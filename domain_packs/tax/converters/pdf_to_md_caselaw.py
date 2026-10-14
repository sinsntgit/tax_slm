"""
Step 10: Convert case-law judgment/order PDFs to clean Markdown.

Unlike 09_pdf_to_md_actdocs.py (income_tax_act_1961.pdf), these PDFs decode
correctly with plain `pdftotext` - confirmed by direct inspection of all 6
files in data/: no (cid:NN) garbling, no non-standard font subsetting. So
this script does NOT need pdftohtml's layout/font metadata at all - it shells
out to plain `pdftotext` and works from the decoded text directly.

WHAT THIS SOURCE ACTUALLY LOOKS LIKE (confirmed by inspecting all 6 files,
not assumed): two distinct export conventions, both present in the corpus:
  1. Indian Kanoon exports (5 of 6 files) - every page repeats the case
     title as a running header, and ends with an "Indian Kanoon -
     http://indiankanoon.org/doc/<id>/   <page>" footer. A short metadata
     block (Equivalent citations / Author / Bench) usually follows the
     title on page 1.
  2. Raw tribunal/court order exports (Reckitt_Benckiser_Ahmedabad.pdf) -
     every page (from page 2 on) repeats a different running header ("ITA
     Nos. .../ Assessee : .../ Asst. Year : ...") and a bare "- N-" page
     number, no Indian Kanoon footer.
Rather than hardcoding either exact pattern, this script detects repeated
header/footer lines GENERICALLY: any line that (after normalizing away the
page number) recurs on a large fraction of pages, near the top or bottom of
its page, is page furniture and gets stripped - so it works on both
conventions above and on formats not yet seen in this corpus.

WHY NO DETERMINISTIC STRUCTURAL PARSER (unlike chunking.py for the Act):
a judgment is discursive prose - paragraphs of reasoning, not a fixed
Section/Sub-section/Clause hierarchy with a stable heading grammar. The
only reliable structural markers here are numbered paragraphs ("2.", "6.1"),
which this script preserves as paragraph starts, and a handful of loose
front-matter labels (Bench/Author/Equivalent citations) it captures
best-effort. Everything else - identifying Facts vs Issues vs Arguments vs
Held, extracting the citation/court/bench/sections/findings structure - is
inherently a reading-comprehension task, not a layout-detection one, and is
deliberately left to the LLM extraction pass in
11_extract_case_law.py (see that script's docstring, and the "separate
chunking program" assessment in this project's notes) rather than guessed
at here with fragile regex.

Usage:
    # Single file:
    python 10_pdf_to_md_caselaw.py --pdf ../data/Reckitt_Benckiser_Ahmedabad.pdf \\
        --out ../data/case_law_md/Reckitt_Benckiser_Ahmedabad.md

    # Whole folder of case-law PDFs at once:
    python 10_pdf_to_md_caselaw.py --dir ../data --out-dir ../data/case_law_md
"""
import argparse
import re
import subprocess
from collections import Counter
from pathlib import Path

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))

from framework.core.convert import poppler_utils
from framework.core.infra import step_logger

# Indian Kanoon's own footer, e.g.
# "Indian Kanoon - http://indiankanoon.org/doc/139970207/        1"
# BUG FOUND VIA A REAL SMOKE TEST: without `-layout`, pdftotext splits this
# into TWO lines - the URL, then the page number alone on the line after -
# the large whitespace gap between them in the original PDF gets read as a
# line break. The trailing `\d+` is therefore optional here; the bare
# page-number line that follows is caught separately by
# _is_bare_trailing_page_number() below.
INDIANKANOON_FOOTER_RE = re.compile(r'^Indian Kanoon\s*-\s*http\S+\s*\d*\s*$')
BARE_DIGITS_RE = re.compile(r'^\d{1,4}$')

# BUG FOUND VIA A REAL SMOKE TEST: a bare page-number marker like "- 2-"
# (the raw-tribunal-order convention, confirmed in Reckitt_Benckiser_
# Ahmedabad.pdf) normalizes to an EMPTY string in
# _normalize_for_repeat_check() (it strips the digit AND the surrounding
# dashes), so it never reached the `len(norm) >= 4` frequency check below
# and leaked straight into the body text mid-sentence ("...Commissioner of
# Income-tax - 2– (Appeals)-9, Ahmedabad..."). Page-number-only lines are
# never legitimate body content regardless of frequency, so they're
# stripped unconditionally instead of via the frequency vote - requires
# both a digit AND a dash so a genuine bare paragraph number (which this
# corpus always writes with a trailing period, e.g. "6.") is never caught
# by this.
PAGE_NUMBER_ONLY_RE = re.compile(r'^(?=[\d\-–—\s]{1,10}$)(?=.*\d)(?=.*[\-–—]).+$')

# A metadata front-matter label Indian Kanoon exports carry on page 1,
# e.g. "Bench: K.S. Hegde, J.C. Shah" / "Author: Pinaki Chandra Ghose" /
# "Equivalent citations: 1970 AIR 691, ...". Captured best-effort into the
# output's front-matter block; not required for anything downstream to
# parse (11_extract_case_law.py re-derives all of this from the full text
# via the LLM anyway) - this is purely a readability nicety.
FRONTMATTER_LABEL_RE = re.compile(r'^(Bench|Author|Equivalent citations)\s*:\s*(.*)$', re.IGNORECASE)

# A numbered paragraph start, e.g. "2." / "6.1" / "10." - common in Indian
# judgments/tribunal orders. Sometimes the number lands alone on its own
# physical line with the paragraph text starting on the next line (a
# pdftotext line-break artifact around the original PDF's tab/indent) -
# handled by _merge_lone_paragraph_numbers() below.
PARA_NUMBER_RE = re.compile(r'^(\d{1,3}(?:\.\d{1,2})?)\.\s*(.*)$')
LONE_PARA_NUMBER_RE = re.compile(r'^(\d{1,3}(?:\.\d{1,2})?)\.\s*$')

# Short standalone structural labels that should stay on their own line
# rather than being folded into a paragraph, e.g. "ORDER", "JUDGMENT",
# "PETITIONER:", "RESPONDENT:", "BEFORE:". Kept intentionally narrow
# (short, mostly-uppercase, optionally colon-terminated) so it doesn't
# accidentally swallow a genuine short SENTENCE of body text.
STANDALONE_LABEL_RE = re.compile(r'^[A-Z][A-Z .,\'"/&()-]{1,40}:?$')


def run_pdftotext(pdf_path: Path) -> str:
    cmd = [poppler_utils.poppler_bin("pdftotext"), str(pdf_path), "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not find/run '{cmd[0]}' (Poppler's pdftotext). Install Poppler and either "
            f"add its bin/ folder to your system PATH, or set POPPLER_BIN_DIR in .env to that "
            f"folder (see scripts/poppler_utils.py)."
        ) from e
    return result.stdout.decode("utf-8", errors="replace")


def split_pages(raw_text: str) -> list[list[str]]:
    """pdftotext separates pages with form-feed (\\f). Returns a list of
    pages, each a list of non-empty, right-stripped lines (blank lines
    dropped here - paragraph breaks are re-derived structurally below, not
    from pdftotext's often-inconsistent blank-line placement)."""
    pages = raw_text.split("\f")
    return [[ln.rstrip() for ln in p.split("\n") if ln.strip()] for p in pages if p.strip()]


def _normalize_for_repeat_check(line: str) -> str:
    """Collapses every run of digits to a single placeholder so the SAME
    running header/footer on different pages - which differs ONLY in
    whatever number is embedded in it - still compares equal across pages."""
    t = line.strip()
    t = re.sub(r'\d+', '#', t)
    t = re.sub(r'\s+', ' ', t)
    return t.strip(" -–—")


# BUG FOUND VIA A REAL SMOKE TEST (Grasim_Industries, 117 pages): this
# document's running header is a watermark-style overlay ("Page | 9 Grasim
# Industries Ltd. 2018-19") that pdftotext interleaves at an ARBITRARY
# position in reading order - not its own line, and not even reliably at a
# page edge; it showed up mid-sentence ("...Aditya Birla Nuvo Ltd Page | 10
# Grasim Industries..."). Line-level frequency detection (detect_running_
# furniture, above) can never catch a fragment embedded inside a line, so
# this is handled as a separate, evidence-gated substring pass: find every
# "Page | <digits> ..." occurrence, capture a generous window of the words
# that follow, and determine the watermark's TRUE length by POSITIONAL
# AGREEMENT across occurrences - the fixed watermark text ("Grasim
# Industries Ltd. <year>") recurs identically (once digits are collapsed)
# at each position, while whatever real body text happens to follow it
# varies every time, so agreement collapses right at the watermark/body
# boundary. (An earlier version captured a FIXED word count instead, which
# routinely over-captured into the next sentence and made every occurrence
# look "different," so nothing ever crossed the repetition threshold - 137
# leaked occurrences in the first real test.)
PAGE_PIPE_RE = re.compile(r'Page\s*\|\s*\d+\s+((?:\S+\s+){0,9}\S+)')
_WATERMARK_MIN_OCCURRENCES = 3
_WATERMARK_MIN_AGREEMENT_FRACTION = 0.5


def strip_page_pipe_watermarks(lines: list[str]) -> list[str]:
    joined = "\n".join(lines)
    matches = list(PAGE_PIPE_RE.finditer(joined))
    if len(matches) < _WATERMARK_MIN_OCCURRENCES:
        return lines

    token_seqs = [re.sub(r'\d+', '#', m.group(1)).split() for m in matches]
    max_len = min(len(seq) for seq in token_seqs)
    min_agreement = max(_WATERMARK_MIN_OCCURRENCES, len(token_seqs) * _WATERMARK_MIN_AGREEMENT_FRACTION)

    watermark_tokens: list[str] = []
    for pos in range(max_len):
        tok, count = Counter(seq[pos] for seq in token_seqs).most_common(1)[0]
        if count < min_agreement:
            break
        watermark_tokens.append(tok)
    if not watermark_tokens:
        return lines  # no confirmed repeating suffix - leave "Page | N" occurrences alone

    # A token can be a COMPOUND placeholder like "#-#" (from "2018-19" with
    # its digit runs collapsed), not just a bare "#" - re.escape(t) alone
    # would then match the literal string "#-#" instead of real digits,
    # silently matching nothing in the actual document (found via a real
    # smoke test: watermark_tokens correctly resolved to ['Grasim',
    # 'Industries', 'Ltd.', '#-#'], yet 137 occurrences still leaked
    # through because this substitution never matched anything real).
    def _token_regex(t: str) -> str:
        return re.escape(t).replace(re.escape('#'), r'\d+')

    watermark_re = re.compile(
        r'Page\s*\|\s*\d+\s+' + r'\s+'.join(_token_regex(t) for t in watermark_tokens)
    )
    cleaned = watermark_re.sub('', joined)
    return cleaned.split("\n")


def detect_running_furniture(pages: list[list[str]]) -> set[str]:
    """Frequency-based header/footer detection: a normalized line occurring
    near the top (first 3) or bottom (last 3) of at least 40% of pages is
    page furniture, not real content - generalizes across the Indian
    Kanoon running-header/footer convention AND the raw tribunal-order
    running-header convention (both confirmed present in this corpus)
    without hardcoding either one's exact wording."""
    if len(pages) < 3:
        return set()  # too few pages for frequency to mean anything
    counts = Counter()
    for page in pages:
        edge_lines = page[:3] + page[-3:]
        seen_this_page = set()
        for ln in edge_lines:
            norm = _normalize_for_repeat_check(ln)
            if norm and len(norm) >= 4 and norm not in seen_this_page:
                counts[norm] += 1
                seen_this_page.add(norm)
    threshold = max(2, int(len(pages) * 0.4))
    return {norm for norm, c in counts.items() if c >= threshold}


def strip_furniture(pages: list[list[str]], furniture: set[str]) -> list[list[str]]:
    out = []
    for page in pages:
        # A bare 1-4 digit line (nothing else) sitting as the LAST line of
        # a page is a page number, not content - real paragraph numbers in
        # this corpus always carry a trailing period ("2.", never a bare
        # "2"), so this is safe. Checked before the main per-line pass
        # because it depends on position within THIS page, not frequency
        # across all pages (a plain, incrementing page number like "1",
        # "2", "3"... never repeats verbatim, so the frequency-based
        # `furniture` set below can never catch it).
        trimmed = list(page)
        if trimmed and BARE_DIGITS_RE.match(trimmed[-1].strip()):
            trimmed = trimmed[:-1]

        kept = []
        for ln in trimmed:
            stripped = ln.strip()
            norm = _normalize_for_repeat_check(ln)
            if INDIANKANOON_FOOTER_RE.match(stripped):
                continue
            if PAGE_NUMBER_ONLY_RE.match(stripped):
                continue
            if norm in furniture:
                continue
            kept.append(ln)
        out.append(kept)
    return out


def _merge_lone_paragraph_numbers(lines: list[str]) -> list[str]:
    """"2." alone on its own line, immediately followed by the paragraph's
    actual text on the next line - merges them into one "2. <text>" line so
    downstream paragraph-building treats the number as what it is: this
    paragraph's own opening marker, not a one-token paragraph of its own."""
    out = []
    i = 0
    n = len(lines)
    while i < n:
        m = LONE_PARA_NUMBER_RE.match(lines[i].strip())
        if m and i + 1 < n and lines[i + 1].strip():
            out.append(f"{m.group(1)}. {lines[i + 1].strip()}")
            i += 2
            continue
        out.append(lines[i])
        i += 1
    return out


def build_paragraphs(all_lines: list[str]) -> list[str]:
    """Reflows physical (page-width-wrapped) lines into logical paragraphs:
    joins consecutive lines with a space until a new numbered-paragraph
    start or a standalone structural label is hit, each of which begins a
    new paragraph/line of its own. Unlike 09's Act converter, case-law text
    has no line-anchored regex grammar to preserve, so reflowing into real
    paragraphs (rather than keeping pdftotext's raggy physical line breaks)
    is what makes this readable prose for the LLM extraction pass."""
    paragraphs: list[str] = []
    current: list[str] = []

    def flush():
        if current:
            paragraphs.append(" ".join(current).strip())
            current.clear()

    for line in all_lines:
        text = line.strip()
        if not text:
            continue
        if PARA_NUMBER_RE.match(text):
            flush()
            current.append(text)
        elif STANDALONE_LABEL_RE.match(text) and len(text) <= 60:
            flush()
            paragraphs.append(text)
        else:
            current.append(text)
    flush()
    return paragraphs


def extract_frontmatter(pages: list[list[str]]) -> dict:
    """Best-effort capture of Indian Kanoon's Bench/Author/Equivalent
    citations labels from page 1 - cosmetic only, see module docstring."""
    info = {}
    if not pages:
        return info
    for ln in pages[0][:15]:
        m = FRONTMATTER_LABEL_RE.match(ln.strip())
        if m:
            key = m.group(1).strip().lower().replace(" ", "_")
            info.setdefault(key, m.group(2).strip())
    return info


def convert_one(pdf_path: Path) -> tuple[str, dict]:
    raw_text = run_pdftotext(pdf_path)
    pages = split_pages(raw_text)
    frontmatter = extract_frontmatter(pages)

    furniture = detect_running_furniture(pages)
    clean_pages = strip_furniture(pages, furniture)

    # Drop the exact lines captured into `frontmatter` from page 1's body
    # flow too, so Bench/Author/Equivalent-citations text doesn't appear
    # twice (once in the header block above, once again inline where it
    # originally sat in the document).
    if frontmatter and clean_pages:
        clean_pages[0] = [
            ln for ln in clean_pages[0] if not FRONTMATTER_LABEL_RE.match(ln.strip())
        ]

    all_lines: list[str] = []
    for page in clean_pages:
        all_lines.extend(_merge_lone_paragraph_numbers(page))

    all_lines = strip_page_pipe_watermarks(all_lines)

    paragraphs = build_paragraphs(all_lines)

    header_block = [f"# {pdf_path.stem.replace('_', ' ')}"]
    if frontmatter:
        header_block.append("")
        for k, v in frontmatter.items():
            header_block.append(f"**{k.replace('_', ' ').title()}:** {v}")
    markdown = "\n".join(header_block) + "\n\n" + "\n\n".join(paragraphs)

    stats = {
        "pages": len(pages),
        "furniture_lines_detected": len(furniture),
        "paragraphs": len(paragraphs),
        "frontmatter_fields": list(frontmatter.keys()),
    }
    return markdown, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", type=Path, default=None, help="Single PDF to convert")
    parser.add_argument("--out", type=Path, default=None, help="Output .md path (with --pdf)")
    parser.add_argument("--dir", type=Path, default=None, help="Convert every PDF in this folder")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output folder (with --dir)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    if args.dir:
        if not args.out_dir:
            raise SystemExit("--out-dir is required with --dir")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        pdfs = sorted(list(args.dir.glob("*.pdf")) + list(args.dir.glob("*.PDF")))
        # Skip the Act PDF itself if it happens to live in the same folder -
        # this script is for case law, not the Act (see 09 for that).
        pdfs = [p for p in pdfs if "income_tax_act_1961" not in p.name.lower()]
        if not pdfs:
            raise SystemExit(f"No case-law PDFs found in {args.dir}")
        step_logger.log(f"Found {len(pdfs)} case-law PDF(s) to convert")
        for pdf_path in pdfs:
            with step_logger.Step(f"Converting {pdf_path.name}"):
                markdown, stats = convert_one(pdf_path)
                out_path = args.out_dir / (pdf_path.stem + ".md")
                out_path.write_text(markdown, encoding="utf-8")
                step_logger.log(f"  {stats} -> {out_path}")
    elif args.pdf:
        if not args.out:
            raise SystemExit("--out is required with --pdf")
        with step_logger.Step(f"Converting {args.pdf.name}"):
            markdown, stats = convert_one(args.pdf)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(markdown, encoding="utf-8")
            step_logger.log(f"  {stats} -> {args.out}")
    else:
        raise SystemExit("Provide either --pdf/--out or --dir/--out-dir")


if __name__ == "__main__":
    main()
