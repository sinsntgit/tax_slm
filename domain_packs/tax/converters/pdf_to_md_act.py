"""
Step 9: Convert a legislative Act PDF (default: income_tax_act_1961.pdf)
straight to the Markdown format scripts/chunking.py expects - Section/
Chapter/Schedule headings it can regex-match, physical line-per-line body
text so its (1)/(a)/(i)/Provided/Explanation/footnote line-anchored parsing
works, and correctly Schedule-scoped headings from the start (no separate
post-processing patch needed for that part).

NOT DOCLING. Deliberately. Both pdfplumber (pdfminer.six) and PyMuPDF/fitz
were tried against this PDF's actual embedded font encoding and BOTH
produced garbled, undecodable text ((cid:NN) placeholders / raw control
bytes) - the fonts are subsetted with a non-standard glyph->Unicode mapping
neither library resolves correctly. `pdftotext`/`pdftohtml` (Poppler's own
CLI, already installed in this environment) decode it perfectly - Poppler
apparently applies a fallback text-extraction heuristic pdfminer/PyMuPDF
don't. This script shells out to `pdftohtml -xml`, which additionally
reports each text run's exact page position (top/left/width/height) and
font reference - real layout metadata, not a markdown guess - which is what
makes reliable heading detection possible at all: this document does NOT
put its headings in a visually distinct font. Confirmed by direct
inspection (see the docstrings on the regexes below): a Section's marginal
heading ("Short title, extent and commencement.") and a Schedule's own
"THE FIRST SCHEDULE" heading both render in the exact same font/size as
ordinary body text. There is no bold/italic/size cue to key off for either
one - only their POSITION in the structural flow (a short line immediately
followed by a "N. " numbered line; or a fixed, closed vocabulary of "THE
<NAME> SCHEDULE" strings) gives them away. Chapter headings are the one
exception that DOES carry a font cue (bold+italic, two-line: "CHAPTER I"
then "PRELIMINARY") and are detected that way.

THE APPROACH FROM 01b_fix_md_schedule_headings.py, INCORPORATED DIRECTLY:
01b was a POST-hoc patch - it re-read an already-converted markdown file,
tracked "## THE <NAME> SCHEDULE" boundaries line-by-line, and relabeled any
"#### Section N." heading found after one to "#### THE <NAME> SCHEDULE --
Rule N. <title>", because the original (undocumented, lost) converter had
no concept of a Schedule boundary at all and kept minting "Section" headings
for every Schedule's own internally-renumbered Rules. This script bakes that
exact same Schedule-boundary tracking into the PRIMARY classifier instead of
patching it in afterwards: the state machine below flips `in_schedule` the
moment it sees a "THE <NAME> SCHEDULE" heading (a real layout signal, not a
markdown regex on already-lossy output) and every heading-shaped line found
after that point is labelled "Rule", not "Section", from the moment it's
first written - never "Section" then silently corrected. As a genuine
second line of defence (not just for show), `fix_markdown()` from
01b_fix_md_schedule_headings.py is still imported and run as a final safety
net pass at the end (see run_01b_safety_net() below) - if the primary
classifier got everything right, as it should on a well-formed page, this
pass finds nothing to change and is a no-op; if it finds something, that's
worth knowing.

Note for future readers: chunking.py's own `section_starts` search is
already scoped to stop at the FIRST Schedule boundary (`main_body = [... if
ln < schedule_boundary]`), and _parse_schedules() treats each Schedule as
ONE flat chunk of body text regardless of what headings appear inside it -
so a mislabeled "Section" heading INSIDE a Schedule was never actually able
to corrupt chunking.py's own output (it's simply never scanned there). The
Schedule/Rule relabeling in this script is still done properly from the
start because it makes the intermediate .md file itself correct and
human-readable, and because nothing guarantees every future consumer of
this file will re-implement chunking.py's same schedule_boundary cutoff.

USAGE:
    # Smoke test on a small page range first:
    python 09_pdf_to_md_actdocs.py --pdf ../data/income_tax_act_1961.pdf \\
        --out /tmp/sample.md --first-page 1 --last-page 60

    # Full conversion + compatibility check against chunking.py:
    python 09_pdf_to_md_actdocs.py --pdf ../data/income_tax_act_1961.pdf \\
        --out ../data/income_tax_act_1961_v3.md --validate --report /tmp/09_report.json
"""
import argparse
import importlib.util
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

from lxml import etree

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))

import chunking  # same directory (domain_packs/tax/converters/)
from framework.core.convert import poppler_utils
from framework.core.infra import step_logger

SCRIPT_DIR = Path(__file__).resolve().parent

# --- Layout-derived structural regexes --------------------------------------
# All confirmed by direct pdftohtml -xml inspection of the real PDF (pages
# 1-6, 47-50, 1030-1094 sampled) - not guessed from the spec or from the
# (already imperfect - see module docstring) existing .md output.

CHAPTER_MARK_RE = re.compile(r'^CHAPTER\s+([IVXLCDM]+(?:-[A-Z])?)\s*$', re.IGNORECASE)
# "THE FIRST SCHEDULE", "THE THIRTEENTH SCHEDULE", etc. - a fixed, closed
# vocabulary (only 14 of these exist in the whole document), rendered
# ALL-CAPS and visually centered (large left offset vs the ~198-270 body
# indent) - both checked, so an incidental body-text mention like "the First
# Schedule to this Act" (Title Case, body-indented) never matches.
SCHEDULE_MARK_RE = re.compile(r'^THE\s+([A-Z][A-Z \-]+?)\s+SCHEDULE\s*$')
SCHEDULE_MARK_MIN_LEFT = 300  # body text starts at left ~198-270; headings observed at left 370-421

# A Section/Rule's own number+period, glued to the start of the FIRST body
# line that follows its marginal heading, e.g. "1. (1) This Act may be
# called..." or "2. In this Act, unless the context otherwise requires,".
# Deliberately does NOT require a following space or "(" - just number,
# 0-3 uppercase letters (for inserted sections like "80CCD"), then ".".
NUMBERED_LINE_START_RE = re.compile(r'^(\d+[A-Z]{0,3})\.(?:\s|\(|$)')

# A marginal heading itself is unremarkable body-style text with no font
# cue at all - only ruled IN by what immediately follows it (see above) and
# ruled OUT if it looks like it could itself be numbered/structural content.
MARGINAL_HEADING_DISQUALIFY_RE = re.compile(r'^[\(\d]|^Provided\b|^Explanation\b', re.IGNORECASE)
MAX_MARGINAL_HEADING_CHARS = 160
# BUG FOUND VIA A REAL SMOKE TEST (pages 1-60): a paragraph that wraps a
# date across a page-width line break - "...the 1st day of April," / "2023."
# on the next physical line - was read the same way: "2023." at line-start
# matches NUMBERED_LINE_START_RE, and the line before it ("to the assessment
# year beginning on or after the 1st day of April,") isn't disqualified by
# anything above, so it got promoted to a bogus "#### Section 2023." Every
# genuine marginal heading in this document (checked against ~20 real
# examples) is a complete title ending in its own period - "Short title,
# extent and commencement.", "Definitions.", "Residence in India." - never a
# mid-sentence fragment ending in a comma or nothing at all. Requiring that
# shape catches this case (and is a second, independent check alongside the
# footnote-line exclusions below, not a replacement for them).
MARGINAL_HEADING_MUST_END_RE = re.compile(r'\.\s*$')

# BUG FOUND VIA A REAL SMOKE TEST (pages 1-60): a footnote-DEFINITION line
# ("1. Sub. by the Act No. 7 of 2025, w.e.f. 1-4-2026.") also starts with
# "N." and was being read by the naive lookahead below as "the next line
# starts a numbered provision" - falsely promoting the ordinary body line
# right before a page's trailing footnote block (e.g. "accordance with the
# provisions of the regulations made under") into a bogus "#### Section 1."
# heading. Reused directly from chunking.py (not re-derived) so this can
# never drift out of sync with what chunking.py itself treats as a
# footnote-definition line.
FOOTNOTE_DEF_RE = chunking.FOOTNOTE_DEF_RE


def run_pdftohtml_xml(pdf_path: Path, first_page: Optional[int], last_page: Optional[int]) -> bytes:
    cmd = [poppler_utils.poppler_bin("pdftohtml"), "-xml", "-stdout", "-q"]
    if first_page:
        cmd += ["-f", str(first_page)]
    if last_page:
        cmd += ["-l", str(last_page)]
    cmd.append(str(pdf_path))
    try:
        result = subprocess.run(cmd, capture_output=True, check=True)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not find/run '{cmd[0]}' (Poppler's pdftohtml). This script shells out to "
            f"Poppler, which is a system tool, not a Python package - 'pip install' doesn't "
            f"provide it. Install Poppler and either add its bin/ folder to your system PATH, "
            f"or set POPPLER_BIN_DIR in .env to that folder (see scripts/poppler_utils.py). "
            f"On Windows: download a prebuilt Poppler release (e.g. "
            f"https://github.com/oschwartz10612/poppler-windows/releases), extract it, and "
            f"point POPPLER_BIN_DIR at its Library\\bin folder."
        ) from e
    return result.stdout


# --- Stage 1: raw text runs from the pdftohtml XML --------------------------

def parse_text_runs(xml_bytes: bytes) -> list[dict]:
    """Returns a flat list of text-run dicts in document order:
    {page, top, left, width, height, size, bold, italic, text}.
    Font size is resolved from the document's cumulative <fontspec> table
    (pdftohtml declares each font id once, the first time it's used, and
    reuses the id afterwards - confirmed by inspecting a full-document dump,
    ids are NOT reset per page when extracting the whole file in one call)."""
    parser = etree.XMLParser(recover=True, huge_tree=True)
    root = etree.fromstring(xml_bytes, parser=parser)

    fontsize_by_id: dict[str, float] = {}
    runs: list[dict] = []

    for page_el in root.iter("page"):
        try:
            page_num = int(page_el.get("number"))
        except (TypeError, ValueError):
            continue
        for child in page_el:
            if child.tag == "fontspec":
                fid = child.get("id")
                try:
                    fontsize_by_id[fid] = float(child.get("size"))
                except (TypeError, ValueError):
                    pass
            elif child.tag == "text":
                text = "".join(child.itertext())
                if text is None or text.strip() == "":
                    continue
                bold = any(el.tag == "b" for el in child.iter())
                italic = any(el.tag == "i" for el in child.iter())
                try:
                    top = int(float(child.get("top", "0")))
                    left = int(float(child.get("left", "0")))
                    width = int(float(child.get("width", "0")))
                    height = int(float(child.get("height", "0")))
                except (TypeError, ValueError):
                    continue
                runs.append({
                    "page": page_num, "top": top, "left": left, "width": width, "height": height,
                    "size": fontsize_by_id.get(child.get("font"), 0.0),
                    "bold": bold, "italic": italic, "text": text,
                })
    return runs


# --- Stage 2: reconstruct physical lines from runs --------------------------

LINE_TOP_TOLERANCE = 8   # px; normal line pitch is ~17-23px, superscript footnote-marker offset is 1-3px
JOIN_GAP_THRESHOLD = 3   # px; runs closer than this are glyph-adjacent (fuse with no inserted space)


def group_into_lines(runs: list[dict]) -> list[dict]:
    """Clusters same-page runs into physical lines by `top` proximity, joins
    each line's runs left-to-right into one text string, and returns lines
    in full-document reading order: {page, top, size, bold, italic, text}.
    `size`/`bold`/`italic` reflect the line's FIRST (leftmost) run only -
    good enough for this script's purposes (Chapter-heading detection,
    which only ever needs the first run of a wholly-bold-or-italic line)."""
    by_page: dict[int, list[dict]] = {}
    for r in runs:
        by_page.setdefault(r["page"], []).append(r)

    lines: list[dict] = []
    for page in sorted(by_page):
        page_runs = sorted(by_page[page], key=lambda r: (r["top"], r["left"]))
        current: list[dict] = []
        current_top_ref = None
        for r in page_runs:
            if current and abs(r["top"] - current_top_ref) > LINE_TOP_TOLERANCE:
                lines.append(_finalize_line(page, current))
                current = []
                current_top_ref = None
            current.append(r)
            current_top_ref = current[0]["top"] if current_top_ref is None else current_top_ref
        if current:
            lines.append(_finalize_line(page, current))
    return lines


def _finalize_line(page: int, run_group: list[dict]) -> dict:
    run_group = sorted(run_group, key=lambda r: r["left"])
    text_parts = []
    prev_right = None
    for r in run_group:
        t = r["text"]
        if prev_right is not None:
            gap = r["left"] - prev_right
            if gap > JOIN_GAP_THRESHOLD and not (text_parts and text_parts[-1].endswith((" ", " "))) and not t.startswith(" "):
                text_parts.append(" ")
        text_parts.append(t)
        prev_right = r["left"] + r["width"]
    text = "".join(text_parts)
    # Collapse the PDF's internal justification spacing (multiple spaces
    # between words on a justified line) down to single spaces - purely
    # cosmetic, doesn't affect any of chunking.py's regexes (all `\s*`/`\s?`).
    text = re.sub(r'[ \t]{2,}', ' ', text).strip()
    first = run_group[0]
    return {
        "page": page, "top": min(r["top"] for r in run_group), "left": first["left"], "text": text,
        "size": first["size"], "bold": first["bold"], "italic": first["italic"],
    }


# --- Stage 3: classify lines into the target Markdown -----------------------

def classify_and_render(lines: list[dict]) -> tuple[str, dict]:
    """Walks the reconstructed physical lines once, in order, and emits the
    target Markdown. Returns (markdown_text, stats)."""
    out: list[str] = []
    stats = Counter()
    size_check_rejection_examples: list[dict] = []

    # Body-text font size - used below as a SECOND, independent signal (real
    # layout, not text pattern) that the line immediately after a candidate
    # marginal heading is an actual numbered provision and not a footnote
    # note. Needed because not every real footnote-definition line matches
    # chunking.FOOTNOTE_DEF_RE's fixed action-word vocabulary (e.g. `2. Words
    # "..." omitt. by the Act No. 7 of 2025...` phrases the action verb,
    # lowercase, mid-sentence - found via a real smoke test on pages 1-60) -
    # but every one of them, regardless of wording, renders smaller than
    # body text.
    #
    # NOT computed as "the single most common line size across the WHOLE
    # document" (the original approach) - that turned out to be an unstable
    # definition of "body text" across Poppler versions. Confirmed via a
    # real run on Windows/Poppler 26.02.0 (vs 22.02.0 used to validate this
    # script originally): the whole-document mode came back as 15.0pt, while
    # every real numbered-provision line (confirmed via several logged
    # examples) still rendered at 14.0pt - something else in the document
    # (headers/footers/footnote blocks/whatever else) had simply become MORE
    # numerous than actual provision text under that build's text-run
    # splitting (74419 runs vs 48214 on the same PDF), skewing the naive
    # mode enough to misclassify every single Section as "too small = a
    # footnote." Instead, restrict the mode computation to lines that
    # already look TEXTUALLY like a numbered-provision start
    # (NUMBERED_LINE_START_RE) - that directly targets "the font size real
    # provision text renders at," which is the actual concept this check
    # needs, rather than "whatever's most common in the document overall."
    size_counts = Counter(round(l["size"], 1) for l in lines if l["size"])
    provision_line_sizes = Counter(
        round(l["size"], 1) for l in lines
        if l["size"] and NUMBERED_LINE_START_RE.match(l["text"])
    )
    if provision_line_sizes:
        body_size = provision_line_sizes.most_common(1)[0][0]
    else:
        body_size = size_counts.most_common(1)[0][0] if size_counts else 14.0
    stats["body_font_size"] = body_size

    # Threshold below which a line's size counts as "smaller than body" (the
    # footnote-note signal in the Section/Rule detection block below).
    # ADAPTIVE, not a fixed epsilon: originally this was a hardcoded
    # `body_size - 0.5`, tuned against Poppler 22.02.0's output on this
    # document (14pt body / 12pt footnote-definitions / 8pt footnote-marker
    # digits - see the comment above). That fixed 0.5pt gap is exactly the
    # kind of magic number that breaks if a different Poppler version scales
    # or rounds reported font sizes even slightly differently - which is
    # suspected to be exactly what happened going from 22.02.0 to 26.02.0 (a
    # 4-version jump) on Windows: every Section got rejected by this one
    # check, while the Chapter/Schedule detectors (which use bold/italic and
    # x-position, not a numeric size threshold) kept working fine. Instead,
    # derive the threshold from THIS document's own observed size
    # distribution each run: find the largest distinct size smaller than
    # body_size, and set the cutoff halfway between the two - so it tracks
    # whatever scale/precision this particular Poppler build happens to
    # report, rather than assuming 22.02.0's exact numbers.
    smaller_sizes = sorted((s for s in size_counts if s < body_size), reverse=True)
    if smaller_sizes:
        size_threshold = (body_size + smaller_sizes[0]) / 2
    else:
        size_threshold = body_size - 0.5  # no smaller size observed at all - fall back to the old constant
    stats["_size_threshold"] = size_threshold

    in_schedule = False
    schedule_name = None
    chapter_pending_roman = None
    i = 0
    n = len(lines)

    # Tracks the tail of `out` that's plain, not-yet-consumed body text -
    # cleared at every heading boundary. Used below to pull in a long
    # marginal heading's earlier line(s): some titles wrap across two
    # physical PDF lines (e.g. Section 9B's real heading is "Income on
    # receipt of capital asset or stock in trade by specified person from" /
    # "specified entity." on the next line - found via a real smoke test on
    # pages 1-60) and only the line immediately before the numbered body
    # would otherwise be captured, silently truncating the title.
    recent_body_texts: list[str] = []
    MAX_HEADING_LOOKBACK = 2

    # Page-boundary markers for downstream page-number citation: (output
    # line count at this point, source PDF page). Coarse (recorded once per
    # source line consumed, not per `out.append()` call inside its
    # handling) - good enough to attribute a chunk to "roughly which PDF
    # page" without threading page-tracking through every render branch.
    page_boundaries: list[tuple[int, int]] = []
    _last_page_seen = None

    while i < n:
        line = lines[i]
        text = line["text"]
        if line.get("page") != _last_page_seen:
            _last_page_seen = line.get("page")
            page_boundaries.append((len(out), _last_page_seen))

        # --- Chapter heading: two-line pattern (bold+italic marker, then
        # bold title) - the one heading type in this document that DOES
        # carry a font/style cue.
        if chapter_pending_roman is not None:
            out.append(f"\n## CHAPTER {chapter_pending_roman} -- {text.strip()}\n")
            stats["chapters"] += 1
            chapter_pending_roman = None
            recent_body_texts = []
            i += 1
            continue
        if line["bold"] and line["italic"]:
            cm = CHAPTER_MARK_RE.match(text)
            if cm:
                chapter_pending_roman = cm.group(1).upper()
                recent_body_texts = []
                i += 1
                continue

        # --- Schedule heading: "THE <NAME> SCHEDULE", centered, all caps.
        sm = SCHEDULE_MARK_RE.match(text)
        if sm and _line_is_centered(line):
            schedule_name = sm.group(1).strip()
            in_schedule = True
            out.append(f"\n## THE {schedule_name} SCHEDULE\n")
            stats["schedules"] += 1
            recent_body_texts = []
            i += 1
            continue

        # --- Section/Rule marginal heading: this line is a plausible short
        # heading AND the next line starts a numbered provision.
        if (
            i + 1 < n
            and not MARGINAL_HEADING_DISQUALIFY_RE.match(text)
            and MARGINAL_HEADING_MUST_END_RE.search(text)
            and len(text) <= MAX_MARGINAL_HEADING_CHARS
            and text
        ):
            stats["_candidates_total"] += 1
            next_line = lines[i + 1]
            next_text = next_line["text"]
            nm = NUMBERED_LINE_START_RE.match(next_text)
            # Diagnostics only (not used for the decision itself): counts
            # WHY a plausible marginal-heading candidate got rejected, so a
            # platform-specific extraction difference (e.g. font sizes
            # reported on a different scale by a different Poppler build)
            # shows up as a spike in one specific counter instead of just a
            # silently low `sections` total - see the hard-fail check in
            # main() below, which reads these to explain itself. Also keeps
            # up to 3 concrete (heading_text, next_text, next_size, body_size)
            # examples of the size-check rejection specifically, since that's
            # the one check that depends on a numeric comparison that could
            # plausibly differ across Poppler builds/platforms.
            if not nm:
                stats["_candidates_rejected_no_numbered_next"] += 1
            elif FOOTNOTE_DEF_RE.match(next_text.strip()):
                nm = None  # a footnote-definition line, not a real numbered provision start
                stats["_candidates_rejected_footnote_def"] += 1
            elif next_line["size"] and round(next_line["size"], 1) < size_threshold:
                nm = None  # smaller-than-body font - a footnote note, whatever its wording
                stats["_candidates_rejected_size_check"] += 1
                if len(size_check_rejection_examples) < 3:
                    size_check_rejection_examples.append(
                        {"heading": text.strip()[:60], "next_line": next_text.strip()[:60],
                         "next_size": next_line["size"], "body_size": body_size,
                         "size_threshold": size_threshold}
                    )
            if nm:
                number = nm.group(1)
                title_parts = [text.strip()]
                # Walk backward through immediately-preceding plain lines
                # that don't look like a complete sentence of their own
                # (no terminal period) - those are wrapped continuations of
                # THIS heading, not the previous section's last sentence.
                lookback = 0
                while (
                    recent_body_texts and lookback < MAX_HEADING_LOOKBACK
                    and out and out[-1] == recent_body_texts[-1]
                ):
                    prev_text = recent_body_texts[-1]
                    if (
                        MARGINAL_HEADING_DISQUALIFY_RE.match(prev_text)
                        or MARGINAL_HEADING_MUST_END_RE.search(prev_text)
                        or len(prev_text) > MAX_MARGINAL_HEADING_CHARS
                    ):
                        break
                    title_parts.insert(0, prev_text)
                    recent_body_texts.pop()
                    out.pop()
                    lookback += 1
                title = " ".join(title_parts)
                if in_schedule:
                    out.append(f"\n#### THE {schedule_name} SCHEDULE — Rule {number}. {title}\n")
                    stats["rules"] += 1
                else:
                    out.append(f"\n---\n\n#### Section {number}. {title}\n")
                    stats["sections"] += 1
                recent_body_texts = []
                i += 1
                continue

        # --- Default: plain body/footnote line, emitted verbatim, one PDF
        # physical line per Markdown line (chunking.py's (1)/(a)/(i)/
        # Provided/Explanation/footnote regexes are all line-anchored, so
        # preserving the PDF's own line breaks - rather than re-flowing into
        # paragraphs - is what makes them matchable at all).
        out.append(text)
        recent_body_texts.append(text)
        i += 1

    markdown = "\n".join(out)
    # Collapse any accidental triple-plus blank lines from adjacent heading
    # insertions back down to a normal single blank line between blocks.
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)
    final_stats = dict(stats)
    if size_check_rejection_examples:
        final_stats["_size_check_rejection_examples"] = size_check_rejection_examples
    final_stats["_page_boundaries"] = page_boundaries
    return markdown, final_stats


def _line_is_centered(line: dict) -> bool:
    # Body text starts at left ~198-270; the 14 real Schedule headings were
    # all observed at left ~370-421 (centered on an ~918pt-wide page). Used
    # alongside the ALL-CAPS "THE ... SCHEDULE" text match so an incidental
    # body-text mention like "the First Schedule to this Act" (Title Case,
    # body-indented) never gets treated as a boundary.
    return line["left"] >= SCHEDULE_MARK_MIN_LEFT


# --- Safety-net pass: reuse 01b's actual fix_markdown() ---------------------

def _load_module_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def run_01b_safety_net(markdown_text: str) -> tuple[str, dict]:
    """Runs 01b_fix_md_schedule_headings.py's own fix_markdown() over the
    output as a genuine second line of defence (not a re-implementation of
    its logic) - see module docstring. Loaded dynamically because the
    filename starts with a digit and isn't a valid `import` target."""
    mod_path = SCRIPT_DIR / "01b_fix_md_schedule_headings.py"
    mod = _load_module_from_path("_fix_md_schedule_headings_01b", mod_path)
    return mod.fix_markdown(markdown_text)


# --- Sanity checks (reused from 00A_pdf_to_md_docling.py - pure regex/
# subprocess helpers with no Docling dependency) -----------------------------

def load_00a_sanity_checks():
    mod_path = SCRIPT_DIR / "00A_pdf_to_md_docling.py"
    mod = _load_module_from_path("_pdf_to_md_docling_00a", mod_path)
    return mod.check_heading_duplicates, mod.check_word_count_vs_pdftotext


# --- Optional: run the produced markdown through chunking.py itself ---------

def run_chunking_compat_check(markdown_text: str) -> dict:
    result = chunking.chunk_act_document(markdown_text)
    entity_types = Counter(e["type"] for e in result["entities"])
    rel_types = Counter(r["type"] for r in result["relationships"])
    chunk_types = Counter(c.element_type for c in result["chunks"])
    return {
        "entities_total": len(result["entities"]),
        "entities_by_type": dict(entity_types),
        "relationships_total": len(result["relationships"]),
        "relationships_by_type": dict(rel_types),
        "chunks_total": len(result["chunks"]),
        "chunks_by_type": dict(chunk_types),
        "ambiguities_total": len(result["ambiguities"]),
        "ambiguities_sample": result["ambiguities"][:10],
    }


def convert(pdf_path: Path, first_page: Optional[int], last_page: Optional[int]) -> tuple[str, dict]:
    with step_logger.Step(f"Extracting text layout via pdftohtml -xml "
                           f"({pdf_path.name}"
                           f"{f', pages {first_page}-{last_page}' if first_page or last_page else ', full document'})"):
        xml_bytes = run_pdftohtml_xml(pdf_path, first_page, last_page)
        step_logger.log(f"  {len(xml_bytes) / 1024:.0f} KB of XML")

    with step_logger.Step("Parsing text runs"):
        runs = parse_text_runs(xml_bytes)
        step_logger.log(f"  {len(runs)} text run(s)")

    with step_logger.Step("Reconstructing physical lines"):
        # group_into_lines/_finalize_line need `left` for the Schedule
        # centeredness check - stash it before dropping down to the merged
        # line representation.
        lines = group_into_lines(runs)
        step_logger.log(f"  {len(lines)} line(s)")

    with step_logger.Step("Classifying structure and rendering Markdown"):
        markdown_text, stats = classify_and_render(lines)
        step_logger.log(f"  {stats}")

    with step_logger.Step("Running 01b safety-net pass"):
        markdown_text, fix_report = run_01b_safety_net(markdown_text)
        step_logger.log(f"  {fix_report['total_headings_rewritten']} heading(s) additionally rewritten "
                         f"by the safety net ({fix_report['rewritten_in_schedule']} schedule/rule, "
                         f"{fix_report['rewritten_in_body']} demoted) - 0 is expected/good here, since "
                         f"the primary classifier already applies the same Schedule-boundary logic.")

    stats["safety_net_rewrites"] = fix_report["total_headings_rewritten"]
    return markdown_text, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--first-page", type=int, default=None)
    parser.add_argument("--last-page", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None, help="Write a JSON stats/sanity-check report here")
    parser.add_argument("--validate", action="store_true",
                         help="Also run the output through chunking.chunk_act_document() and report "
                              "entity/relationship/chunk/ambiguity counts as a compatibility check")
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    markdown_text, stats = convert(args.pdf, args.first_page, args.last_page)

    with step_logger.Step(f"Writing {args.out}"):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(markdown_text, encoding="utf-8")

    with step_logger.Step("Writing page-map sidecar (for downstream page-number citation)"):
        page_boundaries = stats.pop("_page_boundaries", [])
        pagemap_path = args.out.with_suffix(".pagemap.json")
        pagemap_path.write_text(json.dumps(page_boundaries), encoding="utf-8")
        step_logger.log(f"  {len(page_boundaries)} page boundary marker(s) written to {pagemap_path}")

    check_heading_duplicates, check_word_count_vs_pdftotext = load_00a_sanity_checks()
    report = {"conversion_stats": stats}
    with step_logger.Step("Post-conversion sanity checks"):
        # HARD FAILURE, not a warning: a real Act has many sections per
        # chapter (dozens, typically), never fewer sections than chapters.
        # Seeing sections < chapters means the marginal-heading classifier
        # (classify_and_render()'s Section/Rule detection block) rejected
        # essentially every candidate - almost always either a Poppler-
        # build/platform difference in how font SIZE is reported (the one
        # numeric comparison in that block: `next_line["size"] < body_size -
        # 0.5`), or a change to the regexes it depends on. Exiting non-zero
        # here (after the file is already written, so it's still there to
        # inspect) stops a structurally-empty .md file from silently
        # flowing into 02_extract_entities.py, where the actual symptom
        # would otherwise only surface several steps later as "0 entities"
        # with no indication of why.
        n_chapters, n_sections = stats.get("chapters", 0), stats.get("sections", 0)
        if n_chapters >= 5 and n_sections < n_chapters:
            step_logger.log(f"  FAILED: {n_chapters} chapters detected but only {n_sections} sections - "
                             f"a real Act always has far more sections than chapters. The Section/Rule "
                             f"marginal-heading classifier is rejecting nearly every candidate.")
            step_logger.log(f"  Candidate breakdown: {stats.get('_candidates_total', 0)} plausible heading "
                             f"lines seen, rejected as: no-numbered-next-line="
                             f"{stats.get('_candidates_rejected_no_numbered_next', 0)}, "
                             f"footnote-definition-line={stats.get('_candidates_rejected_footnote_def', 0)}, "
                             f"smaller-than-body-font-size={stats.get('_candidates_rejected_size_check', 0)}.")
            if stats.get("_candidates_rejected_size_check", 0) > 0:
                step_logger.log(f"  body_font_size={stats.get('body_font_size')}. Size-check rejection "
                                 f"examples: {stats.get('_size_check_rejection_examples', [])}")
                step_logger.log("  This points at font-size reporting differing between Poppler builds - "
                                 "run `pdftohtml -v` and compare against a known-good version (22.02.0 was "
                                 "used to validate this script), or share the examples above so the "
                                 "size-check tolerance can be adjusted.")
            report["conversion_stats"] = stats
            if args.report:
                args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
                step_logger.log(f"  Diagnostic report written to {args.report} despite the failure.")
            raise SystemExit(
                f"Conversion produced a structurally broken document ({n_sections} sections for "
                f"{n_chapters} chapters) - see the diagnostics above. {args.out} was still written "
                f"for inspection, but do NOT feed it into 02_extract_entities.py as-is."
            )

        h = check_heading_duplicates(markdown_text)
        report["headings"] = h
        step_logger.log(f"  {h['total_headings']} heading(s), {len(h['duplicate_headings'])} "
                         f"distinct heading text repeated more than once")
        if h["duplicate_headings"]:
            for text, count in sorted(h["duplicate_headings"].items(), key=lambda kv: -kv[1])[:10]:
                step_logger.log(f"    x{count}: {text[:80]!r}")

        w = check_word_count_vs_pdftotext(markdown_text, args.pdf)
        report["word_count"] = w
        if w["pdftotext_available"]:
            step_logger.log(f"  word count: PDF={w['pdf_word_count']}, MD={w['md_word_count']} "
                             f"({w['pct_diff']:+.2f}% diff)")
            if w["pct_diff"] is not None and abs(w["pct_diff"]) > 5:
                step_logger.log("  WARNING: word count diverges by more than 5% - check for dropped content.")

    if args.validate:
        with step_logger.Step("Compatibility check: running output through chunking.chunk_act_document()"):
            try:
                compat = run_chunking_compat_check(markdown_text)
                report["chunking_compat"] = compat
                step_logger.log(f"  entities={compat['entities_total']} "
                                 f"relationships={compat['relationships_total']} "
                                 f"chunks={compat['chunks_total']} "
                                 f"ambiguities={compat['ambiguities_total']}")
                step_logger.log(f"  entities by type: {compat['entities_by_type']}")
                step_logger.log(f"  chunks by type: {compat['chunks_by_type']}")
            except Exception as e:
                step_logger.log(f"  FAILED to run chunking.py compatibility check: {type(e).__name__}: {e}")
                report["chunking_compat_error"] = str(e)

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        step_logger.log(f"Report written to {args.report}")

    step_logger.log(f"Done - {args.out} ({len(markdown_text.split())} words)")


if __name__ == "__main__":
    main()
