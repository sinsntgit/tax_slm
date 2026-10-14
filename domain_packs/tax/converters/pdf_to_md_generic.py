"""
Generic document -> Markdown/plain-text converter, for the "Generic
Document" file type in the ingestion UI - anything that isn't the Income
Tax Act (pdf_to_md_act.py) or a case-law judgment (pdf_to_md_caselaw.py):
circulars, notifications, memos, firm opinions, training material, etc.

Deliberately simple, unlike the other two converters: there's no fixed
citation grammar or discursive-judgment structure to exploit here, so this
does the minimum honest thing - decode the file to plain text - and leaves
ALL structure-finding to domain_packs/tax/converters/extract_generic.py's
paragraph-based chunker + LLM enrichment pass, the same division of labor
10_pdf_to_md_caselaw.py already uses for judgments.

Supported inputs: .pdf (via Poppler pdftotext -layout, same tool/approach as
pdf_to_md_caselaw.py), .txt, .md (both read as-is - already plain text).
.docx is explicitly NOT supported here (would need a new dependency,
python-docx/mammoth, not in requirements.txt) - convert to PDF or TXT first
and re-upload; raising a clear error is better than a silent wrong result.

Usage:
    python pdf_to_md_generic.py --in ../data/some_circular.pdf --out ../data/generic_md/some_circular.md
"""
import argparse
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))

import subprocess
from pathlib import Path

from framework.core.convert import poppler_utils
from framework.core.infra import step_logger

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def convert_pdf(pdf_path: Path) -> str:
    cmd = [poppler_utils.poppler_bin("pdftotext"), "-layout", str(pdf_path), "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, check=True, timeout=300)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Could not find/run '{cmd[0]}' (Poppler's pdftotext). Install Poppler and either "
            f"add its bin/ folder to your system PATH, or set POPPLER_BIN_DIR in .env."
        ) from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"pdftotext failed on {pdf_path}: {e.stderr.decode('utf-8', errors='replace')[:500]}") from e
    return result.stdout.decode("utf-8", errors="replace")


def convert_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def convert_one(in_path: Path) -> tuple[str, dict]:
    ext = in_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise SystemExit(
            f"Unsupported file type '{ext}' for the Generic Document converter. Supported: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}. (.docx isn't supported yet - "
            f"convert to PDF or plain text first.)"
        )
    if ext == ".pdf":
        text = convert_pdf(in_path)
    else:
        text = convert_text_file(in_path)

    # Normalize line endings and collapse runs of >2 blank lines - keeps
    # paragraph boundaries meaningful for extract_generic.py's chunker
    # without being overly aggressive about whitespace that might matter
    # (e.g. a table's column spacing).
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    import re
    text = re.sub(r"\n{3,}", "\n\n", text)

    stats = {
        "source_extension": ext,
        "characters": len(text),
        "words": len(text.split()),
        "paragraphs": len([p for p in text.split("\n\n") if p.strip()]),
    }
    return text, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in", dest="infile", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    with step_logger.Step(f"Converting {args.infile.name} (generic document converter)"):
        text, stats = convert_one(args.infile)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        step_logger.log(f"  {stats} -> {args.out}")


if __name__ == "__main__":
    main()
