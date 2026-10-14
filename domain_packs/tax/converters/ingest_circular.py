"""
Single-command CBDT Circular ingestion: run THIS ONE FILE and it does
PDF->text conversion (pdf_to_md_generic.convert_one() - imported and called
in-process, not shelled out to) + extraction (extract_circular.py's own
main(), also called in-process) + graph load + chunk embedding, all in one
Python process and one command - same "one file, one command" pattern
ingest_act.py/ingest_caselaw.py already established for the Act and case law.

WHY THIS ONE ALSO DOES THE PDF->MD STEP (unlike ingest_act.py/ingest_
caselaw.py, which deliberately leave conversion as a separate manual step):
the Act and case law each have their OWN dedicated, document-type-specific
PDF converter (pdf_to_md_act.py, pdf_to_md_caselaw.py) with real per-type
logic worth keeping as a separately-runnable, inspectable step. Circulars
have no such dedicated converter - pdf_to_md_generic.py (a thin pdftotext
-layout wrapper, already built for exactly this "no special structure"
case) is the right tool, and folding its one function call in here removes
a manual step with no corresponding loss of a real conversion stage to
inspect. --md is still accepted directly for a circular that's already been
converted (or hand-corrected after OCR garbling - see extract_circular.py's
module docstring on that risk).

Usage:
    # From a raw PDF (recommended - single command, PDF straight to Neo4j):
    python ingest_circular.py --pdf ../data/rawdata/circulars/Circular-3-2026.pdf \\
        --document-id circular_3_2026 --document-name "CBDT Circular No. 3/2026"

    # From an already-converted .md/.txt:
    python ingest_circular.py --md ../data/circular_md/circular-3-2026.md \\
        --document-id circular_3_2026 --document-name "CBDT Circular No. 3/2026"

    # Whole folder of circular PDFs in one run:
    python ingest_circular.py --pdf-dir ../data/rawdata/circulars
"""
import argparse
import sys
from pathlib import Path

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))

import extract_circular  # same directory - imported and called, not duplicated
import pdf_to_md_generic  # same directory - convert_one(), called in-process (see docstring above)
from framework.core.infra import step_logger

SCRIPT_DIR = Path(__file__).resolve().parent


def _convert_if_needed(pdf_path: Path) -> Path:
    """Runs pdf_to_md_generic's own conversion function in-process and
    writes the result alongside the source PDF (same *.md naming convention
    every other converter in this project uses) - returns the .md path."""
    md_path = pdf_path.with_suffix(".md")
    step_logger.log(f"Converting {pdf_path.name} -> {md_path.name} (generic PDF converter)")
    text, stats = pdf_to_md_generic.convert_one(pdf_path)
    md_path.write_text(text, encoding="utf-8")
    step_logger.log(f"  {stats}")
    return md_path


def _ingest_one(md_path: Path, args) -> None:
    out_dir = md_path.parent
    graph_path = out_dir / f"{md_path.stem}_graph.json"
    chunks_path = out_dir / f"{md_path.stem}_chunks.json"

    step_logger.log(f"=== ingest_circular.py: {md_path.name} -> Neo4j ===")

    extract_argv = [
        "extract_circular.py", "--md", str(md_path), "--schema", args.schema,
        "--out", str(graph_path), "--chunks-out", str(chunks_path),
        "--file-type", args.file_type, "--source-type", args.source_type,
        "--authority-level", args.authority_level, "--jurisdiction", args.jurisdiction,
        "--act-1961-namespace", args.act_1961_namespace, "--act-2025-namespace", args.act_2025_namespace,
    ]
    if args.document_id:
        extract_argv += ["--document-id", args.document_id]
    if args.document_name:
        extract_argv += ["--document-name", args.document_name]
    if args.skip_datasource:
        extract_argv.append("--skip-datasource")
    if args.debug:
        extract_argv.append("--debug")
    if args.verbose:
        extract_argv.append("--verbose")

    old_argv = sys.argv
    try:
        sys.argv = extract_argv
        extract_circular.main()
    finally:
        sys.argv = old_argv

    if not graph_path.exists():
        raise RuntimeError(f"Extraction did not produce {graph_path} - see the log above for what went wrong.")

    from domain_packs.tax.config import DOMAIN_CONFIG
    from framework.core.ingest import pipeline_ops

    step_logger.log(f"=== Loading {graph_path.name} into Neo4j ===")
    ingest_stats = pipeline_ops.ingest_graph_json(
        DOMAIN_CONFIG, str(graph_path), str(chunks_path) if chunks_path.exists() else None, "base"
    )
    step_logger.log(f"Graph loaded: {ingest_stats}")

    if args.skip_embed or not chunks_path.exists():
        step_logger.log("Skipping embedding (--skip-embed set, or no chunks file was produced).")
        return
    step_logger.log(f"=== Embedding {chunks_path.name} ===")
    embed_stats = pipeline_ops.embed_chunks_json(DOMAIN_CONFIG, str(chunks_path))
    step_logger.log(f"Embedding complete: {embed_stats}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pdf", default=None, help="Single circular PDF - converted in-process, then ingested.")
    parser.add_argument("--md", default=None, help="Single already-converted circular .md/.txt file.")
    parser.add_argument("--pdf-dir", default=None, help="Ingest every .pdf in this folder, one after another.")
    parser.add_argument("--schema", default=str(SCRIPT_DIR.parent / "ontology" / "schema.json"))
    parser.add_argument("--file-type", default="circular_pdf")
    parser.add_argument("--document-id", default=None, help="Only meaningful with --md/--pdf (single file).")
    parser.add_argument("--document-name", default=None, help="Only meaningful with --md/--pdf (single file).")
    parser.add_argument("--source-type", default="Circular")
    parser.add_argument("--authority-level", default="Delegated")
    parser.add_argument("--jurisdiction", default="India")
    parser.add_argument("--act-1961-namespace", default="ITA1961")
    parser.add_argument("--act-2025-namespace", default="ITA2025")
    parser.add_argument("--skip-datasource", action="store_true")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    if sum(bool(x) for x in (args.pdf, args.md, args.pdf_dir)) != 1:
        raise SystemExit("Provide exactly one of --pdf, --md, or --pdf-dir.")

    if args.pdf_dir:
        pdf_dir = Path(args.pdf_dir)
        pdf_files = sorted(pdf_dir.glob("*.pdf")) + sorted(pdf_dir.glob("*.PDF"))
        if not pdf_files:
            raise SystemExit(f"No .pdf files found in {pdf_dir}")
        step_logger.log(f"Found {len(pdf_files)} circular PDF(s) to ingest")
        for pdf_path in pdf_files:
            md_path = _convert_if_needed(pdf_path)
            _ingest_one(md_path, args)
    elif args.pdf:
        md_path = _convert_if_needed(Path(args.pdf))
        _ingest_one(md_path, args)
    else:
        _ingest_one(Path(args.md), args)

    step_logger.log("=== ingest_circular.py: done. ===")


if __name__ == "__main__":
    main()
