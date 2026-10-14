"""
Single-command international tax agreement (DTAA/protocol/TIEA) ingestion:
run THIS ONE FILE and it does PDF->text conversion (pdf_to_md_generic.
convert_one(), in-process) + extraction (extract_treaty.py's own main(), in-
process) + graph load + chunk embedding, all in one Python process and one
command - same "one file, one command" pattern ingest_act.py/ingest_
caselaw.py/ingest_circular.py already established. See ingest_circular.py's
docstring for why folding PDF->text conversion in here (unlike the Act/case
law scripts) is the right call for a document type with no dedicated
converter of its own.

Usage:
    # From a raw PDF (recommended - single command, PDF straight to Neo4j):
    python ingest_treaty.py --pdf ../data/rawdata/international_agreements/Albania-Comprehensive-Agreements_....pdf \\
        --document-id treaty_albania --document-name "India-Albania DTAA"

    # From an already-converted .md/.txt:
    python ingest_treaty.py --md ../data/treaty_md/Albania.md \\
        --document-id treaty_albania --document-name "India-Albania DTAA"

    # Whole folder of treaty PDFs in one run (e.g. every country's DTAA at once):
    python ingest_treaty.py --pdf-dir ../data/rawdata/international_agreements
"""
import argparse
import sys
from pathlib import Path

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))

import extract_treaty  # same directory - imported and called, not duplicated
import pdf_to_md_generic  # same directory - convert_one(), called in-process
from framework.core.infra import step_logger

SCRIPT_DIR = Path(__file__).resolve().parent


def _convert_if_needed(pdf_path: Path) -> Path:
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

    step_logger.log(f"=== ingest_treaty.py: {md_path.name} -> Neo4j ===")

    extract_argv = [
        "extract_treaty.py", "--md", str(md_path), "--schema", args.schema,
        "--out", str(graph_path), "--chunks-out", str(chunks_path),
        "--file-type", args.file_type, "--source-type", args.source_type,
        "--authority-level", args.authority_level, "--jurisdiction", args.jurisdiction,
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
        extract_treaty.main()
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
    parser.add_argument("--pdf", default=None, help="Single treaty PDF - converted in-process, then ingested.")
    parser.add_argument("--md", default=None, help="Single already-converted treaty .md/.txt file.")
    parser.add_argument("--pdf-dir", default=None, help="Ingest every .pdf in this folder, one after another.")
    parser.add_argument("--schema", default=str(SCRIPT_DIR.parent / "ontology" / "schema.json"))
    parser.add_argument("--file-type", default="treaty_pdf")
    parser.add_argument("--document-id", default=None, help="Only meaningful with --md/--pdf (single file).")
    parser.add_argument("--document-name", default=None, help="Only meaningful with --md/--pdf (single file).")
    parser.add_argument("--source-type", default="TaxTreaty")
    parser.add_argument("--authority-level", default="Treaty")
    parser.add_argument("--jurisdiction", default="India")
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
        step_logger.log(f"Found {len(pdf_files)} treaty PDF(s) to ingest")
        for pdf_path in pdf_files:
            md_path = _convert_if_needed(pdf_path)
            _ingest_one(md_path, args)
    elif args.pdf:
        md_path = _convert_if_needed(Path(args.pdf))
        _ingest_one(md_path, args)
    else:
        _ingest_one(Path(args.md), args)

    step_logger.log("=== ingest_treaty.py: done. ===")


if __name__ == "__main__":
    main()
