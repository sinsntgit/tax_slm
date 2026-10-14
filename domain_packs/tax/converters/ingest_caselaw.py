"""
Single-command case-law ingestion: run THIS ONE FILE and it does extraction
(extract_case_law.py's LLM structured-extraction pass - imported and called
in-process below, not shelled out to, and not duplicated) + graph load +
chunk embedding, for either one case (--md) or a whole folder of cases
(--dir), all in one Python process. Replaces the previous 3-command
sequence (extract_case_law.py, then `python -m framework.cli.ingest_graph`
once per file, then `python -m framework.cli.embed_chunks` once per file).

Same design note as ingest_act.py (see that file): extract_case_law.py
itself was NOT deleted/inlined-away, because pipeline.py's webapp Ingest
flow still subprocesses it directly. This script imports and calls its
existing main() in-process instead of duplicating ~900 lines of extraction
logic, then continues into the same in-process load/embed helpers the
webapp uses (framework.core.ingest.pipeline_ops).

Usage:
    # One case:
    python ingest_caselaw.py --md ../data/case_law_md/Reckitt_Benckiser.md \\
        --out ../data/case_law_graph/Reckitt_Benckiser_graph.json

    # A whole folder (matches extract_case_law.py's --dir/--out-dir batch mode):
    python ingest_caselaw.py --dir ../data/case_law_md --out-dir ../data/case_law_graph
"""
import argparse
import sys
from pathlib import Path

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))

import extract_case_law  # same directory - imported and called, not duplicated (see docstring above)
from framework.core.infra import step_logger

SCRIPT_DIR = Path(__file__).resolve().parent


def _load_and_embed_pair(domain_config, pipeline_ops, graph_path: Path, chunks_path: Path, mode: str):
    if not graph_path.exists():
        step_logger.log(f"  SKIP load/embed - {graph_path} was never produced (see extraction log above).")
        return
    step_logger.log(f"  Loading {graph_path.name} into Neo4j (mode={mode})...")
    ingest_stats = pipeline_ops.ingest_graph_json(
        domain_config, str(graph_path), str(chunks_path) if chunks_path.exists() else None, mode
    )
    step_logger.log(f"  Graph loaded: {ingest_stats}")
    if not chunks_path.exists():
        step_logger.log(f"  No chunks file for {graph_path.name} - skipping embedding.")
        return
    embed_stats = pipeline_ops.embed_chunks_json(domain_config, str(chunks_path))
    step_logger.log(f"  Embedding complete: {embed_stats}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--md", type=Path, default=None, help="Single case-law .md file")
    parser.add_argument("--out", type=Path, default=None, help="Output graph JSON path (with --md)")
    parser.add_argument("--dir", type=Path, default=None, help="Process every .md file in this folder")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output folder for graph JSON files (with --dir)")
    parser.add_argument("--raw-out-dir", type=Path, default=None)
    parser.add_argument("--chunks-out", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=SCRIPT_DIR.parent / "ontology" / "schema.json")
    parser.add_argument("--file-type", default="caselaw_pdf")
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--document-name", default=None)
    parser.add_argument("--source-type", default="CaseLaw")
    parser.add_argument("--authority-level", default="Judicial")
    parser.add_argument("--jurisdiction", default="India")
    parser.add_argument("--cited-act-namespace", default=None,
                         help="Which Act's id-namespace this batch's citations resolve against - defaults "
                              "to extract_case_law.DEFAULT_CITED_ACT_NAMESPACE if omitted.")
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--mode", choices=["base", "amendment"], default="base")
    parser.add_argument("--skip-embed", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    if not args.md and not args.dir:
        raise SystemExit("Provide either --md/--out or --dir/--out-dir")

    step_logger.log(f"=== ingest_caselaw.py: {'batch ' + str(args.dir) if args.dir else args.md} "
                     f"-> Neo4j (mode={args.mode}) ===")

    # --- Step 1/3: extraction (extract_case_law.py's own main(), called in-process) ---
    extract_argv = ["extract_case_law.py", "--schema", str(args.schema), "--file-type", args.file_type,
                     "--source-type", args.source_type, "--authority-level", args.authority_level,
                     "--jurisdiction", args.jurisdiction]
    if args.md:
        extract_argv += ["--md", str(args.md)]
        if not args.out:
            raise SystemExit("--out is required with --md")
        extract_argv += ["--out", str(args.out)]
    if args.dir:
        extract_argv += ["--dir", str(args.dir)]
        if not args.out_dir:
            raise SystemExit("--out-dir is required with --dir")
        extract_argv += ["--out-dir", str(args.out_dir)]
    for condition, flag, value in [
        (args.raw_out_dir, "--raw-out-dir", str(args.raw_out_dir) if args.raw_out_dir else None),
        (args.chunks_out, "--chunks-out", str(args.chunks_out) if args.chunks_out else None),
        (args.document_id, "--document-id", args.document_id),
        (args.document_name, "--document-name", args.document_name),
        (args.cited_act_namespace, "--cited-act-namespace", args.cited_act_namespace),
        (args.max_chars, "--max-chars", str(args.max_chars) if args.max_chars else None),
    ]:
        if condition:
            extract_argv += [flag, value]
    for flag_name, is_set in [("--debug", args.debug), ("--verbose", args.verbose)]:
        if is_set:
            extract_argv.append(flag_name)

    old_argv = sys.argv
    try:
        sys.argv = extract_argv
        extract_case_law.main()
    finally:
        sys.argv = old_argv

    # --- Step 2-3/3: load + embed every graph/chunks pair extraction produced ---
    from domain_packs.tax.config import DOMAIN_CONFIG
    from framework.core.ingest import pipeline_ops

    if args.skip_embed:
        step_logger.log("--skip-embed set, but load still runs - only the final embedding step is skipped "
                         "per-file below.")

    if args.md:
        chunks_path = args.chunks_out or (args.out.parent / (args.md.stem + "_chunks.json"))
        if args.skip_embed:
            if args.out.exists():
                ingest_stats = pipeline_ops.ingest_graph_json(
                    DOMAIN_CONFIG, str(args.out), str(chunks_path) if chunks_path.exists() else None, args.mode
                )
                step_logger.log(f"Graph loaded: {ingest_stats}")
        else:
            _load_and_embed_pair(DOMAIN_CONFIG, pipeline_ops, args.out, chunks_path, args.mode)
    else:
        md_files = sorted(args.dir.glob("*.md"))
        step_logger.log(f"Loading + embedding {len(md_files)} case(s)...")
        for i, md_path in enumerate(md_files, start=1):
            graph_path = args.out_dir / (md_path.stem + "_graph.json")
            chunks_path = args.out_dir / (md_path.stem + "_chunks.json")
            step_logger.log(f"[{i}/{len(md_files)}] {md_path.name}")
            if args.skip_embed:
                if graph_path.exists():
                    ingest_stats = pipeline_ops.ingest_graph_json(
                        DOMAIN_CONFIG, str(graph_path), str(chunks_path) if chunks_path.exists() else None, args.mode
                    )
                    step_logger.log(f"  Graph loaded: {ingest_stats}")
                else:
                    step_logger.log(f"  SKIP - {graph_path} was never produced.")
            else:
                _load_and_embed_pair(DOMAIN_CONFIG, pipeline_ops, graph_path, chunks_path, args.mode)

    step_logger.log("=== ingest_caselaw.py: done. ===")


if __name__ == "__main__":
    main()
