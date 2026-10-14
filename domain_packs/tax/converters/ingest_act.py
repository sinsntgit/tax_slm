"""
Single-command Act ingestion: run THIS ONE FILE and it does extraction
(chunking.py's deterministic parse + extract_entities.py's LLM semantic
layer - imported and called in-process below, not shelled out to, and not
duplicated - see the note above main()) + graph load + chunk embedding, all
in one Python process. Replaces the previous 3-command sequence
(extract_entities.py, then `python -m framework.cli.ingest_graph`, then
`python -m framework.cli.embed_chunks`) that a manual/CLI ingestion
previously required - the webapp's Ingest tab already did the equivalent
3-step sequence internally (see domain_packs/tax/pipeline.py's
_run_ingestion_job, which still exists and is unchanged - this script is
the same sequence, callable directly from a terminal).

WHY extract_entities.py ITSELF WASN'T DELETED (2026-08-27 design note):
pipeline.py's webapp Ingest flow invokes extract_entities.py as a
subprocess (deliberately - see pipeline_ops.py's own docstring on why
converter scripts are subprocessed but graph-loading isn't). Deleting
extract_entities.py or moving its logic here would have broken the live
webapp Ingest tab. Instead, this script IMPORTS extract_entities.py's
existing, already-tested main() and calls it in-process (no subprocess,
no separate command), then continues straight into the graph-load and
embed steps using the exact same in-process helpers
(framework.core.ingest.pipeline_ops) the webapp uses. Net effect for a
terminal user: ONE file, ONE command, ONE Python process, zero
duplicated extraction logic.

Usage:
    python ingest_act.py --md ../data/ITA2025.md --document-id ITA2025 \\
        --document-name "Income-tax Act, 2025" --act-namespace ITA2025 \\
        --act-year 2025 --effective-from 2026-04-01 --mode base

    python ingest_act.py --md ../data/ITA1961.md --document-id ITA1961 \\
        --document-name "Income-tax Act, 1961" --act-namespace ITA1961 --act-year 1961
"""
import argparse
import sys
from pathlib import Path

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))

import extract_entities  # same directory - imported and called, not duplicated (see docstring above)
from framework.core.infra import step_logger

SCRIPT_DIR = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- extraction args - same names/defaults as extract_entities.py's own, passed straight through ---
    parser.add_argument("--md", required=True)
    parser.add_argument("--schema", default=str(SCRIPT_DIR.parent / "ontology" / "schema.json"))
    parser.add_argument("--max-groups", type=int, default=None)
    parser.add_argument("--max-group-chars", type=int, default=60_000)
    parser.add_argument("--max-output-tokens", type=int, default=24_000)
    parser.add_argument("--extraction-workers", type=int, default=5)
    parser.add_argument("--include-schedules-in-llm", action="store_true")
    parser.add_argument("--skip-semantic-layer", action="store_true")
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--document-name", default="Income Tax Act, 1961")
    parser.add_argument("--source-type", default="Act")
    parser.add_argument("--authority-level", default="Statute")
    parser.add_argument("--jurisdiction", default="India")
    parser.add_argument("--file-type", default="act_pdf")
    parser.add_argument("--act-namespace", default=None,
                         help="e.g. 'ITA2025'/'ITA1961' - REQUIRED in practice if you're ingesting more than "
                              "one Act into the same graph (see chunking.generate_entity_id()'s docstring). "
                              "Defaults to --document-id if omitted.")
    parser.add_argument("--act-year", default=None, type=int)
    parser.add_argument("--act-number", default=None)
    parser.add_argument("--no-act-node", action="store_true")
    parser.add_argument("--effective-from", default=None)
    parser.add_argument("--source-class", default="StatutoryProvisionsSource")
    parser.add_argument("--source-name", default=None)
    parser.add_argument("--source-location", default=None)
    parser.add_argument("--source-notes", default=None)
    parser.add_argument("--priority-tier", default=None)
    parser.add_argument("--update-frequency", default=None)
    parser.add_argument("--skip-datasource", action="store_true")
    parser.add_argument("--skip-entity-resolution", action="store_true")
    parser.add_argument("--review-flags-out", default=None)
    parser.add_argument("--chunking-ambiguities-out", default=None)
    parser.add_argument("--debug", action="store_true")
    # --- graph-load / embed args ---
    parser.add_argument("--mode", choices=["base", "amendment"], default="base")
    parser.add_argument("--skip-embed", action="store_true", help="Load the graph but skip embedding.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    md_path = Path(args.md)
    out_dir = md_path.parent
    graph_path = out_dir / f"{md_path.stem}_graph.json"
    chunks_path = out_dir / f"{md_path.stem}_chunks.json"

    step_logger.log(f"=== ingest_act.py: {md_path.name} -> Neo4j (mode={args.mode}) ===")

    # --- Step 1/3: extraction (extract_entities.py's own main(), called in-process) ---
    extract_argv = [
        "extract_entities.py",  # argv[0] placeholder, argparse ignores it
        "--md", str(md_path), "--schema", args.schema,
        "--out", str(graph_path), "--chunks-out", str(chunks_path),
        "--max-group-chars", str(args.max_group_chars), "--max-output-tokens", str(args.max_output_tokens),
        "--extraction-workers", str(args.extraction_workers),
        "--document-name", args.document_name, "--source-type", args.source_type,
        "--authority-level", args.authority_level, "--jurisdiction", args.jurisdiction,
        "--file-type", args.file_type, "--source-class", args.source_class,
    ]
    _flag_map = [
        (args.max_groups, "--max-groups", str(args.max_groups) if args.max_groups else None),
        (args.document_id, "--document-id", args.document_id),
        (args.act_namespace, "--act-namespace", args.act_namespace),
        (args.act_year, "--act-year", str(args.act_year) if args.act_year else None),
        (args.act_number, "--act-number", args.act_number),
        (args.effective_from, "--effective-from", args.effective_from),
        (args.source_name, "--source-name", args.source_name),
        (args.source_location, "--source-location", args.source_location),
        (args.source_notes, "--source-notes", args.source_notes),
        (args.priority_tier, "--priority-tier", args.priority_tier),
        (args.update_frequency, "--update-frequency", args.update_frequency),
        (args.review_flags_out, "--review-flags-out", args.review_flags_out),
        (args.chunking_ambiguities_out, "--chunking-ambiguities-out", args.chunking_ambiguities_out),
    ]
    for condition, flag, value in _flag_map:
        if condition:
            extract_argv += [flag, value]
    for flag_name, is_set in [
        ("--no-act-node", args.no_act_node), ("--skip-datasource", args.skip_datasource),
        ("--skip-entity-resolution", args.skip_entity_resolution),
        ("--include-schedules-in-llm", args.include_schedules_in_llm),
        ("--skip-semantic-layer", args.skip_semantic_layer),
        ("--debug", args.debug), ("--verbose", args.verbose),
    ]:
        if is_set:
            extract_argv.append(flag_name)

    old_argv = sys.argv
    try:
        sys.argv = extract_argv
        extract_entities.main()
    finally:
        sys.argv = old_argv

    if not graph_path.exists():
        raise RuntimeError(f"Extraction did not produce {graph_path} - see the log above for what went wrong.")

    # --- Step 2/3: load into Neo4j (same in-process helper the webapp uses - no subprocess) ---
    from domain_packs.tax.config import DOMAIN_CONFIG
    from framework.core.ingest import pipeline_ops

    step_logger.log(f"=== Loading {graph_path.name} into Neo4j (mode={args.mode}) ===")
    ingest_stats = pipeline_ops.ingest_graph_json(
        DOMAIN_CONFIG, str(graph_path), str(chunks_path) if chunks_path.exists() else None, args.mode
    )
    step_logger.log(f"Graph loaded: {ingest_stats}")

    # --- Step 3/3: embed chunks ---
    if args.skip_embed or not chunks_path.exists():
        step_logger.log("Skipping embedding (--skip-embed set, or no chunks file was produced).")
        step_logger.log("=== ingest_act.py: done. ===")
        return
    step_logger.log(f"=== Embedding {chunks_path.name} ===")
    embed_stats = pipeline_ops.embed_chunks_json(DOMAIN_CONFIG, str(chunks_path))
    step_logger.log(f"Embedding complete: {embed_stats}")
    step_logger.log("=== ingest_act.py: done. ===")


if __name__ == "__main__":
    main()
