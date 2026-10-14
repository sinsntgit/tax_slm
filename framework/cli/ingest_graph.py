"""
Generic ingestion CLI - domain-parameterized replacement for
tax_kg_poc/scripts/03_build_graph.py. Same behavior (base/amendment modes,
content-hash change detection, chunk-node linking, cross-run fuzzy entity
resolution), driven by whichever domain pack's DomainConfig is loaded - the
graph-loading logic itself lives in framework/core/ingest/graph_loader.py
and has no domain-specific code left in it.

Usage:
    # First ingestion of a domain's base document:
    python -m framework.cli.ingest_graph --domain tax --in ontology/extracted_graph.json --mode base

    # Ingesting an amending document (e.g. a Finance Act) that touches
    # entities already in the graph:
    python -m framework.cli.ingest_graph --domain tax --in ontology/finance_act_2024_graph.json --mode amendment
"""
import argparse
import json
import os

from framework.cli.pack_loader import load_domain_pack
from framework.core.infra import audit_log, neo4j_client, step_logger
from framework.core.ingest.graph_loader import GraphIngestionEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default=None, help="Domain pack name (default: FRAMEWORK_DOMAIN env var, or 'tax')")
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--mode", choices=["base", "amendment"], default="base")
    parser.add_argument("--chunks", default=None,
                         help="Path to chunks.json - lets Chunk nodes carry full metadata, not just chunk_id. "
                              "Defaults to chunks.json next to --in. Missing file is non-fatal.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    domain_config, _prompt_pack = load_domain_pack(args.domain)
    engine = GraphIngestionEngine(domain_config)

    with step_logger.Step(f"Reading {args.infile}"):
        with open(args.infile, "r", encoding="utf-8") as f:
            data = json.load(f)

    if not isinstance(data, dict) or "entities" not in data or "relationships" not in data:
        found_keys = list(data.keys()) if isinstance(data, dict) else [f"<{type(data).__name__}, not an object>"]
        raise SystemExit(
            f"{args.infile} doesn't look like a graph JSON file - expected top-level "
            f"\"entities\" and \"relationships\" keys, found: {found_keys}. A domain's "
            f"converter usually writes several files per document - point --in at the "
            f"one that has this exact shape."
        )

    chunks_path = args.chunks or os.path.join(os.path.dirname(args.infile) or ".", "chunks.json")
    chunk_records = []
    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunk_records = json.load(f)
        step_logger.log(f"Loaded {len(chunk_records)} chunk records with metadata from {chunks_path}")
    else:
        step_logger.log(f"No chunks.json found at {chunks_path} - Chunk nodes will be chunk_id-only.")

    document_id = None
    if data.get("entities"):
        document_id = data["entities"][0].get("properties", {}).get("document_id")
    run_id = audit_log.start_run(script="framework.cli.ingest_graph", document_id=document_id, args=vars(args))

    with step_logger.Step("Connecting to Neo4j"):
        neo4j_client.verify_connectivity()

    with neo4j_client.get_session() as session:
        with step_logger.Step(f"Loading {len(data['entities'])} entities (mode={args.mode})"):
            created_labels, resolved_id_map = engine.load_entities(session, data["entities"], args.mode)

        if resolved_id_map:
            for ent in data["entities"]:
                ent["id"] = resolved_id_map.get(ent["id"], ent["id"])
            for rel in data["relationships"]:
                rel["source"] = resolved_id_map.get(rel["source"], rel["source"])
                rel["target"] = resolved_id_map.get(rel["target"], rel["target"])
            step_logger.log(f"Cross-run entity resolution: rewrote {len(resolved_id_map)} id references "
                             f"to match existing graph nodes.")

        with step_logger.Step("Linking entities to their source Chunk nodes"):
            engine.load_chunk_nodes_and_links(session, data["entities"], chunk_records)

        with step_logger.Step(f"Loading {len(data['relationships'])} relationships"):
            engine.load_relationships(session, data["relationships"])

    step_logger.log("Done. Graph is ready to query.")

    audit_log.finish_run(run_id, status="success", details={
        "entities_in_file": len(data["entities"]),
        "relationships_in_file": len(data["relationships"]),
        "cross_run_merges": len(resolved_id_map),
    })


if __name__ == "__main__":
    main()
