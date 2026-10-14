"""
High-level, single-call ingestion helpers - wraps GraphIngestionEngine +
ChunkEmbedder for callers (the webapp's background ingestion job) that want
"give me a graph.json + chunks.json, get back a graph" without duplicating
framework/cli/ingest_graph.py's / embed_chunks.py's argparse-driven main()
bodies. Those two CLIs remain the direct, scriptable entry points for
command-line use; this module is what framework/webapp/app.py calls
in-process instead of shelling out to them (shelling out to a converter
script is fine/expected - see domain_packs/tax/pipeline.py - but the graph
load itself has no sys.path/import quirks to work around, so in-process is
strictly better here: faster, and errors surface as real Python exceptions
instead of parsed subprocess output).
"""
import json
import os

from framework.core.infra import audit_log, neo4j_client


def ingest_graph_json(domain_config, graph_json_path: str, chunks_json_path: str | None, mode: str) -> dict:
    # Imported here, not at module top: GraphIngestionEngine itself has no
    # heavy deps, but keeping ALL of pipeline_ops's imports lazy means this
    # module (and anything that imports it, like domain_packs/tax/pipeline.py
    # and, transitively, domain_packs/tax/pack.py) stays importable even in
    # an environment where sentence-transformers (embedder.py's dependency,
    # see embed_chunks_json() below) isn't installed yet - e.g. just to read
    # DOMAIN_CONFIG/PROMPT_PACK without running any ML code.
    from framework.core.ingest.graph_loader import GraphIngestionEngine
    with open(graph_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict) or "entities" not in data or "relationships" not in data:
        raise ValueError(f"{graph_json_path} doesn't look like a graph JSON file (expected entities/relationships).")

    chunks_path = chunks_json_path or os.path.join(os.path.dirname(graph_json_path) or ".", "chunks.json")
    chunk_records = []
    if os.path.exists(chunks_path):
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunk_records = json.load(f)

    document_id = None
    if data.get("entities"):
        document_id = data["entities"][0].get("properties", {}).get("document_id")
    run_id = audit_log.start_run(
        script="framework.core.ingest.pipeline_ops.ingest_graph_json",
        document_id=document_id, args={"mode": mode, "in": graph_json_path},
    )

    engine = GraphIngestionEngine(domain_config)
    neo4j_client.verify_connectivity()
    resolved_id_map: dict[str, str] = {}
    with neo4j_client.get_session() as session:
        created_labels, resolved_id_map = engine.load_entities(session, data["entities"], mode)
        if resolved_id_map:
            for ent in data["entities"]:
                ent["id"] = resolved_id_map.get(ent["id"], ent["id"])
            for rel in data["relationships"]:
                rel["source"] = resolved_id_map.get(rel["source"], rel["source"])
                rel["target"] = resolved_id_map.get(rel["target"], rel["target"])
        engine.load_chunk_nodes_and_links(session, data["entities"], chunk_records)
        engine.load_relationships(session, data["relationships"])

    stats = {
        "entities_in_file": len(data["entities"]),
        "relationships_in_file": len(data["relationships"]),
        "cross_run_merges": len(resolved_id_map),
        "labels_touched": sorted(created_labels),
        "chunk_records_found": len(chunk_records),
    }
    audit_log.finish_run(run_id, status="success", details=stats)
    return stats


def embed_chunks_json(domain_config, chunks_json_path: str, batch_size: int = 32) -> dict:
    from framework.core.ingest.embedder import ChunkEmbedder  # see the lazy-import note above

    with open(chunks_json_path, "r", encoding="utf-8") as f:
        chunk_records = json.load(f)

    embedder = ChunkEmbedder(domain_config)
    neo4j_client.verify_connectivity()
    with neo4j_client.get_session() as session:
        embedder.setup_vector_index(session)
        embedder.embed_and_store(session, chunk_records, batch_size=batch_size)

    return {"chunks_embedded": len(chunk_records)}
