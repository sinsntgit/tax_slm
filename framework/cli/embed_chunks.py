"""
Generic embedding CLI - domain-parameterized replacement for
tax_kg_poc/scripts/05_embed_chunks.py. Independent of ingest_graph.py's run
order (both MERGE the same Chunk node by chunk_id).

Usage:
    python -m framework.cli.embed_chunks --domain tax --chunks ontology/chunks.json
"""
import argparse
import json

from framework.cli.pack_loader import load_domain_pack
from framework.core.infra import neo4j_client, step_logger
from framework.core.ingest.embedder import ChunkEmbedder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default=None)
    parser.add_argument("--chunks", required=True, help="Path to chunks.json produced by the domain's extraction step")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    domain_config, _prompt_pack = load_domain_pack(args.domain)
    embedder = ChunkEmbedder(domain_config)

    with open(args.chunks, "r", encoding="utf-8") as f:
        chunk_records = json.load(f)

    with step_logger.Step("Connecting to Neo4j"):
        neo4j_client.verify_connectivity()

    with neo4j_client.get_session() as session:
        with step_logger.Step("Setting up Chunk.embedding vector index"):
            embedder.setup_vector_index(session)
        embedder.embed_and_store(session, chunk_records, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
