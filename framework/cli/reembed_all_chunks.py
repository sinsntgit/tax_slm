"""
Re-embeds EVERY existing Chunk node already in Neo4j with whatever model
EMBEDDING_MODEL_NAME currently points to - needed any time that model
changes (e.g. switching from all-MiniLM-L6-v2 to a domain-specific model
like law-ai/InLegalBERT), since framework/core/ingest/embedder.py's
ChunkEmbedder normally only embeds chunk_records freshly produced during
ingestion, not chunks that already exist in the graph from a PREVIOUS
embedding model.

This does NOT touch entities, documents, or chunk text/metadata - it only
overwrites each Chunk node's `embedding` property (and re-derives `content`
from the same node, unchanged) and then rebuilds the vector index at the new
model's dimension. No re-ingestion of source documents is needed.

Order matters: embeddings are rewritten FIRST, and the vector index is
(re)created AFTER - creating/recreating a vector index while some indexed
nodes still hold the OLD dimension's vectors and others hold the NEW
dimension's is exactly the inconsistent state neo4j_client.ensure_vector_index()'s
dimension check is designed to avoid; doing the embedding rewrite before
touching the index side-steps it entirely (nothing is indexed against a
mismatched dimension at any point).

Usage:
    python -m framework.cli.reembed_all_chunks --domain tax
    python -m framework.cli.reembed_all_chunks --domain tax --batch-size 16
"""
import argparse

from framework.cli.pack_loader import load_domain_pack
from framework.core.infra import neo4j_client, step_logger
from framework.core.ingest.embedder import ChunkEmbedder, EMBEDDING_MODEL_NAME


def fetch_all_chunk_records(session) -> list[dict]:
    """Pulls every existing Chunk node's own properties back out in the same
    shape ChunkEmbedder.embed_and_store() expects for a freshly-produced
    chunk_records list (chunk_id + text, plus whatever metadata keys already
    happen to be on the node - embed_and_store() only ever reads the keys in
    domain_config.embed_chunk_metadata_keys, so extra keys here are harmless)."""
    rows = list(session.run("MATCH (c:Chunk) RETURN c"))
    records = []
    for r in rows:
        props = dict(r["c"])
        chunk_id = props.get("chunk_id")
        if chunk_id is None:
            continue
        record = dict(props)  # includes `content` and every stored metadata key
        record["chunk_id"] = chunk_id
        record["text"] = props.get("content", "")
        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    domain_config, _prompt_pack = load_domain_pack(args.domain)
    embedder = ChunkEmbedder(domain_config)

    with step_logger.Step("Connecting to Neo4j"):
        neo4j_client.verify_connectivity()

    with neo4j_client.get_session() as session:
        with step_logger.Step("Reading every existing Chunk node from Neo4j"):
            chunk_records = fetch_all_chunk_records(session)
        step_logger.log(f"Found {len(chunk_records)} existing chunk(s) to re-embed with "
                         f"'{EMBEDDING_MODEL_NAME}'.")
        if not chunk_records:
            step_logger.log("Nothing to do - no Chunk nodes found.")
            return

        # Embeddings FIRST (see module docstring for why this order matters).
        embedder.embed_and_store(session, chunk_records, batch_size=args.batch_size)

        with step_logger.Step("Rebuilding Chunk.embedding vector index at the new model's dimension"):
            embedder.setup_vector_index(session)

    step_logger.log(f"Done. All {len(chunk_records)} chunk(s) now embedded with "
                     f"'{EMBEDDING_MODEL_NAME}' and searchable via the vector index.")


if __name__ == "__main__":
    main()
