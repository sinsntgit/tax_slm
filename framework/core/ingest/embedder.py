"""
Chunk embedding engine - GENERALIZED from tax_kg_poc/scripts/05_embed_chunks.py.

Embeds chunks.json records with a local sentence-transformers model and
writes `embedding`/`content` (plus a DomainConfig-supplied metadata
allowlist) onto each Chunk node's properties, searchable via Neo4j's native
vector index. Independent of ingestion run order - MERGEs by chunk_id, same
as the graph loader, so it converges to the same node regardless of which
engine ran first.

Usage (see framework/cli/embed_chunks.py):
    from framework.core.ingest.embedder import ChunkEmbedder
    embedder = ChunkEmbedder(domain_config)
    with neo4j_client.get_session() as session:
        embedder.setup_vector_index(session)
        embedder.embed_and_store(session, chunk_records)
"""
import os
import time

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from framework.core.infra import neo4j_client, step_logger
from framework.core.infra.dotenv_path import env_path as _env_path

# Anchored to the repo root by file location, not process cwd - see
# framework.core.infra.dotenv_path's docstring.
load_dotenv(_env_path())

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
VECTOR_INDEX_NAME = "chunk_embedding_index"

_model = None
_embedding_dim = None


def _get_model():
    """Lazy singleton, same reasoning as llm_client's lazy client - importing
    this module (e.g. just to read VECTOR_INDEX_NAME) shouldn't force a
    multi-hundred-MB model download/load."""
    global _model, _embedding_dim
    if _model is None:
        with step_logger.Step(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' "
                               f"(downloads once, then cached locally)"):
            _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            _embedding_dim = _model.get_embedding_dimension()
            step_logger.log(f"Model ready. Embedding dimension: {_embedding_dim}")
    return _model


def embedding_dimension() -> int:
    _get_model()
    return _embedding_dim


def default_embedding_text(record: dict) -> str:
    """Default embedding-text builder: short_title prepended to content, per
    the chunking-strategy spec. A domain pack can override this via
    DomainConfig.embedding_text_fn."""
    short_title = (record.get("short_title") or "").strip()
    content = record.get("text") or record.get("content") or ""
    return f"{short_title}. {content}" if short_title else content


def embed_batch(texts: list[str]) -> list[list[float]]:
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


class ChunkEmbedder:
    def __init__(self, domain_config):
        self.domain_config = domain_config
        self.embedding_text_fn = domain_config.embedding_text_fn or default_embedding_text

    def setup_vector_index(self, session):
        neo4j_client.ensure_vector_index(
            session, VECTOR_INDEX_NAME, label="Chunk", property_name="embedding",
            dimensions=embedding_dimension(), similarity_function="cosine",
        )
        neo4j_client.ensure_property_index(session, "Chunk", "chunk_id")

    def embed_and_store(self, session, chunk_records: list[dict], batch_size: int = 32):
        metadata_keys = self.domain_config.embed_chunk_metadata_keys
        n_batches = (len(chunk_records) + batch_size - 1) // batch_size

        with step_logger.Step(f"Embedding and storing {len(chunk_records)} chunks "
                               f"({n_batches} batches of {batch_size})"):
            for batch_num, i in enumerate(tqdm(range(0, len(chunk_records), batch_size))):
                batch_t0 = time.time()
                batch = chunk_records[i:i + batch_size]
                texts = [self.embedding_text_fn(b) for b in batch]
                embeddings = embed_batch(texts)

                rows = []
                for record, embedding in zip(batch, embeddings):
                    row = {
                        "chunk_id": str(record["chunk_id"]),
                        "content": record.get("text", ""),
                        "embedding": embedding,
                    }
                    for k in metadata_keys:
                        v = record.get(k)
                        if v not in (None, "", []):
                            row[k] = v
                    rows.append(row)

                session.run(
                    """
                    UNWIND $rows AS row
                    MERGE (c:Chunk {chunk_id: row.chunk_id})
                    SET c += row
                    """,
                    rows=rows,
                )
                tqdm.write(step_logger.fmt(
                    f"  batch {batch_num + 1}/{n_batches} ({len(batch)} chunks) done in "
                    f"{time.time() - batch_t0:.1f}s"
                ))

        step_logger.log("Done. Chunks are embedded and stored on Chunk nodes; "
                         f"searchable via the '{VECTOR_INDEX_NAME}' vector index.")
