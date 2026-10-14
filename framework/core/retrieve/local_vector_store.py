"""
Local, non-Neo4j flat-file vector store for the semantic_graph entry-point
indexes (CaseLaw/Section/Circular/TaxTreaty) - see
Semantic_Graph_Retrieval_Architecture.md.

2026-08-29: NEW - replaces the earlier design's use of Neo4j's native
`db.index.vector.queryNodes` / a `retrieval_embedding` node property. Per an
explicit requirement: Neo4j must hold ONLY graph structure, metadata, and a
short text summary per node - NO embedding vectors, anywhere, ever.
Embeddings for entry-point semantic search now live entirely OUTSIDE Neo4j,
in one small local file per index (one .npz per entity type/index_name)
under VECTOR_STORE_DIR (env override) or <repo_root>/data/vector_store/ by
default.

This is a brute-force flat-file store, not an ANN library (FAISS/Chroma/
etc.) - deliberately: this project's entry-point corpora are small (tens to
low hundreds of nodes per entity type, not millions), so an exact numpy
cosine-similarity scan over the whole file is fast enough and avoids adding
a new heavyweight dependency for a POC. If a corpus ever grows large enough
that this becomes the bottleneck, swap search()'s internals for an ANN
library without changing any caller - save()/load()/search()/merge_and_save()
are the extension points; nothing outside this file needs to know the
on-disk format.

Vectors are stored exactly as produced by framework/core/ingest/embedder.py's
embed_batch() - already L2-normalized (SentenceTransformer's
normalize_embeddings=True) - so cosine similarity reduces to a plain dot
product, no extra normalization needed at search time.
"""
import os
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STORE_DIR = _REPO_ROOT / "data" / "vector_store"


def _store_dir() -> Path:
    override = os.environ.get("VECTOR_STORE_DIR")
    return Path(override) if override else DEFAULT_STORE_DIR


def _store_path(index_name: str) -> Path:
    return _store_dir() / f"{index_name}.npz"


def describe_path(index_name: str) -> str:
    return str(_store_path(index_name))


def exists(index_name: str) -> bool:
    return _store_path(index_name).exists()


def save(index_name: str, db_ids: list[str], embeddings: list[list[float]]) -> None:
    """Overwrites the store for `index_name` with EXACTLY these (db_id,
    embedding) pairs. Most callers want merge_and_save() instead (see
    below) so re-embedding after new ingestion doesn't drop earlier
    entries."""
    path = _store_path(index_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    ids_arr = np.array(db_ids, dtype=object)
    vecs_arr = np.array(embeddings, dtype=np.float32)
    np.savez_compressed(path, ids=ids_arr, vectors=vecs_arr)


def load(index_name: str):
    """Returns (db_ids: list[str], vectors: np.ndarray) or (None, None) if
    the store doesn't exist yet (e.g. the matching embed_* CLI hasn't been
    run) - callers should treat that the same as "index not found" did
    under the old Neo4j-native design: log and move on, not a hard error."""
    path = _store_path(index_name)
    if not path.exists():
        return None, None
    with np.load(path, allow_pickle=True) as data:
        return list(data["ids"]), data["vectors"]


def merge_and_save(index_name: str, db_ids: list[str], embeddings: list[list[float]]) -> int:
    """Upserts (db_id, embedding) pairs into the existing store for
    index_name (new db_ids appended, existing ones' vectors replaced) - what
    embed_nodes() in semantic_entry_points.py actually calls, so re-running
    an embed CLI after ingesting MORE documents doesn't wipe out entries for
    documents that weren't touched this run. Returns the total row count
    after merging."""
    existing_ids, existing_vecs = load(index_name)
    by_id = {}
    if existing_ids is not None:
        for db_id, vec in zip(existing_ids, existing_vecs):
            by_id[db_id] = vec
    for db_id, vec in zip(db_ids, embeddings):
        by_id[db_id] = np.array(vec, dtype=np.float32)
    all_ids = list(by_id.keys())
    all_vecs = [by_id[i] for i in all_ids]
    save(index_name, all_ids, all_vecs)
    return len(all_ids)


def list_all(index_name: str) -> list[str]:
    """2026-08-29: NEW - returns every db_id currently in this store (no
    query/similarity involved), for the Data Explorer's "Knowledge Index"
    tab, which needs to show what's IN an index, not what matches a
    question. Empty list if the store doesn't exist yet."""
    ids, _vecs = load(index_name)
    return list(ids) if ids is not None else []


def count(index_name: str) -> int:
    """Row count for an index without loading/returning the full id list -
    used by the Data Explorer's summary view."""
    ids, _vecs = load(index_name)
    return len(ids) if ids is not None else 0


def search(index_name: str, query_embedding: list[float], top_k: int) -> list[tuple[str, float]]:
    """Brute-force cosine-similarity (dot product - vectors pre-normalized,
    see module docstring) search. Returns [(db_id, similarity), ...] sorted
    descending, length <= top_k. Empty list if the store doesn't exist yet."""
    ids, vecs = load(index_name)
    if ids is None or len(ids) == 0:
        return []
    query = np.array(query_embedding, dtype=np.float32)
    scores = vecs @ query  # (N, dim) @ (dim,) -> (N,)
    k = min(top_k, len(ids))
    if k < len(ids):
        top_idx = np.argpartition(-scores, k - 1)[:k]
    else:
        top_idx = np.arange(len(ids))
    top_idx = top_idx[np.argsort(-scores[top_idx])]
    return [(ids[i], float(scores[i])) for i in top_idx]
