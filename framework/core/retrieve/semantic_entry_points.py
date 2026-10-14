"""
Generic per-node semantic entry-point search - the reusable engine behind
the "embed a short text field on each node, search it to find topical entry
points, then traverse the graph from there" pattern used by the
semantic_graph retrieval mode (see Semantic_Graph_Retrieval_Architecture.md).

2026-08-29: REWRITTEN. Embeddings are NOT stored on Neo4j node properties
and NOT searched via Neo4j's native vector index anymore - per an explicit
requirement, Neo4j holds ONLY graph structure, metadata, and short text
summaries; no vectors, anywhere. Embeddings now live in a small local flat
file per index (see local_vector_store.py), searched via brute-force numpy
cosine similarity. Neo4j is used for exactly two things here: (1) reading
each node's text properties to embed (embed_nodes()), and (2) ONE follow-up
lookup, BY THE db_ids the local search already picked, to fetch each hit's
requested return_fields (search_entry_points()) - Neo4j never sees or stores
a vector.

case_law_entry_points.py now ALSO routes through this engine (previously
left as a separate direct-Neo4j-vector-index implementation - see that
file's docstring for why that's no longer an option now that storing ANY
vector on a Neo4j node is prohibited outright).
"""
from framework.core.infra import neo4j_client, step_logger
from framework.core.ingest.embedder import embed_batch
from framework.core.retrieve import local_vector_store


def setup_entry_point_index(session=None, index_name: str = "", label=None,
                             property_name: str = "retrieval_embedding"):
    """No-op - kept only so existing call sites (framework/cli/embed_*.py)
    don't need editing. There is nothing to "set up" for a local flat-file
    store; embed_nodes() below creates/updates the file directly, and Neo4j
    is never touched by this function."""
    return


def embed_nodes(session, label, index_name: str, embedding_text_fn, text_property_names: list[str],
                 where_not_empty_property: str, batch_size: int = 32,
                 property_name: str = "retrieval_embedding") -> int:
    """embedding_text_fn(record: dict) -> str builds the text to embed from
    whatever properties this label's nodes carry (record's keys match
    text_property_names). Only embeds nodes where `where_not_empty_property`
    is set and non-empty.

    `label` may be a single Neo4j label (str) or several (list[str] - e.g.
    circular_entry_points.py's 3 Circular subtypes, which can now all share
    ONE local store since a flat file doesn't care about Neo4j's
    single-label-per-vector-index restriction the way the old native-index
    design did).

    `property_name` is accepted for call-site signature compatibility with
    the pre-2026-08-29 (Neo4j-native-index) design but is now unused - no
    Neo4j property gets written by this function."""
    labels = [label] if isinstance(label, str) else list(label)
    safe_where_prop = neo4j_client.safe_identifier(where_not_empty_property)
    read_props = ", ".join(f"n.{neo4j_client.safe_identifier(p)} AS {neo4j_client.safe_identifier(p)}"
                            for p in text_property_names)

    rows = []
    for lbl in labels:
        safe_label = neo4j_client.safe_identifier(lbl)
        rows.extend(list(session.run(
            f"MATCH (n:{safe_label}) WHERE coalesce(n.{safe_where_prop}, '') <> '' "
            f"RETURN elementId(n) AS db_id, {read_props}"
        )))
    label_desc = "/".join(labels)
    if not rows:
        step_logger.log(f"No {label_desc} nodes with a populated {where_not_empty_property} found - "
                         f"nothing to embed.")
        return 0

    n_batches = (len(rows) + batch_size - 1) // batch_size
    all_db_ids: list[str] = []
    all_embeddings: list[list[float]] = []
    with step_logger.Step(f"Embedding {len(rows)} {label_desc} entry-point node(s) ({n_batches} batches)"):
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            texts = [embedding_text_fn(dict(r)) for r in batch]
            embeddings = embed_batch(texts)
            all_db_ids.extend(r["db_id"] for r in batch)
            all_embeddings.extend(embeddings)

    total = local_vector_store.merge_and_save(index_name, all_db_ids, all_embeddings)
    step_logger.log(f"Done. {len(all_db_ids)} {label_desc} node(s) embedded this run "
                     f"({total} total now in the '{index_name}' local vector store - "
                     f"{local_vector_store.describe_path(index_name)}). No vectors were written to Neo4j.")
    return len(all_db_ids)


def search_entry_points(index_name: str, question: str, top_k: int, return_fields: dict) -> list[dict]:
    """return_fields: {output_key: node_property_name} - lets each caller
    pick which properties come back without this module needing to know
    per-label field names in advance.

    The local flat-file store picks the candidate db_ids first (pure numpy,
    no Neo4j involved); Neo4j is then queried ONLY for those specific ids'
    metadata (`MATCH (n) WHERE elementId(n) IN $db_ids` - label-agnostic, so
    this works whether index_name's store mixes multiple Neo4j labels or
    not, e.g. circular_entry_points.py's 3 subtypes sharing one store)."""
    embedding = embed_batch([question])[0]
    hits = local_vector_store.search(index_name, embedding, top_k)
    if not hits:
        return []
    similarity_by_id = {db_id: score for db_id, score in hits}

    return_clause = ", ".join(
        f"n.{neo4j_client.safe_identifier(prop)} AS {neo4j_client.safe_identifier(out)}"
        for out, prop in return_fields.items()
    )
    with neo4j_client.get_session() as session:
        rows = list(session.run(
            f"MATCH (n) WHERE elementId(n) IN $db_ids RETURN elementId(n) AS db_id, {return_clause}",
            db_ids=list(similarity_by_id.keys()),
        ))
    results = [dict(r) for r in rows]
    for r in results:
        r["similarity"] = similarity_by_id.get(r["db_id"], 0.0)
    results.sort(key=lambda r: r["similarity"], reverse=True)
    return results
