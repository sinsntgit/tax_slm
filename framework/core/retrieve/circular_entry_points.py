"""
Circular-level semantic entry-point search - CBDTCircular/ExplanatoryCircular/
ExtensionCircular's equivalent of section_entry_points.py, built on the
shared semantic_entry_points.py engine. See
Semantic_Graph_Retrieval_Architecture.md.

Circulars are ingested under THREE distinct Neo4j labels depending on
subtype (see extract_circular.py's _SUBTYPE_TO_CLASS): CBDTCircular
(general, the default), ExplanatoryCircular, ExtensionCircular.
graph_loader.py's load_entities() stamps only ONE label per node - no shared
ancestor "Circular" label is added at ingestion time even though "Circular"
is their common ontology superclass.

2026-08-29: SIMPLIFIED. Originally this needed one Neo4j vector index PER
subtype label (Neo4j's native vector index syntax only supports a single
label per index) and merged/re-sorted results from all three at search time.
Now that embeddings live in a local flat file rather than Neo4j (see
local_vector_store.py/semantic_entry_points.py - no vectors are stored in
Neo4j at all anymore), that constraint is gone: a local store doesn't care
about Neo4j labels, so all 3 subtypes now share ONE store
(circular_retrieval_index) - embed_nodes()'s `label` param simply accepts
all 3 labels and reads/embeds nodes from each, and search_entry_points()'s
Neo4j follow-up lookup (`MATCH (n) WHERE elementId(n) IN $db_ids`) is
label-agnostic anyway, so no merge-then-sort step is needed here anymore.
"""
from framework.core.retrieve import semantic_entry_points as sep

CIRCULAR_VECTOR_INDEX_NAME = "circular_retrieval_index"
CIRCULAR_LABELS = ("CBDTCircular", "ExplanatoryCircular", "ExtensionCircular")
CIRCULAR_TEXT_PROPERTIES = ["full_citation", "subject", "applicability_summary", "hasText"]


def _embedding_text(record: dict) -> str:
    citation = (record.get("full_citation") or "").strip()
    subject = (record.get("subject") or "").strip()
    applicability = (record.get("applicability_summary") or "").strip()
    # Capped, same reasoning as section_entry_points.py - hasText can run
    # long for a wordy circular.
    text = (record.get("hasText") or "").strip()[:1500]
    parts = [p for p in (citation, subject, applicability, text) if p]
    return " ".join(parts)


def setup_circular_vector_indexes(session):
    sep.setup_entry_point_index(session, CIRCULAR_VECTOR_INDEX_NAME, label=CIRCULAR_LABELS)


def embed_circular_nodes(session, batch_size: int = 32) -> int:
    return sep.embed_nodes(
        session, label=CIRCULAR_LABELS, index_name=CIRCULAR_VECTOR_INDEX_NAME,
        embedding_text_fn=_embedding_text, text_property_names=CIRCULAR_TEXT_PROPERTIES,
        where_not_empty_property="hasText", batch_size=batch_size,
    )


def search_circular_entry_points(question: str, top_k: int = 15) -> list[dict]:
    return sep.search_entry_points(
        CIRCULAR_VECTOR_INDEX_NAME, question, top_k,
        return_fields={
            "full_citation": "full_citation", "subject": "subject",
            "document_id": "node_id", "sourceMdPath": "sourceMdPath",
        },
    )
