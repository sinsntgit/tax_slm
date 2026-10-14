"""
TaxTreaty-level semantic entry-point index - StatutoryProvision/Section's
equivalent for treaties, built on the shared semantic_entry_points.py
engine. See Semantic_Graph_Retrieval_Architecture.md.

Unlike Circular (see circular_entry_points.py), TaxTreaty is ingested under a
single Neo4j label regardless of treaty_type (DTAA etc. - see
extract_treaty.py's build_graph_from_extraction(), entity type is always
"TaxTreaty"), so this follows the simpler single-index pattern from
section_entry_points.py directly - no multi-index merging needed.
"""
from framework.core.infra import neo4j_client, step_logger
from framework.core.retrieve import semantic_entry_points as sep

TREATY_VECTOR_INDEX_NAME = "treaty_retrieval_index"
TREATY_LABEL = "TaxTreaty"
TREATY_TEXT_PROPERTIES = ["full_citation", "treatyCountry", "treaty_type", "hasText"]


def _embedding_text(record: dict) -> str:
    citation = (record.get("full_citation") or "").strip()
    country = (record.get("treatyCountry") or "").strip()
    treaty_type = (record.get("treaty_type") or "").strip()
    # Capped, same reasoning as section_entry_points.py - hasText can run
    # long for a treaty's captured preamble/full text slice.
    text = (record.get("hasText") or "").strip()[:1500]
    parts = [p for p in (citation, country, treaty_type, text) if p]
    return " ".join(parts)


def setup_treaty_vector_index(session):
    sep.setup_entry_point_index(session, TREATY_VECTOR_INDEX_NAME, label=TREATY_LABEL)


def embed_treaty_nodes(session, batch_size: int = 32) -> int:
    return sep.embed_nodes(
        session, label=TREATY_LABEL, index_name=TREATY_VECTOR_INDEX_NAME,
        embedding_text_fn=_embedding_text, text_property_names=TREATY_TEXT_PROPERTIES,
        where_not_empty_property="hasText", batch_size=batch_size,
    )


def search_treaty_entry_points(question: str, top_k: int = 15) -> list[dict]:
    return sep.search_entry_points(
        TREATY_VECTOR_INDEX_NAME, question, top_k,
        return_fields={
            "full_citation": "full_citation", "treatyCountry": "treatyCountry",
            "document_id": "node_id", "sourceMdPath": "sourceMdPath",
        },
    )
