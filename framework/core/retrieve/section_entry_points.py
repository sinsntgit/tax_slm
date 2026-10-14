"""
Section-level semantic entry-point index - StatutoryProvision's equivalent
of case_law_entry_points.py, built on the shared semantic_entry_points.py
engine. See Semantic_Graph_Retrieval_Architecture.md.

Unlike CaseLaw (whose judgment text is long/discursive enough to need a
dedicated retrievalSummary field composed for retrieval - see
extract_case_law.py's derive_retrieval_summary()), an Act Section's own text
(hasText) plus its marginal_heading/full_citation is ALREADY concise,
already-summary-like statutory text - no separate derived summary field is
needed here, we embed the existing fields directly.
"""
from framework.core.infra import neo4j_client, step_logger
from framework.core.retrieve import semantic_entry_points as sep

SECTION_VECTOR_INDEX_NAME = "section_retrieval_index"
SECTION_LABEL = "Section"
SECTION_TEXT_PROPERTIES = ["full_citation", "marginal_heading", "hasText"]


def _embedding_text(record: dict) -> str:
    citation = (record.get("full_citation") or "").strip()
    heading = (record.get("marginal_heading") or "").strip()
    # Capped, same reasoning as case_law_entry_points.py - the embedding
    # model has its own input limits, and a Section's full hasText can run
    # long for a heavily-amended provision.
    text = (record.get("hasText") or "").strip()[:1500]
    parts = [p for p in (citation, heading, text) if p]
    return " ".join(parts)


def setup_section_vector_index(session):
    sep.setup_entry_point_index(session, SECTION_VECTOR_INDEX_NAME, label=SECTION_LABEL)


def embed_section_nodes(session, batch_size: int = 32) -> int:
    return sep.embed_nodes(
        session, label=SECTION_LABEL, index_name=SECTION_VECTOR_INDEX_NAME,
        embedding_text_fn=_embedding_text, text_property_names=SECTION_TEXT_PROPERTIES,
        where_not_empty_property="hasText", batch_size=batch_size,
    )


def search_section_entry_points(question: str, top_k: int = 15) -> list[dict]:
    return sep.search_entry_points(
        SECTION_VECTOR_INDEX_NAME, question, top_k,
        return_fields={
            "full_citation": "full_citation", "marginal_heading": "marginal_heading",
            "document_id": "document_id", "sourceMdPath": "sourceMdPath",
            "line_start": "line_start", "line_end": "line_end",
        },
    )
