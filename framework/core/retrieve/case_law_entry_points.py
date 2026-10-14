"""
CaseLaw entry-point semantic search - the "graph-first retrieval" pilot's
metadata-only search surface, distinct from the existing Chunk-level
`chunk_embedding_index` the general hybrid pipeline uses.

See Graph_First_Retrieval_Design.md (project root) for the original design,
and Semantic_Graph_Retrieval_Architecture.md for how this feeds
retrieval_mode="semantic_graph"'s live MD-file hydration.

2026-08-29: REWRITTEN onto the shared semantic_entry_points.py engine (was
previously a separate direct-Neo4j-vector-index implementation, deliberately
left alone when section_entry_points.py/etc. were built - see that module's
original history). That tradeoff no longer applies: storing ANY vector on a
Neo4j node property is now prohibited outright (Neo4j holds only graph
structure, metadata, and short text summaries), so this had to move onto
the local-flat-file engine regardless of migration risk.

This migration also fixes a latent bug it surfaced: the OLD Cypher RETURN
clause here never included `sourceMdPath`, so graphrag.py's
semantic_entry_point_search() was silently unable to hydrate ANY case-law
hit (md_hydration.hydrate_text("") always short-circuits to a skip) - this
was masked in testing because the mocked test for that method mocked
search_case_law_entry_points() directly rather than exercising the real
Cypher. Fixed here by adding sourceMdPath to return_fields below.

Embeds each CaseLaw node's `retrievalSummary`/`retrievalKeywords` (composed
DETERMINISTICALLY at ingestion/backfill time - see extract_case_law.py's
derive_retrieval_summary()/derive_retrieval_keywords() - NOT a fresh
LLM-authored abstract).

Also used directly by category_retrieval.py's _case_law_hierarchy_retrieval()
(the stance-balanced for/against path) for topical entry-point discovery -
that caller only reads `db_id` off each hit, so it's unaffected by this
migration.

Usage (see framework/cli/embed_case_law.py for a CLI wrapper):
    from framework.core.retrieve import case_law_entry_points as clep
    with neo4j_client.get_session() as session:
        clep.setup_case_law_vector_index(session)   # no-op, kept for compatibility
        clep.embed_case_law_nodes(session)
    hits = clep.search_case_law_entry_points("software payments taxable as royalty", top_k=15)
"""
from framework.core.retrieve import semantic_entry_points as sep

CASE_LAW_VECTOR_INDEX_NAME = "case_law_retrieval_index"
CASE_LAW_LABEL = "CaseLaw"
CASE_LAW_TEXT_PROPERTIES = ["retrievalSummary", "retrievalKeywords"]


def _embedding_text(record: dict) -> str:
    parts = [p for p in (record.get("retrievalSummary") or "", record.get("retrievalKeywords") or "") if p]
    return " ".join(parts)


def setup_case_law_vector_index(session):
    sep.setup_entry_point_index(session, CASE_LAW_VECTOR_INDEX_NAME, label=CASE_LAW_LABEL)


def embed_case_law_nodes(session, batch_size: int = 32) -> int:
    """Only embeds CaseLaw nodes with a non-empty retrievalSummary (stub
    precedent nodes and any not yet backfilled are skipped - an embedding of
    an empty string is not a meaningful entry point). Returns the number of
    nodes embedded this run."""
    return sep.embed_nodes(
        session, label=CASE_LAW_LABEL, index_name=CASE_LAW_VECTOR_INDEX_NAME,
        embedding_text_fn=_embedding_text, text_property_names=CASE_LAW_TEXT_PROPERTIES,
        where_not_empty_property="retrievalSummary", batch_size=batch_size,
    )


def search_case_law_entry_points(question: str, top_k: int = 15) -> list[dict]:
    """Vector search over the CaseLaw entry-point index alone - callers that
    want stance-balanced coverage should NOT rely on this alone (see
    category_retrieval.py's _case_law_hierarchy_retrieval(), which
    complements this with an explicit stance-split Cypher pull rather than
    replacing it)."""
    return sep.search_entry_points(
        CASE_LAW_VECTOR_INDEX_NAME, question, top_k,
        return_fields={
            "case_name": "case_name", "full_citation": "full_citation", "stance": "stance",
            "document_id": "document_id", "sourceMdPath": "sourceMdPath",
        },
    )
