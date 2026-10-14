"""
opinion_agent.py - 2026-09-11: NEW. The "Opinion" stage of retrieval_mode=
"structured_research" (see hierarchical_orchestrator.py's module docstring),
covering the customer-specified sub-categories: KPMG Opinion - Latest, KPMG
Opinion - Before Change in Law.

The ontology already declares a `FirmOpinion` node label ("A formal KPMG tax
opinion - facts through recommendation, in one record" - see
domain_packs/tax/ontology/schema.json) with SUPERSEDES/SUPERSEDED_BY
relationships to itself and to DelegatedLegislation - exactly the "latest vs
superseded-by-a-law-change-or-newer-opinion" distinction this stage needs.

DESIGN NOTE: latest-vs-superseded classification uses a HAND-WRITTEN Cypher
query (see _fetch_supersession_status() below), not NL2Cypher's free-form
generation - the shape of "does this node have an outgoing SUPERSEDED_BY
edge" is fixed and known regardless of what topic is being searched, so a
deterministic OPTIONAL MATCH is both more reliable and independently
testable than parsing whatever row shape NL2Cypher happens to generate.
Finding CANDIDATE opinions relevant to the tax position, on the other hand,
genuinely needs topical/semantic matching - done via the same vector-search-
then-filter-by-label pattern case_law_agent.find_candidate_case_law() already
uses, just filtered to "FirmOpinion" instead of "CaseLaw".

As of 2026-09-11, no FirmOpinion documents have actually been ingested
(confirmed - see this project's own ingestion-status discussion), so this
agent will currently return "no opinions found" honestly, and will start
returning real results automatically the moment an ingestion pipeline
populates FirmOpinion nodes - no code change needed here when that happens.
"""
from __future__ import annotations

import logging

from framework.core.infra import neo4j_client
from framework.core.retrieve.graphrag import GraphRAGRetriever

logger = logging.getLogger("opinion_agent")

CANDIDATE_TOP_K = 10


def _find_candidate_opinions(retriever: GraphRAGRetriever, tax_position: str) -> list[dict]:
    """Same vector-search-then-filter-by-label pattern as
    case_law_agent.find_candidate_case_law(), filtered to "FirmOpinion"
    instead of "CaseLaw" - see that function's docstring for why vector
    search (not scoped to any one node type) needs this filter."""
    hits = retriever.vector_search_chunks(tax_position, top_k=CANDIDATE_TOP_K)
    chunk_ids = [h["chunk_id"] for h in hits]
    entities = retriever.get_entities_for_chunks(chunk_ids)
    opinion_nodes = [n for n in entities if n.get("label") == "FirmOpinion"]
    seen = set()
    deduped = []
    for n in opinion_nodes:
        if n["db_id"] not in seen:
            deduped.append(n)
            seen.add(n["db_id"])
    return deduped


def _fetch_supersession_status(opinion_ids: list[str]) -> dict[str, dict]:
    """Hand-written, deterministic query (see this module's docstring for
    why NOT NL2Cypher-generated): for each given FirmOpinion node id, finds
    whatever it's SUPERSEDED_BY (another FirmOpinion, a DelegatedLegislation
    node representing a law change, or nothing at all). Returns
    {opinion_id: {"is_superseded": bool, "superseded_by_id": str|None,
    "superseded_by_label": str|None}}."""
    if not opinion_ids:
        return {}
    with neo4j_client.get_session() as session:
        rows = list(session.run(
            """
            MATCH (op:FirmOpinion) WHERE op.id IN $opinion_ids
            OPTIONAL MATCH (op)-[:SUPERSEDED_BY]->(newer)
            RETURN op.id AS opinion_id, newer.id AS superseded_by_id,
                   labels(newer) AS superseded_by_labels
            """,
            opinion_ids=opinion_ids,
        ))
    status = {}
    for r in rows:
        superseded_by_id = r["superseded_by_id"]
        labels = r["superseded_by_labels"] or []
        status[r["opinion_id"]] = {
            "is_superseded": superseded_by_id is not None,
            "superseded_by_id": superseded_by_id,
            "superseded_by_label": labels[0] if labels else None,
        }
    return status


def retrieve_opinions(retriever: GraphRAGRetriever, fact_sheet: dict, trace=None) -> dict:
    """Returns {"latest": [...], "superseded": [...], "candidates_found": int}.
    Each list entry is {"citation"/"id", "superseded_by"} - `superseded_by`
    is None for entries in "latest". `candidates_found` == 0 is the honest,
    expected result until FirmOpinion documents are actually ingested - see
    this module's docstring."""
    tax_position = fact_sheet.get("tax_position", "")
    candidates = _find_candidate_opinions(retriever, tax_position)

    if not candidates:
        logger.info("[opinion_agent] No candidate FirmOpinion nodes found (not yet ingested, or none "
                     "relevant to this tax position).")
        return {"latest": [], "superseded": [], "candidates_found": 0}

    opinion_ids = [n["id"] for n in candidates]
    status_by_id = _fetch_supersession_status(opinion_ids)

    latest, superseded = [], []
    for n in candidates:
        props = n.get("properties", {})
        status = status_by_id.get(n["id"], {"is_superseded": False, "superseded_by_id": None,
                                              "superseded_by_label": None})
        entry = {
            "id": n["id"],
            "citation": props.get("full_citation", props.get("id", "(unknown)")),
            "superseded_by": status["superseded_by_id"],
            "superseded_by_type": status["superseded_by_label"],
        }
        if status["is_superseded"]:
            superseded.append(entry)
        else:
            latest.append(entry)

    logger.info(f"[opinion_agent] {len(candidates)} candidate(s): "
                f"{len(latest)} latest, {len(superseded)} superseded.")
    return {"latest": latest, "superseded": superseded, "candidates_found": len(candidates)}
