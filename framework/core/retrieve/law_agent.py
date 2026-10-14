"""
law_agent.py - 2026-09-11: NEW. The "Law" stage of retrieval_mode=
"structured_research" (see hierarchical_orchestrator.py's module docstring),
covering the customer-specified sub-categories: Income Tax Act, Notifications,
Circulars, Rules, amended versions of all of those, and Tax Treaty.

Unlike retrieval_orchestrator.py's agentic loop (which lets a planner freely
choose ONE tool per iteration and stop as soon as evidence looks sufficient),
this agent ALWAYS runs a fixed set of targeted sub-queries every time - the
customer's hierarchy names six specific things to check, so leaving any of
them to a planner's discretion risks silently skipping one just because an
earlier sub-query already looked "sufficient" in isolation. Each sub-query's
outcome (found evidence, or explicitly found nothing) is preserved separately
in the returned trace, not merged away, so "How this answer was built" can
show exactly which of the six were searched and which came back empty - see
this function's return shape below and hierarchical_orchestrator.py's
evidence-panel trace assembly.

Reuses retrieval_orchestrator.run_tool() for the actual CYPHER/VECTOR
dispatch (GraphRAGRetriever's own methods, unchanged) rather than
duplicating that logic - see that function's own docstring for why it was
promoted from private to public for this purpose.
"""
from __future__ import annotations

import logging

from framework.core.retrieve.graphrag import GraphRAGRetriever
from framework.core.retrieve.retrieval_orchestrator import run_tool

logger = logging.getLogger("law_agent")

# 2026-09-11: NEW - the six sub-categories the customer's hierarchy names
# under "Law". Each is a (label, tool, query_builder) tuple - query_builder
# takes the fact sheet's tax_position and produces the sub-query text to
# hand to the chosen tool. Kept as an explicit, named list (not a loop over
# some generic "try everything" logic) so a reader can see exactly which six
# things this agent checks, matching the hierarchy's own wording. The
# amendment-chain sub-query is ALWAYS attempted here too (not left to a
# planner's discretion the way agentic mode's planner might skip it).
# 2026-10-13: NEW - label of the amendment-chain sub-query below, whose graph
# rows are kept separately in pool["amendment_rows"] (see retrieve_law()) so
# tribunal_case_law_agent._extract_change_in_law_date() can look for the
# change-in-law date in the rows that are actually ABOUT amendments first,
# before falling back to every Law row (any provision/circular's own
# effective_from is a much weaker proxy for "the relevant change in law").
AMENDMENT_SUBQUERY_LABEL = "Amended IT Act, Notifications, Circulars & Rules"

_LAW_SUBQUERIES = [
    ("Income Tax Act", "CYPHER", lambda pos, q: f"Income Tax Act provisions relevant to: {pos}\n\n"
     f"(Original question, for matching any specifically named section/case/authority: {q})"),
    ("Notifications", "CYPHER", lambda pos, q: f"Notifications relevant to: {pos}\n\n"
     f"(Original question, for matching any specifically named section/case/authority: {q})"),
    ("Circulars", "CYPHER", lambda pos, q: f"Circulars relevant to: {pos}\n\n"
     f"(Original question, for matching any specifically named section/case/authority: {q})"),
    ("Rules", "CYPHER", lambda pos, q: f"Rules relevant to: {pos}\n\n"
     f"(Original question, for matching any specifically named section/case/authority: {q})"),
    (AMENDMENT_SUBQUERY_LABEL, "CYPHER",
     lambda pos, q: f"What amendments (AMENDED_BY/SUPERSEDED_BY) apply to provisions relevant to: {pos}\n\n"
     f"(Original question, for matching any specifically named section/case/authority: {q})"),
    ("Tax Treaty", "CYPHER", lambda pos, q: f"Tax Treaty (DTAA) provisions relevant to: {pos}\n\n"
     f"(Original question, for matching any specifically named section/case/authority: {q})"),
]

# 2026-09-11: NEW - vector fallback sub-query, run in addition to the six
# Cypher sub-queries above (not instead of) - catches conceptually relevant
# text that a citation/entity-oriented NL2Cypher query might miss, same
# "vector as a fallback/supplement, not a replacement" reasoning
# GraphRAGRetriever.retrieve()'s own config["vector_mode"] already applies
# to the general hybrid pipeline.
_VECTOR_FALLBACK_LABEL = "Broader conceptual search (vector)"


def retrieve_law(retriever: GraphRAGRetriever, fact_sheet: dict, question: str | None = None,
                  trace=None) -> dict:
    """Returns {"plain_rows": [...], "chunks": [...], "cypher_queries": [...],
    "sub_queries": [...]}. `sub_queries` is a list of {"label", "tool",
    "query", "found_evidence"} - one entry per sub-category ALWAYS attempted
    (see _LAW_SUBQUERIES above), regardless of whether earlier ones found
    anything - this is what lets the evidence panel say "Circulars: not
    found" instead of silently omitting a sub-category that came back empty.

    2026-09-11: `question` param ADDED (optional, defaults to tax_position
    when omitted). Each sub-query now appends the user's ORIGINAL question
    text alongside the category-framed prompt (see _LAW_SUBQUERIES above) -
    previously each sub-query only carried fact_sheet_agent's paraphrased
    tax_position, which can drift enough from the user's actual wording
    (different phrasing, a dropped name) that NL2Cypher's NAMED-CASE/
    NAMED-AMENDING-ACT matching - built and tuned against literal question
    text - failed to fire even when the same underlying question answered
    correctly under retrieval_mode="hybrid". Appending the original question
    gives NL2Cypher the actual name text to match against without dropping
    the category framing each sub-query needs."""
    tax_position = fact_sheet.get("tax_position", "")
    original_question = question if question is not None else tax_position

    pool = {"plain_rows": [], "chunks": [], "cypher_queries": [], "amendment_rows": []}
    sub_queries = []
    seen_chunk_ids = set()

    for label, tool, query_builder in _LAW_SUBQUERIES:
        query_text = query_builder(tax_position, original_question)
        plan = {"tool": tool, "query_text": query_text, "citation": "", "reasoning": label}
        evidence = run_tool(retriever, plan, trace=trace)
        found = bool(evidence.get("plain_rows") or evidence.get("chunks"))

        pool["plain_rows"].extend(evidence.get("plain_rows", []))
        if label == AMENDMENT_SUBQUERY_LABEL:  # 2026-10-13: see AMENDMENT_SUBQUERY_LABEL's comment
            pool["amendment_rows"].extend(evidence.get("plain_rows", []))
        for c in evidence.get("chunks", []):
            if c["chunk_id"] not in seen_chunk_ids:
                pool["chunks"].append(c)
                seen_chunk_ids.add(c["chunk_id"])
        if evidence.get("cypher"):
            pool["cypher_queries"].append(evidence["cypher"])

        sub_queries.append({"label": label, "tool": tool, "query": query_text, "found_evidence": found})
        logger.info(f"[law_agent] sub-query {label!r}: found_evidence={found}")

    # Vector fallback - supplements, does not replace, the Cypher sub-queries above.
    vector_plan = {"tool": "VECTOR", "query_text": tax_position, "citation": "", "reasoning": _VECTOR_FALLBACK_LABEL}
    vector_evidence = run_tool(retriever, vector_plan, trace=trace)
    vector_found = bool(vector_evidence.get("chunks"))
    for c in vector_evidence.get("chunks", []):
        if c["chunk_id"] not in seen_chunk_ids:
            pool["chunks"].append(c)
            seen_chunk_ids.add(c["chunk_id"])
    sub_queries.append({"label": _VECTOR_FALLBACK_LABEL, "tool": "VECTOR", "query": tax_position,
                         "found_evidence": vector_found})

    pool["sub_queries"] = sub_queries
    return pool
