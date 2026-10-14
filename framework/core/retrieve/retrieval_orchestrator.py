"""
retrieval_orchestrator.py - 2026-09-10: NEW. The agentic retrieval loop,
selectable via retrieval_mode="agentic" (see app.py's _run_unstructured_agent()).

Today, GraphRAGRetriever.retrieve() runs one FIXED pass every single call:
citation-shortcut, then one NL2Cypher shot, then one vector search, merged
deterministically - never re-planned within retrieval itself. The only
existing retry mechanism (app.py's quality-checked-ask loop,
answer_evaluation.py's evaluate_answer()) only scores a FINISHED answer and,
on a low score, reruns that entire fixed pass again with the SAME retrieval
calls - a bad NL2Cypher query survives unchanged across every retry, because
nothing about retrieval itself ever adapts (see nl2cypher v11's fix notes
for a concrete case: an entity-blind query kept returning nothing useful,
and no amount of re-scoring the resulting answer would have fixed that).

This module closes that gap with a bounded loop, run BEFORE any answer is
generated:
    plan (retrieval_planner) -> call chosen tool -> accumulate evidence
        -> sufficiency check (evidence_sufficiency) -> SUFFICIENT: stop
                                                     -> INSUFFICIENT: re-plan

GraphRAGRetriever's own tool methods (run_graph_retrieval, nl_to_cypher,
vector_search_chunks, retrieve_by_citation) are called here EXACTLY as they
already exist - nothing about GraphRAGRetriever itself changes. This module
only changes which of those methods gets called, in what order, and with
what sub-query, per iteration.

retrieval_mode="hybrid"/"vector_only"/"semantic_graph" are completely
unaffected - they keep calling GraphRAGRetriever.retrieve() directly, so
they remain a stable, unchanged baseline to compare "agentic" against.
"""
from __future__ import annotations

import logging
import time

from framework.core.retrieve import evidence_sufficiency, retrieval_planner
from framework.core.retrieve.graphrag import GraphRAGRetriever, get_schema_summary

logger = logging.getLogger("retrieval_orchestrator")

# 2026-09-10: NEW - bounds how many plan/retrieve/check passes one question
# can trigger. Mirrors STRUCTURED_AGENT_MAX_ATTEMPTS in app.py - a bounded
# retry, not an open-ended loop, so a question the planner genuinely cannot
# find good evidence for (rather than one it's one reformulation away from
# solving) still terminates and falls through to generation with whatever
# evidence exists, rather than burning further LLM calls indefinitely.
MAX_RETRIEVAL_ITERATIONS = 3


def run_tool(retriever: GraphRAGRetriever, plan: dict, trace=None) -> dict:
    """Dispatches ONE tool call (CYPHER/VECTOR/CITATION - same shape
    retrieval_planner.plan_retrieval() returns) and returns its result
    reshaped into the common {"plain_rows": [...], "chunks": [...],
    "cypher": str|None} evidence shape, so callers don't need tool-specific
    branching beyond this one function. Each branch calls the corresponding
    GraphRAGRetriever method completely unchanged.

    2026-09-11: PROMOTED from a private _run_tool() to a public function -
    law_agent.py (retrieval_mode="structured_research") needs the identical
    CYPHER/VECTOR dispatch logic for its own multiple targeted sub-queries
    (statute, treaty, amendment chain), and duplicating this dispatch there
    would risk the two drifting out of sync. run_agentic_retrieval() below
    is the only other caller and is unaffected - same function, new name,
    now importable."""
    tool = plan["tool"]

    if tool == "CYPHER":
        query_text = plan.get("query_text") or ""
        graph_result = retriever.run_graph_retrieval(query_text, trace=trace)
        matched_nodes = list(graph_result["nodes"].values())
        chunk_ids = retriever.get_chunk_ids_via_graph_traversal([n["db_id"] for n in matched_nodes])
        if not chunk_ids:
            chunk_ids = retriever.extract_source_chunk_ids_legacy(matched_nodes)
        chunks = [{"chunk_id": c["chunk_id"], "text": c["text"], "source": "graph"}
                  for c in retriever.get_chunks_by_ids(chunk_ids)]
        return {"plain_rows": graph_result["plain_rows"], "chunks": chunks, "cypher": graph_result["cypher"]}

    if tool == "VECTOR":
        query_text = plan.get("query_text") or ""
        hits = retriever.vector_search_chunks(query_text, top_k=10)
        chunks = [{"chunk_id": c["chunk_id"], "text": c["text"], "source": "vector",
                   "similarity": c.get("similarity")} for c in hits]
        return {"plain_rows": [], "chunks": chunks, "cypher": None}

    if tool == "CITATION":
        citation = plan.get("citation") or ""
        # 2026-10-14: every Act's match for the citation (the 1961 and 2025
        # Acts share section numbers for unrelated provisions) plus the
        # CORRESPONDS_TO counterpart in the other Act - same helpers
        # GraphRAGRetriever.retrieve()'s citation path uses; was
        # retrieve_by_citation(), which returned one arbitrary Act's chunk.
        hits = retriever.retrieve_citation_matches(citation) if citation else []
        chunks = [{"chunk_id": h["chunk_id"], "text": h["text"], "source": "citation"} for h in hits]
        correspondence = {"rows": [], "chunks": []}
        if hits:
            try:
                correspondence = retriever.get_act_correspondence_context(
                    [h["entity_id"] for h in hits if h.get("entity_id")])
            except Exception as e:
                logger.warning(f"[retrieval_orchestrator] act-correspondence lookup failed: {e}")
        chunks += [c for c in correspondence["chunks"] if c["chunk_id"] not in {x["chunk_id"] for x in chunks}]
        return {"plain_rows": correspondence["rows"], "chunks": chunks, "cypher": None}

    # Should be unreachable - retrieval_planner._parse_plan() already
    # defaults any unrecognized tool value to "CYPHER" before this function
    # ever sees it. Kept as a defensive fallback, not a silent no-op, so a
    # future new tool value that's added to the planner but not here yet
    # fails loudly instead of returning empty evidence with no explanation.
    raise ValueError(f"[retrieval_orchestrator] Unhandled tool {tool!r} from planner output")


def _merge_evidence(pool: dict, new_evidence: dict) -> dict:
    """Accumulates new_evidence into the running pool, deduping chunks by
    chunk_id (the same identity key GraphRAGRetriever.retrieve() itself uses
    for its own seen_ids dedup - see that method's citation_candidate/
    expansion_candidates handling) so a later VECTOR pass re-finding a chunk
    a CYPHER pass already pulled in doesn't double-count it. Graph rows are
    not deduped (they aren't independently keyed the way chunks are), but in
    practice at most one CYPHER call happens per iteration, so duplicate
    rows can only occur if the SAME iteration's Cypher query itself returns
    a duplicate row - the database's own responsibility, not this
    function's."""
    seen_ids = {c["chunk_id"] for c in pool["chunks"]}
    for c in new_evidence.get("chunks", []):
        if c["chunk_id"] not in seen_ids:
            pool["chunks"].append(c)
            seen_ids.add(c["chunk_id"])
    pool["plain_rows"].extend(new_evidence.get("plain_rows", []))
    if new_evidence.get("cypher"):
        pool["cypher_queries"].append(new_evidence["cypher"])
    return pool


def run_agentic_retrieval(retriever: GraphRAGRetriever, question: str, trace=None) -> dict:
    """Runs the bounded plan/retrieve/check loop for one question. Returns:
        {
          "plain_rows": [...],       # accumulated graph rows, across every CYPHER call this ran
          "chunks": [...],           # accumulated + deduped chunks, across every tool call this ran
          "cypher_queries": [...],   # every distinct Cypher string actually executed, in order
          "iterations": [...],       # one record per pass: {tool, query, found_evidence, verdict, missing}
          "final_verdict": "SUFFICIENT" | "INSUFFICIENT",
        }
    `iterations` is intentionally shaped close to app.py's existing
    agent_trace["agents_invoked"] records (see _run_unstructured_agent() /
    _run_structured_agent()) so the evidence panel can render this loop's
    steps the same way it already renders the file-attachment agents'
    steps, without a new UI concept."""
    schema = get_schema_summary()
    pool = {"plain_rows": [], "chunks": [], "cypher_queries": []}
    iterations = []
    prior_attempts_for_planner = []
    verdict = "INSUFFICIENT"

    for i in range(1, MAX_RETRIEVAL_ITERATIONS + 1):
        started = time.monotonic()
        plan = retrieval_planner.plan_retrieval(
            question, schema, prior_attempts=prior_attempts_for_planner, trace=trace)

        new_evidence = run_tool(retriever, plan, trace=trace)
        found_evidence = bool(new_evidence.get("plain_rows") or new_evidence.get("chunks"))
        pool = _merge_evidence(pool, new_evidence)

        check = evidence_sufficiency.check_sufficiency(question, pool, trace=trace)
        verdict = check["verdict"]

        iterations.append({
            "iteration": i,
            "tool": plan["tool"],
            "query": plan.get("query_text") or plan.get("citation") or "",
            "reasoning": plan.get("reasoning", ""),
            "found_evidence": found_evidence,
            "verdict": verdict,
            "missing": check.get("missing", ""),
            "elapsed_seconds": round(time.monotonic() - started, 2),
        })
        logger.info(f"[retrieval_orchestrator] iteration {i}/{MAX_RETRIEVAL_ITERATIONS}: "
                    f"tool={plan['tool']} found_evidence={found_evidence} verdict={verdict}")

        prior_attempts_for_planner.append({
            "tool": plan["tool"], "query_text": plan.get("query_text"),
            "citation": plan.get("citation"), "found_evidence": found_evidence,
        })

        if verdict == "SUFFICIENT":
            break
    else:
        logger.warning(f"[retrieval_orchestrator] Reached MAX_RETRIEVAL_ITERATIONS "
                        f"({MAX_RETRIEVAL_ITERATIONS}) without a SUFFICIENT verdict - "
                        f"proceeding to generation with whatever evidence was found, same "
                        f"'degrade, don't block the answer' reasoning as this codebase's other "
                        f"best-effort retrieval steps (e.g. graphrag.py's graph-expansion step).")

    pool["iterations"] = iterations
    pool["final_verdict"] = verdict
    return pool
