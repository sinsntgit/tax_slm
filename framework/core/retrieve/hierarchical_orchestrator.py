"""
hierarchical_orchestrator.py - 2026-09-11: NEW. Powers retrieval_mode=
"structured_research" - a fixed, customer-specified research hierarchy,
distinct from retrieval_orchestrator.py's agentic loop (which lets a
planner freely choose tools and stop as soon as evidence looks sufficient).
This pipeline always runs every stage and always reports what each one
found - or explicitly did not find - so "How this answer was built" can
show the complete methodology, not just whichever path happened to work.

PIPELINE SHAPE (see each stage's own module for why):
    Fact Sheet (sequential, everyone depends on its output)
        -> Law (sequential - its amendment-chain finding feeds Tribunal)
            -> Case Law    \\
            -> Tribunal     |  parallel (ThreadPoolExecutor, same pattern
            -> Opinion     /   app.py's _run_fast_quality_checked_ask()
                               already uses for its own dual evaluators)
    -> Synthesis (sequential, consumes all of the above)

Tribunal Case Law is NOT fully parallel with Law - see
tribunal_case_law_agent.py's own module docstring for why its
PRE_CHANGE_IN_LAW bucketing needs Law's own amendment-chain finding as
input, and why that dependency was judged worth breaking full 4-way
parallelism for (accuracy over latency, an explicit decision - see this
project's own discussion of the tradeoff).

EVIDENCE TRACE: every stage's result is preserved in the returned
`stage_trace` list, one entry per stage, each carrying enough detail for
the evidence panel to show exactly what was searched at every level of the
hierarchy and explicitly say so wherever nothing was found - see
_build_stage_trace() below and app.py's retrieval_mode=="structured_research"
branch for how this becomes response["agent_trace"].
"""
from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor

from framework.core.retrieve import case_law_agent, fact_sheet_agent, law_agent, opinion_agent, \
    tribunal_case_law_agent
from framework.core.retrieve.graphrag import GraphRAGRetriever

logger = logging.getLogger("hierarchical_orchestrator")


def _safe_call(label: str, fn, *args, **kwargs):
    """Runs one stage, catching any exception so one agent's failure
    degrades that stage to an explicit error result rather than crashing
    the whole structured_research pipeline - same "a bug in one step
    shouldn't take down an answer the other steps could still support"
    reasoning as graphrag.py's own best-effort steps (semantic_entry_point_
    search, graph-expansion). Returns (result, error) - error is None on
    success."""
    try:
        return fn(*args, **kwargs), None
    except Exception as e:
        logger.exception(f"[hierarchical_orchestrator] stage {label!r} failed")
        return None, str(e)


def _build_stage_trace(label: str, started_at: float, found_summary: str, detail: dict,
                        error: str | None) -> dict:
    """One entry in the returned stage_trace list - the shape
    app.py/index.html render as one step in the evidence panel. `detail` is
    stage-specific (e.g. law_agent's sub_queries list, case_law_agent's
    favourable/unfavourable counts) - kept as its own key so the evidence
    panel can render stage-appropriate detail without every stage needing
    the same fields."""
    return {
        "stage": label,
        "found_summary": found_summary if error is None else f"Failed: {error}",
        "detail": detail if error is None else {},
        "error": error,
        "elapsed_seconds": round(time.monotonic() - started_at, 2),
    }


def run_hierarchical_research(retriever: GraphRAGRetriever, question: str,
                               attached_file_context: dict | None = None, trace=None) -> dict:
    """Runs the full Fact Sheet -> Law -> {Case Law, Tribunal, Opinion} ->
    (synthesis inputs assembled) pipeline. Returns {"fact_sheet": {...},
    "law": {...}, "case_law": {...}, "tribunal": {...}, "opinion": {...},
    "stage_trace": [...]}. Does NOT run the final synthesis LLM call itself -
    that is deliberately left to the caller (app.py's retrieval_mode==
    "structured_research" branch), which has direct access to the
    hierarchical_synthesis prompt and the original question text, matching
    how every other agent in this codebase keeps generation calls with the
    caller rather than duplicating that responsibility into an orchestrator
    module."""
    stage_trace = []

    # --- Stage 1: Fact Sheet (sequential) -----------------------------------
    started = time.monotonic()
    fact_sheet, error = _safe_call("Fact Sheet", fact_sheet_agent.build_fact_sheet,
                                    question, attached_file_context, trace=trace)
    if error:
        fact_sheet = {"facts": "(fact-sheet stage failed)", "tax_position": question}
    stage_trace.append(_build_stage_trace(
        "Fact Sheet", started,
        f"Tax position framed: {fact_sheet['tax_position']!r}" if not error else "",
        {"facts": fact_sheet["facts"], "tax_position": fact_sheet["tax_position"]}, error))

    # --- Stage 2: Law (sequential - Tribunal needs its change-in-law date) --
    # 2026-09-11: question= now threaded through (see law_agent.py/
    # case_law_agent.py's docstrings for the bugfix this is part of) - every
    # stage below gets the user's ORIGINAL wording alongside the fact
    # sheet's paraphrase, not the paraphrase alone.
    started = time.monotonic()
    law_result, error = _safe_call("Law", law_agent.retrieve_law, retriever, fact_sheet,
                                    question=question, trace=trace)
    if error:
        law_result = {"plain_rows": [], "chunks": [], "cypher_queries": [], "sub_queries": []}
    found_labels = [sq["label"] for sq in law_result.get("sub_queries", []) if sq["found_evidence"]]
    not_found_labels = [sq["label"] for sq in law_result.get("sub_queries", []) if not sq["found_evidence"]]
    law_summary = (
        f"Found: {', '.join(found_labels) if found_labels else 'none'}. "
        f"Not found: {', '.join(not_found_labels) if not_found_labels else 'none'}."
    )
    stage_trace.append(_build_stage_trace("Law", started, law_summary,
                                           {"sub_queries": law_result.get("sub_queries", [])}, error))

    # --- Stage 3: Case Law, Tribunal Case Law, Opinion (parallel) -----------
    def _run_case_law():
        return case_law_agent.retrieve_case_law(retriever, fact_sheet, question=question, trace=trace)

    def _run_tribunal():
        return tribunal_case_law_agent.retrieve_tribunal_case_law(retriever, fact_sheet, law_result,
                                                                    question=question, trace=trace)

    def _run_opinion():
        return opinion_agent.retrieve_opinions(retriever, fact_sheet, trace=trace)

    with ThreadPoolExecutor(max_workers=3) as pool:
        case_law_started = time.monotonic()
        tribunal_started = time.monotonic()
        opinion_started = time.monotonic()
        case_law_future = pool.submit(_run_case_law)
        tribunal_future = pool.submit(_run_tribunal)
        opinion_future = pool.submit(_run_opinion)

        case_law_result, case_law_error = None, None
        try:
            case_law_result = case_law_future.result()
        except Exception as e:
            logger.exception("[hierarchical_orchestrator] stage 'Case Law' failed")
            case_law_error = str(e)

        tribunal_result, tribunal_error = None, None
        try:
            tribunal_result = tribunal_future.result()
        except Exception as e:
            logger.exception("[hierarchical_orchestrator] stage 'Tribunal Case Law' failed")
            tribunal_error = str(e)

        opinion_result, opinion_error = None, None
        try:
            opinion_result = opinion_future.result()
        except Exception as e:
            logger.exception("[hierarchical_orchestrator] stage 'Opinion' failed")
            opinion_error = str(e)

    if case_law_error:
        case_law_result = {"favourable": [], "unfavourable": [], "unclassified": [], "candidates_found": 0}
    case_law_summary = (
        f"{case_law_result['candidates_found']} candidate(s) found - "
        f"{len(case_law_result['favourable'])} favourable, {len(case_law_result['unfavourable'])} unfavourable."
        if case_law_result["candidates_found"] else "No candidate case law found."
    )
    stage_trace.append(_build_stage_trace("Case Law", case_law_started, case_law_summary,
                                           case_law_result, case_law_error))

    if tribunal_error:
        tribunal_result = {"hc": [], "sc": [], "jurisdictional": [], "pre_change_in_law": [],
                            "change_in_law_date": "", "candidates_found": 0}
    tribunal_summary = (
        f"{tribunal_result['candidates_found']} candidate(s) - "
        f"HC={len(tribunal_result['hc'])}, SC={len(tribunal_result['sc'])}, "
        f"jurisdictional={len(tribunal_result['jurisdictional'])}, "
        f"pre-change-in-law={len(tribunal_result['pre_change_in_law'])}"
        + (f" (change-in-law date: {tribunal_result['change_in_law_date']})"
           if tribunal_result["change_in_law_date"] else " (change-in-law date not established).")
        if tribunal_result["candidates_found"] else "No candidate case law found."
    )
    stage_trace.append(_build_stage_trace("Tribunal Case Law", tribunal_started, tribunal_summary,
                                           tribunal_result, tribunal_error))

    if opinion_error:
        opinion_result = {"latest": [], "superseded": [], "candidates_found": 0}
    opinion_summary = (
        f"{opinion_result['candidates_found']} candidate(s) - "
        f"{len(opinion_result['latest'])} latest, {len(opinion_result['superseded'])} superseded."
        if opinion_result["candidates_found"]
        else "No KPMG opinions found (not yet ingested into the graph, or none relevant to this matter)."
    )
    stage_trace.append(_build_stage_trace("Opinion", opinion_started, opinion_summary,
                                           opinion_result, opinion_error))

    return {
        "fact_sheet": fact_sheet,
        "law": law_result,
        "case_law": case_law_result,
        "tribunal": tribunal_result,
        "opinion": opinion_result,
        "stage_trace": stage_trace,
    }
