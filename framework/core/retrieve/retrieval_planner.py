"""
retrieval_planner.py - 2026-09-10: NEW. The planning step of the agentic
retrieval loop (see retrieval_orchestrator.py). Replaces the FIXED tool
order GraphRAGRetriever.retrieve() always follows today (citation-shortcut,
then NL2Cypher, then vector search, in that order, every single call) with a
per-iteration CHOICE: one LLM call decides which ONE tool to try next and
what sub-query/entity filter to give it, informed by what earlier iterations
in THIS question's retrieval loop already attempted and found insufficient.

This module does not itself execute any retrieval - it only decides what to
try next. retrieval_orchestrator.py calls plan_retrieval(), then dispatches
to the chosen tool via the existing GraphRAGRetriever methods
(run_graph_retrieval / vector_search_chunks / retrieve_by_citation), which
are entirely unchanged by this addition.

Only used under retrieval_mode="agentic" - "hybrid"/"vector_only"/
"semantic_graph" keep calling GraphRAGRetriever.retrieve() directly, exactly
as before, so they remain an unchanged ablation baseline to compare against.
"""
from __future__ import annotations

import json
import logging

from framework.core.infra import llm_client

logger = logging.getLogger("retrieval_planner")

VALID_TOOLS = ("CYPHER", "VECTOR", "CITATION")


def _format_prior_attempts(prior_attempts: list[dict] | None) -> str:
    """Renders prior iterations' (tool, query, verdict) into the prompt's
    PRIOR ATTEMPTS block. Returns "(none - this is the first attempt)" when
    empty so the template always has readable text to show, rather than an
    empty string that reads like a formatting bug."""
    if not prior_attempts:
        return "(none - this is the first attempt)"
    lines = []
    for i, attempt in enumerate(prior_attempts, start=1):
        lines.append(
            f"{i}. tool={attempt.get('tool')!r}, "
            f"query={attempt.get('query_text') or attempt.get('citation')!r}, "
            f"result={'found evidence' if attempt.get('found_evidence') else 'found nothing useful'}"
        )
    return "\n".join(lines)


def _parse_plan(raw: str) -> dict:
    """Parses the model's JSON response defensively - strips markdown code
    fences the same way _generate_structured_query_code() in app.py already
    does for the structured-data-query prompt, since models asked for "JSON
    only" still sometimes wrap it in fences anyway."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(f"[retrieval_planner] Model did not return valid JSON, defaulting to CYPHER "
                        f"with the raw question as query_text. Raw output: {text!r}")
        return {"tool": "CYPHER", "query_text": "", "citation": "", "reasoning": "fallback: unparseable plan"}

    tool = str(parsed.get("tool", "")).strip().upper()
    if tool not in VALID_TOOLS:
        logger.warning(f"[retrieval_planner] Model chose unrecognized tool {tool!r}, defaulting to CYPHER.")
        tool = "CYPHER"

    return {
        "tool": tool,
        "query_text": parsed.get("query_text", "") or "",
        "citation": parsed.get("citation", "") or "",
        "reasoning": parsed.get("reasoning", "") or "",
    }


def plan_retrieval(question: str, schema: dict, prior_attempts: list[dict] | None = None,
                    trace=None) -> dict:
    """Returns a plan dict: {"tool": "CYPHER"|"VECTOR"|"CITATION",
    "query_text": str, "citation": str, "reasoning": str}. `schema` is the
    same {"node_labels": [...], "relationship_types": [...]} shape
    graphrag.get_schema_summary() already returns and nl_to_cypher() already
    consumes - reused here, not re-derived. `prior_attempts` is a list of
    dicts, one per earlier iteration this question's retrieval loop has
    already run (empty/None on the first call) - see
    retrieval_orchestrator.py for how that list is built up across
    iterations."""
    from domain_packs.tax.prompts.tax_prompts import PROMPT_PACK  # local import: avoids a hard,
    # domain-specific dependency at module load time - mirrors how graphrag.py itself takes its
    # prompt_pack as a constructor argument rather than importing tax_prompts directly.

    prompt = PROMPT_PACK.get_prompt("retrieval_plan", version="v1")
    system = prompt.template.format(
        labels=", ".join(schema.get("node_labels", [])),
        rel_types=", ".join(schema.get("relationship_types", [])),
        prior_attempts=_format_prior_attempts(prior_attempts),
        question=question,
    )
    # 2026-10-14: was max_tokens=300 - the plan JSON carries a free-text
    # `reasoning` field plus a query_text, and a cut-off JSON object fails
    # json.loads() and silently degrades to CYPHER with an EMPTY query_text
    # (see _parse_plan()), i.e. NL2Cypher on "". Same headroom reasoning as
    # graphrag.NL2CYPHER_MAX_TOKENS.
    raw = llm_client.complete(system=system, user_content=question, max_tokens=600,
                               trace=trace, name="retrieval_plan")
    plan = _parse_plan(raw)
    logger.info(f"[retrieval_planner] plan={plan}")
    return plan
