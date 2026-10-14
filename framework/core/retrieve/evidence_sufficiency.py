"""
evidence_sufficiency.py - 2026-09-10: NEW. The loop-control step of the
agentic retrieval loop (see retrieval_orchestrator.py). Runs AFTER each
retrieval iteration and BEFORE any answer is generated.

This is the key structural difference from the existing quality-checked-ask
loop (framework/core/retrieve/answer_evaluation.py's evaluate_answer(),
driven from app.py's _run_quality_checked_ask_job()): that evaluator only
ever scores a FINISHED answer, and a low score reruns the ENTIRE fixed
pipeline from scratch with the exact same retrieval calls - retrieval itself
never re-plans. This module instead judges the EVIDENCE POOL directly, prior
to generation, so a bad or empty retrieval result gets caught and re-planned
(see retrieval_planner.py) before a low-quality answer is ever written from
it, rather than after. The two checks are complementary, not redundant: this
one catches bad retrieval, the existing one still catches wording/tone/
hedging problems in an otherwise well-grounded answer.
"""
from __future__ import annotations

import json
import logging

from framework.core.infra import llm_client

logger = logging.getLogger("evidence_sufficiency")

VALID_VERDICTS = ("SUFFICIENT", "INSUFFICIENT")

# Caps how much evidence text goes into the sufficiency-check prompt itself -
# this call only needs enough of each row/chunk to judge relevance, not the
# full text that answer generation will eventually use. Keeps this quick
# judgment call cheap regardless of how large the accumulated evidence pool
# has grown across iterations.
_MAX_CHARS_PER_ITEM = 400
_MAX_ITEMS = 20


def _summarize_evidence(evidence_pool: dict) -> str:
    """Renders the accumulated evidence pool - {"plain_rows": [...],
    "chunks": [...]} - into the prompt's EVIDENCE RETRIEVED SO FAR block.
    Truncates per-item and caps item count (see module constants above);
    this is a relevance judgment, not the material answer generation will
    actually cite. Returns "(no evidence retrieved yet)" when both are empty
    so an all-tools-came-back-empty first pass reads as an obvious
    INSUFFICIENT case rather than an ambiguous blank block."""
    rows = evidence_pool.get("plain_rows") or []
    chunks = evidence_pool.get("chunks") or []
    if not rows and not chunks:
        return "(no evidence retrieved yet)"

    lines = []
    for i, row in enumerate(rows[:_MAX_ITEMS], start=1):
        text = json.dumps(row, default=str)[:_MAX_CHARS_PER_ITEM]
        lines.append(f"[graph row {i}] {text}")
    for i, chunk in enumerate(chunks[:_MAX_ITEMS], start=1):
        text = (chunk.get("text") or "")[:_MAX_CHARS_PER_ITEM]
        lines.append(f"[chunk {i}, source={chunk.get('source', '?')}] {text}")
    return "\n".join(lines)


def _parse_verdict(raw: str) -> dict:
    """Same defensive markdown-fence stripping as retrieval_planner._parse_plan()
    - models asked for "JSON only" still sometimes wrap it in fences."""
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
        logger.warning(f"[evidence_sufficiency] Model did not return valid JSON, defaulting to "
                        f"INSUFFICIENT (safer than assuming good evidence on a parse failure). "
                        f"Raw output: {text!r}")
        return {"verdict": "INSUFFICIENT", "missing": "sufficiency check itself failed to parse"}

    verdict = str(parsed.get("verdict", "")).strip().upper()
    if verdict not in VALID_VERDICTS:
        logger.warning(f"[evidence_sufficiency] Model returned unrecognized verdict {verdict!r}, "
                        f"defaulting to INSUFFICIENT.")
        verdict = "INSUFFICIENT"

    return {"verdict": verdict, "missing": parsed.get("missing", "") or ""}


def check_sufficiency(question: str, evidence_pool: dict, trace=None) -> dict:
    """Returns {"verdict": "SUFFICIENT"|"INSUFFICIENT", "missing": str}.
    `evidence_pool` is the orchestrator's running accumulation across every
    iteration so far - {"plain_rows": [...], "chunks": [...]} - not just the
    most recent tool call's output, so a verdict can recognize evidence
    assembled across multiple different tool calls as sufficient together
    even if no single call alone would be."""
    from domain_packs.tax.prompts.tax_prompts import PROMPT_PACK  # local import - see
    # retrieval_planner.py's plan_retrieval() for why (mirrors graphrag.py's own pattern of
    # taking a prompt_pack rather than importing the domain pack at module load time).

    prompt = PROMPT_PACK.get_prompt("evidence_sufficiency", version="v1")
    system = prompt.template.format(
        question=question,
        evidence_summary=_summarize_evidence(evidence_pool),
    )
    raw = llm_client.complete(system=system, user_content=question, max_tokens=150,
                               trace=trace, name="evidence_sufficiency")
    result = _parse_verdict(raw)
    logger.info(f"[evidence_sufficiency] verdict={result['verdict']!r} missing={result['missing']!r}")
    return result
