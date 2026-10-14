"""
ANSWER EVALUATION - the on-demand, user-facing "Evaluate this answer" button
on the Ask screen. A SEPARATE concern from answer_quality.py's judge_answer():

  - answer_quality.judge_answer() is a thin, 3-dimension INTERNAL quality
    GATE that runs automatically during /api/ask, used only to decide
    whether app.py should silently retry the generation with a wider chunk
    pool. Its output is never shown to the user directly.
  - evaluate_answer() here is a richer, multi-dimension scorecard the user
    triggers explicitly, styled after the manual review scorecards used
    during this project's development (dimension/score/assessment rows, an
    overall score, a plain-language verdict, explicit strengths/issues) -
    the exact process that originally surfaced real bugs (the graph-context
    truncation, the missing precedents_overruled/OVERRULES gap). This makes
    that same kind of review available on demand instead of only through
    manual external evaluation.

Domain-agnostic by design (same reasoning as grounding.py/answer_quality.py):
this module only ever sees the question, the answer, and the same source
material the answer was generated from - the actual evaluation CRITERIA
(what "legal precision" or "citation quality" means for this domain) live in
the domain pack's own "answer_evaluation" prompt (see tax_prompts.py),
passed in by the caller - not hardcoded here, mirroring how nl2cypher's
prompt lives in the domain pack while graphrag.py's retrieval mechanics stay
domain-agnostic.
"""
import json
import logging
import re

from framework.core.retrieve import grounding

logger = logging.getLogger("answer_evaluation")

_EXPECTED_DIMENSION_NAMES = (
    "Identifies relevant authorities/sources",
    "Groundedness / accuracy",
    "Completeness",
    "Legal precision",
    "Citation quality",
    "Directness",
)


def _extract_json(text: str) -> dict | None:
    """Same defensive cleanup pattern used throughout this project's LLM
    call sites (answer_quality.py, extract_case_law.py's parse_llm_json) -
    strip a markdown fence if the model added one despite being told not
    to, then try a direct parse, then fall back to finding the first
    {...} block if the model wrapped the JSON in any surrounding prose."""
    text = text.strip()
    text = re.sub(r"^```(json)?|```$", "", text, flags=re.MULTILINE).strip()
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except (ValueError, TypeError):
                return None
        return None


def _clamp_score(value) -> float | None:
    try:
        return round(max(0.0, min(10.0, float(value))), 1)
    except (TypeError, ValueError):
        return None


def evaluate_answer(question: str, answer_text: str, top_chunks: list[dict], plain_rows: list[dict],
                     system_prompt: str, *, trace=None, no_context_mode: bool = False) -> dict:
    """Runs the rich, user-facing evaluation. Returns:
        {"available": bool,
         "overall": float | None,
         "dimensions": [{"name": str, "score": float | None, "assessment": str}, ...],
         "verdict": str,
         "strengths": [str, ...],
         "issues": [str, ...]}
    `available: False` (all score fields empty/None) only if the evaluator
    call itself fails or returns unparseable output - callers should show
    this as "evaluation unavailable", not as a low score; a failed LLM call
    is not the same thing as a poor answer.

    `no_context_mode` (2026-09-05: NEW) - set True for the separate
    "general knowledge" evaluator (see general_knowledge_evaluation prompt in
    tax_prompts.py), which is deliberately called with empty top_chunks/
    plain_rows so it can only grade off its own trained tax-law knowledge,
    not retrieved material. Without this flag, an empty-context call would
    fall into the "(no source text or graph rows were retrieved for this
    question)" wording below - written for the OTHER evaluator's "did
    retrieval fail" framing, and misleading here since the missing context is
    intentional, not a retrieval failure. This flag swaps in wording that
    says so plainly, and skips the "RETRIEVED CONTEXT" framing entirely."""
    from framework.core.infra import llm_client

    if no_context_mode:
        user_content = (
            f"QUESTION:\n{question}\n\n"
            f"ANSWER TO EVALUATE:\n{answer_text}\n\n"
            f"(No retrieved context is being provided for this evaluation, by design - grade this "
            f"answer using your own general knowledge of Indian tax law, independent of what any "
            f"particular retrieval system may or may not have had available.)"
        )
    else:
        source_block = ""
        if top_chunks:
            source_block += "\n\n---\n\n".join(
                f"[chunk_id={c['chunk_id']}]\n{c['text'][:1200]}" for c in top_chunks[:15]
            )
        if plain_rows:
            source_block += ("\n\nGraph query results (JSON):\n"
                              + grounding.graph_rows_to_context_json(plain_rows, max_chars=6000))
        if not source_block:
            source_block = "(no source text or graph rows were retrieved for this question)"

        user_content = (
            f"QUESTION:\n{question}\n\n"
            f"ANSWER TO EVALUATE:\n{answer_text}\n\n"
            f"RETRIEVED CONTEXT (what the system actually had available):\n{source_block}"
        )

    empty_result = {
        "available": False, "overall": None, "dimensions": [], "verdict": "",
        "strengths": [], "issues": [],
    }

    try:
        raw = llm_client.complete(
            system=system_prompt, user_content=user_content, max_tokens=1200,
            trace=trace, name="answer_evaluation_general_knowledge" if no_context_mode else "answer_evaluation",
        )
        parsed = _extract_json(raw)
        if not parsed:
            raise ValueError(f"evaluator did not return parseable JSON: {raw[:200]!r}")

        raw_dimensions = parsed.get("dimensions") or []
        dimensions = []
        if isinstance(raw_dimensions, list):
            for d in raw_dimensions:
                if not isinstance(d, dict):
                    continue
                dimensions.append({
                    "name": str(d.get("name", "") or "")[:80] or "Unnamed dimension",
                    "score": _clamp_score(d.get("score")),
                    "assessment": str(d.get("assessment", "") or "")[:400],
                })
        # Fall back to the expected dimension list (scoreless) if the model
        # returned something unusable - so the UI always has 6 rows to
        # render rather than an empty/broken table.
        if not dimensions:
            dimensions = [{"name": n, "score": None, "assessment": ""} for n in _EXPECTED_DIMENSION_NAMES]

        def _str_list(key, limit):
            vals = parsed.get(key) or []
            if not isinstance(vals, list):
                return []
            return [str(v)[:300] for v in vals][:limit]

        result = {
            "available": True,
            "overall": _clamp_score(parsed.get("overall")),
            "dimensions": dimensions,
            "verdict": str(parsed.get("verdict", "") or "")[:600],
            "strengths": _str_list("strengths", 3),
            "issues": _str_list("issues", 3),
        }
        logger.info(f"[answer_evaluation] question={question[:50]!r} overall={result['overall']}")
        return result
    except Exception as e:
        logger.warning(f"[answer_evaluation] evaluate_answer failed ({e}) - evaluation unavailable.")
        return empty_result
