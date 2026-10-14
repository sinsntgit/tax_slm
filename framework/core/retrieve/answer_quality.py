"""
ANSWER QUALITY JUDGE - a second, independent LLM pass that scores the
answer /api/ask is about to return (not the retrieval candidates - the
final generated text) and reports back a 0-1 score + concrete gaps.

2026-09-03: EXTENDED, per the uploaded "Legal Research Prompt - Recommended
Revisions" document's LEGAL_ACCURACY_CHECK module (a 20-item silent
pre-answer checklist) and the user's explicit choice to extend THIS
existing judge rather than build a separate second-pass validator. This
judge is NO LONGER domain-agnostic (it was, previously - see the prior
version of this docstring) - it now also scores three tax/legal-specific
dimensions (statutory_fidelity, holding_discipline, authority_hierarchy),
condensed from the LEGAL_ACCURACY_CHECK module's 20 items into 3 scoreable
groups rather than replicating all 20 as separate 0-1 scores, which would
make the judge prompt unwieldy and the individual scores noisy. The
original 3 domain-agnostic dimensions (groundedness, completeness,
specificity) are unchanged and still carried.

This is deliberately a SEPARATE call from answer_generation, not a change
to that prompt - self-grading in the same call that produced the text is
unreliable (a model rarely flags its own answer as incomplete), and a
second, differently-instructed pass catches gaps a generation-focused
prompt has no incentive to notice on its own.

Used by app.py's /api/ask: if the score comes back below QUALITY_MIN_SCORE,
the caller gets back enough signal to retry once with more context (a wider
chunk pool) - see retry_with_more_context() below - rather than silently
returning a thin answer. This is a bounded, single retry, not an open loop:
if the second attempt still scores low, that answer is returned anyway
(honestly, with its lower score attached) rather than looping indefinitely
against a corpus that may simply not have more to give.
"""
import json
import logging
import re

from framework.core.retrieve import grounding

logger = logging.getLogger("answer_quality")

QUALITY_MIN_SCORE = 0.6

JUDGE_SYSTEM_PROMPT = """You are a strict, independent quality reviewer for an AI system that answers \
Indian tax-law questions using retrieved source documents. You did NOT write the answer below - your job \
is to grade it, not defend it.

Score the ANSWER against the QUESTION and the SOURCE TEXT it was supposed to be grounded in, on six \
dimensions, each 0.0-1.0:

1. groundedness: Is every factual claim in the answer actually supported by the source text? An answer \
that states something the source text doesn't say (even if plausible) should score low here.
2. completeness: Does the answer cover what the question actually asked, using the relevant material \
available in the source text? An answer that ignores clearly-relevant source text, or stops short of a \
question with multiple parts, should score low here.
3. specificity: Does the answer cite precise sections/clauses/citations and give concrete detail, rather \
than vague generalities the source text could have made more specific?
4. statutory_fidelity: Where the answer depends on legislation, does it preserve every legally operative \
statutory qualifier (thresholds, "in value" vs "in number", provisos, exceptions, explanations, \
cross-references) rather than paraphrasing them away or compressing a multi-element statutory test into a \
simplified summary? Does it keep the statutory DEFINITION of a concept analytically separate from the \
separate provision that grants a tax consequence (exemption/deduction/exclusion), rather than treating \
satisfaction of one as automatically establishing the other? Score low if a material qualifier, \
threshold, or definition/consequence conflation is missing or wrong.
5. holding_discipline: Does the answer correctly distinguish an actual judicial holding ("held") from \
reasoning, dicta/observation, a factual finding ("found on the facts"), and a party submission ("argued"/\
"contended"), and avoid presenting a fact-specific finding as a general statutory rule? Score low if the \
answer overstates what a case actually decided, or blurs law/fact/submission distinctions.
6. authority_hierarchy: Does the answer correctly treat Supreme Court and jurisdictional High Court \
authority as binding, other High Court/ITAT/AAR authority as persuasive (never described as binding), and \
correctly reflect any retrieved OVERRULES/REVERSES/DISTINGUISHES/FOLLOWS/APPROVES/SUPERSEDES/WITHDRAWS \
relationship's actual scope (not extending a displacement beyond the specific proposition it affects)? \
Score low if a persuasive authority is treated as binding, or an overruling/reversal's scope is \
overstated or ignored.

Return ONLY a JSON object, no other text, in exactly this shape:
{"groundedness": 0.0-1.0, "completeness": 0.0-1.0, "specificity": 0.0-1.0, "statutory_fidelity": 0.0-1.0, \
"holding_discipline": 0.0-1.0, "authority_hierarchy": 0.0-1.0, "overall": 0.0-1.0, \
"gaps": ["short specific gap 1", "short specific gap 2", ...]}

"overall" is your holistic judgment (not required to be a plain average - weight groundedness and \
statutory_fidelity highest, since an ungrounded or statutorily-imprecise answer is worse than a merely \
incomplete one). "gaps" should list 0-3 SHORT, concrete, actionable gaps (e.g. "Drops the 'in value' \
qualifier on the three-fourths shareholding test" or "Treats the ITAT ruling as binding precedent") - \
empty list if there are none. If the question is purely factual/administrative with no statutory or \
case-law dimension (e.g. a due date), score statutory_fidelity/holding_discipline/authority_hierarchy as \
1.0 rather than penalizing an answer for not discussing something the question didn't call for. If the \
source text was empty/unavailable, judge only groundedness (should be 1.0 for an honest "I don't know") \
and completeness (should be 1.0 too, if there was nothing to draw on)."""


def _extract_json(text: str) -> dict | None:
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


def judge_answer(question: str, answer_text: str, top_chunks: list[dict], plain_rows: list[dict],
                  *, trace=None) -> dict:
    """Scores an already-generated answer. Returns:
        {"overall": float, "groundedness": float, "completeness": float,
         "specificity": float, "statutory_fidelity": float,
         "holding_discipline": float, "authority_hierarchy": float,
         "gaps": [str, ...], "available": bool}
    The last three fields are 2026-09-03 NEW - see this module's docstring
    for why the judge is no longer domain-agnostic. All three fall back to
    None (like every other score field) if the judge call fails, and are
    still round-tripped through even for a caller that only inspects
    `overall`/`gaps`, so no existing caller needs to change.
    `available: False` (score fields None) if the judge call itself fails -
    callers should treat that as "couldn't assess", not as a low score; a
    quality gate that silently fails closed on infra hiccups would make the
    app less reliable, not more."""
    from framework.core.infra import llm_client

    source_block = ""
    if top_chunks:
        source_block += "\n\n---\n\n".join(f"[chunk_id={c['chunk_id']}]\n{c['text'][:1200]}" for c in top_chunks[:15])
    if plain_rows:
        # Was a flat json.dumps(plain_rows, indent=2)[:2000] - the same
        # naive-truncation bug fixed in app.py's _build_context_parts() (see
        # grounding.py's module-level comment on graph_rows_to_context_json)
        # existed here too, and at an even tighter 2000-char budget - which
        # matters especially for THIS judge, since its whole job is grading
        # completeness against the source material it's shown.
        source_block += ("\n\nGraph query results (JSON):\n"
                          + grounding.graph_rows_to_context_json(plain_rows, max_chars=4000))
    if not source_block:
        source_block = "(no source text or graph rows were retrieved for this question)"

    user_content = f"QUESTION:\n{question}\n\nANSWER TO GRADE:\n{answer_text}\n\nSOURCE TEXT:\n{source_block}"

    try:
        raw = llm_client.complete(
            system=JUDGE_SYSTEM_PROMPT, user_content=user_content, max_tokens=500,
            trace=trace, name="answer_quality_judge",
        )
        parsed = _extract_json(raw)
        if not parsed:
            raise ValueError(f"judge did not return parseable JSON: {raw[:200]!r}")

        def _f(key):
            try:
                return round(max(0.0, min(1.0, float(parsed.get(key, 0.0)))), 2)
            except (TypeError, ValueError):
                return 0.0

        gaps = parsed.get("gaps") or []
        if not isinstance(gaps, list):
            gaps = []
        gaps = [str(g)[:200] for g in gaps][:3]

        result = {
            "available": True,
            "overall": _f("overall"),
            "groundedness": _f("groundedness"),
            "completeness": _f("completeness"),
            "specificity": _f("specificity"),
            # 2026-09-03: NEW - see module docstring.
            "statutory_fidelity": _f("statutory_fidelity"),
            "holding_discipline": _f("holding_discipline"),
            "authority_hierarchy": _f("authority_hierarchy"),
            "gaps": gaps,
        }
        logger.info(f"[answer_quality] question={question[:50]!r} overall={result['overall']} "
                    f"groundedness={result['groundedness']} completeness={result['completeness']} "
                    f"specificity={result['specificity']} statutory_fidelity={result['statutory_fidelity']} "
                    f"holding_discipline={result['holding_discipline']} "
                    f"authority_hierarchy={result['authority_hierarchy']} gaps={gaps}")
        return result
    except Exception as e:
        logger.warning(f"[answer_quality] judge_answer failed ({e}) - quality unavailable, not scored as low.")
        return {"available": False, "overall": None, "groundedness": None,
                "completeness": None, "specificity": None, "statutory_fidelity": None,
                "holding_discipline": None, "authority_hierarchy": None, "gaps": []}
