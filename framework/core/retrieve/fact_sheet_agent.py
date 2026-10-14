"""
fact_sheet_agent.py - 2026-09-11: NEW. The first, sequential stage of
retrieval_mode="structured_research" (see hierarchical_orchestrator.py's
module docstring for the full pipeline). Distills the user's raw question
(plus any attached file's content) into a structured {"facts", "tax_position"}
pair that every downstream agent (Law/Case Law/Tribunal/Opinion) receives
INSTEAD OF the raw question.

This matters for more than tidiness: the Case Law agent specifically needs a
clear tax_position statement to know which side (taxpayer vs Revenue) counts
as "favourable" for THIS matter (see case_law_classification prompt/
case_law_agent.py) - that mapping is meaningless without a framed position to
test it against.
"""
from __future__ import annotations

import json
import logging

from framework.core.infra import llm_client

logger = logging.getLogger("fact_sheet_agent")


def _format_attached_file_context(attached_file_context: dict | None) -> str:
    """Renders the attached file's schema_summary (the same text
    structured_data_workspace.get_attachment() already builds and
    structured_data_query/agent_routing v2 already reuse) into the prompt's
    ATTACHED FILE CONTENT block. Returns "(none)" when no file is attached
    so the template always has readable text, not an empty string."""
    if not attached_file_context:
        return "(none)"
    filename = attached_file_context.get("filename", "attached file")
    schema_summary = attached_file_context.get("schema_summary", "")
    return f"File: {filename}\n{schema_summary}"


def _parse_fact_sheet(raw: str, question: str) -> dict:
    """Parses the model's JSON response defensively - same markdown-fence
    stripping as retrieval_planner._parse_plan(). Falls back to a minimal
    fact sheet built from the raw question on a parse failure, rather than
    raising - a downstream agent working from an under-specified but
    present fact sheet is better than the whole structured_research pipeline
    failing outright over one malformed JSON response."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        parsed = json.loads(text)
        return {
            "facts": parsed.get("facts", "") or "",
            "tax_position": parsed.get("tax_position", "") or question,
        }
    except json.JSONDecodeError:
        logger.warning(f"[fact_sheet_agent] Model did not return valid JSON, falling back to the raw "
                        f"question as tax_position with no extracted facts. Raw output: {text!r}")
        return {"facts": "(could not be extracted - fact-sheet step failed to parse)", "tax_position": question}


def build_fact_sheet(question: str, attached_file_context: dict | None = None, trace=None) -> dict:
    """Returns {"facts": str, "tax_position": str}. One LLM call. Every
    downstream agent in the structured_research pipeline takes this dict as
    input, not the raw `question` string - see this module's docstring."""
    from domain_packs.tax.prompts.tax_prompts import PROMPT_PACK  # local import - see
    # retrieval_planner.py's plan_retrieval() for why this mirrors graphrag.py's own
    # constructor-injected prompt_pack pattern rather than a module-level import.

    # 2026-09-11: BUGFIX - was pinned to version="v1", which SILENTLY
    # BYPASSED the v2 fix (registered the same day) for doctrinal/case-
    # comparison questions like "How did Azadi Bachao Andolan limit the
    # reach of McDowell?" - v1 only recognizes transaction-shaped questions
    # and misclassifies this shape as "insufficient information," which then
    # poisons every downstream agent's search. Every other agent in this
    # pipeline calls get_prompt() with no explicit version (always resolving
    # to the latest) - this call site was the one exception, meaning the v2
    # fix was registered but never actually took effect here. Omitting
    # version now resolves to the latest (v2) automatically, matching
    # convention everywhere else.
    prompt = PROMPT_PACK.get_prompt("fact_sheet_extraction")
    system = prompt.template.format(
        attached_file_context=_format_attached_file_context(attached_file_context),
        question=question,
    )
    raw = llm_client.complete(system=system, user_content=question, max_tokens=600,
                               trace=trace, name="fact_sheet_extraction")
    fact_sheet = _parse_fact_sheet(raw, question)
    logger.info(f"[fact_sheet_agent] tax_position={fact_sheet['tax_position']!r}")
    return fact_sheet
