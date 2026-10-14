"""
tribunal_case_law_agent.py - 2026-09-11: NEW. The "Tribunal Case Law" stage
of retrieval_mode="structured_research" (see hierarchical_orchestrator.py's
module docstring), covering the customer-specified sub-categories: HC Case
Law, SC Case Law, Jurisdictional Case Law, Case Law Before Change in Law.

IMPORTANT - like case_law_agent.py, this does NOT re-derive forum level or
jurisdiction from scratch. CaseLaw nodes already carry `forum_level`
(Tribunal/High Court/Supreme Court/AAR/Commissioner (Appeals)) and
`jurisdiction_scope` (a territory for High Court rulings, "All India" for
Supreme Court), both computed DETERMINISTICALLY at ingestion (see
extract_case_law.py's classify_forum_level()/derive_jurisdiction_scope()).
This agent's only real judgment call is bucketing relative to the CURRENT
matter: which cases count as jurisdictional for THIS matter's jurisdiction,
and which predate the change-in-law date.

PIPELINE ORDERING NOTE (see hierarchical_orchestrator.py): unlike Case Law/
Opinion, this agent is NOT run fully in parallel with the Law agent - it
needs the Law agent's own amendment-chain findings to know the change-in-law
date, so the orchestrator runs Law first, then this agent (alongside Case
Law/Opinion, which have no such dependency).
"""
from __future__ import annotations

import json
import logging
import re

from framework.core.infra import llm_client
from framework.core.retrieve.case_law_agent import find_candidate_case_law
from framework.core.retrieve.graphrag import GraphRAGRetriever

logger = logging.getLogger("tribunal_case_law_agent")


# 2026-10-13: NEW - full ISO dates (what chunking.py's _normalize_date()
# writes to effective_from/effective_to) vs a bare year (amendingActYear).
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_YEAR_RE = re.compile(r"^\d{4}$")
# Properties that record WHEN a provision's current/omitted text took effect
# (see chunking.py's amendment-props builder: effective_from for insertion/
# substitution, effective_to for an omission) - in priority order.
_DATE_KEYS = ("effective_from", "effective_to", "effective_date")
_YEAR_KEYS = ("amendingActYear",)


def _iter_row_dicts(row):
    """Yields every property dict in one plain_rows row: the row itself (a
    scalar-alias RETURN like `RETURN n.effective_from AS effective_from`),
    plus each node/relationship property dict nested one level down - which
    is the NORMAL shape, since graphrag.run_graph_retrieval() stores a
    returned node as row[alias] = node.properties (e.g. {"provision": {...},
    "r": {...}, "act": {...}})."""
    if not isinstance(row, dict):
        return
    yield row
    for value in row.values():
        if isinstance(value, dict):
            yield value


def _extract_change_in_law_date(law_findings: dict) -> str:
    """Best-effort extraction of a change-in-law date from the Law agent's
    graph rows. Returns "" (not a guess) if none is found - the
    tribunal_bucketing prompt is explicitly instructed to leave
    PRE_CHANGE_IN_LAW unassigned rather than guess when this is empty.

    2026-10-13: BUGFIX - previously only checked TOP-LEVEL keys of each row
    (row.get("effective_from")), but plain_rows nest each returned node's
    properties under its Cypher alias (see _iter_row_dicts()), so a normal
    `RETURN provision, r, act` amendment query never matched and this
    returned "" essentially every time - meaning the PRE_CHANGE_IN_LAW bucket
    could never be populated. Now:
      - looks inside nested node/relationship dicts too;
      - checks the amendment-chain sub-query's own rows (law_agent's
        pool["amendment_rows"]) FIRST and only falls back to all Law rows if
        those carry no date, since an unrelated provision's or circular's
        effective_from is a much weaker proxy for "the change in law";
      - prefers a full ISO date over a bare amendingActYear, and returns the
        MOST RECENT candidate (the latest change is the one a "has the
        position changed" question is about) rather than whichever row
        happened to come first in Cypher's arbitrary row order. A bare year
        is returned as "YYYY" - honest about its precision rather than
        padded into a made-up day/month.
    Deterministic, no LLM call."""
    row_sets = (law_findings.get("amendment_rows") or [], law_findings.get("plain_rows") or [])
    for rows in row_sets:
        dates, years = set(), set()
        for row in rows:
            for props in _iter_row_dicts(row):
                for key in _DATE_KEYS:
                    value = str(props.get(key) or "").strip()
                    if _ISO_DATE_RE.match(value):
                        dates.add(value)
                for key in _YEAR_KEYS:
                    value = str(props.get(key) or "").strip()
                    if _YEAR_RE.match(value):
                        years.add(value)
        if dates:
            return max(dates)  # ISO YYYY-MM-DD sorts lexicographically == chronologically
        if years:
            return max(years)
    return ""


def _format_candidates_for_prompt(case_nodes: list[dict]) -> str:
    lines = []
    for n in case_nodes:
        props = n.get("properties", {})
        lines.append(json.dumps({
            "case_id": n["db_id"],
            "citation": props.get("full_citation", "(unknown citation)"),
            "forum_level": props.get("forum_level", ""),
            "jurisdiction_scope": props.get("jurisdiction_scope", ""),
            "decisionDate": props.get("decisionDate", ""),
        }))
    return "\n".join(lines) if lines else "(no candidates)"


def _parse_buckets(raw: str) -> list[dict]:
    """Same defensive JSON-array parsing as case_law_agent._parse_classifications()."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        logger.warning(f"[tribunal_case_law_agent] Expected a JSON array, got {type(parsed)}: {text!r}")
        return []
    except json.JSONDecodeError:
        logger.warning(f"[tribunal_case_law_agent] Model did not return valid JSON. Raw output: {text!r}")
        return []


def retrieve_tribunal_case_law(retriever: GraphRAGRetriever, fact_sheet: dict, law_findings: dict,
                                question: str | None = None, trace=None) -> dict:
    """Returns {"hc": [...], "sc": [...], "jurisdictional": [...],
    "pre_change_in_law": [...], "change_in_law_date": str,
    "candidates_found": int}. Each list entry is {"citation", "forum_level",
    "jurisdiction_scope", "decisionDate"} - a case can appear in multiple
    lists (e.g. a jurisdictional High Court case decided before the change
    in law appears in both "hc" and "jurisdictional" and "pre_change_in_law").
    `change_in_law_date` is surfaced separately (may be "" if the Law
    agent's findings didn't establish one) so the evidence panel can show
    explicitly whether pre/post-change bucketing was even possible.

    2026-09-11: `question` param ADDED (optional) - forwarded to the shared
    find_candidate_case_law()'s CYPHER path (see case_law_agent.py's
    docstring for the full bugfix rationale: matching on the user's ORIGINAL
    wording, not fact_sheet_agent's paraphrase, is what makes named-case
    matching as reliable here as it already is in retrieval_mode="hybrid")."""
    tax_position = fact_sheet.get("tax_position", "")
    matter_jurisdiction = fact_sheet.get("facts", "")  # jurisdiction is typically stated as part of the facts
    change_in_law_date = _extract_change_in_law_date(law_findings)

    candidates = find_candidate_case_law(retriever, tax_position, question=question)
    if not candidates:
        logger.info("[tribunal_case_law_agent] No candidate case law found via CYPHER or vector search.")
        return {"hc": [], "sc": [], "jurisdictional": [], "pre_change_in_law": [],
                "change_in_law_date": change_in_law_date, "candidates_found": 0}

    from domain_packs.tax.prompts.tax_prompts import PROMPT_PACK  # local import - see
    # retrieval_planner.py's plan_retrieval() for why (mirrors graphrag.py's own constructor-
    # injected prompt_pack pattern rather than a module-level import).

    prompt = PROMPT_PACK.get_prompt("tribunal_bucketing", version="v1")
    system = prompt.template.format(
        matter_jurisdiction=matter_jurisdiction or "(not specified)",
        change_in_law_date=change_in_law_date or "(not established)",
        candidate_cases=_format_candidates_for_prompt(candidates),
    )
    raw = llm_client.complete(system=system, user_content=tax_position, max_tokens=1500,
                               trace=trace, name="tribunal_bucketing")
    bucketed = _parse_buckets(raw)

    by_id = {n["db_id"]: n for n in candidates}
    result = {"hc": [], "sc": [], "jurisdictional": [], "pre_change_in_law": []}
    bucket_key_map = {"HC": "hc", "SC": "sc", "JURISDICTIONAL": "jurisdictional",
                       "PRE_CHANGE_IN_LAW": "pre_change_in_law"}
    for b in bucketed:
        node = by_id.get(b.get("case_id"))
        if not node:
            continue
        props = node.get("properties", {})
        entry = {"citation": props.get("full_citation", "(unknown citation)"),
                 "forum_level": props.get("forum_level", ""),
                 "jurisdiction_scope": props.get("jurisdiction_scope", ""),
                 "decisionDate": props.get("decisionDate", "")}
        for bucket_label in b.get("buckets", []):
            key = bucket_key_map.get(str(bucket_label).upper())
            if key:
                result[key].append(entry)

    logger.info(f"[tribunal_case_law_agent] {len(candidates)} candidate(s): "
                f"HC={len(result['hc'])}, SC={len(result['sc'])}, "
                f"jurisdictional={len(result['jurisdictional'])}, "
                f"pre_change_in_law={len(result['pre_change_in_law'])}.")
    result["change_in_law_date"] = change_in_law_date
    result["candidates_found"] = len(candidates)
    return result
