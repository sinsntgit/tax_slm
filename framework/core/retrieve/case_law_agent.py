"""
case_law_agent.py - 2026-09-11: NEW. The "Case Law" stage of retrieval_mode=
"structured_research" (see hierarchical_orchestrator.py's module docstring),
covering the customer-specified sub-categories: favourable Case Law,
unfavourable Case Law.

IMPORTANT - this does NOT classify each case's outcome with a fresh LLM
judgment call. CaseLaw nodes already carry a `stance` property
(taxpayer_favorable/revenue_favorable/mixed/unknown), derived
DETERMINISTICALLY at ingestion from each case's own extracted
issue_outcomes (see domain_packs/tax/converters/extract_case_law.py's
derive_stance()) - re-judging that here would risk disagreeing with an
already-correct, already-reviewed value for no benefit. What `stance` alone
does NOT know is which side (taxpayer or Revenue) the CURRENT matter's
tax_position is arguing for - a taxpayer_favorable case is "favourable"
evidence only if this matter is also arguing the taxpayer's side. This
agent's only real work is that one cheap, BATCHED mapping step (one LLM
call over every candidate case's stored stance, not one call per case) -
see the case_law_classification prompt.
"""
from __future__ import annotations

import json
import logging

from framework.core.infra import llm_client
from framework.core.retrieve.graphrag import GraphRAGRetriever

logger = logging.getLogger("case_law_agent")

CANDIDATE_TOP_K = 15


def find_candidate_case_law(retriever: GraphRAGRetriever, tax_position: str,
                             question: str | None = None) -> list[dict]:
    """Finds candidate CaseLaw nodes via BOTH of the same two paths
    retrieval_mode="hybrid" (KG + RAG) already uses for a named-case
    question, merged and deduped:

    1. CYPHER, via GraphRAGRetriever.run_graph_retrieval() - the SAME
       NL2Cypher path hybrid mode uses, which already has dedicated
       "NAMED-CASE QUESTIONS" guidance (toLower(full_citation) CONTAINS
       '<name>') for exactly a question that names specific cases - see the
       nl2cypher prompt. This is the primary path for a question like "How
       did Azadi Bachao Andolan limit the reach of McDowell?", which names
       two real cases directly - vector similarity over a paraphrased
       tax_position is not a reliable way to find a specific named entity
       (the same reasoning this project's own nl2cypher NAMED-CASE section
       exists for). run_graph_retrieval()'s own result["nodes"] dict already
       carries the full node shape (db_id/label/properties) directly - no
       chunk round-trip needed, unlike retrieval_orchestrator.run_tool()'s
       CYPHER branch (which additionally resolves chunks for answer
       generation, not needed here).
    2. VECTOR, via vector_search_chunks() + get_entities_for_chunks() - the
       original path, kept as a supplement (not a replacement) for topical/
       conceptual questions with no specific case named, or to catch a
       relevant case the Cypher pass's NL2Cypher call didn't think to match.

    2026-09-11: PUBLIC (not module-private) - tribunal_case_law_agent.py
    needs the IDENTICAL candidate-finding step for its own HC/SC/
    jurisdictional/pre-change-in-law bucketing (same candidate CaseLaw
    nodes, different thing done with them afterward) - shared here rather
    than duplicated, so the two agents can never see a different candidate
    set for the same tax_position.

    2026-09-11: `question` param ADDED (optional, defaults to tax_position
    when omitted - fully backward compatible with any existing caller/test).
    BUGFIX for a reported case: "How did Azadi Bachao Andolan limit the
    reach of McDowell?" found ZERO candidates even though both cases are
    confirmed ingested, while retrieval_mode="hybrid" (KG+RAG) answers the
    SAME literal question correctly. Root cause: this function's Cypher path
    was only ever given fact_sheet_agent's `tax_position` - an LLM
    PARAPHRASE of the question ("The question asks how the Supreme Court's
    ruling in X affected the scope of Y..."), not the user's original
    wording. NL2Cypher's NAMED-CASE QUESTIONS guidance matches on short,
    recognizable case-name text - a paraphrase can drift enough (different
    phrasing, a dropped name, added framing language) that the SAME NL2Cypher
    prompt that reliably matches the literal question fails on its
    restatement. Passing the ORIGINAL question through for the Cypher path
    specifically (vector search and classification still use tax_position -
    see retrieve_case_law() below) restores the same reliability hybrid mode
    already has, per explicit instruction to keep this approach identical to
    hybrid/KG+RAG's."""
    cypher_query_text = question if question is not None else tax_position
    case_law_nodes = []
    seen = set()

    def _add(node):
        if node["db_id"] not in seen:
            case_law_nodes.append(node)
            seen.add(node["db_id"])

    # Path 1: CYPHER (same NL2Cypher path + NAMED-CASE guidance hybrid mode
    # uses) - given the ORIGINAL question text, not the fact sheet's
    # paraphrase, so named-case matching is exactly as reliable as it is in
    # hybrid mode for the identical question (see this function's docstring).
    graph_result = retriever.run_graph_retrieval(cypher_query_text)
    for node in graph_result.get("nodes", {}).values():
        if node.get("label") == "CaseLaw":
            _add(node)

    # Path 2: VECTOR (original path, kept as a supplement - see this
    # function's docstring). Uses tax_position, not the raw question - the
    # fact sheet's clean restatement is a better semantic-similarity anchor
    # than the user's original phrasing, which may include framing language
    # ("how did X limit...") that doesn't itself appear in case text.
    hits = retriever.vector_search_chunks(tax_position, top_k=CANDIDATE_TOP_K)
    chunk_ids = [h["chunk_id"] for h in hits]
    for node in retriever.get_entities_for_chunks(chunk_ids):
        if node.get("label") == "CaseLaw":
            _add(node)

    return case_law_nodes


def _format_candidates_for_prompt(case_nodes: list[dict]) -> str:
    lines = []
    for n in case_nodes:
        props = n.get("properties", {})
        lines.append(json.dumps({
            "case_id": n["db_id"],
            "citation": props.get("full_citation", "(unknown citation)"),
            "stance": props.get("stance", "unknown"),
            "stance_basis": props.get("stanceBasis", ""),
        }))
    return "\n".join(lines) if lines else "(no candidates)"


def _parse_classifications(raw: str) -> list[dict]:
    """Parses the model's JSON array response defensively - same
    markdown-fence stripping as retrieval_planner._parse_plan(). Returns an
    empty list on a parse failure (callers treat that as "could not
    classify," not as an error that should break the whole agent)."""
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
        logger.warning(f"[case_law_agent] Expected a JSON array, got {type(parsed)}: {text!r}")
        return []
    except json.JSONDecodeError:
        logger.warning(f"[case_law_agent] Model did not return valid JSON. Raw output: {text!r}")
        return []


def retrieve_case_law(retriever: GraphRAGRetriever, fact_sheet: dict, question: str | None = None,
                       trace=None) -> dict:
    """Returns {"favourable": [...], "unfavourable": [...], "unclassified": [...],
    "candidates_found": int}. Each list entry is {"citation", "stance",
    "reasoning"}. `candidates_found` is surfaced separately so the evidence
    panel can say "no candidate case law found at all" (distinct from
    "candidates found but none classified as favourable/unfavourable").

    2026-09-11: `question` param ADDED (optional) - forwarded to
    find_candidate_case_law()'s CYPHER path so named-case matching works off
    the user's original wording, not fact_sheet_agent's paraphrase. See that
    function's docstring for the full bugfix rationale. Classification below
    still uses `tax_position`, not `question` - the classification prompt
    needs the FRAMED position (which side is being argued) to map stance
    onto favourable/unfavourable, which the raw question alone doesn't give."""
    tax_position = fact_sheet.get("tax_position", "")
    candidates = find_candidate_case_law(retriever, tax_position, question=question)

    if not candidates:
        logger.info("[case_law_agent] No candidate case law found via CYPHER or vector search.")
        return {"favourable": [], "unfavourable": [], "unclassified": [], "candidates_found": 0}

    from domain_packs.tax.prompts.tax_prompts import PROMPT_PACK  # local import - see
    # retrieval_planner.py's plan_retrieval() for why (mirrors graphrag.py's own constructor-
    # injected prompt_pack pattern rather than a module-level import).

    prompt = PROMPT_PACK.get_prompt("case_law_classification", version="v1")
    system = prompt.template.format(
        tax_position=tax_position,
        candidate_cases=_format_candidates_for_prompt(candidates),
    )
    raw = llm_client.complete(system=system, user_content=tax_position, max_tokens=1500,
                               trace=trace, name="case_law_classification")
    classifications = _parse_classifications(raw)

    by_id = {n["db_id"]: n for n in candidates}
    favourable, unfavourable, unclassified = [], [], []
    for c in classifications:
        node = by_id.get(c.get("case_id"))
        if not node:
            continue
        entry = {
            "citation": node.get("properties", {}).get("full_citation", "(unknown citation)"),
            "stance": node.get("properties", {}).get("stance", "unknown"),
            "reasoning": c.get("reasoning", ""),
        }
        classification = c.get("classification", "").upper()
        if classification == "FAVOURABLE":
            favourable.append(entry)
        elif classification == "UNFAVOURABLE":
            unfavourable.append(entry)
        else:
            unclassified.append(entry)

    logger.info(f"[case_law_agent] {len(candidates)} candidate(s): "
                f"{len(favourable)} favourable, {len(unfavourable)} unfavourable, "
                f"{len(unclassified)} unclassified.")
    return {"favourable": favourable, "unfavourable": unfavourable, "unclassified": unclassified,
            "candidates_found": len(candidates)}
