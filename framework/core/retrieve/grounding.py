"""
CITATION & GROUNDING - PROMOTED unchanged from tax_kg_poc/webapp/grounding.py.
Produces source_references/kg_paths_used/confidence for a question + reranked
chunks + graph rows, and post-generation citation verification. 100%
domain-agnostic - operates on the generic chunk/row shapes every domain pack
produces, no tax concepts anywhere in it.
"""
import json
import logging
import re

logger = logging.getLogger("citation_grounding")


def _kg_path_strings(plain_rows: list[dict]) -> list[str]:
    paths = []
    for row in plain_rows:
        rel = next((v for v in row.values() if isinstance(v, dict) and "type" in v), None)
        if rel:
            paths.append(f"[{rel.get('type', '?')}] relationship in graph")
    return paths


def _confidence_score(reranked_chunks: list[dict], plain_rows: list[dict]) -> float:
    if not reranked_chunks:
        return 0.0

    score = 0.5
    top_chunk = reranked_chunks[0]
    # reranker.py's rerank_score is ALREADY a 0-1 sigmoid-normalized value
    # (see reranker.py's _sigmoid()) - it used to be the raw, unbounded
    # cross-encoder logit, and this line's old "(top_score + 10) / 20"
    # normalization was written for THAT scale. Left unchanged after the
    # sigmoid switch, it silently mis-normalized every confidence score
    # (a perfect rerank_score of 1.0 was compressing down to ~0.55 here) -
    # fixed by using the already-0-1 value directly.
    # 2026-09-11: BUGFIX - reranker.py's skip_reranker path sets
    # rerank_score to an EXPLICIT None (see reranker.rerank()'s skip_rerank
    # branch), not an absent key - `.get("rerank_score", 0.0)` only supplies
    # its default when the key is missing, so a skip-rerank chunk was
    # passing None straight through into the max()/min() call below.
    # `or 0.0` catches both the missing-key AND explicit-None cases.
    top_score = top_chunk.get("rerank_score") or 0.0
    score += 0.3 * max(0.0, min(1.0, top_score))

    # A chunk that only cleared reranker.py's lower FALLBACK bar (see its
    # docstring) is a real but less-certain match - reflect that honestly
    # instead of reporting the same confidence as a high-confidence primary
    # match.
    if top_chunk.get("relevance_tier") == "fallback":
        score -= 0.15

    if plain_rows:
        score += 0.2

    return round(max(0.0, min(1.0, score)), 2)


def ground_answer(question: str, reranked_chunks: list[dict], plain_rows: list[dict]) -> dict:
    source_references = [
        {
            "chunk_id": c["chunk_id"],
            "via": c.get("source", "unknown"),
            "preview": c["text"][:200] + ("…" if len(c["text"]) > 200 else ""),
            # 2026-09-11: BUGFIX - see _confidence_score()'s matching fix
            # above for why `.get(..., 0.0)` alone isn't enough: skip_rerank
            # chunks carry an explicit rerank_score=None (key present), so
            # round(None, 3) was raising TypeError and taking down
            # /api/ask/fast whenever the skip-rerank toggle was used.
            "rerank_score": round(c.get("rerank_score") or 0.0, 3),
        }
        for c in reranked_chunks
    ]

    grounding = {
        "source_references": source_references,
        "kg_paths_used": _kg_path_strings(plain_rows),
        "confidence": _confidence_score(reranked_chunks, plain_rows),
    }

    logger.info(
        f"[citation_grounding] question={question[:50]!r} "
        f"refs={len(source_references)} confidence={grounding['confidence']}"
    )
    return grounding


# Matches the v4 "master tax research prompt"'s mandated primary citation
# style, "**[Document: <name>, p. X, lines Y-Z]**" (see tax_prompts.py's
# answer_generation v4, Section 13/14) - deliberately loose on what's inside
# the brackets since page/line/paragraph are each individually optional per
# that prompt's own "citation hierarchy" fallback list.
_BRACKET_DOC_RE = re.compile(r"\[\s*document\s*:\s*[^\]]+\]", re.IGNORECASE)

# A document-name/citation string has to be at least this long before a
# substring match counts as real evidence of citing - short strings (e.g. a
# 6-char case abbreviation) risk matching coincidentally inside ordinary
# prose rather than because the model actually named the source.
_MIN_DOC_CITATION_LEN = 12


def verify_citations(answer_text: str, reranked_chunks: list[dict], plain_rows: list[dict],
                      doc_details: dict[str, dict] | None = None) -> dict:
    """Checks whether the generated answer actually names the sources it was
    given something to work with - not whether it used ONE specific fixed
    citation format. The active answer_generation prompt's citation style has
    changed across versions (v1/v2's inline "(chunk_id=42)"; v4's "master"
    prompt mandates "[Document: <name>, p. X, lines Y-Z]" as primary, falling
    back to chunk_id/node id only when no page/line metadata exists; a
    user-saved Settings-page override - see prompt_config.py - can use a
    still different style this module has no visibility into). Hardcoding a
    check for only the oldest chunk_id format meant a fully-cited v4 answer
    routinely got flagged "no citation markers found" even though it was
    citing correctly, just not in that one exact shape - the UI's "Grounded"
    badge depends directly on this check, so that false negative showed up as
    a genuinely-grounded answer being labeled "Not grounded".
    Checks, roughly in order of how format-specific they are:
      1. `chunk_id=<N>` inline marker.
      2. A literal graph node id (e.g. "Section_9") appearing in the text.
      3. The `[Document: ...]` bracket format.
      4. Format-agnostic fallback: the real document name / chunk citation /
         entity citation of a chunk that was ACTUALLY retrieved (from
         `doc_details`, as returned by GraphRAGRetriever.get_document_details())
         appears anywhere in the answer text - covers any other citation
         phrasing without this module needing to know what it looks like.
    """
    had_context = bool(reranked_chunks or plain_rows)
    if not had_context:
        return {"citations_found": [], "citation_ok": True}

    citations_found = re.findall(r"chunk_id\s*=\s*\d+", answer_text, re.IGNORECASE)

    node_ids = set()
    for row in plain_rows:
        for v in row.values():
            if isinstance(v, dict) and "id" in v:
                node_ids.add(str(v["id"]))
    node_citations = [nid for nid in node_ids if nid and nid in answer_text]

    bracket_citations = _BRACKET_DOC_RE.findall(answer_text)

    doc_citations = []
    if doc_details:
        answer_lower = answer_text.lower()
        for detail in doc_details.values():
            for field in ("document_name", "chunk_citation", "entity_citation"):
                val = (detail or {}).get(field)
                if val and len(str(val)) >= _MIN_DOC_CITATION_LEN and str(val).lower() in answer_lower:
                    doc_citations.append(str(val))
                    break  # one matched field per chunk is enough signal

    citation_ok = bool(citations_found or node_citations or bracket_citations or doc_citations)
    if not citation_ok:
        logger.warning(
            f"[citation_grounding] Answer had retrieved context but NO citation markers - "
            f"answer_preview={answer_text[:120]!r}"
        )
    return {
        "citations_found": citations_found + node_citations + bracket_citations + doc_citations,
        "citation_ok": citation_ok,
    }


# Matches the "d. Referred Sources" bibliography section heading an
# answer_generation v5-style prompt mandates (see tax_prompts.py) -
# tolerant of the "a./b./c./d." lettering and markdown heading markers
# actually being present or not, since this runs against free-form model
# prose, not a rigid format.
_REFERRED_SOURCES_HEADING_RE = re.compile(r"referred\s+sources", re.IGNORECASE)

# A citation-looking line inside that section: some document-ish text ending
# in a known source-file extension, optionally followed by "- Page N".
_CITATION_LINE_RE = re.compile(r"([^\n]{5,150}?\.(?:txt|pdf))(?:\s*-\s*Page\s*\d+)?", re.IGNORECASE)


# --- Graph query results (JSON) context-sizing -------------------------------
# Shared by app.py's _build_context_parts() (the answer_generation prompt's
# own context) and answer_quality.py/answer_evaluation.py (the independent
# judge/evaluator passes) - all three need to hand a Cypher query's matched
# rows to an LLM as JSON, and all three were vulnerable to the same failure
# mode: a flat `json.dumps(plain_rows, indent=2)[:N]` string slice, which
# silently mangles multi-row results whenever the matched nodes carry rich
# text properties (very common for CaseLaw - `cases_cited` alone routinely
# runs several KB, since it's a merged, sorted list of every followed/
# distinguished/referred/overruled precedent). Confirmed concretely against a
# real "which High Court decisions have been overruled" query: Engineering
# Analysis's 3 OVERRULES rows serialized to 51,504 characters, and a flat
# [:4000] cutoff didn't survive past the FIRST row's own bulky properties -
# the model never even saw the matched case names, let alone a 2nd/3rd row
# like a precedent (e.g. Synopsis International) that has no embedded Chunk
# text of its own and so is ONLY reachable through this JSON block. Two
# changes fix this:
#   1. De-duplicate: the same node (e.g. the overruling CaseLaw) is often
#      repeated verbatim across every row it appears in - shown in full only
#      the first time, referenced by id afterward.
#   2. Cap known-bulky free-text/list properties PER FIELD (not the whole
#      blob) - so every row/case name always survives, only the verbose
#      narrative detail inside any one property gets trimmed.
# The properties actually needed to answer "which cases were overruled by
# whom" (precedents_overruled, holding, final_outcome, case_name,
# full_citation, court, forum_level) are deliberately NOT capped here - only
# the broader/less-targeted narrative fields are.
GRAPH_ROW_FIELD_CAPS = {
    "cases_cited": 1200, "treaty_references": 500, "other_act_references": 400,
    "statutes_cited": 500, "keywords": 300, "legal_concepts": 300,
    "questions_of_law": 400, "hasText": 800, "facts": 800, "extraction_notes": 200,
    # 2026-10-14: BUGFIX - retrievalSummary/retrievalKeywords (added to every
    # CaseLaw node 2026-08-28, see extract_case_law.py's derive_retrieval_*())
    # are embedding-only metadata that DUPLICATE holding/findings/keywords, yet
    # were uncapped here - on the real Engineering Analysis node they are the
    # two largest properties (~2.5KB/~3.1KB), which pushed the 3-row OVERRULES
    # case this module was tuned against from ~13,400 back to ~17,800 chars,
    # i.e. over GRAPH_ROWS_MAX_CHARS_DEFAULT, so the backstop was again slicing
    # the LAST row (the overruled precedent) off mid-JSON. Caught by
    # tests/test_grounding.py::test_engineering_analysis_fixture_rows_fit_backstop.
    "retrievalSummary": 600, "retrievalKeywords": 300,
}
# Overall safety net per caller, now a BACKSTOP rather than the primary
# truncation mechanism - per-field capping above should keep well under this
# for almost every real question. Verified against a real 3-row OVERRULES
# query (Engineering Analysis -> Samsung/Citrix/Synopsis): capping+dedup
# alone brings 51,504 chars down to ~13,400.
GRAPH_ROWS_MAX_CHARS_DEFAULT = 16_000


def cap_bulky_field(value, max_chars: int):
    """Trims a single property value to `max_chars` WITHOUT slicing raw JSON
    text (which can cut mid-string/mid-token and produce invalid or
    misleading fragments). Lists are trimmed item-by-item with a trailing
    "+N more" marker so at least the leading entries stay intact and
    complete; strings are simply cut with an ellipsis. Takes the budget as a
    plain parameter (not a shared/global default) so this stays safe under
    Flask's threaded request handling - see llm_client's USE_MODEL/MODEL_NAME
    race (task #182 in this project's history) for why a module-level
    mutable "current cap" would be the wrong pattern here."""
    if isinstance(value, list):
        kept, total = [], 0
        for item in value:
            item_len = len(str(item))
            if total + item_len > max_chars:
                kept.append(f"...({len(value) - len(kept)} more, truncated for length)")
                break
            kept.append(item)
            total += item_len
        return kept
    if isinstance(value, str) and len(value) > max_chars:
        return value[:max_chars] + "…"
    return value


def prepare_graph_rows_for_context(plain_rows: list[dict], field_caps: dict | None = None) -> list[dict]:
    """Returns a copy of plain_rows safe to json.dumps() into an LLM context:
    bulky fields capped per-property, and any node repeated across multiple
    rows shown in full only once. See module-level comment above for why
    this replaced a flat [:N] string slice at each of its 3 call sites."""
    caps = field_caps if field_caps is not None else GRAPH_ROW_FIELD_CAPS
    seen_node_ids: dict[str, dict] = {}  # node id -> the (capped) props dict already emitted

    def process_value(val):
        if not isinstance(val, dict):
            return val
        node_id = val.get("id") or val.get("node_id")
        if node_id and node_id in seen_node_ids:
            ref = {"id": node_id}
            for label_key in ("case_name", "full_citation", "hasText"):
                if val.get(label_key):
                    ref[label_key] = str(val[label_key])[:120]
                    break
            ref["_note"] = "full properties for this node already shown earlier in this context"
            return ref
        capped = {
            k: (cap_bulky_field(v, caps[k]) if k in caps else v)
            for k, v in val.items()
        }
        if node_id:
            seen_node_ids[node_id] = capped
        return capped

    return [{k: process_value(v) for k, v in row.items()} for row in plain_rows]


def graph_rows_to_context_json(plain_rows: list[dict], max_chars: int = GRAPH_ROWS_MAX_CHARS_DEFAULT,
                                field_caps: dict | None = None) -> str:
    """One-call convenience wrapper: prepare_graph_rows_for_context() +
    json.dumps() + a backstop length cap, since every call site wants
    exactly this sequence."""
    safe_rows = prepare_graph_rows_for_context(plain_rows, field_caps)
    rows_json = json.dumps(safe_rows, indent=2, ensure_ascii=False)
    if len(rows_json) > max_chars:
        rows_json = rows_json[:max_chars] + "\n... (truncated for length)"
    return rows_json


def check_referred_sources(answer_text: str, doc_details: dict[str, dict] | None) -> dict:
    """Best-effort, NON-BLOCKING cross-check for the "d. Referred Sources"
    bibliography section an answer_generation v5-style prompt asks the model
    to compile. A compiled reference list is the part of that response
    format most prone to an LLM inventing a plausible-looking source it
    never actually cited in the body - the model is effectively being asked
    to recall/summarize its own earlier output, which is exactly the kind of
    task LLMs quietly drift on. This never alters the answer or blocks the
    response (unlike verify_citations()'s citation_ok gate) - it only logs a
    warning listing any bibliography entry that doesn't match the document
    name of a chunk that was ACTUALLY retrieved for this question, so it
    shows up in server logs / Langfuse traces for review rather than
    silently. Heuristic (regex-based section/line extraction over free-form
    model prose, not a structured parse) - false positives are possible for
    unusually-formatted answers and are not treated as an error, only a
    signal worth a human glance."""
    if not doc_details:
        return {"checked": False, "unverified_sources": []}

    heading_match = _REFERRED_SOURCES_HEADING_RE.search(answer_text)
    if not heading_match:
        return {"checked": False, "unverified_sources": []}
    section_text = answer_text[heading_match.end():]

    known_names = {
        str(detail.get("document_name")).lower()
        for detail in doc_details.values()
        if detail and detail.get("document_name")
    }
    if not known_names:
        return {"checked": False, "unverified_sources": []}

    cited = {m.group(1).strip() for m in _CITATION_LINE_RE.finditer(section_text)}
    unverified = sorted(
        c for c in cited
        if not any(c.lower() in known or known in c.lower() for known in known_names)
    )
    if unverified:
        logger.warning(
            f"[citation_grounding] 'Referred Sources' lists {len(unverified)} source(s) not matching any "
            f"actually-retrieved document - possible hallucinated bibliography entries: {unverified[:10]}"
        )
    return {"checked": True, "unverified_sources": unverified}
