"""
2026-08-26: NEW FILE. Added to support the "Tax Knowledge & Analysis
Framework" requirement: every /api/ask response must be organized under six
fixed categories (Fact Sheet, Law & Regulatory Framework, Case Law Hierarchy,
Opinion & Expert Views, Client Specific Data, Final Output & Impact Analysis),
and retrieval must actually TRY every category on every question - graph
first, falling back to a vector search over the chunks when the graph has
nothing typed for that category - rather than only ever using the single
general-purpose hybrid retrieval graphrag.GraphRAGRetriever.retrieve() already
runs. Why: that general retrieval is optimized to find the best overall
answer to a question, not to guarantee coverage of each of these six
practice-standard categories individually - a question that happens to match
well on case law would previously starve out any weaker-matching statutory
or opinion content instead of the two being reported side by side.

Domain-agnostic scaffolding (same split as nl2cypher/answer_evaluation.py):
the MECHANICS here (keyword extraction, graph keyword search, vector
fallback, entity-label filtering) are generic to any GraphRAGRetriever-backed
domain pack. The actual six categories - which node labels and text
properties belong to which category, in what order, worded how - are
supplied by the domain pack (see domain_packs/tax/config.py's
TAX_CATEGORY_SPECS) as a list of CategorySpec, exactly how nl2cypher's prompt
text and answer_evaluation's system_prompt are domain-supplied rather than
hardcoded here.
"""
import logging
import re
from dataclasses import dataclass, field

from neo4j import Query

from framework.core.infra import neo4j_client

logger = logging.getLogger("category_retrieval")

# Only used to keep the graph keyword-match pass (below) from matching on
# filler words common to every question regardless of topic - NOT a
# linguistic authority. Precision for paraphrased/conceptual questions comes
# from the vector-search fallback (semantic), not this list.
_STOPWORDS = frozenset({
    "what", "which", "when", "where", "does", "did", "do", "is", "are", "was",
    "were", "the", "a", "an", "of", "for", "to", "in", "on", "under", "with",
    "and", "or", "not", "it", "its", "this", "that", "these", "those", "can",
    "should", "would", "could", "will", "shall", "has", "have", "had", "be",
    "been", "being", "as", "by", "from", "about", "into", "than", "then",
    "there", "their", "any", "all", "some", "if", "so", "such", "we", "you",
    "i", "our", "your", "please", "explain", "tell", "me", "us", "kindly",
})
_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")

DEFAULT_GRAPH_LIMIT = 15
DEFAULT_VECTOR_TOP_K = 8
DEFAULT_GRAPH_TIMEOUT_SECONDS = 20


def extract_keywords(question: str, max_keywords: int = 8) -> list[str]:
    """Lightweight, domain-agnostic keyword extraction for the graph
    pre-filter pass below: lowercase word/number tokens, stopwords dropped,
    de-duplicated, order-preserved. Short alphanumeric tokens containing a
    digit are kept even under the length floor (e.g. "80c", "9(1)(vi)" ->
    "9", "1", "vi" after tokenizing - section-style references are short but
    highly meaningful); short pure-alphabetic filler words are dropped."""
    seen: list[str] = []
    for tok in _TOKEN_RE.findall(question.lower()):
        if tok in _STOPWORDS:
            continue
        if len(tok) < 4 and not any(ch.isdigit() for ch in tok):
            continue
        if tok in seen:
            continue
        seen.append(tok)
        if len(seen) >= max_keywords:
            break
    return seen


@dataclass(frozen=True)
class CategorySpec:
    """One category of the domain's Knowledge & Analysis Framework.

    node_labels / text_properties are only used for categories that ARE
    retrieved (synthesis_only=False). text_properties MUST be SCALAR string
    properties only (never list-valued ones like a `keywords` or
    `cases_cited` array) - the graph keyword search below concatenates them
    inside one Cypher `reduce()`, which errors if it ever hits a list value.

    instructions (2026-08-28: NEW, see Graph_First_Retrieval_Design.md):
    optional extra generation-time guidance appended to this category's
    block in format_category_context() - e.g. an explicit "present both
    favorable and unfavorable authorities" instruction for case law. Empty
    by default (framework mechanics stay domain-agnostic; the actual text is
    domain-pack-supplied, same split as everything else in this file).

    max_chars_per_chunk / max_chars_per_category (2026-08-28: NEW): per-
    category override of format_category_context()'s truncation caps -
    defaults match the framework's original fixed 800/2500 char caps, but a
    domain pack can raise them for a category whose chunk text is already a
    complete composed unit rather than a raw-PDF slice (see tax's
    case_law_hierarchy override and CaseLaw chunk text's construction in
    extract_case_law.py's build_chunks_and_link() - a truncated chunk here
    was a plausible contributor to at least one real mischaracterization,
    see Graph_First_Retrieval_Design.md).
    """
    key: str
    label: str
    description: str
    node_labels: tuple[str, ...] = ()
    text_properties: tuple[str, ...] = ()
    vector_query_prefix: str = ""
    synthesis_only: bool = False
    instructions: str = ""
    max_chars_per_chunk: int = 800
    max_chars_per_category: int = 2500


@dataclass
class CategoryResult:
    key: str
    label: str
    description: str
    status: str  # "graph" | "vector_fallback" | "not_found" | "synthesis"
    chunks: list = field(default_factory=list)
    note: str = ""
    # Copied from the originating CategorySpec at construction time (see
    # _result_from_spec() below) so format_category_context() can render
    # per-category instructions/truncation without needing the spec list
    # passed back in separately.
    instructions: str = ""
    max_chars_per_chunk: int = 800
    max_chars_per_category: int = 2500


def _result_from_spec(spec: "CategorySpec", status: str, chunks: list | None = None, note: str = "") -> CategoryResult:
    """Constructs a CategoryResult carrying spec.instructions/max_chars_*
    forward - use this instead of calling CategoryResult(...) directly so
    every construction site stays in sync automatically."""
    return CategoryResult(
        spec.key, spec.label, spec.description, status=status, chunks=chunks or [], note=note,
        instructions=spec.instructions, max_chars_per_chunk=spec.max_chars_per_chunk,
        max_chars_per_category=spec.max_chars_per_category,
    )


def _graph_keyword_search(spec: CategorySpec, keywords: list[str], limit: int,
                           timeout_seconds: int) -> list[str]:
    """Returns the Neo4j element_ids of nodes labeled one of
    spec.node_labels whose concatenated text_properties contain ANY of the
    extracted keywords (case-insensitive). Deterministic, no LLM call - a
    coarse pre-filter, not the precision mechanism (the reranker/vector path
    elsewhere in this pipeline already carries semantic precision); this
    only decides whether the GRAPH has anything worth using before falling
    back to vector search."""
    if not spec.node_labels or not spec.text_properties or not keywords:
        return []
    # 2026-08-26: single unconstrained MATCH (n), filtered by label overlap
    # THEN keyword match, rather than `UNWIND $labels AS lbl MATCH (n) WHERE
    # lbl IN labels(n) ...` - the UNWIND form re-scans every node once PER
    # label in $labels (up to ~39 for the Law & Regulatory Framework
    # category), while this form does exactly one full-node pass regardless
    # of how many labels are in the category. Neither form can use a label
    # index (there's no single parameterized label in the MATCH pattern -
    # standard Cypher, pre-5.x dynamic labels, has no clean way to express
    # "match any of these N labels" other than a post-MATCH filter), so
    # minimizing the number of passes is what actually matters here.
    cypher = (
        "MATCH (n) "
        "WHERE any(lbl IN $labels WHERE lbl IN labels(n)) "
        "AND any(kw IN $keywords WHERE "
        "  toLower(reduce(acc = '', p IN $text_props | acc + ' ' + coalesce(n[p], ''))) CONTAINS kw) "
        "RETURN DISTINCT elementId(n) AS db_id "
        "LIMIT $limit"
    )
    try:
        with neo4j_client.get_session() as session:
            records = list(session.run(
                Query(cypher, timeout=timeout_seconds),
                labels=list(spec.node_labels), text_props=list(spec.text_properties),
                keywords=keywords, limit=limit,
            ))
    except Exception as e:
        logger.warning(f"[category_retrieval] graph keyword search failed for category '{spec.key}': {e}")
        return []
    return [r["db_id"] for r in records]


def run_category_retrieval(question: str, retriever, specs: list, reranker_module=None,
                            graph_limit: int = DEFAULT_GRAPH_LIMIT,
                            vector_top_k: int = DEFAULT_VECTOR_TOP_K,
                            graph_timeout_seconds: int = DEFAULT_GRAPH_TIMEOUT_SECONDS,
                            retrieval_mode: str = "hybrid",
                            semantic_graph_trace: dict | None = None) -> list:
    """Runs graph-first, vector-fallback retrieval for every non-synthesis
    category in `specs`, using the SAME GraphRAGRetriever instance the main
    /api/ask pipeline already constructed (reuses its chunk-traversal/vector-
    search/session plumbing rather than duplicating it). `reranker_module` is
    optional (pass framework.core.retrieve.reranker) - when given, vector
    fallback candidates are scored through the SAME calibrated relevance bar
    (reranker.DEFAULT_MIN_RELEVANCE / fallback tier) the rest of the app
    uses, instead of inventing a separate raw-cosine threshold; when
    omitted, entity-label-filtered vector candidates are kept as-is.

    `retrieval_mode="vector_only"` (2026-08-26: NEW): skips the graph
    keyword-search pass for every category entirely - every category falls
    straight to its vector fallback, same as graphrag.GraphRAGRetriever's own
    "vector_only" mode. Kept in sync with that flag so the Ask screen's
    "Simple RAG" mode stays graph-free end-to-end, not just for the main
    answer's retrieval.

    `semantic_graph_trace` (2026-08-29: NEW, optional): passed straight
    through to every `retriever.semantic_entry_point_search()` call this
    function makes under retrieval_mode="semantic_graph" - see that method's
    docstring in graphrag.py. Pass the SAME dict the main
    `retriever.retrieve()` call used so ALL files touched across the whole
    /api/ask call (main pass + every category's fallback) land in one
    combined list for the UI's evidence trail. `None` (the default) skips
    this bookkeeping.

    Returns a list[CategoryResult], one per spec, in the given order."""
    keywords = extract_keywords(question)
    vector_only = retrieval_mode == "vector_only"
    results: list[CategoryResult] = []

    for spec in specs:
        if spec.synthesis_only:
            results.append(_result_from_spec(spec, status="synthesis"))
            continue

        # 2026-08-28: NEW. Stance-balanced entry point for case_law_hierarchy
        # specifically (see stance_balanced_case_law.py and
        # Graph_First_Retrieval_Design.md) - explicit "for vs. against" split
        # instead of the generic single-query graph/vector path below. Falls
        # through to that generic path unchanged (same fallback-tier
        # guarantee every other category already has) if the specialized
        # path is thin or errors, so this can never make case_law_hierarchy
        # WORSE than before, only better.
        if spec.key == "case_law_hierarchy" and not vector_only:
            try:
                specialized = _case_law_hierarchy_retrieval(
                    question, retriever, spec, keywords, graph_timeout_seconds)
            except Exception as e:
                logger.warning(f"[category_retrieval] stance-balanced case-law retrieval failed, "
                                f"falling back to generic path: {e}")
                specialized = None
            if specialized is not None:
                results.append(specialized)
                continue

        # --- graph pass (skipped entirely in vector_only mode) ----------
        matched_db_ids = [] if vector_only else _graph_keyword_search(
            spec, keywords, graph_limit, graph_timeout_seconds)
        if matched_db_ids:
            chunk_ids = retriever.get_chunk_ids_via_graph_traversal(matched_db_ids)
            graph_chunks = retriever.get_chunks_by_ids(chunk_ids)
            if graph_chunks:
                results.append(_result_from_spec(
                    spec, status="graph",
                    chunks=[{"chunk_id": c["chunk_id"], "text": c["text"]} for c in graph_chunks[:10]],
                ))
                continue
            # Matched entities exist but none had a linked chunk (e.g. a
            # container-only node) - fall through to vector search below
            # rather than reporting a false "not found".

        # --- vector fallback ---------------------------------------------
        # 2026-08-28: "semantic_graph" mode (see Semantic_Graph_Retrieval_
        # Architecture.md) routes through the same per-node entry-point +
        # md-hydration search graphrag.py's retrieve() uses, instead of the
        # old chunk-embedding ANN search - same {"chunk_id","text",
        # "similarity"} shape, so nothing below this line needs to change.
        query_text = f"{spec.vector_query_prefix}{question}" if spec.vector_query_prefix else question
        if retrieval_mode == "semantic_graph":
            try:
                candidates = retriever.semantic_entry_point_search(
                    query_text, top_k=vector_top_k, semantic_graph_trace=semantic_graph_trace)
            except Exception as e:
                logger.warning(f"[category_retrieval] semantic_entry_point_search failed for "
                                f"category '{spec.key}': {e}")
                candidates = []
        else:
            candidates = retriever.vector_search_chunks(query_text, top_k=vector_top_k)
        if not candidates:
            results.append(_result_from_spec(
                spec, status="not_found",
                note="No matching records found in the knowledge graph or source documents for this category.",
            ))
            continue

        # Keep only candidates whose OWNING ENTITY actually carries one of
        # this category's node_labels - otherwise every category's vector
        # fallback would just return the same generic nearest-neighbour
        # chunks regardless of category (e.g. a "Case Law Hierarchy" search
        # surfacing an Income Tax Act clause just because it was the
        # closest embedding), which defeats the point of having categories.
        entity_labels = retriever.get_entity_labels_for_chunks([c["chunk_id"] for c in candidates])
        wanted = set(spec.node_labels)
        filtered = [c for c in candidates if wanted & set(entity_labels.get(str(c["chunk_id"]), []))]
        if not filtered:
            results.append(_result_from_spec(
                spec, status="not_found",
                note="No matching records found in the knowledge graph or source documents for this category.",
            ))
            continue

        if reranker_module is not None:
            filtered = reranker_module.rerank(question, filtered, top_n=6)
        if not filtered:
            results.append(_result_from_spec(
                spec, status="not_found",
                note="The nearest available records for this category did not clear the relevance bar.",
            ))
            continue

        results.append(_result_from_spec(
            spec, status="vector_fallback",
            chunks=[{"chunk_id": c["chunk_id"], "text": c["text"]} for c in filtered],
        ))

    return results


def _case_law_hierarchy_retrieval(question: str, retriever, spec: CategorySpec, keywords: list[str],
                                   graph_timeout_seconds: int, min_per_stance: int = 2,
                                   top_n_per_stance: int = 8) -> CategoryResult | None:
    """Stance-balanced retrieval for the case_law_hierarchy category. See
    Graph_First_Retrieval_Design.md Section 3.

    Two things happen, BOTH deterministic/explicit rather than
    similarity-hoped-for:
      1. Topical entry points via the CaseLaw-level metadata vector index
         (case_law_entry_points.py) - "what's relevant to THIS question".
      2. An explicit stance-split Cypher pull (stance='taxpayer_favorable'
         vs 'revenue_favorable'), keyword-scoped to the same question - "make
         sure BOTH sides are represented", not left to embedding similarity
         alone to decide.

    Returns None (caller falls through to the generic graph/vector-fallback
    path) when this comes back thin for BOTH stances - the fallback-tier
    guarantee described in Graph_First_Retrieval_Design.md Section 3."""
    from framework.core.retrieve import case_law_entry_points as clep

    try:
        entry_points = clep.search_case_law_entry_points(question, top_k=20)
    except Exception as e:
        logger.warning(f"[category_retrieval] case-law entry-point vector search failed: {e}")
        entry_points = []

    stance_hits: dict[str, list[dict]] = {"taxpayer_favorable": [], "revenue_favorable": []}
    if keywords:
        cypher = (
            "MATCH (n:CaseLaw) "
            "WHERE n.stance = $stance AND any(kw IN $keywords WHERE "
            "  toLower(coalesce(n.retrievalSummary, '') + ' ' + coalesce(n.retrievalKeywords, '')) CONTAINS kw) "
            "RETURN elementId(n) AS db_id, n.case_name AS case_name "
            "LIMIT $limit"
        )
        try:
            with neo4j_client.get_session() as session:
                for stance in ("taxpayer_favorable", "revenue_favorable"):
                    rows = list(session.run(
                        Query(cypher, timeout=graph_timeout_seconds),
                        stance=stance, keywords=keywords, limit=top_n_per_stance,
                    ))
                    stance_hits[stance] = [dict(r) for r in rows]
        except Exception as e:
            logger.warning(f"[category_retrieval] stance-split pull failed: {e}")

    n_favorable = len(stance_hits["taxpayer_favorable"])
    n_revenue = len(stance_hits["revenue_favorable"])
    matched_db_ids = list({r["db_id"] for r in entry_points}
                          | {r["db_id"] for r in stance_hits["taxpayer_favorable"]}
                          | {r["db_id"] for r in stance_hits["revenue_favorable"]})

    if not matched_db_ids or (n_favorable < min_per_stance and n_revenue < min_per_stance):
        # Too thin on EITHER side (or nothing at all) - let the generic
        # graph/vector-fallback path have a shot instead. Not necessarily a
        # failure: some questions genuinely only have authorities on one
        # side, but we don't trust THIS path's judgment on that when it
        # found next to nothing - the generic path's broader vector search
        # is the safer bet in that situation.
        return None

    chunk_ids = retriever.get_chunk_ids_via_graph_traversal(matched_db_ids)
    graph_chunks = retriever.get_chunks_by_ids(chunk_ids)
    if not graph_chunks:
        return None

    note = (f"Stance-balanced retrieval: {n_favorable} taxpayer-favorable, {n_revenue} revenue-favorable "
            f"case(s) matched on topic (plus {len(entry_points)} topical entry point(s) via the metadata "
            f"vector index). If one side still looks thin below, that reflects the ingested corpus, not a "
            f"retrieval cap.")
    return _result_from_spec(
        spec, status="graph",
        chunks=[{"chunk_id": c["chunk_id"], "text": c["text"]} for c in graph_chunks[:20]],
        note=note,
    )


def format_category_context(results: list, max_chars_per_category: int = 2500) -> str:
    """Renders category coverage as a context block for the answer_generation
    LLM call - what actually exists (or doesn't) per category, so the model
    can honestly structure its response under the six headings instead of
    guessing or fabricating content for a category nothing was found for.

    2026-08-28: per-result instructions/max_chars_per_chunk/
    max_chars_per_category (copied from the originating CategorySpec, see
    _result_from_spec()) now override this function's own defaults when
    set - lets a domain pack raise the truncation ceiling or append
    generation-time guidance for one specific category (e.g. tax's
    case_law_hierarchy) without changing every other category's behavior.
    The `max_chars_per_category` PARAMETER here is now only the fallback
    used when a result carries no override (e.g. results built before this
    field existed, or a bare CategoryResult built outside _result_from_spec)."""
    lines = [
        "TAX KNOWLEDGE & ANALYSIS FRAMEWORK - CATEGORY COVERAGE",
        "(Evidence retrieved for each category below. Structure your response using these exact "
        "category headings. For any category marked NOT FOUND, say so plainly - do not imply a "
        "source exists if it is not listed here.)",
    ]
    for r in results:
        lines.append(f"\n### {r.label}")
        lines.append(f"({r.description})")
        if r.instructions:
            lines.append(f"INSTRUCTIONS FOR THIS CATEGORY: {r.instructions}")
        if r.status == "synthesis":
            lines.append("STATUS: SYNTHESIS - not a retrieval category. Derive this section from the "
                          "question itself and/or your analysis of the other categories above, per your "
                          "system prompt's instructions.")
        elif r.status == "not_found":
            lines.append(f"STATUS: NOT FOUND - {r.note}")
        else:
            tier = "graph match" if r.status == "graph" else \
                "vector-search fallback match (the graph had nothing typed for this category)"
            status_line = f"STATUS: FOUND ({tier})"
            if r.note:
                status_line += f" - {r.note}"
            lines.append(status_line)
            per_chunk_cap = r.max_chars_per_chunk or 800
            category_cap = r.max_chars_per_category or max_chars_per_category
            blob = "\n\n".join(f"[chunk_id={c['chunk_id']}]\n{c['text'][:per_chunk_cap]}" for c in r.chunks)
            if len(blob) > category_cap:
                blob = blob[:category_cap] + "\n... (truncated)"
            lines.append(blob)
    return "\n".join(lines)


def summarize_category_coverage(results: list) -> list:
    """Lightweight, UI-safe summary (status + chunk counts, no chunk text) -
    for including in the /api/ask JSON response without bloating the
    payload."""
    return [{"key": r.key, "label": r.label, "status": r.status, "chunk_count": len(r.chunks)} for r in results]
