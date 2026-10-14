"""
BGE RERANKER - PROMOTED unchanged from tax_kg_poc/webapp/reranker.py. Scores
(question, chunk) pairs with a cross-encoder so graph-linked and
vector-similar candidates (not directly comparable on their own scores) get
one unified relevance ranking. 100% domain-agnostic already - no tax
concepts anywhere in it.
"""
import logging
import math
import os

from dotenv import load_dotenv
from sentence_transformers import CrossEncoder

from framework.core.infra.dotenv_path import env_path as _env_path
from framework.core.infra import step_logger
from framework.core.retrieve import reranker_config

# Anchored to the repo root by file location, not process cwd - see
# framework.core.infra.dotenv_path's docstring.
load_dotenv(_env_path())
logger = logging.getLogger("bge_reranker")

RERANKER_MODEL_NAME = os.environ.get("RERANKER_MODEL_NAME", "BAAI/bge-reranker-base")

# Cross-encoder compute scales with input sequence length, and nothing was
# capping chunk text before scoring (unlike app.py's LLM context builder,
# which already truncates each chunk to 1500 chars) - a real request traced
# via Langfuse showed 25-30 candidate pairs taking 90-200+ SECONDS to score,
# an order of magnitude slower than that pair count should need. The LLM
# never sees more than 1500 chars of a chunk anyway (see app.py's
# _build_context_parts), so scoring more than that buys no real relevance
# signal - only slower requests. Matches that same cap for consistency.
RERANK_TEXT_MAX_CHARS = int(os.environ.get("RERANK_TEXT_MAX_CHARS", "1500"))

# BAAI/bge-reranker-base is a single-logit cross-encoder - CrossEncoder.predict()
# returns the RAW logit (roughly -10..+10, unbounded), not a 0-1 relevance
# probability. A "relevance must be > 85%" requirement is meaningless against
# a raw logit, so every score is passed through a sigmoid here to get an
# actual 0-1 probability-like relevance score BEFORE any threshold is applied.
def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


# Minimum relevance (post-sigmoid, 0-1) a chunk must clear to be handed to the
# LLM for answer generation at all.
#
# RECALIBRATED 0.85 -> 0.55 after a real, reproducible failure: the question
# "What did the tribunal hold on non-qualifying demergers?" has genuinely
# relevant chunks sitting right in the case-law corpus (the Tribunal's
# findings/reasoning text, e.g. "the Treasury Undertaking was not
# demonstrated as a genuine business undertaking...") - ChatGPT, given the
# same source PDF, produced a detailed grounded answer from that text. Our
# own reranker scored those chunks below 0.85 (a CONCEPTUAL question's
# wording is a paraphrase of the reasoning, not a near-verbatim match, and
# bge-reranker-base - a general-purpose, NOT tax/legal-fine-tuned
# cross-encoder - systematically under-scores paraphrased-vs-verbatim pairs
# even when the content is exactly what's needed). Result: EVERY finding
# chunk got filtered out, the answer fell back to only the CaseLaw node's
# bare structured properties (statute list, outcome), and the UI showed
# "Not grounded - confidence 0%" for a question the source document actually
# answers well. 0.85 is too strict for this reranker/corpus pairing in
# practice - see also the fallback bar below, a second safety net for
# exactly this situation.
#
# 2026-09-03: CHANGED - was a frozen env-configurable constant. The project
# has since switched RERANKER_MODEL_NAME to bge-reranker-v2-m3 (a different
# cross-encoder with a different score calibration than bge-reranker-base,
# which 0.55 was tuned against) WITHOUT re-tuning this threshold - a
# plausible, concrete cause of a "plain RAG gives better answers" report,
# since a plain top-k comparison has no such cutoff to lose a marginal-but-
# relevant chunk to. Now sourced from reranker_config.load_config() (a
# Settings-page control, hot-reloaded from disk every call - see that
# module's docstring) instead, so it can be tuned empirically against real
# answers without a redeploy. These two module-level constants are kept
# only as the fallback reranker_config.py itself falls back to if its config
# file is missing/corrupt - not read directly by rerank() below anymore.
DEFAULT_MIN_RELEVANCE = reranker_config.DEFAULT_CONFIG["min_relevance"]
DEFAULT_FALLBACK_RELEVANCE = reranker_config.DEFAULT_CONFIG["fallback_relevance"]
FALLBACK_MAX_CHUNKS = reranker_config.DEFAULT_CONFIG["fallback_max_chunks"]

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        logger.info(f"[bge_reranker] Loading reranker model '{RERANKER_MODEL_NAME}'...")
        _reranker = CrossEncoder(RERANKER_MODEL_NAME)
    return _reranker


def rerank(question: str, candidate_chunks: list[dict], top_n: int | None = None,
           min_relevance: float | None = None, fallback_relevance: float | None = None,
           fallback_max_chunks: int | None = None, skip_rerank: bool = False) -> list[dict]:
    """Scores every candidate chunk against the question and returns the
    top_n AMONG those clearing `min_relevance`, sorted by relevance (highest
    first). Each chunk gets "rerank_score" (0-1, sigmoid-normalized - see
    _sigmoid()) and "relevance_tier" ("primary").

    2026-09-03: any of top_n/min_relevance/fallback_relevance/
    fallback_max_chunks left as None (the default) is now sourced from
    reranker_config.load_config() - a Settings-page control, re-read from
    disk on every call - instead of a frozen constant, so it can be tuned
    empirically without a redeploy (see reranker_config.py's docstring for
    why this became necessary). Pass an explicit value to override the
    configured one for a specific call - app.py's internal quality-gate
    retry path does exactly this (deliberately wider/looser than whatever
    baseline is configured, regardless of what that baseline is).

    If NOTHING clears the primary bar, falls back to chunks clearing the
    lower `fallback_relevance` bar, capped at `fallback_max_chunks` - tagged
    relevance_tier="fallback" so the caller/UI can show reduced confidence
    honestly rather than treating it identically to a high-confidence match.
    This exists because a strict single threshold reproducibly rejected
    genuinely on-topic chunks for paraphrased/conceptual questions (see
    DEFAULT_MIN_RELEVANCE's docstring) - the fallback tier is a middle
    ground between "answer off anything" and "refuse to answer whenever the
    top score isn't excellent".

    If NOTHING clears even the fallback bar, this returns an EMPTY list -
    grounding.py's existing hallucination guard then correctly reports "no
    grounded info" rather than answering off a chunk that scored, say, 12%
    relevant.

    Re-scoring is skipped if EVERY chunk in `candidate_chunks` already
    carries a "rerank_score" (i.e. this exact list was already scored by an
    earlier call in the same request) - app.py's quality-judge retry path
    calls rerank() a second time on the IDENTICAL candidate_chunks list,
    just with a wider top_n/min_relevance to admit more of the
    already-computed scores. Same question, same chunk text -> the score
    can only come out identical, so re-running the cross-encoder a second
    time was pure duplicate work - confirmed via Langfuse trace timing to
    cost another 90-200+ seconds on top of the primary pass, for a result
    that was already sitting on each chunk dict. Passing a genuinely
    different `candidate_chunks` list (as the retry path does when it's
    actually a different/wider set) re-scores normally.

    `skip_rerank` (2026-09-07: NEW) - per explicit request to make the BGE
    cross-encoder pass optional for latency-sensitive callers (a real
    Langfuse-measured cost of 90-200+ seconds for 25-30 pairs - see above -
    is the single largest contributor to slow responses). When True, the
    cross-encoder is never invoked: candidates are ordered by whatever
    signal they already carry (vector `similarity` descending, when
    present; graph/citation-sourced candidates with similarity=None are
    kept ahead of vector candidates, mirroring the natural
    graph-then-vector assembly order retrieve() already produces) and
    truncated to `top_n`. Every returned chunk gets `rerank_score: None` and
    `relevance_tier: "unranked"` - deliberately distinct from "primary"/
    "fallback" - so the UI/grounding.py can show honestly that these chunks
    were NOT relevance-filtered, rather than implying a cross-encoder score
    that was never computed. There is no relevance FLOOR in this path (no
    min_relevance/fallback_relevance applied) - skipping the reranker means
    skipping its filtering job entirely, not just its scoring latency; a
    caller that wants speed AND a relevance floor should keep skip_rerank
    False and instead lower top_n/loosen min_relevance via reranker_config.
    """
    if not candidate_chunks:
        return []

    if skip_rerank:
        ordered = sorted(
            candidate_chunks,
            key=lambda c: (c.get("similarity") is not None, -(c.get("similarity") or 0.0)),
        )
        cfg_top_n = top_n if top_n is not None else reranker_config.load_config()["top_n"]
        top = ordered[:cfg_top_n]
        for c in top:
            c["rerank_score"] = None
            c["relevance_tier"] = "unranked"
        logger.info(f"[bge_reranker] skip_rerank=True - bypassed cross-encoder, returning top "
                    f"{len(top)} of {len(candidate_chunks)} candidate(s) by source/similarity order only.")
        return top

    # Only touch disk (load_config()) for whichever params the caller didn't
    # explicitly pin - cheap either way (small JSON file) but no reason to
    # read it at all when every param was already given explicitly.
    cfg = None
    if top_n is None or min_relevance is None or fallback_relevance is None or fallback_max_chunks is None:
        cfg = reranker_config.load_config()
    resolved_top_n = cfg["top_n"] if top_n is None else top_n
    threshold = cfg["min_relevance"] if min_relevance is None else min_relevance
    fallback_threshold = cfg["fallback_relevance"] if fallback_relevance is None else fallback_relevance
    resolved_fallback_max = cfg["fallback_max_chunks"] if fallback_max_chunks is None else fallback_max_chunks
    top_n = resolved_top_n

    if all("rerank_score" in c for c in candidate_chunks):
        step_logger.log(f"[bge_reranker] Reusing {len(candidate_chunks)} already-scored candidates "
                         f"(same list scored earlier this request) - skipping re-predict.")
    else:
        reranker = _get_reranker()
        # Cap text length before scoring - see RERANK_TEXT_MAX_CHARS above.
        pairs = [(question, (c.get("text") or "")[:RERANK_TEXT_MAX_CHARS]) for c in candidate_chunks]
        with step_logger.Step(f"bge_rerank: scoring {len(pairs)} candidate(s)"):
            raw_scores = reranker.predict(pairs)

        for chunk, raw_score in zip(candidate_chunks, raw_scores):
            chunk["rerank_score_raw"] = float(raw_score)
            chunk["rerank_score"] = _sigmoid(float(raw_score))

    # 2026-09-05: NEW - secondary tie-break key, per an explicit answer-
    # inconsistency investigation. Python's sorted() is stable, but that only
    # guarantees a consistent ORDER for a given INPUT order - if
    # candidate_chunks itself arrives in a different order across two calls
    # for the identical question (which it can: Neo4j's vector index is an
    # approximate/HNSW-based ANN search with no guaranteed row order for tied
    # similarity scores - see graphrag.py's vector_search_chunks()), two
    # rerank_score ties can still resolve to a DIFFERENT winner call-to-call,
    # changing which chunk makes the top_n cut and, downstream, the
    # generated answer - even with retrieval/generation otherwise unchanged.
    # Sorting by (rerank_score, chunk_id) makes the winner of any tie
    # deterministic regardless of input order. chunk_id may be a numeric
    # string ("42") or an arbitrary entity id string - the (is_non_numeric,
    # value) tuple sorts all-numeric ids together in numeric order, then
    # falls back to a plain string compare for non-numeric ids, mirroring
    # the exact same pattern graphrag.py's neighbor-listing sort already
    # uses. Negating the tie-break tuple's numeric branch (instead of
    # passing reverse=True, which would also flip the non-numeric/numeric
    # GROUPING itself, and reverse numeric ids into descending order) keeps
    # the tie-break direction independent of the primary reverse=True sort -
    # ties always resolve smallest-numeric-chunk_id-first, deterministically,
    # regardless of how rerank_score itself sorts.
    def _tie_break_key(c):
        cid = str(c.get("chunk_id", ""))
        return (c["rerank_score"], (0, -int(cid)) if cid.isdigit() else (1, cid))
    ranked = sorted(candidate_chunks, key=_tie_break_key, reverse=True)
    above_threshold = [c for c in ranked if c["rerank_score"] >= threshold]

    if above_threshold:
        top = above_threshold[:top_n]
        for c in top:
            c["relevance_tier"] = "primary"
        logger.info(
            f"[bge_reranker] Reranked {len(candidate_chunks)} candidates -> "
            f"{len(above_threshold)} cleared min_relevance={threshold:.2f}, returning top {len(top)} "
            f"(scores: {[round(c['rerank_score'], 3) for c in top]})"
        )
        return top

    # Nothing cleared the primary bar - try the lower fallback bar before
    # giving up entirely.
    above_fallback = [c for c in ranked if c["rerank_score"] >= fallback_threshold]
    if above_fallback:
        top = above_fallback[:min(top_n, resolved_fallback_max)]
        for c in top:
            c["relevance_tier"] = "fallback"
        logger.warning(
            f"[bge_reranker] {len(candidate_chunks)} candidates reranked, none cleared "
            f"min_relevance={threshold:.2f} (best score: {ranked[0]['rerank_score']:.3f}) - "
            f"using FALLBACK tier (>={fallback_threshold:.2f}), returning {len(top)} lower-confidence "
            f"chunk(s) (scores: {[round(c['rerank_score'], 3) for c in top]})."
        )
        return top

    logger.warning(
        f"[bge_reranker] {len(candidate_chunks)} candidates reranked, but NONE cleared even the "
        f"fallback_relevance={fallback_threshold:.2f} bar (best score: "
        f"{ranked[0]['rerank_score']:.3f}) - returning empty, answer generation will report "
        f"no grounded info rather than answer off a low-relevance chunk."
    )
    return []
