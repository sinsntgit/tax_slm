"""
CACHE + AGENT STATE STORE (Redis) - PROMOTED unchanged from
tax_kg_poc/webapp/cache.py. Response cache + short-lived per-request agent
state, both fail-safe if Redis isn't running. 100% domain-agnostic - keyed
purely by question text + arbitrary filters dict, no tax concepts.
"""
import hashlib
import json
import logging
import os
import re
import uuid

import redis
from dotenv import load_dotenv

from framework.core.infra.dotenv_path import env_path as _env_path

# Anchored to the repo root by file location, not process cwd - see
# framework.core.infra.dotenv_path's docstring.
load_dotenv(_env_path())
logger = logging.getLogger("cache_and_state")

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
CACHE_TTL_SECONDS = int(os.environ.get("CACHE_TTL_SECONDS", "3600"))

AGENT_STATE_TTL_SECONDS = 600

try:
    _redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True, socket_connect_timeout=1)
    _redis_client.ping()
    REDIS_AVAILABLE = True
    logger.info(f"[cache] Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
except Exception as e:
    REDIS_AVAILABLE = False
    logger.warning(f"[cache] Redis not available ({e}) - running WITHOUT caching or agent state tracking.")


def _normalize_question_for_cache(question: str) -> str:
    """2026-09-05: NEW - per explicit report that asking "the same question"
    with trivial punctuation/whitespace variation ("What is Section 9?" vs
    "What is Section 9" vs "what is section 9.") produced visibly different
    answers. Root cause: the previous key_material only did
    question.strip().lower() - it never touched trailing punctuation or
    internal whitespace, so each of those three variants hashed to a
    DIFFERENT cache key and each independently re-ran the full pipeline
    (intent detection -> retrieval -> rerank -> generation), each subject to
    its own sampling/ANN-search variance (see the temperature default change
    and the reranker tie-breaker change alongside this one - this
    normalization only fixes the caching half of the inconsistency; it does
    nothing for two GENUINELY different question strings, nor for the first
    time a given normalized question is ever asked).

    Normalization is deliberately conservative - it must never make two
    MEANINGFULLY different questions collide:
      - Collapses any run of whitespace to a single space (mirrors how a
        question renders/reads regardless of stray double-spaces).
      - Strips ONE trailing run of sentence-ending punctuation (?!.) - not
        punctuation anywhere else in the string, so "Section 9(1)?" keeps
        its internal parenthesis/digit untouched and only the trailing "?"
        is dropped.
      - Does NOT strip a trailing "!" and "?" together as meaningfully
        different (both map to the same normalized form) - a legal-research
        question's trailing terminator carries no semantic weight, unlike
        casing (already lowercased before this call was even added) or
        internal wording.
    Case-folding to lowercase is already handled by the caller
    (question.strip().lower() below) - this function only fixes the
    punctuation/whitespace gap, kept as a separate function so its narrow
    scope is testable in isolation."""
    normalized = re.sub(r"\s+", " ", question).strip()
    normalized = re.sub(r"[?!.]+$", "", normalized).strip()
    return normalized


def _cache_key(question: str, metadata_filters: dict | None = None, extra_key: str = "") -> str:
    # 2026-08-26: added extra_key - folded into the hash so answers computed
    # under different pipeline VARIANTS for the identical question/filters
    # (e.g. retrieval_mode="hybrid" vs "vector_only", see graphrag.py) get
    # separate cache entries instead of one silently overwriting/serving the
    # other. Empty string by default - existing callers that don't pass it
    # produce the exact same key as before, so this is fully backward
    # compatible with whatever is already cached.
    # 2026-09-05: question text now runs through _normalize_question_for_cache()
    # (whitespace/trailing-punctuation normalization) BEFORE the existing
    # .strip().lower() - see that function's docstring for why. This changes
    # the resulting hash relative to before (an existing cached entry for a
    # not-yet-normalized key will simply MISS once and get recomputed under
    # the new key - a one-time cache-miss cost, not a correctness issue).
    normalized_question = _normalize_question_for_cache(question).lower()
    key_material = normalized_question + "|" + json.dumps(metadata_filters or {}, sort_keys=True) + "|" + extra_key
    digest = hashlib.sha256(key_material.encode("utf-8")).hexdigest()
    return f"qa_cache:{digest}"


def get_cached_answer(question: str, metadata_filters: dict | None = None, extra_key: str = "") -> dict | None:
    if not REDIS_AVAILABLE:
        return None
    try:
        raw = _redis_client.get(_cache_key(question, metadata_filters, extra_key))
        if raw:
            logger.info(f"[cache] HIT for question: {question[:60]!r}")
            return json.loads(raw)
        logger.info(f"[cache] MISS for question: {question[:60]!r}")
        return None
    except Exception as e:
        logger.warning(f"[cache] read failed: {e}")
        return None


def set_cached_answer(question: str, response: dict, metadata_filters: dict | None = None, extra_key: str = ""):
    if not REDIS_AVAILABLE:
        return
    try:
        _redis_client.setex(_cache_key(question, metadata_filters, extra_key), CACHE_TTL_SECONDS, json.dumps(response))
    except Exception as e:
        logger.warning(f"[cache] write failed: {e}")


def start_agent_run() -> str:
    run_id = str(uuid.uuid4())
    _write_state(run_id, {"status": "started", "steps": []})
    return run_id


def update_agent_state(run_id: str, step_name: str, data: dict):
    if not REDIS_AVAILABLE:
        return
    try:
        state = _read_state(run_id) or {"status": "started", "steps": []}
        state["steps"].append({"step": step_name, "data": data})
        _write_state(run_id, state)
    except Exception as e:
        logger.warning(f"[agent_state] update failed for run {run_id}: {e}")


def finish_agent_run(run_id: str):
    if not REDIS_AVAILABLE:
        return
    try:
        state = _read_state(run_id) or {}
        state["status"] = "finished"
        _write_state(run_id, state)
    except Exception as e:
        logger.warning(f"[agent_state] finish failed for run {run_id}: {e}")


def get_agent_state(run_id: str) -> dict | None:
    return _read_state(run_id)


def _state_key(run_id: str) -> str:
    return f"agent_state:{run_id}"


def _write_state(run_id: str, state: dict):
    if not REDIS_AVAILABLE:
        return
    _redis_client.setex(_state_key(run_id), AGENT_STATE_TTL_SECONDS, json.dumps(state))


def clear_all_cache() -> int:
    """Deletes every response-cache and agent-state key this module owns
    (the `qa_cache:*` / `agent_state:*` prefixes) - used by the "Clear All
    Data" flow so stale cached answers from before a graph reset can never
    be served again. Deliberately scoped to these two prefixes rather than
    `FLUSHDB`, since Redis may be shared with other uses in a real
    deployment. Returns the number of keys deleted; 0 (not an error) if
    Redis isn't available."""
    if not REDIS_AVAILABLE:
        return 0
    deleted = 0
    try:
        for pattern in ("qa_cache:*", "agent_state:*"):
            keys = list(_redis_client.scan_iter(match=pattern, count=500))
            if keys:
                deleted += _redis_client.delete(*keys)
        logger.info(f"[cache] cleared {deleted} cache/agent-state key(s)")
    except Exception as e:
        logger.warning(f"[cache] clear_all_cache failed: {e}")
    return deleted


def _read_state(run_id: str) -> dict | None:
    if not REDIS_AVAILABLE:
        return None
    try:
        raw = _redis_client.get(_state_key(run_id))
        return json.loads(raw) if raw else None
    except Exception as e:
        logger.warning(f"[agent_state] read failed for run {run_id}: {e}")
        return None
