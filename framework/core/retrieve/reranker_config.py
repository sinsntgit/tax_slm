"""
User-editable configuration for reranker.py's relevance filtering - 2026-09-03:
NEW, per explicit report "the answers being generated from plain RAG... is
much better than what we are generating."

ROOT CAUSE THIS ADDRESSES: reranker.py's DEFAULT_MIN_RELEVANCE (0.55) was
empirically calibrated against the bge-reranker-base model's score
distribution (see that constant's own docstring - a documented incident
where 0.85 was too strict). The project has since switched the actual
reranker model to bge-reranker-v2-m3 (a different cross-encoder with a
different score calibration) via RERANKER_MODEL_NAME, but the 0.55/0.35
thresholds were never re-tuned for it - a plausible, concrete cause of
under-recall (a genuinely relevant chunk scoring just under the new model's
0.55 gets silently dropped before the LLM ever sees it) that a plain-RAG
comparison, which has no such cutoff, would never hit.

Rather than guess a new fixed threshold without a labeled test set to tune
it against, this makes min_relevance/fallback_relevance/top_n a Settings-
page control (same hot-reload-from-disk pattern as response_config.py /
comparative_response.py) - re-read from disk on every rerank() call, so a
Settings-page edit takes effect on the very next question, no restart
needed, and can be tuned empirically against real answers instead of
theoretically against an unlabeled corpus.
"""
import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "reranker_config.json"

# Defaults: min_relevance nudged down from reranker.py's old hardcoded 0.55
# as a reasonable starting point for bge-reranker-v2-m3 (a stronger, more
# discriminating model than bge-reranker-base tends to spread scores wider,
# so a slightly lower absolute bar is a safer default pending real tuning)
# - NOT a substitute for actually watching real answers and adjusting via
# Settings, which is the whole point of this being configurable at all.
DEFAULT_CONFIG = {
    "min_relevance": 0.45,
    "fallback_relevance": 0.30,
    "fallback_max_chunks": 5,
    "top_n": 20,
}


def _config_path() -> Path:
    override = os.environ.get("RERANKER_CONFIG_PATH", "").strip()
    return Path(override) if override else DEFAULT_CONFIG_PATH


def load_config() -> dict:
    """Re-reads the config file from disk every call (cheap - it's a small
    JSON file). Falls back to DEFAULT_CONFIG - whole file missing, invalid
    JSON, or not a dict - so a bad hand-edit can never break /api/ask, it
    just silently reverts to the default for that request. Same pattern as
    response_config.load_response_config()."""
    path = _config_path()
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            return dict(DEFAULT_CONFIG)
    except (OSError, ValueError):
        return dict(DEFAULT_CONFIG)
    merged = dict(DEFAULT_CONFIG)
    merged.update({k: v for k, v in cfg.items() if k in DEFAULT_CONFIG})
    return merged


def save_config(cfg: dict) -> dict:
    """Validates + writes cfg to disk. Returns the normalized dict actually
    written (unknown keys dropped, out-of-range/wrong-type values fall back
    to the default for that one key rather than rejecting the whole save)."""
    normalized = dict(DEFAULT_CONFIG)

    def _clamped_float(key: str, lo: float, hi: float):
        val = cfg.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and lo <= float(val) <= hi:
            normalized[key] = float(val)

    def _clamped_int(key: str, lo: int, hi: int):
        val = cfg.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and lo <= int(val) <= hi:
            normalized[key] = int(val)

    _clamped_float("min_relevance", 0.0, 1.0)
    _clamped_float("fallback_relevance", 0.0, 1.0)
    _clamped_int("fallback_max_chunks", 1, 50)
    _clamped_int("top_n", 1, 100)

    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
    return normalized
