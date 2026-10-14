"""
Small JSON-file-backed settings store - currently just the Settings screen's
"Data sources" enable/disable toggles. Same hot-reload-from-disk pattern as
framework.core.retrieve.response_config (re-read on every GET, write-through
on every POST) - deliberately NOT a Neo4j :Setting node, since these toggles
are a UI/operator preference about the *webapp*, not a fact about the graph
itself, and this way they survive even a full "Delete Neo4j Data" wipe.

Judgment call (documented per the task's request): toggling a data source
here does NOT actually pause/stop any ingestion pipeline - there is no
scheduler/watcher process in this codebase for any of the five sources to
pause (Upload/Bulk-batch are the only ones with real backends at all; the
other three - Scrape URL/Paste text/Watch folder - are explicitly not-yet-
implemented, see api_ingest_not_implemented() in app.py). The toggle is
persisted honestly as a preference the UI reads back, not wired to any
enforcement, and the Settings screen says so.
"""
import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SETTINGS_PATH = _REPO_ROOT / "data" / "webapp_settings.json"

# Real, truthful data-source list derived from domain_packs/tax/pipeline.py's
# FILE_TYPES registry (the only source of truth for "what file types does
# this app's ingestion pipeline actually support") - NOT the prototype's 5
# hardcoded rows (Income-tax Act statutes / ITAT-HC-SC scraper / DTAA & CBDT
# circulars / GST-customs notifications / Tax discussion forums), most of
# which don't correspond to any real backend in this codebase. Each row here
# maps 1:1 onto a FILE_TYPES key so "enabled" has real meaning (whether this
# webapp should offer that file type for new ingestion), and the ids are
# stable regardless of which domain pack is loaded (falls back to whatever
# PIPELINE.FILE_TYPES actually exposes - see api_settings_data_sources()).
DEFAULT_DATA_SOURCE_IDS = ("act_pdf", "caselaw_pdf", "generic")

# 2026-08-27: NEW - LLM sampling temperature, settable from the Settings page
# (per explicit request) rather than only via .env, with the SAME "takes
# effect on the very next LLM call, no restart" hot-reload guarantee
# llm_client.py's model-name config already has - this file is already
# re-read from disk on every load_settings() call, so wiring llm_client's
# refresh_model_config() to read from here (see llm_client.py) gets that
# behavior for free.
# 2026-09-05: CHANGED (was None) - per an explicit answer-inconsistency
# investigation: pinning temperature=0.0 removes one of several compounding
# sources of "same question, different answer" (sampling randomness on top
# of whatever retrieval/reranking variance already exists - see
# cache.py's _normalize_question_for_cache() docstring and graphrag.py's
# GRAPH_EXPANSION note for the others). 0.0 = most deterministic sampling
# the API allows; still fully overridable from the Settings page (including
# back to "no override" by clearing it there) for anyone who wants sampling
# variety back. NOTE: this has NO effect on OpenAI's reasoning-family models
# (gpt-5-*, o-series) unless reasoning_effort="none" is ALSO set for that
# purpose (see llm_client.py's GPT_REASONING_EFFORT_<PURPOSE> env var) -
# those models reject temperature outright at any other effort level and
# this app already falls back to omitting it gracefully when rejected.
DEFAULT_TEMPERATURE = 0.0
# Clamped to 0.0-1.0 (not the up-to-2.0 some providers technically accept) -
# this is the range EVERY supported provider (Claude, GPT, Gemini) accepts
# without a 400, so a value set here is always valid regardless of which
# USE_MODEL is currently active or gets switched to later.
TEMPERATURE_MIN, TEMPERATURE_MAX = 0.0, 1.0


def _normalize_temperature(value) -> float | None:
    """None (or anything unparseable) means "no override, use the provider's
    own default" - never raises, since a stray bad value in the settings
    file shouldn't take down every LLM call in the app."""
    if value is None or value == "":
        return None
    try:
        return round(max(TEMPERATURE_MIN, min(TEMPERATURE_MAX, float(value))), 2)
    except (TypeError, ValueError):
        return None


def _settings_path() -> Path:
    override = os.environ.get("WEBAPP_SETTINGS_PATH", "").strip()
    return Path(override) if override else DEFAULT_SETTINGS_PATH


def _default_settings() -> dict:
    return {
        "data_sources": {source_id: {"enabled": True} for source_id in DEFAULT_DATA_SOURCE_IDS},
        # Langfuse tracing on/off preference - deliberately just a bool, NOT
        # the keys themselves. The actual public/secret key/host always come
        # from .env only (see framework.core.retrieve.observability) - this
        # flag lets a user turn tracing off without touching .env, and back
        # on again, but can never enable tracing on its own if .env has no
        # real keys configured (observability.refresh_langfuse_config()
        # checks both).
        "langfuse_tracing_enabled": True,
        "temperature": DEFAULT_TEMPERATURE,
    }


def load_settings() -> dict:
    path = _settings_path()
    if not path.exists():
        return _default_settings()
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return _default_settings()
    except (OSError, ValueError):
        return _default_settings()
    merged = _default_settings()
    if isinstance(raw.get("data_sources"), dict):
        for source_id, val in raw["data_sources"].items():
            merged["data_sources"][source_id] = {"enabled": bool((val or {}).get("enabled", True))}
    if "langfuse_tracing_enabled" in raw:
        merged["langfuse_tracing_enabled"] = bool(raw["langfuse_tracing_enabled"])
    if "temperature" in raw:
        merged["temperature"] = _normalize_temperature(raw["temperature"])
    return merged


def save_settings(settings: dict) -> dict:
    normalized = _default_settings()
    if isinstance(settings.get("data_sources"), dict):
        for source_id, val in settings["data_sources"].items():
            normalized["data_sources"][source_id] = {"enabled": bool((val or {}).get("enabled", True))}
    if "langfuse_tracing_enabled" in settings:
        normalized["langfuse_tracing_enabled"] = bool(settings["langfuse_tracing_enabled"])
    if "temperature" in settings:
        normalized["temperature"] = _normalize_temperature(settings["temperature"])
    path = _settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
    return normalized


def set_data_source_enabled(source_id: str, enabled: bool) -> dict:
    settings = load_settings()
    settings["data_sources"].setdefault(source_id, {"enabled": True})
    settings["data_sources"][source_id]["enabled"] = bool(enabled)
    return save_settings(settings)


def set_langfuse_tracing_enabled(enabled: bool) -> dict:
    settings = load_settings()
    settings["langfuse_tracing_enabled"] = bool(enabled)
    return save_settings(settings)


def set_temperature(value) -> dict:
    """`value=None` resets to "use provider default" (same as never having
    set it). See llm_client.refresh_model_config() for how this is picked
    up - re-read fresh from this file on every single LLM call, so a save
    here takes effect on the very next question with no app restart."""
    settings = load_settings()
    settings["temperature"] = _normalize_temperature(value)
    return save_settings(settings)
