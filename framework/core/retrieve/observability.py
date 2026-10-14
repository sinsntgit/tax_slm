"""
OBSERVABILITY LAYER (LangFuse) - PROMOTED unchanged from
tax_kg_poc/webapp/observability.py. Thin, no-op-if-unconfigured tracing
wrapper - 100% domain-agnostic.
"""
import logging
import os

from dotenv import load_dotenv

from framework.core.infra.dotenv_path import env_path as _env_path

# Anchored to the repo root by file location, NOT process cwd - see
# framework.core.infra.dotenv_path's docstring. A plain, path-less
# load_dotenv() here previously caused real .env keys (confirmed present by
# the user) to silently never be read whenever this app was launched from a
# working directory other than the repo root.
load_dotenv(_env_path())
logger = logging.getLogger("observability")

LANGFUSE_PUBLIC_KEY = os.environ.get("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.environ.get("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")

_langfuse_client = None
LANGFUSE_CONFIGURED = False  # True once a client built from real .env keys and passed auth_check()
LANGFUSE_ENABLED = False     # LANGFUSE_CONFIGURED AND the Settings-page on/off toggle
LANGFUSE_DISABLED_REASON = ""


def _tracing_toggle_enabled() -> bool:
    """Reads the Settings-page on/off preference - this is the ONLY thing a
    user can flip from the UI; the keys themselves always come from .env
    (see module docstring / refresh_langfuse_config()). Defaults to True
    (enabled) so a fresh install with real .env keys traces immediately,
    matching this module's pre-toggle behavior."""
    try:
        from framework.core.infra import settings_store
        return bool(settings_store.load_settings().get("langfuse_tracing_enabled", True))
    except Exception:
        return True


def refresh_langfuse_config(force: bool = False) -> dict:
    """(Re)reads LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST from .env ONLY (never
    from the Settings UI - the UI only ever sets the on/off toggle read by
    _tracing_toggle_enabled() above) and (re)constructs the module-level
    Langfuse client - unlike framework.core.infra.llm_client's
    refresh_model_config() (which just swaps a couple of strings), this
    rebuilds an actual client object, so it's called explicitly (at import
    time below, and from the Settings UI's enable/disable toggle handler)
    rather than on every request. Safe to call repeatedly - a no-op
    reconnect if force=False and the keys haven't changed since the client
    was last built.
    Returns {"enabled": bool, "reason": str} - the same shape the Settings
    UI needs to show an honest status, no fabricated success. `enabled`
    reflects LANGFUSE_ENABLED (i.e. also respects the toggle), not just
    whether the keys work."""
    global LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST
    global _langfuse_client, LANGFUSE_CONFIGURED, LANGFUSE_ENABLED, LANGFUSE_DISABLED_REASON

    load_dotenv(_env_path(), override=True)
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com").strip()
    toggle_on = _tracing_toggle_enabled()

    unchanged = (
        public_key == LANGFUSE_PUBLIC_KEY and secret_key == LANGFUSE_SECRET_KEY
        and host == LANGFUSE_HOST and _langfuse_client is not None
        and LANGFUSE_ENABLED == (LANGFUSE_CONFIGURED and toggle_on)
    )
    if unchanged and not force:
        return {"enabled": LANGFUSE_ENABLED, "reason": LANGFUSE_DISABLED_REASON}

    LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST = public_key, secret_key, host
    _langfuse_client = None
    LANGFUSE_CONFIGURED = False
    LANGFUSE_ENABLED = False
    LANGFUSE_DISABLED_REASON = ""

    if not (public_key and secret_key):
        LANGFUSE_DISABLED_REASON = "LANGFUSE_PUBLIC_KEY/SECRET_KEY not set in .env - tracing unavailable."
        logger.info(f"[observability] {LANGFUSE_DISABLED_REASON}")
        return {"enabled": False, "reason": LANGFUSE_DISABLED_REASON}

    try:
        from langfuse import Langfuse
        client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        # requirements.txt pins langfuse<3.0.0 because v3/v4 removed
        # .trace()/.span()/.generation() entirely (confirmed via a live
        # install test) - everything below depends on that v2 API. If the
        # environment actually has v3+ installed anyway (requirements.txt
        # not (re)installed since the pin was added, or something else
        # upgraded it), fail loudly ONCE here with the exact fix, instead
        # of silently degrading into a "log_generation failed: 'Langfuse'
        # object has no attribute 'generation'" warning on every single
        # LLM call.
        if not hasattr(client, "generation"):
            try:
                import importlib.metadata as _im
                _installed_version = _im.version("langfuse")
            except Exception:
                _installed_version = "unknown"
            LANGFUSE_DISABLED_REASON = (
                f"Installed langfuse version ({_installed_version}) is v3+ and no longer has "
                f".trace()/.span()/.generation() - tracing disabled. Fix: run "
                f"`pip install \"langfuse>=2.50.0,<3.0.0\" --force-reinstall` in this project's venv "
                f"(requirements.txt is already pinned correctly - the installed package just wasn't "
                f"reinstalled to match yet)."
            )
            logger.warning(f"[observability] {LANGFUSE_DISABLED_REASON}")
            return {"enabled": False, "reason": LANGFUSE_DISABLED_REASON}
        # A real round-trip, not just "the constructor didn't throw" - bad
        # keys/host construct fine but fail on first actual API call, so
        # auth_check() (part of the v2 SDK) is what actually proves the
        # connection works before we tell the user "enabled".
        try:
            client.auth_check()
        except Exception as e:
            LANGFUSE_DISABLED_REASON = f"Langfuse auth check failed ({e}) - check LANGFUSE_PUBLIC_KEY/SECRET_KEY/HOST in .env."
            logger.warning(f"[observability] {LANGFUSE_DISABLED_REASON}")
            return {"enabled": False, "reason": LANGFUSE_DISABLED_REASON}

        _langfuse_client = client
        LANGFUSE_CONFIGURED = True
        if not toggle_on:
            LANGFUSE_DISABLED_REASON = "Langfuse is configured in .env but tracing is switched off in Settings."
            logger.info(f"[observability] {LANGFUSE_DISABLED_REASON}")
            return {"enabled": False, "reason": LANGFUSE_DISABLED_REASON}

        LANGFUSE_ENABLED = True
        logger.info("[observability] LangFuse tracing ENABLED")
        return {"enabled": True, "reason": ""}
    except Exception as e:
        LANGFUSE_DISABLED_REASON = f"LangFuse init failed ({e}) - tracing disabled."
        logger.warning(f"[observability] {LANGFUSE_DISABLED_REASON}")
        return {"enabled": False, "reason": LANGFUSE_DISABLED_REASON}


refresh_langfuse_config()  # populate LANGFUSE_ENABLED/_langfuse_client at import time, same as llm_client.py


def start_trace(name: str, input_data: dict, run_id: str | None = None):
    if not LANGFUSE_ENABLED:
        return None
    try:
        return _langfuse_client.trace(name=name, input=input_data, id=run_id)
    except Exception as e:
        logger.warning(f"[observability] start_trace failed: {e}")
        return None


def log_span(trace, name: str, input_data, output_data, metadata: dict | None = None):
    if trace is None:
        return
    try:
        trace.span(name=name, input=input_data, output=output_data, metadata=metadata or {})
    except Exception as e:
        logger.warning(f"[observability] log_span failed for '{name}': {e}")


# 2026-08-29: NEW. A real request under retrieval_mode="semantic_graph"
# showed Langfuse repeatedly logging "Item exceeds size limit (size:
# 95029230)" (95MB!) and dropping that span's input/output/metadata
# entirely. Root cause: app.py's "bge_rerank"/"citation_grounding" spans
# pass the FULL candidate/top chunk dicts (list[dict] with a "text" key) as
# trace input/output. reranker.rerank() only caps chunk TEXT for its own
# cross-encoder scoring pairs (RERANK_TEXT_MAX_CHARS, a local copy) - it
# never mutates chunk["text"] itself. Under semantic_graph mode, chunk
# "text" can be an ENTIRE whole-file md_hydration.py read (capped at
# DEFAULT_WHOLE_FILE_MAX_CHARS = 4,000,000 chars there), and 30-39 such
# candidates easily sum to tens of MB - Langfuse's ingestion API has its own
# (much smaller) per-item size ceiling, so the whole span's data got
# silently dropped, losing that debugging trail entirely.
#
# This is PURELY a trace-payload concern: it does not touch what actually
# reaches the LLM (that's a separate, already-existing 1500-char-per-chunk
# cap in app.py's _build_context_parts) or what the reranker scores on
# (RERANK_TEXT_MAX_CHARS, already 1500 chars). Callers that hand a
# list[dict] of chunks to log_span()'s input_data/output_data should route
# it through this first so Langfuse gets a small, readable preview per
# chunk instead of the full hydrated text.
TRACE_CHUNK_TEXT_PREVIEW_CHARS = int(os.environ.get("TRACE_CHUNK_TEXT_PREVIEW_CHARS", "300"))


def summarize_chunks_for_trace(chunks, text_chars: int = TRACE_CHUNK_TEXT_PREVIEW_CHARS) -> list:
    """Returns a small, Langfuse-safe copy of a chunk-dict list: every key
    EXCEPT "text" is kept as-is (chunk_id, similarity, rerank_score, source,
    relevance_tier, etc. - all small scalars), and "text" is replaced with a
    `text_chars`-char preview plus the ORIGINAL full length (so a trace
    reviewer can still see "this was a 2.1MB whole-file hydration" without
    Langfuse having to actually store 2.1MB). Not chunk-dict-shaped input
    (e.g. a plain string or a non-list) is returned unchanged - callers that
    pass a question string or an already-small dict to log_span() don't need
    this and shouldn't be forced through it."""
    if not isinstance(chunks, list):
        return chunks
    summarized = []
    for c in chunks:
        if not isinstance(c, dict):
            summarized.append(c)
            continue
        row = {k: v for k, v in c.items() if k != "text"}
        full_text = c.get("text") or ""
        row["text_preview"] = full_text[:text_chars]
        row["text_full_chars"] = len(full_text)
        summarized.append(row)
    return summarized


def end_trace(trace, output_data):
    if trace is None:
        return
    try:
        trace.update(output=output_data)
    except Exception as e:
        logger.warning(f"[observability] end_trace failed: {e}")


def flush_langfuse() -> None:
    """The v2 SDK batches trace/span/generation calls in a background thread
    and ships them on its own timer - fine for a long-lived process left
    running, but nothing here ever explicitly flushed that queue, so any
    short-lived process (a CLI ingestion script that calls llm_client.
    complete() then exits) risked losing whatever hadn't been auto-flushed
    yet by exit time, and even in the long-lived Flask app there's no
    guarantee a trace is actually ON THE WIRE by the time a request
    completes and someone checks the Langfuse console. Call this after
    end_trace() at the end of a request/run that should be fully shipped
    before moving on. Safe no-op if tracing isn't configured."""
    if not LANGFUSE_ENABLED or _langfuse_client is None:
        return
    try:
        _langfuse_client.flush()
    except Exception as e:
        logger.warning(f"[observability] flush_langfuse failed: {e}")


def _iso(dt) -> str | None:
    if dt is None:
        return None
    try:
        return dt.isoformat()
    except Exception:
        return str(dt)


def get_recent_traces(hours: int = 24, limit: int = 100) -> dict:
    """Read-side query for the Monitoring screen + Dashboard's Langfuse-24h
    card: fetches real traces from the last `hours` hours via the v2 SDK's
    `fetch_traces()` (confirmed against the installed langfuse==2.60.10
    client.py - returns FetchTracesResponse(data=[TraceWithDetails], meta)).
    Degrades gracefully (empty stats, not an exception) whenever
    LANGFUSE_ENABLED is False or the API call itself fails (network down,
    bad keys, etc.) - callers should treat `available: False` as "show an
    honest empty state", never synthesize numbers to fill the gap."""
    if not LANGFUSE_ENABLED:
        return {"available": False, "reason": "Langfuse is not configured (no public/secret key in .env).",
                "traces": [], "stats": {"count": 0, "avg_latency_seconds": None, "total_cost": None, "error_rate": None}}
    try:
        import datetime as dt
        from_ts = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
        resp = _langfuse_client.fetch_traces(from_timestamp=from_ts, limit=limit, order_by="timestamp.desc")
        rows = []
        latencies = []
        costs = []
        error_count = 0
        for t in resp.data:
            latency = getattr(t, "latency", None)
            cost = getattr(t, "total_cost", None)
            # A trace's own observations carry `level` (DEFAULT/WARNING/ERROR) -
            # any ERROR-level observation marks the whole trace as an error for
            # the status column/error-rate stat.
            has_error = False
            try:
                for obs in (getattr(t, "observations", None) or []):
                    lvl = getattr(obs, "level", None) if not isinstance(obs, str) else None
                    if lvl == "ERROR":
                        has_error = True
                        break
            except Exception:
                pass
            if has_error:
                error_count += 1
            if latency is not None:
                latencies.append(latency)
            if cost is not None:
                costs.append(cost)
            question = None
            try:
                inp = getattr(t, "input", None)
                if isinstance(inp, dict):
                    question = inp.get("question")
            except Exception:
                pass
            status = "error" if has_error else ("slow" if (latency or 0) > 8 else "ok")
            rows.append({
                "id": t.id, "name": getattr(t, "name", None), "question": question,
                "timestamp": _iso(getattr(t, "timestamp", None)),
                "latency_seconds": latency, "cost": cost, "status": status,
            })
        n = len(resp.data)
        stats = {
            "count": n,
            "avg_latency_seconds": (sum(latencies) / len(latencies)) if latencies else None,
            "total_cost": sum(costs) if costs else None,
            "error_rate": (error_count / n) if n else None,
        }
        return {"available": True, "traces": rows, "stats": stats}
    except Exception as e:
        logger.warning(f"[observability] get_recent_traces failed: {e}")
        return {"available": False, "reason": f"Langfuse query failed: {e}",
                "traces": [], "stats": {"count": 0, "avg_latency_seconds": None, "total_cost": None, "error_rate": None}}


def get_generation_stats(hours: int = 24, limit: int = 500) -> dict:
    """Read-side query for token/latency stats across GENERATION-type
    observations in the last `hours` hours (used for the Dashboard's "avg
    answer latency" KPI, sourced from real generation observations rather
    than a guess). Same graceful-degradation contract as get_recent_traces()."""
    if not LANGFUSE_ENABLED:
        return {"available": False, "avg_latency_seconds": None, "count": 0}
    try:
        import datetime as dt
        from_ts = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=hours)
        resp = _langfuse_client.fetch_observations(type="GENERATION", from_start_time=from_ts, limit=limit)
        latencies = []
        for o in resp.data:
            start = getattr(o, "start_time", None)
            end = getattr(o, "end_time", None)
            if start is not None and end is not None:
                try:
                    latencies.append((end - start).total_seconds())
                except Exception:
                    pass
        return {
            "available": True,
            "avg_latency_seconds": (sum(latencies) / len(latencies)) if latencies else None,
            "count": len(resp.data),
        }
    except Exception as e:
        logger.warning(f"[observability] get_generation_stats failed: {e}")
        return {"available": False, "avg_latency_seconds": None, "count": 0}


def log_generation(trace, name: str, model: str, input_data, output_data, *,
                    usage: dict | None = None, metadata: dict | None = None) -> None:
    """Logs ONE completed LLM call as a Langfuse "generation" (model, prompt,
    completion, token usage if known) - unlike log_span() above (a generic
    step with no model/usage semantics), this is Langfuse's dedicated LLM-
    call record type, which is what powers its token/cost dashboards. Called
    from framework.core.infra.llm_client.complete() itself, not by each
    caller, so every LLM call anywhere in the app - ingestion and retrieval
    alike - is traced without any call site needing to know about Langfuse.
    `trace` is optional: pass one to nest this generation under an existing
    Langfuse trace (e.g. app.py's per-/api/ask-request trace); omit it (None)
    and this becomes its own standalone trace - either way it's logged."""
    if not LANGFUSE_ENABLED:
        return
    try:
        target = trace if trace is not None else _langfuse_client
        target.generation(
            name=name, model=model, input=input_data, output=output_data,
            usage=usage, metadata=metadata or {},
        )
    except Exception as e:
        logger.warning(f"[observability] log_generation failed for '{name}': {e}")
