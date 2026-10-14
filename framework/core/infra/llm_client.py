"""
Unified LLM client - PROMOTED from tax_kg_poc/scripts/llm_client.py, then
extended with three things every caller (every domain pack's extraction/
retrieval LLM calls) gets for free without any call-site changes:

  1. Per-provider model name is read from .env (CLAUDE_MODEL_NAME /
     GEMINI_MODEL_NAME / GPT_MODEL_NAME), not hardcoded - falls back to this
     file's own defaults (MODEL_NAME_DEFAULTS below) if the env var is unset
     or blank, so an empty/missing .env still behaves exactly as before.
  2. USE_MODEL and the active model name are re-read from .env on EVERY
     complete() call (see refresh_model_config()) - not just once at process
     start - so editing .env while the Flask app (or a long-running
     ingestion subprocess) is already running takes effect on the very next
     LLM call, no restart needed. This also means switching PROVIDER at
     runtime (USE_MODEL=CLAUDE -> GEMINI) works: _get_client() rebuilds its
     cached SDK client whenever the active provider changes, instead of
     permanently caching whichever provider was configured at import time.
  3. Every call is logged to Langfuse as a "generation" (model, prompt,
     completion, token usage, latency) via framework.core.retrieve.
     observability - the same no-op-if-unconfigured LangFuse wrapper
     webapp/app.py's /api/ask pipeline already uses for its coarser
     per-stage spans. Doing it HERE instead of in every caller means every
     LLM call anywhere in the app - ingestion (extract_entities.py/
     extract_case_law.py/extract_generic.py) as much as retrieval
     (intent.py/graphrag.py/app.py's answer generation) - gets traced
     automatically, with zero changes needed at any of those 6 call sites.
     An optional `trace` kwarg lets a caller that already has its own
     Langfuse trace (app.py's /api/ask does) nest this generation under it
     instead of it becoming its own standalone trace - purely additive,
     every existing call site keeps working unmodified since it's optional.

2026-10-14: Phase 0 of the "Tax SLM" product (a fine-tuned open-weight model
served by vLLM inside the customer's environment, replacing Claude/Gemini/
GPT API calls) added three more things, all purely additive/opt-in:

  4. A LOCAL provider (USE_MODEL=LOCAL) for ANY OpenAI-compatible endpoint -
     vLLM, Ollama, LM Studio - configured by LOCAL_LLM_BASE_URL /
     LOCAL_MODEL_NAME / LOCAL_LLM_API_KEY / LOCAL_LLM_MAX_TOKENS /
     LOCAL_LLM_CONTEXT_LENGTH (see _complete_local()).
  5. Per-CALL routing: LLM_ROUTE_<CALL_NAME>=<PROVIDER> (and optionally
     LLM_ROUTE_<CALL_NAME>_MODEL=<model>) sends one named call (the `name=`
     every call site already passes, e.g. nl2cypher / intent_detection) to a
     different provider than its purpose/global setting - so the SLM can be
     rolled out one call type at a time and compared side by side (see
     resolve_call_route()). Hot-reloaded like everything else.
  6. Opt-in training-data capture (LLM_CALL_LOG=1) of every complete() call
     to JSONL - see framework.core.infra.llm_call_log, and
     framework/cli/export_training_data.py to turn it into chat-format
     fine-tuning data.
"""
import os
import re
import threading
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from framework.core.infra import llm_call_log, step_logger
from framework.core.infra.dotenv_path import env_path as _env_path
from framework.core.retrieve import observability

# Anchored to the repo root by file location, not process cwd - see
# framework.core.infra.dotenv_path's docstring.
load_dotenv(_env_path())

# --- Per-provider model name: env var name -> fallback default ------------
# CLAUDE_MODEL_NAME / GEMINI_MODEL_NAME / GPT_MODEL_NAME in .env override
# these; see refresh_model_config(). Kept as the single source of truth for
# "what model do we use if .env doesn't say" so there's exactly one place to
# update when a provider ships a new default model.
MODEL_NAME_DEFAULTS = {
    "CLAUDE": "claude-haiku-4-5-20251001",
    "GEMINI": "gemini-1.5-flash",
    "GPT": "gpt-5-nano",
    # 2026-10-14: NEW - blank on purpose. There is no sensible universal
    # default for a self-hosted model, and guessing one would just produce a
    # confusing 404 from the server. Blank means "ask the server": the first
    # LOCAL call queries GET <base_url>/models and uses the first model it
    # serves (vLLM normally serves exactly one) - see _discover_local_model().
    "LOCAL": "",
}
# Old name, kept as an alias for anything importing MODEL_NAMES directly.
MODEL_NAMES = MODEL_NAME_DEFAULTS


@dataclass(frozen=True)
class ModelConfig:
    """Immutable snapshot of provider/model/timeout for ONE complete() call.

    THREAD-SAFETY FIX: this used to be three bare module globals (USE_MODEL/
    MODEL_NAME/LLM_TIMEOUT_SECONDS) that refresh_model_config() rewrote in
    place on every call. Flask's dev server runs each request on its own
    thread, so page-load requests (which call complete() with
    purpose="ingestion" or "retrieval" depending on route) could interleave
    with an in-flight /api/ask call - one thread's refresh_model_config()
    would overwrite the globals mid-request, and since _complete_gpt/
    _complete_claude/_complete_gemini and _get_client() each re-read the
    globals independently (not a value captured once), a single logical
    call could end up using one thread's PROVIDER (e.g. the OpenAI client,
    already selected via _DISPATCH[USE_MODEL]) with ANOTHER thread's
    MODEL_NAME (e.g. a Claude model string) - exactly the "GPT client asked
    for claude-haiku-4-5-20251001" bug this was built to fix. Resolving one
    ModelConfig up front and threading it explicitly through every function
    call below (instead of touching the globals again) makes each complete()
    call immune to concurrent refreshes on other threads, no matter how they
    interleave."""
    use_model: str
    model_name: str
    timeout: float
    # 2026-08-27: NEW - sampling temperature, read from settings_store.py (a
    # Settings-page control, not .env - see that module's docstring on the
    # new "temperature" setting) rather than from os.environ like the other
    # ModelConfig fields, since this one is meant to be changed from the UI
    # without touching .env at all. None = don't send an explicit temperature
    # (defer to the provider's own default) - the behavior before this field
    # existed, still the default.
    temperature: float | None
    # 2026-09-05: NEW - purpose-scoped GPT reasoning_effort (was a single
    # flat GPT_REASONING_EFFORT global read once at import time - see the
    # module-level constant of the same name below, now only a fallback
    # default). Added per an explicit consistency investigation: OpenAI's
    # gpt-5.1 family accepts an explicit `temperature` ONLY when
    # reasoning_effort="none" (confirmed against OpenAI's own published
    # reasoning-effort/sampling-parameter compatibility matrix,
    # 2026-09 community thread) - every other effort level on that family,
    # and gpt-5/gpt-5.4-style models generally, reject temperature outright
    # (this app already handles that gracefully - see _complete_gpt below).
    # Threaded through ModelConfig (not read as a bare global inside
    # _complete_gpt) for the SAME thread-safety reason temperature/model_name
    # already are - see this class's main docstring.
    reasoning_effort: str | None
    # 2026-10-14: NEW (all optional, defaulted, so existing positional/keyword
    # constructions keep working) - per-call routing provenance plus the
    # LOCAL provider's endpoint settings. Resolved in refresh_model_config()
    # and threaded through like every other field here, for the same
    # thread-safety reason (see this class's docstring): a concurrent .env
    # edit can never pair one thread's base_url with another's model name.
    # `route` = which LLM_ROUTE_* env key (if any) picked the provider.
    route: str | None = None
    base_url: str | None = None
    api_key: str | None = field(default=None, repr=False)   # never printed in logs/reprs
    max_retries: int = 1
    max_tokens_cap: int | None = None
    context_length: int | None = None
    # None = auto (send a system message; fall back to prepending it to the
    # user turn if the server's chat template rejects the system role).
    system_role: bool | None = None
    strip_think: bool = True


_client_lock = threading.Lock()
_clients: dict[str, object] = {}   # keyed by provider name, one cached SDK client per provider


# Purpose-scoped provider override - USE_MODEL_INGESTION / USE_MODEL_RETRIEVAL
# in .env pick a DIFFERENT provider for extraction (extract_entities.py/
# extract_case_law.py/extract_generic.py, all called with purpose="ingestion")
# vs. the /api/ask pipeline (intent.py/graphrag.py/app.py's answer generation,
# all default to purpose="retrieval") - e.g. a cheap/fast model for bulk
# ingestion, a stronger one for user-facing answers. Either or both may be
# left unset, in which case that purpose just falls back to plain USE_MODEL -
# so this is purely additive, existing single-USE_MODEL setups are unaffected.
PURPOSE_ENV_KEYS = {"ingestion": "USE_MODEL_INGESTION", "retrieval": "USE_MODEL_RETRIEVAL"}


# 2026-10-14: BUGFIX - MOVED here from further down this file.
# refresh_model_config() runs once at IMPORT time (see the call right after
# its definition) and reads this constant as its last-resort fallback, but it
# used to be defined ~120 lines below that call, so importing this module
# raised NameError on any machine whose .env sets neither
# GPT_REASONING_EFFORT_<PURPOSE> nor GPT_REASONING_EFFORT - i.e. every fresh
# install. It only worked on the dev machine because its .env happens to set
# GPT_REASONING_EFFORT_RETRIEVAL, which short-circuits the `or` chain before
# the undefined name is reached. Found by the offline import smoke test.
GPT_REASONING_EFFORT_DEFAULT = os.environ.get("GPT_REASONING_EFFORT", "minimal")


# 2026-10-14: NEW - per-call routing for the gradual SLM rollout. Every call
# site already passes a stable `name=` (nl2cypher, intent_detection,
# answer_generation, ...) for Langfuse; LLM_ROUTE_<NAME> reuses it as the
# routing key so moving one call type onto the local model needs no code
# change, only a .env line. Some names carry a dynamic suffix (e.g.
# "case_law_extraction_<file stem>", "semantic_layer_group_3_of_12"), so the
# lookup is LONGEST-PREFIX over underscore-separated segments:
# LLM_ROUTE_CASE_LAW_EXTRACTION matches every case_law_extraction_* call, and
# LLM_ROUTE_ANSWER would deliberately match answer_generation,
# answer_evaluation and answer_quality_judge together. Candidates ending in
# _MODEL are skipped because LLM_ROUTE_<NAME>_MODEL is the companion
# model-name override, not a provider.
ROUTE_ENV_PREFIX = "LLM_ROUTE_"


def _route_candidates(name: str) -> list[str]:
    norm = re.sub(r"[^A-Za-z0-9]+", "_", name or "").strip("_").upper()
    if not norm:
        return []
    parts = norm.split("_")
    return ["_".join(parts[:i]) for i in range(len(parts), 0, -1)
            if not "_".join(parts[:i]).endswith("_MODEL") and "_".join(parts[:i]) != "MODEL"]


def resolve_call_route(name: str | None) -> tuple[str | None, str | None, str | None]:
    """Returns (env_key, provider, model_override) for the most specific
    LLM_ROUTE_* key set (non-blank) for this call name, or (None, None,
    None) when no route applies. Reads os.environ only - the caller
    (refresh_model_config) has just re-read .env, so this is hot-reloaded.
    Note: python-dotenv never REMOVES a key that was deleted from .env, so
    to un-route a call while running, set it blank (LLM_ROUTE_X=) rather
    than deleting the line - same caveat as every other setting here."""
    for cand in _route_candidates(name or ""):
        key = f"{ROUTE_ENV_PREFIX}{cand}"
        provider = os.environ.get(key, "").strip().upper()
        if provider:
            model = os.environ.get(f"{key}_MODEL", "").strip() or None
            return key, provider, model
    return None, None, None


def _env_int(key: str, default: int | None) -> int | None:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        step_logger.log(f"llm_client: WARNING - {key}={raw!r} is not a number; ignoring it.")
        return default


def _env_bool(key: str, default: bool | None) -> bool | None:
    raw = os.environ.get(key, "").strip().lower()
    if not raw or raw == "auto":
        return default
    return raw in ("1", "true", "yes", "on")


# Last config logged per (purpose, route) - so alternating routed/unrouted
# calls don't spam the step log with "config (re)loaded" on every call.
_last_logged_config: dict[tuple, tuple] = {}


def refresh_model_config(purpose: str = "retrieval", name: str | None = None) -> ModelConfig:
    """Re-reads USE_MODEL(_INGESTION/_RETRIEVAL) / <PROVIDER>_MODEL_NAME /
    LLM_TIMEOUT_SECONDS from .env and returns a fresh, immutable ModelConfig
    snapshot for THIS call only (see ModelConfig's docstring for why this is
    no longer stashed in mutable module globals). Called at import time (to
    populate the back-compat USE_MODEL/MODEL_NAME/LLM_TIMEOUT_SECONDS globals
    below, which nothing in this file's hot path reads anymore, but are kept
    for any external code that imported them directly) AND at the top of
    every complete() call (so a .env edit made while the process is already
    running - Flask app or an ingestion subprocess - takes effect on the very
    next call, genuinely at runtime, no restart). `override=True` is what
    makes this pick up an EDITED value for a key that was already present in
    os.environ at process start - plain load_dotenv() only fills in keys
    that aren't set yet, which would silently ignore a runtime edit. Cheap
    enough to call on every LLM call - it's a small local file read,
    negligible next to the seconds an LLM call itself takes."""
    load_dotenv(_env_path(), override=True)

    # 2026-10-14: precedence is now call route > purpose > global. `name` is
    # optional so every existing caller of refresh_model_config(purpose)
    # (pipeline.py's llm_chunk_info(), etc.) behaves exactly as before.
    route_key, route_provider, route_model = resolve_call_route(name)
    purpose_key = PURPOSE_ENV_KEYS.get(purpose)
    purpose_override = os.environ.get(purpose_key, "").strip().upper() if purpose_key else ""
    use_model = route_provider or purpose_override or os.environ.get("USE_MODEL", "CLAUDE").strip().upper()
    if use_model not in MODEL_NAME_DEFAULTS:
        source = route_key or (purpose_key if purpose_override else "USE_MODEL")
        raise ValueError(
            f"{source}='{use_model}' in .env is "
            f"not recognized. Set it to one of: {', '.join(MODEL_NAME_DEFAULTS)}."
        )
    # Purpose-scoped MODEL NAME override (on top of the purpose-scoped
    # PROVIDER override above) - e.g. CLAUDE_MODEL_NAME_INGESTION=<cheap
    # model> + CLAUDE_MODEL_NAME_RETRIEVAL=<stronger model>, so ingestion
    # and retrieval can use different models even when both use the SAME
    # provider, not just when they use different providers. Falls back to
    # the plain <PROVIDER>_MODEL_NAME, then the hardcoded default - purely
    # additive, unset means "no purpose-specific override".
    model_name = (
        (route_model or "")
        or os.environ.get(f"{use_model}_MODEL_NAME_{purpose.upper()}", "").strip()
        or os.environ.get(f"{use_model}_MODEL_NAME", "").strip()
        or MODEL_NAME_DEFAULTS[use_model]
    )
    timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", "120"))

    # 2026-09-05: NEW - purpose-scoped reasoning_effort, same override chain
    # as model_name above: GPT_REASONING_EFFORT_<PURPOSE> -> plain
    # GPT_REASONING_EFFORT -> the hardcoded "minimal" module default. Only
    # meaningful for GPT (Claude/Gemini ignore this field entirely - see
    # _complete_claude/_complete_gemini, neither reads config.reasoning_effort),
    # but resolved for every purpose/provider combination the same way
    # model_name is, for consistency and so a future non-GPT use isn't
    # blocked by this being GPT-specific here.
    reasoning_effort = (
        os.environ.get(f"GPT_REASONING_EFFORT_{purpose.upper()}", "").strip()
        or os.environ.get("GPT_REASONING_EFFORT", "").strip()
        or GPT_REASONING_EFFORT_DEFAULT
    )

    # 2026-08-27: NEW - re-read on every call, same "hot, no restart" contract
    # as everything else in this function - settings_store.load_settings()
    # re-reads its JSON file from disk each call, so a Settings-page save
    # takes effect on the very next question. Deliberately a LOCAL import
    # (not at module top) to avoid a circular import: settings_store is pure
    # infra with no dependency back on llm_client, so this is safe either
    # way, but keeping it local mirrors this function's existing pattern of
    # only touching outside state right where it's used.
    from framework.core.infra import settings_store
    temperature = settings_store.load_settings().get("temperature")

    # 2026-10-14: NEW - LOCAL endpoint settings, only resolved when LOCAL is
    # the provider for this call (other providers leave them None).
    local_kwargs = {}
    if use_model == "LOCAL":
        local_kwargs = dict(
            base_url=os.environ.get("LOCAL_LLM_BASE_URL", "").strip().rstrip("/") or None,
            # Placeholder rather than None on purpose: the openai SDK falls
            # back to OPENAI_API_KEY when api_key is None, which would send
            # the real OpenAI key to a self-hosted server. vLLM/Ollama/LM
            # Studio accept any string unless started with --api-key.
            api_key=os.environ.get("LOCAL_LLM_API_KEY", "").strip() or "not-needed",
            max_retries=max(0, _env_int("LOCAL_LLM_MAX_RETRIES", 1)),
            max_tokens_cap=_env_int("LOCAL_LLM_MAX_TOKENS", None),
            context_length=_env_int("LOCAL_LLM_CONTEXT_LENGTH", None),
            system_role=_env_bool("LOCAL_LLM_SYSTEM_ROLE", None),
            strip_think=_env_bool("LOCAL_LLM_STRIP_THINK", True),
        )

    config = ModelConfig(use_model=use_model, model_name=model_name, timeout=timeout, temperature=temperature,
                          reasoning_effort=reasoning_effort, route=route_key, **local_kwargs)

    global USE_MODEL, MODEL_NAME, LLM_TIMEOUT_SECONDS, TEMPERATURE
    log_key = (purpose, route_key)
    log_val = (use_model, model_name, temperature, local_kwargs.get("base_url"))
    changed = _last_logged_config.get(log_key) != log_val
    _last_logged_config[log_key] = log_val
    # Back-compat globals only - see docstring. Note these ARE still racy
    # under concurrent calls, same as before; the fix is that nothing in
    # THIS file reads them anymore (everything threads the returned
    # ModelConfig explicitly instead), so the race can no longer corrupt an
    # in-flight request. Left in place only in case other modules import
    # llm_client.MODEL_NAME/.USE_MODEL directly for logging/display.
    USE_MODEL, MODEL_NAME, LLM_TIMEOUT_SECONDS, TEMPERATURE = use_model, model_name, timeout, temperature
    if changed:
        step_logger.log(f"llm_client: config (re)loaded for purpose='{purpose}'"
                         f"{f' (routed by {route_key})' if route_key else ''} - "
                         f"USE_MODEL={use_model}, MODEL_NAME={model_name or '(auto-discover)'}, "
                         f"timeout={timeout:.0f}s, "
                         f"temperature={temperature if temperature is not None else 'provider default'}")
    return config


refresh_model_config()  # populate back-compat USE_MODEL/MODEL_NAME/LLM_TIMEOUT_SECONDS globals at import time


def _get_client(config: ModelConfig):
    """Returns the SDK client for config.use_model, building + caching it
    the first time each provider is needed. Cached per-provider (dict keyed
    by provider name, e.g. one Anthropic client AND one OpenAI client can
    coexist) rather than in a single _client/_client_provider slot - the old
    single-slot cache plus a provider-changed check was itself part of the
    thread-safety bug (see ModelConfig's docstring): under concurrent
    requests using different purposes/providers, one thread rebuilding the
    single slot for its provider could hand a freshly-built client of the
    WRONG provider to another thread that was about to call it. A dict
    keyed by provider name has no such shared mutable "current provider"
    state to race on - each provider's client, once built, never changes."""
    with _client_lock:
        client = _clients.get(config.use_model)
        if client is not None:
            return client

        if config.use_model == "CLAUDE":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "USE_MODEL=CLAUDE but ANTHROPIC_API_KEY is not set (or is blank) in .env."
                )
            from anthropic import Anthropic
            client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"], timeout=config.timeout)

        elif config.use_model == "GEMINI":
            if not os.environ.get("GEMINI_API_KEY"):
                raise RuntimeError(
                    "USE_MODEL=GEMINI but GEMINI_API_KEY is not set (or is blank) in .env."
                )
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
            client = genai

        elif config.use_model == "GPT":
            if not os.environ.get("OPENAI_API_KEY"):
                raise RuntimeError(
                    "USE_MODEL=GPT but OPENAI_API_KEY is not set (or is blank) in .env."
                )
            from openai import OpenAI
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], timeout=config.timeout)

        else:
            raise ValueError(f"Unknown USE_MODEL={config.use_model!r}")

        _clients[config.use_model] = client
        step_logger.log(f"llm_client: initialized {config.use_model} client (model={config.model_name}, "
                         f"timeout={config.timeout:.0f}s)")
        return client


def _complete_claude(config: ModelConfig, system: str, user_content: str, max_tokens: int) -> tuple[str, dict | None]:
    client = _get_client(config)
    kwargs = {} if config.temperature is None else {"temperature": config.temperature}
    response = client.messages.create(
        model=config.model_name,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_content}],
        **kwargs,
    )
    if getattr(response, "stop_reason", None) == "max_tokens":
        # The response was cut off mid-generation because it hit max_tokens -
        # for a JSON-producing caller (e.g. extract_entities.py's semantic
        # layer) this almost always means truncated/unparseable JSON, i.e.
        # this call's findings get silently dropped by the caller's own
        # try/except unless this is logged here. Raising max_tokens (or
        # shrinking the input group so there's less to describe) fixes it.
        step_logger.log(f"llm_client: WARNING - Claude ({config.model_name}) response TRUNCATED "
                         f"(stop_reason='max_tokens', budget was {max_tokens}) - likely incomplete/"
                         f"unparseable output. Raise the caller's max_tokens or reduce its input group size.")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    usage = None
    try:
        u = getattr(response, "usage", None)
        if u is not None:
            usage = {"input": u.input_tokens, "output": u.output_tokens,
                      "total": u.input_tokens + u.output_tokens}
    except Exception:
        pass
    return text, usage


def _complete_gemini(config: ModelConfig, system: str, user_content: str, max_tokens: int) -> tuple[str, dict | None]:
    genai = _get_client(config)
    model = genai.GenerativeModel(config.model_name, system_instruction=system)
    generation_config = {"max_output_tokens": max_tokens}
    if config.temperature is not None:
        generation_config["temperature"] = config.temperature
    response = model.generate_content(
        user_content,
        generation_config=generation_config,
        request_options={"timeout": config.timeout},
    )
    if not response.candidates:
        return "", None
    finish_reason = getattr(response.candidates[0], "finish_reason", None)
    # google-generativeai reports this as an enum whose *name* is "MAX_TOKENS"
    # when truncated - same silent-truncation risk/fix as the Claude branch above.
    if getattr(finish_reason, "name", str(finish_reason)) == "MAX_TOKENS":
        step_logger.log(f"llm_client: WARNING - Gemini ({config.model_name}) response TRUNCATED "
                         f"(finish_reason=MAX_TOKENS, budget was {max_tokens}) - likely incomplete/"
                         f"unparseable output. Raise the caller's max_tokens or reduce its input group size.")
    text = (response.text or "").strip()
    usage = None
    try:
        um = getattr(response, "usage_metadata", None)
        if um is not None:
            usage = {"input": um.prompt_token_count, "output": um.candidates_token_count,
                      "total": um.total_token_count}
    except Exception:
        pass
    return text, usage


# 2026-09-05: RENAMED from GPT_REASONING_EFFORT (which is now resolved
# purpose-scoped, per-call, in refresh_model_config() -> ModelConfig.
# reasoning_effort - see that field's docstring). This constant now only
# supplies the hardcoded fallback when NEITHER GPT_REASONING_EFFORT_<PURPOSE>
# NOR plain GPT_REASONING_EFFORT is set in .env, mirroring how
# MODEL_NAME_DEFAULTS is the last-resort fallback for model_name.
# 2026-10-14: GPT_REASONING_EFFORT_DEFAULT itself MOVED up to sit before
# refresh_model_config() - see the note there.
GPT_REASONING_TOKEN_BUDGET = int(os.environ.get("GPT_REASONING_TOKEN_BUDGET", "2000"))

_reasoning_effort_supported = None
# 2026-08-27: NEW - OpenAI's reasoning-family models (o-series, gpt-5-*,
# which is what GPT_MODEL_NAME defaults to) reject an explicit `temperature`
# param outright (400 error) unless it's left at the provider default -
# unlike reasoning_effort, which they DO support. Same "remember it failed,
# skip it for the rest of this run" pattern as _reasoning_effort_supported
# above, so a temperature setting that this model can't honor doesn't retry
# and fail on every single call.
_temperature_supported = None


def _complete_gpt(config: ModelConfig, system: str, user_content: str, max_tokens: int) -> tuple[str, dict | None]:
    client = _get_client(config)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]
    budgeted_tokens = max_tokens + GPT_REASONING_TOKEN_BUDGET

    def _call(**extra):
        try:
            return client.chat.completions.create(
                model=config.model_name, max_completion_tokens=budgeted_tokens, messages=messages, **extra,
            )
        except TypeError:
            return client.chat.completions.create(
                model=config.model_name, max_tokens=budgeted_tokens, messages=messages,
            )

    global _reasoning_effort_supported, _temperature_supported
    # 2026-08-27: NEW - build the optional-kwargs set once, then fall back in
    # stages if the model rejects any of it: full set -> drop temperature ->
    # drop reasoning_effort too. Mirrors the single-flag retry this function
    # already did for reasoning_effort alone, just widened to two
    # independent optional params that can each be unsupported separately.
    kwargs = {}
    if _reasoning_effort_supported is not False and config.reasoning_effort:
        kwargs["reasoning_effort"] = config.reasoning_effort
    if _temperature_supported is not False and config.temperature is not None:
        kwargs["temperature"] = config.temperature

    try:
        response = _call(**kwargs)
        if "reasoning_effort" in kwargs:
            _reasoning_effort_supported = True
        if "temperature" in kwargs:
            _temperature_supported = True
    except Exception as e1:
        if "temperature" in kwargs:
            step_logger.log(f"llm_client: GPT call with temperature={config.temperature} rejected "
                             f"({type(e1).__name__}: {e1}) - retrying without temperature (this model's "
                             f"reasoning-effort setting doesn't allow overriding it), and will skip "
                             f"temperature for the rest of this run.")
            _temperature_supported = False
            kwargs.pop("temperature", None)
            try:
                response = _call(**kwargs)
                if "reasoning_effort" in kwargs:
                    _reasoning_effort_supported = True
            except Exception as e2:
                step_logger.log(f"llm_client: GPT call with reasoning_effort='{config.reasoning_effort}' "
                                 f"rejected ({type(e2).__name__}: {e2}) - retrying with neither, and will "
                                 f"skip reasoning_effort for the rest of this run.")
                _reasoning_effort_supported = False
                response = _call()
        else:
            step_logger.log(f"llm_client: GPT call with reasoning_effort='{config.reasoning_effort}' "
                             f"rejected ({type(e1).__name__}: {e1}) - retrying without it, and will "
                             f"skip reasoning_effort for the rest of this run.")
            _reasoning_effort_supported = False
            response = _call()

    choice = response.choices[0]
    finish_reason = getattr(choice, "finish_reason", None)
    usage_obj = getattr(response, "usage", None)
    completion_tokens = getattr(usage_obj, "completion_tokens", None) if usage_obj else None
    prompt_tokens = getattr(usage_obj, "prompt_tokens", None) if usage_obj else None
    reasoning_tokens = None
    if usage_obj is not None:
        details = getattr(usage_obj, "completion_tokens_details", None)
        reasoning_tokens = getattr(details, "reasoning_tokens", None) if details else None
    step_logger.log(f"llm_client: GPT response - finish_reason={finish_reason}, "
                     f"completion_tokens={completion_tokens}, reasoning_tokens={reasoning_tokens}, "
                     f"budget_sent={budgeted_tokens}")

    content = (choice.message.content or "").strip()
    if not content and finish_reason == "length":
        step_logger.log(
            f"llm_client: WARNING - GPT ({config.model_name}) returned EMPTY content with finish_reason="
            f"'length': it used its entire {budgeted_tokens}-token budget "
            f"({reasoning_tokens if reasoning_tokens is not None else 'most'} on reasoning) and had "
            f"none left to write the JSON answer. This chunk will be skipped. To fix for future runs: "
            f"raise GPT_REASONING_TOKEN_BUDGET and/or the caller's max_tokens in .env."
        )
    usage = None
    if usage_obj is not None:
        usage = {"input": prompt_tokens, "output": completion_tokens,
                 "total": (prompt_tokens or 0) + (completion_tokens or 0)}
    return content, usage


# --- LOCAL provider (vLLM / Ollama / LM Studio / any OpenAI-compatible server) --
# 2026-10-14: NEW - Phase 0 of the Tax SLM product. Uses the openai SDK (already
# a dependency for GPT) pointed at LOCAL_LLM_BASE_URL rather than a bespoke
# HTTP client: vLLM, Ollama (/v1) and LM Studio all implement the same
# /v1/chat/completions contract, and the SDK already gives us timeouts,
# retries and typed errors. Deliberately a SEPARATE function from
# _complete_gpt: none of GPT's reasoning_effort / max_completion_tokens /
# reasoning-token-budget handling applies to a self-hosted model, and
# keeping it apart means a change for OpenAI can never break the SLM path.


class LocalLLMError(RuntimeError):
    """Raised for any LOCAL-provider failure (timeout, connection refused,
    HTTP error) with a message that says what to check - callers already
    catch Exception around complete(), so this changes no control flow."""


_local_discovered_models: dict[str, str] = {}             # base_url -> model id served there
_local_no_system_role: set[tuple[str, str]] = set()       # (base_url, model) whose template rejects "system"
_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)
# Rough chars-per-token for prompt-size estimation against
# LOCAL_LLM_CONTEXT_LENGTH. Deliberately conservative (real English legal text
# is ~4 chars/token on Llama/Qwen tokenizers) so the estimate errs towards
# leaving room, not overflowing the server's context window.
_CHARS_PER_TOKEN_ESTIMATE = 3.0
_CHAT_TEMPLATE_OVERHEAD_TOKENS = 64


def _get_local_client(config: ModelConfig):
    if not config.base_url:
        raise RuntimeError(
            "USE_MODEL=LOCAL (or an LLM_ROUTE_*=LOCAL route) but LOCAL_LLM_BASE_URL is not set in .env - "
            "set it to the server's OpenAI-compatible base URL, e.g. http://localhost:8000/v1 (vLLM) or "
            "http://localhost:11434/v1 (Ollama)."
        )
    # Keyed by endpoint+credentials (not just "LOCAL" like the other
    # providers' cache) so a hot .env edit of LOCAL_LLM_BASE_URL/API key
    # takes effect on the next call instead of reusing a stale client.
    key = ("LOCAL", config.base_url, config.api_key, config.max_retries)
    with _client_lock:
        client = _clients.get(key)
        if client is None:
            from openai import OpenAI
            client = OpenAI(base_url=config.base_url, api_key=config.api_key,
                            timeout=config.timeout, max_retries=config.max_retries)
            _clients[key] = client
            step_logger.log(f"llm_client: initialized LOCAL client (base_url={config.base_url}, "
                             f"max_retries={config.max_retries})")
        return client


def _local_error(config: ModelConfig, model: str, e: Exception) -> LocalLLMError:
    import openai
    where = f"LOCAL model server at {config.base_url} (model={model or '?'})"
    if isinstance(e, openai.APITimeoutError):
        return LocalLLMError(f"{where} did not respond within {config.timeout:.0f}s (LLM_TIMEOUT_SECONDS). "
                             f"The server may be overloaded or still loading the model; raise "
                             f"LLM_TIMEOUT_SECONDS or check the server's logs.")
    if isinstance(e, openai.APIConnectionError):
        return LocalLLMError(f"Could not connect to {where}: {e}. Is the server running and is "
                             f"LOCAL_LLM_BASE_URL correct (it must include the /v1 suffix)? If this host "
                             f"uses an HTTP proxy, add the server's host to NO_PROXY.")
    if isinstance(e, openai.APIStatusError):
        body = ""
        try:
            body = (e.response.text or "")[:500]
        except Exception:
            pass
        hint = ""
        if e.status_code == 404:
            hint = (" A 404 usually means LOCAL_MODEL_NAME doesn't match the server's served model name "
                    "(vLLM --served-model-name) or LOCAL_LLM_BASE_URL is missing /v1.")
        elif e.status_code in (401, 403):
            hint = " Check LOCAL_LLM_API_KEY matches the server's --api-key."
        elif e.status_code == 400 and "context" in body.lower():
            hint = " The prompt likely exceeds the model's context window - set LOCAL_LLM_CONTEXT_LENGTH."
        return LocalLLMError(f"{where} returned HTTP {e.status_code}: {body or e}.{hint}")
    return LocalLLMError(f"{where} failed: {type(e).__name__}: {e}")


def _discover_local_model(client, config: ModelConfig) -> str:
    cached = _local_discovered_models.get(config.base_url)
    if cached:
        return cached
    try:
        models = client.models.list(timeout=min(config.timeout, 30))
        ids = [m.id for m in getattr(models, "data", []) or []]
    except Exception as e:
        raise _local_error(config, "", e) from e
    if not ids:
        raise LocalLLMError(f"LOCAL_MODEL_NAME is blank and the server at {config.base_url} reports no models "
                            f"at /models - set LOCAL_MODEL_NAME explicitly.")
    if len(ids) > 1:
        step_logger.log(f"llm_client: LOCAL server serves {len(ids)} models {ids[:5]} and LOCAL_MODEL_NAME is "
                         f"blank - using the first, '{ids[0]}'. Set LOCAL_MODEL_NAME (or a per-call "
                         f"LLM_ROUTE_<NAME>_MODEL, e.g. a LoRA adapter name) to choose explicitly.")
    _local_discovered_models[config.base_url] = ids[0]
    return ids[0]


def _local_effective_max_tokens(config: ModelConfig, system: str, user_content: str, max_tokens: int) -> int:
    """Applies LOCAL_LLM_MAX_TOKENS (hard cap) and LOCAL_LLM_CONTEXT_LENGTH
    (prompt + completion must fit) to the caller's max_tokens. Callers were
    tuned for 128k-1M-context API models (answer_generation asks for 5000
    output tokens on a very long prompt); a 7-14B model served with an 8-32k
    window would otherwise reject the request outright with HTTP 400."""
    effective = max_tokens
    if config.max_tokens_cap:
        effective = min(effective, config.max_tokens_cap)
    if config.context_length:
        est_prompt = int((len(system) + len(user_content)) / _CHARS_PER_TOKEN_ESTIMATE) + _CHAT_TEMPLATE_OVERHEAD_TOKENS
        room = config.context_length - est_prompt
        if room < effective:
            if room < 64:
                step_logger.log(f"llm_client: WARNING - LOCAL prompt (~{est_prompt} tokens est.) nearly fills or "
                                 f"exceeds LOCAL_LLM_CONTEXT_LENGTH={config.context_length}; the server will "
                                 f"probably reject or truncate it.")
            effective = max(64, room)
    if effective != max_tokens:
        step_logger.log(f"llm_client: LOCAL max_tokens reduced {max_tokens} -> {effective} "
                         f"(LOCAL_LLM_MAX_TOKENS/LOCAL_LLM_CONTEXT_LENGTH)")
    return effective


def _looks_like_system_role_rejection(e: Exception) -> bool:
    """Chat templates that have no system role (Gemma, older Mistral) make
    vLLM return HTTP 400 with e.g. 'System role not supported' or
    'Conversation roles must alternate user/assistant/...'. Matched narrowly
    so an unrelated 400 (context overflow, bad param) is NOT retried."""
    import openai
    if not isinstance(e, openai.BadRequestError):
        return False
    try:
        text = (e.response.text or "") + " " + str(e)
    except Exception:
        text = str(e)
    text = text.lower()
    return ("system" in text and ("role" in text or "not supported" in text)) or "roles must alternate" in text


def _complete_local(config: ModelConfig, system: str, user_content: str, max_tokens: int) -> tuple[str, dict | None]:
    import openai
    client = _get_local_client(config)
    model = config.model_name or _discover_local_model(client, config)
    effective_max = _local_effective_max_tokens(config, system, user_content, max_tokens)

    def _messages(merge_system: bool) -> list[dict]:
        if not system:
            return [{"role": "user", "content": user_content}]
        if merge_system:
            # Fallback for templates with no system role: same instructions,
            # prepended to the user turn - the common convention those
            # models were themselves trained with.
            return [{"role": "user", "content": f"{system}\n\n{user_content}"}]
        return [{"role": "system", "content": system}, {"role": "user", "content": user_content}]

    merge = config.system_role is False or (config.base_url, model) in _local_no_system_role
    extra = {} if config.temperature is None else {"temperature": config.temperature}

    def _call(merge_system: bool):
        return client.chat.completions.create(
            model=model, messages=_messages(merge_system), max_tokens=effective_max,
            timeout=config.timeout, **extra,
        )

    try:
        try:
            response = _call(merge)
        except openai.BadRequestError as e:
            if merge or not system or config.system_role is True or not _looks_like_system_role_rejection(e):
                raise
            step_logger.log(f"llm_client: LOCAL model '{model}' rejected the system role ({e}) - retrying with "
                             f"the system prompt prepended to the user message, and will keep doing so for "
                             f"this model (set LOCAL_LLM_SYSTEM_ROLE=0 to skip the first attempt).")
            _local_no_system_role.add((config.base_url, model))
            response = _call(True)
    except openai.OpenAIError as e:
        if isinstance(e, openai.NotFoundError) and not config.model_name:
            _local_discovered_models.pop(config.base_url, None)   # server may have swapped models - rediscover next call
        raise _local_error(config, model, e) from e

    choices = getattr(response, "choices", None) or []
    if not choices:
        step_logger.log(f"llm_client: WARNING - LOCAL ({model}) returned no choices.")
        return "", None
    choice = choices[0]
    text = (getattr(choice.message, "content", None) or "")
    if config.strip_think:
        # Reasoning-style open models (Qwen3, DeepSeek-R1 distills) emit a
        # <think>...</think> preamble unless vLLM runs a reasoning parser -
        # every JSON-parsing caller here would choke on it.
        text = _THINK_RE.sub("", text, count=1)
    text = text.strip()
    if getattr(choice, "finish_reason", None) == "length":
        step_logger.log(f"llm_client: WARNING - LOCAL ({model}) response TRUNCATED (finish_reason='length', "
                         f"budget was {effective_max}) - likely incomplete/unparseable output.")

    # Usage is optional in the OpenAI-compatible contract (some Ollama/LM
    # Studio builds and proxies omit it). Leave it None rather than guess -
    # Langfuse and the call log both accept a missing usage.
    usage = None
    u = getattr(response, "usage", None)
    if u is not None:
        pt, ct = getattr(u, "prompt_tokens", None), getattr(u, "completion_tokens", None)
        if pt is not None or ct is not None:
            usage = {"input": pt, "output": ct, "total": getattr(u, "total_tokens", None) or (pt or 0) + (ct or 0)}
    return text, usage


def context_window_tokens(purpose: str = "ingestion", name: str | None = None) -> int | None:
    """2026-10-14: NEW - the configured context window for the provider that
    would serve this purpose/call, when known from config. Only LOCAL has one
    (LOCAL_LLM_CONTEXT_LENGTH); API providers return None so callers keep
    using their own MODEL_CONTEXT_WINDOWS tables. Intended for
    extract_generic.model_chunk_budget()/pipeline.llm_chunk_info(), which
    currently fall back to 100k tokens for any provider they don't know -
    far too large for a 7-14B model's window."""
    config = refresh_model_config(purpose, name=name)
    return config.context_length if config.use_model == "LOCAL" else None


def describe_call_model(purpose: str = "retrieval", name: str | None = None) -> dict:
    """2026-10-14 (phase0/followups): NEW - which provider/model WOULD serve
    this purpose + call name right now, for evaluation integrity (the
    /api/ask answer-cache key and the model_info block in responses and
    /api/settings/summary). Deliberately contains NO secrets: never the API
    key and never LOCAL_LLM_BASE_URL (a URL can embed credentials). A LOCAL
    model left blank (server-chosen, see _discover_local_model()) reports
    the model already discovered for that endpoint when there is one, else
    "" with model_source="server_default" - leave unset rather than guess."""
    config = refresh_model_config(purpose, name=name)
    model, source = config.model_name, "configured"
    if config.use_model == "LOCAL" and not model:
        model = _local_discovered_models.get(config.base_url or "", "")
        source = "discovered" if model else "server_default"
    return {"provider": config.use_model, "model": model, "model_source": source, "route": config.route}


def model_identity_key(purpose: str = "retrieval", name: str | None = None) -> str:
    """2026-10-14 (phase0/followups): NEW - a compact, stable string naming
    the model that would serve this call, for cache keys. Unlike
    describe_call_model() it is FIXED for a given configuration: a blank
    LOCAL model name is identified by a short hash of the endpoint URL
    (never the URL itself) instead of the discovered model, which only
    becomes known after the first call - so a cache read key computed
    before generation always equals the write key computed after it."""
    config = refresh_model_config(purpose, name=name)
    model = config.model_name
    if config.use_model == "LOCAL" and not model:
        import hashlib
        model = "auto@" + hashlib.sha256((config.base_url or "").encode("utf-8")).hexdigest()[:10]
    return f"{config.use_model}/{model}"


# 2026-10-14 (phase0/followups): chunk-group sizing for a LOCAL window. The
# ingestion converters count document text with tiktoken cl100k (~4 chars per
# token on English legal text), while _local_effective_max_tokens() estimates
# the prompt at _CHARS_PER_TOKEN_ESTIMATE (3.0) chars per token. If a group
# filled `window - reserves` tiktoken tokens, that estimate would come out ~33%
# larger than the window and clamp the completion to 64 tokens (truncated,
# unparseable JSON). So the text budget is scaled into the same conservative
# units the clamp uses - errs on smaller groups (more calls), never overflow.
_TIKTOKEN_CHARS_PER_TOKEN = 4.0


def local_text_token_budget(window: int, reserve_system: int, reserve_output: int) -> int:
    """How many tiktoken-counted tokens of document text fit in one call to a
    LOCAL model with this context window, after reserving room for the
    system prompt, the completion and the chat-template overhead. Shared by
    extract_generic.model_chunk_budget() and pipeline.llm_chunk_info() so
    the UI shows exactly the budget ingestion uses. Floor of 256 tokens (a
    paragraph): below that the window is misconfigured and the per-call
    clamp/warning in _local_effective_max_tokens() takes over."""
    usable = window - reserve_system - reserve_output - _CHAT_TEMPLATE_OVERHEAD_TOKENS
    return max(int(usable * _CHARS_PER_TOKEN_ESTIMATE / _TIKTOKEN_CHARS_PER_TOKEN), 256)


_DISPATCH = {
    "CLAUDE": _complete_claude,
    "GEMINI": _complete_gemini,
    "GPT": _complete_gpt,
    "LOCAL": _complete_local,
}


def complete(system: str, user_content: str, max_tokens: int = 1000, *,
             trace=None, name: str = "llm_call", purpose: str = "retrieval",
             prompt_version: str | None = None) -> str:
    """Calls the currently configured provider/model and returns the
    completion text. `purpose` ("ingestion" or "retrieval") selects which of
    USE_MODEL_INGESTION/USE_MODEL_RETRIEVAL to resolve (falls back to plain
    USE_MODEL if that one's unset) - see refresh_model_config(). Ingestion
    call sites (extract_entities.py/extract_case_law.py/extract_generic.py)
    pass purpose="ingestion"; every retrieval call site (intent.py/
    graphrag.py/app.py's answer generation) uses the default. `trace`/`name`
    are optional - pass a Langfuse trace object (see framework.core.retrieve.
    observability.start_trace) from a caller that already has one (e.g.
    app.py's /api/ask) to nest this call's Langfuse generation under it;
    omit them (the default) and this call is still logged to Langfuse as its
    own standalone generation - either way, every call is observable, with
    zero required changes at any call site that doesn't need purpose/trace.

    THREAD-SAFETY: `config` is resolved ONCE here and threaded explicitly
    into _DISPATCH[...] and every _complete_* function - nothing below this
    point re-reads module globals for provider/model/timeout. This is what
    makes a single complete() call immune to another thread's concurrent
    refresh_model_config() call changing the active provider/model
    mid-request - see ModelConfig's docstring for the bug this fixes.

    2026-10-14: `name` now also selects a per-call route (LLM_ROUTE_<NAME>,
    see resolve_call_route()), and the optional `prompt_version` is recorded
    in Langfuse metadata and the training-data call log (LLM_CALL_LOG=1) so
    exported examples can be filtered to the current production prompt."""
    # pick up any .env edit since the last call - see module docstring
    config = refresh_model_config(purpose, name=name)
    t0 = time.time()
    step_logger.log(f"llm_client: -> calling {config.use_model} ({config.model_name or 'auto'}) [purpose={purpose}"
                     f"{', route=' + config.route if config.route else ''}], "
                     f"prompt={len(user_content)} chars, max_tokens={max_tokens}...")
    meta = {"provider": config.use_model, "purpose": purpose, "max_tokens": max_tokens}
    if config.route:
        meta["route"] = config.route
    if prompt_version:
        meta["prompt_version"] = prompt_version
    try:
        if config.use_model == "LOCAL" and not config.model_name:
            # Resolve the auto-discovered model name up front so Langfuse and
            # the call log record the real model id, not a blank.
            import dataclasses
            config = dataclasses.replace(config, model_name=_discover_local_model(_get_local_client(config), config))
        result, usage = _DISPATCH[config.use_model](config, system, user_content, max_tokens)
    except Exception as e:
        elapsed = time.time() - t0
        step_logger.log(f"llm_client: <- {config.use_model} FAILED after {elapsed:.1f}s - {type(e).__name__}: {e}")
        observability.log_generation(
            trace, name=name, model=config.model_name,
            input_data={"system": system, "user": user_content},
            output_data=f"ERROR: {type(e).__name__}: {e}",
            metadata={**meta, "elapsed_s": round(elapsed, 2), "error": True},
        )
        llm_call_log.record_call(
            name=name, purpose=purpose, provider=config.use_model, model=config.model_name,
            route=config.route, prompt_version=prompt_version, system=system, user=user_content,
            output=None, error=f"{type(e).__name__}: {e}", latency_s=elapsed, usage=None,
            max_tokens=max_tokens, temperature=config.temperature,
        )
        raise
    elapsed = time.time() - t0
    step_logger.log(f"llm_client: <- {config.use_model} responded in {elapsed:.1f}s ({len(result)} chars)")
    observability.log_generation(
        trace, name=name, model=config.model_name,
        input_data={"system": system, "user": user_content},
        output_data=result,
        usage=usage,
        metadata={**meta, "elapsed_s": round(elapsed, 2)},
    )
    llm_call_log.record_call(
        name=name, purpose=purpose, provider=config.use_model, model=config.model_name,
        route=config.route, prompt_version=prompt_version, system=system, user=user_content,
        output=result, error=None, latency_s=elapsed, usage=usage,
        max_tokens=max_tokens, temperature=config.temperature,
    )
    return result
