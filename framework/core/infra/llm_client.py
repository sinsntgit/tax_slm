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
"""
import os
import threading
import time
from dataclasses import dataclass

from dotenv import load_dotenv

from framework.core.infra import step_logger
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


def refresh_model_config(purpose: str = "retrieval") -> ModelConfig:
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

    purpose_key = PURPOSE_ENV_KEYS.get(purpose)
    purpose_override = os.environ.get(purpose_key, "").strip().upper() if purpose_key else ""
    use_model = purpose_override or os.environ.get("USE_MODEL", "CLAUDE").strip().upper()
    if use_model not in MODEL_NAME_DEFAULTS:
        raise ValueError(
            f"USE_MODEL{'' if not purpose_override else '_' + purpose.upper()}='{use_model}' in .env is "
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
        os.environ.get(f"{use_model}_MODEL_NAME_{purpose.upper()}", "").strip()
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

    config = ModelConfig(use_model=use_model, model_name=model_name, timeout=timeout, temperature=temperature,
                          reasoning_effort=reasoning_effort)

    global USE_MODEL, MODEL_NAME, LLM_TIMEOUT_SECONDS, TEMPERATURE
    changed = (use_model != globals().get("USE_MODEL") or model_name != globals().get("MODEL_NAME")
               or temperature != globals().get("TEMPERATURE"))
    # Back-compat globals only - see docstring. Note these ARE still racy
    # under concurrent calls, same as before; the fix is that nothing in
    # THIS file reads them anymore (everything threads the returned
    # ModelConfig explicitly instead), so the race can no longer corrupt an
    # in-flight request. Left in place only in case other modules import
    # llm_client.MODEL_NAME/.USE_MODEL directly for logging/display.
    USE_MODEL, MODEL_NAME, LLM_TIMEOUT_SECONDS, TEMPERATURE = use_model, model_name, timeout, temperature
    if changed:
        step_logger.log(f"llm_client: config (re)loaded for purpose='{purpose}' - "
                         f"USE_MODEL={use_model}, MODEL_NAME={model_name}, timeout={timeout:.0f}s, "
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
GPT_REASONING_EFFORT_DEFAULT = os.environ.get("GPT_REASONING_EFFORT", "minimal")
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


_DISPATCH = {
    "CLAUDE": _complete_claude,
    "GEMINI": _complete_gemini,
    "GPT": _complete_gpt,
}


def complete(system: str, user_content: str, max_tokens: int = 1000, *,
             trace=None, name: str = "llm_call", purpose: str = "retrieval") -> str:
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
    mid-request - see ModelConfig's docstring for the bug this fixes."""
    config = refresh_model_config(purpose)  # pick up any .env edit since the last call - see module docstring
    t0 = time.time()
    step_logger.log(f"llm_client: -> calling {config.use_model} ({config.model_name}) [purpose={purpose}], "
                     f"prompt={len(user_content)} chars, max_tokens={max_tokens}...")
    try:
        result, usage = _DISPATCH[config.use_model](config, system, user_content, max_tokens)
    except Exception as e:
        elapsed = time.time() - t0
        step_logger.log(f"llm_client: <- {config.use_model} FAILED after {elapsed:.1f}s - {type(e).__name__}: {e}")
        observability.log_generation(
            trace, name=name, model=config.model_name,
            input_data={"system": system, "user": user_content},
            output_data=f"ERROR: {type(e).__name__}: {e}",
            metadata={"provider": config.use_model, "purpose": purpose, "max_tokens": max_tokens,
                      "elapsed_s": round(elapsed, 2), "error": True},
        )
        raise
    elapsed = time.time() - t0
    step_logger.log(f"llm_client: <- {config.use_model} responded in {elapsed:.1f}s ({len(result)} chars)")
    observability.log_generation(
        trace, name=name, model=config.model_name,
        input_data={"system": system, "user": user_content},
        output_data=result,
        usage=usage,
        metadata={"provider": config.use_model, "purpose": purpose, "max_tokens": max_tokens,
                  "elapsed_s": round(elapsed, 2)},
    )
    return result
