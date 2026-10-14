"""
Opt-in training-data capture for every llm_client.complete() call.

2026-10-14: NEW - Phase 0 of the "Tax SLM" product. The fine-tuned local
model needs (system prompt, user content, good answer) examples per call type
(nl2cypher, intent_detection, answer_generation, ...), and the cheapest
source of those is the production traffic the API models already handle.
With LLM_CALL_LOG=1, llm_client.complete() appends one JSON line per call to
<LLM_CALL_LOG_DIR>/llm_calls_<UTC date>.jsonl (default data/llm_calls/,
which .gitignore already excludes via data/). framework/cli/
export_training_data.py turns those logs into chat-format fine-tuning JSONL.

  *** WARNING - CLIENT DATA ***
  These logs contain full prompts and answers: user questions, uploaded
  client documents, retrieved case law and generated opinions. That is why
  capture is OFF by default and must be switched on deliberately per
  environment. Never commit, email or copy these files outside the
  customer's environment. The domain PII masker's regex detectors (PAN,
  Aadhaar, e-mail, phone for the tax pack) are applied before anything is
  written, but regex masking cannot catch names, addresses or business facts
  in free text - treat the files as confidential regardless.

PII masking details / why it's done this way:
  - The masker's RULES come from the active domain pack (FRAMEWORK_DOMAIN),
    loaded lazily via framework.cli.pack_loader at first use, NOT at import.
    Importing the pack at module import time would be circular (pack ->
    pipeline/converters -> llm_client -> this module), which is exactly the
    kind of import-time failure tests/test_smoke_imports.py guards against.
  - Only the regex detectors + pii.mask_value() are used, not
    PiiMasker.mask_free_text(): the latter writes a PiiAccessLog node to
    Neo4j for every match, which would put a Neo4j round-trip (or a 10s
    connect timeout when Neo4j is down) inside every LLM call. Tokens are the
    same HMAC tokens the rest of the app produces ([PAN_REDACTED_xxxx]).
  - PiiMasker's always-masked FIELDS (CaseLaw.case_name etc.) are node-
    property rules and have no meaning for free text, so they don't apply.
  - FAIL CLOSED: if the masker can't be loaded, the record is NOT written
    (a warning is logged once) unless LLM_CALL_LOG_PII_MASK=0 explicitly
    disables masking (e.g. for synthetic/public test data).

Logging must never break or slow an LLM call materially: every failure in
here is caught and logged, never raised.
"""
import contextvars
import hashlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from framework.core.infra import step_logger

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LOG_DIR = _REPO_ROOT / "data" / "llm_calls"
SCHEMA_VERSION = 1

_write_lock = threading.Lock()
_masker_lock = threading.Lock()
_detectors_cache: dict[str, tuple] = {}     # domain name -> regex detectors
_warned: set[str] = set()
# The id of the most recent call recorded on THIS thread/context - lets a
# caller that later scores an answer (e.g. answer_quality_judge) attach the
# score to the right record via annotate(last_call_id(), judge_score=...).
_last_call_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("llm_last_call_id", default=None)


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def is_enabled() -> bool:
    """Read from os.environ on every call - llm_client.complete() has just
    re-read .env (refresh_model_config), so toggling LLM_CALL_LOG in .env is
    hot like every other LLM setting."""
    return _truthy(os.environ.get("LLM_CALL_LOG"))


def log_dir() -> Path:
    override = os.environ.get("LLM_CALL_LOG_DIR", "").strip()
    return Path(override) if override else DEFAULT_LOG_DIR


def _masking_enabled() -> bool:
    raw = os.environ.get("LLM_CALL_LOG_PII_MASK", "").strip()
    return True if not raw else _truthy(raw)


def _warn_once(key: str, msg: str) -> None:
    if key not in _warned:
        _warned.add(key)
        step_logger.log(msg)


def _load_detectors() -> tuple:
    """Returns the active domain pack's PII regex detectors; raises if they
    can't be loaded (caller fails closed)."""
    domain = os.environ.get("FRAMEWORK_DOMAIN", "tax").strip() or "tax"
    with _masker_lock:
        if domain in _detectors_cache:
            return _detectors_cache[domain]
        from framework.cli.pack_loader import load_domain_pack   # lazy - see module docstring
        try:
            domain_config, _prompt_pack = load_domain_pack(domain)
        except SystemExit as e:   # pack_loader reports a missing pack via SystemExit
            raise RuntimeError(str(e)) from e
        detectors = tuple(domain_config.pii_regex_detectors or ())
        _detectors_cache[domain] = detectors
        return detectors


def mask_text(text: str | None, detectors: tuple) -> tuple[str | None, list[str]]:
    if not text or not isinstance(text, str):
        return text, []
    from framework.core.retrieve.pii import mask_value
    found: list[str] = []
    for label, pattern in detectors:
        def _replace(m, label=label):
            found.append(label)
            return mask_value(m.group(0), label)
        text = pattern.sub(_replace, text)
    return text, found


def _append(record: dict) -> None:
    directory = log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    day = record["ts"][:10]
    path = directory / f"llm_calls_{day}.jsonl"
    line = json.dumps(record, ensure_ascii=False)
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def record_call(*, name: str, purpose: str, provider: str, model: str | None, route: str | None,
                prompt_version: str | None, system: str, user: str, output: str | None, error: str | None,
                latency_s: float, usage: dict | None, max_tokens: int | None,
                temperature: float | None) -> str | None:
    """Appends one call record if LLM_CALL_LOG is on. Returns the record's
    call_id (or None when disabled/skipped). Never raises."""
    if not is_enabled():
        return None
    try:
        pii_types: list[str] = []
        masked = False
        if _masking_enabled():
            try:
                detectors = _load_detectors()
            except Exception as e:
                _warn_once("masker", f"llm_call_log: WARNING - could not load the domain PII masker ({e}); "
                                     f"NOT writing LLM call logs (fail closed). Fix FRAMEWORK_DOMAIN, or set "
                                     f"LLM_CALL_LOG_PII_MASK=0 only for non-client data.")
                return None
            system, f1 = mask_text(system, detectors)
            user, f2 = mask_text(user, detectors)
            output, f3 = mask_text(output, detectors)
            error, _ = mask_text(error, detectors)
            pii_types = sorted(set(f1 + f2 + f3))
            masked = True
        call_id = uuid.uuid4().hex
        record = {
            "schema_version": SCHEMA_VERSION,
            "type": "call",
            "call_id": call_id,
            "ts": datetime.now(timezone.utc).isoformat(),
            "name": name,
            "purpose": purpose,
            "provider": provider,
            "model": model,
            "route": route,
            "prompt_version": prompt_version,
            "system_sha256": hashlib.sha256((system or "").encode("utf-8")).hexdigest()[:16],
            "system": system,
            "user": user,
            "output": output,
            "error": error,
            "latency_s": round(float(latency_s), 3),
            "usage": usage,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "pii_masked": masked,
            "pii_types_found": pii_types,
        }
        _append(record)
        _last_call_id.set(call_id)
        return call_id
    except Exception as e:
        _warn_once(f"write:{type(e).__name__}", f"llm_call_log: WARNING - failed to write call log ({type(e).__name__}: "
                                                  f"{e}); LLM call itself is unaffected.")
        return None


def last_call_id() -> str | None:
    return _last_call_id.get()


def annotate(call_id: str | None, **fields) -> bool:
    """Attaches extra fields (e.g. judge_score=0.85, human_label="good") to
    an earlier call record by appending a small {"type": "annotation"} line -
    the log is append-only, so records are never rewritten. The export CLI
    merges annotations into their call before filtering (--min-judge-score).
    No-op (False) when logging is off or call_id is None."""
    if not call_id or not is_enabled():
        return False
    try:
        _append({"schema_version": SCHEMA_VERSION, "type": "annotation", "call_id": call_id,
                 "ts": datetime.now(timezone.utc).isoformat(), **fields})
        return True
    except Exception as e:
        _warn_once("annotate", f"llm_call_log: WARNING - failed to write annotation ({e}).")
        return False
