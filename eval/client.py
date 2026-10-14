"""
Minimal JSON-over-HTTP client with retries, for the eval runner.

2026-10-14: NEW. stdlib urllib (not requests/httpx) so the harness has no
dependency the app itself doesn't already pin. Classifies every failure so
the run report can tell "the product answered badly" apart from "the call
never completed":

  timeout            - no complete response within --timeout seconds
  connection_error   - refused/reset/DNS (server down, Docker not up)
  http_error         - non-2xx status (body kept - /api/ask returns {"error": ...})
  malformed_response - 2xx but not JSON, or JSON without the expected key

Retried (with exponential backoff): connection_error, HTTP 429/500/502/503/504
(/api/ask turns ANY pipeline exception - LLM provider hiccup, Neo4j blip -
into a 500, so a 500 is treated as transient), and timeout only when
retry_on_timeout=True (a timed-out research call can cost minutes; retrying
it by default would make one slow item stall the whole run). 4xx other than
429 is a request bug - never retried.
"""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

RETRYABLE_HTTP = {429, 500, 502, 503, 504}


@dataclass
class CallResult:
    status: str                      # "ok" | "error"
    http_status: int | None = None
    latency_s: float | None = None   # wall-clock of the FINAL attempt
    attempts: int = 0
    response: dict | None = None
    raw_text: str | None = None
    error: dict | None = None
    attempt_log: list = field(default_factory=list)


def _one_call(url: str, body: dict | None, timeout: float, method: str = "POST") -> tuple:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", "Accept": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return "done", resp.status, raw, time.perf_counter() - started
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode("utf-8", errors="replace")
        except Exception:
            raw = ""
        return "done", e.code, raw, time.perf_counter() - started
    except (TimeoutError, socket.timeout):
        return "timeout", None, None, time.perf_counter() - started
    except urllib.error.URLError as e:
        if isinstance(e.reason, (TimeoutError, socket.timeout)):
            return "timeout", None, None, time.perf_counter() - started
        return "connection_error", None, str(e.reason), time.perf_counter() - started
    except (ConnectionError, OSError) as e:
        return "connection_error", None, str(e), time.perf_counter() - started


def call_json(url: str, body: dict | None, *, timeout: float = 300.0, retries: int = 2,
              backoff_s: float = 2.0, retry_on_timeout: bool = False, expect_key: str | None = None,
              method: str = "POST", sleep=time.sleep) -> CallResult:
    attempts_allowed = 1 + max(0, retries)
    result = CallResult(status="error")
    for attempt in range(1, attempts_allowed + 1):
        outcome, http_status, raw, elapsed = _one_call(url, body, timeout, method)
        result.attempts, result.latency_s, result.http_status = attempt, round(elapsed, 3), http_status
        retryable = False
        if outcome == "timeout":
            result.error = {"type": "timeout", "message": f"no response within {timeout}s"}
            retryable = retry_on_timeout
        elif outcome == "connection_error":
            result.error = {"type": "connection_error", "message": raw}
            retryable = True
        elif not (200 <= http_status < 300):
            msg = raw
            try:
                msg = json.loads(raw).get("error", raw)
            except Exception:
                pass
            result.error = {"type": "http_error", "http_status": http_status, "message": str(msg)[:2000]}
            result.raw_text = raw[:20000] if raw else None
            retryable = http_status in RETRYABLE_HTTP
        else:
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                parsed = None
            if not isinstance(parsed, dict) or (expect_key and expect_key not in parsed):
                result.error = {"type": "malformed_response",
                                "message": f"2xx response was not a JSON object"
                                           + (f" with '{expect_key}'" if expect_key else "")}
                result.raw_text = (raw or "")[:20000]
                retryable = False
            else:
                result.status, result.response, result.error, result.raw_text = "ok", parsed, None, None
        result.attempt_log.append({"attempt": attempt, "latency_s": round(elapsed, 3),
                                   "outcome": "ok" if result.status == "ok" else result.error["type"],
                                   "http_status": http_status})
        if result.status == "ok" or not retryable or attempt == attempts_allowed:
            break
        sleep(backoff_s * (2 ** (attempt - 1)))
    return result
