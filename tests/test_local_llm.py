"""2026-10-14: NEW - LOCAL (OpenAI-compatible) provider, per-call routing,
hot reload, training-data call log and the export CLI. The LOCAL provider is
tested end to end over real HTTP against a tiny fake OpenAI-compatible server
(http.server in a thread) - no real model, no network, no real .env."""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from framework.core.infra import llm_call_log, llm_client
from framework.core.retrieve import observability


# --- fake OpenAI-compatible server -------------------------------------------

class FakeServer:
    def __init__(self):
        self.mode = "ok"            # ok | no_usage | slow | error500 | no_system_role | think
        self.models = ["tax-slm-7b"]
        self.requests: list[dict] = []
        self.reply = "Section 194J applies."
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def _send(self, code, payload):
                body = json.dumps(payload).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.endswith("/models"):
                    self._send(200, {"object": "list", "data": [{"id": m, "object": "model", "created": 0,
                                                                  "owned_by": "x"} for m in server.models]})
                else:
                    self._send(404, {"error": {"message": "not found"}})

            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                req = json.loads(self.rfile.read(length) or b"{}")
                req["_auth"] = self.headers.get("Authorization")
                server.requests.append(req)
                if not self.path.endswith("/chat/completions"):
                    return self._send(404, {"error": {"message": "not found"}})
                if req.get("model") not in server.models:
                    return self._send(404, {"error": {"message": f"The model `{req.get('model')}` does not exist."}})
                if server.mode == "slow":
                    time.sleep(3)
                if server.mode == "error500":
                    return self._send(500, {"error": {"message": "CUDA out of memory"}})
                if server.mode == "no_system_role" and any(m["role"] == "system" for m in req["messages"]):
                    return self._send(400, {"error": {"message": "System role not supported", "type": "BadRequestError"}})
                content = server.reply
                if server.mode == "think":
                    content = "<think>let me reason about TDS</think>\n" + server.reply
                payload = {
                    "id": "cmpl-1", "object": "chat.completion", "created": 0, "model": req["model"],
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                                 "finish_reason": "stop"}],
                }
                if server.mode != "no_usage":
                    payload["usage"] = {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}
                self._send(200, payload)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def fake_server():
    s = FakeServer()
    yield s
    s.close()


_PREFIXES = ("USE_MODEL", "LLM_ROUTE_", "LOCAL_", "LLM_CALL_LOG", "LLM_TIMEOUT", "CLAUDE_", "GPT_", "GEMINI_")


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated .env: refresh_model_config() re-reads ENV_FILE_PATH with
    override=True on every call, which writes straight into os.environ - so
    snapshot and restore the whole environment rather than relying on
    monkeypatch alone. Never touches the real .env."""
    saved = dict(os.environ)
    for k in list(os.environ):
        if k.startswith(_PREFIXES):
            del os.environ[k]
    env_file = tmp_path / "test.env"
    env_file.write_text("")
    os.environ["ENV_FILE_PATH"] = str(env_file)
    os.environ["WEBAPP_SETTINGS_PATH"] = str(tmp_path / "settings.json")   # missing file -> defaults
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    llm_client._local_discovered_models.clear()
    llm_client._local_no_system_role.clear()
    generations = []
    monkeypatch.setattr(observability, "log_generation",
                        lambda trace, name, model, input_data, output_data, **kw:
                        generations.append({"name": name, "model": model, "output": output_data, **kw}))

    class Env:
        path = env_file
        gens = generations

        def write(self, **values):
            env_file.write_text("".join(f"{k}={v}\n" for k, v in values.items()))

    yield Env()
    os.environ.clear()
    os.environ.update(saved)


def _local_env(server, **extra):
    values = dict(USE_MODEL="LOCAL", LOCAL_LLM_BASE_URL=server.base_url, LOCAL_MODEL_NAME="tax-slm-7b",
                  LOCAL_LLM_MAX_RETRIES="0")
    values.update(extra)
    return values


# --- LOCAL provider over real HTTP -------------------------------------------

def test_local_success_with_usage(env, fake_server):
    env.write(**_local_env(fake_server))
    out = llm_client.complete("You are a tax assistant.", "Which TDS section?", max_tokens=50, name="nl2cypher")
    assert out == "Section 194J applies."
    req = fake_server.requests[-1]
    assert req["model"] == "tax-slm-7b"
    assert req["messages"][0] == {"role": "system", "content": "You are a tax assistant."}
    assert req["max_tokens"] == 50
    assert req["_auth"] == "Bearer not-needed"     # never falls back to OPENAI_API_KEY
    g = env.gens[-1]
    assert g["usage"] == {"input": 11, "output": 5, "total": 16}
    assert g["model"] == "tax-slm-7b" and g["metadata"]["provider"] == "LOCAL"


def test_local_api_key_and_openai_key_not_leaked(env, fake_server):
    os.environ["OPENAI_API_KEY"] = "sk-real-openai-key"
    env.write(**_local_env(fake_server))
    llm_client.complete("s", "u", name="x")
    assert fake_server.requests[-1]["_auth"] == "Bearer not-needed"
    env.write(**_local_env(fake_server, LOCAL_LLM_API_KEY="local-secret"))
    llm_client.complete("s", "u", name="x")
    assert fake_server.requests[-1]["_auth"] == "Bearer local-secret"
    assert "local-secret" not in repr(llm_client.refresh_model_config())


def test_local_no_usage_response(env, fake_server):
    fake_server.mode = "no_usage"
    env.write(**_local_env(fake_server))
    assert llm_client.complete("s", "u", name="intent_detection") == "Section 194J applies."
    assert env.gens[-1]["usage"] is None


def test_local_timeout_raises_clear_error(env, fake_server):
    fake_server.mode = "slow"
    env.write(**_local_env(fake_server, LLM_TIMEOUT_SECONDS="1"))
    t0 = time.time()
    with pytest.raises(llm_client.LocalLLMError, match="did not respond within 1s"):
        llm_client.complete("s", "u", name="answer_generation")
    assert time.time() - t0 < 2.9            # no retries with LOCAL_LLM_MAX_RETRIES=0
    assert env.gens[-1]["metadata"]["error"] is True


def test_local_http_error_raises_clear_error(env, fake_server):
    fake_server.mode = "error500"
    env.write(**_local_env(fake_server))
    with pytest.raises(llm_client.LocalLLMError, match=r"HTTP 500.*CUDA out of memory"):
        llm_client.complete("s", "u")


def test_local_wrong_model_name_404_hint(env, fake_server):
    env.write(**_local_env(fake_server, LOCAL_MODEL_NAME="wrong-name"))
    with pytest.raises(llm_client.LocalLLMError, match="served-model-name"):
        llm_client.complete("s", "u")


def test_local_connection_refused(env):
    env.write(USE_MODEL="LOCAL", LOCAL_LLM_BASE_URL="http://127.0.0.1:9/v1", LOCAL_MODEL_NAME="m",
              LOCAL_LLM_MAX_RETRIES="0")
    with pytest.raises(llm_client.LocalLLMError, match="Could not connect"):
        llm_client.complete("s", "u")


def test_local_missing_base_url(env):
    env.write(USE_MODEL="LOCAL")
    with pytest.raises(RuntimeError, match="LOCAL_LLM_BASE_URL is not set"):
        llm_client.complete("s", "u")


def test_local_model_autodiscovery(env, fake_server):
    env.write(USE_MODEL="LOCAL", LOCAL_LLM_BASE_URL=fake_server.base_url, LOCAL_LLM_MAX_RETRIES="0")
    assert llm_client.complete("s", "u") == "Section 194J applies."
    assert fake_server.requests[-1]["model"] == "tax-slm-7b"
    assert env.gens[-1]["model"] == "tax-slm-7b"


def test_local_system_role_fallback(env, fake_server):
    fake_server.mode = "no_system_role"
    env.write(**_local_env(fake_server))
    assert llm_client.complete("SYS", "USER") == "Section 194J applies."
    assert fake_server.requests[-1]["messages"] == [{"role": "user", "content": "SYS\n\nUSER"}]
    n = len(fake_server.requests)
    llm_client.complete("SYS", "USER")            # remembered: no wasted system-role attempt
    assert len(fake_server.requests) == n + 1


def test_local_system_role_forced_off(env, fake_server):
    env.write(**_local_env(fake_server, LOCAL_LLM_SYSTEM_ROLE="0"))
    llm_client.complete("SYS", "USER")
    assert [m["role"] for m in fake_server.requests[-1]["messages"]] == ["user"]


def test_local_strips_think_block(env, fake_server):
    fake_server.mode = "think"
    env.write(**_local_env(fake_server))
    assert llm_client.complete("s", "u") == "Section 194J applies."


def test_local_max_tokens_cap_and_context_length(env, fake_server):
    env.write(**_local_env(fake_server, LOCAL_LLM_MAX_TOKENS="300"))
    llm_client.complete("s", "u", max_tokens=5000)
    assert fake_server.requests[-1]["max_tokens"] == 300
    env.write(**_local_env(fake_server, LOCAL_LLM_CONTEXT_LENGTH="2000"))
    llm_client.complete("s", "x" * 3000, max_tokens=5000)     # ~1000 est. prompt tokens + overhead
    sent = fake_server.requests[-1]["max_tokens"]
    assert 64 <= sent < 1000
    assert llm_client.context_window_tokens("ingestion") == 2000


def test_local_temperature_from_settings(env, fake_server, tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"temperature": 0.2}))
    env.write(**_local_env(fake_server))
    llm_client.complete("s", "u")
    assert fake_server.requests[-1]["temperature"] == 0.2


# --- routing precedence + hot reload ------------------------------------------

def test_routing_precedence(env):
    env.write(USE_MODEL="CLAUDE", USE_MODEL_RETRIEVAL="GPT", LLM_ROUTE_NL2CYPHER="LOCAL",
              LOCAL_LLM_BASE_URL="http://x/v1", LLM_ROUTE_NL2CYPHER_MODEL="nl2cypher-lora")
    c = llm_client.refresh_model_config("retrieval", name="nl2cypher")
    assert (c.use_model, c.model_name, c.route) == ("LOCAL", "nl2cypher-lora", "LLM_ROUTE_NL2CYPHER")
    c = llm_client.refresh_model_config("retrieval", name="intent_detection")
    assert (c.use_model, c.route) == ("GPT", None)                    # purpose override
    c = llm_client.refresh_model_config("ingestion", name="temporal_metadata")
    assert c.use_model == "CLAUDE"                                    # global
    c = llm_client.refresh_model_config("retrieval")                  # no name: old behaviour
    assert c.use_model == "GPT"


def test_routing_prefix_match_for_dynamic_names(env):
    env.write(USE_MODEL="CLAUDE", LLM_ROUTE_CASE_LAW_EXTRACTION="LOCAL", LOCAL_LLM_BASE_URL="http://x/v1",
              LLM_ROUTE_SEMANTIC_LAYER_GROUP="GPT")
    assert llm_client.refresh_model_config("ingestion", name="case_law_extraction_ITA_123_2024").use_model == "LOCAL"
    assert llm_client.refresh_model_config("ingestion", name="semantic_layer_group_3_of_12").use_model == "GPT"
    assert llm_client.refresh_model_config("retrieval", name="case_law_classification").use_model == "CLAUDE"


def test_route_invalid_provider_names_the_key(env):
    env.write(LLM_ROUTE_NL2CYPHER="LLAMA")
    with pytest.raises(ValueError, match="LLM_ROUTE_NL2CYPHER='LLAMA'"):
        llm_client.refresh_model_config("retrieval", name="nl2cypher")


def test_hot_reload_route_and_endpoint(env, fake_server):
    env.write(USE_MODEL="CLAUDE")
    assert llm_client.refresh_model_config(name="nl2cypher").use_model == "CLAUDE"
    env.write(USE_MODEL="CLAUDE", LLM_ROUTE_NL2CYPHER="LOCAL", LOCAL_LLM_BASE_URL=fake_server.base_url,
              LOCAL_MODEL_NAME="tax-slm-7b", LOCAL_LLM_MAX_RETRIES="0")
    assert llm_client.complete("s", "u", name="nl2cypher") == "Section 194J applies."   # no restart
    second = FakeServer()
    try:
        second.reply = "from second server"
        env.write(USE_MODEL="CLAUDE", LLM_ROUTE_NL2CYPHER="LOCAL", LOCAL_LLM_BASE_URL=second.base_url,
                  LOCAL_MODEL_NAME="tax-slm-7b", LOCAL_LLM_MAX_RETRIES="0")
        assert llm_client.complete("s", "u", name="nl2cypher") == "from second server"
    finally:
        second.close()
    env.write(USE_MODEL="CLAUDE", LLM_ROUTE_NL2CYPHER="")             # blank un-routes
    assert llm_client.refresh_model_config(name="nl2cypher").use_model == "CLAUDE"


# --- training-data call log ---------------------------------------------------

def _read_log(directory):
    lines = []
    for p in sorted(directory.glob("llm_calls_*.jsonl")):
        lines += [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    return lines


def test_call_log_off_by_default(env, fake_server, tmp_path):
    os.environ["LLM_CALL_LOG_DIR"] = str(tmp_path / "calls")
    env.write(**_local_env(fake_server))
    llm_client.complete("s", "u", name="nl2cypher")
    assert not (tmp_path / "calls").exists()


def test_call_log_on_masks_pii_and_records_fields(env, fake_server, tmp_path):
    log_dir = tmp_path / "calls"
    env.write(**_local_env(fake_server, LLM_CALL_LOG="1", LLM_CALL_LOG_DIR=str(log_dir)))
    llm_client.complete("You are a tax assistant.", "Assessee PAN ABCDE1234F, email a.b@example.com - TDS?",
                        max_tokens=50, name="answer_generation", prompt_version="v3")
    recs = _read_log(log_dir)
    assert len(recs) == 1
    r = recs[0]
    assert r["type"] == "call" and r["name"] == "answer_generation" and r["purpose"] == "retrieval"
    assert r["provider"] == "LOCAL" and r["model"] == "tax-slm-7b" and r["prompt_version"] == "v3"
    assert r["output"] == "Section 194J applies." and r["error"] is None
    assert r["usage"] == {"input": 11, "output": 5, "total": 16} and r["latency_s"] >= 0
    assert "ABCDE1234F" not in r["user"] and "a.b@example.com" not in r["user"]
    assert "[PAN_REDACTED_" in r["user"] and r["pii_masked"] is True
    assert set(r["pii_types_found"]) == {"PAN", "EMAIL"}
    assert llm_call_log.last_call_id() == r["call_id"]


def test_call_log_records_errors(env, fake_server, tmp_path):
    fake_server.mode = "error500"
    log_dir = tmp_path / "calls"
    env.write(**_local_env(fake_server, LLM_CALL_LOG="1", LLM_CALL_LOG_DIR=str(log_dir)))
    with pytest.raises(llm_client.LocalLLMError):
        llm_client.complete("s", "u", name="nl2cypher")
    r = _read_log(log_dir)[0]
    assert r["output"] is None and "HTTP 500" in r["error"]


def test_call_log_fails_closed_without_masker(env, fake_server, tmp_path, monkeypatch):
    log_dir = tmp_path / "calls"
    monkeypatch.setattr(llm_call_log, "_load_detectors", lambda: (_ for _ in ()).throw(RuntimeError("no pack")))
    env.write(**_local_env(fake_server, LLM_CALL_LOG="1", LLM_CALL_LOG_DIR=str(log_dir)))
    assert llm_client.complete("s", "u") == "Section 194J applies."   # the call itself still works
    assert not log_dir.exists() or _read_log(log_dir) == []
    env.write(**_local_env(fake_server, LLM_CALL_LOG="1", LLM_CALL_LOG_DIR=str(log_dir), LLM_CALL_LOG_PII_MASK="0"))
    llm_client.complete("s", "u")
    assert _read_log(log_dir)[-1]["pii_masked"] is False


def test_call_log_hot_toggle(env, fake_server, tmp_path):
    log_dir = tmp_path / "calls"
    env.write(**_local_env(fake_server, LLM_CALL_LOG_DIR=str(log_dir)))
    llm_client.complete("s", "u1")
    env.write(**_local_env(fake_server, LLM_CALL_LOG="1", LLM_CALL_LOG_DIR=str(log_dir)))
    llm_client.complete("s", "u2")
    env.write(**_local_env(fake_server, LLM_CALL_LOG="0", LLM_CALL_LOG_DIR=str(log_dir)))
    llm_client.complete("s", "u3")
    assert [r["user"] for r in _read_log(log_dir)] == ["u2"]


# --- export CLI ---------------------------------------------------------------

def _write_records(directory, records):
    directory.mkdir(parents=True, exist_ok=True)
    by_day = {}
    for r in records:
        by_day.setdefault(r["ts"][:10], []).append(r)
    for day, rs in by_day.items():
        with open(directory / f"llm_calls_{day}.jsonl", "a", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps(r) + "\n")
        # plus one corrupt line - must be skipped, not crash the export
        with open(directory / f"llm_calls_{day}.jsonl", "a", encoding="utf-8") as f:
            f.write("{not json\n")


def _rec(cid, name, ts, user="q", output="a", system="SYS", **kw):
    base = dict(type="call", call_id=cid, ts=ts, name=name, purpose="retrieval", provider="CLAUDE",
                model="m", system=system, user=user, output=output, error=None, pii_masked=True)
    base.update(kw)
    return base


def test_export_cli(tmp_path, capsys):
    from framework.cli import export_training_data as cli
    logs = tmp_path / "logs"
    _write_records(logs, [
        _rec("1", "nl2cypher", "2026-10-01T10:00:00+00:00", user="q1", output="MATCH (n) RETURN n"),
        _rec("2", "nl2cypher", "2026-10-02T10:00:00+00:00", user="q1", output="MATCH (s) RETURN s"),  # dup prompt, newer
        _rec("3", "nl2cypher", "2026-10-03T10:00:00+00:00", user="q2", output=""),                    # empty output
        _rec("4", "intent_detection", "2026-10-02T11:00:00+00:00", output="LAW"),
        _rec("5", "case_law_extraction_ITA_1_2024", "2026-10-02T12:00:00+00:00", purpose="ingestion", output="{}"),
        _rec("6", "case_law_extraction_ITA_2_2024", "2026-10-02T12:00:00+00:00", purpose="ingestion",
             user="other", output="{}"),
        _rec("7", "answer_generation", "2026-09-15T10:00:00+00:00", output="old"),                     # before --since
        _rec("8", "answer_generation", "2026-10-05T10:00:00+00:00", user="g1", output="good"),
        _rec("9", "answer_generation", "2026-10-05T11:00:00+00:00", user="g2", output="bad"),
        _rec("10", "answer_generation", "2026-10-06T11:00:00+00:00", user="g3", output="raw", pii_masked=False),
        _rec("11", "nl2cypher", "2026-10-04T10:00:00+00:00", user="q9", error="HTTP 500", output=None),
    ])
    with open(logs / "llm_calls_2026-10-05.jsonl", "a") as f:
        f.write(json.dumps({"type": "annotation", "call_id": "8", "ts": "x", "judge_score": 0.9}) + "\n")
        f.write(json.dumps({"type": "annotation", "call_id": "9", "ts": "x", "judge_score": 0.3}) + "\n")

    out = tmp_path / "out"
    rc = cli.main(["--log-dir", str(logs), "--out", str(out), "--since", "2026-10-01", "--until", "2026-10-31"])
    assert rc == 0
    files = sorted(p.name for p in out.glob("*.jsonl"))
    assert files == ["answer_generation.jsonl", "case_law_extraction.jsonl", "intent_detection.jsonl", "nl2cypher.jsonl"]
    nl = [json.loads(line) for line in (out / "nl2cypher.jsonl").read_text().splitlines()]
    assert len(nl) == 1                                                  # deduped, empty + error dropped
    assert nl[0]["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "q1"},
                                 {"role": "assistant", "content": "MATCH (s) RETURN s"}]   # newest kept
    assert len((out / "case_law_extraction.jsonl").read_text().splitlines()) == 2      # family grouping
    ans = [json.loads(line) for line in (out / "answer_generation.jsonl").read_text().splitlines()]
    assert [a["messages"][-1]["content"] for a in ans] == ["good", "bad"]   # old (date) + unmasked excluded

    out2 = tmp_path / "out2"
    cli.main(["--log-dir", str(logs), "--out", str(out2), "--call", "answer_generation", "--min-judge-score", "0.8",
              "--with-metadata", "--merge-system"])
    rows = [json.loads(line) for line in (out2 / "answer_generation.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["metadata"]["judge_score"] == 0.9
    assert rows[0]["messages"] == [{"role": "user", "content": "SYS\n\ng1"}, {"role": "assistant", "content": "good"}]
    assert sorted(p.name for p in out2.glob("*.jsonl")) == ["answer_generation.jsonl"]

    out3 = tmp_path / "out3"
    cli.main(["--log-dir", str(logs), "--out", str(out3), "--call", "case_law_extraction", "--purpose", "ingestion"])
    assert len((out3 / "case_law_extraction.jsonl").read_text().splitlines()) == 2


def test_annotate_roundtrip_into_export(env, fake_server, tmp_path):
    from framework.cli import export_training_data as cli
    log_dir = tmp_path / "calls"
    env.write(**_local_env(fake_server, LLM_CALL_LOG="1", LLM_CALL_LOG_DIR=str(log_dir)))
    llm_client.complete("s", "u", name="answer_generation")
    assert llm_call_log.annotate(llm_call_log.last_call_id(), judge_score=0.95)
    stats = cli.export(log_dir, tmp_path / "o", min_judge_score=0.9)
    assert stats["written"] == {"answer_generation": 1}
