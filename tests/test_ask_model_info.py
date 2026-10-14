"""
2026-10-14: NEW (phase0/followups) - evaluation-integrity API requests from the
eval engineer (eval/README.md):
  a. /api/ask accepts bypass_cache (no cache read, no cache write);
  b. the answer-cache key includes the answer model identity + answer prompt
     version, and the read key still equals the write key;
  c. responses and /api/settings/summary carry model_info, with no secrets;
  d. the judge calls (answer_quality_judge / answer_evaluation*) can be routed
     to a different model via LLM_ROUTE_<NAME>, and model_info reports it.
Drives the real Flask route and _api_ask_impl() with only the heavy edges faked
(same harness shape as tests/test_webapp_ask_cache.py). Isolated .env file -
the real .env is never read.
"""
import json
import os

import pytest

from framework.core.infra import llm_client

_PREFIXES = ("USE_MODEL", "LLM_ROUTE_", "LOCAL_", "CLAUDE_", "GPT_", "GEMINI_")
SECRET = "sk-test-SECRET-value-123"


@pytest.fixture(scope="module")
def app_module():
    import framework.webapp.app as app_module
    return app_module


@pytest.fixture
def env(tmp_path):
    saved = dict(os.environ)
    for k in list(os.environ):
        if k.startswith(_PREFIXES):
            del os.environ[k]
    env_file = tmp_path / "test.env"
    env_file.write_text("")
    os.environ["ENV_FILE_PATH"] = str(env_file)
    os.environ["WEBAPP_SETTINGS_PATH"] = str(tmp_path / "settings.json")
    os.environ["PROMPT_CONFIG_PATH"] = str(tmp_path / "prompt_config.json")
    os.environ["RERANKER_CONFIG_PATH"] = str(tmp_path / "reranker_config.json")
    os.environ["COMPARATIVE_RESPONSE_CONFIG_PATH"] = str(tmp_path / "comparative.json")

    class Env:
        prompt_config = tmp_path / "prompt_config.json"

        def write(self, **values):
            # python-dotenv never removes a key, so clear the routed ones first
            for k in list(os.environ):
                if k.startswith(_PREFIXES):
                    del os.environ[k]
            env_file.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    yield Env()
    os.environ.clear()
    os.environ.update(saved)
    llm_client.refresh_model_config("retrieval")


class FakeRetriever:
    def __init__(self):
        self.retrieve_calls = 0

    def retrieve(self, question, **kwargs):
        self.retrieve_calls += 1
        return {
            "cypher": None, "plain_rows": [],
            "candidate_chunks": [{"chunk_id": "1", "text": "royalty for software is income deemed to accrue",
                                  "similarity": 0.9, "source": "vector"}],
            "retrieval_scope": "hybrid", "nodes": [], "edges": [],
        }

    def get_document_details(self, ids):
        return {str(i): {"document_name": "Income-tax Act, 1961", "page": 12} for i in ids}

    def get_parent_section_context(self, ids):
        return {}

    def get_entities_for_chunks(self, ids):
        return []

    def get_edges_among_nodes(self, ids):
        return []


@pytest.fixture
def harness(app_module, monkeypatch, env):
    store, ev = {}, {"get_keys": [], "set_keys": [], "llm_names": []}
    cache = app_module.cache

    def fake_get(question, metadata_filters=None, extra_key=""):
        key = cache._cache_key(question, metadata_filters, extra_key)
        ev["get_keys"].append(key)
        return dict(store[key]) if key in store else None

    def fake_set(question, response, metadata_filters=None, extra_key=""):
        key = cache._cache_key(question, metadata_filters, extra_key)
        ev["set_keys"].append(key)
        store[key] = response

    def fake_complete(*a, **k):
        ev["llm_names"].append(k.get("name"))
        return "Royalty for software is income deemed to accrue (chunk_id=1)."

    retriever = FakeRetriever()
    monkeypatch.setattr(cache, "get_cached_answer", fake_get)
    monkeypatch.setattr(cache, "set_cached_answer", fake_set)
    monkeypatch.setattr(app_module, "detect_intent", lambda *a, **k: "CONCEPTUAL")
    monkeypatch.setattr(app_module, "retriever", retriever)
    monkeypatch.setattr(app_module, "CATEGORY_SPECS", None)
    monkeypatch.setattr(app_module.llm_client, "complete", fake_complete)
    monkeypatch.setattr(app_module.answer_quality, "judge_answer",
                        lambda *a, **k: {"available": True, "overall": 9, "gaps": []})
    ev["retriever"] = retriever
    return app_module, ev, env


Q = "Is software royalty taxable in India?"


def _ask(client, **extra):
    r = client.post("/api/ask", json={"question": Q, **extra})
    assert r.status_code == 200, r.get_json()
    return r.get_json()


def test_bypass_cache_on_route_skips_read_and_write(harness):
    app_module, ev, env = harness
    env.write(USE_MODEL="CLAUDE")
    client = app_module.app.test_client()
    _ask(client)                               # populates the cache
    assert len(ev["set_keys"]) == 1
    body = _ask(client, bypass_cache=True)
    assert body["from_cache"] is False
    assert len(ev["get_keys"]) == 1 and len(ev["set_keys"]) == 1   # untouched by the bypassed call
    assert ev["retriever"].retrieve_calls == 2
    assert _ask(client)["from_cache"] is True                       # normal calls still use the cache


def test_switching_answer_model_never_serves_the_other_models_answer(harness):
    app_module, ev, env = harness
    client = app_module.app.test_client()
    env.write(USE_MODEL="CLAUDE")
    first = _ask(client)
    assert first["model_info"]["provider"] == "CLAUDE"

    env.write(USE_MODEL="CLAUDE", USE_MODEL_RETRIEVAL="LOCAL", LOCAL_LLM_BASE_URL="http://slm:8000/v1",
              LOCAL_MODEL_NAME="tax-slm-7b")
    second = _ask(client)
    assert second["from_cache"] is False
    assert (second["model_info"]["provider"], second["model_info"]["model"]) == ("LOCAL", "tax-slm-7b")
    # a different model name on the same provider is a different key too
    env.write(USE_MODEL="CLAUDE", USE_MODEL_RETRIEVAL="LOCAL", LOCAL_LLM_BASE_URL="http://slm:8000/v1",
              LOCAL_MODEL_NAME="tax-slm-7b-v2")
    assert _ask(client)["from_cache"] is False
    # back on the first model -> its own entry is a hit again
    env.write(USE_MODEL="CLAUDE")
    again = _ask(client)
    assert again["from_cache"] is True and again["model_info"]["provider"] == "CLAUDE"
    assert ev["retriever"].retrieve_calls == 3
    assert len(set(ev["set_keys"])) == 3


def test_routing_only_answer_generation_changes_the_key(harness):
    app_module, ev, env = harness
    env.write(USE_MODEL="CLAUDE")
    a, _ = app_module._api_ask_impl(Q, None)
    env.write(USE_MODEL="CLAUDE", LLM_ROUTE_ANSWER_GENERATION="GPT")
    b, _ = app_module._api_ask_impl(Q, None)
    assert b["from_cache"] is False and b["model_info"]["provider"] == "GPT"
    assert b["model_info"]["route"] == "LLM_ROUTE_ANSWER_GENERATION"
    # routing an unrelated call (intent detection) does not invalidate answers
    env.write(USE_MODEL="CLAUDE", LLM_ROUTE_INTENT_DETECTION="GPT")
    c, _ = app_module._api_ask_impl(Q, None)
    assert c["from_cache"] is True


def test_prompt_version_change_changes_the_key(harness):
    app_module, ev, env = harness
    env.write(USE_MODEL="CLAUDE")
    a, _ = app_module._api_ask_impl(Q, None)
    builtin = a["model_info"]["answer_prompt"]
    assert builtin["name"] == "answer_generation"
    assert builtin["version"] == app_module.PROMPT_PACK.get_prompt("answer_generation").version
    env.prompt_config.write_text(json.dumps({"enabled": True, "template": "You are a careful tax researcher."}))
    b, _ = app_module._api_ask_impl(Q, None)
    assert b["from_cache"] is False
    assert b["model_info"]["answer_prompt"]["version"].startswith("settings_override:")
    env.prompt_config.write_text(json.dumps({"enabled": True, "template": "You are a terse tax researcher."}))
    c, _ = app_module._api_ask_impl(Q, None)
    assert c["from_cache"] is False
    assert c["model_info"]["answer_prompt"]["version"] != b["model_info"]["answer_prompt"]["version"]
    # lookup mode reports its own prompt
    d, _ = app_module._api_ask_impl(Q, None, answer_mode="lookup")
    assert d["model_info"]["answer_prompt"]["name"] == "answer_generation_lookup"


@pytest.mark.parametrize("kwargs", [{}, {"fast_mode": True, "skip_rerank": True},
                                    {"retrieval_mode": "vector_only", "include_parent_context": True,
                                     "mode_label": "agentic"}])
def test_read_key_equals_write_key_with_model_identity(harness, kwargs):
    app_module, ev, env = harness
    env.write(USE_MODEL="LOCAL", LOCAL_LLM_BASE_URL="http://slm:8000/v1")   # blank model name -> server default
    resp, _ = app_module._api_ask_impl(Q, None, **kwargs)
    assert ev["set_keys"] == ev["get_keys"] and len(ev["get_keys"]) == 1
    assert resp["model_info"]["model_source"] in ("server_default", "discovered")
    resp2, _ = app_module._api_ask_impl(Q, None, **kwargs)
    assert resp2["from_cache"] is True


def test_model_info_and_settings_summary_never_leak_secrets(harness):
    app_module, ev, env = harness
    env.write(USE_MODEL="LOCAL", LOCAL_LLM_BASE_URL=f"http://user:{SECRET}@slm:8000/v1",
              LOCAL_LLM_API_KEY=SECRET, LOCAL_MODEL_NAME="tax-slm-7b", CLAUDE_API_KEY=SECRET,
              LLM_ROUTE_ANSWER_QUALITY_JUDGE="CLAUDE", LLM_ROUTE_ANSWER_EVALUATION="CLAUDE",
              LLM_ROUTE_ANSWER_EVALUATION_MODEL="claude-sonnet-judge")
    client = app_module.app.test_client()
    body = _ask(client, bypass_cache=True)
    info = body["model_info"]
    assert set(info) >= {"provider", "model", "answer_prompt", "nl2cypher_prompt_version", "judge"}
    assert info["nl2cypher_prompt_version"] == app_module.PROMPT_PACK.get_prompt("nl2cypher").version
    assert SECRET not in json.dumps(info) and "slm:8000" not in json.dumps(info)

    summary = client.get("/api/settings/summary").get_json()
    assert summary["model_info"]["provider"] == "LOCAL"
    assert summary["model_info"]["model"] == "tax-slm-7b"
    assert SECRET not in json.dumps(summary["model_info"])


def test_judge_routed_away_from_the_model_under_test(harness, monkeypatch):
    """The SLM answers; the judge + evaluator run on a different model, purely
    via LLM_ROUTE_ANSWER_QUALITY_JUDGE / LLM_ROUTE_ANSWER_EVALUATION."""
    app_module, ev, env = harness
    env.write(USE_MODEL="LOCAL", LOCAL_LLM_BASE_URL="http://slm:8000/v1", LOCAL_MODEL_NAME="tax-slm-7b",
              LLM_ROUTE_ANSWER_QUALITY_JUDGE="CLAUDE", LLM_ROUTE_ANSWER_QUALITY_JUDGE_MODEL="claude-judge",
              LLM_ROUTE_ANSWER_EVALUATION="CLAUDE", LLM_ROUTE_ANSWER_EVALUATION_MODEL="claude-judge")
    info, _ = app_module._ask_model_info("research")
    for name in ("answer_quality_judge", "answer_evaluation"):
        assert (info["judge"][name]["provider"], info["judge"][name]["model"]) == ("CLAUDE", "claude-judge")
        assert info["judge"][name]["same_as_answer_model"] is False

    # and the real judge functions really do go to that model: intercept the
    # provider dispatch (below complete()'s config resolution)
    monkeypatch.undo()   # drop the harness's fake complete / fake judge
    seen = []

    def fake_provider(config, system, user_content, max_tokens):
        seen.append((config.use_model, config.model_name))
        return '{"overall": 8, "groundedness": 8, "completeness": 8, "specificity": 8, "gaps": []}', None

    monkeypatch.setitem(llm_client._DISPATCH, "CLAUDE", fake_provider)
    monkeypatch.setitem(llm_client._DISPATCH, "LOCAL", fake_provider)
    from framework.core.retrieve import answer_evaluation, answer_quality
    chunks = [{"chunk_id": "1", "text": "royalty text"}]
    answer_quality.judge_answer(Q, "answer", chunks, [])
    answer_evaluation.evaluate_answer(Q, "answer", chunks, [], "Grade it.")
    answer_evaluation.evaluate_answer(Q, "answer", [], [], "Grade it.", no_context_mode=True)
    assert seen == [("CLAUDE", "claude-judge")] * 3
    # without the routes they fall back to today's behaviour (the answer model)
    env.write(USE_MODEL="LOCAL", LOCAL_LLM_BASE_URL="http://slm:8000/v1", LOCAL_MODEL_NAME="tax-slm-7b")
    seen.clear()
    answer_quality.judge_answer(Q, "answer", chunks, [])
    assert seen == [("LOCAL", "tax-slm-7b")]
    info, _ = app_module._ask_model_info("research")
    assert info["judge"]["answer_quality_judge"]["same_as_answer_model"] is True
