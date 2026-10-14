"""2026-10-14: NEW - /api/ask answer-cache consistency (framework/webapp/app.py).

Regression for the 2026-10-13 bug where the cache READ key carried a
':fast'/':normal' suffix but the WRITE key did not, so every /api/ask was a
guaranteed miss. Drives the REAL route + _api_ask_impl() through Flask's test
client with only the heavy edges replaced: intent detection, the
GraphRAGRetriever, llm_client.complete, the answer-quality judge and Redis
(an in-memory dict keyed by the real cache._cache_key()). The reranker,
grounding, prompt resolution, category formatting etc. all run for real."""
import pytest


@pytest.fixture(scope="module")
def app_module():
    import framework.webapp.app as app_module  # noqa: WPS433 - import is the point
    return app_module


class FakeRetriever:
    def __init__(self):
        self.retrieve_calls = 0

    def retrieve(self, question, **kwargs):
        self.retrieve_calls += 1
        return {
            "cypher": "MATCH (s:Section {section_no:'9'}) RETURN s",
            "plain_rows": [{"s": {"id": "Section_9", "hasText": "income deemed to accrue"}}],
            "candidate_chunks": [
                {"chunk_id": "1", "text": "royalty paid for software is income deemed to accrue in india",
                 "similarity": 0.9, "source": "vector"},
                {"chunk_id": "2", "text": "agricultural income is exempt", "similarity": 0.4, "source": "vector"},
            ],
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
def harness(app_module, monkeypatch, tmp_path):
    store, events = {}, {"llm": 0, "get_keys": [], "set_keys": []}
    cache = app_module.cache

    def fake_get(question, metadata_filters=None, extra_key=""):
        key = cache._cache_key(question, metadata_filters, extra_key)
        events["get_keys"].append(key)
        return dict(store[key]) if key in store else None

    def fake_set(question, response, metadata_filters=None, extra_key=""):
        key = cache._cache_key(question, metadata_filters, extra_key)
        events["set_keys"].append(key)
        store[key] = response

    def fake_complete(*a, **k):
        events["llm"] += 1
        return "Royalty for software is taxable as income deemed to accrue (chunk_id=1)."

    retriever = FakeRetriever()
    monkeypatch.setattr(cache, "get_cached_answer", fake_get)
    monkeypatch.setattr(cache, "set_cached_answer", fake_set)
    monkeypatch.setattr(app_module, "detect_intent", lambda *a, **k: "CONCEPTUAL")
    monkeypatch.setattr(app_module, "retriever", retriever)
    monkeypatch.setattr(app_module, "CATEGORY_SPECS", None)
    monkeypatch.setattr(app_module.llm_client, "complete", fake_complete)
    monkeypatch.setattr(app_module.answer_quality, "judge_answer",
                        lambda *a, **k: {"available": True, "overall": 9, "gaps": []})
    # deterministic config files (no dependence on whatever is committed at repo root)
    monkeypatch.setenv("PROMPT_CONFIG_PATH", str(tmp_path / "prompt_config.json"))
    monkeypatch.setenv("RERANKER_CONFIG_PATH", str(tmp_path / "reranker_config.json"))
    monkeypatch.setenv("COMPARATIVE_RESPONSE_CONFIG_PATH", str(tmp_path / "comparative.json"))
    events["retriever"] = retriever
    return app_module, events


def test_second_identical_ask_is_served_from_cache(harness):
    app_module, ev = harness
    client = app_module.app.test_client()

    first = client.post("/api/ask", json={"question": "What is the tax treatment of software royalty?"})
    assert first.status_code == 200, first.get_json()
    body1 = first.get_json()
    assert body1["from_cache"] is False
    assert "Royalty for software" in body1["answer"]
    assert body1["grounding"]["citation_check"]["citation_ok"] is True
    llm_calls_first = ev["llm"]
    assert llm_calls_first >= 1 and ev["retriever"].retrieve_calls == 1

    # same question modulo case/whitespace/trailing punctuation -> same key
    second = client.post("/api/ask", json={"question": "what is the tax treatment of  software royalty"})
    body2 = second.get_json()
    assert second.status_code == 200
    assert body2["from_cache"] is True
    assert body2["answer"] == body1["answer"]
    assert ev["llm"] == llm_calls_first and ev["retriever"].retrieve_calls == 1  # no second pipeline run

    # the key written on the miss is exactly the key read on the hit
    assert len(ev["set_keys"]) == 1
    assert ev["get_keys"] == [ev["set_keys"][0], ev["set_keys"][0]]


@pytest.mark.parametrize("kwargs", [{}, {"fast_mode": True, "skip_rerank": True},
                                    {"retrieval_mode": "vector_only", "answer_mode": "lookup"}])
def test_api_ask_impl_read_key_equals_write_key(harness, kwargs):
    app_module, ev = harness
    resp, _internal = app_module._api_ask_impl("Is software royalty taxable?", {"year": "2021"}, **kwargs)
    assert resp["from_cache"] is False
    assert len(ev["get_keys"]) == 1 and ev["set_keys"] == ev["get_keys"]
    resp2, _ = app_module._api_ask_impl("Is software royalty taxable?", {"year": "2021"}, **kwargs)
    assert resp2["from_cache"] is True


def test_different_modes_do_not_share_cache_entries(harness):
    app_module, ev = harness
    app_module._api_ask_impl("Is software royalty taxable?", None)
    app_module._api_ask_impl("Is software royalty taxable?", None, fast_mode=True)
    app_module._api_ask_impl("Is software royalty taxable?", None, answer_mode="opinion")
    assert len(set(ev["set_keys"])) == 3
    assert ev["retriever"].retrieve_calls == 3


def test_bypass_cache_neither_reads_nor_writes(harness):
    app_module, ev = harness
    resp, _ = app_module._api_ask_impl("Is software royalty taxable?", None, bypass_cache=True)
    assert resp["from_cache"] is False
    assert ev["get_keys"] == [] and ev["set_keys"] == []


def test_ask_validation_errors(app_module):
    client = app_module.app.test_client()
    assert client.post("/api/ask", json={"question": "  "}).status_code == 400
    assert client.post("/api/ask", json={"question": "q", "retrieval_mode": "nope"}).status_code == 400
    assert client.post("/api/ask", json={"question": "q", "answer_mode": "nope"}).status_code == 400
