"""2026-10-14: NEW - agentic retrieval_mode used to lose parent-section
context because it generates through _api_ask_impl(retrieval_mode=
"vector_only"). These tests run the REAL _api_ask_impl() (fast_mode, so no
intent/quality-gate LLM calls) against a fake retriever/LLM and check what
actually reaches the answer-generation prompt and the cache key."""
import pytest

from framework.webapp import app as webapp

PARENT_TEXT = "PARENT-SECTION-FULL-TEXT: (1) lead-in ... (2) the clause the chunk was cut from ..."


class FakeRetriever:
    def __init__(self):
        self.parent_calls = []

    def retrieve(self, question, **kw):
        return {"cypher": None, "plain_rows": [], "nodes": [], "edges": [],
                "candidate_chunks": [], "retrieval_scope": {"retrieval_mode": kw.get("retrieval_mode")}}

    def get_document_details(self, chunk_ids):
        return {str(c): {"document_name": "Income-tax Act, 2025", "page": 7} for c in chunk_ids}

    def get_entities_for_chunks(self, chunk_ids):
        return []

    def get_edges_among_nodes(self, node_db_ids):
        return []

    def get_parent_section_context(self, chunk_ids):
        self.parent_calls.append(list(chunk_ids))
        return {str(c): {"parent_id": "ITA2025__Section_159", "parent_citation": "Section 159",
                         "parent_label": "Section", "parent_text": PARENT_TEXT} for c in chunk_ids}


@pytest.fixture
def harness(monkeypatch):
    fake = FakeRetriever()
    prompts, cache_writes, cache_reads = [], [], []
    monkeypatch.setattr(webapp, "retriever", fake)
    monkeypatch.setattr(webapp, "CATEGORY_SPECS", None)
    monkeypatch.setattr(webapp.reranker, "rerank", lambda q, chunks, **kw: list(chunks))

    def fake_complete(system, user_content, max_tokens=1000, **kw):
        prompts.append(user_content)
        return "ANSWER"

    monkeypatch.setattr(webapp.llm_client, "complete", fake_complete)
    monkeypatch.setattr(webapp.cache, "get_cached_answer",
                        lambda q, f, extra_key=None: cache_reads.append(extra_key) or None)
    monkeypatch.setattr(webapp.cache, "set_cached_answer",
                        lambda q, r, f, extra_key=None: cache_writes.append(extra_key))
    return {"retriever": fake, "prompts": prompts, "cache_writes": cache_writes, "cache_reads": cache_reads}


CHUNKS = [{"chunk_id": "ITA2025__12", "text": "(b) the clause text only", "source": "graph"}]


def _ask(**kw):
    return webapp._api_ask_impl("What does section 159 provide?", None, fast_mode=True, skip_rerank=True,
                                supplementary_chunks=CHUNKS, **kw)


def _mode_part(extra_key):
    return extra_key.split(":model=")[0]


def test_resolve_rule():
    r = webapp._resolve_include_parent_context
    assert r("hybrid", None) is True
    assert r("semantic_graph", None) is True
    assert r("vector_only", None) is False
    assert r("vector_only", True) is True
    assert r("hybrid", False) is False


def test_simple_rag_vector_only_still_excludes_parent_context(harness):
    response, _ = _ask(retrieval_mode="vector_only")
    assert harness["retriever"].parent_calls == []
    assert PARENT_TEXT not in harness["prompts"][0]
    assert response["retrieval_mode"] == "vector_only"
    # 2026-10-14: the key now also carries ":model=...|prompt=..." (see
    # app._ask_model_info()); the mode part is what this test pins down.
    assert [_mode_part(k) for k in harness["cache_writes"]] == ["vector_only:research:fast"]


def test_explicit_include_parent_context_reaches_the_llm(harness):
    response, internal = _ask(retrieval_mode="vector_only", include_parent_context=True, mode_label="agentic")
    assert harness["retriever"].parent_calls == [["ITA2025__12"]]
    assert PARENT_TEXT in harness["prompts"][0]
    ref = response["grounding"]["source_references"][0]
    assert ref["parent_context"]["parent_id"] == "ITA2025__Section_159"
    # cached + reported under its own mode, not as a Simple RAG answer
    assert response["retrieval_mode"] == "agentic"
    assert [_mode_part(k) for k in harness["cache_writes"]] == ["agentic:research:fast"]
    assert harness["cache_reads"] == harness["cache_writes"]


def test_agentic_branch_passes_parent_context_through(harness, monkeypatch):
    orchestrator_result = {"chunks": CHUNKS, "cypher_queries": ["MATCH (n) RETURN n"], "plain_rows": [],
                           "iterations": [{}], "final_verdict": "sufficient"}
    monkeypatch.setattr(webapp.retrieval_orchestrator, "run_agentic_retrieval",
                        lambda retriever, question, trace=None: orchestrator_result)
    response, internal, trace_record = webapp._run_unstructured_agent(
        "What does section 159 provide?", None, "agentic", "research", fast_mode=True, skip_rerank=True)
    assert PARENT_TEXT in harness["prompts"][0]
    assert response["grounding"]["source_references"][0]["parent_context"]["parent_text"] == PARENT_TEXT
    assert response["retrieval_mode"] == "agentic"
    assert response["cypher"] == "MATCH (n) RETURN n"
    assert all(k.startswith("agentic:") for k in harness["cache_writes"])
