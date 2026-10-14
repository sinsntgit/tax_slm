"""
2026-10-14: phase0/review-pii - regression tests for the PII paths fixed after
the review/phase0 audit (see tests/test_review_phase0.py section 3). Each test
drives the REAL masking/route code with fakes (no Neo4j/LLM): FirmOpinion
client_name must never leave unmasked, and regex PII (a PAN here) must be
masked on every path that reaches the answer LLM / browser.
"""
import pytest

from domain_packs.tax.config import DOMAIN_CONFIG
from framework.core.retrieve import pii as pii_mod
from framework.core.retrieve import retrieval_orchestrator
from framework.core.retrieve.category_retrieval import CategoryResult
from framework.webapp import app as webapp
from test_agentic_parent_context import FakeRetriever

PAN = "ABCDE1234F"


@pytest.fixture(autouse=True)
def _no_audit_log(monkeypatch):
    monkeypatch.setattr(pii_mod, "_log_pii_access", lambda *a, **k: None)


@pytest.fixture
def masker():
    return pii_mod.PiiMasker(DOMAIN_CONFIG)


# --- PiiMasker.mask_plain_rows ---------------------------------------------------

def test_plain_rows_mask_always_masked_fields_in_every_shape(masker):
    node = {"id": "FirmOpinion_OP_01", "label": "FirmOpinion",
            "properties": {"client_name": "Zomato Limited", "opinion_id": "OP-01", "id": "FirmOpinion_OP_01"}}
    rows = masker.mask_plain_rows([{
        "o.client_name": "Zomato Limited",                       # scalar, dotted key
        "collect(o.counterparty_name)": ["Acme Inc", "Beta LLP"],  # unaliased collect()
        "o.partner_name": None,                                  # null stays null
        "o": {"client_name": "Zomato Limited", "title": "Opinion for [CLIENT]", "id": "FirmOpinion_OP_01"},
        "n": node,                                               # node-shaped dict, one level deeper
        "rows": [{"client_name": "Zomato Limited"}],             # list of maps
        "o.title": f"PAN {PAN}",                                 # non-masked field: regex only
    }])
    row = rows[0]
    text = str(row)
    for raw in ("Zomato", "Acme", "Beta", PAN):
        assert raw not in text
    assert row["o.client_name"].startswith("[CLIENT_NAME_REDACTED_")
    assert all(v.startswith("[COUNTERPARTY_NAME_REDACTED_") for v in row["collect(o.counterparty_name)"])
    assert row["o.partner_name"] is None
    # non-PII fields survive untouched; never-mask ids are not altered
    assert row["o"]["title"] == "Opinion for [CLIENT]" and row["o"]["id"] == "FirmOpinion_OP_01"
    assert row["n"]["properties"]["opinion_id"] == "OP-01" and row["n"]["id"] == "FirmOpinion_OP_01"
    # same raw value -> same stable token wherever it appears
    assert row["o.client_name"] == row["o"]["client_name"] == row["n"]["properties"]["client_name"]
    # a key that merely CONTAINS the word is not a masked field
    assert masker.always_masked_field_for_key("client_name_count") is None
    assert masker.always_masked_field_for_key("o.client_name") == "client_name"


def test_plain_row_masking_is_idempotent(masker):
    once = masker.mask_plain_rows([{"client_name": "Zomato Limited", "t": f"PAN {PAN}"}])
    assert masker.mask_plain_rows(once) == once


# --- retrieval_orchestrator.run_tool (agentic + structured_research law_agent) ----

class _ToolRetriever:
    def __init__(self):
        self.pii = pii_mod.PiiMasker(DOMAIN_CONFIG)

    def run_graph_retrieval(self, query_text, trace=None):
        return {"cypher": "MATCH (o:FirmOpinion) RETURN o.client_name",
                "plain_rows": [{"o.client_name": "Zomato Limited"}], "nodes": {}, "edges": []}

    def get_chunk_ids_via_graph_traversal(self, db_ids):
        return ["c1"]

    def extract_source_chunk_ids_legacy(self, nodes):
        return []

    def get_chunks_by_ids(self, ids):
        return [{"chunk_id": "c1", "text": f"Client PAN {PAN}"}]

    def vector_search_chunks(self, q, top_k=10):
        return [{"chunk_id": "c2", "text": f"PAN {PAN}", "similarity": 0.9}]


def test_run_tool_masks_rows_and_chunks_for_every_consumer():
    r = _ToolRetriever()
    ev = retrieval_orchestrator.run_tool(r, {"tool": "CYPHER", "query_text": "who"})
    assert "Zomato" not in str(ev["plain_rows"]) and PAN not in str(ev["chunks"])
    assert ev["cypher"] and ev["chunks"][0]["chunk_id"] == "c1"
    ev = retrieval_orchestrator.run_tool(r, {"tool": "VECTOR", "query_text": "who"})
    assert PAN not in str(ev["chunks"]) and ev["chunks"][0]["similarity"] == 0.9


def test_agentic_loop_sends_only_masked_evidence_to_sufficiency_llm(monkeypatch):
    seen = []
    monkeypatch.setattr(retrieval_orchestrator.retrieval_planner, "plan_retrieval",
                        lambda *a, **k: {"tool": "CYPHER", "query_text": "who", "citation": "", "reasoning": ""})
    monkeypatch.setattr(retrieval_orchestrator, "get_schema_summary", lambda: {})

    def check(question, pool, trace=None):
        seen.append(str(pool))
        return {"verdict": "SUFFICIENT", "reasoning": "", "missing": ""}

    monkeypatch.setattr(retrieval_orchestrator.evidence_sufficiency, "check_sufficiency", check)
    result = retrieval_orchestrator.run_agentic_retrieval(_ToolRetriever(), "who is the client?")
    assert seen and all("Zomato" not in s and PAN not in s for s in seen)
    assert "Zomato" not in str(result["plain_rows"])


# --- app: agentic branch, supplementary chunks, parent context, category chunks ---

@pytest.fixture
def harness(monkeypatch):
    prompts = []
    monkeypatch.setattr(webapp, "CATEGORY_SPECS", None)
    monkeypatch.setattr(webapp.reranker, "rerank", lambda q, chunks, **kw: list(chunks))
    monkeypatch.setattr(webapp.llm_client, "complete",
                        lambda system, user_content, max_tokens=1000, **kw: prompts.append(user_content) or "ANSWER")
    monkeypatch.setattr(webapp.cache, "get_cached_answer", lambda *a, **k: None)
    monkeypatch.setattr(webapp.cache, "set_cached_answer", lambda *a, **k: None)
    return prompts


def test_agentic_internal_rows_and_evidence_chunks_are_masked(harness, monkeypatch):
    monkeypatch.setattr(webapp, "retriever", FakeRetriever())
    orchestrator_result = {
        "chunks": [{"chunk_id": "opinion_x_2", "text": f"[CLIENT] PAN {PAN}", "source": "graph"}],
        "cypher_queries": ["MATCH (o:FirmOpinion) RETURN o.client_name"],
        "plain_rows": [{"o.client_name": "Zomato Limited"}],
        "iterations": [{}], "final_verdict": "sufficient"}
    monkeypatch.setattr(webapp.retrieval_orchestrator, "run_agentic_retrieval",
                        lambda retriever, question, trace=None: orchestrator_result)
    response, internal, _ = webapp._run_unstructured_agent(
        "Which client?", None, "agentic", "research", fast_mode=True, skip_rerank=True)
    # internal rows feed /api/ask/evaluate judges + evaluators (and so the LLM call log)
    assert "Zomato" not in str(internal["plain_rows"]) and "Zomato" not in str(response["plain_rows"])
    assert PAN not in "".join(harness) and PAN not in str(response)
    assert PAN not in str(internal["top_chunks"])


class _PanParentRetriever(FakeRetriever):
    def get_parent_section_context(self, chunk_ids):
        return {str(c): {"parent_id": "GenericDocument_x", "parent_citation": "client upload",
                         "parent_label": "GenericDocument", "parent_text": f"Assessee PAN {PAN}"}
                for c in chunk_ids}


def test_parent_context_is_masked_before_prompt_and_response(harness, monkeypatch):
    monkeypatch.setattr(webapp, "retriever", _PanParentRetriever())
    response, _ = webapp._api_ask_impl(
        "q", None, fast_mode=True, skip_rerank=True, retrieval_mode="vector_only",
        include_parent_context=True, supplementary_chunks=[{"chunk_id": "c1", "text": "t", "source": "graph"}])
    assert PAN not in "".join(harness) and PAN not in str(response)
    assert "PAN_REDACTED" in response["grounding"]["source_references"][0]["parent_context"]["parent_text"]


def test_category_chunks_are_masked_before_prompt(harness, monkeypatch):
    monkeypatch.setattr(webapp, "retriever", FakeRetriever())
    monkeypatch.setattr(webapp, "CATEGORY_SPECS", ("x",))
    cat = CategoryResult("client_specific_data", "Client Specific Data", "d", status="graph",
                         chunks=[{"chunk_id": "c9", "text": f"Client PAN {PAN}"}])
    monkeypatch.setattr(webapp.category_retrieval, "run_category_retrieval", lambda *a, **k: [cat])
    response, _ = webapp._api_ask_impl("q", None, fast_mode=True, skip_rerank=True, retrieval_mode="hybrid",
                                       supplementary_chunks=[{"chunk_id": "c1", "text": "t", "source": "graph"}])
    assert PAN not in "".join(harness) and PAN not in str(response)


# --- data explorer: no search oracle over always-masked fields ---------------------

def test_data_explorer_does_not_match_on_masked_fields(fake_neo4j):
    fake_neo4j.on("MATCH (n:FirmOpinion)", [
        {"db_id": "4:x:1", "props": {"id": "FirmOpinion_OP_01", "client_name": "Zomato Limited",
                                    "title": "Opinion on FTS"}}])
    client = webapp.app.test_client()
    hit = client.get("/api/data-explorer/nodes?label=FirmOpinion&search=zomato").get_json()
    assert hit["total"] == 0 and hit["rows"] == []
    ok = client.get("/api/data-explorer/nodes?label=FirmOpinion&search=fts").get_json()
    assert ok["total"] == 1 and "Zomato" not in str(ok)
