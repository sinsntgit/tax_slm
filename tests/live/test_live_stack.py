"""2026-10-14: NEW - live end-to-end checks for the owner's dev machine
(Docker Neo4j + Redis, a filled-in .env, real LLM + Hugging Face models, an
ingested graph). Skipped by default; run with:

    set TAXKG_LIVE_TESTS=1            (Windows)   /   export TAXKG_LIVE_TESTS=1
    python -m pytest tests/live -v

With TAXKG_LIVE_TESTS=1 conftest.py does NOT install the sentence_transformers
stub, so the real embedding/reranker models are used. Optional:
TAXKG_LIVE_QUESTION overrides the /api/ask question (pick one your ingested
corpus can answer)."""
import os

import pytest

pytestmark = pytest.mark.live

DEFAULT_QUESTION = ("Are payments for shrink-wrapped computer software royalty under section 9(1)(vi)? "
                    "What did the Supreme Court hold in Engineering Analysis?")


def test_neo4j_connectivity_and_graph_not_empty():
    from framework.core.infra import neo4j_client
    neo4j_client.verify_connectivity()
    with neo4j_client.get_session() as session:
        n = session.run("MATCH (n) RETURN count(n) AS n").single()["n"]
    assert n > 0, "Neo4j is reachable but the graph is empty - ingest something first"


def test_validate_graph_endpoint_returns_report():
    from framework.webapp.app import app
    resp = app.test_client().get("/api/validate")
    assert resp.status_code == 200, resp.get_data(as_text=True)[:500]
    body = resp.get_json()
    assert isinstance(body, dict) and "error" not in body


def test_validate_graph_sanity_checks_direct():
    from domain_packs.tax.config import DOMAIN_CONFIG
    from framework.core.infra import neo4j_client
    from framework.core.ingest.validate import run_sanity_checks
    with neo4j_client.get_session() as session:
        report = run_sanity_checks(session, DOMAIN_CONFIG)
    assert isinstance(report, dict) and report


def test_api_ask_round_trip_has_answer_and_citations():
    from framework.webapp.app import app
    question = os.environ.get("TAXKG_LIVE_QUESTION", DEFAULT_QUESTION)
    resp = app.test_client().post("/api/ask", json={"question": question})
    assert resp.status_code == 200, resp.get_data(as_text=True)[:500]
    body = resp.get_json()
    assert body.get("answer", "").strip(), body
    assert "couldn't find anything" not in body["answer"], "nothing retrieved - pick a question your corpus covers"
    grounding = body.get("grounding") or {}
    assert body.get("source_chunks"), "no source chunks were cited"
    assert grounding.get("citation_check", {}).get("citation_ok") is True, grounding.get("citation_check")
