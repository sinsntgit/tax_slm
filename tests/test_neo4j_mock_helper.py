"""2026-10-14: NEW - self-test for tests/helpers/neo4j_mock.py (so the other
workstreams can rely on its documented behavior)."""
import pytest
from neo4j import Query

from tests.helpers.neo4j_mock import FakeNeo4j


def test_substring_rules_records_and_query_log():
    neo = FakeNeo4j().on("MATCH (c:CaseLaw)", [{"db_id": "1", "name": "A"}])
    with neo.session() as s:
        res = s.run(Query("MATCH (c:CaseLaw) WHERE c.x = $x RETURN c", timeout=5), {"x": 1}, y=2)
        rows = list(res)
    assert rows[0]["name"] == "A" and dict(rows[0]) == {"db_id": "1", "name": "A"}
    q = neo.last("CaseLaw")
    assert q.params == {"x": 1, "y": 2} and q.timeout == 5
    assert neo.ran("WHERE c.x") and not neo.ran("Section")


def test_callable_once_errors_and_defaults():
    neo = FakeNeo4j()
    neo.on("A", [{"v": 1}], once=True).on("A", lambda q, p: [{"v": p["n"] * 2}]).on("BOOM", RuntimeError)
    s = neo.session()
    assert s.run("A").single()["v"] == 1
    assert s.run("A", n=21).single()["v"] == 42
    assert s.run("unscripted").single() is None
    with pytest.raises(RuntimeError):
        s.run("BOOM")
    assert s.execute_read(lambda tx: tx.run("A", n=1).data()) == [{"v": 2}]


def test_strict_mode_and_install(monkeypatch):
    neo = FakeNeo4j(strict=True).install(monkeypatch)
    from framework.core.infra import neo4j_client
    with neo4j_client.get_session() as s:
        with pytest.raises(AssertionError, match="Unscripted"):
            s.run("MATCH (n) RETURN n")
    neo4j_client.verify_connectivity()
    assert neo.sessions_opened == 1
    with pytest.raises(AssertionError, match="No query containing"):
        neo.last("nothing")
