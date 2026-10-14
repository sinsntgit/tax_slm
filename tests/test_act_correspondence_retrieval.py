"""2026-10-14: NEW - Income-tax Act 1961 vs 2025 renumbering in retrieval.

With both Acts ingested, "Section 90" names two unrelated provisions
(ITA1961__Section_90 treaty relief vs ITA2025__Section_90), and the 1961
section's counterpart is ITA2025__Section_159 via CORRESPONDS_TO (written by
build_act_correspondence.py). These tests drive GraphRAGRetriever's real
code against a small fake Neo4j session."""
import sys
from pathlib import Path

import pytest

from domain_packs.tax.config import DOMAIN_CONFIG
from domain_packs.tax.prompts.tax_prompts import PROMPT_PACK
from framework.core.retrieve import graphrag, retrieval_orchestrator


class FakeResult:
    def __init__(self, rows):
        self._rows = [dict(r) for r in rows]

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._rows[0] if self._rows else None


class FakeSession:
    """Dispatches each query to the first handler whose marker substring is
    in the Cypher text; records every (query, params) for assertions."""

    def __init__(self, handlers):
        self.handlers = handlers
        self.calls = []

    def run(self, query, **params):
        text = getattr(query, "text", query)
        self.calls.append((text, params))
        for marker, handler in self.handlers:
            if marker in text:
                return FakeResult(handler(params))
        raise AssertionError(f"unexpected query: {text[:120]}")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


GRAPH = {
    # Chunk full_citation carries no Act - both Acts' chunks say "Section 90".
    "chunks_by_citation": {
        "Section 90": [
            {"chunk_id": "ITA1961__812", "text": "90. (1) The Central Government may enter into an agreement...",
             "full_citation": "Section 90", "entity_id": "ITA1961__Section_90",
             "document_name": "Income-tax Act, 1961"},
            {"chunk_id": "ITA2025__455", "text": "90. Unrelated capital-gains definition...",
             "full_citation": "Section 90", "entity_id": "ITA2025__Section_90",
             "document_name": "Income-tax Act, 2025"},
        ],
    },
    "acts": {"ITA1961": "Income-tax Act, 1961", "ITA2025": "Income-tax Act, 2025"},
    "correspondence": [
        {"from_id": "ITA1961__Section_90", "from_citation": "Section 90", "from_document_name": "Income-tax Act, 1961",
         "to_id": "ITA2025__Section_159", "to_citation": "Section 159", "to_document_name": "Income-tax Act, 2025",
         "to_heading": "Agreement with foreign countries", "to_db_id": "4:x:159",
         "basis": "llm_text_comparison_reindexed", "confidence": 0.7},
    ],
    "section_chunks": {"4:x:159": [
        {"chunk_id": "ITA2025__1001", "text": "159. (1) The Central Government may enter into an agreement...",
         "line_start": 10},
        {"chunk_id": "ITA2025__1002", "text": "159. (2) Where ... the provisions of this Act shall apply...",
         "line_start": 20},
    ]},
}


def make_session(graph=GRAPH):
    def citation_rows(p):
        return [r for r in graph["chunks_by_citation"].get(p["citation"], [])][: p["limit"]]

    def act_rows(p):
        return [{"id": i, "name": graph["acts"][i]} for i in p["ids"] if i in graph["acts"]]

    def correspondence_rows(p):
        return [r for r in graph["correspondence"] if r["from_id"] in p["eids"]][: p["limit"]]

    def section_chunk_rows(p):
        return graph["section_chunks"].get(p["db_id"], [])[: p["k"]]

    return FakeSession([
        ("MATCH (c:Chunk {full_citation: $citation})", citation_rows),
        ("MATCH (a:Act) WHERE a.id IN $ids", act_rows),
        ("-[r:CORRESPONDS_TO]->(t:Section)", correspondence_rows),
        ("MATCH (t)<-[:PART_OF*0..6]-(d)-[:hasSourceChunk]->(c:Chunk)", section_chunk_rows),
    ])


@pytest.fixture
def session(monkeypatch):
    s = make_session()
    monkeypatch.setattr(graphrag.neo4j_client, "get_session", lambda: s)
    return s


@pytest.fixture
def retriever():
    return graphrag.GraphRAGRetriever(DOMAIN_CONFIG, PROMPT_PACK)


@pytest.mark.parametrize("entity_id,expected", [
    ("ITA1961__Section_90", "ITA1961"),
    ("ITA2025__SubSection_159_1", "ITA2025"),
    ("Income_Tax_Act_2025__Clause_10_23C_a", "Income_Tax_Act_2025"),
    ("Proviso_to_ITA2025__Section_9", "ITA2025"),
    ("Proviso_2_to_ITA1961__SubSection_9_1", "ITA1961"),
    ("Section_90", None),                       # un-namespaced single-Act graph
    ("Section_9__superseded__abc123", None),    # versioned clone of an un-namespaced id
    ("CaseLaw_Samsung_v_ITO", None),
    (None, None),
])
def test_act_namespace_of(entity_id, expected):
    assert graphrag.act_namespace_of(entity_id) == expected


def test_citation_lookup_returns_every_acts_match_labelled(session, retriever):
    matches = retriever.retrieve_citation_matches("Section 90")
    assert [m["chunk_id"] for m in matches] == ["ITA1961__812", "ITA2025__455"]
    assert [m["act_name"] for m in matches] == ["Income-tax Act, 1961", "Income-tax Act, 2025"]
    assert [m["act_namespace"] for m in matches] == ["ITA1961", "ITA2025"]


def test_citation_lookup_leaves_act_unset_when_unknown(monkeypatch, retriever):
    graph = dict(GRAPH, acts={}, chunks_by_citation={"Section 5": [
        {"chunk_id": "5", "text": "t", "full_citation": "Section 5", "entity_id": "Section_5",
         "document_name": None}]})
    monkeypatch.setattr(graphrag.neo4j_client, "get_session", lambda: make_session(graph))
    (m,) = retriever.retrieve_citation_matches("Section 5")
    assert m["act_namespace"] is None and m["act_name"] is None


def test_correspondence_context_names_both_provisions_and_acts(session, retriever):
    ctx = retriever.get_act_correspondence_context(["ITA1961__Section_90", "ITA2025__Section_90"])
    (row,) = ctx["rows"]
    assert row["act_correspondence"] == ("Section 90 of Income-tax Act, 1961 corresponds to "
                                         "Section 159 of Income-tax Act, 2025")
    assert row["corresponding_provision_act"] == "Income-tax Act, 2025"
    assert row["correspondence_confidence"] == 0.7
    assert [c["chunk_id"] for c in ctx["chunks"]] == ["ITA2025__1001", "ITA2025__1002"]
    assert {c["source"] for c in ctx["chunks"]} == {"act_correspondence"}
    # starts from the cited provision's Section via PART_OF*0..6, so a cited
    # SubSection/Clause also reaches its Section's correspondence
    corr_query = next(q for q, _ in session.calls if "CORRESPONDS_TO" in q)
    assert "[:PART_OF*0..6]->(s:Section)" in corr_query


def test_no_correspondence_is_empty_not_an_error(monkeypatch, retriever):
    monkeypatch.setattr(graphrag.neo4j_client, "get_session",
                        lambda: make_session(dict(GRAPH, correspondence=[])))
    assert retriever.get_act_correspondence_context(["ITA2025__Section_1"]) == {"rows": [], "chunks": []}


def _stub_other_paths(monkeypatch, retriever):
    monkeypatch.setattr(retriever, "run_graph_retrieval",
                        lambda q, trace=None: {"cypher": None, "plain_rows": [], "nodes": {}, "edges": []})
    monkeypatch.setattr(retriever, "vector_search_chunks", lambda q, top_k=10: [])
    monkeypatch.setattr(graphrag.retrieval_router, "route", lambda q, cfg: "recall")


def test_retrieve_surfaces_both_acts_and_the_corresponding_provision(session, monkeypatch, retriever):
    _stub_other_paths(monkeypatch, retriever)
    out = retriever.retrieve("What does Section 90 provide?", intent="EXACT_LOOKUP")
    ids = [(c["chunk_id"], c["source"]) for c in out["candidate_chunks"]]
    assert ids[:4] == [("ITA1961__812", "citation"), ("ITA2025__455", "citation"),
                       ("ITA2025__1001", "act_correspondence"), ("ITA2025__1002", "act_correspondence")]
    assert any("corresponds to Section 159 of Income-tax Act, 2025" in r.get("act_correspondence", "")
               for r in out["plain_rows"])
    assert out["retrieval_scope"]["graph_found_something"] is True


def test_retrieve_vector_only_never_touches_correspondence(session, monkeypatch, retriever):
    _stub_other_paths(monkeypatch, retriever)
    out = retriever.retrieve("What does Section 90 provide?", retrieval_mode="vector_only")
    assert out["candidate_chunks"] == [] and out["plain_rows"] == []
    assert not any("CORRESPONDS_TO" in q for q, _ in session.calls)


def test_nl2cypher_matched_section_nodes_also_seed_correspondence(session, monkeypatch, retriever):
    _stub_other_paths(monkeypatch, retriever)
    node = {"db_id": "4:x:90", "id": "ITA1961__Section_90", "label": "Section", "properties": {}}
    monkeypatch.setattr(retriever, "run_graph_retrieval", lambda q, trace=None: {
        "cypher": "MATCH (n:Section) RETURN n", "plain_rows": [], "nodes": {"4:x:90": node}, "edges": []})
    monkeypatch.setattr(retriever, "get_chunk_ids_via_graph_traversal", lambda ids: [])
    out = retriever.retrieve("what is the treaty relief provision?", intent="RELATIONAL")
    assert ("ITA2025__1001", "act_correspondence") in [(c["chunk_id"], c["source"]) for c in out["candidate_chunks"]]


def test_correspondence_is_an_expansion_edge():
    assert "CORRESPONDS_TO" in graphrag.GRAPH_EXPANSION_REL_TYPES


def test_agentic_citation_tool_is_act_aware(session, retriever):
    ev = retrieval_orchestrator.run_tool(retriever, {"tool": "CITATION", "citation": "Section 90"})
    assert [c["chunk_id"] for c in ev["chunks"]] == ["ITA1961__812", "ITA2025__455", "ITA2025__1001", "ITA2025__1002"]
    assert ev["plain_rows"] and "Section 159" in ev["plain_rows"][0]["act_correspondence"]


# --- ingestion side: Act chunk ids must not collide across Acts --------------

def test_act_chunk_ids_are_namespaced():
    conv = Path(__file__).resolve().parents[1] / "domain_packs" / "tax" / "converters"
    sys.path.insert(0, str(conv))
    try:
        import extract_entities
    finally:
        sys.path.remove(str(conv))
    assert extract_entities.act_chunk_id(5, "ITA1961") == "ITA1961__5"
    assert extract_entities.act_chunk_id(5, "ITA2025") == "ITA2025__5"
    assert extract_entities.act_chunk_id(5, None) == 5  # un-namespaced run unchanged
