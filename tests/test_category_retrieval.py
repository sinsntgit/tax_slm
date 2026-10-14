"""2026-10-14: NEW - framework/core/retrieve/category_retrieval.py: keyword
extraction, graph-first/vector-fallback per category, stance-balanced
case-law path, context formatting. Neo4j via tests/helpers/neo4j_mock.py."""
import pytest

from framework.core.retrieve import category_retrieval as cr
from framework.core.retrieve.category_retrieval import CategoryResult, CategorySpec


# --- extract_keywords ----------------------------------------------------------

def test_extract_keywords_drops_stopwords_and_short_words_keeps_digits():
    kws = cr.extract_keywords("What is the TDS rate under Section 194J(1)(b) for fees of a CA?")
    assert kws == ["rate", "section", "194j", "1", "fees"]


def test_extract_keywords_dedup_order_and_cap():
    assert cr.extract_keywords("royalty royalty software Royalty copyright") == ["royalty", "software", "copyright"]
    assert cr.extract_keywords("alpha bravo charlie delta echo", max_keywords=2) == ["alpha", "bravo"]
    assert cr.extract_keywords("") == []
    assert cr.extract_keywords("is it the a of") == []


# --- fixtures ----------------------------------------------------------------

class FakeRetriever:
    """Records calls; scripted returns for the 5 retriever methods category_retrieval uses."""

    def __init__(self, graph_chunks=None, vector=None, labels=None, semantic=None):
        self.graph_chunks = graph_chunks or []
        self.vector = vector or []
        self.labels = labels or {}
        self.semantic = semantic
        self.calls = []

    def get_chunk_ids_via_graph_traversal(self, db_ids):
        self.calls.append(("traverse", list(db_ids)))
        return [c["chunk_id"] for c in self.graph_chunks]

    def get_chunks_by_ids(self, ids):
        self.calls.append(("by_ids", list(ids)))
        return list(self.graph_chunks)

    def vector_search_chunks(self, query, top_k):
        self.calls.append(("vector", query, top_k))
        return [dict(c) for c in self.vector]

    def semantic_entry_point_search(self, query, top_k, semantic_graph_trace=None):
        self.calls.append(("semantic", query, top_k, semantic_graph_trace))
        if isinstance(self.semantic, Exception):
            raise self.semantic
        return [dict(c) for c in (self.semantic or [])]

    def get_entity_labels_for_chunks(self, ids):
        self.calls.append(("labels", list(ids)))
        return self.labels


LAW = CategorySpec(key="law", label="Law", description="statutes", node_labels=("Section", "Rule"),
                   text_properties=("hasText", "full_citation"), vector_query_prefix="Statute: ")
SYNTH = CategorySpec(key="final", label="Final Output", description="synth", synthesis_only=True)
CASE = CategorySpec(key="case_law_hierarchy", label="Case Law", description="cases",
                    node_labels=("CaseLaw",), text_properties=("case_name", "holding"),
                    instructions="Present both sides.", max_chars_per_chunk=2000, max_chars_per_category=6000)


# --- run_category_retrieval -------------------------------------------------------

def test_synthesis_category_is_not_retrieved(fake_neo4j):
    r = FakeRetriever()
    [res] = cr.run_category_retrieval("royalty question", r, [SYNTH])
    assert res.status == "synthesis" and r.calls == [] and fake_neo4j.queries == []


def test_graph_hit_returns_graph_status_and_params(fake_neo4j):
    fake_neo4j.on("RETURN DISTINCT elementId(n)", [{"db_id": "4:x:1"}, {"db_id": "4:x:2"}])
    r = FakeRetriever(graph_chunks=[{"chunk_id": i, "text": f"t{i}", "extra": 1} for i in range(12)])
    [res] = cr.run_category_retrieval("royalty on software", r, [LAW], graph_limit=7, graph_timeout_seconds=3)
    assert res.status == "graph"
    assert len(res.chunks) == 10 and res.chunks[0] == {"chunk_id": 0, "text": "t0"}
    q = fake_neo4j.last("RETURN DISTINCT elementId(n)")
    assert q.params == {"labels": ["Section", "Rule"], "text_props": ["hasText", "full_citation"],
                        "keywords": ["royalty", "software"], "limit": 7}
    assert q.timeout == 3
    assert ("traverse", ["4:x:1", "4:x:2"]) in r.calls


def test_graph_match_without_chunks_falls_through_to_vector(fake_neo4j):
    fake_neo4j.on("RETURN DISTINCT elementId(n)", [{"db_id": "4:x:1"}])
    r = FakeRetriever(graph_chunks=[], vector=[{"chunk_id": 5, "text": "v5"}, {"chunk_id": 6, "text": "v6"}],
                      labels={"5": ["Section"], "6": ["CaseLaw"]})
    [res] = cr.run_category_retrieval("royalty software", r, [LAW], vector_top_k=4)
    assert res.status == "vector_fallback"
    assert res.chunks == [{"chunk_id": 5, "text": "v5"}]          # label-filtered: 6 is CaseLaw
    assert ("vector", "Statute: royalty software", 4) in r.calls  # spec prefix applied


def test_vector_only_mode_never_touches_neo4j(fake_neo4j):
    r = FakeRetriever(vector=[{"chunk_id": 5, "text": "v5"}], labels={"5": ["Rule"]})
    law, case = cr.run_category_retrieval("royalty", r, [LAW, CASE], retrieval_mode="vector_only")
    assert fake_neo4j.queries == []        # neither the keyword pass nor the stance-split pull ran
    assert law.status == "vector_fallback"
    assert case.status == "not_found"       # chunk 5 is a Rule, not CaseLaw


def test_graph_search_error_degrades_to_vector(fake_neo4j):
    fake_neo4j.on("RETURN DISTINCT elementId(n)", RuntimeError("neo4j down"))
    r = FakeRetriever(vector=[{"chunk_id": 1, "text": "v"}], labels={"1": ["Section"]})
    [res] = cr.run_category_retrieval("royalty", r, [LAW])
    assert res.status == "vector_fallback"


@pytest.mark.parametrize("vector,labels,note_fragment", [
    ([], {}, "No matching records"),
    ([{"chunk_id": 1, "text": "v"}], {"1": ["Circular"]}, "No matching records"),
])
def test_not_found_paths(fake_neo4j, vector, labels, note_fragment):
    [res] = cr.run_category_retrieval("royalty", FakeRetriever(vector=vector, labels=labels), [LAW])
    assert res.status == "not_found" and note_fragment in res.note


def test_reranker_applied_to_vector_fallback(fake_neo4j):
    calls = {}

    class FakeReranker:
        @staticmethod
        def rerank(question, chunks, top_n=None):
            calls["args"] = (question, [c["chunk_id"] for c in chunks], top_n)
            return []

    r = FakeRetriever(vector=[{"chunk_id": 1, "text": "v"}], labels={"1": ["Section"]})
    [res] = cr.run_category_retrieval("royalty", r, [LAW], reranker_module=FakeReranker)
    assert calls["args"] == ("royalty", [1], 6)   # scored against the ORIGINAL question, not the prefix
    assert res.status == "not_found" and "relevance bar" in res.note


def test_semantic_graph_mode_uses_entry_points_and_passes_trace(fake_neo4j):
    trace = {"files_used": []}
    r = FakeRetriever(semantic=[{"chunk_id": "mdhydrate_1", "text": "x"}], labels={"mdhydrate_1": ["Rule"]})
    [res] = cr.run_category_retrieval("royalty", r, [LAW], retrieval_mode="semantic_graph",
                                      semantic_graph_trace=trace)
    assert res.status == "vector_fallback"
    sem = [c for c in r.calls if c[0] == "semantic"][0]
    assert sem[3] is trace
    assert not any(c[0] == "vector" for c in r.calls)


def test_semantic_graph_error_is_not_found(fake_neo4j):
    r = FakeRetriever(semantic=RuntimeError("store missing"))
    [res] = cr.run_category_retrieval("royalty", r, [LAW], retrieval_mode="semantic_graph")
    assert res.status == "not_found"


# --- stance-balanced case_law_hierarchy -------------------------------------------

def _stance_rows(query, params):
    n = 2 if params["stance"] == "taxpayer_favorable" else 1
    return [{"db_id": f"{params['stance']}_{i}", "case_name": f"Case {i}"} for i in range(n)]


def test_case_law_stance_balanced_path(fake_neo4j, monkeypatch):
    from framework.core.retrieve import case_law_entry_points
    monkeypatch.setattr(case_law_entry_points, "search_case_law_entry_points",
                        lambda q, top_k=20: [{"db_id": "ep_1"}])
    fake_neo4j.on("MATCH (n:CaseLaw)", _stance_rows)
    r = FakeRetriever(graph_chunks=[{"chunk_id": i, "text": f"c{i}"} for i in range(25)])
    [res] = cr.run_category_retrieval("royalty software", r, [CASE])
    assert res.status == "graph" and len(res.chunks) == 20
    assert "2 taxpayer-favorable, 1 revenue-favorable" in res.note
    stances = [q.params["stance"] for q in fake_neo4j.matching("MATCH (n:CaseLaw)")]
    assert stances == ["taxpayer_favorable", "revenue_favorable"]
    traversed = set([c for c in r.calls if c[0] == "traverse"][0][1])
    assert traversed == {"ep_1", "taxpayer_favorable_0", "taxpayer_favorable_1", "revenue_favorable_0"}
    assert res.instructions == "Present both sides." and res.max_chars_per_chunk == 2000


def test_case_law_thin_stances_fall_through_to_generic(fake_neo4j, monkeypatch):
    from framework.core.retrieve import case_law_entry_points
    monkeypatch.setattr(case_law_entry_points, "search_case_law_entry_points",
                        lambda q, top_k=20: (_ for _ in ()).throw(RuntimeError("no store")))
    fake_neo4j.on("MATCH (n:CaseLaw)", [{"db_id": "only_one", "case_name": "X"}])
    fake_neo4j.on("RETURN DISTINCT elementId(n)", [])  # generic graph pass finds nothing
    r = FakeRetriever(vector=[{"chunk_id": 9, "text": "v"}], labels={"9": ["CaseLaw"]})
    [res] = cr.run_category_retrieval("royalty software", r, [CASE])
    assert res.status == "vector_fallback"          # generic path still produced something
    assert fake_neo4j.ran("RETURN DISTINCT elementId(n)")


# --- format_category_context / summarize ------------------------------------------

def test_format_category_context_statuses_instructions_and_truncation():
    results = [
        CategoryResult("final", "Final Output", "synth", status="synthesis"),
        CategoryResult("law", "Law", "statutes", status="not_found", note="nothing here"),
        CategoryResult("vec", "Vec", "d", status="vector_fallback",
                       chunks=[{"chunk_id": 1, "text": "a" * 50}], max_chars_per_chunk=10),
        CategoryResult("case", "Case Law", "cases", status="graph", note="balanced",
                       chunks=[{"chunk_id": i, "text": "b" * 100} for i in range(5)],
                       instructions="Present both sides.", max_chars_per_chunk=100, max_chars_per_category=150),
    ]
    text = cr.format_category_context(results)
    assert text.startswith("TAX KNOWLEDGE & ANALYSIS FRAMEWORK - CATEGORY COVERAGE")
    assert "### Final Output" in text and "STATUS: SYNTHESIS" in text
    assert "STATUS: NOT FOUND - nothing here" in text
    assert "STATUS: FOUND (vector-search fallback match" in text
    assert "[chunk_id=1]\n" + "a" * 10 + "\n" in text + "\n"
    assert "INSTRUCTIONS FOR THIS CATEGORY: Present both sides." in text
    assert "STATUS: FOUND (graph match) - balanced" in text
    case_block = text.split("STATUS: FOUND (graph match) - balanced\n", 1)[1]
    assert case_block.endswith("\n... (truncated)")
    assert len(case_block) == 150 + len("\n... (truncated)")


def test_summarize_category_coverage():
    res = [CategoryResult("k", "L", "d", status="graph", chunks=[{"chunk_id": 1, "text": "x"}])]
    assert cr.summarize_category_coverage(res) == [{"key": "k", "label": "L", "status": "graph", "chunk_count": 1}]


def test_tax_category_specs_contract():
    """Domain-pack contract the graph keyword search relies on: retrieved
    categories have labels + SCALAR text properties (reduce() over a list
    property errors in Cypher)."""
    from domain_packs.tax.config import TAX_CATEGORY_SPECS
    known_list_props = {"keywords", "cases_cited", "legal_concepts", "statutes_cited", "questions_of_law",
                        "precedents_overruled", "treaty_references", "other_act_references"}
    keys = [s.key for s in TAX_CATEGORY_SPECS]
    assert len(keys) == len(set(keys)) == 6
    assert "case_law_hierarchy" in keys
    for s in TAX_CATEGORY_SPECS:
        if s.synthesis_only:
            continue
        assert s.node_labels and s.text_properties, s.key
        assert not (set(s.text_properties) & known_list_props), s.key
