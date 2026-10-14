"""2026-10-14: NEW - nl2cypher v12 (Neo4j 5, not Apache AGE) + truncated-query
detection in GraphRAGRetriever.nl_to_cypher(). See graphrag.py's
NL2CYPHER_MAX_TOKENS / cypher_incompleteness_reason()."""
import pytest

from domain_packs.tax.config import DOMAIN_CONFIG
from domain_packs.tax.prompts.tax_prompts import PROMPT_PACK
from framework.core.retrieve import graphrag


# --- prompt versions ----------------------------------------------------------

def test_v12_is_current_and_v11_is_untouched():
    versions = PROMPT_PACK.list_versions("nl2cypher")
    assert versions[-1] == "v12" and "v11" in versions
    v11 = PROMPT_PACK.get_prompt("nl2cypher", version="v11")
    # v11 is a registered version - its text must stay byte-for-byte (the AGE
    # wording is the bug v12 fixes, so its presence proves v11 wasn't edited).
    assert "openCypher query for Apache AGE" in v11.template
    v12 = PROMPT_PACK.get_prompt("nl2cypher")
    assert v12.notes and "Apache AGE" in v12.notes


def test_v12_template_targets_neo4j5_and_namespaced_act_ids():
    t = PROMPT_PACK.get_prompt("nl2cypher").template
    rendered = t.format(labels="Section, CaseLaw", rel_types="PART_OF, CORRESPONDS_TO")
    assert "Apache AGE" not in rendered
    assert "Neo4j 5" in rendered
    assert "Section, CaseLaw" in rendered and "PART_OF, CORRESPONDS_TO" in rendered
    # Neo4j-5 specifics
    assert "IS NOT NULL" in rendered and "exists(n.prop)" in rendered
    assert "id(n)" in rendered
    # Act-namespaced ids, matched in both forms
    assert "n.id ENDS WITH '__Section_12A'" in rendered
    # 1961 <-> 2025 routed to CORRESPONDS_TO, not SUPERSEDED_BY
    assert "(n)-[r:CORRESPONDS_TO]->(m:Section)" in rendered
    assert "replacing same-numbered sections of the 1961" not in rendered
    # The old bare exact-id example that matched nothing on a namespaced graph is gone
    assert "MATCH (n {id: 'Section_17'})" not in rendered


# --- truncation detection -----------------------------------------------------

@pytest.mark.parametrize("cypher", [
    "MATCH (n:CaseLaw) WHERE toLower(n.full_citation) CONTAINS 'engineering analysis' RETURN n LIMIT 5",
    "MATCH (n:Section)\nWHERE n.id = 'Section_90' OR n.id ENDS WITH '__Section_90'\n"
    "OPTIONAL MATCH (n)-[r:CORRESPONDS_TO]->(m:Section)\nRETURN n, r, m LIMIT 25",
    "MATCH (n:CaseLaw) RETURN count(n)",
    "MATCH (n) WHERE ANY(x IN n.legal_concepts WHERE toLower(x) CONTAINS 'demerger') RETURN n LIMIT 25",
    "MATCH (a)-[r]->(b) RETURN *",
    "MATCH (n) WHERE n.full_citation = 'Act No. 15 of 2024, and (more)' RETURN n",  # brackets/keywords in a literal
])
def test_complete_queries_pass(cypher):
    assert graphrag.cypher_incompleteness_reason(cypher) is None


@pytest.mark.parametrize("cypher,expected", [
    ("MATCH (n:CaseLaw) WHERE toLower(n.full_citation) CONTAINS 'engineering ana", "unterminated"),
    ("MATCH (overruling:CaseLaw)-[r:OVERRULES]->(overruled:CaseLaw)\nWHERE ANY(x IN overruling.legal_concepts "
     "WHERE toLower(x) CONTAINS 'royalty')", "no RETURN"),
    ("MATCH (n) WHERE ANY(x IN n.legal_concepts WHERE toLower(x) CONTAINS 'demerger'", "unclosed"),
    ("MATCH (n:Section) WHERE n.id = 'Section_9' OR", "ends mid-clause"),  # also no RETURN, either is fine
    ("MATCH (n:Section) RETURN n,", "ends mid-clause"),
    ("MATCH (n:Section) RETURN n ORDER BY", "ends mid-clause"),
    ("MATCH (n)-[r:CORRESPONDS_TO]->(m)) RETURN n", "unbalanced"),
])
def test_truncated_queries_are_flagged(cypher, expected):
    reason = graphrag.cypher_incompleteness_reason(cypher)
    assert reason is not None
    if expected != "ends mid-clause":
        assert expected in reason


def test_no_query_and_empty_are_not_flagged():
    assert graphrag.cypher_incompleteness_reason("NO_QUERY") is None
    assert graphrag.cypher_incompleteness_reason("") is None


# --- nl_to_cypher wiring --------------------------------------------------------

@pytest.fixture
def retriever():
    return graphrag.GraphRAGRetriever(DOMAIN_CONFIG, PROMPT_PACK)


SCHEMA = {"node_labels": ["Section", "CaseLaw"], "relationship_types": ["PART_OF", "CORRESPONDS_TO"]}


def test_nl_to_cypher_raises_token_cap_and_returns_complete_query(monkeypatch, retriever):
    seen = {}

    def fake_complete(system, user_content, max_tokens=1000, **kw):
        seen["max_tokens"] = max_tokens
        seen["system"] = system
        return "```cypher\nMATCH (n:Section) WHERE n.id ENDS WITH '__Section_90' RETURN n LIMIT 5;\n```"

    monkeypatch.setattr(graphrag.llm_client, "complete", fake_complete)
    out = retriever.nl_to_cypher("What does section 90 say?", SCHEMA)
    assert out == "MATCH (n:Section) WHERE n.id ENDS WITH '__Section_90' RETURN n LIMIT 5"
    assert seen["max_tokens"] == graphrag.NL2CYPHER_MAX_TOKENS >= 1000
    assert "Neo4j 5" in seen["system"]  # v12 is what's served


def test_truncated_llm_output_becomes_no_query_and_never_hits_neo4j(monkeypatch, retriever):
    monkeypatch.setattr(graphrag.llm_client, "complete",
                        lambda *a, **k: "MATCH (n:CaseLaw) WHERE toLower(n.full_citation) CONTAINS 'samsu")
    monkeypatch.setattr(graphrag, "get_schema_summary", lambda: SCHEMA)

    def boom():
        raise AssertionError("a truncated query must not be executed")

    monkeypatch.setattr(graphrag.neo4j_client, "get_session", boom)
    assert retriever.nl_to_cypher("q", SCHEMA) == "NO_QUERY"
    result = retriever.run_graph_retrieval("What did Samsung hold?")
    assert result["cypher"] is None and result["plain_rows"] == []


def test_retrieval_planner_cap_raised(monkeypatch):
    from framework.core.retrieve import retrieval_planner
    seen = {}

    def fake_complete(system, user_content, max_tokens=1000, **kw):
        seen["max_tokens"] = max_tokens
        return '{"tool": "VECTOR", "query_text": "x", "citation": "", "reasoning": "r"}'

    monkeypatch.setattr(retrieval_planner.llm_client, "complete", fake_complete)
    plan = retrieval_planner.plan_retrieval("q", SCHEMA)
    assert plan["tool"] == "VECTOR"
    assert seen["max_tokens"] >= 600
