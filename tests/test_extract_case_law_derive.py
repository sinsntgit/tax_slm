"""2026-10-14: NEW - extract_case_law.py's DETERMINISTIC derivations
(classify_forum_level, derive_jurisdiction_scope, derive_stance,
derive_retrieval_summary/keywords, resolve_section_citation) and
build_graph_from_extraction(), checked against the real Engineering Analysis
fixture graph (tests/fixtures/graphs/*Engineering_Analysis*_graph.json).

The fixture is the graph a real extraction produced, so an `extraction` dict
is reconstructed from it (citation/court/issues/findings/outcomes/keywords/
statutes/sections) and the derivations must reproduce the stored values.
No LLM call anywhere."""
import ast
import importlib
import json
import sys
from pathlib import Path

import pytest

CONVERTERS = Path(__file__).resolve().parents[1] / "domain_packs" / "tax" / "converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))  # the converters are scripts: `import chunking` is top-level
ecl = importlib.import_module("extract_case_law")


@pytest.fixture(scope="module")
def ea_graph(fixtures_dir=Path(__file__).resolve().parent / "fixtures"):
    path = next((fixtures_dir / "graphs").glob("*Engineering_Analysis*_graph.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def _as_list(v):
    return ast.literal_eval(v) if isinstance(v, str) else (v or [])


@pytest.fixture(scope="module")
def ea_extraction(ea_graph):
    """Rebuild the extraction-JSON shape from the stored graph."""
    ents = ea_graph["entities"]
    case = ents[0]["properties"]
    issues = [e for e in ents if e["type"] == "Issue"]
    concls = [e for e in ents if e["type"] == "Conclusion"]
    issue_text = [i["properties"]["hasText"] for i in issues]
    sections = {"interpreted": [], "applied": [], "referred": []}
    for r in ea_graph["relationships"]:
        if r["type"] == "INTERPRETS" and r["properties"].get("relation_type") in sections:
            sections[r["properties"]["relation_type"]].append(
                {"number": r["properties"]["cited_as"], "act": r["properties"]["act"]})
    return {
        "citation": case["full_citation"],
        "court": case["court"],
        "facts": case["hasText"],
        "final_outcome": case["final_outcome"],
        "keywords": _as_list(case["keywords"]),
        "statutes_cited": _as_list(case["statutes_cited"]),
        "issues": issue_text,
        "findings": [{"issue": t, "holding": c["properties"]["holding"], "reasoning": c["properties"]["hasText"]}
                     for t, c in zip(issue_text, concls)],
        "issue_outcomes": [{"issue": t, "winner": c["properties"]["winner"]} for t, c in zip(issue_text, concls)],
        "sections_interpreted": sections["interpreted"],
        "sections_applied": sections["applied"],
        "sections_referred": sections["referred"],
        "precedents_overruled": _as_list(case["precedents_overruled"]),
    }


# --- classify_forum_level / derive_jurisdiction_scope ------------------------------

@pytest.mark.parametrize("court,level", [
    ("Supreme Court of India", "Supreme Court"),
    ("High Court of Karnataka", "High Court"),
    ("Hon'ble Bombay High Court", "High Court"),
    ("Income Tax Appellate Tribunal, Delhi Bench", "Tribunal"),
    ("ITAT Mumbai", "Tribunal"),
    ("Authority for Advance Ruling (Income Tax)", "AAR"),
    ("AAR New Delhi", "AAR"),
    ("Commissioner of Income Tax (Appeals)-3", "Commissioner (Appeals)"),
    ("CIT(A) Pune", "Commissioner (Appeals)"),
    ("District Court, Pune", ""),
    ("", ""),
    (None, ""),
])
def test_classify_forum_level(court, level):
    assert ecl.classify_forum_level(court) == level


@pytest.mark.parametrize("court,level,scope", [
    ("Supreme Court of India", "Supreme Court", "All India"),
    ("Bombay High Court", "High Court", "Maharashtra & Goa (Bombay High Court)"),
    ("High Court of Punjab and Haryana", "High Court", "Punjab, Haryana & Chandigarh (Punjab & Haryana High Court)"),
    ("High Court of Telangana", "High Court", "Andhra Pradesh & Telangana High Court"),
    ("High Court of Uttarakhand", "High Court", ""),          # unknown HC -> unset, never guessed
    ("ITAT Delhi", "Tribunal", ""),                             # tribunals -> unset
    (None, "High Court", ""),
])
def test_derive_jurisdiction_scope(court, level, scope):
    assert ecl.derive_jurisdiction_scope(court, level) == scope


# --- derive_stance -----------------------------------------------------------------

@pytest.mark.parametrize("outcomes,stance,basis_start", [
    ([{"winner": "Assessee"}, {"winner": " assessee "}], "taxpayer_favorable", "2 issue(s) decided for assessee, 0"),
    ([{"winner": "revenue"}], "revenue_favorable", "0 issue(s) decided for assessee, 1"),
    ([{"winner": "assessee"}, {"winner": "revenue"}, {"winner": "remanded"}], "mixed", "1 issue(s) decided for assessee, 1"),
    ([{"winner": "partly"}, {}, None], "unknown", "No issue_outcomes"),
    ([], "unknown", "No issue_outcomes"),
])
def test_derive_stance(outcomes, stance, basis_start):
    got_stance, basis = ecl.derive_stance({"issue_outcomes": outcomes})
    assert got_stance == stance and basis.startswith(basis_start)


def test_derive_stance_tolerates_null_winner():
    """LLM JSON routinely contains `"winner": null` for an undecided issue -
    must be treated as 'no recognized winner', not crash the whole case."""
    assert ecl.derive_stance({"issue_outcomes": [{"winner": None}, {"winner": "revenue"}]})[0] == "revenue_favorable"
    assert ecl.derive_stance({"issue_outcomes": None})[0] == "unknown"


def test_derive_retrieval_summary_tolerates_null_fields():
    s = ecl.derive_retrieval_summary({"court": None, "findings": [{"issue": None, "holding": "Held X."},
                                                                  {"issue": "I", "holding": None}],
                                      "final_outcome": None}, "A v B")
    assert s == "A v B. Held X."


def test_derive_retrieval_summary_and_keywords_small():
    ext = {"court": "ITAT", "findings": [{"issue": "Is it royalty", "holding": "No."}, {"holding": "Only holding."},
                                         {"issue": "orphan issue"}],
           "final_outcome": "Allowed", "keywords": ["Royalty", "royalty ", "TDS"],
           "sections_interpreted": [{"number": "9(1)(vi)"}, "195"], "statutes_cited": [{"section": "195"}, "x"],
           "legal_principles": ["tds"]}
    assert ecl.derive_retrieval_summary(ext, "A v B") == \
        "A v B (ITAT). Is it royalty: No. Only holding. Final outcome: Allowed."
    assert ecl.derive_retrieval_keywords(ext) == "Royalty, TDS, 9(1)(vi), 195"


# --- resolve_section_citation ----------------------------------------------------------

@pytest.mark.parametrize("raw,expected_id", [
    ("9(1)(vi)", "ITA1961__Clause_9_1_vi"),
    ("2(19AA)", "ITA1961__SubSection_2_19AA"),
    ("195", "ITA1961__Section_195"),
    ("40(a)(ia)", "ITA1961__Clause_40_a_ia"),
    (" 10(23C)(iv)(a) ", "ITA1961__SubClause_10_23C_iv_a"),
    ("Article 12", None),
    ("General reference", None),
])
def test_resolve_section_citation(raw, expected_id):
    assert ecl.resolve_section_citation(raw, namespace="ITA1961")[0] == expected_id


def test_resolve_section_citation_unnamespaced_parts():
    eid, parts = ecl.resolve_section_citation("80IAC(2)")
    assert eid == "SubSection_80IAC_2"
    assert parts == {"section": "80IAC", "subsection": "2", "clause": None, "sub_clause": None}


# --- against the real fixture ------------------------------------------------------------

def test_fixture_derivations_reproduce_stored_values(ea_graph, ea_extraction):
    stored = ea_graph["entities"][0]["properties"]
    level = ecl.classify_forum_level(ea_extraction["court"])
    assert level == stored["forum_level"] == "Supreme Court"
    assert ecl.derive_jurisdiction_scope(ea_extraction["court"], level) == stored["jurisdiction_scope"] == "All India"
    assert ecl.derive_stance(ea_extraction) == (stored["stance"], stored["stanceBasis"])
    assert ecl.derive_retrieval_summary(ea_extraction, ea_extraction["citation"]) == stored["retrievalSummary"]
    # legal_principles aren't stored on the node, so the stored keyword string
    # must START with what keywords + section numbers + statutes produce.
    assert stored["retrievalKeywords"].startswith(ecl.derive_retrieval_keywords(ea_extraction))


def test_fixture_build_graph_core_props_and_issue_edges(ea_graph, ea_extraction):
    out = ecl.build_graph_from_extraction(ea_extraction, "ea.md", doc_meta={"document_id": "doc1"})
    case = out["entities"][0]
    assert case["type"] == "CaseLaw"
    assert case["id"] == ea_graph["entities"][0]["id"]
    for key in ("forum_level", "jurisdiction_scope", "stance", "stanceBasis", "retrievalSummary"):
        assert case["properties"][key] == ea_graph["entities"][0]["properties"][key], key
    raises = [r for r in out["relationships"] if r["properties"].get("relation_type") == "raises_issue"]
    assert len(raises) == 5
    winners = [e["properties"]["winner"] for e in out["entities"] if e["type"] == "Conclusion"]
    assert winners == ["assessee"] * 5


def test_non_income_tax_act_sections_do_not_get_wired_to_ita_nodes(ea_extraction):
    """Copyright Act s.14 ('exclusive rights') must NOT become an INTERPRETS
    edge to Income-tax Act Section 14 ('heads of income') - it is recorded in
    other_act_references instead."""
    out = ecl.build_graph_from_extraction(ea_extraction, "ea.md")
    statute_edges = [r for r in out["relationships"]
                     if r["type"] == "INTERPRETS" and r["properties"].get("relation_type") != "raises_issue"]
    assert statute_edges, "Income-tax Act citations must still be wired"
    assert all("income" in r["properties"]["act"].lower() for r in statute_edges)
    targets = {r["target"] for r in statute_edges}
    assert "ITA1961__Clause_9_1_vi" in targets and "ITA1961__Section_195" in targets
    assert "ITA1961__Section_14" not in targets and "ITA1961__Section_30" not in targets
    other = out["entities"][0]["properties"]["other_act_references"]
    assert "Copyright Act 1957 Section 14" in other


def test_unspecified_act_and_unhyphenated_income_tax_act_still_resolve():
    ext = {"citation": "A v B", "court": "ITAT Delhi",
           "sections_interpreted": [{"number": "68", "act": ""}, {"number": "69A", "act": "Income Tax Act, 1961"},
                                    {"number": "56(2)(viib)"}]}
    out = ecl.build_graph_from_extraction(ext, "x.md")
    targets = [r["target"] for r in out["relationships"] if r["type"] == "INTERPRETS"]
    assert targets == ["ITA1961__Section_68", "ITA1961__Section_69A", "ITA1961__Clause_56_2_viib"]
    assert out["entities"][0]["properties"]["other_act_references"] == []
