"""
2026-10-14: NEW (phase0/followups) - deterministic case-name normalizer for the
CaseLaw RESOLUTION key (domain_packs/tax/case_names.py, registered through
entity_resolution.KeyProperties in domain_packs/tax/config.py), tested with the
real names in tests/fixtures: the opinions' case_law_cited lists and the
Engineering Analysis judgment's graph fixture. Cited-case stubs must converge on
the ingested judgment; genuinely different cases must not merge; node ids never
change.
"""
import itertools
import json
import re
import sys
from pathlib import Path

import pytest

from domain_packs.tax.case_names import normalize_case_name
from domain_packs.tax.config import DOMAIN_CONFIG, RESOLUTION_KEY_PROPERTIES
from framework.core.ingest import entity_resolution as er

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
CONVERTERS = REPO / "domain_packs" / "tax" / "converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

EA_GRAPH = FIXTURES / "graphs" / ("325ebe7a_Engineering_Analysis_Centre_Of_vs_The_Commissioner_Of_Income_Tax"
                                  "_on_2_March_2021_graph.json")
EA_STUB = "Engineering Analysis Centre of Excellence Pvt Ltd v. CIT, (2021) 432 ITR 471 (SC)"


def _graph_case_names():
    g = json.loads(EA_GRAPH.read_text())
    return [e["properties"]["case_name"] for e in g["entities"] if e["type"] == "CaseLaw"]


def _opinion_case_names():
    names = []
    for f in sorted((FIXTURES / "opinions").glob("*.md")):
        m = re.search(r"case_law_cited:\s*\n((?:\s+- .*\n)+)", f.read_text())
        if m:
            names += [line.strip()[2:].strip().strip('"') for line in m.group(1).splitlines()]
    return names


def _key(name):
    return er.resolution_key("CaseLaw", {"case_name": name}, "CaseLaw_x", DOMAIN_CONFIG.resolution_key_properties)


def _score(a, b):
    return er.similarity(_key(a), _key(b))


def _would_merge(a, b):
    """What the loader/batch dedup does: the type's compatibility veto, then
    the fuzzy score against HIGH_SIMILARITY."""
    guard = er.key_guard("CaseLaw", DOMAIN_CONFIG.resolution_key_properties)
    _, score = er.best_match(_key(a), [("x", _key(b))], compatible=guard)
    return score >= er.HIGH_SIMILARITY


EA_REAL = next(n for n in _graph_case_names() if n.startswith("Engineering Analysis"))


def test_fixture_names_are_what_this_test_assumes():
    assert EA_STUB in _opinion_case_names()
    assert EA_REAL.startswith("Engineering Analysis Centre of Excellence Private Limited v. The Commissioner of Income Tax")


def test_engineering_analysis_stub_converges_on_the_ingested_judgment():
    assert normalize_case_name(EA_STUB) == normalize_case_name(EA_REAL) == \
        "engineering analysis centre of excellence ltd v cit"
    assert _score(EA_STUB, EA_REAL) >= er.HIGH_SIMILARITY
    # the bug being fixed: the plain key scored well below the merge bar
    plain = {"CaseLaw": ["case_name"]}
    raw = er.similarity(er.resolution_key("CaseLaw", {"case_name": EA_STUB}, "a", plain),
                        er.resolution_key("CaseLaw", {"case_name": EA_REAL}, "b", plain))
    assert raw < er.HIGH_SIMILARITY


@pytest.mark.parametrize("cited, judgment", [
    # opinion fixture vs the EA judgment's own precedent list (different reporters, UOI vs Union of India)
    ("Vodafone International Holdings BV v. UOI, (2012) 341 ITR 1 (SC)",
     "Vodafone International Holdings BV v. Union of India, (2012) 6 SCC 613"),
    # same case, one with an editorial note
    ("CIT v. Samsung Electronics Co. Ltd., (2012) 345 ITR 494",
     "CIT v. Samsung Electronics Co. Ltd., (2012) 345 ITR 494 (overruled on substantive legal question of "
     "whether computer software payments constitute royalty)"),
    # spelling variants of the same parties
    ("CIT v. De Beers India Minerals (P) Ltd, (2012) 346 ITR 467 (Kar)",
     "Commissioner of Income-tax v. M/s. De Beers India Minerals Private Limited, [2012] 21 taxmann.com 214 (Karnataka)"),
    ("Magneti Marelli Powertrain India (P) Ltd v. DCIT",
     "Magneti Marelli Powertrain India Pvt. Ltd. vs. Deputy Commissioner of Income Tax, Circle 1(1), ITA 1234/2015"),
    ("Director of Income Tax v. A.P. Moller Maersk AS, (2017) 5 SCC 651",
     "DIT vs A P Moller Maersk A.S., (2017) 392 ITR 186 (SC)"),
    ("PILCOM v. CIT, West Bengal-VII, 2020 SCC Online SC 426", "PILCOM v. Commissioner of Income Tax (2020)"),
    ("Indo Rama Synthetics (I) Ltd v. CIT, (2011) 330 ITR 363 (Del)",
     "Indo Rama Synthetics (India) Limited versus CIT [2011] 330 ITR 363 (Delhi)"),
])
def test_variants_of_the_same_case_converge(cited, judgment):
    assert normalize_case_name(cited) == normalize_case_name(judgment)
    assert _would_merge(cited, judgment)


@pytest.mark.parametrize("a, b", [
    ("CIT v. Samsung Electronics Co. Ltd., (2012) 345 ITR 494", "CIT v. ZTE Corporation, (2017) 392 ITR 80"),
    ("Director of Income Tax v. Ericsson A.B., (2012) 343 ITR 470",
     "Director of Income Tax v. Nokia Networks OY, (2013) 358 ITR 259"),
    ("Union of India v. Ashish Agarwal, (2022) 444 ITR 1 (SC)", "Union of India v. Rajeev Bansal (2024) (SC)"),
    # different revenue officers are NOT folded together
    ("Agro Portfolio (P) Ltd v. ITO (2018)", "Agro Portfolio (P) Ltd v. CIT (2018)"),
])
def test_genuinely_different_cases_do_not_merge(a, b):
    assert normalize_case_name(a) != normalize_case_name(b)
    assert not _would_merge(a, b)


def test_designation_veto_where_fuzzy_alone_would_merge():
    a, b = "Agro Portfolio (P) Ltd v. ITO (2018)", "Agro Portfolio (P) Ltd v. CIT (2018)"
    assert _score(a, b) >= er.HIGH_SIMILARITY          # the fuzzy ratio alone is fooled...
    assert not _would_merge(a, b)                        # ...the same_designations veto is not
    assert _would_merge(EA_STUB, EA_REAL)


def test_no_two_distinct_fixture_cases_auto_merge():
    """Every pair of fixture case names that are NOT the same case (by the
    known same-case groups below) stays below the auto-merge threshold."""
    names = sorted(set(_graph_case_names() + _opinion_case_names()))
    same_case = [
        {EA_REAL, EA_STUB},
        {n for n in names if n.startswith("Vodafone International Holdings")},
        {n for n in names if n.startswith("CIT v. Samsung Electronics")},
    ]
    assert all(len(g) == 2 for g in same_case)

    def same(a, b):
        return any(a in g and b in g for g in same_case)

    offenders = [(a, b, _score(a, b)) for a, b in itertools.combinations(names, 2)
                 if not same(a, b) and _would_merge(a, b)]
    assert offenders == []


def test_single_party_documents_are_not_collapsed_to_the_authority():
    # "CBDT Circular ..." must keep its number, never reduce to "cbdt"
    a, b = "CBDT Circular No. 7/2017 (GAAR clarifications)", "CBDT Circular No. 18/2020 (ESOP deferral FAQs)"
    assert normalize_case_name(a) != normalize_case_name(b)
    assert "7" in normalize_case_name(a) and not _would_merge(a, b)


def test_key_properties_is_still_a_plain_property_list():
    spec = RESOLUTION_KEY_PROPERTIES["CaseLaw"]
    assert spec == ["case_name"] and list(spec) == ["case_name"]
    assert spec.normalizer is normalize_case_name
    # a normalizer failure never breaks resolution - falls back to the raw text
    broken = {"CaseLaw": er.KeyProperties(["case_name"], normalizer=lambda s: 1 / 0)}
    assert er.resolution_key("CaseLaw", {"case_name": "A v. B"}, "x", broken) == "caselaw a v b"
    assert normalize_case_name("") == "" and normalize_case_name(None) == ""


# --- end to end through the real graph loader --------------------------------

def _ea_real_entity():
    g = json.loads(EA_GRAPH.read_text())
    return next(e for e in g["entities"] if e["type"] == "CaseLaw" and e["properties"]["case_name"] == EA_REAL)


def _op18_graph():
    import extract_opinion
    path = next((FIXTURES / "opinions").glob("OP-18*.md"))
    return extract_opinion.build_graph_from_opinion(extract_opinion.parse_opinion(path.read_text(), path.name),
                                                    path.name)


def _stub_from(graph):
    return next(e for e in graph["entities"] if e["type"] == "CaseLaw")


def test_opinion_stub_joins_already_ingested_judgment():
    from test_provenance_loader import FakeGraph
    from framework.core.ingest.graph_loader import GraphIngestionEngine

    fake, engine = FakeGraph(), GraphIngestionEngine(DOMAIN_CONFIG)
    real = _ea_real_entity()
    engine.load_entities(fake, [real], mode="base")
    op = _op18_graph()
    stub = _stub_from(op)
    assert stub["id"] != real["id"] and stub["properties"]["provenance"] == "cited_reference"
    _, resolved = engine.load_entities(fake, op["entities"], mode="base")
    assert resolved[stub["id"]] == real["id"]                 # stub resolved onto the judgment's (unchanged) id
    assert stub["id"] not in fake.nodes                       # no orphan stub node
    assert fake.props(real["id"])["case_name"] == EA_REAL     # real content not overwritten by the stub


def test_judgment_ingested_after_stub_upgrades_it():
    from test_provenance_loader import FakeGraph
    from framework.core.ingest.graph_loader import GraphIngestionEngine

    fake, engine = FakeGraph(), GraphIngestionEngine(DOMAIN_CONFIG)
    op = _op18_graph()
    stub = _stub_from(op)
    engine.load_entities(fake, op["entities"], mode="base")
    real = _ea_real_entity()
    _, resolved = engine.load_entities(fake, [real], mode="base")
    assert resolved[real["id"]] == stub["id"]                 # converged on the existing node's id
    node = fake.props(stub["id"])
    assert node["case_name"] == EA_REAL and node["provenance"] == "ingested_document"
    assert real["id"] not in fake.nodes


def test_samsung_and_zte_stay_separate_nodes():
    from test_provenance_loader import FakeGraph
    from framework.core.ingest.graph_loader import GraphIngestionEngine

    g = json.loads(EA_GRAPH.read_text())
    pick = [e for e in g["entities"] if e["type"] == "CaseLaw"
            and e["properties"]["case_name"].startswith(("CIT v. ZTE", "CIT v. Samsung Electronics Co. Ltd., (2012) 345 ITR 494"))]
    zte = next(e for e in pick if "ZTE" in e["id"])
    samsung = next(e for e in pick if "Samsung" in e["id"] and "overruled" not in e["id"])
    fake, engine = FakeGraph(), GraphIngestionEngine(DOMAIN_CONFIG)
    engine.load_entities(fake, [samsung], mode="base")
    _, resolved = engine.load_entities(fake, [zte], mode="base")
    assert resolved == {} and {samsung["id"], zte["id"]} <= set(fake.nodes)


def test_within_batch_dedup_merges_variants_but_not_different_cases():
    ents = [
        {"id": "CaseLaw_a", "type": "CaseLaw", "properties": {"case_name": EA_REAL}},
        {"id": "CaseLaw_b", "type": "CaseLaw", "properties": {"case_name": EA_STUB}},
        {"id": "CaseLaw_c", "type": "CaseLaw", "properties": {"case_name": "CIT v. ZTE Corporation, (2017) 392 ITR 80"}},
        {"id": "CaseLaw_d", "type": "CaseLaw",
         "properties": {"case_name": "CIT v. Samsung Electronics Co. Ltd., (2012) 345 ITR 494"}},
        {"id": "CaseLaw_e", "type": "CaseLaw", "properties": {"case_name": "Agro Portfolio (P) Ltd v. ITO (2018)"}},
        {"id": "CaseLaw_f", "type": "CaseLaw", "properties": {"case_name": "Agro Portfolio (P) Ltd v. CIT (2018)"}},
    ]
    kept, alias_map, _ = er.resolve_within_batch(ents, DOMAIN_CONFIG.resolution_key_properties)
    assert alias_map == {"CaseLaw_b": "CaseLaw_a"}
    assert sorted(e["id"] for e in kept) == ["CaseLaw_a", "CaseLaw_c", "CaseLaw_d", "CaseLaw_e", "CaseLaw_f"]


def test_merge_duplicate_caselaw_cli_respects_the_veto():
    from framework.cli import merge_duplicate_caselaw as m
    nodes = [{"id": "a", "case_name": EA_REAL}, {"id": "b", "case_name": EA_STUB},
             {"id": "e", "case_name": "Agro Portfolio (P) Ltd v. ITO (2018)"},
             {"id": "f", "case_name": "Agro Portfolio (P) Ltd v. CIT (2018)"}]
    clusters = m._cluster(nodes, DOMAIN_CONFIG.resolution_key_properties, er.HIGH_SIMILARITY)
    assert [sorted(n["id"] for n in c) for c in clusters] == [["a", "b"]]
