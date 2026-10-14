"""
2026-10-14: NEW (phase0/review-integrity) - entity IDENTITY for cross-run
resolution, following the review/phase0 findings (tests/test_review_phase0.py
items 1-2) and the audit of every label the converters emit.

Audit result this file pins (scores = entity_resolution.similarity on the real
id shapes, all >= HIGH_SIMILARITY 95 = silent auto-merge before the fix):
  * Loading the 20 fictitious opinion fixtures one after another collapsed
    OP-02..OP-20 onto FirmOpinion_OP_01 (each overwriting the previous one's
    properties) and re-pointed opinions' authority edges Rule 10B -> 10U,
    Rule 3(9) -> 3(8), Rule 11UB -> 11UA, CBDT Circular 6/2017 -> 7/2017.
  * Same hazard, no fixture hit yet: Article_12 -> _13 of a treaty (96.9),
    TaxTreaty_Austria -> _Australia (96.4), TreatyNotification_<c>_1 -> _2
    (96.2), LegalEvent ids differing by one digit (98.8), DataSource_<doc>
    ids (97.3), CaseCitation "(2021) 432 ITR 471" -> "... 417".
Fixes: exact-id labels (config.STRUCTURAL_LABELS), a canonical circular_number
key, a numeric-token veto on every fuzzy path (entity_resolution.key_guard),
and id-fallback keys may flag but never auto-merge (graph_loader).
"""
import json
import re
import sys
from pathlib import Path

import pytest

from domain_packs.tax import ingest_dedup
from domain_packs.tax.config import (DOMAIN_CONFIG, RESOLUTION_KEY_PROPERTIES, STRUCTURAL_LABELS,
                                     canonical_circular_number)
from framework.core.ingest import entity_resolution as er
from framework.core.ingest import provenance
from framework.core.ingest.graph_loader import GraphIngestionEngine

REPO = Path(__file__).resolve().parents[1]
CONVERTERS = REPO / "domain_packs" / "tax" / "converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

from test_provenance_loader import FakeGraph  # noqa: E402

FIXTURES = REPO / "tests" / "fixtures"
OPINIONS = sorted((FIXTURES / "opinions").glob("OP-*.md"))
CIRCULAR_GRAPH = FIXTURES / "graphs" / "143de5b6_circular-2-2026-pdf_graph.json"
CASELAW_GRAPH = next((FIXTURES / "graphs").glob("*Engineering_Analysis*_graph.json"))


@pytest.fixture
def graph():
    return FakeGraph()


@pytest.fixture
def engine():
    return GraphIngestionEngine(DOMAIN_CONFIG)


def _opinion_graph(path: Path) -> dict:
    import extract_opinion
    return extract_opinion.build_graph_from_opinion(
        extract_opinion.parse_opinion(path.read_text(encoding="utf-8"), path.name), path.name)


def _stub(entity_id, label, **props):
    return {"id": entity_id, "type": label, "properties": provenance.mark_stub(dict(props), cited_in="opinion_OP_99")}


def _real(entity_id, label, **props):
    return {"id": entity_id, "type": label, "properties": {"hasText": f"real text of {entity_id}", **props}}


# --- end to end: all 20 opinion fixtures through the real loader ----------------------

def test_twenty_opinions_load_without_any_cross_entity_merge(graph, engine):
    engine.load_entities(graph, json.loads(CIRCULAR_GRAPH.read_text())["entities"], mode="base")
    titles = {}
    for path in OPINIONS:
        g = _opinion_graph(path)
        own = g["entities"][0]
        assert own["type"] == "FirmOpinion"
        titles[own["id"]] = own["properties"]["title"]
        _labels, resolved = engine.load_entities(graph, g["entities"], mode="base")
        assert resolved == {}, f"{path.name}: {resolved}"
    assert len(titles) == 20
    for op_id, title in titles.items():
        assert graph.props(op_id)["title"] == title
    # the four authority pairs that used to be re-pointed onto a neighbour all exist separately
    for a, b in (("ITR1962__Rule_10B", "ITR1962__Rule_10U"), ("ITR1962__Rule_3_9", "ITR1962__Rule_3_8"),
                 ("ITR1962__Rule_11UB", "ITR1962__Rule_11UA"),
                 ("CBDTCircular_6_of_2017", "CBDTCircular_7_of_2017")):
        assert a in graph.nodes and b in graph.nodes


def _emitted_labels() -> set:
    labels = {e["type"] for p in OPINIONS for e in _opinion_graph(p)["entities"]}
    for path in (CIRCULAR_GRAPH, CASELAW_GRAPH):
        labels |= {e["type"] for e in json.loads(path.read_text())["entities"]
                   if not e["id"].startswith("DataSource_")}  # label = caller's --source-class, see below
    # Not covered by a fixture graph - minted by f-strings in chunking.py (Act path),
    # extract_treaty.py, extract_circular.py (_SUBTYPE_TO_CLASS) and extract_case_law.py.
    labels |= {"Act", "Section", "SubSection", "Clause", "SubClause", "Proviso", "Explanation",
               "AmendingAct", "DefinedTerm", "LegalEvent", "TaxTreaty", "Article", "TreatyNotification",
               "CBDTCircular", "ExplanatoryCircular", "ExtensionCircular", "TaxpayerArgument",
               "RevenueArgument", "GenericDocument", "DocumentSection"}
    return labels


def test_every_emitted_label_has_an_identity_rule():
    """Guard for the next converter: every label must be exact-id OR keyed on
    real properties - never left to the id-text fallback. (DataSource's label
    is the caller's --source-class, so it cannot be listed; it is covered by
    the id-fallback cap tested below.)"""
    missing = sorted(l for l in _emitted_labels()
                     if l not in STRUCTURAL_LABELS and l not in RESOLUTION_KEY_PROPERTIES)
    assert missing == []


# --- exact-id labels: a neighbouring authority is never a match ---------------------

@pytest.mark.parametrize("label, real_id, stub_id", [
    ("Article", "Article_13_TaxTreaty_USA", "Article_12_TaxTreaty_USA"),
    ("Rule", "ITR1962__Rule_11", "ITR1962__Rule_10"),
    ("Rule", "ITR1962__Rule_11UA", "ITR1962__Rule_11UB"),  # same digits - only exact-id catches it
    ("TaxTreaty", "TaxTreaty_Australia", "TaxTreaty_Austria"),
    ("TreatyNotification", "TreatyNotification_USA_1_GSR_990_E", "TreatyNotification_USA_2_GSR_990_E"),
    ("LegalEvent", "LegalEvent_supersession_CBDTCircular_8_of_2026_CBDTCircular_3_of_2025",
     "LegalEvent_supersession_CBDTCircular_8_of_2026_CBDTCircular_4_of_2025"),
    ("FirmOpinion", "FirmOpinion_OP_01", "FirmOpinion_OP_02"),
])
def test_exact_id_label_stub_never_rewired_to_neighbour(graph, engine, label, real_id, stub_id):
    assert er.similarity(er.resolution_key(label, {}, real_id), er.resolution_key(label, {}, stub_id)) >= 95
    engine.load_entities(graph, [_real(real_id, label)], mode="base")
    _labels, resolved = engine.load_entities(graph, [_stub(stub_id, label, full_citation=stub_id)], mode="base")
    assert resolved == {}
    assert graph.props(stub_id)["provenance"] == provenance.CITED_REFERENCE
    assert graph.props(real_id)["hasText"] == f"real text of {real_id}"


def test_exact_id_stub_still_joins_its_own_authority(graph, engine):
    engine.load_entities(graph, [_real("Article_12_TaxTreaty_USA", "Article", article_number="12")], mode="base")
    _labels, resolved = engine.load_entities(
        graph, [_stub("Article_12_TaxTreaty_USA", "Article", article_number="12", cited_as="Art 12 India-USA")],
        mode="base")
    assert resolved == {}
    props = graph.props("Article_12_TaxTreaty_USA")
    assert props["hasText"] == "real text of Article_12_TaxTreaty_USA"
    assert props["provenance"] == provenance.INGESTED_DOCUMENT and "cited_as" not in props


def test_real_opinion_upgrades_its_supersedes_stub(graph, engine):
    """FirmOpinion exact-id still lets an opinion's `supersedes:` stub be
    upgraded in place when the superseded opinion itself is ingested."""
    engine.load_entities(graph, [_stub("FirmOpinion_OP_01", "FirmOpinion", opinion_id="OP-01")], mode="base")
    engine.load_entities(graph, [_real("FirmOpinion_OP_01", "FirmOpinion", opinion_id="OP-01", title="FTS")],
                         mode="base")
    props = graph.props("FirmOpinion_OP_01")
    assert props["title"] == "FTS" and props["upgraded_from_provenance"] == provenance.CITED_REFERENCE


# --- CBDT circulars: canonical number key + numeric veto ----------------------------

@pytest.mark.parametrize("raw, expected", [
    ("8/2026", "8 of 2026"), ("8 of 2026", "8 of 2026"), ("Circular No. 08/2026", "8 of 2026"),
    ("CBDT Circular No. 8-2026", "8 of 2026"), ("Not clearly legible", "Not clearly legible"),
])
def test_canonical_circular_number(raw, expected):
    assert canonical_circular_number(raw) == expected


def test_circular_stub_resolves_onto_the_same_circular_only(graph, engine):
    # real document: id from the LLM-read "8/2026"; opinion stub: CBDTCircular_8_of_2026
    engine.load_entities(graph, [_real("CBDTCircular_8_2026", "CBDTCircular", circular_number="8/2026"),
                                 _real("CBDTCircular_3_2026", "CBDTCircular", circular_number="3/2026")],
                         mode="base")
    _labels, resolved = engine.load_entities(graph, [
        _stub("CBDTCircular_8_of_2026", "CBDTCircular", circular_number="8 of 2026"),
        _stub("CBDTCircular_18_of_2026", "CBDTCircular", circular_number="18 of 2026"),
    ], mode="base")
    assert resolved == {"CBDTCircular_8_of_2026": "CBDTCircular_8_2026"}
    assert graph.props("CBDTCircular_8_2026")["provenance"] == provenance.INGESTED_DOCUMENT
    assert "CBDTCircular_18_of_2026" in graph.nodes


# --- id-fallback labels: flag, never auto-merge --------------------------------------

def test_id_fallback_label_is_flagged_not_merged(graph, engine):
    label = "KnowledgeRepositoryItem"  # a DataSource --source-class: no key, not exact-id
    a, b = "DataSource_vodafone_international_holdings_part_a", "DataSource_vodafone_international_holdings_part_b"
    assert label not in STRUCTURAL_LABELS and label not in RESOLUTION_KEY_PROPERTIES
    assert er.similarity(er.resolution_key(label, {}, a), er.resolution_key(label, {}, b)) >= 95
    engine.load_entities(graph, [_real(a, label)], mode="base")
    _labels, resolved = engine.load_entities(graph, [_real(b, label)], mode="base")
    assert resolved == {}
    assert graph.props(b)["possibleDuplicateOf"] == a
    assert graph.props(a)["hasText"] == f"real text of {a}"


def test_keyed_label_still_auto_merges_same_entity(graph, engine):
    """The cap only applies to id-fallback keys: a keyed type (Issue) with the
    same substantive text re-extracted under a new id still merges."""
    text = "Whether payment for software licence is royalty under section 9(1)(vi)"
    engine.load_entities(graph, [_real("Issue_1_to_CaseLaw_X", "Issue", full_citation="X", hasText=text)],
                         mode="base")
    _labels, resolved = engine.load_entities(
        graph, [_real("Issue_1_to_CaseLaw_X_v2", "Issue", full_citation="X", hasText=text)], mode="base")
    assert resolved == {"Issue_1_to_CaseLaw_X_v2": "Issue_1_to_CaseLaw_X"}


# --- numeric veto on every fuzzy path ----------------------------------------------

def test_same_number_tokens():
    assert er.same_number_tokens("rule 10 1", "1 rule 10")
    assert not er.same_number_tokens("article 12", "article 13")
    assert not er.same_number_tokens("432 itr 471", "432 itr 417")
    assert er.same_number_tokens("no digits", "none here")


@pytest.mark.parametrize("label, a, b", [
    ("CaseCitation", {"full_citation": "(2021) 432 ITR 471 (SC)", "reporter_name": "ITR"},
     {"full_citation": "(2021) 432 ITR 417 (SC)", "reporter_name": "ITR"}),
    ("AmendingAct", {"amendingActNo": 5, "amendingActYear": 2024}, {"amendingActNo": 15, "amendingActYear": 2024}),
])
def test_key_guard_vetoes_differing_numbers(label, a, b):
    ka, kb = (er.resolution_key(label, p, "x", RESOLUTION_KEY_PROPERTIES) for p in (a, b))
    assert not er.key_guard(label, RESOLUTION_KEY_PROPERTIES)(ka, kb)
    assert er.best_match(ka, [("other", kb)], compatible=er.key_guard(label, RESOLUTION_KEY_PROPERTIES)) == (None, 0.0)


def test_key_guard_keeps_the_caselaw_designation_veto():
    rk = RESOLUTION_KEY_PROPERTIES
    guard = er.key_guard("CaseLaw", rk)
    ito = er.resolution_key("CaseLaw", {"case_name": "Samsung Electronics Co Ltd v. ITO"}, "a", rk)
    cit = er.resolution_key("CaseLaw", {"case_name": "Samsung Electronics Co Ltd v. CIT"}, "b", rk)
    same = er.resolution_key("CaseLaw", {"case_name": "Samsung Electronics Co. Ltd. vs ITO"}, "c", rk)
    assert not guard(ito, cit) and guard(ito, same)


def test_resolve_within_batch_applies_numeric_veto():
    keys = {"LegalConcept": ["name"]}
    ents = [{"id": "LC_a12", "type": "LegalConcept", "properties": {"name": "Royalty under Article 12"}},
            {"id": "LC_a13", "type": "LegalConcept", "properties": {"name": "Royalty under Article 13"}}]
    kept, alias, _flags = er.resolve_within_batch(ents, keys)
    assert alias == {} and {e["id"] for e in kept} == {"LC_a12", "LC_a13"}


# --- merge_duplicate_caselaw: provenance-aware canonical + anchored re-point filter ---

class _DegreeSession:
    def __init__(self, degrees):
        self.degrees = degrees

    def run(self, q, **p):
        deg = self.degrees[p["id"]]

        class _R:
            def single(self_inner):
                return {"deg": deg}
        return _R()


def test_pick_canonical_legacy_stub_without_provenance_loses():
    from framework.cli import merge_duplicate_caselaw as merge
    legacy_stub = {"id": "CaseLaw_A", "provenance": None, "extraction_notes": "Stub created from a citation in X"}
    real = {"id": "CaseLaw_A_2012", "provenance": None, "extraction_notes": "Extracted from a.pdf"}
    assert merge._pick_canonical(_DegreeSession({"CaseLaw_A": 40, "CaseLaw_A_2012": 2}),
                                 [legacy_stub, real])["id"] == "CaseLaw_A_2012"


def test_pick_canonical_degree_still_decides_between_ingested_nodes():
    from framework.cli import merge_duplicate_caselaw as merge
    a = {"id": "CaseLaw_A", "provenance": provenance.INGESTED_DOCUMENT, "ingestion_date": "2026-01-01"}
    b = {"id": "CaseLaw_B", "provenance": provenance.INGESTED_DOCUMENT, "ingestion_date": "2026-02-01"}
    assert merge._pick_canonical(_DegreeSession({"CaseLaw_A": 3, "CaseLaw_B": 7}), [a, b])["id"] == "CaseLaw_B"


def test_file_key_ignores_opinion_stub_with_node_id():
    from framework.cli import merge_duplicate_caselaw as merge
    stub = {"id": "CaseLaw_X", "node_id": "CaseLaw_X", "provenance": provenance.CITED_REFERENCE,
            "source_file_sha256": "f" * 64}
    assert merge._file_key(stub) is None


class _RepointCapture:
    def __init__(self):
        self.queries = []

    def run(self, q, **p):
        self.queries.append((q, p))

        class _R:
            def single(self_inner):
                return {"c": 0, "p": {}}

            def __iter__(self_inner):
                return iter([])
        return _R()


def test_repoint_exclusion_uses_the_same_anchored_predicates():
    from framework.cli import merge_duplicate_caselaw as merge
    s = _RepointCapture()
    dup = "CaseLaw_Samsung_Electronics_Co_Ltd"
    merge._merge_duplicate_into_canonical(s, dup, "CaseLaw_Canon", dry_run=True, drop_subtree=True)
    repoint = [(q, p) for q, p in s.queries if "RETURN type(r)" in q]
    assert len(repoint) == 2
    for q, p in repoint:
        assert "=~ ('caselaw_' + $id_re + '_[0-9]+')" in q and "EVENT_INSTRUMENT" in q
        assert "STARTS WITH" not in q and "CONTAINS" not in q
        assert p["id_re"] == re.escape(dup)


# --- ingest_dedup stub detection --------------------------------------------------

@pytest.mark.parametrize("props, is_stub", [
    ({"provenance": provenance.CITED_REFERENCE, "extraction_notes": ""}, True),
    ({"provenance": provenance.MODEL_SUGGESTED}, True),
    ({"extraction_notes": "Stub created from a reference in X"}, True),  # legacy, pre-provenance
    ({"provenance": provenance.INGESTED_DOCUMENT, "extraction_notes": "Extracted from x.pdf"}, False),
    ({"extraction_notes": ""}, False),  # missing provenance = ingested_document (load default)
])
def test_is_stub(props, is_stub):
    assert ingest_dedup._is_stub({"id": "x", "type": "CaseLaw", "properties": props}) is is_stub


def test_main_case_law_id_ignores_stub_with_node_id():
    graph = {"entities": [
        {"id": "CaseLaw_Real", "type": "CaseLaw", "properties": {"node_id": "CaseLaw_Real"}},
        {"id": "CaseLaw_Cited", "type": "CaseLaw",
         "properties": provenance.mark_stub({"node_id": "CaseLaw_Cited"})},
    ]}
    assert ingest_dedup.main_case_law_id(graph) == "CaseLaw_Real"
