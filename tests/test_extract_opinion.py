"""
2026-10-14: NEW (phase0/opinions) - domain_packs/tax/converters/extract_opinion.py
against ALL 20 fictitious fixtures in tests/fixtures/opinions/.

tests/fixtures/opinion_snapshots.json is a reviewed snapshot of the key
fields per opinion (stub ids, relationship types + provenance, confidence
band, ISO date, chunked sections, ambiguities). If a deliberate converter
change alters it, regenerate it and REVIEW the diff - the hand-written tests
below pin the legally important cases independently of the snapshot.
Regenerate with: python tests/test_extract_opinion.py --regenerate-snapshot
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONVERTERS = REPO / "domain_packs" / "tax" / "converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

import chunking  # noqa: E402
import extract_case_law  # noqa: E402
import extract_opinion as eo  # noqa: E402
from framework.core.ingest import provenance  # noqa: E402

OPINIONS = sorted((REPO / "tests" / "fixtures" / "opinions").glob("OP-*.md"))
SNAPSHOT = json.loads((REPO / "tests" / "fixtures" / "opinion_snapshots.json").read_text(encoding="utf-8"))
SCHEMA = json.loads((REPO / "domain_packs" / "tax" / "ontology" / "schema.json").read_text(encoding="utf-8"))


def _build(path: Path, **kw):
    parsed = eo.parse_opinion(path.read_text(encoding="utf-8"), path.name)
    return parsed, eo.build_graph_from_opinion(parsed, path.name, source_md_path=str(path), **kw)


def _by_prefix(prefix):
    return next(p for p in OPINIONS if p.name.startswith(prefix))


def test_all_twenty_fixtures_present():
    assert len(OPINIONS) == 20


def snapshot(graph: dict) -> dict:
    op = graph["entities"][0]["properties"]
    return {
        "node_id": graph["entities"][0]["id"],
        "opinion_id": op["opinion_id"],
        "date": op.get("date"),
        "category": op["category"],
        "confidence_level": op["confidence_level"],
        "confidence_band": op["confidence_band"],
        "statute_regime": op["statute_regime"],
        "chunk_sections": [c["element_type"] for c in graph["chunks"]],
        "stubs": sorted(f"{e['type']}:{e['id']}" for e in graph["entities"][1:]),
        "relationships": sorted(f"{r['source']}-[{r['type']}|{r['properties']['provenance']}]->{r['target']}"
                                for r in graph["relationships"]),
        "other_act_references": op["other_act_references"],
        "other_authorities_cited": op["other_authorities_cited"],
        "ambiguity_types": sorted(a["type"] for a in graph["ambiguities"]),
    }


@pytest.mark.parametrize("path", OPINIONS, ids=lambda p: p.name[:5])
def test_snapshot_key_fields(path):
    _parsed, graph = _build(path)
    assert snapshot(graph) == SNAPSHOT[path.name]


@pytest.mark.parametrize("path", OPINIONS, ids=lambda p: p.name[:5])
def test_every_fixture_parses_validates_and_is_provenance_marked(path):
    parsed, graph = _build(path)
    opinion = graph["entities"][0]
    props = opinion["properties"]
    assert opinion["type"] == "FirmOpinion"
    assert eo.validate_against_schema(graph, SCHEMA) == []
    # frontmatter-derived fields
    fm = parsed["frontmatter"]
    assert props["opinion_id"] == fm["opinion_id"]
    assert props["opinion_date"] == fm["date"] and re.fullmatch(r"\d{4}-\d{2}-\d{2}", props["date"])
    assert props["effective_from"] == props["date"]
    assert props["confidence_level"] == fm["confidence_level"] and props["confidence_band"]
    assert props["category"] == fm["category"]
    assert props["is_fictitious"] is True
    # retrieval-facing text properties (config.py opinion_expert_views text_properties)
    for key in ("facts", "questions", "analysis", "opinion_conclusion", "reliance_clauses"):
        assert props[key], key
    assert "Confidence level" not in props["opinion_conclusion"]
    # document metadata like the other converters
    assert props["document_id"] and props["document_name"] and props["file_type"] == "firm_opinion"
    assert props["sourceMdPath"] == str(path)
    # provenance on EVERYTHING
    assert props["provenance"] == provenance.INGESTED_DOCUMENT
    for e in graph["entities"][1:]:
        assert e["properties"]["provenance"] == provenance.CITED_REFERENCE, e["id"]
        assert e["properties"]["sourceMdPath"] == "", "a stub must never hydrate the citing opinion's text"
        assert "document_id" not in e["properties"]
        assert e["properties"]["confidence"] == eo.STUB_CONFIDENCE
    for r in graph["relationships"]:
        assert r["properties"]["provenance"] in (provenance.INGESTED_DOCUMENT, provenance.CITED_REFERENCE)
        if r["source"] == opinion["id"]:
            assert r["properties"]["provenance"] == provenance.INGESTED_DOCUMENT
        else:
            assert r["type"] == "PART_OF" and r["properties"]["provenance"] == provenance.CITED_REFERENCE
    # every stub is reachable from the opinion
    targets = {r["target"] for r in graph["relationships"] if r["source"] == opinion["id"]}
    stub_ids = {e["id"] for e in graph["entities"][1:]}
    assert stub_ids - targets <= {e["id"] for e in graph["entities"] if e["type"] == "TaxTreaty"}


@pytest.mark.parametrize("path", OPINIONS, ids=lambda p: p.name[:5])
def test_chunks_link_to_opinion_and_skip_boilerplate(path):
    parsed, graph = _build(path)
    opinion = graph["entities"][0]
    chunks = graph["chunks"]
    ids = [c["chunk_id"] for c in chunks]
    assert len(ids) == len(set(ids)) and all(i.startswith(f"opinion_{opinion['id']}_") for i in ids)
    assert opinion["properties"]["source_chunk_ids"].split(",") == ids
    assert {c["entity_id"] for c in chunks} == {opinion["id"]}  # opinion_agent joins on Chunk.entity_id
    assert {"facts", "questions", "analysis", "opinion_conclusion"} <= {c["element_type"] for c in chunks}
    assert not eo.BOILERPLATE_SECTION_KEYS & {c["element_type"] for c in chunks}
    conclusion = next(c for c in chunks if c["element_type"] == "opinion_conclusion")
    assert conclusion["text"].endswith(f"Confidence level: {parsed['confidence_level']}")
    for c in chunks:
        assert c["text"].strip() and c["file_type"] == "firm_opinion" and c["short_title"].startswith(parsed["opinion_id"])


@pytest.mark.parametrize("path", OPINIONS, ids=lambda p: p.name[:5])
def test_party_names_pseudonymised_in_text_but_kept_in_masked_fields(path):
    parsed, graph = _build(path)
    props = graph["entities"][0]["properties"]
    assert props["client_name"] == parsed["client"] and props["partner_name"] == parsed["partner"]
    names = [parsed["client"], parsed["partner"]]
    if eo.looks_like_party_name(parsed["counterparty"]):
        names.append(re.sub(r"\s*\([^)]*\)\s*$", "", parsed["counterparty"]))
        assert props["counterparty_name"] == parsed["counterparty"]
    texts = [c["text"] for c in graph["chunks"]] + [c["full_citation"] for c in graph["chunks"]] + [
        props[k] for k in ("facts", "questions", "analysis", "opinion_conclusion", "reliance_clauses",
                           "purpose_and_basis", "limitations", "hasText", "title", "full_citation")]
    for name in names:
        for t in texts:
            assert name not in t, f"{name!r} leaked into text"
    assert "[CLIENT]" in props["reliance_clauses"]


def test_pii_config_masks_opinion_party_fields():
    from domain_packs.tax.config import ALWAYS_MASKED_FIELDS, DOMAIN_CONFIG
    from framework.core.retrieve.pii import PiiMasker
    assert {"partner_name", "client_industry", "client_name", "counterparty_name"} <= set(ALWAYS_MASKED_FIELDS["FirmOpinion"])
    _parsed, graph = _build(_by_prefix("OP-01"))
    masked = PiiMasker(DOMAIN_CONFIG).mask_node_properties("FirmOpinion", graph["entities"][0]["properties"])
    for key in ("client_name", "counterparty_name", "partner_name"):
        assert "REDACTED" in masked[key]
    assert masked["firm_name"] == "Everest Tax Advisory LLP"


def test_redaction_never_mangles_cited_authorities():
    pairs = eo.build_redactions("Apollo Holdings Private Limited", "", "",
                                protected="Apollo Tyres Ltd v. CIT, (2002) 255 ITR 273 (SC)")
    assert ("Apollo", "[CLIENT]") not in pairs
    assert eo.redact("Apollo Holdings Private Limited relied on Apollo Tyres", pairs) == \
        "[CLIENT] relied on Apollo Tyres"
    assert eo.party_variants("Resident angel investor") == []
    assert eo.party_variants("Not applicable") == []


# --- id parity with the other converters -----------------------------------------

def test_section_ids_match_case_law_resolver_and_chunking():
    _parsed, graph = _build(_by_prefix("OP-01"))
    ids = {e["id"]: e["type"] for e in graph["entities"]}
    expected, _ = extract_case_law.resolve_section_citation("9(1)(vii)", namespace="ITA1961")
    assert expected == chunking.generate_entity_id("9", "1", "vii", namespace="ITA1961") == "ITA1961__Clause_9_1_vii"
    assert ids[expected] == "Clause"
    assert ids["Explanation_2_to_ITA1961__Clause_9_1_vii"] == "Explanation"
    assert ids["ITA1961__SubSection_90_2"] == "SubSection"


def test_case_stub_ids_use_case_law_precedent_scheme():
    for path in OPINIONS:
        parsed, graph = _build(path)
        for e in graph["entities"]:
            if e["type"] == "CaseLaw":
                assert e["id"] == f"CaseLaw_{extract_case_law._slug(e['properties']['case_name'])}"
                assert e["properties"]["case_name"] in parsed["cases_cited_raw"]


@pytest.mark.parametrize("raw,numbers,expl,other", [
    ("Section 9(1)(vii) & Explanation 2", ["9(1)(vii)"], "2", None),
    ("Section 9(1)(i), Explanation 5", ["9(1)(i)"], "5", None),
    ("Explanation 1 to Section 115JB", ["115JB"], "1", None),
    ("Section 56(2)(viib)", ["56(2)(viib)"], None, None),
    ("Section 2(29A) / 2(42A)", ["2(29A)", "2(42A)"], None, None),
    ("Section 115JB(2A)-(2C)", ["115JB(2A)", "115JB(2B)", "115JB(2C)"], None, None),
    ("Section 115QA (as amended)", ["115QA"], None, None),
    ("Section 80-IAC", ["80IAC"], None, None),
    ("Section 194-IC", ["194IC"], None, None),
    ("Section 165, Finance Act, 2016", [], None, "Finance Act, 2016"),
    ("Section 6(3) of the Income-tax Act, 1961", ["6(3)"], None, None),
])
def test_split_section_citation(raw, numbers, expl, other):
    assert eo.split_section_citation(raw) == (numbers, expl, other)


@pytest.mark.parametrize("raw,kind", [
    ("Section 195", "section"), ("Explanation 1 to Section 115JB", "section"), ("Rule 3(8) / 3(9)", "rule"),
    ("CBDT Circular No. 7/2017 (GAAR clarifications)", "circular"), ("CBDT Circular No. 6/2017", "circular"),
    ("India-USA DTAA, Article 12 (Fees for Included Services)", "treaty"), ("India-Mauritius DTAA", "treaty"),
    ("Union of India v. Ashish Agarwal, (2022) 444 ITR 1 (SC)", "case"), ("Not applicable", "skip"),
    ("Not applicable / no specific provisions beyond the Income-tax Act general scheme.", "skip"),
    ("TOLA, 2020", "other"), ("CBDT FAQs on Equalisation Levy", "other"), ("DPIIT Notification G.S.R. 127(E)", "other"),
])
def test_classify_citation(raw, kind):
    assert eo.classify_citation(raw) == kind


def test_mixed_lists_are_classified_by_content_not_by_field():
    # OP-04 lists a CBDT circular under case_law_cited; OP-11 lists one under income_tax_sections.
    for prefix, circ in (("OP-04", "CBDTCircular_7_of_2017"), ("OP-11", "CBDTCircular_6_of_2017")):
        _parsed, graph = _build(_by_prefix(prefix))
        rels = {(r["type"], r["target"]) for r in graph["relationships"]}
        assert ("RELIES_ON_AUTHORITY", circ) in rels
        assert not any(e["type"] == "CaseLaw" and "Circular" in e["id"] for e in graph["entities"])


def test_other_act_and_unclassified_authorities_are_not_forced_onto_ids():
    _parsed, graph = _build(_by_prefix("OP-12"))
    props = graph["entities"][0]["properties"]
    assert props["other_act_references"] == ["Section 165, Finance Act, 2016"]
    assert props["other_authorities_cited"] == ["CBDT FAQs on Equalisation Levy"]
    assert not any("165" in e["id"] for e in graph["entities"])


def test_treaty_stubs_match_extract_treaty_id_scheme():
    _parsed, graph = _build(_by_prefix("OP-01"))
    ents = {e["id"]: e for e in graph["entities"]}
    assert ents["TaxTreaty_USA"]["type"] == "TaxTreaty"
    assert ents["Article_12_TaxTreaty_USA"]["properties"]["article_title"] == "Fees for Included Services"
    _parsed, graph = _build(_by_prefix("OP-04"))
    rels = {(r["type"], r["target"], r["properties"].get("authority_kind")) for r in graph["relationships"]}
    assert ("RELIES_ON_PROVISION", "TaxTreaty_Mauritius", "treaty") in rels


def test_act_namespace_detection():
    assert eo.detect_act_namespace("under the Income-tax Act, 1961") == "ITA1961"
    assert eo.detect_act_namespace("under the Income-tax Act, 2025 only") == "ITA2025"
    assert eo.detect_act_namespace("Income-tax Act, 1961 and Income-tax Act, 2025") == "ITA1961"
    assert eo.detect_act_namespace("no act named") == "ITA1961"
    assert eo.detect_act_namespace("Income-tax Act, 1961", override="ITA2025") == "ITA2025"
    text = _by_prefix("OP-15").read_text(encoding="utf-8").replace("Income-tax Act, 1961", "Income-tax Act, 2025")
    graph = eo.build_graph_from_opinion(eo.parse_opinion(text, "x.md"), "x.md")
    assert "ITA2025__Section_44ADA" in {e["id"] for e in graph["entities"]}


def test_confidence_band_earliest_phrase_wins():
    assert eo.derive_confidence_band("Moderate; fairly arguable if the valuation assumptions are robust") == "moderate"
    assert eo.derive_confidence_band("Fairly arguable / at risk absent remedial steps") == "fairly_arguable"
    assert eo.derive_confidence_band("Reasonable / more likely than not, contingent on documentation") == "more_likely_than_not"
    assert eo.derive_confidence_band("High degree of comfort; may attract scrutiny") == "high"
    assert eo.derive_confidence_band("") == ""


# --- no-frontmatter fallback (PDF/TXT opinions) -----------------------------------

@pytest.mark.parametrize("path", OPINIONS, ids=lambda p: p.name[:5])
def test_body_only_parse_recovers_the_same_graph(path):
    text = path.read_text(encoding="utf-8")
    _fm, body = eo.parse_frontmatter(text)
    with_fm, g_fm = _build(path)
    no_fm = eo.parse_opinion(body, path.name)
    g_body = eo.build_graph_from_opinion(no_fm, path.name)
    assert not no_fm["has_frontmatter"]
    for key in ("opinion_id", "title", "client", "partner", "firm", "date", "confidence_level"):
        assert no_fm[key] == with_fm[key], key
    stubs = lambda g: sorted(e["id"] for e in g["entities"][1:])  # noqa: E731
    assert stubs(g_body) == stubs(g_fm)
    # The counterparty is ONLY known from frontmatter (the letter body never
    # names it as such), so a body-only opinion can't pseudonymise it - the
    # chunk texts must differ by exactly that and nothing else.
    protected = " | ".join(with_fm["cases_cited_raw"] + with_fm["sections_cited_raw"] + with_fm["treaty_raw"])
    cp_only = eo.build_redactions("", with_fm["counterparty"], "", protected=protected)
    assert len(g_body["chunks"]) == len(g_fm["chunks"])
    for cb, cf in zip(g_body["chunks"], g_fm["chunks"]):
        body_text = eo.redact(cb["text"], cp_only)
        # where a counterparty shares the client's first word (OP-19 "Hillcrest
        # ...") the body-only run shows "[CLIENT] Global Pte Ltd" instead -
        # still pseudonymised, so allow anything at [COUNTERPARTY] positions.
        pattern = re.escape(cf["text"]).replace(re.escape("[COUNTERPARTY]"), ".+?")
        assert re.fullmatch(pattern, body_text, re.S), cb["chunk_id"]
        assert with_fm["client"] not in body_text and with_fm["partner"] not in body_text


def test_frontmatter_parser_subset():
    fm, body = eo.parse_frontmatter('---\na: "x: y"\nb: []\nc:\n  - "one"\n  - two\nd: true\ne: [p, "q"]\n---\nBODY')
    assert fm == {"a": "x: y", "b": [], "c": ["one", "two"], "d": True, "e": ["p", "q"]}
    assert body == "BODY"
    assert eo.parse_frontmatter("no frontmatter") == ({}, "no frontmatter")


def test_supersession_frontmatter_emits_both_directions():
    text = _by_prefix("OP-11").read_text(encoding="utf-8").replace("fictitious: true", "fictitious: true\nsupersedes: OP-01", 1)
    graph = eo.build_graph_from_opinion(eo.parse_opinion(text, "x.md"), "x.md")
    rels = {(r["source"], r["type"], r["target"]) for r in graph["relationships"]}
    assert ("FirmOpinion_OP_11", "SUPERSEDES", "FirmOpinion_OP_01") in rels
    assert ("FirmOpinion_OP_01", "SUPERSEDED_BY", "FirmOpinion_OP_11") in rels  # what opinion_agent queries
    stub = next(e for e in graph["entities"] if e["id"] == "FirmOpinion_OP_01")
    assert stub["properties"]["provenance"] == provenance.CITED_REFERENCE


def test_cli_dir_mode_writes_graph_and_chunks(tmp_path):
    proc = subprocess.run([sys.executable, str(CONVERTERS / "extract_opinion.py"),
                           "--dir", str(REPO / "tests" / "fixtures" / "opinions"), "--out-dir", str(tmp_path)],
                          capture_output=True, text=True, cwd=str(CONVERTERS), timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    graphs = sorted(tmp_path.glob("*_graph.json"))
    assert len(graphs) == 20 and len(list(tmp_path.glob("*_chunks.json"))) == 20
    data = json.loads(graphs[0].read_text(encoding="utf-8"))
    assert data["entities"][0]["type"] == "FirmOpinion" and data["relationships"]
    assert "SCHEMA WARNINGS" not in proc.stdout + proc.stderr


if __name__ == "__main__" and "--regenerate-snapshot" in sys.argv:
    _out = {p.name: snapshot(_build(p)[1]) for p in OPINIONS}
    (REPO / "tests" / "fixtures" / "opinion_snapshots.json").write_text(
        json.dumps(_out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(_out)} snapshots - review the diff before committing")
