"""2026-10-14: NEW - domain_packs/tax/converters/chunking.py on a small
synthetic Act markdown: heading/marker regexes, footnote amendment parsing,
citation + entity-id building (incl. Act namespacing), provisos/explanations,
cross-references, schedules, page mapping, LLM batching.

tiktoken: chunking.py calls tiktoken.get_encoding("cl100k_base") at IMPORT
time, which downloads the BPE file from openaipublic.blob.core.windows.net on
first use. That host is blocked in the cloud sandbox (it fails fast and
chunking falls back to len(text)//4), but is reachable on GitHub Actions - so
token counts would differ between environments. Every test here forces the
deterministic fallback (`_ENCODING = None`) for reproducibility."""
import importlib
import sys
from pathlib import Path

import pytest

CONVERTERS = Path(__file__).resolve().parents[1] / "domain_packs" / "tax" / "converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))
chunking = importlib.import_module("chunking")

ACT_MD = """## CHAPTER I - PRELIMINARY
#### Section 2. Definitions.
2. In this Act, unless the context otherwise requires,--
(1) "agricultural income" means any rent of 6[twenty-four] months, see section 10.
(a) any rent or revenue derived from land;
(b) any income derived from such land by agriculture;
6. Sub. for "thirty-six" by the Act No. 15 of 2024, w.e.f. 23-7-2024.
(1A) "assessee" means a person by whom any tax 7[***] is payable under section 4.
Provided that this shall not apply to a firm.
Explanation 1.-- For the purposes of this clause, sub-section (2) of section 10 applies.
7. Omitted by the Act No. 3 of 2025, w.e.f. 1-4-2025.
## CHAPTER II - BASIS OF CHARGE
#### Section 4. Charge of income-tax.
4. Income-tax shall be charged for every assessment year.
## THE FIRST SCHEDULE
[See section 115]
Rates table.
"""


@pytest.fixture(autouse=True)
def deterministic_tokens(monkeypatch):
    monkeypatch.setattr(chunking, "_ENCODING", None)


@pytest.fixture
def small_limits(monkeypatch):
    """Force splitting on tiny synthetic text (real limits are ~1000 tokens)."""
    for key in ("section", "subsection"):
        monkeypatch.setitem(chunking.SIZE_GUIDELINES, key, dict(chunking.SIZE_GUIDELINES[key], max=10))


@pytest.fixture
def parsed(small_limits):
    return chunking.chunk_act_document(ACT_MD, namespace="ITA2025", act_props={"actName": "Test Act"})


def _ents(out):
    return {e["id"]: e for e in out["entities"]}


# --- regexes / small helpers ------------------------------------------------------

@pytest.mark.parametrize("line,expected", [
    ("#### Section 80IAC. Special provision.", ("80IAC", "Special provision.")),
    ("#### Section 2. Definitions.", ("2", "Definitions.")),
    ("### Section 2. Definitions.", None),
])
def test_section_heading_re(line, expected):
    m = chunking.SECTION_HEADING_RE.match(line)
    assert (m.groups() if m else None) == expected


@pytest.mark.parametrize("line,expected", [
    ("## CHAPTER IV-A -- DEDUCTIONS", ("IV-A", "DEDUCTIONS")),
    ("## Chapter XII: Special rates", ("XII", "Special rates")),
])
def test_chapter_heading_re(line, expected):
    assert chunking.CHAPTER_HEADING_RE.match(line).groups() == expected


@pytest.mark.parametrize("pattern,line,marker", [
    ("SUBSECTION_RE", "(1A) text", "1A"), ("SUBSECTION_RE", "(42) x", "42"), ("SUBSECTION_RE", "(a) x", None),
    ("CLAUSE_RE", "(aa) x", "aa"), ("CLAUSE_RE", "(1) x", None),
    ("SUB_CLAUSE_RE", "(iv) x", "iv"), ("SUB_CLAUSE_RE", "(XII) x", "XII"),
])
def test_marker_res(pattern, line, marker):
    m = getattr(chunking, pattern).match(line)
    assert (m.group(1) if m else None) == marker


@pytest.mark.parametrize("line,action,date", [
    ('6. Sub. for "thirty-six" by the Act No. 15 of 2024, w.e.f. 23-7-2024.', "substitution", "2024-07-23"),
    ("3a. Ins. by Act No. 7 of 2017, w.r.e.f. 1-4-2016.", "insertion", "2016-04-01"),
    ("9. Omitt. by the Act No. 15 of 2024, w.e.f. 1-4-2025.", "omission", "2025-04-01"),
    ("1. Amended by Act No. 2 of 2020.", "amendment", None),
])
def test_footnote_index(line, action, date):
    cleaned, idx = chunking._extract_footnote_index(["body text", line])
    assert cleaned == ["body text"]
    [entry] = idx.values()
    assert entry["type"] == action and entry["effective_from"] == date


def test_normalize_date():
    assert chunking._normalize_date("5-9-2019") == "2019-09-05"
    assert chunking._normalize_date("bad") is None


def test_strip_inline_markers():
    fi = {"6": {"type": "substitution", "act_number": 15}}
    text, amendments, redacted = chunking._strip_inline_markers("rent of 6[twenty-four] and 9a[***] and 8[ ]", fi)
    assert text == "rent of twenty-four and [***] and [***]"
    assert redacted is True and amendments == [{"marker": "6", "type": "substitution", "act_number": 15}]


def test_cross_references():
    refs = chunking.extract_cross_references(
        "as per sub-section (2) of section 10 and section 10 and section 4 and Section 4", self_section="4")
    assert refs == [{"text": "sub-section (2) of section 10", "target_section": "10", "target_subsection": "2"},
                    {"text": "section 10", "target_section": "10"}]
    assert chunking.related_sections_from_refs(refs) == ["10"]


@pytest.mark.parametrize("args,citation,eid", [
    (("9",), "Section 9", "Section_9"),
    (("9", "1"), "Section 9(1)", "SubSection_9_1"),
    (("9", "1", "vi"), "Section 9(1)(vi)", "Clause_9_1_vi"),
    (("10", "23C", "iv", "a"), "Section 10(23C)(iv)(a)", "SubClause_10_23C_iv_a"),
])
def test_citation_and_entity_id(args, citation, eid):
    assert chunking.generate_full_citation(*args) == citation
    assert chunking.generate_entity_id(*args) == eid
    assert chunking.generate_entity_id(*args, namespace="ITA1961") == f"ITA1961__{eid}"


@pytest.mark.parametrize("etype,tokens,verdict", [
    ("section", 50, "too_small"), ("section", 500, "good"), ("section", 1001, "too_large"), ("nope", 1, "unknown"),
])
def test_decide_chunk_size(etype, tokens, verdict):
    assert chunking.decide_chunk_size(etype, tokens)[0] == verdict


def test_token_count_fallback():
    assert chunking.token_count("") == 0
    assert chunking.token_count("abc") == 1
    assert chunking.token_count("a" * 40) == 10


def test_make_short_title():
    assert chunking.make_short_title("  a   b  ") == "a b"
    assert chunking.make_short_title("word " * 30, max_len=20) == "word word word word..."


# --- chunk_act_document -------------------------------------------------------------

def test_every_id_is_namespaced_and_hierarchy_connected(parsed):
    ents = _ents(parsed)
    structural = [e for e in ents.values() if e["type"] in ("Section", "SubSection", "Clause", "SubClause")]
    assert {e["id"] for e in structural} == {
        "ITA2025__Section_2", "ITA2025__SubSection_2_1", "ITA2025__Clause_2_1_a", "ITA2025__Clause_2_1_b",
        "ITA2025__SubSection_2_1A", "ITA2025__Section_4"}
    part_of = {(r["source"], r["target"]) for r in parsed["relationships"] if r["type"] == "PART_OF"}
    assert part_of == {
        ("ITA2025__Section_2", "ITA2025"), ("ITA2025__SubSection_2_1", "ITA2025__Section_2"),
        ("ITA2025__Clause_2_1_a", "ITA2025__SubSection_2_1"), ("ITA2025__Clause_2_1_b", "ITA2025__SubSection_2_1"),
        ("ITA2025__SubSection_2_1A", "ITA2025__Section_2"), ("ITA2025__Section_4", "ITA2025")}
    # every PART_OF endpoint is a real entity (no dangling un-namespaced wrapper ids)
    assert all(s in ents and t in ents for s, t in part_of)
    assert ents["ITA2025"]["type"] == "Act"


def test_split_section_wrapper_gets_marginal_heading_and_lead_in(parsed):
    sec2 = _ents(parsed)["ITA2025__Section_2"]["properties"]
    assert sec2["marginal_heading"] == "Definitions."
    assert sec2["hasText"] == "In this Act, unless the context otherwise requires,--"
    assert _ents(parsed)["ITA2025__Section_4"]["properties"]["marginal_heading"] == "Charge of income-tax."


def test_chapter_headings_do_not_leak_into_previous_section_text(parsed):
    for e in parsed["entities"]:
        assert "CHAPTER" not in e["properties"].get("hasText", ""), e["id"]
    for c in parsed["chunks"]:
        assert "## CHAPTER" not in c.content, c.entity_id
    chapters = {c.entity_id: c.chapter for c in parsed["chunks"]}
    assert chapters["ITA2025__Clause_2_1_a"] == "Chapter I: PRELIMINARY"
    assert chapters["ITA2025__Section_4"] == "Chapter II: BASIS OF CHARGE"


def test_amendments_redaction_and_amending_act_nodes(parsed):
    ents = _ents(parsed)
    sub21 = ents["ITA2025__SubSection_2_1"]["properties"]
    assert "twenty-four months" in sub21["hasText"] and "6[" not in sub21["hasText"]
    assert sub21["amendmentNature"] == "substitution" and sub21["effective_from"] == "2024-07-23"
    sub21a = ents["ITA2025__SubSection_2_1A"]["properties"]
    assert sub21a["content_status"] == "deleted_or_redacted" and "[***]" in sub21a["hasText"]
    assert sub21a["status"] == "omitted" and sub21a["effective_to"] == "2025-04-01"
    amended = {(r["source"], r["target"]) for r in parsed["relationships"] if r["type"] == "AMENDED_BY"}
    assert amended == {("ITA2025__SubSection_2_1", "AmendingAct_15_2024"),
                       ("ITA2025__SubSection_2_1A", "AmendingAct_3_2025")}
    assert ents["AmendingAct_15_2024"]["properties"]["full_citation"] == "Act No. 15 of 2024"
    assert set(parsed["footnote_index"]) == {"6", "7"}


def test_provisos_explanations_and_refs(parsed):
    rels = [(r["type"], r["source"], r["target"]) for r in parsed["relationships"]]
    assert ("HAS_PROVISO", "ITA2025__SubSection_2_1A", "Proviso_to_ITA2025__SubSection_2_1A") in rels
    assert ("HAS_EXPLANATION", "ITA2025__SubSection_2_1A", "Explanation_1_to_ITA2025__SubSection_2_1A") in rels
    assert ("REFERS_TO", "ITA2025__SubSection_2_1A", "ITA2025__Section_4") in rels
    assert ("CROSS_REFERS_TO_ALLIED_LAW", "ITA2025__Section_115", "Schedule_FIRST") in rels
    leaf = next(c for c in parsed["chunks"] if c.entity_id == "ITA2025__SubSection_2_1A")
    assert leaf.has_proviso and leaf.has_explanation and leaf.related_sections == ["4"]
    assert "Provided that" not in leaf.content  # peeled into its own chunk
    expl = next(c for c in parsed["chunks"] if c.element_type == "explanation")
    assert expl.full_citation == "Explanation 1 to Section 2(1A)"


def test_schedule_chunk(parsed):
    sched = next(c for c in parsed["chunks"] if c.chunk_type == "schedule")
    assert sched.entity_id == "Schedule_FIRST" and sched.full_citation == "Schedule FIRST"
    assert sched.related_sections == ["115"] and "Rates table." in sched.content


def test_unnamespaced_single_act_keeps_bare_ids(small_limits):
    out = chunking.chunk_act_document(ACT_MD)
    ids = {e["id"] for e in out["entities"]}
    assert {"Section_2", "SubSection_2_1", "Clause_2_1_a", "Section_4"} <= ids
    assert not any("__" in i for i in ids)


def test_small_sections_stay_single_leaf_without_small_limits():
    out = chunking.chunk_act_document(ACT_MD, namespace="X")
    sec2 = [c for c in out["chunks"] if c.entity_id == "X__Section_2"]
    assert len(sec2) == 1 and "(1A)" in sec2[0].content


def test_duplicate_section_heading_flagged(small_limits):
    md = "#### Section 5. A.\n5. First text.\n#### Section 5. B.\n5. Second text.\n"
    out = chunking.chunk_act_document(md)
    assert _ents(out)["Section_5"]["properties"]["hasText"] == "First text."
    assert any(a["type"] == "duplicate_entity_id" for a in out["ambiguities"])


def test_nested_repeat_markers_fold_into_current_block():
    lines = [(1, "(1) first"), (2, "(a) x"), (3, "(1) nested repeat"), (4, "(2) second")]
    blocks = chunking._split_marked_blocks(lines, chunking.SUBSECTION_RE)
    assert [b[0] for b in blocks] == ["1", "2"]
    assert blocks[0][2] == [(1, "first"), (2, "(a) x"), (3, "(1) nested repeat")]


# --- page mapping / batching ----------------------------------------------------------

def test_page_for_line_and_apply():
    bounds = [(0, 1), (10, 2), (25, 3)]
    assert chunking.page_for_line(0, bounds) == 1
    assert chunking.page_for_line(9, bounds) == 1
    assert chunking.page_for_line(10, bounds) == 2
    assert chunking.page_for_line(99, bounds) == 3
    assert chunking.page_for_line(5, []) is None
    cs = [chunking.Chunk(section="1", line_start=12)]
    chunking.apply_page_numbers(cs, bounds)
    assert cs[0].page_number == 2


def test_group_chunks_for_llm():
    cs = [chunking.Chunk(section=str(i), content="x" * 40) for i in range(5)]
    groups = chunking.group_chunks_for_llm(cs, max_chars=100)
    assert [len(g) for g in groups] == [2, 2, 1]
    assert chunking.group_chunks_for_llm([], 10) == []
    big = [chunking.Chunk(section="1", content="y" * 500)]
    assert chunking.group_chunks_for_llm(big, 10) == [big]  # an oversized chunk still gets its own group


@pytest.mark.xfail(strict=True, reason=(
    "KNOWN BUG (2026-10-14, reported, not fixed here): chunking.py records line_start/line_end as "
    "0-indexed positions in the FOOTNOTE-STRIPPED line list, while md_hydration.hydrate_text() treats "
    "them as 1-indexed lines of the RAW .md file - a line-range hydration returns the wrong lines "
    "(off by one, plus one more per footnote line above). Latent today because Section nodes don't "
    "carry line_start yet (only Chunk records do), so semantic_graph falls back to whole-file reads."))
def test_chunk_line_numbers_round_trip_through_md_hydration(tmp_path):
    from framework.core.retrieve import md_hydration
    p = tmp_path / "act.md"
    p.write_text(ACT_MD, encoding="utf-8")
    out = chunking.chunk_act_document(ACT_MD)
    sec4 = next(c for c in out["chunks"] if c.entity_id == "Section_4")
    text, _ = md_hydration.hydrate_text(str(p), sec4.line_start, sec4.line_end)
    assert "Income-tax shall be charged" in text
