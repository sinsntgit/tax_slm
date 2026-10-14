"""2026-10-14: NEW - framework/core/retrieve/grounding.py: citation verification,
graph-row context sizing, referred-sources cross-check, confidence scoring."""
import json

import pytest

from framework.core.retrieve import grounding as g


# --- _confidence_score / ground_answer --------------------------------------

def test_confidence_zero_without_chunks():
    assert g._confidence_score([], [{"a": 1}]) == 0.0


def test_confidence_uses_sigmoid_score_directly():
    # 0.5 base + 0.3 * 1.0 = 0.8 (NOT the old (score+10)/20 normalization)
    assert g._confidence_score([{"rerank_score": 1.0}], []) == 0.8
    assert g._confidence_score([{"rerank_score": 0.5}], []) == 0.65


def test_confidence_clamps_out_of_range_scores():
    assert g._confidence_score([{"rerank_score": 7.3}], []) == 0.8
    assert g._confidence_score([{"rerank_score": -3}], []) == 0.5


@pytest.mark.parametrize("chunk", [{"rerank_score": None}, {}])
def test_confidence_handles_skip_rerank_none_and_missing(chunk):
    # skip_rerank chunks carry an explicit None - must not raise TypeError
    assert g._confidence_score([chunk], []) == 0.5


def test_confidence_fallback_tier_penalty_and_graph_bonus():
    top = {"rerank_score": 1.0, "relevance_tier": "fallback"}
    assert g._confidence_score([top], []) == 0.65
    assert g._confidence_score([top], [{"x": 1}]) == 0.85
    # bonus never pushes above 1.0
    assert g._confidence_score([{"rerank_score": 1.0}], [{"x": 1}]) == 1.0


def test_ground_answer_with_none_rerank_score_and_long_text():
    chunks = [
        {"chunk_id": "7", "text": "x" * 250, "rerank_score": None, "source": "vector"},
        {"chunk_id": "8", "text": "short", "rerank_score": 0.91234},
    ]
    rows = [{"n": {"id": "Section_9"}, "r": {"type": "PART_OF"}}]
    out = g.ground_answer("q?", chunks, rows)
    refs = out["source_references"]
    assert refs[0]["rerank_score"] == 0.0
    assert refs[0]["preview"].endswith("…") and len(refs[0]["preview"]) == 201
    assert refs[0]["via"] == "vector"
    assert refs[1] == {"chunk_id": "8", "via": "unknown", "preview": "short", "rerank_score": 0.912}
    assert out["kg_paths_used"] == ["[PART_OF] relationship in graph"]
    assert out["confidence"] == 0.7  # 0.5 + 0 (None) + 0.2 graph bonus


# --- verify_citations --------------------------------------------------------

CHUNKS = [{"chunk_id": "42", "text": "t"}]


def test_verify_no_context_is_trivially_ok():
    assert g.verify_citations("anything", [], []) == {"citations_found": [], "citation_ok": True}


def test_verify_chunk_id_marker():
    out = g.verify_citations("per (chunk_id = 42) the rate is 10%", CHUNKS, [])
    assert out["citation_ok"] and out["citations_found"] == ["chunk_id = 42"]


def test_verify_graph_node_id_in_text():
    rows = [{"s": {"id": "Section_9", "hasText": "..."}}]
    out = g.verify_citations("Section_9 says so", [], rows)
    assert out == {"citations_found": ["Section_9"], "citation_ok": True}


def test_verify_bracket_document_format_case_insensitive():
    out = g.verify_citations("Royalty excluded **[document: ITA 1961, p. 4, lines 1-3]**", CHUNKS, [])
    assert out["citation_ok"]
    assert out["citations_found"] == ["[document: ITA 1961, p. 4, lines 1-3]"]


def test_verify_doc_details_fallback_and_min_length():
    details = {"42": {"document_name": "Engineering Analysis Centre (2021) 2 SCC 1"},
               "43": {"document_name": "CIT v X"}}  # < 12 chars: too short to count
    ok = g.verify_citations("As held in engineering analysis centre (2021) 2 scc 1, ...", CHUNKS, [], details)
    assert ok["citation_ok"] and ok["citations_found"] == ["Engineering Analysis Centre (2021) 2 SCC 1"]
    short = g.verify_citations("In CIT v X the court held", CHUNKS, [], {"43": details["43"]})
    assert short == {"citations_found": [], "citation_ok": False}


def test_verify_flags_uncited_answer():
    out = g.verify_citations("The rate is 10 percent.", CHUNKS, [], {"42": {"document_name": None}})
    assert out == {"citations_found": [], "citation_ok": False}


# --- prepare_graph_rows_for_context / graph_rows_to_context_json ------------

def test_cap_bulky_field_list_and_string():
    assert g.cap_bulky_field(["aaaa", "bbbb", "cccc"], 9) == ["aaaa", "bbbb", "...(1 more, truncated for length)"]
    assert g.cap_bulky_field(["aa", "bb"], 100) == ["aa", "bb"]
    assert g.cap_bulky_field("abcdef", 3) == "abc…"
    assert g.cap_bulky_field("abc", 3) == "abc"
    assert g.cap_bulky_field(12345, 1) == 12345


def test_prepare_rows_caps_only_listed_fields_and_dedups_repeated_nodes():
    case = {"id": "CaseLaw_EA", "case_name": "Engineering Analysis", "cases_cited": ["c" * 50] * 30,
            "holding": "h" * 5000, "facts": "f" * 900}
    rows = [
        {"m": case, "r": {"type": "OVERRULES"}, "n": {"id": "CaseLaw_Samsung", "case_name": "Samsung"}},
        {"m": dict(case), "r": {"type": "OVERRULES"}, "n": {"node_id": "CaseLaw_Citrix", "case_name": "Citrix"}},
    ]
    out = g.prepare_graph_rows_for_context(rows)
    first = out[0]["m"]
    assert first["holding"] == "h" * 5000            # uncapped, needed for the answer
    assert first["facts"] == "f" * 800 + "…"          # capped at 800
    assert first["cases_cited"][-1].startswith("...(")  # list capped item-by-item
    assert len(first["cases_cited"]) == 25  # 24 items x 50 chars fit in 1200, then the marker
    # second occurrence of the same node -> compact reference only
    assert out[1]["m"] == {"id": "CaseLaw_EA", "case_name": "Engineering Analysis",
                           "_note": "full properties for this node already shown earlier in this context"}
    # node_id key also identifies a node; relationship dicts pass through
    assert out[1]["n"]["case_name"] == "Citrix"
    assert out[0]["r"] == {"type": "OVERRULES"}
    # input not mutated
    assert rows[0]["m"]["facts"] == "f" * 900


def test_prepare_rows_custom_caps_and_scalars():
    rows = [{"x": {"id": "A", "note": "abcdef"}, "count": 3}]
    out = g.prepare_graph_rows_for_context(rows, field_caps={"note": 2})
    assert out == [{"x": {"id": "A", "note": "ab…"}, "count": 3}]


def test_graph_rows_to_context_json_valid_json_and_backstop():
    rows = [{"n": {"id": f"N{i}", "hasText": "t" * 2000}} for i in range(3)]
    js = g.graph_rows_to_context_json(rows)
    parsed = json.loads(js)
    assert len(parsed) == 3 and all(len(r["n"]["hasText"]) == 801 for r in parsed)
    tiny = g.graph_rows_to_context_json(rows, max_chars=100)
    assert tiny.endswith("\n... (truncated for length)") and len(tiny) == 100 + len("\n... (truncated for length)")


def test_engineering_analysis_fixture_rows_fit_backstop(fixtures_dir):
    """Real CaseLaw node props (from the fixture graph) repeated across 3 rows,
    the shape of the OVERRULES query that motivated per-field capping: capping +
    dedup must bring it under the 16K backstop so no row is sliced off."""
    path = next((fixtures_dir / "graphs").glob("*Engineering_Analysis*_graph.json"))
    ents = json.loads(path.read_text(encoding="utf-8"))["entities"]
    ea = dict(ents[0]["properties"], id=ents[0]["id"])
    stubs = [e for e in ents if e["type"] == "CaseLaw"][1:4]
    rows = [{"m": dict(ea), "r": {"type": "OVERRULES"}, "n": dict(s["properties"], id=s["id"])} for s in stubs]
    raw = json.dumps(rows, indent=2, ensure_ascii=False)
    js = g.graph_rows_to_context_json(rows)
    assert len(js) < len(raw)
    assert len(js) < g.GRAPH_ROWS_MAX_CHARS_DEFAULT, len(js)
    parsed = json.loads(js)  # not cut by the backstop -> still valid JSON
    assert [r["n"]["id"] for r in parsed] == [s["id"] for s in stubs]


# --- check_referred_sources --------------------------------------------------

DETAILS = {"1": {"document_name": "income_tax_act_1961.pdf"}, "2": {"document_name": "Circular 2 of 2026.pdf"}}


def test_referred_sources_flags_unknown_entries():
    answer = ("Body text.\n\n## d. Referred Sources\n"
              "- income_tax_act_1961.pdf - Page 12\n"
              "- Some Invented Ruling 2019.pdf - Page 3\n")
    out = g.check_referred_sources(answer, DETAILS)
    assert out["checked"] is True
    assert out["unverified_sources"] == ["- Some Invented Ruling 2019.pdf"]


def test_referred_sources_all_known():
    answer = "Referred sources:\nCircular 2 of 2026.pdf - Page 1\n"
    assert g.check_referred_sources(answer, DETAILS) == {"checked": True, "unverified_sources": []}


@pytest.mark.parametrize("answer,details", [
    ("no heading here foo.pdf", DETAILS),
    ("Referred Sources\nfoo.pdf", None),
    ("Referred Sources\nfoo.pdf", {"1": {"document_name": ""}}),
])
def test_referred_sources_not_checked(answer, details):
    assert g.check_referred_sources(answer, details) == {"checked": False, "unverified_sources": []}
