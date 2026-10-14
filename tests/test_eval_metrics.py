"""2026-10-14: NEW - deterministic per-item metrics of the eval harness."""
import pytest

from eval import metrics
from eval.report import aggregate, percentile
from eval_fakes import (EA_DOC, FABRICATED_ANSWER, GOOD_RESEARCH_ANSWER, MALFORMED_FORMAT_ANSWER, ask_response,
                        evaluation_response, gold_item)


def _found(auth, text):
    return metrics.find_authority(auth, metrics.normalize(text)) is not None


@pytest.mark.parametrize("text,expected", [
    ("tax must be deducted under Section 195 of the Act", True),
    ("deduct u/s 195 before remitting", True),
    ("as per s.195(1)", True),
    ("Sec. 195 applies", True),
    ("Section 195A deals with something else", False),
    ("Section 1951 does not exist", False),
    ("see section 19 and 5", False),
])
def test_section_matching_is_number_aware(text, expected):
    auth = {"id": "s195", "kind": "section", "label": "Section 195", "aliases": ["Section 195"], "section": "195"}
    assert _found(auth, text) is expected


def test_subclause_and_suffix_matching():
    s9 = {"id": "s9", "kind": "section", "label": "Section 9(1)(vi)", "aliases": ["Section 9(1)(vi)"], "section": "9(1)(vi)"}
    assert _found(s9, "royalty under section 9(1)(vi), Explanation 2")
    assert not _found(s9, "fees under section 9(1)(vii)")
    s80 = {"id": "s80", "kind": "section", "label": "Section 80-IAC", "aliases": ["Section 80-IAC"], "section": "80-IAC"}
    assert _found(s80, "deduction under section 80IAC") and _found(s80, "Section 80 - IAC")
    case = {"id": "ea", "kind": "case", "label": "EA", "aliases": ["Engineering Analysis Centre of Excellence"]}
    assert _found(case, "325ebe7a_Engineering_Analysis_Centre_of_Excellence_vs_CIT.md")
    assert _found(case, "ENGINEERING ANALYSIS CENTRE OF EXCELLENCE PVT. LTD.")


def test_strip_server_additions_removes_disclaimer_and_note():
    raw = "Body text.\n\n(Note: this answer should have cited specific sources but didn't - x)\n\nDISCLAIMER\nboiler"
    assert metrics.strip_server_additions(raw) == "Body text."


def test_citation_check_good_vs_fabricated():
    sources = metrics.known_sources(ask_response(GOOD_RESEARCH_ANSWER))
    good = metrics.check_citations(GOOD_RESEARCH_ANSWER, sources)
    assert good["fabricated_count"] == 0 and good["citation_count"] >= 4
    assert good["page_unverified"] == []

    bad = metrics.check_citations(FABRICATED_ANSWER, sources)
    assert bad["fabricated_count"] == 2
    assert any("Imaginary Software" in f for f in bad["fabricated"])
    assert "chunk_id=999" in bad["fabricated"]

    wrong_page = metrics.check_citations(f"Held so. {EA_DOC} - Page 77", sources)
    assert wrong_page["fabricated_count"] == 0 and len(wrong_page["page_unverified"]) == 1


def test_citation_with_no_retrieval_is_fabricated():
    resp = ask_response(GOOD_RESEARCH_ANSWER, chunks=[])
    res = metrics.check_citations(GOOD_RESEARCH_ANSWER, metrics.known_sources(resp))
    assert res["fabricated_count"] == 1  # same doc cited several times -> deduped to one


def test_format_compliance():
    ok = metrics.check_format(GOOD_RESEARCH_ANSWER, "research")
    assert ok["compliant"] and ok["strict_litigation_risk_bullet"] and not ok["strict_referred_categories_missing"]
    bad = metrics.check_format(MALFORMED_FORMAT_ANSWER, "research")
    assert not bad["compliant"] and bad["headings_missing"] == ["a", "b", "d"]
    swapped = GOOD_RESEARCH_ANSWER.replace("## c. Conclusion", "## x").replace("## d. Referred Sources",
                                                                              "## c. Conclusion\n## d. Referred Sources")
    reordered = "## c. Conclusion\nfirst\n" + GOOD_RESEARCH_ANSWER.replace("## c. Conclusion", "")
    assert metrics.check_format(swapped, "research")["compliant"]
    r = metrics.check_format(reordered, "research")
    assert not r["compliant"] and not r["in_order"]
    assert metrics.check_format("anything", "lookup") == {"applicable": False, "compliant": None}


def test_must_not_claim_negation_guard():
    claims = [{"claim": "royalty", "patterns": [r"(?:constitutes?|is|are)\s+royalt(?:y|ies)\b"]}]
    assert metrics.check_must_not_claim("The payments do not constitute royalty.", claims) == []
    assert metrics.check_must_not_claim("It cannot be said the payment is royalty.", claims) == []
    v = metrics.check_must_not_claim("We conclude the payment is royalty under the DTAA.", claims)
    assert len(v) == 1 and "is royalty" in v[0]["snippet"]
    # negation INSIDE the pattern is part of the wrong claim and must still fire
    inside = [{"claim": "excluded", "patterns": [r"not\s+includible\s+in\s+book\s+profit"]}]
    assert len(metrics.check_must_not_claim("The gain is not includible in book profit.", inside)) == 1
    unguarded = [{"claim": "royalty", "patterns": [r"is\s+royalty"], "negation_guard": False}]
    assert len(metrics.check_must_not_claim("It is not true that it is royalty.", unguarded)) == 1


def test_score_item_end_to_end_and_disclaimer_ignored():
    gold = gold_item("t1", "Are software licence payments royalty under section 195?")
    rec = {"status": "ok", "latency_s": 12.0, "attempts": 1, "effective_answer_mode": "research",
           "response": ask_response(GOOD_RESEARCH_ANSWER),
           "judge": {"status": "ok", "response": evaluation_response(7.5)}}
    row = metrics.score_item(gold, rec)
    assert row["authorities"]["recall_answer"] == 1.0 and row["authorities"]["recall_sources"] == 1.0
    assert row["citations"]["fabricated_count"] == 0 and row["citations"]["citation_ok"] is True
    assert row["format"]["compliant"] is True
    # DISCLAIMER says "is taxable as fees for included services"/"Royalty is taxable" - must not count
    assert row["must_not_claim"] == []
    assert row["key_points"]["coverage"] == 1.0
    assert row["judge"]["answer_quality.overall"] == 0.8
    assert row["judge"]["evaluation.overall"] == 7.5
    assert row["judge"]["evaluation.groundedness_accuracy"] == 7.5
    assert row["judge"]["general_knowledge.overall"] == 6.5


def test_score_item_missing_authority_and_error_record():
    gold = gold_item("t2", "Are software licence payments royalty under section 195?",
                     required=[{"id": "s90", "kind": "section", "label": "Section 90", "aliases": ["Section 90"],
                                "section": "90"},
                               {"id": "s195", "kind": "section", "label": "Section 195", "aliases": ["Section 195"],
                                "section": "195"}])
    row = metrics.score_item(gold, {"status": "ok", "response": ask_response(GOOD_RESEARCH_ANSWER)})
    assert row["authorities"]["recall_answer"] == 1.0  # "s.90(2)" in the answer counts for Section 90
    err = metrics.score_item(gold, {"status": "error", "error": {"type": "timeout"}, "latency_s": 30})
    assert err["status"] == "error" and "authorities" not in err


def test_unavailable_judge_contributes_nothing():
    resp = ask_response(GOOD_RESEARCH_ANSWER)
    resp["grounding"]["answer_quality"] = {"available": False, "overall": None}
    assert metrics.extract_judge_scores(resp, {"status": "error"}) == {}


def test_aggregate_denominators():
    gold = gold_item("t3", "Are software licence payments royalty under section 195?")
    ok = metrics.score_item(gold, {"status": "ok", "latency_s": 10, "response": ask_response(GOOD_RESEARCH_ANSWER),
                                   "effective_answer_mode": "research"})
    fab = metrics.score_item(gold, {"status": "ok", "latency_s": 20, "response": ask_response(FABRICATED_ANSWER),
                                    "effective_answer_mode": "research"})
    err = metrics.score_item(gold, {"status": "error", "latency_s": 90, "error": {"type": "timeout"},
                                    "effective_answer_mode": "research"})
    s = aggregate([ok, fab, err])
    assert s["counts"] == {"items": 3, "ok": 2, "error": 1, "gold_verified": 0, "gold_unverified": 3}
    assert s["error_rate"] == pytest.approx(1 / 3)
    assert s["citations"]["fabricated_total"] == 2 and s["citations"]["items_with_fabricated"] == 1
    assert s["format"]["compliance_rate"] == 1.0
    assert s["latency"]["research"]["max"] == 90  # timeouts count toward latency
    assert s["authority"]["items_meeting_threshold_rate"] == 1.0


def test_percentile():
    assert percentile([1, 2, 3, 4, 5], 50) == 3
    assert percentile([10, 20], 95) == pytest.approx(19.5)
    assert percentile([], 95) is None
