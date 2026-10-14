"""2026-10-14: NEW - gold-set schema/loader + starter-set traceability tests."""
import json
import re
from collections import Counter

import pytest

from eval import metrics
from eval.goldset import GoldValidationError, gold_fingerprint, load_gold, main as goldset_main
from eval_fakes import gold_item, write_gold

from conftest import REPO_ROOT

GOLD_DIR = REPO_ROOT / "eval" / "gold"
CATEGORIES = {"lookup", "research", "for_against", "change_in_law", "treaty", "computation", "opinion"}
# Key points that are derived arithmetic rather than verbatim fixture facts - flagged in the item's notes.
DERIVED_KEY_POINTS = {("op12-equalisation-levy-computation", 0)}


def test_starter_gold_set_is_valid_and_all_drafts():
    items = load_gold(GOLD_DIR)
    assert 38 <= len(items) <= 45
    assert {i["category"] for i in items} == CATEGORIES
    assert all(i["verified"] is False and i["verified_by"] == "claude-draft" for i in items)
    assert len({i["id"] for i in items}) == len(items)
    assert min(Counter(i["category"] for i in items).values()) >= 3


@pytest.mark.parametrize("item", load_gold(GOLD_DIR), ids=lambda i: i["id"])
def test_starter_item_traceable_to_fixture(item):
    """Every required authority and every scored key point must be findable in
    the fixture the item cites - the drafts must not invent facts."""
    assert item["sources"], "starter items must cite their fixture"
    text = (REPO_ROOT / item["sources"][0]).read_text(encoding="utf-8")
    norm = metrics.normalize(text)
    for a in item["required_authorities"]:
        assert metrics.find_authority(a, norm), f"{a['label']} not found in {item['sources'][0]}"
    for idx, kp in enumerate(item["expected_key_points"]):
        if (item["id"], idx) in DERIVED_KEY_POINTS or not kp.get("match_any"):
            continue
        assert any(re.search(rx, text, re.IGNORECASE) for rx in kp["match_any"]), kp["point"]


def test_validation_reports_all_errors_with_locations(tmp_path):
    good = gold_item("good-item-1", "What is the TDS position on software payments?")
    bad_enum = dict(gold_item("bad-enum", "What is the TDS position on software payments?"), category="vibes")
    missing = gold_item("missing-field", "What is the TDS position on software payments?")
    del missing["verified_by"]
    bad_regex = gold_item("bad-regex", "What is the TDS position on software payments?",
                          mnc=[{"claim": "x is y", "patterns": ["(unclosed"]}])
    self_verified = dict(gold_item("self-verified", "What is the TDS position on software payments?"), verified=True)
    extra = dict(gold_item("extra-field", "What is the TDS position on software payments?"), answer="typo field")
    path = write_gold(tmp_path / "g.jsonl", [good, bad_enum, missing, bad_regex, self_verified, extra, good])
    with path.open("a") as fh:
        fh.write("{not json\n")
    with pytest.raises(GoldValidationError) as ei:
        load_gold(path)
    errs = "\n".join(ei.value.errors)
    assert "g.jsonl:2 [bad-enum] $.category" in errs
    assert "missing required field 'verified_by'" in errs
    assert "invalid regular expression" in errs
    assert "cannot be verified=true" in errs
    assert "unknown field 'answer'" in errs
    assert "duplicate id (first defined at g.jsonl:1)" in errs
    assert "g.jsonl:8: not valid JSON" in errs


def test_filters_and_fingerprint(tmp_path):
    a = gold_item("item-a", "What is the TDS position on software payments?", category="lookup")
    b = gold_item("item-b", "What is the TDS position on software payments?", category="treaty")
    path = write_gold(tmp_path / "g.jsonl", [a, b])
    assert [i["id"] for i in load_gold(path, categories=["treaty"])] == ["item-b"]
    assert [i["id"] for i in load_gold(path, ids=["item-a"])] == ["item-a"]
    with pytest.raises(GoldValidationError):
        load_gold(path, ids=["nope"])
    fp = gold_fingerprint(load_gold(path))
    b2 = dict(b, question="A different question about software payments?")
    assert gold_fingerprint(load_gold(write_gold(tmp_path / "g2.jsonl", [a, b2]))) != fp


def test_goldset_cli(capsys):
    assert goldset_main([str(GOLD_DIR)]) == 0
    assert "unverified drafts" in capsys.readouterr().out


def test_schema_file_is_valid_json():
    schema = json.loads((REPO_ROOT / "eval" / "gold_item.schema.json").read_text())
    assert set(schema["properties"]["category"]["enum"]) == CATEGORIES
