"""2026-10-14: NEW - eval.compare side-by-side diff + release-gate evaluation."""
import json

import pytest

from eval import compare as eval_compare
from eval import run as eval_run
from eval_fakes import FABRICATED_ANSWER, FakeTarget, ask_response, gold_item, write_gold

from conftest import REPO_ROOT


@pytest.fixture
def gold(tmp_path):
    return write_gold(tmp_path / "gold.jsonl", [
        gold_item(f"item-{i:02d}", f"Q{i:02d}: are software payments royalty under section 195?") for i in range(12)])


def _run(gold, out, *, quality=0.8, judge_overall=8.0, behaviors=None, judge=True):
    with FakeTarget(behaviors, judge_overall=judge_overall, quality=quality) as fake:
        args = ["--gold", str(gold), "--target", fake.url, "--out", str(out), "--retries", "0", "--concurrency", "4"]
        assert eval_run.main(args + (["--judge"] if judge else [])) == 0
    return out


def _gate(result, gid):
    return next(g for g in result["gates"] if g["id"] == gid)


def test_equal_runs_pass_all_measured_gates(gold, tmp_path, capsys):
    base = _run(gold, tmp_path / "frontier")
    cand = _run(gold, tmp_path / "slm", quality=0.76, judge_overall=7.4)   # 95% / 92.5% of baseline
    code = eval_compare.main([str(cand), str(base), "--quiet"])
    result = json.loads((cand / "compare_vs_frontier" / "compare.json").read_text())
    assert result["verdict"] == "PASS" and code == 0
    assert result["provisional"] is True
    assert _gate(result, "zero_fabricated_citations")["status"] == "PASS"
    assert _gate(result, "judge_quality_vs_baseline")["status"] == "PASS"
    assert _gate(result, "cypher_validity")["status"] == "NOT_MEASURED"  # disabled, non-deciding
    md = (cand / "compare_vs_frontier" / "compare.md").read_text()
    assert "Verdict: **PASS**" in md and "PROVISIONAL" in md and "Side by side" in md


def test_fabrication_and_quality_regression_fail(gold, tmp_path):
    base = _run(gold, tmp_path / "frontier")
    cand = _run(gold, tmp_path / "slm", quality=0.6,
                behaviors={"Q03": lambda n: ("json", 200, ask_response(FABRICATED_ANSWER, quality=0.6))})
    result = eval_compare.compare(cand, base)
    assert result["verdict"] == "FAIL"
    assert _gate(result, "zero_fabricated_citations")["status"] == "FAIL"
    jq = _gate(result, "judge_quality_vs_baseline")
    assert jq["status"] == "FAIL"
    ratio = next(d for d in jq["dimensions"] if d["metric"] == "answer_quality.overall")
    assert ratio["n_paired"] == 12 and ratio["ratio"] == pytest.approx(0.75)
    assert result["item_diffs"][0]["id"] in {f"item-{i:02d}" for i in range(12)}
    assert eval_compare.main([str(cand), str(base), "--quiet"]) == 1


def test_missing_judge_is_incomplete_not_pass(gold, tmp_path):
    base = _run(gold, tmp_path / "frontier", judge=False)
    cand = _run(gold, tmp_path / "slm", judge=False)
    result = eval_compare.compare(cand, base)
    assert _gate(result, "judge_quality_vs_baseline")["status"] == "NOT_MEASURED"
    assert result["verdict"] == "INCOMPLETE"
    assert any("--judge was not used" in w for w in result["warnings"])
    assert eval_compare.main([str(cand), str(base), "--quiet"]) == 3


def test_format_latency_and_authority_gates(gold, tmp_path):
    base = _run(gold, tmp_path / "frontier")
    cand = _run(gold, tmp_path / "slm", behaviors={
        "Q05": lambda n: ("json", 200, ask_response("Plain answer, no structure, mentions nothing."))})
    result = eval_compare.compare(cand, base)
    fmt = _gate(result, "format_compliance")
    assert fmt["status"] == "FAIL" and fmt["candidate"] == pytest.approx(11 / 12)
    auth = _gate(result, "authority_recall_per_question")
    assert auth["status"] == "FAIL" and auth["candidate"] == pytest.approx(11 / 12)
    assert _gate(result, "research_latency")["status"] == "PASS"
    assert _gate(result, "must_not_claim")["blocking"] is False


def test_custom_gates_file_and_cache_validity(gold, tmp_path):
    base = _run(gold, tmp_path / "frontier")
    cand = _run(gold, tmp_path / "slm", behaviors={
        "Q01": lambda n: ("json", 200, dict(ask_response("x"), from_cache=True))})
    gates = json.loads((REPO_ROOT / "eval" / "gates.json").read_text())
    gates["gates"] = [g for g in gates["gates"] if g["id"] == "run_validity_no_cache_hits"]
    p = tmp_path / "gates.json"
    p.write_text(json.dumps(gates))
    result = eval_compare.compare(cand, base, p)
    assert [g["id"] for g in result["gates"]] == ["run_validity_no_cache_hits"]
    assert result["verdict"] == "FAIL"
    assert any("cache hit" in w for w in result["warnings"])


def test_gates_json_shape():
    gates = json.loads((REPO_ROOT / "eval" / "gates.json").read_text())
    ids = {g["id"] for g in gates["gates"]}
    assert {"zero_fabricated_citations", "judge_quality_vs_baseline", "authority_recall_per_question",
            "format_compliance", "research_latency", "cypher_validity", "extraction_accuracy"} <= ids
    for g in gates["gates"]:
        assert g["op"] in eval_compare.OPS and g["type"] in {"absolute", "relative_to_baseline"}
