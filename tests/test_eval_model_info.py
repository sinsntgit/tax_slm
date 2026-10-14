"""2026-10-14: NEW (phase0/followups) - the eval harness sends bypass_cache=true,
records each response's model_info, and flags mixed-model runs, self-graded
judges and same-model comparisons (eval/run.py, metrics.py, report.py,
compare.py)."""
import json

from eval import compare as eval_compare
from eval import run as eval_run
from eval_fakes import FakeTarget, GOOD_RESEARCH_ANSWER, ask_response, gold_item, model_info, write_gold


def _with_info(**kw):
    def behave(n):
        resp = ask_response(GOOD_RESEARCH_ANSWER)
        resp["model_info"] = model_info(**kw)
        return "json", 200, resp
    return behave


def _gold(tmp_path, n=4):
    return write_gold(tmp_path / "gold.jsonl", [
        gold_item(f"item-{i}", f"Q{i}: are software payments royalty under section 195?") for i in range(n)])


def _run(gold, out, behaviors):
    with FakeTarget(behaviors) as fake:
        assert eval_run.main(["--gold", str(gold), "--target", fake.url, "--out", str(out),
                              "--retries", "0", "--concurrency", "2"]) == 0
    return out


def test_runner_sends_bypass_cache_and_records_model_info(tmp_path):
    gold = _gold(tmp_path)
    out = _run(gold, tmp_path / "slm", {"Q": _with_info(provider="LOCAL", model="tax-slm-7b")})
    rec = json.loads((out / "responses" / "item-0.json").read_text())
    assert rec["request"]["body"]["bypass_cache"] is True
    assert rec["model_info"]["provider"] == "LOCAL"
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["server_settings"]["model_info"]["answer_prompt"]["version"] == "v7"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["overall"]["models"] == {"LOCAL/tax-slm-7b answer_generation@v7": 4}
    assert summary["overall"]["judge_self_graded_items"] == 0
    report = (out / "report.md").read_text()
    assert "LOCAL/tax-slm-7b answer_generation@v7 x4" in report
    assert "MIXED MODELS" not in report and "SELF-GRADED" not in report


def test_mixed_models_and_self_grading_are_flagged(tmp_path):
    gold = _gold(tmp_path)
    out = _run(gold, tmp_path / "mixed", {
        "Q0": _with_info(provider="CLAUDE", model="claude-x"),
        "Q": _with_info(provider="LOCAL", model="tax-slm-7b", judge_same=True),
    })
    o = json.loads((out / "summary.json").read_text())["overall"]
    assert len(o["models"]) == 2 and o["judge_self_graded_items"] == 3
    report = (out / "report.md").read_text()
    assert "MIXED MODELS" in report and "SELF-GRADED" in report


def test_compare_warns_on_mixed_self_graded_and_same_model(tmp_path):
    gold = _gold(tmp_path)
    base = _run(gold, tmp_path / "frontier", {"Q": _with_info(provider="CLAUDE", model="claude-x")})
    same = _run(gold, tmp_path / "same", {"Q": _with_info(provider="CLAUDE", model="claude-x")})
    warnings = eval_compare.compare(same, base)["warnings"]
    assert any("same answer model/prompt" in w for w in warnings)

    cand = _run(gold, tmp_path / "slm", {
        "Q0": _with_info(provider="CLAUDE", model="claude-x"),
        "Q": _with_info(provider="LOCAL", model="tax-slm-7b", judge_same=True),
    })
    warnings = eval_compare.compare(cand, base)["warnings"]
    assert any("mixes answer models" in w for w in warnings)
    assert any("judged by the answer model itself" in w for w in warnings)
    assert not any("same answer model/prompt" in w for w in warnings)


def test_old_server_without_model_info_is_unknown_not_an_error(tmp_path):
    gold = _gold(tmp_path, n=2)
    out = _run(gold, tmp_path / "old", None)
    o = json.loads((out / "summary.json").read_text())["overall"]
    assert o["models"] == {"unknown": 2}
