"""2026-10-14: NEW - eval.run over real HTTP against a fake target: metrics,
error classification, retries, resume and the judge pass."""
import json

from eval import run as eval_run
from eval.client import call_json
from eval_fakes import FABRICATED_ANSWER, FakeTarget, ask_response, gold_item, write_gold


def _gold(tmp_path):
    return write_gold(tmp_path / "gold" / "g.jsonl", [
        gold_item("good-item", "GOOD: are software payments royalty under section 195?"),
        gold_item("fab-item", "FAB: are software payments royalty under section 195?"),
        gold_item("malformed-item", "MALFORMED: are software payments royalty under section 195?"),
        gold_item("slow-item", "SLOW: are software payments royalty under section 195?"),
        gold_item("flaky-item", "FLAKY: are software payments royalty under section 195?"),
        gold_item("cached-item", "CACHED: are software payments royalty under section 195?", category="lookup",
                  answer_mode="lookup"),
    ])


def _behaviors(state):
    return {
        "FAB:": lambda n: ("json", 200, ask_response(FABRICATED_ANSWER)),
        "MALFORMED:": lambda n: ("raw", 200, "<html>proxy error</html>"),
        "SLOW:": lambda n: ("sleep", 3.0, ask_response("late")) if state["slow"] else ("json", 200, ask_response("## a. x")),
        "FLAKY:": lambda n: ("json", 500, {"error": "LLM provider unreachable"}) if n == 1 else
                            ("json", 200, ask_response("Not royalty. Engineering Analysis says Section 195 TDS is not required.")),
        "CACHED:": lambda n: ("json", 200, ask_response("cached answer", from_cache=True)),
    }


def _args(gold, out, url, *extra):
    return ["--gold", str(gold), "--target", url, "--out", str(out), "--timeout", "1", "--retries", "1",
            "--backoff", "0.01", "--concurrency", "3", *extra]


def test_run_scores_classifies_and_resumes(tmp_path):
    gold, out = _gold(tmp_path), tmp_path / "runs" / "r1"
    state = {"slow": True}
    with FakeTarget(_behaviors(state)) as fake:
        assert eval_run.main(_args(gold, out, fake.url)) == 0
        recs = {p.stem: json.loads(p.read_text()) for p in (out / "responses").glob("*.json")}
        assert recs["good-item"]["status"] == "ok"
        assert recs["good-item"]["request"]["body"]["include_full_chunks"] is True
        assert recs["malformed-item"]["error"]["type"] == "malformed_response"
        assert recs["slow-item"]["error"]["type"] == "timeout"
        assert recs["flaky-item"]["status"] == "ok" and recs["flaky-item"]["attempts"] == 2
        assert fake.calls[next(q for q in fake.calls if q.startswith("MALFORMED"))] == 1  # not retried
        assert fake.calls[next(q for q in fake.calls if q.startswith("SLOW"))] == 1  # timeouts not retried by default

        summary = json.loads((out / "summary.json").read_text())
        o = summary["overall"]
        assert o["counts"] == {"items": 6, "ok": 4, "error": 2, "gold_verified": 0, "gold_unverified": 6}
        assert o["error_types"] == {"malformed_response": 1, "timeout": 1}
        assert o["citations"]["fabricated_total"] == 2 and o["citations"]["items_with_fabricated"] == 1
        assert o["cache_hits"] == 1
        assert set(summary["by_category"]) == {"research", "lookup"}
        assert summary["run"]["server_settings"]["llm_provider"] == "FAKE"
        report = (out / "report.md").read_text()
        assert "Imaginary Software" in report and "INVALID FOR COMPARISON" in report and "PROVISIONAL" in report
        items = [json.loads(line) for line in (out / "items.jsonl").read_text().splitlines()]
        assert len(items) == 6

        # Resume: only the two errored items are re-asked; OK items are not.
        good_q = next(q for q in fake.calls if q.startswith("GOOD"))
        state["slow"] = False
        assert eval_run.main(_args(gold, out, fake.url)) == 0
        assert fake.calls[good_q] == 1
        assert fake.calls[next(q for q in fake.calls if q.startswith("SLOW"))] == 2
        o2 = json.loads((out / "summary.json").read_text())["overall"]
        assert o2["counts"]["error"] == 1  # malformed still malformed

        # Resume with --judge: judge-only pass for answered items, no re-asking.
        assert eval_run.main(_args(gold, out, fake.url, "--judge")) == 0
        assert fake.calls[good_q] == 1
        assert fake.judge_calls == 5
        o3 = json.loads((out / "summary.json").read_text())["overall"]
        assert o3["judge"]["evaluation.overall"]["n"] == 5


def test_resume_refuses_different_settings(tmp_path):
    gold, out = _gold(tmp_path), tmp_path / "r"
    with FakeTarget() as fake:
        assert eval_run.main(_args(gold, out, fake.url, "--ids", "good-item")) == 0
        assert eval_run.main(_args(gold, out, fake.url, "--ids", "good-item", "--answer-mode", "lookup")) == 2
        # a mode override that changes the request body re-asks the item (stale record) when forced
        assert eval_run.main(_args(gold, out, fake.url, "--ids", "good-item", "--answer-mode", "lookup", "--force")) == 0
        rec = json.loads((out / "responses" / "good-item.json").read_text())
        assert rec["request"]["body"]["answer_mode"] == "lookup"


def test_resume_with_new_filter_keeps_earlier_items(tmp_path):
    gold, out = _gold(tmp_path), tmp_path / "r"
    with FakeTarget() as fake:
        assert eval_run.main(_args(gold, out, fake.url, "--ids", "good-item")) == 0
        assert eval_run.main(_args(gold, out, fake.url, "--ids", "flaky-item")) == 0
    ids = [json.loads(line)["id"] for line in (out / "items.jsonl").read_text().splitlines()]
    assert sorted(ids) == ["flaky-item", "good-item"]


def test_preflight_fails_fast_when_target_down(tmp_path, capsys):
    gold = _gold(tmp_path)
    assert eval_run.main(_args(gold, tmp_path / "r", "http://127.0.0.1:9")) == 2
    assert "cannot reach" in capsys.readouterr().err
    assert not (tmp_path / "r" / "responses").exists()


def test_invalid_gold_exits_1(tmp_path):
    bad = write_gold(tmp_path / "bad.jsonl", [{"id": "x"}])
    assert eval_run.main(["--gold", str(bad), "--out", str(tmp_path / "r"), "--no-preflight"]) == 1


def test_client_retries_then_gives_up_on_http_400(tmp_path):
    with FakeTarget({"BAD": lambda n: ("json", 400, {"error": "Invalid retrieval_mode"})}) as fake:
        res = call_json(fake.url + "/api/ask", {"question": "BAD request"}, retries=3, backoff_s=0.01, expect_key="answer")
    assert res.status == "error" and res.attempts == 1
    assert res.error == {"type": "http_error", "http_status": 400, "message": "Invalid retrieval_mode"}
