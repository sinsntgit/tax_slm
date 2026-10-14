"""
Side-by-side comparison of two eval runs + PASS/FAIL against release gates.

2026-10-14: NEW. This is what decides "SLM vs frontier baseline":

    python -m eval.compare <candidate_run_dir> <baseline_run_dir> [--gates eval/gates.json] [--out DIR]

The FIRST run is the candidate under test (e.g. the fine-tuned SLM), the
SECOND is the reference baseline (the frontier model). Gates come from
eval/gates.json (thresholds are config, not code):
  - absolute gates read a candidate metric from its re-aggregated summary
    (dotted path, e.g. "citations.fabricated_total", "latency.research.p95");
  - relative_to_baseline gates compare candidate/baseline MEANS of judge
    dimensions over PAIRED items (answered OK in both runs), so a candidate
    that errors on hard questions can't look better by skipping them.
A metric that can't be computed (no research-mode items, judge not run,
too few paired items) is NOT_MEASURED - never silently PASS.

Verdict: FAIL if any enabled blocking gate FAILs; INCOMPLETE if none fail
but an enabled blocking gate is NOT_MEASURED; otherwise PASS. A verdict on
an unverified (draft) gold set is labelled PROVISIONAL.
Exit code: 0 PASS, 1 FAIL, 3 INCOMPLETE (2 = usage error, argparse).

Writes compare.json + compare.md (default: <candidate>/compare_vs_<baseline name>/).
"""
from __future__ import annotations

import argparse
import json
import operator
import sys
from pathlib import Path

from eval.report import aggregate, fmt_num
from eval.score import load_items_jsonl, read_json, score_run, write_json_atomic

DEFAULT_GATES = Path(__file__).resolve().parent / "gates.json"
OPS = {"<=": operator.le, "<": operator.lt, ">=": operator.ge, ">": operator.gt, "==": operator.eq}


def load_run(run_dir: Path) -> tuple[dict, list[dict]]:
    if not (run_dir / "manifest.json").exists():
        raise SystemExit(f"ERROR: {run_dir} is not an eval run directory (no manifest.json)")
    if not (run_dir / "items.jsonl").exists():
        score_run(run_dir)
    return read_json(run_dir / "manifest.json"), load_items_jsonl(run_dir / "items.jsonl")


def get_path(d: dict, dotted: str):
    cur = d
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def paired_judge(cand_rows, base_rows, metric: str) -> dict:
    base = {r["id"]: r for r in base_rows if r.get("status") == "ok"}
    c_vals, b_vals = [], []
    for r in cand_rows:
        b = base.get(r["id"])
        if r.get("status") != "ok" or b is None:
            continue
        cv, bv = (r.get("judge") or {}).get(metric), (b.get("judge") or {}).get(metric)
        if cv is not None and bv is not None:
            c_vals.append(cv)
            b_vals.append(bv)
    cm, bm = _mean(c_vals), _mean(b_vals)
    ratio = (cm / bm) if (cm is not None and bm) else None
    return {"metric": metric, "n_paired": len(c_vals), "candidate_mean": cm, "baseline_mean": bm, "ratio": ratio}


def evaluate_gates(gates_cfg: dict, cand_sum: dict, base_sum: dict, cand_rows, base_rows) -> list[dict]:
    results = []
    for g in gates_cfg["gates"]:
        res = {"id": g["id"], "description": g.get("description", ""), "blocking": g.get("blocking", True),
               "enabled": g.get("enabled", True), "op": g.get("op"), "threshold": g.get("threshold")}
        if not res["enabled"]:
            res.update(status="NOT_MEASURED", detail="gate not implemented yet (enabled=false)")
            results.append(res)
            continue
        op = OPS[g["op"]]
        if g["type"] == "absolute":
            val, bval = get_path(cand_sum, g["metric"]), get_path(base_sum, g["metric"])
            res.update(metric=g["metric"], candidate=val, baseline=bval)
            if val is None:
                res.update(status="NOT_MEASURED", detail="metric not available for the candidate run")
            else:
                res["status"] = "PASS" if op(val, g["threshold"]) else "FAIL"
        elif g["type"] == "relative_to_baseline":
            dims = [paired_judge(cand_rows, base_rows, m) for m in g["judge_metrics"]]
            min_n = g.get("min_paired_items", 1)
            for d in dims:
                if d["ratio"] is None or d["n_paired"] < min_n:
                    d["status"] = "NOT_MEASURED"
                else:
                    d["status"] = "PASS" if op(d["ratio"], g["threshold"]) else "FAIL"
            res["dimensions"] = dims
            statuses = {d["status"] for d in dims}
            res["status"] = "FAIL" if "FAIL" in statuses else ("NOT_MEASURED" if "NOT_MEASURED" in statuses else "PASS")
            if res["status"] == "NOT_MEASURED":
                missing = [d["metric"] for d in dims if d["status"] == "NOT_MEASURED"]
                res["detail"] = (f"not measurable on >= {min_n} paired items: {', '.join(missing)} "
                                 f"(run BOTH runs with --judge for evaluation.*)")
        else:
            raise ValueError(f"gate {g['id']}: unknown type {g['type']!r}")
        results.append(res)
    return results


def verdict_of(results: list[dict]) -> str:
    blocking = [r for r in results if r["enabled"] and r["blocking"]]
    if any(r["status"] == "FAIL" for r in blocking):
        return "FAIL"
    if any(r["status"] == "NOT_MEASURED" for r in blocking):
        return "INCOMPLETE"
    return "PASS"


def comparability_warnings(cm: dict, bm: dict, cand_rows, base_rows, cand_sum, base_sum) -> list[str]:
    w = []
    if cm.get("gold_fingerprint") != bm.get("gold_fingerprint"):
        w.append(f"Gold sets differ (fingerprints {cm.get('gold_fingerprint')} vs {bm.get('gold_fingerprint')}) - "
                 f"only paired items are comparable.")
    for k in ("endpoint", "retrieval_mode", "answer_mode"):
        if cm.get(k) != bm.get(k):
            w.append(f"Runs used different {k}: {cm.get(k)!r} vs {bm.get(k)!r}.")
    cids, bids = {r["id"] for r in cand_rows}, {r["id"] for r in base_rows}
    if cids != bids:
        w.append(f"Item sets differ: {len(cids - bids)} only in candidate, {len(bids - cids)} only in baseline.")
    for name, s in (("candidate", cand_sum), ("baseline", base_sum)):
        if s["cache_hits"]:
            w.append(f"{name} run has {s['cache_hits']} cache hit(s) - flush Redis and re-run those items.")
        # 2026-10-14 (phase0/followups): model_info-based integrity checks.
        models = s.get("models") or {}
        if len(models) > 1:
            w.append(f"{name} run mixes answer models/prompts {sorted(models)} - not a clean run.")
        if s.get("judge_self_graded_items"):
            w.append(f"{name} run has {s['judge_self_graded_items']} answer(s) judged by the answer model itself - "
                     f"judge.* scores are self-grades (route LLM_ROUTE_ANSWER_QUALITY_JUDGE / "
                     f"LLM_ROUTE_ANSWER_EVALUATION to a reference model).")
    cand_models, base_models = set(cand_sum.get("models") or {}), set(base_sum.get("models") or {})
    if cand_models and cand_models == base_models and cand_models != {"unknown"}:
        w.append(f"Candidate and baseline report the same answer model/prompt {sorted(cand_models)} - "
                 f"is this really a comparison of two different configurations?")
    if not (cm.get("judge") and bm.get("judge")):
        w.append("--judge was not used for both runs, so evaluation.* (the /api/ask/evaluate scorecard) "
                 "cannot be compared.")
    unverified = cand_sum["counts"]["gold_unverified"]
    if unverified:
        w.append(f"PROVISIONAL: {unverified} gold item(s) are unverified drafts (verified=false) - a release "
                 f"decision needs a tax-professional-verified gold set.")
    return w


SIDE_BY_SIDE = [
    ("Items ok / error", lambda s: f"{s['counts']['ok']} / {s['counts']['error']}", None),
    ("Error rate", lambda s: fmt_num(s["error_rate"], pct=True), "error_rate"),
    ("Fabricated citations", lambda s: str(s["citations"]["fabricated_total"]), "citations.fabricated_total"),
    ("Items with fabricated citations", lambda s: str(s["citations"]["items_with_fabricated"]), "citations.items_with_fabricated"),
    ("citation_ok rate", lambda s: fmt_num(s["citations"]["citation_ok_rate"], pct=True), "citations.citation_ok_rate"),
    ("Authority recall in answer (micro)", lambda s: fmt_num(s["authority"]["recall_answer_micro"], pct=True), "authority.recall_answer_micro"),
    ("Items meeting authority threshold", lambda s: fmt_num(s["authority"]["items_meeting_threshold_rate"], pct=True), "authority.items_meeting_threshold_rate"),
    ("Authority recall in sources (micro)", lambda s: fmt_num(s["authority"]["recall_sources_micro"], pct=True), "authority.recall_sources_micro"),
    ("Format compliance (research)", lambda s: fmt_num(s["format"]["compliance_rate"], pct=True), "format.compliance_rate"),
    ("must_not_claim violations", lambda s: str(s["must_not_claim"]["violations_total"]), "must_not_claim.violations_total"),
    ("Key-point coverage", lambda s: fmt_num(s["key_points"]["coverage_mean"], pct=True), "key_points.coverage_mean"),
    ("Latency p50 (s)", lambda s: fmt_num(s["latency"]["all"]["p50"], 1), "latency.all.p50"),
    ("Latency p95 (s)", lambda s: fmt_num(s["latency"]["all"]["p95"], 1), "latency.all.p95"),
    ("Research latency p95 (s)", lambda s: fmt_num(get_path(s, "latency.research.p95"), 1), "latency.research.p95"),
]


def _delta(c, b):
    if isinstance(c, (int, float)) and isinstance(b, (int, float)):
        return f"{c - b:+.3g}"
    return ""


def render_md(out: dict) -> str:
    L = [f"# Eval comparison: {out['candidate']['name']} (candidate) vs {out['baseline']['name']} (baseline)", ""]
    L.append(f"## Verdict: **{out['verdict']}**" + (" (PROVISIONAL - unverified gold)" if out["provisional"] else ""))
    L.append("")
    if out["warnings"]:
        L += ["### Warnings", ""] + [f"- {w}" for w in out["warnings"]] + [""]
    L += ["## Release gates", "", "| Gate | Blocking | Candidate | Baseline | Threshold | Result |", "|---|---|---|---|---|---|"]
    for g in out["gates"]:
        if "dimensions" in g:
            L.append(f"| {g['id']} | {'yes' if g['blocking'] else 'no'} | see below | | ratio {g['op']} {g['threshold']} | **{g['status']}** |")
        else:
            L.append(f"| {g['id']} | {'yes' if g['blocking'] else 'no'} | {fmt_num(g.get('candidate'), 3)} | "
                     f"{fmt_num(g.get('baseline'), 3)} | {g['op']} {g['threshold']} | **{g['status']}** |")
    for g in out["gates"]:
        if "dimensions" in g:
            L += ["", f"### {g['id']} - {g['description']}", "",
                  "| Judge dimension | paired n | candidate mean | baseline mean | ratio | Result |", "|---|---|---|---|---|---|"]
            for d in g["dimensions"]:
                L.append(f"| {d['metric']} | {d['n_paired']} | {fmt_num(d['candidate_mean'], 3)} | {fmt_num(d['baseline_mean'], 3)} | "
                         f"{fmt_num(d['ratio'], 3)} | {d['status']} |")
    notes = [g for g in out["gates"] if g.get("detail")]
    if notes:
        L += [""] + [f"- `{g['id']}`: {g['detail']}" for g in notes]

    cs, bs = out["candidate"]["summary"], out["baseline"]["summary"]
    L += ["", "## Side by side", "", "| Metric | Candidate | Baseline | Delta |", "|---|---|---|---|"]
    for label, fn, path in SIDE_BY_SIDE:
        L.append(f"| {label} | {fn(cs)} | {fn(bs)} | {_delta(get_path(cs, path), get_path(bs, path)) if path else ''} |")
    jkeys = sorted(set(cs["judge"]) | set(bs["judge"]))
    if jkeys:
        L += ["", "| Judge (unpaired means) | Candidate | Baseline |", "|---|---|---|"]
        L += [f"| {k} | {fmt_num((cs['judge'].get(k) or {}).get('mean'), 3)} | {fmt_num((bs['judge'].get(k) or {}).get('mean'), 3)} |"
              for k in jkeys]

    L += ["", "## By category", "",
          "| Category | auth recall C / B | fabricated C / B | format C / B | judge overall C / B | p95 s C / B |",
          "|---|---|---|---|---|---|"]
    for cat in sorted(set(out["by_category"]["candidate"]) | set(out["by_category"]["baseline"])):
        c, b = out["by_category"]["candidate"].get(cat), out["by_category"]["baseline"].get(cat)

        def cell(s, path, pct=False, digits=2):
            return fmt_num(get_path(s, path), digits, pct) if s else "-"
        L.append(f"| {cat} | {cell(c, 'authority.recall_answer_micro', True)} / {cell(b, 'authority.recall_answer_micro', True)} | "
                 f"{cell(c, 'citations.fabricated_total')} / {cell(b, 'citations.fabricated_total')} | "
                 f"{cell(c, 'format.compliance_rate', True)} / {cell(b, 'format.compliance_rate', True)} | "
                 f"{fmt_num((c or {}).get('judge', {}).get('answer_quality.overall', {}).get('mean'))} / "
                 f"{fmt_num((b or {}).get('judge', {}).get('answer_quality.overall', {}).get('mean'))} | "
                 f"{cell(c, 'latency.all.p95', digits=1)} / {cell(b, 'latency.all.p95', digits=1)} |")

    if out["item_diffs"]:
        L += ["", "## Per-item differences (largest judge regressions first)", "",
              "| id | category | status C / B | judge overall C / B | auth C / B | fabricated C / B | format C / B |",
              "|---|---|---|---|---|---|---|"]
        for d in out["item_diffs"]:
            L.append(f"| {d['id']} | {d['category']} | {d['status'][0]} / {d['status'][1]} | "
                     f"{fmt_num(d['judge_overall'][0])} / {fmt_num(d['judge_overall'][1])} | "
                     f"{fmt_num(d['auth_recall'][0], pct=True)} / {fmt_num(d['auth_recall'][1], pct=True)} | "
                     f"{d['fabricated'][0]} / {d['fabricated'][1]} | {d['format'][0]} / {d['format'][1]} |")
    L.append("")
    return "\n".join(L)


def item_diffs(cand_rows, base_rows) -> list[dict]:
    base = {r["id"]: r for r in base_rows}
    out = []
    for c in cand_rows:
        b = base.get(c["id"])
        if b is None:
            continue

        def pick(r):
            ok = r.get("status") == "ok"
            return {
                "status": r.get("status") if ok else (r.get("error") or {}).get("type", r.get("status")),
                "judge": (r.get("judge") or {}).get("answer_quality.overall") if ok else None,
                "auth": r["authorities"]["recall_answer"] if ok else None,
                "fab": r["citations"]["fabricated_count"] if ok else None,
                "fmt": ("n/a" if not r["format"].get("applicable") else ("ok" if r["format"]["compliant"] else "FAIL")) if ok else None,
            }
        pc, pb = pick(c), pick(b)
        if pc == pb:
            continue
        delta = (pc["judge"] - pb["judge"]) if (pc["judge"] is not None and pb["judge"] is not None) else None
        out.append({"id": c["id"], "category": c["category"], "status": [pc["status"], pb["status"]],
                    "judge_overall": [pc["judge"], pb["judge"]], "judge_delta": delta,
                    "auth_recall": [pc["auth"], pb["auth"]], "fabricated": [pc["fab"], pb["fab"]],
                    "format": [pc["fmt"], pb["fmt"]]})
    # Worst regressions first; status changes (errors) before score changes.
    out.sort(key=lambda d: (d["status"][0] == d["status"][1], d["judge_delta"] if d["judge_delta"] is not None else 0.0))
    return out


def compare(cand_dir: str | Path, base_dir: str | Path, gates_path: str | Path = DEFAULT_GATES) -> dict:
    cand_dir, base_dir = Path(cand_dir), Path(base_dir)
    gates_cfg = read_json(Path(gates_path))
    threshold = gates_cfg.get("settings", {}).get("authority_item_threshold", 0.85)
    cm, cand_rows = load_run(cand_dir)
    bm, base_rows = load_run(base_dir)
    cand_sum = aggregate(cand_rows, authority_item_threshold=threshold)
    base_sum = aggregate(base_rows, authority_item_threshold=threshold)

    def by_cat(rows):
        cats = {}
        for r in rows:
            cats.setdefault(r["category"], []).append(r)
        return {k: aggregate(v, authority_item_threshold=threshold) for k, v in sorted(cats.items())}

    results = evaluate_gates(gates_cfg, cand_sum, base_sum, cand_rows, base_rows)
    out = {
        "candidate": {"name": cm.get("name", cand_dir.name), "dir": str(cand_dir), "manifest": cm, "summary": cand_sum},
        "baseline": {"name": bm.get("name", base_dir.name), "dir": str(base_dir), "manifest": bm, "summary": base_sum},
        "gates_file": str(gates_path),
        "gates": results,
        "verdict": verdict_of(results),
        "provisional": cand_sum["counts"]["gold_unverified"] > 0,
        "warnings": comparability_warnings(cm, bm, cand_rows, base_rows, cand_sum, base_sum),
        "by_category": {"candidate": by_cat(cand_rows), "baseline": by_cat(base_rows)},
        "item_diffs": item_diffs(cand_rows, base_rows),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.compare",
                                 description="Compare a CANDIDATE run against a BASELINE run and apply release gates.")
    ap.add_argument("candidate", help="run dir of the model/config under test (e.g. the SLM)")
    ap.add_argument("baseline", help="run dir of the reference (e.g. the frontier model)")
    ap.add_argument("--gates", default=str(DEFAULT_GATES))
    ap.add_argument("--out", default=None, help="output dir (default <candidate>/compare_vs_<baseline name>)")
    ap.add_argument("--quiet", action="store_true", help="don't print the markdown report")
    args = ap.parse_args(argv)

    result = compare(args.candidate, args.baseline, args.gates)
    out_dir = Path(args.out) if args.out else Path(args.candidate) / f"compare_vs_{Path(args.baseline).name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_dir / "compare.json", result)
    md = render_md(result)
    (out_dir / "compare.md").write_text(md, encoding="utf-8")
    if not args.quiet:
        print(md)
    print(f"[eval.compare] verdict {result['verdict']} -> {out_dir / 'compare.md'}", file=sys.stderr)
    return {"PASS": 0, "FAIL": 1, "INCOMPLETE": 3}[result["verdict"]]


if __name__ == "__main__":
    sys.exit(main())
