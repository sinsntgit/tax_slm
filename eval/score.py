"""
(Re-)score a run directory from its saved raw responses - no API calls.

2026-10-14: NEW. `python -m eval.run` calls this at the end of every run;
it is also a CLI of its own so a run can be re-scored after a metric fix or
a gold-item correction without paying for the answers again:

    python -m eval.score eval/runs/<name>                 # uses the run's gold snapshot
    python -m eval.score eval/runs/<name> --gold eval/gold/   # re-score against the current gold set

Run directory layout (written by eval/run.py):
    manifest.json          run configuration + server settings snapshot
    gold_snapshot.jsonl    the exact gold items this run was asked
    responses/<id>.json    one raw record per item (request, status, latency, raw API JSON, judge)
    items.jsonl            per-item metrics (this module)
    summary.json           aggregates: overall, by_category, by_answer_mode (this module)
    report.md              human-readable report (this module)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from eval import metrics, report
from eval.goldset import gold_fingerprint, load_gold

HARNESS_VERSION = "2026-10-14.1"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def record_path(run_dir: Path, item_id: str) -> Path:
    return run_dir / "responses" / f"{item_id}.json"


def load_records(run_dir: Path) -> dict[str, dict]:
    out = {}
    resp_dir = run_dir / "responses"
    if resp_dir.is_dir():
        for p in sorted(resp_dir.glob("*.json")):
            try:
                rec = read_json(p)
            except (ValueError, OSError):
                continue  # half-written/corrupt record -> treated as not done, re-run on resume
            if isinstance(rec, dict) and rec.get("id"):
                out[rec["id"]] = rec
    return out


def load_items_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def score_run(run_dir: str | Path, *, gold_path: str | Path | None = None,
              authority_item_threshold: float = 0.85) -> dict:
    run_dir = Path(run_dir)
    manifest = read_json(run_dir / "manifest.json")
    gold_items = load_gold(gold_path or (run_dir / "gold_snapshot.jsonl"))
    selected = set(manifest.get("item_ids") or [g["id"] for g in gold_items])
    gold_items = [g for g in gold_items if g["id"] in selected]
    records = load_records(run_dir)

    rows = []
    for g in gold_items:
        rec = records.get(g["id"]) or {"id": g["id"], "status": "missing",
                                        "error": {"type": "missing", "message": "item was never run"}}
        rows.append(metrics.score_item(g, rec, authority_item_threshold=authority_item_threshold))

    with open(run_dir / "items.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = dict(manifest)
    manifest.setdefault("gold_fingerprint", gold_fingerprint(gold_items))
    summary = report.build_summary(rows, manifest, authority_item_threshold=authority_item_threshold)
    summary["scored_at"] = now_iso()
    write_json_atomic(run_dir / "summary.json", summary)
    (run_dir / "report.md").write_text(report.render_report(summary, rows), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m eval.score", description="Re-score a saved eval run offline.")
    ap.add_argument("run_dir")
    ap.add_argument("--gold", default=None, help="gold path to score against (default: the run's gold_snapshot.jsonl)")
    ap.add_argument("--authority-threshold", type=float, default=0.85)
    args = ap.parse_args(argv)
    summary = score_run(args.run_dir, gold_path=args.gold, authority_item_threshold=args.authority_threshold)
    o = summary["overall"]
    print(f"scored {o['counts']['items']} items ({o['counts']['error']} errors) -> {Path(args.run_dir) / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
