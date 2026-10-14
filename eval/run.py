"""
Eval runner - asks every gold question through the REAL HTTP API and saves
the raw responses, then scores them (eval/score.py).

2026-10-14: NEW. Calls the running Flask app over HTTP rather than importing
_api_ask_impl() so it measures exactly what a user gets (routing, caching,
the agentic wrapper, disclaimer append, real latency) - and so the same
command can later be pointed at an SLM-backed deployment unchanged.

    python -m eval.run --gold eval/gold/ --target http://localhost:5000 \\
        --retrieval-mode hybrid --answer-mode research --out eval/runs/<name> [--judge]

Behaviour:
  - Resumable: every item's record is written atomically to
    <out>/responses/<id>.json as soon as it finishes; re-running the same
    command skips items already answered OK (errored items are retried).
    Ctrl-C at any point loses at most the in-flight items.
  - --concurrency N parallel requests (default 2 - the Flask dev server and
    the LLM provider's rate limits, not this harness, are the bottleneck).
  - --timeout per request, --retries on transient failures (see client.py
    for which failures are retried).
  - --judge also POSTs each answer to /api/ask/evaluate (grounded v1
    scorecard + no-context general-knowledge scorecard). Done as a SEPARATE
    call after the answer is saved, so judge time never pollutes the
    measured answer latency (auto_evaluate=true on /api/ask would).
  - --retrieval-mode / --answer-mode, when given, override every item's own
    value (comparability across a run); when omitted each item's gold
    answer_mode / retrieval_mode (default hybrid) is used.
  - Always sends include_full_chunks=true so the scorer can see the full
    retrieved chunk text (authority-in-sources + fabricated-citation checks).
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from eval.client import call_json
from eval.goldset import GoldValidationError, gold_fingerprint, load_gold
from eval.score import (HARNESS_VERSION, load_records, now_iso, read_json, record_path, score_run,
                        write_json_atomic)

ENDPOINTS = {"ask": "/api/ask", "fast": "/api/ask/fast"}
EVALUATE_PATH = "/api/ask/evaluate"
SETTINGS_PATH = "/api/settings/summary"
# Manifest fields that must match for a resume to append to an existing run -
# mixing answers produced under different targets/modes would make the run's
# numbers meaningless.
RESUME_KEYS = ("target", "endpoint", "retrieval_mode", "answer_mode")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m eval.run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--gold", default="eval/gold/", help="gold .jsonl file or directory (default eval/gold/)")
    ap.add_argument("--target", default="http://localhost:5000", help="base URL of the running app")
    ap.add_argument("--out", required=True, help="run directory, e.g. eval/runs/2026-10-14_frontier_hybrid")
    ap.add_argument("--endpoint", choices=sorted(ENDPOINTS), default="ask",
                    help="ask = /api/ask (default, the research path); fast = /api/ask/fast")
    ap.add_argument("--retrieval-mode", default=None,
                    choices=["hybrid", "vector_only", "semantic_graph", "agentic", "structured_research"])
    ap.add_argument("--answer-mode", default=None, choices=["lookup", "research", "opinion"])
    ap.add_argument("--judge", action="store_true", help="also score each answer via /api/ask/evaluate")
    ap.add_argument("--concurrency", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=300.0, help="per-request timeout, seconds (default 300)")
    ap.add_argument("--judge-timeout", type=float, default=300.0)
    ap.add_argument("--retries", type=int, default=2, help="retries on transient errors (default 2)")
    ap.add_argument("--backoff", type=float, default=2.0, help="base backoff seconds, doubled per retry")
    ap.add_argument("--retry-on-timeout", action="store_true", help="also retry timed-out requests")
    ap.add_argument("--ids", nargs="*", default=None, help="only these gold ids")
    ap.add_argument("--category", action="append", default=None, help="only these categories (repeatable)")
    ap.add_argument("--limit", type=int, default=None, help="only the first N selected items (smoke test)")
    ap.add_argument("--rerun", action="store_true", help="re-ask items already answered OK in this run dir")
    ap.add_argument("--force", action="store_true",
                    help="resume even if target/endpoint/modes differ from the existing manifest")
    ap.add_argument("--no-preflight", action="store_true", help="skip the reachability check")
    ap.add_argument("--authority-threshold", type=float, default=0.85)
    ap.add_argument("--notes", default="", help="free text stored in the manifest (e.g. model/prompt under test)")
    return ap


def effective_modes(item: dict, args) -> tuple[str, str]:
    answer_mode = args.answer_mode or item["answer_mode"]
    retrieval_mode = args.retrieval_mode or item.get("retrieval_mode") or "hybrid"
    return answer_mode, retrieval_mode


def ask_body(item: dict, answer_mode: str, retrieval_mode: str) -> dict:
    return {"question": item["question"], "answer_mode": answer_mode, "retrieval_mode": retrieval_mode,
            "include_full_chunks": True}


def _server_settings(target: str) -> tuple[dict | None, str | None]:
    """Best-effort snapshot of what the server says it is running (provider,
    temperature, embedding model) for the manifest - comparisons are only
    meaningful when you can see what changed between two runs. Only a
    connection failure is fatal (server not up); anything else is recorded."""
    res = call_json(target.rstrip("/") + SETTINGS_PATH, None, method="GET", timeout=15, retries=0)
    if res.status != "ok":
        return None, (res.error or {}).get("type")
    s = res.response
    snapshot = {
        "llm_provider": ((s.get("temperature") or {}).get("provider")),
        "temperature": ((s.get("temperature") or {}).get("value")),
        "embedding_model": ((s.get("embedding_model") or {}).get("name")),
        "neo4j_connected": ((s.get("neo4j") or {}).get("connected")),
    }
    return snapshot, None


def run(args) -> int:
    out = Path(args.out)
    try:
        items = load_gold(args.gold, ids=args.ids, categories=args.category)
    except (GoldValidationError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if args.limit:
        items = items[: args.limit]
    if not items:
        print("ERROR: no gold items selected", file=sys.stderr)
        return 1
    target = args.target.rstrip("/")

    manifest_path = out / "manifest.json"
    new_cfg = {"target": target, "endpoint": ENDPOINTS[args.endpoint],
               "retrieval_mode": args.retrieval_mode, "answer_mode": args.answer_mode}
    old = read_json(manifest_path) if manifest_path.exists() else None
    if old and not args.force:
        diffs = {k: (old.get(k), new_cfg[k]) for k in RESUME_KEYS if old.get(k) != new_cfg[k]}
        if diffs:
            print(f"ERROR: {out} already holds a run with different settings {diffs}. "
                  f"Use a new --out, or --force to mix them (not recommended).", file=sys.stderr)
            return 2

    settings, settings_err = (None, None) if args.no_preflight else _server_settings(target)
    if settings_err == "connection_error":
        print(f"ERROR: cannot reach {target} - is the app running (docker compose up -d; "
              f"python -m framework.webapp.app --domain tax)? Use --no-preflight to skip this check.",
              file=sys.stderr)
        return 2

    out.mkdir(parents=True, exist_ok=True)
    (out / "responses").mkdir(exist_ok=True)
    # Self-contained run: the exact gold items asked are snapshotted beside the
    # answers, so re-scoring/comparing later never depends on eval/gold/ being unchanged.
    # Merged with any earlier snapshot so a resumed run with a different
    # --ids/--category filter still has every item it ever asked.
    gold_src = Path(args.gold)
    snapshot_path = out / "gold_snapshot.jsonl"
    snapshot = {}
    if snapshot_path.exists():
        try:
            snapshot = {g["id"]: g for g in load_gold(snapshot_path)}
        except GoldValidationError:
            snapshot = {}
    snapshot.update({it["id"]: it for it in items})
    with open(snapshot_path, "w", encoding="utf-8") as fh:
        for it in sorted(snapshot.values(), key=lambda g: g["id"]):
            fh.write(json.dumps({k: v for k, v in it.items() if not k.startswith("_")}, ensure_ascii=False) + "\n")
    manifest = dict(old or {})
    manifest.update(new_cfg)
    manifest.update({
        "name": out.name,
        "judge": bool(args.judge or (old or {}).get("judge")),
        "gold_path": str(gold_src),
        "gold_fingerprint": gold_fingerprint(items),
        "item_ids": sorted(set((old or {}).get("item_ids") or []) | {i["id"] for i in items}),
        "started_at": (old or {}).get("started_at") or now_iso(),
        "last_resumed_at": now_iso() if old else None,
        "finished_at": None,
        "server_settings": settings if settings is not None else (old or {}).get("server_settings"),
        "server_settings_error": settings_err,
        "harness_version": HARNESS_VERSION,
        "concurrency": args.concurrency, "timeout_s": args.timeout, "retries": args.retries,
        "notes": args.notes or (old or {}).get("notes", ""),
    })
    write_json_atomic(manifest_path, manifest)

    existing = load_records(out)
    todo = []
    for it in items:
        answer_mode, retrieval_mode = effective_modes(it, args)
        body = ask_body(it, answer_mode, retrieval_mode)
        rec = existing.get(it["id"])
        stale = rec is not None and (rec.get("request") or {}).get("body") != body
        need_answer = args.rerun or rec is None or rec.get("status") != "ok" or stale
        need_judge = args.judge and (need_answer or (rec.get("judge") or {}).get("status") != "ok")
        if need_answer or need_judge:
            todo.append((it, body, answer_mode, retrieval_mode, need_answer, need_judge))
    print(f"[eval] {len(items)} items selected, {len(items) - len(todo)} already done, {len(todo)} to run "
          f"-> {out}")

    lock = threading.Lock()
    done_count = [0]

    def work(it, body, answer_mode, retrieval_mode, need_answer, need_judge):
        rec = existing.get(it["id"]) if not need_answer else None
        if need_answer:
            started = now_iso()
            res = call_json(target + ENDPOINTS[args.endpoint], body, timeout=args.timeout, retries=args.retries,
                            backoff_s=args.backoff, retry_on_timeout=args.retry_on_timeout, expect_key="answer")
            rec = {
                "id": it["id"], "question": it["question"],
                "effective_answer_mode": answer_mode, "effective_retrieval_mode": retrieval_mode,
                "request": {"endpoint": ENDPOINTS[args.endpoint], "body": body},
                "status": res.status, "http_status": res.http_status, "attempts": res.attempts,
                "latency_s": res.latency_s, "attempt_log": res.attempt_log,
                "started_at": started, "finished_at": now_iso(),
                "response": res.response, "raw_text": res.raw_text, "error": res.error,
            }
            write_json_atomic(record_path(out, it["id"]), rec)
        if need_judge and rec and rec.get("status") == "ok":
            jbody = {"question": it["question"], "answer": rec["response"].get("answer", ""),
                     "retrieval_mode": retrieval_mode}
            jres = call_json(target + EVALUATE_PATH, jbody, timeout=args.judge_timeout, retries=args.retries,
                             backoff_s=args.backoff, retry_on_timeout=args.retry_on_timeout, expect_key="evaluation")
            rec["judge"] = {"status": jres.status, "http_status": jres.http_status, "latency_s": jres.latency_s,
                            "attempts": jres.attempts, "response": jres.response, "error": jres.error}
            write_json_atomic(record_path(out, it["id"]), rec)
        with lock:
            done_count[0] += 1
            status = rec.get("status") if rec else "?"
            extra = f" ({(rec.get('error') or {}).get('type')})" if rec and rec.get("error") else ""
            print(f"[eval] {done_count[0]}/{len(todo)} {it['id']}: {status}{extra} "
                  f"{rec.get('latency_s') if rec else ''}s", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
        futures = [pool.submit(work, *t) for t in todo]
        for f in as_completed(futures):
            exc = f.exception()
            if exc is not None:  # one bad item never stops the batch (project convention)
                print(f"[eval] internal error on an item: {exc!r}", file=sys.stderr)

    manifest["finished_at"] = now_iso()
    write_json_atomic(manifest_path, manifest)
    summary = score_run(out, authority_item_threshold=args.authority_threshold)
    o = summary["overall"]
    print(f"[eval] done: {o['counts']['ok']} ok, {o['counts']['error']} errors; "
          f"fabricated citations {o['citations']['fabricated_total']}; "
          f"format {o['format']['compliant']}/{o['format']['applicable']}; "
          f"latency p95 {o['latency']['all']['p95']}s -> {out / 'report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
