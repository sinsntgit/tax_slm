"""
Export LLM call logs (LLM_CALL_LOG=1, see framework.core.infra.llm_call_log)
as chat-format fine-tuning data: one JSONL file per call family, each line

    {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

- the format TRL's SFTTrainer, Axolotl, LLaMA-Factory and Unsloth all accept.

2026-10-14: NEW - Phase 0 of the Tax SLM product. Call names with a dynamic
suffix (case_law_extraction_<file>, semantic_layer_group_3_of_12, ...) are
grouped into one family file, because the per-file suffix carries no
training signal and one adapter/dataset per task is what fine-tuning needs.

The output contains client data exactly like the input logs do - keep it
inside the customer's environment.

Usage:
    python -m framework.cli.export_training_data --out data/training_exports/run1
    python -m framework.cli.export_training_data --call nl2cypher --call intent_detection \
        --since 2026-10-01 --until 2026-10-31 --min-judge-score 0.8 --out data/training_exports/run2
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from framework.core.infra import llm_call_log

# Dynamic per-document / per-group suffixes appended by ingestion call sites
# (see the call-name table in README "Running with a local model"). Stripped
# to get the stable family name used for grouping and --call matching.
_FAMILY_PATTERNS = [
    (re.compile(r"^(semantic_layer_group|enrichment_group)_\d+_of_\d+$"), r"\1"),
    (re.compile(r"^(case_law_extraction|circular_extraction|treaty_preamble)_.+$"), r"\1"),
]


def call_family(name: str) -> str:
    for pattern, repl in _FAMILY_PATTERNS:
        if pattern.match(name or ""):
            return pattern.sub(repl, name)
    return name or "llm_call"


def _parse_when(value: str, end_of_day: bool = False) -> datetime:
    """Accepts YYYY-MM-DD or a full ISO timestamp; naive values are UTC.
    A bare --until date is inclusive (end of that day)."""
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        dt = datetime.fromisoformat(value + ("T23:59:59.999999" if end_of_day else "T00:00:00"))
    else:
        dt = datetime.fromisoformat(value)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _score(rec: dict) -> float:
    try:
        return float(rec.get("judge_score"))
    except (TypeError, ValueError):
        return -1.0


def iter_log_records(log_dir: Path):
    for path in sorted(log_dir.glob("llm_calls_*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    # One bad line never stops an export (e.g. a partial
                    # write from a killed process) - skip it and say so.
                    print(f"WARNING: skipping unparseable line {path.name}:{lineno}", file=sys.stderr)


def load_calls(log_dir: Path) -> list[dict]:
    """Returns call records with any later annotations merged in."""
    calls: dict[str, dict] = {}
    annotations: dict[str, dict] = defaultdict(dict)
    for rec in iter_log_records(log_dir):
        if rec.get("type", "call") == "annotation":
            extra = {k: v for k, v in rec.items() if k not in ("type", "call_id", "ts", "schema_version")}
            annotations[rec.get("call_id")].update(extra)
        else:
            calls[rec.get("call_id") or hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()] = rec
    for cid, extra in annotations.items():
        if cid in calls:
            calls[cid].update(extra)
    return list(calls.values())


def to_messages(rec: dict, merge_system: bool = False) -> list[dict]:
    system, user, output = rec.get("system") or "", rec.get("user") or "", rec.get("output") or ""
    if merge_system and system:
        return [{"role": "user", "content": f"{system}\n\n{user}"}, {"role": "assistant", "content": output}]
    msgs = [{"role": "system", "content": system}] if system else []
    return msgs + [{"role": "user", "content": user}, {"role": "assistant", "content": output}]


def export(log_dir: Path, out_dir: Path, *, calls: list[str] | None = None, purpose: str | None = None,
           provider: str | None = None, since: datetime | None = None, until: datetime | None = None,
           min_judge_score: float | None = None, dedup: str = "prompt", merge_system: bool = False,
           allow_unmasked: bool = False, with_metadata: bool = False) -> dict:
    """Returns a stats dict: {"written": {family: n}, "skipped": {reason: n}, "files": [...]}."""
    wanted = {c.strip() for c in (calls or []) if c.strip()}
    skipped: Counter = Counter()
    kept: dict[str, dict[str, dict]] = defaultdict(dict)   # family -> dedup key -> record

    for rec in load_calls(log_dir):
        name = rec.get("name") or ""
        family = call_family(name)
        if wanted and name not in wanted and family not in wanted:
            skipped["call_filter"] += 1
            continue
        if purpose and rec.get("purpose") != purpose:
            skipped["purpose_filter"] += 1
            continue
        if provider and (rec.get("provider") or "").upper() != provider.upper():
            skipped["provider_filter"] += 1
            continue
        if rec.get("error") or not (rec.get("output") or "").strip():
            skipped["error_or_empty_output"] += 1
            continue
        if not rec.get("pii_masked") and not allow_unmasked:
            skipped["not_pii_masked"] += 1
            continue
        try:
            ts = _parse_when(rec.get("ts") or "")
        except ValueError:
            skipped["bad_timestamp"] += 1
            continue
        if (since and ts < since) or (until and ts > until):
            skipped["date_filter"] += 1
            continue
        if min_judge_score is not None:
            score = rec.get("judge_score")
            try:
                ok = score is not None and float(score) >= min_judge_score
            except (TypeError, ValueError):
                ok = False
            if not ok:
                skipped["judge_score_filter"] += 1
                continue

        if dedup == "none":
            key = rec.get("call_id") or str(id(rec))
        else:
            parts = [rec.get("system") or "", rec.get("user") or ""]
            if dedup == "prompt_output":
                parts.append(rec.get("output") or "")
            key = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
        prev = kept[family].get(key)
        if prev is not None:
            skipped["duplicate"] += 1
            # Keep the better-judged example, else the most recent one.
            if (_score(prev), prev.get("ts") or "") > (_score(rec), rec.get("ts") or ""):
                continue
        kept[family][key] = rec

    out_dir.mkdir(parents=True, exist_ok=True)
    written, files = {}, []
    for family in sorted(kept):
        records = sorted(kept[family].values(), key=lambda r: r.get("ts") or "")
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", family)
        path = out_dir / f"{safe}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for rec in records:
                row = {"messages": to_messages(rec, merge_system=merge_system)}
                if with_metadata:
                    row["metadata"] = {k: rec.get(k) for k in ("call_id", "ts", "name", "purpose", "provider", "model",
                                                               "prompt_version", "system_sha256", "judge_score")}
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        written[family] = len(records)
        files.append(str(path))
    return {"written": written, "skipped": dict(skipped), "files": files}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Convert LLM call logs into chat-format fine-tuning JSONL, one file per call family.")
    p.add_argument("--log-dir", default=None, help="Call-log directory (default: LLM_CALL_LOG_DIR or data/llm_calls/)")
    p.add_argument("--out", required=True, help="Output directory for <call_family>.jsonl files")
    p.add_argument("--call", action="append", default=[], help="Only this call name or family (repeatable)")
    p.add_argument("--purpose", choices=["ingestion", "retrieval"], default=None)
    p.add_argument("--provider", default=None, help="Only calls served by this provider, e.g. CLAUDE (teacher data)")
    p.add_argument("--since", default=None, help="YYYY-MM-DD or ISO timestamp (UTC), inclusive")
    p.add_argument("--until", default=None, help="YYYY-MM-DD (inclusive, whole day) or ISO timestamp (UTC)")
    p.add_argument("--min-judge-score", type=float, default=None,
                   help="Keep only calls annotated with judge_score >= this (calls without a score are dropped)")
    p.add_argument("--dedup", choices=["prompt", "prompt_output", "none"], default="prompt",
                   help="prompt (default): one example per identical system+user; prompt_output: identical triples only")
    p.add_argument("--merge-system", action="store_true",
                   help="Prepend the system prompt to the user turn (for models whose chat template has no system role)")
    p.add_argument("--allow-unmasked", action="store_true",
                   help="Include records written with LLM_CALL_LOG_PII_MASK=0 (default: excluded)")
    p.add_argument("--with-metadata", action="store_true", help="Add a 'metadata' object to each line")
    args = p.parse_args(argv)

    log_dir = Path(args.log_dir) if args.log_dir else llm_call_log.log_dir()
    if not log_dir.is_dir():
        print(f"No call-log directory at {log_dir} - enable capture with LLM_CALL_LOG=1 first.", file=sys.stderr)
        return 1
    stats = export(
        log_dir, Path(args.out), calls=args.call, purpose=args.purpose, provider=args.provider,
        since=_parse_when(args.since) if args.since else None,
        until=_parse_when(args.until, end_of_day=True) if args.until else None,
        min_judge_score=args.min_judge_score, dedup=args.dedup, merge_system=args.merge_system,
        allow_unmasked=args.allow_unmasked, with_metadata=args.with_metadata,
    )
    for family, n in stats["written"].items():
        print(f"{family}: {n} examples")
    if stats["skipped"]:
        print("skipped: " + ", ".join(f"{k}={v}" for k, v in sorted(stats["skipped"].items())))
    if not stats["written"]:
        print("No examples matched the filters.", file=sys.stderr)
    print(f"Wrote {sum(stats['written'].values())} examples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
