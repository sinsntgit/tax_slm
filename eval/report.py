"""
Aggregation (summary.json) and the human-readable report.md for one run.

2026-10-14: NEW. Aggregates are always recomputed from the per-item metric
rows (items.jsonl), never accumulated during the run, so a re-score or a
different gate threshold (compare.py re-aggregates with eval/gates.json's
settings) gives numbers consistent with the per-item rows shown beside them.

Denominator rules (stated in eval/README.md, keep in sync):
  - error_rate: errored items / all items.
  - answer-quality metrics (citations, authorities, format, must_not_claim,
    judge): over items that returned an answer (status ok). Errors are
    reported separately, never scored as zeros.
  - latency: over every item that has a measured latency (ok AND failed
    calls - a timeout counts at its timeout value, so slow failures can't
    flatter the latency gate).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

LATENCY_LIMIT_S = 60.0  # proposed release gate; the gate itself is configured in eval/gates.json


def percentile(values: list[float], q: float) -> float | None:
    """Linear-interpolation percentile (same as numpy's default), q in 0-100."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * q / 100.0
    lo, hi = math.floor(pos), math.ceil(pos)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def _mean(values):
    vals = [v for v in values if v is not None]
    return (sum(vals) / len(vals)) if vals else None


def _rate(num, den):
    return (num / den) if den else None


def latency_stats(values: list[float], limit_s: float = LATENCY_LIMIT_S) -> dict:
    vals = [v for v in values if v is not None]
    return {
        "n": len(vals),
        "mean": _mean(vals),
        "p50": percentile(vals, 50),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
        "max": max(vals) if vals else None,
        f"share_under_{int(limit_s)}s": _rate(sum(1 for v in vals if v < limit_s), len(vals)),
    }


def aggregate(rows: list[dict], *, authority_item_threshold: float = 0.85) -> dict:
    n = len(rows)
    ok = [r for r in rows if r.get("status") == "ok"]
    errors = [r for r in rows if r.get("status") != "ok"]

    cit = [r["citations"] for r in ok]
    fabricated_total = sum(c["fabricated_count"] for c in cit)
    items_with_fab = sum(1 for c in cit if c["fabricated_count"] > 0)

    auth = [r["authorities"] for r in ok if r["authorities"]["required"] > 0]
    req_total = sum(a["required"] for a in auth)
    meets = [a["recall_answer"] >= authority_item_threshold for a in auth]

    fmt = [r["format"] for r in ok if r["format"].get("applicable")]
    mnc_total = sum(len(r["must_not_claim"]) for r in ok)
    kp = [r["key_points"]["coverage"] for r in ok if r["key_points"]["coverage"] is not None]

    lat_by_mode = defaultdict(list)
    for r in rows:
        if r.get("latency_s") is not None:
            lat_by_mode[r.get("answer_mode") or "unknown"].append(r["latency_s"])

    judge_vals = defaultdict(list)
    for r in ok:
        for k, v in (r.get("judge") or {}).items():
            judge_vals[k].append(v)

    return {
        "counts": {
            "items": n, "ok": len(ok), "error": len(errors),
            "gold_verified": sum(1 for r in rows if r.get("verified")),
            "gold_unverified": sum(1 for r in rows if not r.get("verified")),
        },
        "error_rate": _rate(len(errors), n),
        "error_types": dict(Counter(((r.get("error") or {}).get("type") or "missing") for r in errors)),
        "cache_hits": sum(1 for r in ok if r.get("from_cache")),
        "no_context_answers": sum(1 for r in ok if r.get("no_context")),
        "citations": {
            "fabricated_total": fabricated_total,
            "items_with_fabricated": items_with_fab,
            "fabricated_item_rate": _rate(items_with_fab, len(ok)),
            "citation_ok_rate": _rate(sum(1 for c in cit if c["citation_ok"]), len(cit)),
            "citations_total": sum(c["count"] for c in cit),
            "page_unverified_total": sum(len(c["page_unverified"]) for c in cit),
            "referred_sources_unverified_total": sum(len(c["referred_sources_unverified"]) for c in cit),
            "unsupported_case_mentions_total": sum(len(c["unsupported_case_mentions"]) for c in cit),
        },
        "authority": {
            "threshold": authority_item_threshold,
            "items_scored": len(auth),
            "required_total": req_total,
            "recall_answer_micro": _rate(sum(a["cited_in_answer"] for a in auth), req_total),
            "recall_answer_macro": _mean([a["recall_answer"] for a in auth]),
            "recall_sources_micro": _rate(sum(a["found_in_sources"] for a in auth), req_total),
            "recall_sources_macro": _mean([a["recall_sources"] for a in auth]),
            "items_meeting_threshold": sum(meets),
            "items_meeting_threshold_rate": _rate(sum(meets), len(meets)),
        },
        "format": {
            "applicable": len(fmt),
            "compliant": sum(1 for f in fmt if f["compliant"]),
            "compliance_rate": _rate(sum(1 for f in fmt if f["compliant"]), len(fmt)),
            "litigation_risk_rate": _rate(sum(1 for f in fmt if f.get("strict_litigation_risk_bullet")), len(fmt)),
            "all_referred_categories_rate": _rate(
                sum(1 for f in fmt if f.get("compliant") and not f.get("strict_referred_categories_missing")), len(fmt)),
        },
        "must_not_claim": {
            "violations_total": mnc_total,
            "items_with_violations": sum(1 for r in ok if r["must_not_claim"]),
        },
        "key_points": {"items_scored": len(kp), "coverage_mean": _mean(kp)},
        "latency": {
            "all": latency_stats([r.get("latency_s") for r in rows]),
            **{mode: latency_stats(vals) for mode, vals in sorted(lat_by_mode.items())},
        },
        "judge": {k: {"mean": _mean(v), "n": len(v)} for k, v in sorted(judge_vals.items())},
    }


def build_summary(rows: list[dict], manifest: dict, *, authority_item_threshold: float = 0.85) -> dict:
    by_category = defaultdict(list)
    by_mode = defaultdict(list)
    for r in rows:
        by_category[r["category"]].append(r)
        by_mode[r.get("answer_mode") or "unknown"].append(r)
    return {
        "run": {k: manifest.get(k) for k in ("name", "target", "endpoint", "retrieval_mode", "answer_mode",
                                              "judge", "gold_path", "gold_fingerprint", "started_at",
                                              "finished_at", "server_settings", "harness_version")},
        "overall": aggregate(rows, authority_item_threshold=authority_item_threshold),
        "by_category": {c: aggregate(v, authority_item_threshold=authority_item_threshold)
                        for c, v in sorted(by_category.items())},
        "by_answer_mode": {m: aggregate(v, authority_item_threshold=authority_item_threshold)
                           for m, v in sorted(by_mode.items())},
    }


# ---------------------------------------------------------------- markdown

def fmt_num(v, digits: int = 2, pct: bool = False) -> str:
    if v is None:
        return "n/a"
    if pct:
        return f"{v * 100:.1f}%"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def _headline_rows(o: dict) -> list[tuple[str, str]]:
    return [
        ("Items (ok / error)", f"{o['counts']['ok']} / {o['counts']['error']}"),
        ("Error rate", fmt_num(o["error_rate"], pct=True)),
        ("Fabricated citations (total / items)",
         f"{o['citations']['fabricated_total']} / {o['citations']['items_with_fabricated']}"),
        ("citation_ok rate", fmt_num(o["citations"]["citation_ok_rate"], pct=True)),
        ("Authority recall in answer (micro / macro)",
         f"{fmt_num(o['authority']['recall_answer_micro'], pct=True)} / {fmt_num(o['authority']['recall_answer_macro'], pct=True)}"),
        (f"Items with >= {int(o['authority']['threshold'] * 100)}% required authorities cited",
         f"{o['authority']['items_meeting_threshold']}/{o['authority']['items_scored']} "
         f"({fmt_num(o['authority']['items_meeting_threshold_rate'], pct=True)})"),
        ("Authority recall in retrieved sources (micro)", fmt_num(o["authority"]["recall_sources_micro"], pct=True)),
        ("a/b/c/d format compliance (research mode)",
         f"{o['format']['compliant']}/{o['format']['applicable']} ({fmt_num(o['format']['compliance_rate'], pct=True)})"),
        ("must_not_claim violations (total / items)",
         f"{o['must_not_claim']['violations_total']} / {o['must_not_claim']['items_with_violations']}"),
        ("Key-point coverage (mean)", fmt_num(o["key_points"]["coverage_mean"], pct=True)),
        ("Latency p50 / p95 / max (s)",
         f"{fmt_num(o['latency']['all']['p50'], 1)} / {fmt_num(o['latency']['all']['p95'], 1)} / {fmt_num(o['latency']['all']['max'], 1)}"),
        ("Cache hits (must be 0 for a valid run)", str(o["cache_hits"])),
    ]


def render_report(summary: dict, rows: list[dict]) -> str:
    run, o = summary["run"], summary["overall"]
    lines = [f"# Eval run: {run.get('name')}", ""]
    lines.append(f"- Target: `{run.get('target')}` endpoint `{run.get('endpoint')}`; "
                 f"retrieval_mode `{run.get('retrieval_mode') or 'per item'}`, answer_mode `{run.get('answer_mode') or 'per item'}`; "
                 f"judge: {'yes (/api/ask/evaluate)' if run.get('judge') else 'internal answer_quality only'}")
    lines.append(f"- Gold: `{run.get('gold_path')}` (fingerprint `{run.get('gold_fingerprint')}`), "
                 f"{o['counts']['gold_verified']} verified / {o['counts']['gold_unverified']} unverified items")
    lines.append(f"- Started {run.get('started_at')} - finished {run.get('finished_at')}")
    if o["counts"]["gold_unverified"]:
        lines += ["", "> **PROVISIONAL:** this run includes gold items not yet signed off by a tax professional "
                      "(verified=false). Treat authority/key-point/must-not-claim numbers as indicative only."]
    if o["cache_hits"]:
        lines += ["", f"> **INVALID FOR COMPARISON:** {o['cache_hits']} answer(s) were served from the response cache "
                      "(from_cache=true) - latency is not real and the answer may come from a different model. "
                      "Flush Redis and re-run those items (see eval/README.md)."]

    lines += ["", "## Headline", "", "| Metric | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in _headline_rows(o)]

    lines += ["", "## Latency by answer mode", "", "| Mode | n | p50 | p90 | p95 | max | share < 60s |", "|---|---|---|---|---|---|---|"]
    for mode, st in o["latency"].items():
        lines.append(f"| {mode} | {st['n']} | {fmt_num(st['p50'], 1)} | {fmt_num(st['p90'], 1)} | {fmt_num(st['p95'], 1)} | "
                     f"{fmt_num(st['max'], 1)} | {fmt_num(st.get('share_under_60s'), pct=True)} |")

    if o["judge"]:
        lines += ["", "## Judge scores (means over answered items)", "",
                  "answer_quality.* = in-pipeline judge (0-1); evaluation.* / general_knowledge.* = /api/ask/evaluate (0-10).",
                  "", "| Dimension | mean | n |", "|---|---|---|"]
        lines += [f"| {k} | {fmt_num(v['mean'], 3)} | {v['n']} |" for k, v in o["judge"].items()]

    lines += ["", "## By category", "",
              "| Category | items | errors | fabricated | auth recall (micro) | items >= threshold | format | mnc viol. | p95 s | judge overall |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for cat, a in summary["by_category"].items():
        jo = a["judge"].get("answer_quality.overall", {}).get("mean")
        lines.append(
            f"| {cat} | {a['counts']['items']} | {a['counts']['error']} | {a['citations']['fabricated_total']} | "
            f"{fmt_num(a['authority']['recall_answer_micro'], pct=True)} | "
            f"{a['authority']['items_meeting_threshold']}/{a['authority']['items_scored']} | "
            f"{a['format']['compliant']}/{a['format']['applicable']} | {a['must_not_claim']['violations_total']} | "
            f"{fmt_num(a['latency']['all']['p95'], 1)} | {fmt_num(jo, 2)} |")

    lines += ["", "## Per item", "",
              "| id | category | status | latency s | auth (answer/sources) | fabricated | format | mnc | judge overall |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"| {r['id']} | {r['category']} | **{(r.get('error') or {}).get('type', r['status'])}** | "
                         f"{fmt_num(r.get('latency_s'), 1)} | - | - | - | - | - |")
            continue
        a = r["authorities"]
        f = r["format"]
        fcell = "n/a" if not f.get("applicable") else ("ok" if f["compliant"] else "MISSING " + ",".join(f["headings_missing"]) if f["headings_missing"] else "ORDER")
        lines.append(
            f"| {r['id']} | {r['category']} | ok{' (cache)' if r.get('from_cache') else ''} | {fmt_num(r.get('latency_s'), 1)} | "
            f"{a['cited_in_answer']}/{a['required']} , {a['found_in_sources']}/{a['required']} | "
            f"{r['citations']['fabricated_count']} | {fcell} | {len(r['must_not_claim'])} | "
            f"{fmt_num(r['judge'].get('answer_quality.overall'), 2)} |")

    problems = [r for r in rows if r["status"] == "ok" and (r["citations"]["fabricated"] or r["must_not_claim"]
                                                         or r["authorities"]["missing_in_answer"]
                                                         or r["citations"]["page_unverified"]
                                                         or r["citations"]["unsupported_case_mentions"])]
    if problems:
        lines += ["", "## Item details (review these)", ""]
        for r in problems:
            lines.append(f"### {r['id']} ({r['category']})")
            c = r["citations"]
            if c["fabricated"]:
                lines.append(f"- **Fabricated citations:** " + "; ".join(f"`{x}`" for x in c["fabricated"]))
            if r["must_not_claim"]:
                for v in r["must_not_claim"]:
                    lines.append(f"- **must_not_claim** \"{v['claim']}\": ...{v['snippet']}...")
            if r["authorities"]["missing_in_answer"]:
                lines.append(f"- Required authorities not cited: {', '.join(r['authorities']['missing_in_answer'])}"
                             + (f" (not in retrieved sources either: {', '.join(r['authorities']['missing_in_sources'])})"
                                if r['authorities']['missing_in_sources'] else ""))
            if c["page_unverified"]:
                lines.append(f"- Page numbers not among retrieved chunk pages: " + "; ".join(c["page_unverified"]))
            if c["unsupported_case_mentions"]:
                lines.append(f"- Case names not found in retrieved material (heuristic): " + "; ".join(c["unsupported_case_mentions"]))
            lines.append("")

    errs = [r for r in rows if r["status"] != "ok"]
    if errs:
        lines += ["", "## Errors", ""]
        for r in errs:
            e = r.get("error") or {}
            lines.append(f"- `{r['id']}`: {e.get('type')} {('HTTP ' + str(e.get('http_status'))) if e.get('http_status') else ''} "
                         f"- {str(e.get('message', ''))[:300]} (attempts: {r.get('attempts')})")
    lines.append("")
    return "\n".join(lines)
