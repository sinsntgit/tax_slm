# Evaluation harness (Phase 0)

This harness is how every model, prompt and retrieval change gets judged. It is
also how we will show the fine-tuned Tax SLM matching the frontier-model
baseline before release. It does three things:

1. It asks a gold question set through the **real HTTP API** of the running app
   (`/api/ask`, or optionally `/api/ask/fast`) and saves every raw response.
2. It scores each answer **deterministically and offline**: authority recall,
   fabricated citations, a/b/c/d format compliance, must-not-claim screens and
   latency. It also records the judge scores that the API itself produces.
3. It compares a candidate run with a baseline run and gives a **PASS / FAIL /
   INCOMPLETE** verdict against the release gates in `eval/gates.json`.

It uses only the Python standard library plus the repo's own
`framework/core/retrieve/grounding.py`. There are no extra dependencies, no
LLM calls of its own, and it never reads `.env`.

> **The starter gold set is a DRAFT.** All 40 items in `eval/gold/` were drafted
> by Claude from the test fixtures, and every one is `verified=false,
> verified_by="claude-draft"`. The opinions they come from are **fictitious**.
> Until tax professionals have reviewed and signed off the items, every report
> and comparison is labelled **PROVISIONAL**, and a release decision cannot rest
> on it.

---

## 1. Running a baseline on the owner's machine

Prerequisites: Docker Desktop running, `.env` filled in from `.env.example`, the
corpus already ingested, and the venv active.

```bat
cd D:\tax_kg_poc_revised
docker compose up -d
python -m framework.webapp.app --domain tax          :: leave running (http://localhost:5000)
```

In a second terminal:

```bat
:: 1. Flush the response cache. Its key ignores the model, so a cached answer
::    could be another model's answer, with fake latency. Do this before EVERY run.
docker exec tax_kg_revised_redis redis-cli FLUSHDB

:: 2. Check the gold set
python -m eval.goldset eval/gold/

:: 3. Smoke test (3 items)
python -m eval.run --gold eval/gold/ --target http://localhost:5000 --retrieval-mode hybrid --answer-mode research --out eval/runs/smoke --limit 3

:: 4. The baseline run (frontier model, set via USE_MODEL / USE_MODEL_RETRIEVAL in .env)
python -m eval.run --gold eval/gold/ --target http://localhost:5000 --retrieval-mode hybrid --answer-mode research --judge --out eval/runs/2026-10-xx_frontier_hybrid_research --notes "USE_MODEL_RETRIEVAL=CLAUDE <model>, answer_generation v6"
```

The results land in `eval/runs/<name>/report.md`. That folder is git-ignored
because it holds full answers and corpus text.

**Resuming.** If a run is interrupted (Ctrl-C, laptop sleeps, provider outage),
rerun **the same command**. Items already answered OK are skipped and errored
items are retried. Adding `--judge` later runs only the judge pass. If `--out`
already holds a run with a different target, endpoint or mode, the runner
refuses to mix them; use a new `--out`, or `--force` if you really mean it.

**Comparing.** Later, run the candidate (for example the SLM-backed app) the same
way into another folder, then:

```bat
python -m eval.compare eval/runs/<candidate> eval/runs/<baseline>
```

The **first** argument is the candidate under test and the **second** is the
reference baseline. The command prints a markdown report and writes
`compare.json` and `compare.md` to `<candidate>/compare_vs_<baseline>/`. Its exit
code is `0` for PASS, `1` for FAIL and `3` for INCOMPLETE, so CI can use it.

Useful options for `eval.run`:

| Option | Default | Meaning |
|---|---|---|
| `--endpoint ask\|fast` | `ask` | `/api/ask` (research path) or `/api/ask/fast` (evaluate-and-retry loop, no NL2Cypher) |
| `--retrieval-mode` / `--answer-mode` | per gold item | When given, overrides every item (use this for comparability) |
| `--judge` | off | Also POSTs each answer to `/api/ask/evaluate` as a separate call, so its time never pollutes latency |
| `--concurrency` | 2 | Parallel requests |
| `--timeout` | 300 | Per-request seconds. A timeout is recorded at that latency |
| `--retries` / `--backoff` | 2 / 2s | Retries on connection errors and HTTP 429/5xx (exponential backoff) |
| `--retry-on-timeout` | off | Timeouts are not retried by default because they are slow |
| `--ids`, `--category`, `--limit` | | Subsets, for example `--category treaty` |
| `--rerun` | off | Re-ask items already answered OK |

To re-score a saved run after a metric fix or a gold correction, with no API
calls: `python -m eval.score eval/runs/<name> [--gold eval/gold/]`.

## 2. Run folder layout

```
manifest.json         target, endpoint, modes, gold fingerprint, server settings snapshot, notes
gold_snapshot.jsonl   the exact gold items asked (the run is self-contained)
responses/<id>.json   raw record: request, status, attempts, latency, full API JSON, judge result
items.jsonl           per-item metrics
summary.json          aggregates: overall, by_category, by_answer_mode
report.md             human-readable report, including a "review these" section per item
```

## 3. What each metric means

Unless noted otherwise, answer metrics are computed over items that **returned
an answer**. Errors are counted in `error_rate` and are never scored as zeros.
Before any scoring, the server-appended `DISCLAIMER` block and the "(Note: ...
should have cited ...)" caution line are stripped.

| Metric | Definition |
|---|---|
| **error_rate** | Items with no usable answer divided by all items. Error types: `timeout`, `connection_error`, `http_error` (with status), `malformed_response` (2xx but not JSON, or no `answer`). |
| **cache_hits** | Answers with `from_cache=true`. Any non-zero value makes a run invalid for comparison (see section 1). |
| **authority recall (answer)** | For each gold item: required authorities found **in the answer text** divided by the number required. Micro is the ratio of totals; macro is the mean per item. |
| **items meeting authority threshold** | Share of items whose answer recall is at least 85% (`settings.authority_item_threshold` in gates.json). The release gate needs this to be 100%. |
| **authority recall (sources)** | The same test run against **everything retrieval returned**: document names, citations, chunk previews, full chunk text (the runner always sends `include_full_chunks=true`), parent-section text and graph nodes. If an authority is in the sources but not the answer, generation dropped it. If it is in neither, retrieval missed it. |
| **fabricated citations** | Citations in the answer that name a document or chunk **not among the retrieved sources**. Recognised forms are the v6 style `<doc name>.txt/.pdf - Page N`, `[Document: ...]` and `chunk_id=N`. A source cited several times counts once. The gate is **zero in the whole run**. If nothing was retrieved, every citation counts as fabricated. |
| **page_unverified** | The cited document is real, but none of its retrieved chunks carry the cited page. This is a review signal only and is not counted as fabricated, because parent-section context can legitimately span pages. |
| **referred_sources_unverified** | The output of `grounding.check_referred_sources()` (the server's own "d. Referred Sources" check), re-run on the stripped answer. |
| **unsupported_case_mentions** | A heuristic. It lists "X v. Y" case names whose distinctive party word appears nowhere in the retrieved material. It catches invented authorities cited without a file-style citation. It is informational and never gated. |
| **citation_ok** | The server's `grounding.citation_check.citation_ok` (`verify_citations`). If that is missing, the value is recomputed locally with the same function. |
| **format compliance** | Research mode only (the `answer_generation` v6 Section 6 skeleton). Headings `a. Executive Summary`, `b. Detailed Analysis`, `c. Conclusion` and `d. Referred Sources` must all be at line start, in that order, with markdown decoration tolerated. The report also covers the litigation-risk bullet and the six Referred Sources category labels as "strict" detail, but these are not part of the gate: v6 Section 7 allows dropping the litigation-risk bullet for simple questions. Lookup and opinion modes are N/A. |
| **must_not_claim** | Regex screens for wrong claims. With `negation_guard` (the default), a match is ignored if one of the 6 preceding words in the same sentence is a negation ("does **not** constitute royalty"). An attributed claim ("the Revenue argued it is royalty") can still fire. Every hit carries its snippet for human review, and the gate is advisory (non-blocking). |
| **key-point coverage** | For key points that have `match_any` regexes: the share matched. It is indicative only; the judge scores substance. |
| **latency** | Client wall-clock time of the final attempt, covering **all** items including failures (a timeout counts at its timeout value). Reported as p50, p90, p95 and max, plus the share under 60s, overall and per answer mode. |
| **judge: `answer_quality.*`** | The in-pipeline judge (`answer_quality.judge_answer`, scored 0–1) that every `/api/ask` call already runs. Its dimensions are overall, groundedness, completeness, specificity, statutory_fidelity, holding_discipline and authority_hierarchy. |
| **judge: `evaluation.*` / `general_knowledge.overall`** | With `--judge`: the `/api/ask/evaluate` grounded scorecard (answer_evaluation v1, 0–10, which re-retrieves in the same retrieval_mode) and its no-context general-knowledge scorecard. |

An unavailable judge (`available:false`) contributes nothing; it is never counted as 0.

## 4. Release gates (`eval/gates.json`)

The thresholds are configuration, not code. In the current proposed gates, the
candidate must meet each of these:

| Gate | Rule |
|---|---|
| `zero_fabricated_citations` | `citations.fabricated_total <= 0` |
| `judge_quality_vs_baseline` | For **each** listed judge dimension, the candidate mean divided by the baseline mean must be `>= 0.90`. This is computed over **paired** items (answered in both runs), with at least 10 of them. It includes `evaluation.overall`, so **both** runs need `--judge`. |
| `authority_recall_per_question` | Every question has at least 85% of its required authorities cited |
| `format_compliance` | At least 98% of research-mode answers have the a/b/c/d skeleton |
| `research_latency` | Research-mode p95 below 60s |
| `run_validity_no_cache_hits`, `run_validity_error_rate` | The run is valid: 0 cache hits and at most 2% errors |
| `must_not_claim` | Advisory (`blocking: false`) |
| `cypher_validity`, `extraction_accuracy` | `enabled: false` until Phase 1, so they show as NOT_MEASURED |

The verdict is decided as follows. If any enabled blocking gate fails, the
verdict is **FAIL**. If none fail but one cannot be measured (judge not run,
no research items, too few paired items), the verdict is **INCOMPLETE**, never a
silent PASS. Otherwise the verdict is **PASS**. A verdict on an unverified gold
set is also marked **PROVISIONAL**. Compare also warns when the two runs used
different gold sets, endpoints or modes.

## 5. Gold set: format, adding and verifying items

Gold items live in `eval/gold/*.jsonl`, one JSON object per line. Lines starting
with `//` are comments. The schema is in `eval/gold_item.schema.json`.
`python -m eval.goldset` validates it and lists **every** problem with
`file:line [id]`.

JSONL was chosen over YAML for three reasons: there is no YAML dependency in the
repo, each line gives a clean per-question git diff and blame for sign-off, and
appending an item can never break another item.

```json
{"id": "op18-software-licence-ireland", "question": "...", "category": "treaty",
 "answer_mode": "research", "retrieval_mode": "hybrid",
 "required_authorities": [{"id": "s195", "kind": "section", "label": "Section 195",
                           "aliases": ["Section 195"], "section": "195"},
                          {"id": "engineering-analysis", "kind": "case", "label": "Engineering Analysis ... (SC)",
                           "aliases": ["Engineering Analysis Centre of Excellence", "Engineering Analysis"]}],
 "optional_authorities": [],
 "expected_key_points": [{"point": "...", "source": "OP-18 s.6", "match_any": ["not\\s+royalt"]}],
 "must_not_claim": [{"claim": "The payments constitute royalty", "patterns": ["(?:constitutes?|is|are)\\s+royalt(?:y|ies)\\b"]}],
 "sources": ["tests/fixtures/opinions/OP-18_software_royalty_engineering_analysis.md"],
 "verified_by": "claude-draft", "verified": false, "notes": "..."}
```

- **category** is one of `lookup | research | for_against | change_in_law | treaty | computation | opinion`.
  **answer_mode** is one of `lookup | research | opinion`.
- **Authority matching** is case-insensitive and normalises dashes, quotes,
  underscores and `80-IAC`/`80IAC`. It is section-number aware: `"section":
  "195"` also matches `s.195`, `sec. 195` and `u/s 195`, and never matches
  `195A` or `1951`. `9(1)(vi)` never matches `9(1)(vii)`, but `Section 195`
  does match `Section 195(1)`. Add an alias for each spelling that should count.
- **required vs optional authorities**: required ones count toward recall and
  the gate. Optional ones are only reported (`optional_cited`). For the starter
  set, *required* means the frontmatter authorities that the opinion's
  Issue/Analysis/Conclusion actually relies on, and *optional* means the
  frontmatter-only ones.
- Every key point carries a `source` (fixture and paragraph or section) so a
  reviewer can trace it.
  `tests/test_eval_goldset.py` checks that every required authority and every
  key-point regex of the starter set really occurs in the cited fixture. The one
  derived figure (6% × INR 1.5 crore = INR 9 lakh) is whitelisted and flagged in
  that item's notes.

**Verification workflow (tax professional):**

1. Read the question, `expected_key_points`, `required_authorities` and
   `must_not_claim` against the primary sources, not just the fixture. The
   opinions are fictitious and their conclusions may not be correct law.
2. Correct anything wrong and record it in `notes`.
3. Set `"verified": true`, set `"verified_by": "<name, qualification>"` and add
   `"verified_at": "YYYY-MM-DD"`. The loader rejects `verified=true` while
   `verified_by` is still `claude-draft`.
4. Commit one reviewer's sign-offs per commit, so the history shows who
   approved what.

Real client-style questions (anonymised) should be added over time. The starter
set is mostly synthetic scenarios over a tiny fixture corpus. For a baseline on
the real corpus, the owner should make sure the corpus contains the authorities
the gold set expects, or add items grounded in what is actually ingested.
