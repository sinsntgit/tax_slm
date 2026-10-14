"""
Evaluation harness for the tax research app / Tax SLM (Phase 0).

2026-10-14: NEW. Measures the REAL product over HTTP (/api/ask, /api/ask/fast,
/api/ask/evaluate) against a gold question set, scores each answer offline
(authority recall, fabricated citations, a/b/c/d format compliance,
must-not-claim screens, latency, judge scores), and compares two runs
(candidate vs baseline) against the release gates in eval/gates.json.
See eval/README.md for how to run it and what every metric means.

Deliberately stdlib-only (urllib, concurrent.futures, json) so it runs on the
owner's Windows machine and in the offline cloud sandbox with no extra
dependency beyond what the repo already has.
"""
