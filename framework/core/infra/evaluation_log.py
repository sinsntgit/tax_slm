"""
2026-08-26: NEW. Evaluation feedback log - PERSISTS EVERY independent
evaluation the app runs (Neo4j-backed, same storage/pattern as
chat_history.py) so a user can review the full history of question/answer
scores and reviewer feedback later, from a dedicated page, instead of losing
it once they navigate away from the Ask screen. Added per explicit request:
"make sure you have all feedback available to see in a separate option
under the left panel... should open all feedback (question, feedback,
score) to view in a separate page" - and, per the follow-up clarification,
covers BOTH kinds of evaluation this app runs:
  - "manual"                a one-off /api/ask/evaluate click
  - "quality_checked_attempt"  one attempt inside the auto-evaluate-and-retry
                             loop (_run_quality_checked_ask_job in app.py) -
                             every attempt is logged, not just the final one,
                             so the full progression is reviewable here too
                             (this DUPLICATES what already lands in chat
                             history's quality_check.attempts for a *saved*
                             chat turn, but this log is the only place that
                             also captures evaluations that never got saved
                             to a chat, e.g. a manual Evaluate click, or a
                             quality-checked run whose chat-history save
                             failed).

Model: (:EvaluationRecord {id, question, answer_text, source, retrieval_mode,
        attempt_number, overall, verdict, dimensions_json, strengths_json,
        issues_json, reason, created_at})
Flat/standalone nodes (no relationships) - this is a pure append-only log for
human review, not something the retrieval/graph layer ever queries.

Deliberately NOT threaded into answer_evaluation.py itself (which stays a
pure, storage-agnostic scoring function) - callers (the /api/ask/evaluate
route and the quality-checked job) call record_evaluation() themselves,
right after they already have an `evaluation` dict, exactly like how
chat_history is called from the webapp layer rather than from retrieval
code. Best-effort/non-fatal by design: if logging fails, the evaluation the
user is looking at right now still renders fine - see try/except at each
call site in app.py.
"""
import json
import uuid
from datetime import datetime, timezone

from framework.core.infra import neo4j_client

RECORD_LABEL = "EvaluationRecord"

# How much of the answer text to store per record - full answers can be long
# (multi-thousand-char legal analyses); capping keeps the log table/page
# fast to load while still showing enough to be useful on the feedback page
# (the full answer is always still viewable from the chat/Ask screen itself
# for anything recent enough to matter).
MAX_STORED_ANSWER_CHARS = 6000


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


def ensure_indexes():
    """Idempotent - safe to call on every webapp start, same pattern as
    chat_history.ensure_indexes()."""
    with neo4j_client.get_session() as session:
        neo4j_client.ensure_id_constraint(session, RECORD_LABEL)
        neo4j_client.ensure_property_index(session, RECORD_LABEL, "created_at")
        neo4j_client.ensure_property_index(session, RECORD_LABEL, "source")


def record_evaluation(question: str, answer_text: str, evaluation: dict, *,
                       source: str, retrieval_mode: str | None = None,
                       attempt_number: int | None = None,
                       chat_id: str | None = None) -> dict:
    """Logs one evaluation run. `source` is "manual" (one-off Evaluate click)
    or "quality_checked_attempt" (one attempt inside the auto-retry loop).
    Safe to call even when `evaluation["available"]` is False (the evaluator
    call itself failed) - that's still worth a row (`overall` just stays
    null), it shows up on the feedback page as "evaluation unavailable"
    rather than silently vanishing."""
    record_id = _new_id()
    ts = _now_iso()
    answer_text = (answer_text or "")[:MAX_STORED_ANSWER_CHARS]
    evaluation = evaluation or {}

    with neo4j_client.get_session() as db:
        db.run(
            f"""
            CREATE (r:{RECORD_LABEL} {{
                id: $id, question: $question, answer_text: $answer_text,
                source: $source, retrieval_mode: $retrieval_mode,
                attempt_number: $attempt_number, chat_id: $chat_id,
                overall: $overall, verdict: $verdict,
                dimensions_json: $dimensions_json, strengths_json: $strengths_json,
                issues_json: $issues_json, reason: $reason,
                available: $available, created_at: $ts
            }})
            """,
            id=record_id, question=question or "", answer_text=answer_text,
            source=source, retrieval_mode=retrieval_mode, attempt_number=attempt_number,
            chat_id=chat_id,
            overall=evaluation.get("overall"), verdict=evaluation.get("verdict") or "",
            dimensions_json=json.dumps(evaluation.get("dimensions") or []),
            strengths_json=json.dumps(evaluation.get("strengths") or []),
            issues_json=json.dumps(evaluation.get("issues") or []),
            reason=evaluation.get("reason") or "",
            available=bool(evaluation.get("available")),
            ts=ts,
        )
    return {"id": record_id, "created_at": ts}


def _row_to_dict(r) -> dict:
    d = dict(r)
    for key in ("dimensions_json", "strengths_json", "issues_json"):
        out_key = key.replace("_json", "")
        raw = d.pop(key, None)
        try:
            d[out_key] = json.loads(raw) if raw else []
        except (TypeError, json.JSONDecodeError):
            d[out_key] = []
    return d


def list_evaluations(limit: int = 300) -> list[dict]:
    """Most-recent-first. `limit` bounds the feedback page's initial load -
    this is an append-only log with no pagination UI (yet), so a generous
    but finite cap keeps a long-running install's page load reasonable."""
    with neo4j_client.get_session() as db:
        rows = db.run(
            f"""
            MATCH (r:{RECORD_LABEL})
            RETURN r.id AS id, r.question AS question, r.answer_text AS answer_text,
                   r.source AS source, r.retrieval_mode AS retrieval_mode,
                   r.attempt_number AS attempt_number, r.chat_id AS chat_id,
                   r.overall AS overall, r.verdict AS verdict,
                   r.dimensions_json AS dimensions_json, r.strengths_json AS strengths_json,
                   r.issues_json AS issues_json, r.reason AS reason,
                   r.available AS available, r.created_at AS created_at
            ORDER BY r.created_at DESC
            LIMIT $limit
            """,
            limit=limit,
        )
        return [_row_to_dict(r) for r in rows]
