"""
Chat history storage - PERSISTS SAVED ASK-PANEL CHATS IN NEO4J (per explicit
user request: "store it in current neo4j itself"), not a separate database.

Model: (:ChatSession {id, name, created_at, updated_at})
       -[:HAS_MESSAGE]->
       (:ChatMessage {id, session_id, seq, question, answer_json, created_at})

A "message" here is one full question+answer TURN from the Ask panel, not a
free-form chat line - `answer_json` is the entire /api/ask response payload
(JSON-serialized, since Neo4j node properties can't hold nested objects),
stored verbatim so re-opening an old turn can feed it straight back into the
SAME renderAskAnswer()/showAskGraph() the live path already uses, evidence
trail and graph included, with no separate replay logic needed.

Deliberately NOT threaded into /api/ask itself - that route stays stateless
and domain-agnostic. The webapp calls these functions AFTER a /api/ask
response comes back, purely additive: if saving history fails for any
reason, the user still sees their answer, they just don't get it saved.
"""
import json
import uuid
from datetime import datetime, timezone

from framework.core.infra import neo4j_client, step_logger

SESSION_LABEL = "ChatSession"
MESSAGE_LABEL = "ChatMessage"
HAS_MESSAGE_REL = "HAS_MESSAGE"

# How much of a fresh chat's first question to use as its auto-generated
# name (see AUTO-naming: "auto-name from first question" chosen over a
# generic "Chat 1"/"Chat 2" default) - trimmed at a word boundary so it
# doesn't cut off mid-word, with an ellipsis marker if truncated.
AUTO_NAME_MAX_CHARS = 60


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


def auto_name_from_question(question: str) -> str:
    q = (question or "").strip()
    if len(q) <= AUTO_NAME_MAX_CHARS:
        return q or "New chat"
    cut = q[:AUTO_NAME_MAX_CHARS]
    last_space = cut.rfind(" ")
    if last_space > 20:  # don't cut too aggressively short if the first "word" is long
        cut = cut[:last_space]
    return cut.rstrip() + "…"


def ensure_indexes():
    """Idempotent - safe to call on every webapp start, same pattern as
    every other engine's ensure_*_index() calls."""
    with neo4j_client.get_session() as session:
        neo4j_client.ensure_id_constraint(session, SESSION_LABEL)
        neo4j_client.ensure_id_constraint(session, MESSAGE_LABEL)
        neo4j_client.ensure_property_index(session, MESSAGE_LABEL, "session_id")


def create_session(question: str, answer: dict, name: str | None = None) -> dict:
    """Creates a new ChatSession (auto-named from `question` unless `name`
    is given) and immediately appends `question`/`answer` as its first
    message, in one call - mirrors how the UI actually uses this: a chat
    isn't created until the user's first question in it actually completes.
    Returns {"session": {...}, "message": {...}}."""
    session_id = _new_id()
    ts = _now_iso()
    session_name = (name or "").strip() or auto_name_from_question(question)

    with neo4j_client.get_session() as db:
        db.run(
            f"""
            CREATE (s:{SESSION_LABEL} {{id: $id, name: $name, created_at: $ts, updated_at: $ts}})
            """,
            id=session_id, name=session_name, ts=ts,
        )
    session_obj = {"id": session_id, "name": session_name, "created_at": ts, "updated_at": ts, "message_count": 0}
    message_obj = append_message(session_id, question, answer)
    session_obj["message_count"] = 1
    session_obj["updated_at"] = message_obj["created_at"]
    return {"session": session_obj, "message": message_obj}


def append_message(session_id: str, question: str, answer: dict) -> dict:
    message_id = _new_id()
    ts = _now_iso()
    answer_json = json.dumps(answer if answer is not None else {})

    with neo4j_client.get_session() as db:
        seq_row = db.run(
            f"MATCH (:{SESSION_LABEL} {{id: $sid}})-[:{HAS_MESSAGE_REL}]->(m:{MESSAGE_LABEL}) "
            f"RETURN count(m) AS n",
            sid=session_id,
        ).single()
        seq = (seq_row["n"] if seq_row else 0) + 1

        db.run(
            f"""
            MATCH (s:{SESSION_LABEL} {{id: $sid}})
            CREATE (m:{MESSAGE_LABEL} {{
                id: $mid, session_id: $sid, seq: $seq,
                question: $question, answer_json: $answer_json, created_at: $ts
            }})
            CREATE (s)-[:{HAS_MESSAGE_REL}]->(m)
            SET s.updated_at = $ts
            """,
            sid=session_id, mid=message_id, seq=seq,
            question=question or "", answer_json=answer_json, ts=ts,
        )
    return {"id": message_id, "seq": seq, "question": question or "", "answer": answer or {}, "created_at": ts}


def list_sessions() -> list[dict]:
    with neo4j_client.get_session() as db:
        rows = db.run(
            f"""
            MATCH (s:{SESSION_LABEL})
            OPTIONAL MATCH (s)-[:{HAS_MESSAGE_REL}]->(m:{MESSAGE_LABEL})
            WITH s, count(m) AS message_count
            RETURN s.id AS id, s.name AS name, s.created_at AS created_at,
                   s.updated_at AS updated_at, message_count
            ORDER BY s.updated_at DESC
            """
        )
        return [dict(r) for r in rows]


def get_session(session_id: str) -> dict | None:
    with neo4j_client.get_session() as db:
        s_row = db.run(
            f"MATCH (s:{SESSION_LABEL} {{id: $sid}}) RETURN s.id AS id, s.name AS name, "
            f"s.created_at AS created_at, s.updated_at AS updated_at",
            sid=session_id,
        ).single()
        if not s_row:
            return None
        session_obj = dict(s_row)

        m_rows = db.run(
            f"""
            MATCH (:{SESSION_LABEL} {{id: $sid}})-[:{HAS_MESSAGE_REL}]->(m:{MESSAGE_LABEL})
            RETURN m.id AS id, m.seq AS seq, m.question AS question,
                   m.answer_json AS answer_json, m.created_at AS created_at
            ORDER BY m.seq ASC
            """,
            sid=session_id,
        )
        messages = []
        for r in m_rows:
            d = dict(r)
            raw = d.pop("answer_json", "{}")
            try:
                d["answer"] = json.loads(raw) if raw else {}
            except (TypeError, json.JSONDecodeError):
                d["answer"] = {}
            messages.append(d)
        session_obj["messages"] = messages
        session_obj["message_count"] = len(messages)
        return session_obj


def rename_session(session_id: str, new_name: str) -> dict | None:
    new_name = (new_name or "").strip()
    if not new_name:
        return None
    ts = _now_iso()
    with neo4j_client.get_session() as db:
        row = db.run(
            f"""
            MATCH (s:{SESSION_LABEL} {{id: $sid}})
            SET s.name = $name, s.updated_at = $ts
            RETURN s.id AS id, s.name AS name, s.created_at AS created_at, s.updated_at AS updated_at
            """,
            sid=session_id, name=new_name, ts=ts,
        ).single()
        return dict(row) if row else None


def delete_session(session_id: str) -> bool:
    with neo4j_client.get_session() as db:
        result = db.run(
            f"""
            MATCH (s:{SESSION_LABEL} {{id: $sid}})
            OPTIONAL MATCH (s)-[:{HAS_MESSAGE_REL}]->(m:{MESSAGE_LABEL})
            DETACH DELETE s, m
            RETURN count(s) AS deleted
            """,
            sid=session_id,
        ).single()
        return bool(result and result["deleted"])
