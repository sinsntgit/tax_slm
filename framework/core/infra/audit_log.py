"""
Neo4j-backed pipeline audit / data-lineage log - PROMOTED unchanged from
tax_kg_poc/scripts/audit_log.py. `AuditRun` is pipeline/operational metadata,
not a domain concept, so every domain pack's CLI commands share this same
audit trail in the same graph.
"""
import json
import socket
import uuid
from datetime import datetime, timezone

from framework.core.infra import neo4j_client, step_logger


def start_run(script: str, document_id: str | None = None, args: dict | None = None) -> str:
    run_id = str(uuid.uuid4())
    step_logger.log(f"[audit_log] Connecting to Neo4j to record run start "
                     f"(timeout={neo4j_client.NEO4J_CONNECT_TIMEOUT_SECONDS}s)...")
    try:
        with neo4j_client.get_session() as session:
            session.run(
                "CREATE CONSTRAINT IF NOT EXISTS FOR (r:AuditRun) REQUIRE r.run_id IS UNIQUE"
            )
            session.run(
                """
                MERGE (r:AuditRun {run_id: $run_id})
                SET r.script = $script,
                    r.document_id = $document_id,
                    r.host = $host,
                    r.started_at = $started_at,
                    r.status = 'running',
                    r.args = $args
                """,
                run_id=run_id, script=script, document_id=document_id,
                host=socket.gethostname(),
                started_at=datetime.now(timezone.utc).isoformat(),
                args=json.dumps(args or {}, default=str),
            )
        step_logger.log(f"[audit_log] Run {run_id} started ({script}, document_id={document_id}).")
    except Exception as e:
        step_logger.log(f"[audit_log] WARNING: could not record run start ({e}). Continuing without audit logging.")
    return run_id


def finish_run(run_id: str, status: str, details: dict | None = None):
    try:
        with neo4j_client.get_session() as session:
            session.run(
                """
                MATCH (r:AuditRun {run_id: $run_id})
                SET r.finished_at = $finished_at,
                    r.status = $status,
                    r.details = $details
                """,
                run_id=run_id,
                finished_at=datetime.now(timezone.utc).isoformat(),
                status=status,
                details=json.dumps(details or {}, default=str),
            )
        step_logger.log(f"[audit_log] Run {run_id} finished (status={status}).")
    except Exception as e:
        step_logger.log(f"[audit_log] WARNING: could not record run end ({e}).")
