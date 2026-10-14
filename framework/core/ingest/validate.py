"""
Post-ingestion sanity-check engine - refactored out of
tax_kg_poc/scripts/04_query_graph.py (and framework/cli/query_graph.py's
first port of it) into a plain function that RETURNS structured data instead
of only printing via step_logger, so both the CLI and the webapp's
/api/validate route can share one implementation.
"""
import os
import time

from neo4j import Query
from neo4j.exceptions import ClientError

from framework.core.infra import neo4j_client

STATEMENT_TIMEOUT_SECONDS = int(os.environ.get("NEO4J_STATEMENT_TIMEOUT_SECONDS", "20"))
MAX_ORPHANS = 20


def _run(session, cypher: str, **params):
    return list(session.run(Query(cypher, timeout=STATEMENT_TIMEOUT_SECONDS), **params))


def run_sanity_checks(session, domain_config) -> dict:
    """Returns:
    {
      "node_counts": [{"label": str, "count": int}, ...],
      "relationship_counts": [{"type": str, "count": int}, ...],
      "sample_relationship_checks": [
          {"description": str, "source_label": str, "rel_type": str, "target_label": str,
           "samples": [{"source": str, "target": str}, ...]}
      ],
      "orphans": [{"label": str, "id": str}, ...],
      "orphan_check_truncated": bool,
      "orphan_check_labels_checked": int,
      "citation_lookup": {"example_citation": str, "chunk": {...}, "refers_to": [str, ...]} | None,
    }
    """
    report: dict = {}

    node_rows = _run(session, "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS cnt ORDER BY cnt DESC")
    report["node_counts"] = [{"label": r["label"], "count": r["cnt"]} for r in node_rows]

    rel_rows = _run(session, "MATCH ()-[r]->() RETURN type(r) AS rel_type, count(*) AS cnt ORDER BY cnt DESC")
    report["relationship_counts"] = [{"type": r["rel_type"], "count": r["cnt"]} for r in rel_rows]

    checks = []
    for source_label, rel_type, target_label, description in (domain_config.sample_relationship_checks or []):
        source_label_safe = neo4j_client.safe_identifier(source_label)
        rel_type_safe = neo4j_client.safe_identifier(rel_type)
        target_label_safe = neo4j_client.safe_identifier(target_label)
        rows = _run(
            session,
            f"MATCH (s:{source_label_safe})-[:{rel_type_safe}]->(t:{target_label_safe}) "
            f"RETURN s.id AS source, t.id AS target LIMIT 10",
        )
        checks.append({
            "description": description, "source_label": source_label, "rel_type": rel_type,
            "target_label": target_label,
            "samples": [{"source": r["source"], "target": r["target"]} for r in rows],
        })
    report["sample_relationship_checks"] = checks

    labels = [r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")]
    found = []
    truncated = False
    labels_checked = 0
    for label in labels:
        if len(found) >= MAX_ORPHANS:
            truncated = True
            break
        labels_checked += 1
        label_safe = neo4j_client.safe_identifier(label)
        try:
            rows = _run(
                session,
                f"MATCH (n:{label_safe}) WHERE NOT (n)--() RETURN n.id AS id LIMIT {MAX_ORPHANS - len(found)}",
            )
            found.extend({"label": label, "id": r["id"]} for r in rows)
        except ClientError as e:
            if "timeout" not in str(e).lower():
                raise
    report["orphans"] = found
    report["orphan_check_truncated"] = truncated
    report["orphan_check_labels_checked"] = labels_checked
    report["orphan_check_labels_total"] = len(labels)

    sample = _run(session, "MATCH (c:Chunk) WHERE c.full_citation IS NOT NULL RETURN c.full_citation AS citation LIMIT 1")
    if not sample:
        report["citation_lookup"] = None
    else:
        example_citation = sample[0]["citation"]
        rows = _run(
            session,
            """
            MATCH (c:Chunk {full_citation: $citation})
            RETURN c.chunk_id AS chunk_id, c.chunk_type AS chunk_type, c.section AS section,
                   c.subsection AS subsection, c.clause AS clause
            """,
            citation=example_citation,
        )
        refs = _run(
            session,
            "MATCH (n {full_citation: $citation})-[:REFERS_TO]->(t) RETURN t.id AS target LIMIT 5",
            citation=example_citation,
        )
        report["citation_lookup"] = {
            "example_citation": example_citation,
            "chunks": [dict(r) for r in rows],
            "refers_to": [r["target"] for r in refs],
        }

    return report
