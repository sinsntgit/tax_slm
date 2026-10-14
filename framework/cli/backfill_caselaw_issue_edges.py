"""
One-off backfill: adds the CaseLaw -[:INTERPRETS {relation_type:
"raises_issue"}]-> Issue edge that domain_packs/tax/converters/
extract_case_law.py now creates for every newly-ingested case (see that
file's "issues / issue_outcomes / findings" section), but which earlier
ingestion runs never wrote - those Issue nodes exist in the graph with no
relationship at all connecting them back to the CaseLaw that raised them,
reachable only by string-matching the "_to_<case_id>" suffix baked into
their id.

This script is idempotent (MERGE, not CREATE) and safe to re-run: already-
linked pairs are left untouched, and running it again after a fresh
ingestion (which now writes the edge itself) is a harmless no-op for those.

Usage:
    python -m framework.cli.backfill_caselaw_issue_edges --domain tax
    python -m framework.cli.backfill_caselaw_issue_edges --domain tax --dry-run
"""
import argparse

from framework.cli.pack_loader import load_domain_pack
from framework.core.infra import neo4j_client, step_logger


def find_missing_links(session) -> list[dict]:
    """Every (CaseLaw, Issue) pair whose Issue.id ends with '_to_' + the
    CaseLaw's own id - i.e. the naming convention extract_case_law.py has
    always used to mint that Issue's id - that does NOT yet have an
    INTERPRETS edge between them in either direction."""
    rows = session.run(
        """
        MATCH (c:CaseLaw), (i:Issue)
        WHERE i.id STARTS WITH 'Issue_' AND i.id ENDS WITH ('_to_' + c.id)
          AND NOT (c)-[:INTERPRETS]->(i)
          AND NOT (i)-[:INTERPRETED_BY]->(c)
        RETURN c.id AS case_id, i.id AS issue_id
        """
    )
    return [dict(r) for r in rows]


def backfill(session, dry_run: bool) -> int:
    pairs = find_missing_links(session)
    step_logger.log(f"Found {len(pairs)} CaseLaw->Issue pair(s) missing the INTERPRETS edge.")
    if dry_run:
        for p in pairs[:20]:
            step_logger.log(f"  WOULD LINK: {p['case_id']} -[INTERPRETS]-> {p['issue_id']}")
        if len(pairs) > 20:
            step_logger.log(f"  ...and {len(pairs) - 20} more.")
        return 0
    if not pairs:
        return 0
    result = session.run(
        """
        MATCH (c:CaseLaw), (i:Issue)
        WHERE i.id STARTS WITH 'Issue_' AND i.id ENDS WITH ('_to_' + c.id)
          AND NOT (c)-[:INTERPRETS]->(i)
          AND NOT (i)-[:INTERPRETED_BY]->(c)
        MERGE (c)-[r:INTERPRETS {relation_type: 'raises_issue'}]->(i)
        RETURN count(r) AS created
        """
    )
    return result.single()["created"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Report what would be linked without writing anything.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    load_domain_pack(args.domain)  # validates --domain / env the same way every other CLI entry point does

    with step_logger.Step("Connecting to Neo4j"):
        neo4j_client.verify_connectivity()

    with neo4j_client.get_session() as session:
        with step_logger.Step("Backfilling CaseLaw -> Issue INTERPRETS edges"):
            created = backfill(session, args.dry_run)

    if args.dry_run:
        step_logger.log("Dry run - nothing written. Re-run without --dry-run to apply.")
    else:
        step_logger.log(f"Done - created {created} new INTERPRETS edge(s).")


if __name__ == "__main__":
    main()
