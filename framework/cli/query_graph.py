"""
Generic post-ingestion sanity-check CLI - domain-parameterized replacement
for tax_kg_poc/scripts/04_query_graph.py. Thin wrapper around
framework/core/ingest/validate.py's run_sanity_checks(), which is the same
function the webapp's /api/validate route calls - one implementation, two
front-ends.

Usage:
    python -m framework.cli.query_graph --domain tax
"""
import argparse

from framework.cli.pack_loader import load_domain_pack
from framework.core.infra import neo4j_client, step_logger
from framework.core.ingest.validate import run_sanity_checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    domain_config, _prompt_pack = load_domain_pack(args.domain)

    with step_logger.Step("Connecting to Neo4j"):
        neo4j_client.verify_connectivity()

    with neo4j_client.get_session() as session:
        with step_logger.Step("Running sanity checks"):
            report = run_sanity_checks(session, domain_config)

    step_logger.log("1. Total node count by label:")
    for row in report["node_counts"]:
        step_logger.log(f"  {row['label']}: {row['count']}")

    step_logger.log("2. Total relationship count by type:")
    for row in report["relationship_counts"]:
        step_logger.log(f"  {row['type']}: {row['count']}")

    if report["sample_relationship_checks"]:
        step_logger.log(f"3. Sample structural-relationship checks ({len(report['sample_relationship_checks'])} configured):")
        for check in report["sample_relationship_checks"]:
            step_logger.log(f"  {check['description']}:")
            for s in check["samples"]:
                step_logger.log(f"    {s['source']} -> {s['target']}")
    else:
        step_logger.log("3. No sample_relationship_checks configured for this domain - skipping.")

    step_logger.log(f"4. Orphan check ({report['orphan_check_labels_checked']}/{report['orphan_check_labels_total']} labels checked"
                     f"{', stopped early - hit the cap' if report['orphan_check_truncated'] else ''}):")
    if report["orphans"]:
        for o in report["orphans"]:
            step_logger.log(f"  ORPHAN: {o['label']} - {o['id']}")
    else:
        step_logger.log("  None found - every node has at least one relationship.")

    step_logger.log("5. Direct citation lookup (Chunk.full_citation):")
    cl = report["citation_lookup"]
    if not cl:
        step_logger.log("  No Chunk nodes with full_citation found - run the domain's extraction + "
                         "ingest_graph steps first.")
    else:
        step_logger.log(f"  Looked up citation {cl['example_citation']!r} directly:")
        for c in cl["chunks"]:
            step_logger.log(f"    {c}")
        if cl["refers_to"]:
            step_logger.log(f"  {cl['example_citation']!r} REFERS_TO: {cl['refers_to']}")


if __name__ == "__main__":
    main()
