"""
One-time backfill: populate stance/stanceBasis/retrievalSummary/
retrievalKeywords (see Graph_First_Retrieval_Design.md Section 2, and
extract_case_law.py's derive_stance()/derive_retrieval_summary()/
derive_retrieval_keywords() - REUSED here, not reimplemented, so backfilled
and newly-ingested CaseLaw nodes are derived identically) onto CaseLaw nodes
that were ingested BEFORE these fields existed.

Pure graph read -> Python compose -> graph write. NO LLM call: every input
these derive_*() functions need (issue_outcomes/findings/keywords/sections/
legal_principles) is already sitting in the graph from the original
extraction - re-running extraction would just be re-deriving the same
inputs at LLM cost for no benefit. Same "prefer a deterministic pass over a
paid LLM re-run when the graph already has what's needed" reasoning as
classify_forum_level()/derive_jurisdiction_scope() themselves.

Usage:
    python backfill_case_law_stance.py            # writes changes
    python backfill_case_law_stance.py --dry-run   # prints what WOULD change, writes nothing
"""
import argparse
import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this_dir, "..", "..", ".."))
sys.path.insert(0, _this_dir)  # for `import extract_case_law` (same directory)

from framework.core.infra import neo4j_client, step_logger
import extract_case_law as excl


FETCH_QUERY = """
MATCH (case:CaseLaw)
OPTIONAL MATCH (case)-[:INTERPRETS {relation_type: 'raises_issue'}]->(issue:Issue)-[:HAS_CONCLUSION]->(concl:Conclusion)
WITH case, collect(DISTINCT {issue: issue.hasText, holding: concl.holding, winner: concl.winner}) AS raw_findings
OPTIONAL MATCH (case)-[interp:INTERPRETS]->(sp)
WHERE interp.relation_type IN ['interpreted', 'applied', 'referred'] AND interp.cited_as IS NOT NULL
WITH case, raw_findings, collect(DISTINCT interp.cited_as) AS cited_sections
OPTIONAL MATCH (case)-[:INSTANTIATES_CONCEPT]->(concept:LegalConcept)
WITH case, raw_findings, cited_sections, collect(DISTINCT concept.hasText) AS legal_principles
RETURN
    elementId(case) AS db_id,
    case.case_name AS case_name,
    case.full_citation AS full_citation,
    case.court AS court,
    case.final_outcome AS final_outcome,
    case.keywords AS keywords,
    case.legal_concepts AS legal_concepts,
    raw_findings,
    cited_sections,
    legal_principles
"""

WRITE_QUERY = """
UNWIND $rows AS row
MATCH (case:CaseLaw)
WHERE elementId(case) = row.db_id
SET case.stance = row.stance,
    case.stanceBasis = row.stanceBasis,
    case.retrievalSummary = row.retrievalSummary,
    case.retrievalKeywords = row.retrievalKeywords
"""


def _reconstruct_extraction_dict(record: dict) -> dict:
    """Rebuilds an extraction-shaped dict from graph data alone - just
    enough of the shape derive_stance()/derive_retrieval_summary()/
    derive_retrieval_keywords() actually read, nothing more."""
    findings = [
        {"issue": f.get("issue") or "", "holding": f.get("holding") or ""}
        for f in (record.get("raw_findings") or [])
        if f.get("issue") or f.get("holding")
    ]
    issue_outcomes = [
        {"issue": f.get("issue") or "", "winner": f.get("winner") or ""}
        for f in (record.get("raw_findings") or [])
        if f.get("winner")
    ]
    keywords = record.get("keywords") or record.get("legal_concepts") or []
    sections_interpreted = [{"number": s} for s in (record.get("cited_sections") or [])]
    return {
        "citation": record.get("full_citation") or record.get("case_name") or "",
        "court": record.get("court") or "",
        "final_outcome": record.get("final_outcome") or "",
        "keywords": keywords,
        "issue_outcomes": issue_outcomes,
        "findings": findings,
        "sections_interpreted": sections_interpreted,
        "sections_applied": [],
        "sections_referred": [],
        "statutes_cited": [],
        "legal_principles": record.get("legal_principles") or [],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print what would change, write nothing.")
    args = parser.parse_args()

    with neo4j_client.get_session() as session:
        records = [dict(r) for r in session.run(FETCH_QUERY)]

    step_logger.log(f"Found {len(records)} CaseLaw node(s) in the graph.")

    rows = []
    stance_counts: dict[str, int] = {}
    for record in records:
        extraction = _reconstruct_extraction_dict(record)
        stance, basis = excl.derive_stance(extraction)
        citation = extraction["citation"]
        summary = excl.derive_retrieval_summary(extraction, citation)
        keywords = excl.derive_retrieval_keywords(extraction)
        stance_counts[stance] = stance_counts.get(stance, 0) + 1
        rows.append({
            "db_id": record["db_id"],
            "stance": stance,
            "stanceBasis": basis,
            "retrievalSummary": summary,
            "retrievalKeywords": keywords,
        })
        if args.dry_run:
            step_logger.log(f"  [{stance:18s}] {citation[:70]:70s} | {basis}")

    step_logger.log("Stance breakdown:")
    for stance, count in sorted(stance_counts.items()):
        step_logger.log(f"  {stance}: {count}")
    if stance_counts.get("unknown"):
        step_logger.log(
            f"  NOTE: {stance_counts['unknown']} case(s) have no recognized issue_outcomes[].winner in the "
            f"graph - these are likely either stub nodes (referenced as a precedent but never ingested from "
            f"their own judgment) or cases ingested before issue_outcomes extraction existed. Re-ingesting "
            f"from the source PDF is the only way to populate stance for these, not this backfill."
        )

    if args.dry_run:
        step_logger.log("--dry-run set: no changes written.")
        return

    with neo4j_client.get_session() as session:
        session.run(WRITE_QUERY, rows=rows)
    step_logger.log(f"Wrote stance/stanceBasis/retrievalSummary/retrievalKeywords onto {len(rows)} CaseLaw node(s).")


if __name__ == "__main__":
    main()
