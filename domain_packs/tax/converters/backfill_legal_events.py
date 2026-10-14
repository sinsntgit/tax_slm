"""
One-time backfill: reify OVERRULES (CaseLaw -> CaseLaw) and SUPERSEDES /
WITHDRAWS (DelegatedLegislation subtype -> same, in practice Circulars) edges
that are ALREADY sitting in the graph as dated LegalEvent nodes - see
chunking.build_legal_event()'s docstring for what a LegalEvent is and why.
Covers everything ingested BEFORE 2026-08-31, when extract_case_law.py and
extract_circular.py started creating these events themselves at extraction
time (see those two files' OVERRULES / SUPERSEDES / WITHDRAWS loops).

Pure graph read -> Python compose -> graph write. NO LLM call: the event
date is always the INSTRUMENT node's own already-ingested date (a case's
judgment_date, a circular's date_issued/effective_from) and the
subject/instrument ids are already sitting on the existing relationship's
two endpoints - same "prefer a deterministic pass over a paid LLM re-run
when the graph already has what's needed" reasoning as
backfill_case_law_stance.py / backfill_source_md_path.py.

Idempotent / safe to run repeatedly: chunking.build_legal_event() builds a
deterministic id (LegalEvent_<type>_<instrument_id>_<subject_id>), and
GraphIngestionEngine.load_entities()/load_relationships() MERGE by id - so
re-running this after extract_case_law.py/extract_circular.py have ALSO
started creating these events going forward just MERGEs onto the same nodes/
edges instead of duplicating anything.

Usage:
    python backfill_legal_events.py --dry-run   # prints what WOULD be created, writes nothing
    python backfill_legal_events.py             # writes changes
    python backfill_legal_events.py --verbose
"""
import argparse
import os
import sys

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this_dir, "..", "..", ".."))
sys.path.insert(0, _this_dir)  # for `import chunking` (same directory)

from framework.core.infra import neo4j_client, step_logger
from framework.core.ingest.graph_loader import GraphIngestionEngine
import chunking

# (Cypher relationship type, LegalEvent eventType, ordered list of date
# properties to try on the INSTRUMENT node - first non-empty wins). CaseLaw's
# own date lives in judgment_date; a DelegatedLegislation subtype's (in
# practice CBDTCircular/ExplanatoryCircular/ExtensionCircular) lives in
# date_issued, falling back to effective_from if date_issued was never
# captured - same fallback order extract_circular.py's own event-creation
# call effectively uses (date_issued first).
EVENT_SOURCES = [
    ("OVERRULES", "overruling", ["judgment_date", "effective_from"]),
    ("SUPERSEDES", "supersession", ["date_issued", "effective_from", "judgment_date"]),
    ("WITHDRAWS", "withdrawal", ["date_issued", "effective_from", "judgment_date"]),
]

FETCH_QUERY_TEMPLATE = """
MATCH (instrument)-[:{rel_type}]->(subject)
RETURN instrument.id AS instrument_id,
       coalesce(instrument.full_citation, instrument.case_name, instrument.id) AS instrument_label,
       {date_coalesce} AS instrument_date,
       subject.id AS subject_id,
       coalesce(subject.full_citation, subject.case_name, subject.id) AS subject_label
"""


def _date_coalesce_expr(date_props: list[str]) -> str:
    parts = ", ".join(f"instrument.{p}" for p in date_props) + ", ''"
    return f"coalesce({parts})"


def collect_legal_events(session) -> tuple[list[dict], list[dict], dict[str, int]]:
    """Returns (entities, relationships, counts_by_rel_type) - ready for
    GraphIngestionEngine.load_entities()/load_relationships()."""
    entities: list[dict] = []
    relationships: list[dict] = []
    counts: dict[str, int] = {}

    for rel_type, event_type, date_props in EVENT_SOURCES:
        query = FETCH_QUERY_TEMPLATE.format(rel_type=rel_type, date_coalesce=_date_coalesce_expr(date_props))
        rows = [dict(r) for r in session.run(query)]
        counts[rel_type] = len(rows)
        step_logger.log(f"{rel_type}: {len(rows)} existing edge(s) found in the graph.")
        for row in rows:
            if not row["instrument_id"] or not row["subject_id"]:
                continue  # shouldn't happen (both come from a real matched edge), guard anyway
            event_entity, event_rels = chunking.build_legal_event(
                event_type=event_type, event_date=row["instrument_date"] or "",
                instrument_id=row["instrument_id"], instrument_label=row["instrument_label"] or row["instrument_id"],
                subject_id=row["subject_id"], subject_label=row["subject_label"] or row["subject_id"],
                confidence=0.75,  # slightly below extraction-time 0.8 - reified from a pre-existing edge, not the original extraction call itself
            )
            if not row["instrument_date"]:
                step_logger.vlog(f"  NOTE: no date found on instrument {row['instrument_id']!r} for this "
                                  f"{event_type} event (checked {date_props}) - LegalEvent created without eventDate.")
            entities.append(event_entity)
            relationships.extend(event_rels)

    return entities, relationships, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be created, write nothing.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    with neo4j_client.get_session() as session:
        with step_logger.Step("Backfilling LegalEvent nodes from existing OVERRULES/SUPERSEDES/WITHDRAWS edges"):
            entities, relationships, counts = collect_legal_events(session)

        if not entities:
            step_logger.log("Nothing to do - no OVERRULES/SUPERSEDES/WITHDRAWS edges found in the graph "
                             "(has case law / circular ingestion run yet?).")
            return

        step_logger.log(f"Would create/merge {len(entities)} LegalEvent node(s) "
                         f"({', '.join(f'{k}={v}' for k, v in counts.items())}) and "
                         f"{len(relationships)} EVENT_SUBJECT/EVENT_INSTRUMENT edge(s).")

        if args.dry_run:
            step_logger.log("--dry-run set: not writing to Neo4j. Sample of what would be created:")
            for e in entities[:10]:
                step_logger.log(f"  LegalEvent {e['id']} - eventType={e['properties']['eventType']!r}, "
                                 f"eventDate={e['properties'].get('eventDate', '(unset)')!r}")
            if len(entities) > 10:
                step_logger.log(f"  ...and {len(entities) - 10} more.")
            return

        engine = GraphIngestionEngine(domain_config=None)  # load_entities()/load_relationships() don't touch domain_config
        engine.load_entities(session, entities, mode="base")
        engine.load_relationships(session, relationships)
        step_logger.log(f"Done - {len(entities)} LegalEvent node(s) and {len(relationships)} "
                         f"EVENT_SUBJECT/EVENT_INSTRUMENT edge(s) written to Neo4j.")


if __name__ == "__main__":
    main()
