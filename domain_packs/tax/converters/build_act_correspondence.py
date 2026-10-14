"""
Populates the ontology's CORRESPONDS_TO relationship (StatutoryProvision <->
StatutoryProvision, declared in schema.json since early in this project but
never populated by any converter until now) between an OLD Act and a NEW Act
that have both already been ingested into Neo4j under DIFFERENT id-namespaces
- see chunking.generate_entity_id()'s docstring (2026-08-27) for why two Acts
need distinct namespaces at all.

WHY THIS EXISTS (see Ontology_Legal_Review_2026-08-27.md for the full
analysis): every judgment in the currently-ingested case-law corpus was
decided under the Income-tax Act, 1961, but the Act actually ingested as
"the Act" is the brand-new Income-tax Act, 2025. A section NUMBER means
something different under each - e.g. old-Act Section 90 (DTAA/treaty
override, the section AT ISSUE in Union of India v. Azadi Bachao Andolan)
was renumbered to new-Act Section 159; new-Act Section 90 is an unrelated
capital-gains definitional provision. extract_case_law.py now resolves a
case's citations against the OLD Act's namespace (correct - see its
DEFAULT_CITED_ACT_NAMESPACE), so a case's INTERPRETS edge lands on the RIGHT
old-Act provision. But a user asking "what does the CURRENT Act say about
the provision Azadi Bachao Andolan interpreted" still has no graph path from
old-Act Section 90 to new-Act Section 159 without this correspondence -
that's what this script builds.

No official government concordance/comparative table was found in either
ingested Act PDF (checked via pdftotext + grep for "comparative table" /
"corresponding provision" - only generic transitional-clause language, no
lookup table) - see the legal review. So this is LLM-assisted, comparing
REAL provision text already sitting in the graph (not re-parsing PDFs),
Section-level only (not subsection-level - a full concordance at that
granularity is a much bigger undertaking than this corpus needs), and
scoped ONLY to sections actually cited by the ingested case law (a few
dozen, not the whole ~500-section Act) - see the module docstring's
"Tier 1/Tier 2" approach below.

Usage:
    python build_act_correspondence.py --old-namespace ITA1961 --new-namespace ITA2025
    python build_act_correspondence.py --old-namespace ITA1961 --new-namespace ITA2025 --dry-run
    python build_act_correspondence.py --old-namespace ITA1961 --new-namespace ITA2025 \\
        --review-out correspondence_review.json   # human-reviewable record of every mapping made/skipped
"""
import argparse
import json
import re
from pathlib import Path

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))

from framework.core.infra import llm_client, neo4j_client, step_logger
from framework.core.ingest.graph_loader import GraphIngestionEngine

SCRIPT_DIR = Path(__file__).resolve().parent

TIER1_SYSTEM_PROMPT = (
    "You compare two statutory provisions from two different (but related) "
    "Acts and decide whether they cover the SAME subject matter - i.e. "
    "whether a court's holding interpreting one would be relevant precedent "
    "for interpreting the other. Minor wording/numbering differences don't "
    "matter; SUBJECT MATTER is what matters. Respond with exactly one line: "
    "either 'MATCH' or 'NO_MATCH', optionally followed by ' - ' and a short "
    "(<20 word) reason."
)

TIER2_SYSTEM_PROMPT = (
    "You are given the full text of one section from an OLD Act, and a list "
    "of section numbers + short headings from a NEW Act that replaced it. "
    "Identify which NEW Act section (if any) now covers the SAME subject "
    "matter as the OLD section - i.e. where a court's holding on the OLD "
    "section would now be relevant precedent. Respond with exactly one "
    "line: either the new section number alone (e.g. '159') or 'NONE' if "
    "nothing in the list plausibly corresponds."
)


def _truncate(text: str, max_chars: int = 1200) -> str:
    text = text or ""
    return text if len(text) <= max_chars else text[:max_chars] + " …[truncated]"


def get_cited_old_section_numbers(session, old_namespace: str) -> list[str]:
    """Every DISTINCT top-level section number that ingested case law's
    INTERPRETS edges point at (directly or via a SubSection/Clause/SubClause
    child), under the OLD Act's namespace. Deliberately section-level only -
    see module docstring for why."""
    result = session.run(
        """
        MATCH (:CaseLaw)-[:INTERPRETS]->(t)
        WHERE t.id STARTS WITH $prefix AND t.section_no IS NOT NULL
        RETURN DISTINCT t.section_no AS section_no
        """,
        prefix=f"{old_namespace}__",
    )
    return sorted({r["section_no"] for r in result if r["section_no"]})


def get_section_node(session, namespace: str, section_no: str) -> dict | None:
    result = session.run(
        "MATCH (n:Section {id: $id}) RETURN n.section_no AS section_no, "
        "n.hasText AS hasText, n.marginal_heading AS marginal_heading, "
        "n.full_citation AS full_citation LIMIT 1",
        id=f"{namespace}__Section_{section_no}",
    )
    record = result.single()
    return dict(record) if record else None


def get_all_new_section_headings(session, new_namespace: str) -> list[dict]:
    result = session.run(
        "MATCH (n:Section) WHERE n.id STARTS WITH $prefix AND n.section_no IS NOT NULL "
        "RETURN n.section_no AS section_no, n.marginal_heading AS marginal_heading",
        prefix=f"{new_namespace}__",
    )
    return [dict(r) for r in result]


def tier1_same_number_check(old_section: dict, new_section: dict) -> tuple[bool, str]:
    user_content = (
        f"OLD Act section text (heading: {old_section.get('marginal_heading', '')}):\n"
        f"{_truncate(old_section.get('hasText', ''))}\n\n"
        f"NEW Act section with the SAME number (heading: {new_section.get('marginal_heading', '')}):\n"
        f"{_truncate(new_section.get('hasText', ''))}"
    )
    raw = llm_client.complete(system=TIER1_SYSTEM_PROMPT, user_content=user_content,
                               max_tokens=60, name="act_correspondence_tier1", purpose="ingestion").strip()
    is_match = raw.upper().startswith("MATCH")
    return is_match, raw


def tier2_reindex_search(old_section: dict, new_headings: list[dict]) -> tuple[str | None, str]:
    heading_lines = "\n".join(
        f"{h['section_no']}: {h.get('marginal_heading') or '(no heading)'}" for h in new_headings
    )
    user_content = (
        f"OLD Act section {old_section.get('full_citation', '')} "
        f"(heading: {old_section.get('marginal_heading', '')}):\n"
        f"{_truncate(old_section.get('hasText', ''), 2000)}\n\n"
        f"NEW Act section index (number: heading):\n{heading_lines}"
    )
    raw = llm_client.complete(system=TIER2_SYSTEM_PROMPT, user_content=user_content,
                               max_tokens=30, name="act_correspondence_tier2", purpose="ingestion").strip()
    m = re.match(r'^\s*([0-9]+[A-Z]{0,3})\s*$', raw, re.IGNORECASE)
    return (m.group(1) if m else None), raw


def build_correspondence(session, old_namespace: str, new_namespace: str) -> tuple[list[dict], list[dict]]:
    """Returns (relationships, review_records). relationships is ready for
    GraphIngestionEngine.load_relationships(); review_records is a full
    audit trail (matched, unmatched, reasoning) for human review - written
    out via --review-out regardless of --dry-run."""
    cited = get_cited_old_section_numbers(session, old_namespace)
    step_logger.log(f"{len(cited)} distinct old-Act section(s) actually cited by ingested case law: "
                     f"{', '.join(cited) if cited else '(none found - has case law been ingested yet?)'}")

    relationships: list[dict] = []
    review: list[dict] = []

    for i, section_no in enumerate(cited, start=1):
        step_logger.log(f"  [{i}/{len(cited)}] old-Act Section {section_no}...")
        old_node = get_section_node(session, old_namespace, section_no)
        if not old_node:
            step_logger.log(f"    SKIP - old-Act Section {section_no} not found under namespace "
                             f"'{old_namespace}' (has the old Act actually been ingested?)")
            review.append({"old_section": section_no, "status": "old_section_not_found"})
            continue

        old_id = f"{old_namespace}__Section_{section_no}"
        new_id_same_number = f"{new_namespace}__Section_{section_no}"
        new_node_same_number = get_section_node(session, new_namespace, section_no)

        if new_node_same_number:
            is_match, reason = tier1_same_number_check(old_node, new_node_same_number)
            if is_match:
                step_logger.log(f"    MATCH (same number) - {reason}")
                relationships.append({
                    "type": "CORRESPONDS_TO", "source": old_id, "target": new_id_same_number,
                    "properties": {"basis": "llm_text_comparison_same_number", "confidence": 0.85,
                                   "notes": reason},
                })
                relationships.append({
                    "type": "CORRESPONDS_TO", "source": new_id_same_number, "target": old_id,
                    "properties": {"basis": "llm_text_comparison_same_number", "confidence": 0.85,
                                   "notes": reason},
                })
                review.append({"old_section": section_no, "new_section": section_no,
                                "status": "matched_same_number", "reason": reason})
                continue
            step_logger.log(f"    NO_MATCH at same number ({reason}) - trying reindex search...")
        else:
            step_logger.log(f"    new-Act has no Section {section_no} at all - trying reindex search...")

        new_headings = get_all_new_section_headings(session, new_namespace)
        if not new_headings:
            step_logger.log(f"    SKIP - new Act has no sections under namespace '{new_namespace}' at all "
                             f"(has it been ingested?)")
            review.append({"old_section": section_no, "status": "new_act_not_ingested"})
            continue

        matched_number, raw = tier2_reindex_search(old_node, new_headings)
        if matched_number:
            new_id = f"{new_namespace}__Section_{matched_number}"
            step_logger.log(f"    MATCH (reindexed) - old Section {section_no} -> new Section {matched_number}")
            relationships.append({
                "type": "CORRESPONDS_TO", "source": old_id, "target": new_id,
                "properties": {"basis": "llm_text_comparison_reindexed", "confidence": 0.7, "notes": raw},
            })
            relationships.append({
                "type": "CORRESPONDS_TO", "source": new_id, "target": old_id,
                "properties": {"basis": "llm_text_comparison_reindexed", "confidence": 0.7, "notes": raw},
            })
            review.append({"old_section": section_no, "new_section": matched_number,
                            "status": "matched_reindexed", "reason": raw})
        else:
            step_logger.log(f"    NO MATCH FOUND for old Section {section_no} anywhere in the new Act "
                             f"({raw}) - flagged for manual review, no edge created.")
            review.append({"old_section": section_no, "status": "no_match_found", "reason": raw})

    return relationships, review


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--old-namespace", required=True, help="e.g. ITA1961 - the Act case law was decided under.")
    parser.add_argument("--new-namespace", required=True, help="e.g. ITA2025 - the currently-in-force Act.")
    parser.add_argument("--dry-run", action="store_true", help="Compute + log/print the mapping, write nothing to Neo4j.")
    parser.add_argument("--review-out", default=None,
                         help="Path to write the full audit trail (matched/unmatched/reasoning) as JSON. "
                              "Defaults to correspondence_review_<old>_<new>.json alongside this script.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    review_out = Path(args.review_out) if args.review_out else \
        SCRIPT_DIR / f"correspondence_review_{args.old_namespace}_{args.new_namespace}.json"

    with neo4j_client.get_session() as session:
        with step_logger.Step(f"Building {args.old_namespace} <-> {args.new_namespace} CORRESPONDS_TO mapping"):
            relationships, review = build_correspondence(session, args.old_namespace, args.new_namespace)

        with open(review_out, "w", encoding="utf-8") as f:
            json.dump(review, f, indent=2, ensure_ascii=False)
        step_logger.log(f"Wrote full audit trail ({len(review)} record(s)) to {review_out}")

        matched = sum(1 for r in review if r["status"].startswith("matched"))
        step_logger.log(f"Summary: {matched} section(s) mapped, {len(review) - matched} unmatched/skipped "
                         f"(see {review_out} for detail) - {len(relationships)} CORRESPONDS_TO edge(s) "
                         f"(bidirectional, so 2x the mapped-section count).")

        if args.dry_run:
            step_logger.log("--dry-run: not writing to Neo4j. Relationships that WOULD be created:")
            for rel in relationships:
                step_logger.log(f"  {rel['source']} -[CORRESPONDS_TO]-> {rel['target']} "
                                 f"({rel['properties']['basis']}, confidence={rel['properties']['confidence']})")
            return

        if not relationships:
            step_logger.log("Nothing to write - no matches found.")
            return

        engine = GraphIngestionEngine(domain_config=None)  # load_relationships() doesn't touch domain_config
        engine.load_relationships(session, relationships)
        step_logger.log("Done - CORRESPONDS_TO edges written to Neo4j.")


if __name__ == "__main__":
    main()
