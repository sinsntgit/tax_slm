"""
One-time backfill: populate sourceMdPath (see Semantic_Graph_Retrieval_
Architecture.md) onto CaseLaw/Circular/TaxTreaty/Section entities that were
ingested BEFORE that property existed (2026-08-28/29).

Pure graph read -> path resolution -> graph write. NO LLM call, NO
re-extraction - same "prefer a deterministic pass over a paid LLM re-run
when the graph already has what's needed" reasoning as
backfill_case_law_stance.py.

Path resolution differs by entity type because they store file provenance
differently:

  - CaseLaw / CBDTCircular / ExplanatoryCircular / ExtensionCircular /
    TaxTreaty: each node already carries `source_document`, the bare .md
    filename it was extracted from (e.g. extract_case_law.py's
    `"source_document": source_file`) - resolved as `--md-dir/<that filename>`.

  - Section (StatutoryProvision): does NOT carry a per-node source_document
    - extract_entities.py's attach_pipeline_metadata() only stamps
      `document_id` (the Act namespace, e.g. "ITA1961"), because every
      Section under one ingestion run shares the SAME source .md file (the
      whole Act). This script groups Sections by document_id and resolves
      ONE .md path per Act, either via an explicit --act-map
      ("ITA1961=1b52526e_income_tax_act_1961.md,ITA2025=...") or, if not
      given, by matching a 4-digit year found in document_id against
      filenames in --md-dir (printed either way during --dry-run so you can
      sanity-check it before writing).

Usage:
    python backfill_source_md_path.py --md-dir ../../../data/converted_md --dry-run
    python backfill_source_md_path.py --md-dir ../../../data/converted_md \\
        --act-map "ITA1961=1b52526e_income_tax_act_1961.md,ITA2025=14456eeb_Income_Tax_Act_2025_as_amended_by_FA_Act_2026.md"
"""
import argparse
import os
import re
import sys
from pathlib import Path

_this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_this_dir, "..", "..", ".."))

from framework.core.infra import neo4j_client, step_logger

# 2026-08-29: kept in sync with graph_loader.py's single-label-per-node
# scheme (see circular_entry_points.py's module docstring) - these are the
# ONLY labels sourceMdPath is defined on (schema.json's domain list).
FILE_PROVENANCE_LABELS = ("CaseLaw", "CBDTCircular", "ExplanatoryCircular", "ExtensionCircular", "TaxTreaty")

FETCH_BY_SOURCE_DOCUMENT = """
MATCH (n:{label})
WHERE coalesce(n.source_document, '') <> '' AND coalesce(n.sourceMdPath, '') = ''
RETURN elementId(n) AS db_id, n.source_document AS source_document,
       coalesce(n.full_citation, n.case_name, n.id) AS label_text
"""

FETCH_SECTIONS = """
MATCH (n:Section)
WHERE coalesce(n.sourceMdPath, '') = ''
RETURN elementId(n) AS db_id, n.document_id AS document_id
"""

WRITE_QUERY_TEMPLATE = """
UNWIND $rows AS row
MATCH (n:{label}) WHERE elementId(n) = row.db_id
SET n.sourceMdPath = row.sourceMdPath
"""


def _parse_act_map(raw: str | None) -> dict[str, str]:
    if not raw:
        return {}
    out = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        doc_id, _, filename = pair.partition("=")
        if not doc_id or not filename:
            raise SystemExit(f"--act-map entry {pair!r} is not in DOCUMENT_ID=filename.md form")
        out[doc_id.strip()] = filename.strip()
    return out


def _guess_act_md_file(document_id: str, md_dir: Path) -> str | None:
    """Year-substring heuristic fallback when --act-map doesn't cover a
    document_id - see module docstring. Returns a bare filename, or None if
    no confident single match was found (caller must then require --act-map
    for that document_id rather than guess wrong)."""
    years = re.findall(r"(19\d{2}|20\d{2})", document_id)
    if not years:
        return None
    year = years[0]
    candidates = [p.name for p in md_dir.glob("*.md") if year in p.name and
                  re.search(r"income.?tax.?act|\bact\b", p.name, re.IGNORECASE)]
    if len(candidates) == 1:
        return candidates[0]
    return None  # 0 or >1 matches - not confident enough to auto-pick


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--md-dir", type=Path, required=True,
                         help="Directory containing the converted .md files (e.g. data/converted_md).")
    parser.add_argument("--act-map", default=None,
                         help="Comma-separated DOCUMENT_ID=filename.md pairs for Section resolution "
                              "(see module docstring). Falls back to a year-substring guess if omitted.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change, write nothing.")
    args = parser.parse_args()

    md_dir = args.md_dir.resolve()
    if not md_dir.is_dir():
        raise SystemExit(f"--md-dir {md_dir} is not a directory")
    act_map = _parse_act_map(args.act_map)

    total_written = 0
    total_skipped = 0

    with neo4j_client.get_session() as session:
        # --- CaseLaw / Circular subtypes / TaxTreaty: per-node source_document ---
        for label in FILE_PROVENANCE_LABELS:
            records = [dict(r) for r in session.run(FETCH_BY_SOURCE_DOCUMENT.format(label=label))]
            if not records:
                continue
            step_logger.log(f"{label}: {len(records)} node(s) missing sourceMdPath.")
            rows = []
            for r in records:
                candidate = md_dir / r["source_document"]
                if not candidate.exists():
                    step_logger.log(f"  SKIP [{label}] {str(r['label_text'])[:60]!r} - "
                                     f"{candidate} does not exist")
                    total_skipped += 1
                    continue
                step_logger.log(f"  {str(r['label_text'])[:60]!r} -> {candidate}")
                rows.append({"db_id": r["db_id"], "sourceMdPath": str(candidate)})
            if rows and not args.dry_run:
                session.run(WRITE_QUERY_TEMPLATE.format(label=label), rows=rows)
            total_written += len(rows)

        # --- Section: shared one-file-per-Act resolution ---
        section_records = [dict(r) for r in session.run(FETCH_SECTIONS)]
        if section_records:
            by_doc_id: dict[str, list[str]] = {}
            for r in section_records:
                by_doc_id.setdefault(r["document_id"] or "", []).append(r["db_id"])
            step_logger.log(f"Section: {len(section_records)} node(s) missing sourceMdPath, "
                             f"across {len(by_doc_id)} document_id(s): {sorted(by_doc_id)}")

            rows = []
            for document_id, db_ids in by_doc_id.items():
                filename = act_map.get(document_id) or _guess_act_md_file(document_id, md_dir)
                if not filename:
                    step_logger.log(f"  SKIP document_id={document_id!r} ({len(db_ids)} Section node(s)) - "
                                     f"no --act-map entry and no confident filename guess. Re-run with "
                                     f"--act-map \"{document_id}=<the Act's .md filename>\".")
                    total_skipped += len(db_ids)
                    continue
                candidate = md_dir / filename
                if not candidate.exists():
                    step_logger.log(f"  SKIP document_id={document_id!r} - {candidate} does not exist")
                    total_skipped += len(db_ids)
                    continue
                step_logger.log(f"  document_id={document_id!r} ({len(db_ids)} Section node(s)) -> {candidate}")
                rows.extend({"db_id": db_id, "sourceMdPath": str(candidate)} for db_id in db_ids)
            if rows and not args.dry_run:
                session.run(WRITE_QUERY_TEMPLATE.format(label="Section"), rows=rows)
            total_written += len(rows)

    if args.dry_run:
        step_logger.log(f"--dry-run set: no changes written. Would set sourceMdPath on "
                         f"{total_written} node(s); {total_skipped} skipped (see SKIP lines above).")
    else:
        step_logger.log(f"Wrote sourceMdPath onto {total_written} node(s); {total_skipped} skipped "
                         f"(see SKIP lines above - fix the underlying issue and re-run, this script is "
                         f"safe to run repeatedly).")


if __name__ == "__main__":
    main()
