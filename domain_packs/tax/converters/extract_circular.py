"""
CBDT Circular structured extraction - the circular-specific counterpart to
11_extract_case_law.py (judgments) and 02_extract_entities.py (the Act):
a ONE-CALL LLM extraction pass over a converted CBDT circular/order (Markdown
or plain text from pdf_to_md_generic.py), mapped onto the ontology's
CBDTCircular / ExplanatoryCircular / ExtensionCircular classes (all under
DelegatedLegislation - see schema.json), with IMPLEMENTS/CLARIFIES edges to
whichever Act sections the circular references and SUPERSEDES/WITHDRAWS edges
to any earlier circular it names.

WHY A DEDICATED SCRIPT RATHER THAN extract_generic.py: extract_generic.py
already CAN ingest a circular (it tags DocumentSection chunks against the
full ontology class list via its enrichment pass), but it never produces a
proper typed CBDTCircular node with the circular's own number/date/subject/
F.No. as first-class properties, never classifies Explanatory vs Extension,
and never resolves section citations into real edges onto the Act graph -
exactly the gap a dedicated Act/case-law-style script closes, matching the
specificity those two document types already get.

WHY ONE LLM CALL (not chunking.py-style deterministic parsing, and not
extract_generic.py's paragraph+enrichment split): every circular inspected
while building this script (data/rawdata/circulars/*.pdf) is short (1-3
pages) with NO stable citation grammar the way the Act has ("#### Section N."
headings) - each is a free-form CBDT letter with a Subject line, numbered
paragraphs, and a signature block, closer in shape to a short judgment than
to the Act. A single extraction call over the whole (short) text is both
cheaper and more reliable than trying to force a fixed regex grammar onto
prose that varies letter-to-letter.

WHAT THIS SCRIPT DOES:
  1. Reads one (or a folder of) circular .md/.txt file(s).
  2. Sends the full text through llm_client.complete() asking for circular
     number/year/date/F.No./subject/issuing authority, a classification
     (explanatory / extension / general), the Act sections it references
     (with a implements-vs-clarifies judgment call per reference), any
     earlier circular(s) it supersedes or withdraws, and a short applicability
     summary + keyword list.
  3. Maps that JSON onto CBDTCircular (or its ExplanatoryCircular/
     ExtensionCircular subtype) + IMPLEMENTS/CLARIFIES edges (resolved onto
     the SAME Act node ids chunking.py/extract_entities.py already use, via
     chunking.generate_entity_id - exactly the pattern extract_case_law.py's
     resolve_section_citation() established) + SUPERSEDES/WITHDRAWS edges to
     stub CBDTCircular nodes for any earlier circular referenced by name/number
     only (same idempotent-stub pattern extract_case_law.py uses for
     precedents).
  4. Produces embeddable Chunk records via extract_generic.py's own
     split_into_paragraph_chunks() (reused directly, not reimplemented) so a
     circular's full text is vector-searchable exactly like every other
     document type.
  5. Reuses extract_entities.py's attach_pipeline_metadata() / compute_
     content_hash() / build_datasource_entity_and_links() UNCHANGED, so
     document_id/document_name/source_type/authority_level/jurisdiction/
     file_type/DataSource-provenance work identically across every converter
     in this project - no new plumbing invented here.

ACT-NAMESPACE RESOLUTION: a circular can reference sections of EITHER the
Income-tax Act, 1961 or the Income-tax Act, 2025 (the sample corpus this
script was built against has both - Circular 3/2026 references Schedule V of
"the Income-tax Act, 2025"). Each extracted section reference carries the
Act name the LLM read directly off the text; --act-1961-namespace/--act-2025-
namespace map those names to the same short --document-id values used when
that Act was itself ingested (defaults ITA1961/ITA2025, matching this
project's established convention - see ingest_act.py). A reference to an Act
name that doesn't match either pattern is left unresolved and flagged in
ambiguities rather than guessed.

KNOWN LIMITATIONS (explicit, not silently approximated):
  - Some circulars/orders in the wild are hand-scanned and heavily OCR-
    garbled (see the real circular-2-2026-pdf.pdf sample - "lndia" for
    "India", "cp---l" garbage in the letterhead); the LLM is asked to read
    past this, but a badly-garbled subject/number is possible and worth a
    human skim on first use with a new source.
  - implements-vs-clarifies is a judgment call the LLM makes per reference,
    not a rule; both edge types point at the same target node either way, so
    a wrong choice only affects which relationship label is used, not
    whether the citation resolves at all.
  - "General"-classified circulars stay CBDTCircular (the ontology's own
    least-specific concrete Circular subtype for this case) rather than
    inventing a third subtype the ontology doesn't declare.

Usage:
    python extract_circular.py --md ../data/circular_md/circular-2-2026.md \\
        --out ../data/circular_graph/circular-2-2026_graph.json \\
        --document-id circular_2_2026 --document-name "CBDT Circular No. 2/2026"

    # Every circular in a folder:
    python extract_circular.py --dir ../data/circular_md --out-dir ../data/circular_graph
"""
import argparse
import hashlib
import json
import os as _os
import re
import sys as _sys
from datetime import datetime, timezone
from pathlib import Path

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))

import chunking  # same directory (domain_packs/tax/converters/)
import extract_entities  # same directory - reused for attach_pipeline_metadata()/
                          # compute_content_hash()/build_datasource_entity_and_links()
import extract_generic  # same directory - reused for split_into_paragraph_chunks()
from framework.core.infra import llm_client, step_logger
from framework.core.retrieve import observability

SCRIPT_DIR = Path(__file__).resolve().parent

# --- The extraction prompt --------------------------------------------------

EXTRACTION_INSTRUCTIONS = """You are a tax law analyst extracting structured data from an official CBDT
(Central Board of Direct Taxes, India) circular, order, or instruction.

CRITICAL RULES:
- Read the FULL text provided, including any letterhead/signature block for the circular number,
  F.No., and date.
- The text may be OCR-extracted from a scanned PDF and contain minor garbling (e.g. "lndia" for
  "India") - read past obvious OCR noise, don't let it block extraction.
- "circular_subtype" MUST be exactly one of: "explanatory" (clarifies/interprets how a provision
  should be read or applied - a notification, condonation-of-delay order, or clarificatory circular
  counts as explanatory unless it is purely a deadline extension), "extension" (its SOLE substantive
  effect is extending a statutory/procedural deadline under section 119 or similar), "general"
  (neither of the above cleanly applies).
- "sections_referenced": every Income-tax Act section/schedule this circular cites, with which Act
  it belongs to read EXACTLY as written (e.g. "Income-tax Act, 1961" vs "Income-tax Act, 2025" -
  these are two DIFFERENT statutes currently in force together in India, do not assume one).
- "relation" per section reference: "implements" if the circular is exercising/operationalising
  delegated authority under that section (e.g. issuing a prescribed form, laying down a procedure);
  "clarifies" if it is explaining/interpreting how the section applies without creating new
  procedure. Prefer "clarifies" when genuinely unsure.
- "supersedes"/"withdraws": ONLY circulars/notifications this document explicitly says it
  supersedes, withdraws, or replaces - by number if given, else by clear description. Leave empty
  if none are mentioned.

Return ONLY valid JSON, no markdown, no backticks.

{
  "circular_number": "e.g. 3 of 2026 (as written, including any leading zeros)",
  "circular_year": 2026,
  "f_no": "the F.No./file number exactly as written",
  "date_issued": "date exactly as written, e.g. 30th March, 2026",
  "subject": "the Subject/Sub line, verbatim or near-verbatim",
  "issuing_authority": "e.g. Central Board of Direct Taxes, TPL Division",
  "signatory": "name and designation of the signing officer, if given",
  "circular_subtype": "explanatory | extension | general",
  "sections_referenced": [
    {"number": "119", "act": "Income-tax Act, 1961", "relation": "implements"}
  ],
  "applicability_summary": "2-4 sentence plain-language summary of who/what this circular affects and how",
  "key_provisions": ["short bullet-style statements of what the circular actually directs/clarifies"],
  "supersedes": ["circular/notification names or numbers this document explicitly supersedes"],
  "withdraws": ["circular/notification names or numbers this document explicitly withdraws"],
  "effective_from": "date this circular's own directions take effect, if stated, else empty string",
  "keywords": ["5-10 search terms"]
}

CIRCULAR TEXT (read fully):
{text}"""

EXTRACTION_MAX_TOKENS = 4_000
DEFAULT_MAX_CHARS = 200_000  # circulars are short; generous safety rail, not a real-world limit

_SUBTYPE_TO_CLASS = {
    "explanatory": "ExplanatoryCircular",
    "extension": "ExtensionCircular",
    "general": "CBDTCircular",
}

# Which Act name (as the LLM read it off the circular's own text) maps to
# which Act's id-namespace - see module docstring's "ACT-NAMESPACE
# RESOLUTION" section. Matched case-insensitively, substring, most-specific
# pattern first (the 2025 Act's name is a substring-safe superset check
# against "1961" so order doesn't actually matter here, but kept explicit).
_ACT_NAME_PATTERNS = [
    (re.compile(r"income.?tax act,?\s*2025", re.IGNORECASE), "2025"),
    (re.compile(r"income.?tax act,?\s*1961", re.IGNORECASE), "1961"),
]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text or "").strip("_")[:60]


def resolve_act_namespace(act_name: str, ns_1961: str, ns_2025: str) -> str | None:
    for pattern, year in _ACT_NAME_PATTERNS:
        if pattern.search(act_name or ""):
            return ns_2025 if year == "2025" else ns_1961
    return None


SECTION_CITATION_RE = re.compile(r'^\s*(\d+[A-Z]{0,3})((?:\s*\([^)]+\)){0,3})\s*$', re.IGNORECASE)


def resolve_section_citation(number: str, namespace: str | None) -> str | None:
    """Same parse as extract_case_law.py's resolve_section_citation() -
    reimplemented here (rather than imported) only because that function
    lives in extract_case_law.py under a case-law-specific module docstring;
    the citation grammar itself (chunking.py's Section/SubSection/Clause/
    SubClause id scheme) is identical and deliberately kept in sync."""
    m = SECTION_CITATION_RE.match((number or "").strip())
    if not m:
        return None
    section = m.group(1)
    parts = re.findall(r'\(([^)]+)\)', m.group(2) or "")
    subsection = parts[0] if len(parts) > 0 else None
    clause = parts[1] if len(parts) > 1 else None
    sub_clause = parts[2] if len(parts) > 2 else None
    return chunking.generate_entity_id(section, subsection, clause, sub_clause, namespace=namespace)


def call_extraction(text: str, max_chars: int, trace=None, name: str = "circular_extraction") -> tuple[str, bool]:
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True
    prompt = EXTRACTION_INSTRUCTIONS.replace("{text}", text)
    raw = llm_client.complete(
        system="You are a precise legal-document data-extraction assistant. Follow the user's "
               "instructions exactly and return only the requested JSON.",
        user_content=prompt, max_tokens=EXTRACTION_MAX_TOKENS,
        trace=trace, name=name, purpose="ingestion",
    ).strip()
    return raw, truncated


def parse_llm_json(raw: str) -> dict | None:
    cleaned = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def build_graph_from_extraction(extraction: dict, full_text: str, source_file: str,
                                 doc_meta: dict, ns_1961: str, ns_2025: str,
                                 target_chunk_tokens: int = 500) -> dict:
    entities: list[dict] = []
    relationships: list[dict] = []
    ambiguities: list[dict] = []
    run_timestamp = datetime.now(timezone.utc).isoformat()

    number = (extraction.get("circular_number") or "").strip()
    subject = (extraction.get("subject") or "").strip()
    label = number or Path(source_file).stem
    circular_id = f"CBDTCircular_{_slug(label)}"
    subtype = (extraction.get("circular_subtype") or "general").strip().lower()
    entity_type = _SUBTYPE_TO_CLASS.get(subtype, "CBDTCircular")

    circular_props = {
        "hasText": full_text[:8000],  # full text also lives in the Chunk records below
        "full_citation": f"CBDT Circular No. {number}" if number else label,
        "circular_number": number,
        "circular_year": extraction.get("circular_year"),
        "f_no": extraction.get("f_no", ""),
        "date_issued": extraction.get("date_issued", ""),
        "subject": subject,
        "issuing_authority": extraction.get("issuing_authority", ""),
        "signatory": extraction.get("signatory", ""),
        "circular_subtype": subtype,
        "notificationNumber": extraction.get("f_no", "") or number,
        "applicability_summary": extraction.get("applicability_summary", ""),
        "key_provisions": extraction.get("key_provisions", []) or [],
        "keywords": extraction.get("keywords", []) or [],
        **({"effective_from": extraction["effective_from"].strip()} if (extraction.get("effective_from") or "").strip() else {}),
        "confidence": 0.8,
        "extraction_notes": f"Extracted from {source_file} via extract_circular.py",
        "source_document": source_file,
        "ingestion_date": run_timestamp,
        "node_id": circular_id,
        "content_hash": hashlib.sha256((full_text or "").encode("utf-8")).hexdigest()[:16],
    }
    entities.append({"id": circular_id, "type": entity_type, "properties": circular_props})

    for ref in extraction.get("sections_referenced", []) or []:
        number_ref = (ref or {}).get("number", "")
        act_name = (ref or {}).get("act", "")
        relation = ((ref or {}).get("relation") or "clarifies").strip().lower()
        if not number_ref:
            continue
        namespace = resolve_act_namespace(act_name, ns_1961, ns_2025)
        if namespace is None:
            ambiguities.append({
                "type": "unresolved_act_name", "circular": circular_id,
                "raw_section": number_ref, "raw_act": act_name,
            })
            continue
        target_id = resolve_section_citation(number_ref, namespace)
        if not target_id:
            ambiguities.append({
                "type": "unresolved_section_citation", "circular": circular_id,
                "raw_section": number_ref, "raw_act": act_name,
            })
            continue
        rel_type = "IMPLEMENTS" if relation == "implements" else "CLARIFIES"
        relationships.append({
            "type": rel_type, "source": circular_id, "target": target_id,
            "properties": {"cited_as": number_ref, "act": act_name},
        })

    for field, rel_type, event_type in [
        ("supersedes", "SUPERSEDES", "supersession"),
        ("withdraws", "WITHDRAWS", "withdrawal"),
    ]:
        for name in extraction.get(field, []) or []:
            if not name:
                continue
            other_id = f"CBDTCircular_{_slug(name)}"
            entities.append({"id": other_id, "type": "CBDTCircular", "properties": {
                "full_citation": name, "circular_number": name,
                "confidence": 0.3,
                "extraction_notes": f"Stub created from a reference in {label}; will be filled in "
                                     f"properly if/when this circular is itself ingested.",
            }})
            relationships.append({"type": rel_type, "source": circular_id, "target": other_id, "properties": {}})
            # 2026-08-31: NEW - reify SUPERSEDES/WITHDRAWS as a dated
            # LegalEvent too (see chunking.build_legal_event()'s docstring) -
            # the event date is THIS circular's own date_issued, since this
            # circular is the INSTRUMENT that did the superseding/
            # withdrawing; `other_id` (the earlier circular) is the SUBJECT.
            event_entity, event_rels = chunking.build_legal_event(
                event_type=event_type, event_date=extraction.get("date_issued", ""),
                instrument_id=circular_id, instrument_label=label,
                subject_id=other_id, subject_label=name,
            )
            entities.append(event_entity)
            relationships.extend(event_rels)

    for e in entities:
        extract_entities.attach_pipeline_metadata(e, doc_meta, run_timestamp, None)

    chunk_texts = extract_generic.split_into_paragraph_chunks(full_text, target_tokens=target_chunk_tokens)
    chunks = []
    for i, text in enumerate(chunk_texts):
        chunk_id = f"circular_{circular_id}_{i}"
        chunks.append({
            "chunk_id": chunk_id, "text": text,
            "short_title": chunking.make_short_title(text),
            "chunk_type": "circular_text", "element_type": "circular_text",
            "full_citation": circular_props["full_citation"],
            "entity_id": circular_id, "file_type": doc_meta.get("file_type", "circular_pdf"),
        })
    if chunks:
        existing = circular_props.get("source_chunk_ids", "")
        ids = [c for c in existing.split(",") if c] if existing else []
        ids.extend(c["chunk_id"] for c in chunks)
        circular_props["source_chunk_ids"] = ",".join(ids)

    return {"entities": entities, "relationships": relationships, "ambiguities": ambiguities,
            "circular_id": circular_id, "chunks": chunks}


def process_one(md_path: Path, max_chars: int, debug: bool, doc_meta: dict, ns_1961: str, ns_2025: str,
                 target_chunk_tokens: int, trace=None) -> tuple[dict | None, dict | None, dict]:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    stats = {"source_chars": len(text)}

    with step_logger.Step(f"Extracting structured data from {md_path.name} via LLM"):
        raw, truncated = call_extraction(text, max_chars, trace=trace, name=f"circular_extraction_{md_path.stem}")
        stats["truncated"] = truncated
        if truncated:
            step_logger.log(f"  WARNING: document exceeded --max-chars ({max_chars}) and was truncated.")
        extraction = parse_llm_json(raw)
        if extraction is None:
            step_logger.log(f"  WARNING: could not parse model output as JSON ({len(raw)} chars) - skipping.")
            step_logger.log(f"  ----- start of raw output -----\n{raw[:600]}")
            if debug:
                print("----- RAW MODEL OUTPUT (first 1500 chars) -----")
                print(raw[:1500])
            return None, None, stats

    with step_logger.Step("Mapping extraction onto graph entities/relationships"):
        graph = build_graph_from_extraction(extraction, text, md_path.name, doc_meta, ns_1961, ns_2025,
                                             target_chunk_tokens=target_chunk_tokens)
        stats["entities"] = len(graph["entities"])
        stats["relationships"] = len(graph["relationships"])
        stats["ambiguities"] = len(graph["ambiguities"])
        stats["chunks"] = len(graph["chunks"])
        step_logger.log(f"  {stats['entities']} entities, {stats['relationships']} relationships, "
                         f"{stats['chunks']} chunks, {stats['ambiguities']} ambiguities flagged")

    return extraction, graph, stats


def validate_against_schema(graph: dict, schema: dict) -> list[str]:
    class_names = {c["name"] for c in schema.get("classes", [])}
    prop_names = {p["name"] for p in schema.get("object_properties", [])}
    problems = []
    for e in graph["entities"]:
        if e["type"] not in class_names:
            problems.append(f"Entity type {e['type']!r} (id={e['id']}) not found in ontology classes")
    for r in graph["relationships"]:
        if r["type"] not in prop_names:
            problems.append(f"Relationship type {r['type']!r} not found in ontology object_properties")
    return sorted(set(problems))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--md", type=Path, default=None, help="Single circular .md/.txt file")
    parser.add_argument("--out", type=Path, default=None, help="Output graph JSON path (with --md)")
    parser.add_argument("--dir", type=Path, default=None, help="Process every .md file in this folder")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output folder for graph JSON files (with --dir)")
    parser.add_argument("--raw-out-dir", type=Path, default=None)
    parser.add_argument("--chunks-out", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=SCRIPT_DIR.parent / "ontology" / "schema.json")
    parser.add_argument("--file-type", default="circular_pdf")
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--document-name", default=None)
    parser.add_argument("--source-type", default="Circular")
    parser.add_argument("--authority-level", default="Delegated")
    parser.add_argument("--jurisdiction", default="India")
    parser.add_argument("--act-1961-namespace", default="ITA1961",
                         help="Id-namespace of the already-ingested Income-tax Act, 1961 (see ingest_act.py).")
    parser.add_argument("--act-2025-namespace", default="ITA2025",
                         help="Id-namespace of the already-ingested Income-tax Act, 2025.")
    parser.add_argument("--target-chunk-tokens", type=int, default=500)
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--skip-datasource", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    schema = None
    if args.schema.exists():
        with open(args.schema, "r", encoding="utf-8") as f:
            schema = json.load(f)

    llm_trace = observability.start_trace(
        "extract_circular_run", {"md": str(args.md) if args.md else None, "dir": str(args.dir) if args.dir else None}
    )

    def make_doc_meta(md_path: Path) -> dict:
        return {
            "document_id": args.document_id or f"circular_{_slug(md_path.stem)}",
            "document_name": args.document_name or md_path.stem,
            "source_type": args.source_type, "authority_level": args.authority_level,
            "jurisdiction": args.jurisdiction, "file_type": args.file_type,
            # 2026-08-29: stamps sourceMdPath onto every entity via the shared
            # attach_pipeline_metadata() (see extract_entities.py) - same
            # semantic_graph hydration path as CaseLaw/Section, see
            # Semantic_Graph_Retrieval_Architecture.md.
            "source_md_path": str(md_path.resolve()),
        }

    def handle(md_path: Path, out_path: Path, raw_out_path: Path | None, chunks_out_path: Path):
        doc_meta = make_doc_meta(md_path)
        extraction, graph, stats = process_one(
            md_path, args.max_chars, args.debug, doc_meta, args.act_1961_namespace, args.act_2025_namespace,
            args.target_chunk_tokens, trace=llm_trace,
        )
        if graph is None:
            return
        entities, relationships = graph["entities"], graph["relationships"]
        if not args.skip_datasource:
            run_timestamp = datetime.now(timezone.utc).isoformat()
            fake_args = argparse.Namespace(source_class="RegulatoryPublication", source_name=None,
                                            source_location=None, source_notes=None,
                                            priority_tier=None, update_frequency=None)
            source_entity, source_links = extract_entities.build_datasource_entity_and_links(
                [e["id"] for e in entities], doc_meta, fake_args, run_timestamp, len(graph["chunks"])
            )
            entities.append(source_entity)
            relationships.extend(source_links)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"entities": entities, "relationships": relationships}, f, indent=2, ensure_ascii=False)
        if graph["ambiguities"]:
            step_logger.log(f"  {len(graph['ambiguities'])} ambiguity(ies) - see raw extraction / re-run with --debug")
        if raw_out_path:
            raw_out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(raw_out_path, "w", encoding="utf-8") as f:
                json.dump(extraction, f, indent=2, ensure_ascii=False)
        chunks_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_out_path, "w", encoding="utf-8") as f:
            json.dump(graph["chunks"], f, indent=2, ensure_ascii=False)
        if schema:
            problems = validate_against_schema(graph, schema)
            if problems:
                step_logger.log(f"  SCHEMA WARNINGS ({len(problems)}):")
                for p in problems:
                    step_logger.log(f"    - {p}")
        step_logger.log(f"  -> {out_path}")
        step_logger.log(f"  -> {chunks_out_path} ({len(graph['chunks'])} chunks)")

    if args.dir:
        if not args.out_dir:
            raise SystemExit("--out-dir is required with --dir")
        md_files = sorted(args.dir.glob("*.md")) + sorted(args.dir.glob("*.txt"))
        if not md_files:
            raise SystemExit(f"No .md/.txt files found in {args.dir}")
        step_logger.log(f"Found {len(md_files)} circular file(s) to process")
        for md_path in md_files:
            out_path = args.out_dir / (md_path.stem + "_graph.json")
            raw_out_path = (args.raw_out_dir or args.out_dir) / (md_path.stem + "_extraction.json")
            chunks_out_path = args.out_dir / (md_path.stem + "_chunks.json")
            handle(md_path, out_path, raw_out_path, chunks_out_path)
    elif args.md:
        if not args.out:
            raise SystemExit("--out is required with --md")
        raw_out_path = args.raw_out_dir / (args.md.stem + "_extraction.json") if args.raw_out_dir else \
            args.out.parent / (args.md.stem + "_extraction.json")
        chunks_out_path = args.chunks_out or (args.out.parent / (args.md.stem + "_chunks.json"))
        handle(args.md, args.out, raw_out_path, chunks_out_path)
    else:
        observability.end_trace(llm_trace, {"error": "no --md or --dir given"})
        raise SystemExit("Provide either --md/--out or --dir/--out-dir")

    observability.end_trace(llm_trace, {"status": "done"})


if __name__ == "__main__":
    main()
