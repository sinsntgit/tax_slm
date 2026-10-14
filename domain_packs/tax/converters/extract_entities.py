"""
Step 2: Read the Income Tax Act markdown file, parse it into a deterministic
Section -> Subsection -> Clause -> Sub-clause hierarchy (chunking.py), and
use the LLM only for the narrower semantic layer that deterministic regex
parsing genuinely can't do: identifying DefinedTerm definitions/usages,
Illustrations, and Memoranda.

REWRITTEN per the "Act Document Chunking Strategy" spec (instructions.md):
previously this script chunked purely by markdown heading boundaries + a
character cap, then asked the LLM to GUESS every entity's section_no,
sub_section_no, clause_id, full_citation, and amendment info from raw text
on every single chunk. That's both imprecise (LLM citation guesses can be
wrong) and wasteful (chunking.py computes all of this exactly, for free,
via regex, before any LLM call happens). Now:

  1. chunking.chunk_act_document() parses the WHOLE document once and
     returns, with ZERO LLM calls: every Section/SubSection/Clause/
     SubClause/Proviso/Explanation entity (citation-precise ids, hasText,
     section_no/sub_section_no/clause_id/sub_clause_id, amendment info
     parsed from the source's footnote markers, redacted-content flags)
     plus their structural relationships (PART_OF, HAS_PROVISO,
     HAS_EXPLANATION, REFERS_TO from cross-references, CROSS_REFERS_TO_ALLIED_LAW
     for Schedule<->Section links) - see chunking.py's module docstring for
     the parsing strategy and its known limitations.
  2. The LLM is only asked to fill in what genuinely needs judgment and
     can't be regexed: DefinedTerm extraction (which term does a provision
     DEFINE vs merely USE), Illustration detection, Memorandum detection -
     see EXTRACTION_SYSTEM_PROMPT below. Every chunk sent to the LLM already
     carries its own citation/id as GROUND TRUTH context, so the LLM never
     needs to invent or guess an id - it only ever references the ids
     it's given, plus mints new ids for DefinedTerm/Illustration/Memorandum.
  3. Fine-grained chunks (subsection/clause/sub-clause/proviso/explanation
     level - see chunking.py's Chunk dataclass) are batched into
     ~60,000-char groups per LLM call (chunking.group_chunks_for_llm,
     groups may now span multiple Sections - see its docstring), rather
     than one call per tiny fine-grained chunk, while chunks.json / pgvector
     still store each fine-grained chunk individually for citation-precise
     retrieval (05_embed_chunks.py, webapp/retrieval.py). Raised from the
     original ~7000-char/~1 group-per-Section default after a real full-Act
     run timed out: with the old per-Section-forced grouping, raising
     --max-group-chars alone barely moved the group count (855 Sections ->
     ~855 groups no matter the char cap), so it was the boundary rule, not
     the size, that needed to change. At the new default this is ~45 calls
     for the full Act instead of ~1015 - see --max-output-tokens below for
     the output-side half of this trade-off.

Output: a JSON file with two lists - "entities" and "relationships" -
ready to be loaded into Apache AGE in Step 3. Unchanged shape from before,
so 03_build_graph.py needs no format changes.

Usage:
    python 02_extract_entities.py --schema ../ontology/schema.json --out ../ontology/extracted_graph.json \\
        --document-id ITA1961 --document-name "Income Tax Act, 1961" --source-type Act \\
        --authority-level Statute --jurisdiction India --max-groups 20
    # --md defaults to ../data/income_tax_act_1961_v2.md - override with --md if using a different file.
    # start small with --max-groups while testing, remove/raise for a full run.
"""
import argparse
import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from dotenv import load_dotenv
from tqdm import tqdm

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))

import chunking  # same directory (domain_packs/tax/converters/)
from framework.core.ingest import entity_resolution
from framework.core.infra import audit_log
from framework.core.infra import llm_client
from framework.core.infra import step_logger
from framework.core.infra.dotenv_path import env_path as _env_path
from framework.core.retrieve import observability

# Anchored to the repo root by file location, not process cwd - see
# framework.core.infra.dotenv_path's docstring.
load_dotenv(_env_path())

DEFAULT_MD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "income_tax_act_1961_v2.md")


def load_markdown_text(md_path: str) -> str:
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


# --- SEMANTIC-LAYER EXTRACTION PROMPT ---------------------------------------
# Deliberately narrow: chunking.py already built every Section/SubSection/
# Clause/SubClause/Proviso/Explanation entity with citation-precise ids and
# structural relationships (PART_OF/HAS_PROVISO/HAS_EXPLANATION/REFERS_TO)
# with NO LLM involvement. This prompt's ONLY job is the semantic layer regex
# can't reliably do: which defined terms does a chunk introduce or use, and
# does it contain a worked Illustration or an explanatory Memorandum.

EXTRACTION_SYSTEM_PROMPT = """You are a legal knowledge-graph enrichment engine for Indian income tax law.

You will be given a list of TEXT CHUNKS from the Income Tax Act. Each chunk already
belongs to an EXISTING graph entity - its "id" and "citation" are given to you and
ALREADY EXIST in the knowledge graph. Do NOT re-emit, redefine, or duplicate these
existing entities. Your job is ONLY to identify three additional things the text
supports, which regex-based parsing cannot reliably detect:

1. DEFINED TERMS - a specific word/phrase given a legal definition (e.g. "assessee",
   "previous year", "short-term capital asset"). For each one you find:
   - If THIS chunk is the one DEFINING the term (e.g. '"assessee" means...'), emit a
     DefinedTerm entity (id "DefinedTerm_<snake_case_term>", e.g.
     "DefinedTerm_short_term_capital_asset") and a "DEFINES" relationship from the
     GIVEN chunk's existing entity id to that DefinedTerm id.
   - If THIS chunk merely USES a term that's defined elsewhere (you don't need to know
     where - just that it's clearly a term-of-art, not a definition site), emit a
     "USES_TERM" relationship from the given entity id to the same DefinedTerm id
     (minting the DefinedTerm entity if you haven't already in this same batch).
   - Do not flag ordinary English words - only terms the Act itself treats as having a
     specific, citable legal meaning.

2. ILLUSTRATIONS - a worked numeric example, usually explicitly labeled "Illustration".
   Emit an Illustration entity (id "Illustration_to_<parent_id>") and a
   "HAS_ILLUSTRATION" relationship from the given chunk's entity id to it.

3. MEMORANDA - explanatory memorandum text attached to a provision (e.g. "Memorandum
   explaining the provisions of the Finance Bill"). Emit a Memorandum entity (id
   "Memorandum_to_<parent_id>") and a "HAS_MEMORANDUM" relationship from the given
   chunk's entity id to it.

You may ALSO, only where the text clearly and explicitly supports it (never guess),
suggest a short topical "tag" (e.g. "capital-gains", "tds") for a given chunk's
existing entity - return this as an entry in "entity_updates", NOT as a new entity.

4. RETRIEVAL METADATA - also as entity_updates properties on the given chunk's
   existing entity, ONLY where the text clearly and explicitly supports it (never
   guess or infer something not actually stated):
   - "legal_concepts": 2-6 short topical tags a tax professional would search by,
     e.g. ["demerger", "cost-of-acquisition", "deemed-dividend", "slump-sale",
     "treaty-relief"]. Lowercase, hyphenated, no spaces. Leave the list empty if
     nothing distinctive applies - do not force generic tags like "income-tax".
   - "statutes_cited": other Income-tax Act section citations this text
     explicitly references (NOT this chunk's own citation), e.g. ["Section 46(2)",
     "Section 50B"]. Plain citation strings, not ids.
   - "other_act_references": references to a DIFFERENT statute (Companies Act,
     SEBI regulations, etc.), e.g. ["Companies Act 2013 Section 233"].
   - "treaty_references": references to a tax treaty/DTAA article, e.g.
     ["India-Mauritius DTAA Article 13"].
   Omit any of these four keys entirely from entity_updates if the chunk has
   nothing to report for it - do not emit empty lists.

If a chunk has none of the above, that's a completely normal, expected result - do
not force a match. Most chunks will contribute nothing.

Output ONLY valid JSON matching this exact structure, no prose, no markdown fences:

{
  "new_entities": [
    {"id": "DefinedTerm_assessee", "type": "DefinedTerm",
     "properties": {"term": "assessee", "hasText": "the definition text...", "confidence": 0.95, "extraction_notes": ""}}
  ],
  "new_relationships": [
    {"type": "DEFINES", "source": "SubSection_2_7", "target": "DefinedTerm_assessee", "properties": {}},
    {"type": "USES_TERM", "source": "Section_4", "target": "DefinedTerm_assessee", "properties": {}}
  ],
  "entity_updates": [
    {"id": "SubSection_2_7", "properties": {"tag": "definitions",
      "legal_concepts": ["definitions"], "statutes_cited": ["Section 46(2)"],
      "other_act_references": ["Companies Act 2013 Section 233"],
      "treaty_references": ["India-Mauritius DTAA Article 13"]}}
  ]
}

Each chunk below is shown as: [id=<entity id> | citation=<full citation>] followed by its text.
"""


def build_semantic_schema_excerpt(full_schema: dict) -> dict:
    """Trims the full ontology schema down to just what this prompt's
    narrower job needs (DefinedTerm/Illustration/Memorandum + the object
    properties that attach them) - smaller prompt, faster/cheaper calls,
    and no risk of the LLM wandering back into re-deriving structural
    entities chunking.py already built deterministically."""
    keep_classes = {"DefinedTerm", "Illustration", "Memorandum", "StatutoryInstrument"}
    keep_props = {"DEFINES", "USES_TERM", "HAS_ILLUSTRATION", "HAS_MEMORANDUM"}
    return {
        "classes": [c for c in full_schema.get("classes", []) if c["name"] in keep_classes],
        "object_properties": [p for p in full_schema.get("object_properties", []) if p["name"] in keep_props],
        "data_properties": [p for p in full_schema.get("data_properties", []) if p["name"] in ("tag", "hasText")],
    }


def format_chunk_group_for_prompt(group: list) -> str:
    parts = []
    for c in group:
        parts.append(f"[id={c.entity_id} | citation={c.full_citation}]\n{c.content}")
    return "\n\n---\n\n".join(parts)


def extract_semantic_layer(group: list, schema_excerpt: dict, debug: bool = False,
                            max_tokens: int = 8000, trace=None, group_label: str = "semantic_layer") -> dict:
    user_content = (
        f"ONTOLOGY SCHEMA EXCERPT:\n{json.dumps(schema_excerpt, indent=2)}\n\n"
        f"TEXT CHUNKS:\n{format_chunk_group_for_prompt(group)}"
    )
    raw = llm_client.complete(system=EXTRACTION_SYSTEM_PROMPT, user_content=user_content,
                               max_tokens=max_tokens, trace=trace, name=group_label,
                               purpose="ingestion").strip()
    cleaned = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    empty = {"new_entities": [], "new_relationships": [], "entity_updates": []}
    if not cleaned:
        step_logger.log("  WARNING: model returned EMPTY output for this group (see llm_client log "
                         "line above for why, if provider-reported) - skipping.")
        return empty
    try:
        parsed = json.loads(cleaned)
        parsed.setdefault("new_entities", [])
        parsed.setdefault("new_relationships", [])
        parsed.setdefault("entity_updates", [])
        return parsed
    except json.JSONDecodeError:
        step_logger.log("  WARNING: could not parse model output as JSON for this group. Skipping.")
        if debug:
            print("----- RAW MODEL OUTPUT (first 1000 chars) -----")
            print(raw[:1000])
            print("----- END RAW OUTPUT -----")
        return empty


# --- PIPELINE-LEVEL METADATA -------------------------------------------------
# Everything here is computed by the SCRIPT, not the LLM - these are facts
# about the ingestion run itself (which document, when, what exact text
# produced this entity), not facts extracted from legal text.

def compute_content_hash(text: str) -> str:
    """Stable hash of the entity's own text - lets you later detect whether a
    re-ingestion actually changed anything for a given entity."""
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()[:16]


def attach_pipeline_metadata(entity: dict, doc_meta: dict, run_timestamp: str, chunk_index: int | None):
    """
    Mutates entity["properties"] in place, adding all the pipeline-derived
    metadata fields the ontology defines but nothing else should be
    guessing. `chunk_index` is this entity's position in chunks.json (see
    main()'s entity_id -> chunk_index map built from chunking.py's fine-
    grained Chunk list) - None for entities chunking.py didn't produce a
    standalone Chunk record for (e.g. a Section that further split into
    subsections has no single leaf chunk of its own); those entities are
    still fully queryable as graph nodes, just not directly linked to a
    pgvector chunk for embedding-based retrieval.
    """
    props = entity["properties"]
    props.setdefault("document_id", doc_meta["document_id"])
    props.setdefault("document_name", doc_meta["document_name"])
    props.setdefault("source_type", doc_meta["source_type"])
    props.setdefault("authority_level", doc_meta["authority_level"])
    props.setdefault("jurisdiction", doc_meta["jurisdiction"])
    props.setdefault("file_type", doc_meta.get("file_type", ""))
    props.setdefault("sourceMdPath", doc_meta.get("source_md_path", ""))
    # Document-level effective_from (currently only set by extract_generic.py's
    # extract_document_temporal_metadata() - see its docstring) - a
    # setdefault, same as every other field here, so it never overwrites a
    # MORE PRECISE per-provision effective_from chunking.py's own
    # _amendment_entity_props() may already have set for the Act path (that
    # function runs before this one for every Act entity that had an
    # amendment footnote - this is purely a fallback for entities/documents
    # that don't have their own more specific date).
    if doc_meta.get("effective_from"):
        props.setdefault("effective_from", doc_meta["effective_from"])
        props.setdefault("status", doc_meta.get("status", "in force"))
    props.setdefault("ingestion_date", run_timestamp)
    props.setdefault("node_id", entity["id"])
    if chunk_index is not None:
        props.setdefault("source_chunk_ids", str(chunk_index))
        props.setdefault("chunk_id", str(chunk_index))
    props.setdefault("content_hash", compute_content_hash(props.get("hasText", "")))
    # 2026-10-14: provenance (framework/core/ingest/provenance.py) - a
    # setdefault, so a stub the caller already marked cited_reference keeps it;
    # everything else this shared stamper sees was extracted from the document
    # being ingested. Literal (no import) - same value as provenance.INGESTED_DOCUMENT.
    props.setdefault("provenance", "ingested_document")


def build_datasource_entity_and_links(
    entity_ids: list[str], doc_meta: dict, args, run_timestamp: str, n_chunks: int
) -> tuple[dict, list[dict]]:
    """Builds ONE synthetic DataSource entity for this ingestion run plus an
    `INGESTED_FROM` relationship from every extracted entity to it. Unchanged
    from the prior version of this script."""
    source_id = f"DataSource_{doc_meta['document_id']}"
    source_entity = {
        "id": source_id,
        "type": args.source_class,
        "properties": {
            "source_name": args.source_name or doc_meta["document_name"],
            "document_count": 1,
            "is_available": True,
            "in_poc": True,
            "ingestion_date": run_timestamp,
            "jurisdiction": doc_meta["jurisdiction"],
            "file_type": doc_meta.get("file_type", ""),
        },
    }
    if args.source_location:
        source_entity["properties"]["source_location"] = args.source_location
    if args.source_notes:
        source_entity["properties"]["source_notes"] = args.source_notes
    if args.priority_tier:
        source_entity["properties"]["priority_tier"] = args.priority_tier
    if args.update_frequency:
        source_entity["properties"]["update_frequency"] = args.update_frequency

    links = [
        {"type": "INGESTED_FROM", "source": eid, "target": source_id, "properties": {}}
        for eid in entity_ids
    ]
    return source_entity, links


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--md", default=DEFAULT_MD_PATH,
                         help=f"Path to the markdown file. Defaults to {DEFAULT_MD_PATH}")
    parser.add_argument("--schema", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-groups", type=int, default=None,
                         help="Limit number of LLM-call groups processed (useful for a quick test run). "
                              "The deterministic structural pass (Section/SubSection/Clause/.../Proviso/"
                              "Explanation) always runs on the FULL document regardless of this flag - "
                              "it's free (no LLM calls) and fast (~5s for the whole Act).")
    parser.add_argument("--max-group-chars", type=int, default=60_000,
                         help="Target max characters per LLM-call group (chunking.group_chunks_for_llm). "
                              "Groups may span multiple Sections to actually reduce call count as this is "
                              "raised (see chunking.group_chunks_for_llm's docstring) - raise together with "
                              "--max-output-tokens so each call's output budget keeps pace with how much a "
                              "bigger group is likely to find.")
    parser.add_argument("--max-output-tokens", type=int, default=24_000,
                         help="Max output tokens per semantic-layer LLM call (llm_client.complete). Bigger "
                              "--max-group-chars means more source text per call, so potentially more "
                              "DefinedTerm/Illustration/Memorandum entities to return as JSON per call - if "
                              "this is too low for a given group, the model truncates mid-JSON, that group's "
                              "output fails to parse, and the whole group's findings are silently skipped "
                              "(logged as a WARNING, not an error). Watch the log for that warning after "
                              "raising --max-group-chars; lower the group size or raise this if you see it. "
                              "Raised from the original 8000 default after a real full-Act run showed 39/46 "
                              "groups (85%%) truncating at that budget (responses already 24-29k chars at the "
                              "cutoff) and being silently dropped - 24000 is comfortably within Claude Haiku "
                              "4.5's 64k-token output ceiling, leaving headroom for the largest groups.")
    parser.add_argument("--extraction-workers", type=int, default=5,
                         help="Number of semantic-layer LLM calls to run concurrently (thread pool). Each "
                              "group's extraction is fully independent (own text, own output, merged into "
                              "the shared entity/relationship dicts only after the call returns), and "
                              "llm_client.complete()'s provider/model config is resolved per-call rather "
                              "than via shared mutable globals specifically so concurrent calls are safe "
                              "(see llm_client.ModelConfig's docstring). A real full-Act run had 46 groups "
                              "running strictly sequentially at ~30-70s each - 5 concurrent workers should "
                              "cut that phase to roughly a fifth of the time, budget/rate-limit permitting. "
                              "Set to 1 to restore the old strictly-sequential behavior.")
    parser.add_argument("--include-schedules-in-llm", action="store_true",
                         help="Also run the DefinedTerm/Illustration/Memorandum LLM pass over Schedule "
                              "chunks (skipped by default - schedules are typically tables/procedural "
                              "content, low-yield for this pass, and can be large).")
    parser.add_argument("--skip-semantic-layer", action="store_true",
                         help="Skip the LLM pass entirely - output only the deterministically parsed "
                              "structural entities/relationships (no DefinedTerm/Illustration/Memorandum).")
    parser.add_argument("--debug", action="store_true",
                         help="Print extracted text stats and raw model output on parse failure")
    parser.add_argument("--chunks-out", default=None,
                         help="Path to write the raw chunk texts as JSON (for later embedding into pgvector). "
                              "Defaults to the same folder as --out, named chunks.json")

    # --- Document-level metadata (pipeline-derived, applied to every entity) ---
    parser.add_argument("--document-id", default=None,
                         help="Stable ID for the source document, e.g. 'ITA1961'. Defaults to the md filename.")
    parser.add_argument("--document-name", default="Income Tax Act, 1961",
                         help="Human-readable document name.")
    parser.add_argument("--source-type", default="Act",
                         help="e.g. Act, Rule, Notification, Circular, CaseLaw.")
    parser.add_argument("--authority-level", default="Statute",
                         help="e.g. Statute, DelegatedLegislation, Judicial, FirmOpinion.")
    parser.add_argument("--jurisdiction", default="India")
    parser.add_argument("--file-type", default="act_pdf",
                         help="Stamped onto every entity/chunk this run produces as a `file_type` "
                              "property, so the webapp's 'Filter by file type' control can narrow "
                              "retrieval to it later.")
    parser.add_argument("--act-namespace", default=None,
                         help="Short, stable per-Act id used to prefix every entity id this run mints "
                              "(e.g. 'ITA2025', 'ITA1961') - see chunking.generate_entity_id()'s docstring. "
                              "Defaults to --document-id. Needed so a SECOND Act (e.g. the old 1961 Act, "
                              "ingested alongside the 2025 Act so old case law can cite the RIGHT provision) "
                              "doesn't collide node ids with the first Act's same-numbered sections.")
    parser.add_argument("--act-year", default=None, type=int,
                         help="This Act's year, for the Act node's actYear property (e.g. 2025). "
                              "Auto-detected from --document-name if omitted (looks for a 4-digit year).")
    parser.add_argument("--act-number", default=None,
                         help="This Act's number, e.g. '30' for '[30 OF 2025]'. Optional.")
    parser.add_argument("--no-act-node", action="store_true",
                         help="Skip creating the Act node / PART_OF wiring - keep namespaced ids but no Act "
                              "entity. Rarely needed; mainly for a quick re-chunk where the caller manages "
                              "the Act node itself.")
    parser.add_argument("--effective-from", default=None,
                         help="This document's own overall commencement date (YYYY-MM-DD), e.g. a "
                              "brand-new Act's enactment date. Stamped as a FALLBACK effective_from/"
                              "status=in force on every entity that doesn't already have a more "
                              "precise, PER-PROVISION effective_from from chunking.py's own w.e.f./"
                              "omission footnote parsing (attach_pipeline_metadata() never overwrites "
                              "that - see its docstring). Needed for temporal/'as of <date>' retrieval "
                              "to work on provisions that have no amendment footnote of their own - "
                              "which, for a freshly-enacted Act, is most of them.")

    # --- DataSource node (provenance layer) ---
    parser.add_argument("--source-class", default="StatutoryProvisionsSource",
                         help="Ontology DataSource subclass for this run's source, e.g. "
                              "StatutoryProvisionsSource, CaseLawsSource, CBDTCircularsSource, "
                              "VATInfolineSource, IncomeTaxRulesSource, NotificationsSource.")
    parser.add_argument("--source-name", default=None,
                         help="Human-readable source name (defaults to --document-name).")
    parser.add_argument("--source-location", default=None,
                         help="Where this source lives, e.g. a URL or shared-drive path.")
    parser.add_argument("--source-notes", default=None)
    parser.add_argument("--priority-tier", default=None, help="e.g. P1, P2, P3.")
    parser.add_argument("--update-frequency", default=None, help="e.g. 'annual', 'as-amended'.")
    parser.add_argument("--skip-datasource", action="store_true",
                         help="Skip creating the DataSource node / INGESTED_FROM edges entirely.")
    parser.add_argument("--skip-entity-resolution", action="store_true",
                         help="Skip the fuzzy dedup pass and keep every id as-is.")
    parser.add_argument("--review-flags-out", default=None,
                         help="Path to write medium-confidence possible-duplicate flags as JSON "
                              "(for human review). Defaults to the same folder as --out, named "
                              "entity_resolution_review.json")
    parser.add_argument("--chunking-ambiguities-out", default=None,
                         help="Path to write chunking.py's parse ambiguities (oversized leaves, "
                              "duplicate section/marker ids in the source) for human review. Defaults "
                              "to the same folder as --out, named chunking_ambiguities.json")
    parser.add_argument("--verbose", action="store_true",
                         help="Print extra per-group detail on top of the always-on step/group progress logging.")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    step_logger.log(f"Provider: USE_MODEL={llm_client.USE_MODEL} (model={llm_client.MODEL_NAME}, "
                     f"timeout={llm_client.LLM_TIMEOUT_SECONDS:.0f}s)")

    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)
    schema_excerpt = build_semantic_schema_excerpt(schema)

    doc_meta = {
        "document_id": args.document_id or os.path.splitext(os.path.basename(args.md))[0],
        "document_name": args.document_name,
        "source_type": args.source_type,
        "authority_level": args.authority_level,
        "jurisdiction": args.jurisdiction,
        "file_type": args.file_type,
        # 2026-08-28: NEW. Absolute path to the converted .md file this run
        # read from - see md_hydration.py / Semantic_Graph_Retrieval_
        # Architecture.md. Lets retrieval read the ACTUAL current text of a
        # provision live from the source file at query time (a precise
        # line_start:line_end slice, using the line_start/line_end this
        # entity already gets from chunking.py) instead of only ever seeing
        # whatever text was captured in the Chunk node at ingestion time.
        "source_md_path": os.path.abspath(args.md),
    }
    if args.effective_from:
        doc_meta["effective_from"] = args.effective_from
        doc_meta["status"] = "in force"
    run_timestamp = datetime.now(timezone.utc).isoformat()

    # --- Act-namespaced entity ids (2026-08-27) -----------------------------
    # See chunking.generate_entity_id()'s docstring for why this exists: a
    # bare section NUMBER isn't a stable id across more than one Act. Only
    # relevant for --source-type Act (case law/generic docs don't mint
    # Section/SubSection/etc ids via chunking.py at all).
    act_namespace = None
    act_props = None
    if args.source_type == "Act":
        act_namespace = args.act_namespace or doc_meta["document_id"]
        if not args.no_act_node:
            act_year = args.act_year
            if act_year is None:
                year_match = re.search(r'\b(1[89]\d{2}|20\d{2})\b', doc_meta["document_name"])
                act_year = int(year_match.group(1)) if year_match else None
            act_props = {
                "actName": doc_meta["document_name"],
                "actNamespace": act_namespace,
                "hasText": doc_meta["document_name"],
                "full_citation": doc_meta["document_name"],
                "jurisdiction": doc_meta["jurisdiction"],
                "confidence": 1.0, "extraction_notes": "",
            }
            if act_year:
                act_props["actYear"] = act_year
            if args.act_number:
                act_props["actNumber"] = args.act_number
        step_logger.log(f"Act namespace for this run: {act_namespace!r} "
                         f"(every Section/SubSection/Clause/SubClause id will be prefixed '{act_namespace}__' - "
                         f"needed so this Act's sections don't collide with a differently-numbered Act "
                         f"ingested into the same graph).")

    run_id = audit_log.start_run(
        script="02_extract_entities.py", document_id=doc_meta["document_id"], args=vars(args)
    )
    # One Langfuse trace for this whole extraction run - every semantic-layer
    # LLM call below is nested under it (via extract_semantic_layer's `trace`
    # kwarg), so a full-document ingestion shows up as ONE trace containing
    # all its LLM calls in Langfuse, not one disconnected trace per call.
    # No-op if LANGFUSE_* isn't configured in .env - see observability.py.
    llm_trace = observability.start_trace(
        "extract_entities_run", {"document_id": doc_meta["document_id"], "md": args.md}, run_id=run_id
    )

    with step_logger.Step(f"Reading markdown file ({args.md})"):
        full_text = load_markdown_text(args.md)
        step_logger.log(f"Loaded {len(full_text)} characters of text from the markdown file.")
        if args.debug:
            print("----- FIRST 500 CHARS OF TEXT -----")
            print(full_text[:500])
            print("----- END SAMPLE -----")

    with step_logger.Step("Deterministic structural parse (chunking.py - no LLM calls)"):
        parsed = chunking.chunk_act_document(full_text, namespace=act_namespace, act_props=act_props)
        step_logger.log(f"Structural pass: {len(parsed['entities'])} entities, "
                         f"{len(parsed['relationships'])} relationships, {len(parsed['chunks'])} fine-grained "
                         f"chunks, {len(parsed['footnote_index'])} amendment footnotes indexed, "
                         f"{len(parsed['ambiguities'])} items flagged for review.")

    with step_logger.Step("Applying page numbers from pdf_to_md_act.py's page-map sidecar (if present)"):
        pagemap_path = os.path.splitext(args.md)[0] + ".pagemap.json"
        page_boundaries = []
        if os.path.exists(pagemap_path):
            try:
                with open(pagemap_path, "r", encoding="utf-8") as f:
                    page_boundaries = json.load(f)
            except (OSError, ValueError) as e:
                step_logger.log(f"  Could not read page-map sidecar ({pagemap_path}): {e} - "
                                 f"chunks will have no page_number.")
        chunking.apply_page_numbers(parsed["chunks"], page_boundaries)
        step_logger.log(f"  {len(page_boundaries)} page boundary marker(s) loaded from {pagemap_path}"
                         if page_boundaries else "  No page-map sidecar found - chunks will have no page_number "
                                                  "(only affects Act PDFs converted before this feature was added).")

    all_entities: dict[str, dict] = {}
    all_relationships: list[dict] = list(parsed["relationships"])

    # entity_id -> its position in chunks.json, so 05_embed_chunks.py and
    # 03_build_graph.py's Chunk-node linking can find each entity's source
    # text. Built from chunking.py's fine-grained Chunk list (index order is
    # preserved verbatim into chunks.json below).
    chunk_index_by_entity: dict[str, int] = {c.entity_id: i for i, c in enumerate(parsed["chunks"])}

    # Entities chunking.py produced deterministically (Section/SubSection/
    # Clause/SubClause/Proviso/Explanation) are ALREADY guaranteed unique -
    # their ids are generated straight from parsed citations, and
    # chunking.py's own add_entity() already rejects/flags true duplicates.
    # Tracked separately so entity resolution below (fuzzy STRING similarity)
    # is only ever run against LLM-sourced entities like DefinedTerm, where
    # it's actually useful - fuzzy-matching structural ids is actively
    # harmful here: "Section 55" and "Section 56" are ~94% similar AS
    # STRINGS despite being completely different provisions, and at full-Act
    # scale (750+ sections, thousands of subsections/clauses sharing the
    # same short numeric/lettered vocabulary) that produces exactly the kind
    # of false-positive pile-up fuzzy dedup was never meant to solve here.
    structural_entity_ids: set[str] = set()
    for ent in parsed["entities"]:
        ent.setdefault("properties", {})
        attach_pipeline_metadata(ent, doc_meta, run_timestamp, chunk_index_by_entity.get(ent["id"]))
        all_entities[ent["id"]] = ent
        structural_entity_ids.add(ent["id"])

    if not args.skip_semantic_layer:
        llm_chunks = [c for c in parsed["chunks"] if args.include_schedules_in_llm or c.chunk_type != "schedule"]
        groups = chunking.group_chunks_for_llm(llm_chunks, max_chars=args.max_group_chars)
        if args.max_groups:
            groups = groups[: args.max_groups]
        total_groups = len(groups)

        n_workers = max(1, args.extraction_workers)
        with step_logger.Step(f"Semantic-layer extraction (DefinedTerm/Illustration/Memorandum) - "
                               f"{total_groups} LLM-call groups via {llm_client.USE_MODEL}, "
                               f"{n_workers} concurrent worker(s)"):

            def _run_group(i: int, group: list) -> tuple[int, dict, float]:
                """Runs ONE group's LLM extraction call. Deliberately does NOT touch
                all_entities/all_relationships (the shared, non-thread-safe dicts/list
                built up below) - it only returns the result, so every thread here is
                purely read-only against shared state and merging stays single-threaded
                in the loop after as_completed(), same as the old sequential code just
                run concurrently first.

                Any exception from the LLM call itself (timeout, transient API error,
                etc.) is caught HERE and turned into the same "empty result + WARNING
                log" outcome extract_semantic_layer() already uses for an unparseable
                response - llm_client.complete() re-raises on failure (see its own
                try/except), and left uncaught that would propagate out of fut.result()
                below and crash the WHOLE run, discarding every other group's
                already-completed work over one transient failure. One slow/failed
                group out of 40+ should cost that group's findings, not the run."""
                group_t0 = time.time()
                group_chars = sum(len(c.content) for c in group)
                step_logger.log(
                    f"  group {i + 1}/{total_groups} ({len(group)} chunks, {group_chars} chars) - extracting..."
                )
                try:
                    result = extract_semantic_layer(group, schema_excerpt, debug=args.debug,
                                                     max_tokens=args.max_output_tokens, trace=llm_trace,
                                                     group_label=f"semantic_layer_group_{i + 1}_of_{total_groups}")
                except Exception as e:
                    elapsed = time.time() - group_t0
                    step_logger.log(
                        f"  group {i + 1}/{total_groups} FAILED after {elapsed:.1f}s - "
                        f"{type(e).__name__}: {e} - skipping this group's findings."
                    )
                    return i, {"new_entities": [], "new_relationships": [], "entity_updates": []}, elapsed
                elapsed = time.time() - group_t0
                n_new, n_rel, n_upd = (len(result["new_entities"]), len(result["new_relationships"]),
                                        len(result["entity_updates"]))
                step_logger.log(
                    f"  group {i + 1}/{total_groups} done in {elapsed:.1f}s "
                    f"- {n_new} new entities, {n_rel} new relationships, {n_upd} property updates"
                )
                step_logger.vlog(f"    new entity ids: {[e.get('id') for e in result['new_entities']]}")
                return i, result, elapsed

            results_by_index: dict[int, dict] = {}
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = [pool.submit(_run_group, i, group) for i, group in enumerate(groups)]
                for fut in tqdm(as_completed(futures), total=total_groups, desc="Semantic layer"):
                    i, result, _elapsed = fut.result()
                    results_by_index[i] = result

            # Merge sequentially, in ORIGINAL group order (not completion order) - keeps
            # output deterministic/reproducible run-to-run regardless of which thread
            # happened to finish first, same as the old strictly-sequential loop.
            for i in range(total_groups):
                result = results_by_index[i]

                for ent in result["new_entities"]:
                    eid = ent.get("id")
                    if not eid:
                        continue
                    ent.setdefault("properties", {})
                    if eid in all_entities:
                        # Same DefinedTerm etc. mentioned again in a later group -
                        # merge properties, don't overwrite pipeline metadata.
                        all_entities[eid]["properties"].update(
                            {k: v for k, v in ent["properties"].items()
                             if k not in ("document_id", "document_name", "source_type",
                                          "authority_level", "jurisdiction", "ingestion_date",
                                          "node_id", "content_hash")}
                        )
                    else:
                        attach_pipeline_metadata(ent, doc_meta, run_timestamp, None)
                        all_entities[eid] = ent

                for rel in result["new_relationships"]:
                    if rel.get("source") and rel.get("target") and rel.get("type"):
                        rel.setdefault("properties", {})
                        all_relationships.append(rel)

                for upd in result["entity_updates"]:
                    eid = upd.get("id")
                    if eid and eid in all_entities and isinstance(upd.get("properties"), dict):
                        all_entities[eid]["properties"].update(upd["properties"])

            # defines_terms - flatten the DEFINES relationships just collected into a
            # plain searchable list property on the DEFINING entity, so a "meaning of
            # X" question can filter directly on defines_terms instead of requiring a
            # graph traversal to discover which entity defines a given term. DefinedTerm
            # entities carry their own "term" property (set above by the LLM) - reused
            # here rather than re-deriving the term name from the target id.
            for rel in all_relationships:
                if rel.get("type") != "DEFINES":
                    continue
                source_id, target_id = rel.get("source"), rel.get("target")
                if source_id not in all_entities or target_id not in all_entities:
                    continue
                term = all_entities[target_id]["properties"].get("term")
                if not term:
                    continue
                props = all_entities[source_id]["properties"]
                terms = props.get("defines_terms") or []
                if term not in terms:
                    props["defines_terms"] = terms + [term]

            step_logger.log(f"Semantic-layer pass complete: {len(all_entities)} total entities, "
                             f"{len(all_relationships)} total relationships so far (pre-resolution).")
    else:
        step_logger.log("Semantic-layer LLM pass skipped (--skip-semantic-layer) - structural-only output.")

    # --- ENTITY RESOLUTION (deterministic + fuzzy dedup) ---
    # Deliberately scoped to ONLY the non-structural (LLM-sourced) entities -
    # see structural_entity_ids' comment above for why running fuzzy string
    # similarity against citation-precise ids like "Section 55"/"Section 56"
    # is actively harmful, not just unnecessary. DefinedTerm entities minted
    # independently across different LLM-call groups (e.g. "assessee"
    # surfacing in many chunks) can still end up with slightly different
    # ids/casing - this catches those.
    review_flags = []
    non_structural = [e for eid, e in all_entities.items() if eid not in structural_entity_ids]
    if not args.skip_entity_resolution and non_structural:
        with step_logger.Step(f"Entity resolution (intra-run fuzzy dedup, {len(non_structural)} "
                               f"non-structural entities - structural entities are skipped, already unique)"):
            # Local import (not at module top) - avoids a hard dependency on the tax
            # DomainConfig for any caller that only needs chunking.py's structural parse.
            from domain_packs.tax.config import DOMAIN_CONFIG
            resolved_list, alias_map, review_flags = entity_resolution.resolve_within_batch(
                non_structural, DOMAIN_CONFIG.resolution_key_properties
            )
            for eid in [e["id"] for e in non_structural]:
                all_entities.pop(eid, None)
            for e in resolved_list:
                all_entities[e["id"]] = e
            if alias_map:
                for rel in all_relationships:
                    rel["source"] = alias_map.get(rel["source"], rel["source"])
                    rel["target"] = alias_map.get(rel["target"], rel["target"])
            step_logger.log(f"Auto-merged {len(alias_map)} near-duplicate entities within this run, "
                             f"flagged {len(review_flags)} for manual review.")

    if not args.skip_datasource and all_entities:
        with step_logger.Step("Creating DataSource node + INGESTED_FROM links"):
            source_entity, source_links = build_datasource_entity_and_links(
                list(all_entities.keys()), doc_meta, args, run_timestamp, len(parsed["chunks"])
            )
            all_entities[source_entity["id"]] = source_entity
            all_relationships.extend(source_links)
            step_logger.log(f"Added DataSource node '{source_entity['id']}' ({source_entity['type']}) "
                             f"with {len(source_links)} INGESTED_FROM links.")

    output = {
        "entities": list(all_entities.values()),
        "relationships": all_relationships,
    }

    with step_logger.Step(f"Writing output files ({len(output['entities'])} entities, "
                           f"{len(output['relationships'])} relationships)"):
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        step_logger.log(f"Written to {args.out}")

        chunks_out_path = args.chunks_out or os.path.join(os.path.dirname(args.out) or ".", "chunks.json")
        # Fine-grained Chunk records, WITH their rich metadata (full_citation,
        # chunk_type, cross_references, amendment_info, etc.) - not just raw
        # text - so 03_build_graph.py can store it on the Chunk node and
        # 05_embed_chunks.py / webapp/retrieval.py can filter/cite on it.
        chunk_records = [
            {"chunk_id": i, "text": c.content, **c.to_dict(), "file_type": doc_meta.get("file_type", "")}
            for i, c in enumerate(parsed["chunks"])
        ]
        with open(chunks_out_path, "w", encoding="utf-8") as f:
            json.dump(chunk_records, f, indent=2)
        step_logger.log(f"{len(chunk_records)} fine-grained chunks (with citation metadata) "
                         f"written to {chunks_out_path}")

        if review_flags:
            review_out_path = args.review_flags_out or os.path.join(
                os.path.dirname(args.out) or ".", "entity_resolution_review.json"
            )
            with open(review_out_path, "w", encoding="utf-8") as f:
                json.dump(review_flags, f, indent=2)
            step_logger.log(f"{len(review_flags)} possible-duplicate entities flagged for "
                             f"human review -> {review_out_path}")

        if parsed["ambiguities"]:
            amb_out_path = args.chunking_ambiguities_out or os.path.join(
                os.path.dirname(args.out) or ".", "chunking_ambiguities.json"
            )
            with open(amb_out_path, "w", encoding="utf-8") as f:
                json.dump(parsed["ambiguities"], f, indent=2)
            step_logger.log(f"{len(parsed['ambiguities'])} chunking-parse items flagged for "
                             f"human review -> {amb_out_path}")

    run_stats = {
        "entities": len(output["entities"]),
        "relationships": len(output["relationships"]),
        "fine_grained_chunks": len(parsed["chunks"]),
        "entity_resolution_flags": len(review_flags),
        "chunking_ambiguities": len(parsed["ambiguities"]),
    }
    audit_log.finish_run(run_id, status="success", details=run_stats)
    observability.end_trace(llm_trace, run_stats)
    step_logger.log("All done.")


if __name__ == "__main__":
    main()
