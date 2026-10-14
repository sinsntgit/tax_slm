"""
Generic entity/relationship extraction for the "Generic Document" file type -
anything that isn't the Act (extract_entities.py, deterministic
Section/SubSection/... parse) or a case-law judgment (extract_case_law.py,
one-call-per-case LLM extraction): circulars, notifications, memos, firm
opinions, training material, and anything else in the v5 ontology's 164
classes that doesn't have its own dedicated converter yet.

THE CHUNK-BUDGET REQUIREMENT this script exists to satisfy: ingestion should
be sized to "the maximum chunk allowed by the LLM selected" - load the
WHOLE file in one extraction call if it fits within that model's context
window, otherwise split into as few token-budgeted call-groups as needed.
MODEL_CONTEXT_WINDOWS below gives each of this project's three supported
providers' real context window; model_chunk_budget() subtracts headroom for
the system prompt + the model's own output, and group_chunks_for_llm() packs
paragraph-level chunks into groups no larger than that budget - so a short
circular becomes ONE LLM call, and only a genuinely long document gets split
at all.

Two-layer design (same shape as chunking.py + extract_entities.py for the
Act, and 11_extract_case_law.py for judgments):
  1. DETERMINISTIC layer (this script, no LLM): paragraph-boundary chunking
     into DocumentSection entities under one GenericDocument entity - so
     ingestion always produces a citable, embeddable, queryable graph even
     if the LLM enrichment pass below finds nothing.
  2. LLM ENRICHMENT layer (best-effort, narrow ask): for each token-budgeted
     group of DocumentSection chunks, ask the model which of the v5
     ontology's classes it recognizes in the text (CaseLaw mentions,
     DefinedTerm-like concepts, Circulars/Notifications referenced, etc.)
     and how they relate back to the given DocumentSection ids - never
     asked to re-derive the DocumentSection structure itself, exactly the
     same narrowing extract_entities.py's EXTRACTION_SYSTEM_PROMPT applies
     to the Act's semantic layer.

GenericDocument/DocumentSection are NOT formal classes in New_tax_ontology_v5.owl,
same deliberate-non-ontology-label pattern this project already uses for
Chunk/AuditRun/PiiAccessLog - see domain_packs/tax/config.py's note on this.

Output: <out> (entities/relationships JSON, same {"entities":[...],
"relationships":[...]} shape framework.cli.ingest_graph expects unmodified)
and <chunks-out> (chunk records, same shape framework.cli.embed_chunks expects
unmodified) - zero format translation needed at either call site, matching
every other converter in this project.

Usage:
    python extract_generic.py --md ../data/generic_md/some_circular.md \\
        --schema ../ontology/schema.json --out ../data/generic_graph/some_circular_graph.json \\
        --document-id circular_23_2024 --document-name "CBDT Circular No. 23/2024" \\
        --source-type Circular
"""
import argparse
import json
import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))

import re
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from tqdm import tqdm

import chunking  # same directory (domain_packs/tax/converters/) - reused only for
                  # token_count()/make_short_title(), NOT its Act-specific regex parsing
import extract_entities  # same directory - reused for compute_content_hash()/
                          # attach_pipeline_metadata()/build_datasource_entity_and_links(),
                          # so DataSource/provenance handling stays IDENTICAL across converters
from framework.core.ingest import entity_resolution
from framework.core.infra import audit_log, llm_client, step_logger
from framework.core.infra.dotenv_path import env_path as _env_path
from framework.core.retrieve import observability

# Anchored to the repo root by file location, not process cwd - see
# framework.core.infra.dotenv_path's docstring.
load_dotenv(_env_path())

# --- LLM chunk-budget sizing --------------------------------------------------
# Real context windows for this project's three supported providers/models
# (framework/core/infra/llm_client.py's MODEL_NAMES). Conservative published
# figures, used ONLY to decide extraction call-group sizes - not billing.
MODEL_CONTEXT_WINDOWS = {
    "CLAUDE": 200_000,   # claude-haiku-4-5
    "GEMINI": 1_000_000,  # gemini-1.5-flash
    "GPT": 128_000,       # gpt-5-nano
}
DEFAULT_RESERVE_FOR_SYSTEM_PROMPT = 1_500
DEFAULT_RESERVE_FOR_OUTPUT = 2_000


def model_chunk_budget(reserve_system: int = DEFAULT_RESERVE_FOR_SYSTEM_PROMPT,
                        reserve_output: int = DEFAULT_RESERVE_FOR_OUTPUT) -> int:
    """How many tokens of DOCUMENT TEXT can go into one extraction call for
    whichever provider is currently selected (llm_client.USE_MODEL),
    reserving headroom for the system prompt and the model's own JSON
    output.

    2026-10-14 (phase0/followups): a LOCAL model (vLLM/Ollama, typically an
    8-32k window) has no entry in MODEL_CONTEXT_WINDOWS, so it used to fall
    through to the 100k default - one "whole document" group far bigger than
    the server accepts. When llm_client.context_window_tokens() knows the
    window (LOCAL_LLM_CONTEXT_LENGTH, honouring USE_MODEL_INGESTION and an
    LLM_ROUTE_ENRICHMENT* route - the enrichment calls are named
    enrichment_group_<i>_of_<n>), that window is used via
    llm_client.local_text_token_budget(). API providers are unchanged."""
    local_window = _configured_local_window()
    if local_window:
        return llm_client.local_text_token_budget(local_window, reserve_system, reserve_output)
    window = MODEL_CONTEXT_WINDOWS.get(llm_client.USE_MODEL, 100_000)
    budget = window - reserve_system - reserve_output
    return max(budget, 2_000)  # never go below a sane floor even for a tiny/misconfigured window


def _configured_local_window() -> int | None:
    """2026-10-14: LOCAL_LLM_CONTEXT_LENGTH for the provider that will serve
    the enrichment calls, or None (API provider, or LOCAL with no window
    configured - then the old table/100k default applies and a warning says
    so, rather than guessing a window for an unknown model). Never raises:
    a bad USE_MODEL value surfaces on the real call, not in sizing."""
    try:
        window = llm_client.context_window_tokens("ingestion", name="enrichment_group")
        if not window:
            config = llm_client.refresh_model_config("ingestion", name="enrichment_group")
            if config.use_model == "LOCAL":
                step_logger.log("  WARNING: LOCAL model with no LOCAL_LLM_CONTEXT_LENGTH set - sizing "
                                "extraction groups for a 100k window; set it to the server's max_model_len.")
        return window
    except Exception:
        return None


def generic_chunk_id(index: int, document_id: str) -> str:
    """2026-10-14: NEW. The chunk_id a generic-document run gives its
    index-th Chunk - `<document_id>__<index>`, the same `<namespace>__<i>`
    shape extract_entities.act_chunk_id() gives Act chunks.
    BUGFIX: this was the bare position index (0, 1, 2, ...) for every generic
    document, while graph_loader/embedder MERGE Chunk nodes on chunk_id
    ALONE - so ingesting a second generic document MERGEd its chunk "0" onto
    the first document's chunk "0" (and onto chunk "0" of any Act ingested
    without a namespace), overwriting text/citation and leaving the earlier
    document's DocumentSection pointing at the wrong text. Unlike an Act run
    a generic run ALWAYS has a document_id (main() defaults it to the file
    stem), so this is always namespaced - deterministic for re-runs of the
    same document, distinct across documents. Commas are replaced because
    entities carry chunk ids as a comma-joined `source_chunk_ids` string
    (graph_loader.load_chunk_nodes_and_links() splits on ",")."""
    namespace = str(document_id).strip().replace(",", "_")
    if not namespace:
        raise ValueError("generic_chunk_id() needs a non-empty document_id")
    return f"{namespace}__{index}"


# --- deterministic paragraph-level chunking -----------------------------------

def split_into_paragraph_chunks(text: str, target_tokens: int = 600) -> list[str]:
    """Greedily packs paragraphs (blank-line separated) into chunks close to
    target_tokens, never splitting a paragraph across two chunks unless a
    SINGLE paragraph alone already exceeds target_tokens (rare - a huge
    table dump, say), in which case it's kept whole rather than corrupted
    with a mid-sentence cut. This is the generic-document equivalent of
    chunking.py's SIZE_GUIDELINES tiers, minus any assumption about what a
    "section" or "clause" is."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for p in paragraphs:
        p_tokens = chunking.token_count(p)
        if current and current_tokens + p_tokens > target_tokens:
            chunks.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(p)
        current_tokens += p_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks or ([text.strip()] if text.strip() else [])


def estimate_page_numbers(full_text: str, chunk_texts: list[str]) -> list[int | None]:
    """Best-effort 1-indexed page number per chunk, for PDF sources only.
    pdf_to_md_generic.py's convert_pdf() runs `pdftotext -layout` and does
    NOT strip Poppler's page-separator form-feed (\\x0c) character when
    normalizing the text - it survives untouched into the .md file - so the
    page a chunk started on is just "how many form-feeds precede this
    chunk's start offset in the full text, + 1". Locates each chunk via a
    forward-advancing substring search (chunks are exact substrings of
    full_text, formed by joining paragraphs split from it) so this stays
    correct even if near-duplicate paragraphs repeat later in the document.
    Returns None for a chunk that can't be located (shouldn't normally
    happen) or for a .txt/.md source with no form-feeds at all (every
    "page" comes back as 1, which is honest - there ARE no pages)."""
    pages: list[int | None] = []
    search_from = 0
    for chunk in chunk_texts:
        idx = full_text.find(chunk, search_from)
        if idx == -1:
            pages.append(None)
            continue
        page = full_text.count("\x0c", 0, idx) + 1
        pages.append(page)
        search_from = idx + max(len(chunk), 1)
    return pages


def group_chunks_for_llm(chunk_texts: list[str], token_budget: int) -> list[list[int]]:
    """Packs chunk INDICES into call-groups whose combined token count stays
    under token_budget - the direct implementation of this script's "load
    the whole file in one call if it fits" requirement: if every chunk's
    total is under budget, this returns exactly one group."""
    groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0

    for i, text in enumerate(chunk_texts):
        t = chunking.token_count(text)
        if current and current_tokens + t > token_budget:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(i)
        current_tokens += t

    if current:
        groups.append(current)
    return groups


# --- Document-level temporal metadata (effective_from) ----------------------
# Circulars/notifications/memos routinely state their own applicability date
# in the opening text ("This circular is issued in supersession... and shall
# take effect from 1st April 2023", "w.e.f. 01-04-2021", etc.) - unlike the
# Act (chunking.py parses w.e.f./omission footnotes deterministically per
# provision) or case law (extraction.date == the judgment date), NOTHING in
# this script previously looked for this at all, so generic documents could
# never be found via a temporal/"as of <date>" question or the metadata
# filter's `as_of` parameter (apply_metadata_filters()) - effective_from was
# always unset. One small, single-call, best-effort LLM read of just the
# document's opening text (not per-chunk - a document states its own
# effective date ONCE, not per paragraph) closes that gap.

TEMPORAL_METADATA_SYSTEM_PROMPT = """You read the OPENING TEXT of a tax/legal document (circular,
notification, memo, or similar) and identify its own stated effective/applicability date - the
date FROM WHICH this document's contents apply, if the document itself states one (e.g. "shall
take effect from 1st April 2023", "w.e.f. 01-04-2021", "with immediate effect" combined with a
visible issue date). This is NOT the document's issue/signing date unless the text says the
document takes effect on that same date - only extract effective_from if you can point to
actual wording establishing when it applies, never guess or infer one that isn't stated.

Output ONLY valid JSON, no prose, no markdown fences:
{"effective_from": "YYYY-MM-DD or empty string if not stated", "reference_number": "the document's own circular/notification number, or empty string"}
"""


def extract_document_temporal_metadata(full_text: str, trace=None) -> dict:
    """One cheap best-effort call over just the opening ~4000 chars (a
    document's own effective-date statement is essentially always in its
    header/preamble, never buried deep in the body) - returns {} (not a
    crash) on any parse failure, since this is a nice-to-have enrichment,
    not something that should ever block ingestion."""
    excerpt = full_text[:4000]
    try:
        raw = llm_client.complete(
            system=TEMPORAL_METADATA_SYSTEM_PROMPT, user_content=excerpt, max_tokens=200,
            trace=trace, name="temporal_metadata", purpose="ingestion",
        ).strip()
        cleaned = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
        parsed = json.loads(cleaned)
        if not isinstance(parsed, dict):
            return {}
        return {k: v for k, v in parsed.items() if isinstance(v, str) and v.strip()}
    except Exception as e:
        step_logger.log(f"  (non-fatal) Could not extract document-level temporal metadata: {e}")
        return {}


# --- LLM enrichment prompt ----------------------------------------------------

ENRICHMENT_SYSTEM_PROMPT = """You are a knowledge-graph enrichment engine for an Indian tax-advisory firm's
document repository. You will be given EXCERPTS from a document that has already been split
into numbered sections, each already an EXISTING graph entity - its "id" is given to you and
ALREADY EXISTS in the knowledge graph. Do NOT re-emit or duplicate these existing entities.

Your job: identify any concepts in the text that match one of the ontology classes listed
below, that a legal/tax professional would want to find again later - e.g. a cited case,
a defined term, a referenced circular/notification, a named party or transaction type. Do
NOT force a match - most excerpts may have nothing you can confidently tag, and that's a
completely normal, expected result.

ALLOWED ontology classes for new entities: {class_names}

Also, ONLY where the text clearly and explicitly supports it (never guess), tag the
GIVEN DocumentSection's existing entity itself with retrieval metadata - return this
as an entry in "entity_updates", NOT as a new entity:
- "legal_concepts": 2-6 short topical tags a tax professional would search by, e.g.
  ["demerger", "cost-of-acquisition", "treaty-relief"]. Lowercase, hyphenated.
- "statutes_cited": Income-tax Act section citations explicitly referenced, e.g.
  ["Section 46(2)", "Section 55(2)(b)(iii)"].
- "other_act_references": references to a different statute (Companies Act, SEBI
  regulations, etc.), e.g. ["Companies Act 2013 Section 233"].
- "treaty_references": tax treaty/DTAA article references, e.g.
  ["India-Mauritius DTAA Article 13"].
Omit any of these keys entirely if the excerpt has nothing to report for it - do not
emit empty lists, and do not force a match on every excerpt.

Output ONLY valid JSON matching this exact structure, no prose, no markdown fences:

{{
  "new_entities": [
    {{"id": "<Type>_<short_snake_case_slug>", "type": "<one of the allowed classes>",
     "properties": {{"hasText": "the relevant excerpt or definition...", "confidence": 0.8, "extraction_notes": ""}}}}
  ],
  "new_relationships": [
    {{"type": "MENTIONS", "source": "<the given DocumentSection id>", "target": "<new entity id>", "properties": {{}}}}
  ],
  "entity_updates": [
    {{"id": "<the given DocumentSection id>", "properties": {{
      "legal_concepts": ["demerger"], "statutes_cited": ["Section 46(2)"],
      "other_act_references": ["Companies Act 2013 Section 233"],
      "treaty_references": ["India-Mauritius DTAA Article 13"]}}}}
  ]
}}

Each excerpt below is shown as: [id=<entity id>] followed by its text.
"""


def build_class_excerpt(full_schema: dict, max_classes: int = 60) -> str:
    """A compact class-name list, not the full schema (164 classes x full
    property lists would bloat every call) - enough for the model to pick a
    valid `type` value without inventing one. Capped at max_classes for
    prompt-size sanity; the ontology's own most broadly useful classes for
    document enrichment (CaseLaw, definitions, circulars/notifications,
    legal concepts, parties) are common early entries, but this is a
    best-effort cap, not a curated priority list."""
    names = [c["name"] for c in full_schema.get("classes", [])]
    return ", ".join(names[:max_classes])


def format_group_for_prompt(chunk_texts: list[str], entity_ids: list[str], indices: list[int]) -> str:
    parts = [f"[id={entity_ids[i]}]\n{chunk_texts[i]}" for i in indices]
    return "\n\n---\n\n".join(parts)


def extract_enrichment(group_text: str, class_excerpt: str, debug: bool = False,
                        trace=None, group_label: str = "enrichment") -> dict:
    system = ENRICHMENT_SYSTEM_PROMPT.format(class_names=class_excerpt)
    raw = llm_client.complete(system=system, user_content=group_text, max_tokens=2000,
                               trace=trace, name=group_label, purpose="ingestion").strip()
    cleaned = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    empty = {"new_entities": [], "new_relationships": [], "entity_updates": []}
    if not cleaned:
        step_logger.log("  WARNING: model returned EMPTY output for this group - skipping.")
        return empty
    try:
        parsed = json.loads(cleaned)
        parsed.setdefault("new_entities", [])
        parsed.setdefault("new_relationships", [])
        parsed.setdefault("entity_updates", [])
        return parsed
    except json.JSONDecodeError:
        step_logger.log("  WARNING: could not parse model output as JSON for this group - skipping.")
        if debug:
            print("----- RAW MODEL OUTPUT (first 1000 chars) -----")
            print(raw[:1000])
        return empty


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--md", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--chunks-out", default=None)
    parser.add_argument("--target-chunk-tokens", type=int, default=600,
                         help="Target token size per fine-grained DocumentSection chunk.")
    parser.add_argument("--skip-enrichment", action="store_true",
                         help="Skip the LLM enrichment pass entirely - output only the "
                              "deterministic GenericDocument/DocumentSection structure.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    parser.add_argument("--document-id", default=None)
    parser.add_argument("--document-name", default=None)
    parser.add_argument("--source-type", default="KnowledgeRepositoryItem")
    parser.add_argument("--authority-level", default="Unspecified")
    parser.add_argument("--jurisdiction", default="India")
    parser.add_argument("--file-type", default="generic",
                         help="Stamped onto every entity/chunk this run produces as a `file_type` "
                              "property, so the webapp's 'Filter by file type' control can narrow "
                              "retrieval to it later.")
    parser.add_argument("--source-class", default="KnowledgeRepositoryItem")
    parser.add_argument("--skip-datasource", action="store_true")
    parser.add_argument("--skip-entity-resolution", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    step_logger.log(f"Provider: USE_MODEL={llm_client.USE_MODEL} (model={llm_client.MODEL_NAME}) - "
                     f"chunk budget for this run: {model_chunk_budget()} tokens per extraction call.")

    with open(args.schema, "r", encoding="utf-8") as f:
        schema = json.load(f)
    class_excerpt = build_class_excerpt(schema)

    doc_meta = {
        "document_id": args.document_id or _os.path.splitext(_os.path.basename(args.md))[0],
        "document_name": args.document_name or _os.path.splitext(_os.path.basename(args.md))[0],
        "source_type": args.source_type,
        "authority_level": args.authority_level,
        "jurisdiction": args.jurisdiction,
        "file_type": args.file_type,
    }
    run_timestamp = datetime.now(timezone.utc).isoformat()
    run_id = audit_log.start_run(script="extract_generic.py", document_id=doc_meta["document_id"], args=vars(args))
    # One Langfuse trace for this whole run - see extract_entities.py's
    # matching comment. No-op if LANGFUSE_* isn't configured in .env.
    llm_trace = observability.start_trace(
        "extract_generic_run", {"document_id": doc_meta["document_id"], "md": args.md}, run_id=run_id
    )

    with step_logger.Step(f"Reading {args.md}"):
        with open(args.md, "r", encoding="utf-8") as f:
            full_text = f.read()
        step_logger.log(f"Loaded {len(full_text)} characters.")

    with step_logger.Step("Extracting document-level temporal metadata (effective_from)"):
        temporal_meta = extract_document_temporal_metadata(full_text, trace=llm_trace)
        if temporal_meta.get("effective_from"):
            doc_meta["effective_from"] = temporal_meta["effective_from"]
            doc_meta["status"] = "in force"
            step_logger.log(f"  Found effective_from={temporal_meta['effective_from']!r} - will be "
                             f"stamped on every entity this document produces, for temporal/'as of "
                             f"<date>' retrieval (apply_metadata_filters()'s as_of parameter).")
        else:
            step_logger.log("  No stated effective date found in the document's opening text - "
                             "effective_from left unset (temporal filtering won't match this document).")
        if temporal_meta.get("reference_number"):
            doc_meta["reference_number"] = temporal_meta["reference_number"]

    with step_logger.Step("Deterministic paragraph-level chunking (no LLM calls)"):
        chunk_texts = split_into_paragraph_chunks(full_text, target_tokens=args.target_chunk_tokens)
        chunk_pages = estimate_page_numbers(full_text, chunk_texts)
        step_logger.log(f"{len(chunk_texts)} DocumentSection chunk(s) produced.")

    all_entities: dict[str, dict] = {}
    all_relationships: list[dict] = []
    structural_entity_ids: set[str] = set()

    doc_id = doc_meta["document_id"]
    doc_entity_id = f"GenericDocument_{doc_id}"
    doc_content_hash = extract_entities.compute_content_hash(full_text)
    doc_entity = {
        "id": doc_entity_id, "type": "GenericDocument",
        "properties": {
            "hasText": "", "full_citation": doc_meta["document_name"],
            "confidence": 1.0, "extraction_notes": "",
            "content_hash": doc_content_hash,
        },
    }
    extract_entities.attach_pipeline_metadata(doc_entity, doc_meta, run_timestamp, None)
    all_entities[doc_entity_id] = doc_entity
    structural_entity_ids.add(doc_entity_id)

    section_entity_ids: list[str] = []
    for i, text in enumerate(chunk_texts):
        sec_id = f"DocumentSection_{doc_id}_{i}"
        props = {
            "hasText": text, "full_citation": f"{doc_meta['document_name']} — Part {i + 1}",
            "confidence": 1.0, "extraction_notes": "",
        }
        if chunk_pages[i] is not None:
            props["page_number"] = chunk_pages[i]
        entity = {"id": sec_id, "type": "DocumentSection", "properties": props}
        # 2026-10-14: namespaced chunk id (was the bare index i) - see generic_chunk_id().
        extract_entities.attach_pipeline_metadata(entity, doc_meta, run_timestamp, generic_chunk_id(i, doc_id))
        all_entities[sec_id] = entity
        structural_entity_ids.add(sec_id)
        section_entity_ids.append(sec_id)
        all_relationships.append({"type": "PART_OF", "source": sec_id, "target": doc_entity_id, "properties": {}})

    if not args.skip_enrichment and chunk_texts:
        budget = model_chunk_budget()
        groups = group_chunks_for_llm(chunk_texts, budget)
        with step_logger.Step(f"LLM enrichment pass - {len(groups)} call-group(s) "
                               f"(budget={budget} tokens/call, {'single call - whole document fit' if len(groups) == 1 else 'split across ' + str(len(groups)) + ' calls'})"):
            for gi, indices in enumerate(tqdm(groups, desc="Enrichment")):
                group_text = format_group_for_prompt(chunk_texts, section_entity_ids, indices)
                t0 = time.time()
                tqdm.write(step_logger.fmt(f"  group {gi + 1}/{len(groups)} ({len(indices)} chunks) - extracting..."))
                result = extract_enrichment(group_text, class_excerpt, debug=args.debug, trace=llm_trace,
                                             group_label=f"enrichment_group_{gi + 1}_of_{len(groups)}")
                tqdm.write(step_logger.fmt(f"  group {gi + 1}/{len(groups)} done in {time.time() - t0:.1f}s - "
                                            f"{len(result['new_entities'])} entities, {len(result['new_relationships'])} relationships"))
                for ent in result["new_entities"]:
                    eid = ent.get("id")
                    if not eid or not ent.get("type"):
                        continue
                    ent.setdefault("properties", {})
                    if eid in all_entities:
                        all_entities[eid]["properties"].update(ent["properties"])
                    else:
                        extract_entities.attach_pipeline_metadata(ent, doc_meta, run_timestamp, None)
                        all_entities[eid] = ent
                for rel in result["new_relationships"]:
                    if rel.get("source") and rel.get("target") and rel.get("type"):
                        rel.setdefault("properties", {})
                        all_relationships.append(rel)
                for upd in result["entity_updates"]:
                    eid = upd.get("id")
                    if eid and eid in all_entities and isinstance(upd.get("properties"), dict):
                        all_entities[eid]["properties"].update(upd["properties"])
                time.sleep(0.2)
    else:
        step_logger.log("LLM enrichment pass skipped.")

    review_flags = []
    non_structural = [e for eid, e in all_entities.items() if eid not in structural_entity_ids]
    if not args.skip_entity_resolution and non_structural:
        with step_logger.Step(f"Entity resolution ({len(non_structural)} LLM-sourced entities)"):
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
            step_logger.log(f"Auto-merged {len(alias_map)}, flagged {len(review_flags)} for review.")

    if not args.skip_datasource:
        with step_logger.Step("Creating DataSource node + INGESTED_FROM links"):
            fake_args = argparse.Namespace(
                source_class=args.source_class, source_name=None,
                source_location=None, source_notes=None, priority_tier=None, update_frequency=None,
            )
            source_entity, source_links = extract_entities.build_datasource_entity_and_links(
                list(all_entities.keys()), doc_meta, fake_args, run_timestamp, len(chunk_texts)
            )
            all_entities[source_entity["id"]] = source_entity
            all_relationships.extend(source_links)

    output = {"entities": list(all_entities.values()), "relationships": all_relationships}

    with step_logger.Step(f"Writing output ({len(output['entities'])} entities, {len(output['relationships'])} relationships)"):
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        step_logger.log(f"Written to {args.out}")

        chunks_out_path = args.chunks_out or _os.path.join(_os.path.dirname(args.out) or ".", "chunks.json")
        chunk_records = [
            {
                # 2026-10-14: namespaced (was the bare index i) - see generic_chunk_id().
                "chunk_id": generic_chunk_id(i, doc_id), "text": text, "entity_id": section_entity_ids[i],
                "full_citation": f"{doc_meta['document_name']} — Part {i + 1}",
                "short_title": chunking.make_short_title(text), "chunk_type": "document_section",
                "element_type": "document_section", "file_type": doc_meta.get("file_type", ""),
                **({"page_number": chunk_pages[i]} if chunk_pages[i] is not None else {}),
            }
            for i, text in enumerate(chunk_texts)
        ]
        with open(chunks_out_path, "w", encoding="utf-8") as f:
            json.dump(chunk_records, f, indent=2)
        step_logger.log(f"{len(chunk_records)} chunks written to {chunks_out_path}")

    run_stats = {
        "entities": len(output["entities"]), "relationships": len(output["relationships"]),
        "chunks": len(chunk_texts), "entity_resolution_flags": len(review_flags),
    }
    audit_log.finish_run(run_id, status="success", details=run_stats)
    observability.end_trace(llm_trace, run_stats)
    step_logger.log("All done.")


if __name__ == "__main__":
    main()
