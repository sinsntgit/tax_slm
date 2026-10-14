"""
International tax agreement (DTAA/protocol/TIEA) structured extraction - the
treaty-specific counterpart to extract_circular.py/extract_case_law.py/
02_extract_entities.py: a HYBRID deterministic + LLM pass over a converted
treaty notification PDF (data/rawdata/international_agreements/*.pdf, as
published on incometaxindia.gov.in), mapped onto the ontology's TaxTreaty /
Article (a declared TaxTreaty subclass) / TreatyNotification classes.

WHY HYBRID (not pure-LLM like extract_case_law.py, not pure-regex like
chunking.py): inspecting the two sample treaties this script was built
against (Albania, China) shows a SPLIT reality -
  - The Article-level body is extremely regular across both: "ARTICLE N" (a
    bare line, no other text) followed by an optional all-caps title line,
    repeated maybe 20-35 times per treaty - exactly the kind of fixed
    citation grammar chunking.py already proves is safe to parse
    deterministically, no LLM needed, and far cheaper for what can be a
    50+ page document.
  - The NOTIFICATION PREAMBLE (which government notification gave the treaty
    domestic effect, when it was signed, when it entered into force) is
    NOT consistent: Albania's preamble reads "...was signed at New Delhi on
    the 8th day of July, 2013... entry into force... is the 4th day of
    December, 2013... Notification No: 2/2014 [F.NO.501/1/2003-FTD-I]/SO
    47(E), dated 7-1-2014", while China's reads "...has come into force on
    the 21st day of November, 1994... Notification : No. GSR 331(E), dated
    5-4-1995, as corrected by Notification No. GSR 574(E)... and as amended
    by Notification No. S.O. 2562(E)... Dated 17-7-2019" - two genuinely
    different sentence structures, and India's treaty corpus spans 90+
    countries signed across 5 decades of varying drafting/notification
    conventions. A single fixed regex would silently mis-parse or blank out
    a meaningful fraction of that corpus; a short, narrow LLM read of just
    the ~3000-char preamble (same cost/reliability tradeoff extract_
    generic.py's extract_document_temporal_metadata() already makes for the
    same "free-text preamble, varies per document" problem) is safer.

WHAT THIS SCRIPT DOES:
  1. Reads one (or a folder of) converted treaty .md/.txt file(s).
  2. Deterministically splits the body into "ARTICLE N" sections via
     split_into_articles() below - zero LLM calls for this part.
  3. Sends ONLY the opening ~3000 chars (the preamble, before "ANNEXURE"/
     the treaty body proper) through one short LLM call for treaty_country,
     treaty_type, the OTHER contracting state's full name, and every
     notification that has given this treaty (or an amendment to it) legal
     effect - each with its own number/date, since (see China above) a
     single treaty is very often given effect and later amended by MULTIPLE
     notifications over the years, not just one.
  4. Maps the result onto ONE TaxTreaty node + one Article node per detected
     ARTICLE (PART_OF -> the TaxTreaty, an ontology-declared structural-
     hierarchy edge - see schema.json's 2026-08-27 PART_OF extension) + one
     TreatyNotification node per notification found (GIVES_EFFECT_TO -> the
     TaxTreaty, the ontology's own declared relation for exactly this).
  5. Produces one embeddable Chunk per Article (chunk_type "treaty_article")
     plus one for the notification preamble - same shape every other
     converter in this project already produces.
  6. Reuses extract_entities.py's attach_pipeline_metadata()/compute_
     content_hash()/build_datasource_entity_and_links() unchanged, same as
     extract_circular.py.

KNOWN LIMITATIONS (explicit, not silently approximated):
  - split_into_articles() assumes the "ARTICLE N" heading convention this
    project's entire sample corpus (Albania, China, and every other treaty
    on incometaxindia.gov.in downloaded in the same export format) uses. A
    treaty published in a genuinely different layout (no ARTICLE headings -
    e.g. a bare amending Protocol with only numbered paragraphs) falls back
    to ONE whole-body chunk with a logged warning, never a silent guess at
    article boundaries.
  - MODIFIED_BY_PROTOCOL/MODIFIED_BY_MLI (TaxTreaty->TaxTreaty, both already
    declared in schema.json) are NOT populated by this script - reliably
    matching an amending Protocol back to its correct base treaty node
    requires resolving free-text references ("the Agreement signed on
    [date]") against already-ingested treaties, which needs either a second
    lookup pass against the graph or much stronger LLM cross-referencing
    than a single short preamble read can safely do. Left as a documented
    follow-up rather than guessed.
  - treaty_type is an LLM read of the document's own title/preamble wording,
    not validated against an external treaty-status database - correct for
    every sample checked (DTAA for both Albania/China) but not independently
    verified against India's actual current treaty network.

Usage:
    python extract_treaty.py --md ../data/treaty_md/Albania.md \\
        --out ../data/treaty_graph/Albania_graph.json \\
        --document-id treaty_albania --document-name "India-Albania DTAA"

    # Every treaty in a folder:
    python extract_treaty.py --dir ../data/treaty_md --out-dir ../data/treaty_graph
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
import extract_entities  # same directory - attach_pipeline_metadata()/compute_content_hash()/
                          # build_datasource_entity_and_links(), reused unchanged
from framework.core.infra import llm_client, step_logger
from framework.core.retrieve import observability

SCRIPT_DIR = Path(__file__).resolve().parent

# --- Deterministic ARTICLE-level split ---------------------------------------
# Matches an "ARTICLE N" heading line, tolerating up to 3 leading/trailing
# markup characters (pdftotext -layout centers these with leading
# whitespace, hence the loose leading [ \t]*). The leading/trailing
# tolerance matters for real consolidated treaty texts: a treaty later
# amended by a Protocol (e.g. the China DTAA sample this script was tested
# against) republishes BOTH the original article (bracketed, e.g.
# "[ARTICLE 1") AND its amended replacement (quoted, e.g. '"ARTICLE 1',
# sometimes with a leading footnote digit like '1[ARTICLE 5') as a redline -
# a plain "^ARTICLE N$" pattern would miss every bracketed/quoted
# occurrence entirely. An optional all-caps title line immediately
# following the heading (e.g. "GENERAL DEFINITIONS") is captured separately
# as the article's title.
ARTICLE_HEADING_RE = re.compile(r'^[ \t]*[\[\"\'0-9]{0,3}ARTICLE[ \t]+(\d+[A-Z]?)[ \t\]\"\']*$', re.MULTILINE)


def split_into_articles(text: str) -> list[dict]:
    """Returns [{"number","title","text"}] in document order. Empty list if
    no ARTICLE heading is found at all (caller falls back to one whole-body
    chunk - see module docstring's limitations note).

    DEDUPLICATION: when the same article number appears more than once (the
    bracketed-original + quoted-amendment redline pattern described above),
    only the FIRST occurrence is kept. Verified against the real China DTAA
    sample this script was tested against: its consolidated text places the
    CURRENT (substituted/in-force) article FIRST, bracketed, immediately
    following the prior article's text - the SECOND, quoted occurrence is
    introduced by an explicit footnote reading "Substituted by Notification
    No. ... Prior to its substitution, Article N read as under:" i.e. the
    quoted repeat is the OLD, superseded wording kept only for historical
    reference, not the current text. "First wins" is therefore the correct
    default for this real-world convention (the opposite of what a naive
    reading of "brackets = original, quotes = replacement" might suggest -
    confirmed by actually reading the footnote text, not assumed)."""
    matches = list(ARTICLE_HEADING_RE.finditer(text))
    by_number: dict[str, dict] = {}
    duplicate_count = 0
    for i, m in enumerate(matches):
        number = m.group(1)
        if number in by_number:
            duplicate_count += 1
            continue  # keep the FIRST occurrence only - see docstring
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip("\n")
        # Title: the first non-blank line of the body, IF it looks like a
        # short all-caps heading (not the start of a numbered paragraph) -
        # otherwise there is no separate title line for this article.
        lines = [ln for ln in body.split("\n") if ln.strip()]
        title = ""
        if lines:
            first = lines[0].strip()
            if len(first) <= 80 and first == first.upper() and not first[:1].isdigit():
                title = first
                body = "\n".join(lines[1:]).strip()
        by_number[number] = {"number": number, "title": title, "text": body.strip()}
    if duplicate_count:
        step_logger.log(f"  {duplicate_count} duplicate ARTICLE N heading(s) found (consolidated/amended-"
                         f"text redline) - kept only each article's FIRST (current, in-force) occurrence; "
                         f"see extract_treaty.py's split_into_articles() docstring.")
    return list(by_number.values())


# --- LLM preamble read --------------------------------------------------------

PREAMBLE_MAX_CHARS = 3000
PREAMBLE_SYSTEM_PROMPT = """You read the OPENING/PREAMBLE text of an Indian government notification
giving domestic legal effect to a bilateral tax agreement (DTAA/protocol/TIEA) with another
country. Extract ONLY what the text actually states - never guess a date, number, or country name
that isn't explicitly present.

Return ONLY valid JSON, no markdown, no backticks:
{
  "country": "the other contracting state's short common name, e.g. 'Albania'",
  "country_official_name": "the other contracting state's full official name as used in the text, e.g. 'the Republic of Albania'",
  "treaty_type": "DTAA | Protocol | TIEA | MLI | Other - your best classification of THIS agreement from the title/preamble wording",
  "signed_date": "the date the underlying Agreement itself was SIGNED, exactly as written (e.g. '8th day of July, 2013'), or empty string if not stated",
  "entry_into_force_date": "the date the Agreement itself entered into force / came into force, exactly as written, or empty string if not stated",
  "notifications": [
    {"number": "the notification's own number exactly as written, e.g. 'No. 2/2014' or 'GSR 331(E)'",
     "f_no": "the F.No./file number if separately given, else empty string",
     "date": "the date this specific notification was issued/dated, exactly as written",
     "note": "very short note on what this notification did if stated, e.g. 'original notification' or 'amended by S.O. 2562(E)'"}
  ]
}
List EVERY distinct notification number mentioned (a treaty is often first notified, then later
corrected or amended by one or more further notifications - list each one separately, in the
order mentioned).

PREAMBLE TEXT:
{text}"""


def call_preamble_extraction(preamble_text: str, trace=None, name: str = "treaty_preamble") -> str:
    prompt = PREAMBLE_SYSTEM_PROMPT.replace("{text}", preamble_text[:PREAMBLE_MAX_CHARS])
    return llm_client.complete(
        system="You are a precise legal-document data-extraction assistant. Follow the user's "
               "instructions exactly and return only the requested JSON.",
        user_content=prompt, max_tokens=1200, trace=trace, name=name, purpose="ingestion",
    ).strip()


def parse_llm_json(raw: str) -> dict | None:
    cleaned = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text or "").strip("_")[:60]


def build_graph_from_extraction(preamble: dict, articles: list[dict], full_text: str,
                                 source_file: str, doc_meta: dict) -> dict:
    entities: list[dict] = []
    relationships: list[dict] = []
    ambiguities: list[dict] = []
    run_timestamp = datetime.now(timezone.utc).isoformat()

    country = (preamble.get("country") or Path(source_file).stem).strip()
    treaty_id = f"TaxTreaty_{_slug(country)}"
    treaty_type = (preamble.get("treaty_type") or "DTAA").strip()

    treaty_props = {
        "hasText": full_text[:4000],
        "full_citation": f"India-{country} {treaty_type}",
        "treatyCountry": country,
        "country_official_name": preamble.get("country_official_name", ""),
        "treaty_type": treaty_type,
        "signed_date": preamble.get("signed_date", ""),
        **({"effective_from": preamble["entry_into_force_date"].strip()}
           if (preamble.get("entry_into_force_date") or "").strip() else {}),
        "keywords": [country, treaty_type],
        "confidence": 0.8,
        "extraction_notes": f"Extracted from {source_file} via extract_treaty.py",
        "source_document": source_file,
        "ingestion_date": run_timestamp,
        "node_id": treaty_id,
        "content_hash": hashlib.sha256((full_text or "").encode("utf-8")).hexdigest()[:16],
    }
    entities.append({"id": treaty_id, "type": "TaxTreaty", "properties": treaty_props})

    # OVERRIDES (sec 90(2)) - see Graph_First_Retrieval_Design.md Section 2.2
    # and schema.json's OVERRIDES comment. Seeded here deterministically for
    # the ONE well-established, black-letter structural rule this project is
    # confident enough to state without case-specific legal review: sec
    # 90(2) of the Income-tax Act, 1961 (renumbered under the 2025 Act - see
    # extract_case_law.py's DEFAULT_CITED_ACT_NAMESPACE docstring for that
    # renumbering) provides that where a DTAA applies, the Act's own
    # provisions apply ONLY to the extent they are MORE BENEFICIAL to the
    # assessee - i.e. the treaty conditionally overrides the Act, not
    # unconditionally. Target is the Act's own root entity (id ==
    # act_entity_id == the namespace value - see chunking.py's
    # `builder.add_entity(namespace, "Act", ...)`), NOT any specific
    # section - the whole Act is conditionally subordinate, not one
    # provision. Deliberately NOT extended to per-case or per-article
    # precedence questions (e.g. GAAR/PPT limitations on a specific treaty
    # benefit) without legal-domain review - see the schema comment.
    #
    # Safe to always emit: graph_loader.py's load_relationships() drops
    # (with a logged count, not a crash) any relationship whose target
    # doesn't exist yet - so this is a no-op until the referenced Act is
    # actually ingested, and self-heals on the next treaty re-ingestion (or
    # simply becomes live retroactively once the loader's MERGE resolves it)
    # once it is. Pass --overridden-act-namespace '' to skip entirely.
    overridden_act_namespace = (doc_meta or {}).get("overridden_act_namespace", "ITA1961")
    if overridden_act_namespace:
        relationships.append({
            "type": "OVERRIDES", "source": treaty_id, "target": overridden_act_namespace,
            "properties": {
                "overrideCondition": (
                    "Per sec 90(2) of the Income-tax Act, 1961 (as renumbered under the Income-tax Act, "
                    "2025): where this DTAA applies to an assessee, the Act's provisions apply ONLY to the "
                    "extent they are MORE BENEFICIAL to the assessee than this treaty - not an unconditional "
                    "override. Also subject to GAAR (Chapter X-A) / the MLI's Principal Purpose Test, which "
                    "can in turn limit a treaty benefit - not modeled as a separate edge here; a specific "
                    "PPT-driven denial of benefit is a case-specific determination, not a structural rule."
                ),
            },
        })

    notifications = preamble.get("notifications") or []
    if not notifications:
        ambiguities.append({"type": "no_notification_found", "treaty": treaty_id})
    for idx, notif in enumerate(notifications, start=1):
        number = (notif or {}).get("number", "").strip()
        notif_id = f"TreatyNotification_{_slug(country)}_{idx}_{_slug(number) or idx}"
        entities.append({"id": notif_id, "type": "TreatyNotification", "properties": {
            "hasText": (notif or {}).get("note", ""),
            "full_citation": f"Notification {number} (India-{country} {treaty_type})" if number else f"Notification {idx} (India-{country} {treaty_type})",
            "notificationNumber": number,
            "f_no": (notif or {}).get("f_no", ""),
            "date_issued": (notif or {}).get("date", ""),
            **({"effective_from": (notif or {}).get("date", "").strip()} if (notif or {}).get("date", "").strip() else {}),
            "confidence": 0.75,
            "extraction_notes": "",
        }})
        relationships.append({"type": "GIVES_EFFECT_TO", "source": notif_id, "target": treaty_id, "properties": {}})

    if not articles:
        ambiguities.append({"type": "no_articles_detected", "treaty": treaty_id,
                             "note": "ARTICLE heading pattern not found - falling back to whole-body chunking."})
    article_ids: list[str] = []
    for art in articles:
        art_id = f"Article_{art['number']}_{treaty_id}"
        article_ids.append(art_id)
        entities.append({"id": art_id, "type": "Article", "properties": {
            "hasText": art["text"],
            "full_citation": f"Article {art['number']}" + (f" ({art['title']})" if art["title"] else "") + f" of India-{country} {treaty_type}",
            "article_number": art["number"], "article_title": art["title"],
            "confidence": 0.95,  # deterministic split, not LLM-inferred
            "extraction_notes": "",
        }})
        relationships.append({"type": "PART_OF", "source": art_id, "target": treaty_id, "properties": {}})

    for e in entities:
        extract_entities.attach_pipeline_metadata(e, doc_meta, run_timestamp, None)

    chunks = []
    if notifications or not articles:
        preamble_chunk_id = f"treaty_{treaty_id}_notification"
        preamble_text = full_text.split("ANNEXURE", 1)[0].strip() or full_text[:2000]
        chunks.append({
            "chunk_id": preamble_chunk_id, "text": preamble_text,
            "short_title": f"Notification giving effect to India-{country} {treaty_type}",
            "chunk_type": "treaty_notification", "element_type": "treaty_notification",
            "full_citation": treaty_props["full_citation"],
            "entity_id": (entities[1]["id"] if notifications else treaty_id),
            "file_type": doc_meta.get("file_type", "treaty_pdf"),
        })
    if articles:
        for art, art_id in zip(articles, article_ids):
            if not art["text"]:
                continue
            chunk_id = f"treaty_{art_id}"
            chunks.append({
                "chunk_id": chunk_id, "text": art["text"],
                "short_title": f"Article {art['number']}" + (f" - {art['title']}" if art["title"] else ""),
                "chunk_type": "treaty_article", "element_type": "treaty_article",
                "full_citation": f"Article {art['number']} of India-{country} {treaty_type}",
                "entity_id": art_id, "file_type": doc_meta.get("file_type", "treaty_pdf"),
            })
    else:
        # Fallback: no ARTICLE headings found - chunk the whole body via the
        # same generic paragraph splitter extract_generic.py/extract_circular.py
        # use, all linked to the TaxTreaty node itself.
        import extract_generic
        for i, text in enumerate(extract_generic.split_into_paragraph_chunks(full_text, target_tokens=600)):
            chunks.append({
                "chunk_id": f"treaty_{treaty_id}_body_{i}", "text": text,
                "short_title": chunking.make_short_title(text),
                "chunk_type": "treaty_body_fallback", "element_type": "treaty_body_fallback",
                "full_citation": treaty_props["full_citation"], "entity_id": treaty_id,
                "file_type": doc_meta.get("file_type", "treaty_pdf"),
            })

    # Stamp source_chunk_ids onto whichever entity each chunk points at, same
    # convention extract_case_law.py's build_chunks_and_link() uses.
    by_id = {e["id"]: e for e in entities}
    for c in chunks:
        entity = by_id.get(c["entity_id"])
        if entity is None:
            continue
        existing = entity["properties"].get("source_chunk_ids", "")
        ids = [x for x in existing.split(",") if x] if existing else []
        ids.append(c["chunk_id"])
        entity["properties"]["source_chunk_ids"] = ",".join(ids)

    return {"entities": entities, "relationships": relationships, "ambiguities": ambiguities,
            "treaty_id": treaty_id, "chunks": chunks}


def process_one(md_path: Path, doc_meta: dict, trace=None) -> tuple[dict | None, dict | None, dict]:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    stats = {"source_chars": len(text)}

    with step_logger.Step(f"Deterministic ARTICLE-level split of {md_path.name}"):
        articles = split_into_articles(text)
        stats["articles"] = len(articles)
        step_logger.log(f"  {len(articles)} article(s) found" if articles else
                         "  WARNING: no 'ARTICLE N' headings found - will fall back to whole-body chunking.")

    with step_logger.Step("Extracting notification preamble metadata via LLM"):
        raw = call_preamble_extraction(text, trace=trace, name=f"treaty_preamble_{md_path.stem}")
        preamble = parse_llm_json(raw)
        if preamble is None:
            step_logger.log(f"  WARNING: could not parse preamble model output as JSON ({len(raw)} chars) - "
                             f"continuing with an empty preamble (country will fall back to the filename).")
            preamble = {}

    with step_logger.Step("Mapping extraction onto graph entities/relationships"):
        graph = build_graph_from_extraction(preamble, articles, text, md_path.name, doc_meta)
        stats["entities"] = len(graph["entities"])
        stats["relationships"] = len(graph["relationships"])
        stats["ambiguities"] = len(graph["ambiguities"])
        stats["chunks"] = len(graph["chunks"])
        step_logger.log(f"  {stats['entities']} entities, {stats['relationships']} relationships, "
                         f"{stats['chunks']} chunks, {stats['ambiguities']} ambiguities flagged")

    return preamble, graph, stats


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
    parser.add_argument("--md", type=Path, default=None, help="Single treaty .md/.txt file")
    parser.add_argument("--out", type=Path, default=None, help="Output graph JSON path (with --md)")
    parser.add_argument("--dir", type=Path, default=None, help="Process every .md file in this folder")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output folder for graph JSON files (with --dir)")
    parser.add_argument("--raw-out-dir", type=Path, default=None)
    parser.add_argument("--chunks-out", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=SCRIPT_DIR.parent / "ontology" / "schema.json")
    parser.add_argument("--file-type", default="treaty_pdf")
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--document-name", default=None)
    parser.add_argument("--source-type", default="TaxTreaty")
    parser.add_argument("--authority-level", default="Treaty")
    parser.add_argument("--jurisdiction", default="India")
    parser.add_argument("--overridden-act-namespace", default="ITA1961",
                         help="Which Act's id-namespace (see chunking.py's generate_entity_id()) this treaty's "
                              "sec 90(2) OVERRIDES edge points at - see build_graph_from_extraction()'s "
                              "docstring section on OVERRIDES. Defaults to the old Act (ITA1961), matching "
                              "extract_case_law.py's DEFAULT_CITED_ACT_NAMESPACE default for the same reason: "
                              "it's the Act the current corpus is anchored on. Pass '' to skip creating this "
                              "edge entirely for a treaty where sec 90(2)-style override doesn't apply as-is.")
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
        "extract_treaty_run", {"md": str(args.md) if args.md else None, "dir": str(args.dir) if args.dir else None}
    )

    def make_doc_meta(md_path: Path) -> dict:
        return {
            "document_id": args.document_id or f"treaty_{_slug(md_path.stem)}",
            "document_name": args.document_name or md_path.stem,
            "source_type": args.source_type, "authority_level": args.authority_level,
            "jurisdiction": args.jurisdiction, "file_type": args.file_type,
            "overridden_act_namespace": args.overridden_act_namespace,
            # 2026-08-29: stamps sourceMdPath onto every entity via the shared
            # attach_pipeline_metadata() (see extract_entities.py) - same
            # semantic_graph hydration path as CaseLaw/Section, see
            # Semantic_Graph_Retrieval_Architecture.md.
            "source_md_path": str(md_path.resolve()),
        }

    def handle(md_path: Path, out_path: Path, raw_out_path: Path | None, chunks_out_path: Path):
        doc_meta = make_doc_meta(md_path)
        preamble, graph, stats = process_one(md_path, doc_meta, trace=llm_trace)
        if graph is None:
            return
        entities, relationships = graph["entities"], graph["relationships"]
        if not args.skip_datasource:
            run_timestamp = datetime.now(timezone.utc).isoformat()
            fake_args = argparse.Namespace(source_class="TreatyNetworkSource", source_name=None,
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
                json.dump(preamble, f, indent=2, ensure_ascii=False)
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
        step_logger.log(f"Found {len(md_files)} treaty file(s) to process")
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
