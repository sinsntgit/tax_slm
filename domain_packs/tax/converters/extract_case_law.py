"""
Step 11: LLM structured-extraction pass for case-law judgments (converted
to Markdown by 10_pdf_to_md_caselaw.py) - the case-law equivalent of
chunking.py + 02_extract_entities.py for the Act, but a genuinely DIFFERENT
approach, not a reuse of either.

WHY THIS ISN'T A "CHUNKING" PROGRAM LIKE chunking.py:
chunking.py works because income_tax_act_1961_v2.md has a fixed, regex-
matchable citation grammar - "#### Section N." headings, "(1)"/"(a)"/"(i)"
markers - so structure can be parsed deterministically, with the LLM only
filling in judgment calls chunking.py's regex can't make (DefinedTerm
detection, etc.). A judgment has no such grammar: it's discursive prose -
facts, arguments, reasoning, holding - identifiable only by READING it, not
by a citation pattern. (Confirmed by inspecting all 6 files in data/: no
stable heading structure beyond loose numbered paragraphs; two different
export conventions even within this small corpus - see 10's docstring.)
The ontology (ontology/schema.json) independently confirms this was always
the intended approach for case law: it already defines Issue, SubIssue,
Argument, TaxpayerArgument, RevenueArgument, TaxQuestion, FactPattern,
CaseOutcome, CaseCitation, Judge, Bench, LegalConcept and the INTERPRETS /
HAS_ARGUMENT / HAS_CONCLUSION / FOLLOWS / DISTINGUISHES_CASE / RELIES_ON_CASE
/ HAS_CITATION relationships - a full reasoning-structure schema that only
makes sense as the target of LLM extraction, not a deterministic parser.
Tax_KG_Architecture_Blueprint.docx's own pipeline design (Sec 8.2) says
exactly this: case law gets "header-pattern segmentation... via regex + LLM
fallback" and "an LLM extraction pass (structured JSON output)" - not a
chunking.py-style parser.

WHAT THIS SCRIPT DOES:
  1. Reads one (or a folder of) case-law .md file(s).
  2. Sends the FULL text through llm_client.complete() with the extraction
     prompt below (verbatim from the KPMG case-law extraction spec this
     project is standardizing on), asking for one rich structured JSON
     object per case: citation/court/bench/date, sections interpreted/
     applied/referred, issues + per-issue findings (holding + reasoning),
     assessee/revenue arguments, precedents followed/distinguished/
     referred, legal principles, and the final outcome.
  3. Maps that JSON onto the ontology's CaseLaw/Issue/Argument/Conclusion/
     CaseCitation/LegalConcept classes and the object properties above,
     producing entities/relationships in the EXACT shape 03_build_graph.py
     already consumes (`{"id","type","properties"}` / `{"type","source",
     "target","properties"}`) - so no new graph-loader is needed, only a
     new extraction step feeding the existing one.
  4. Resolves sections_interpreted/applied/referred citations (e.g.
     "2(19AA)") into the SAME entity ids chunking.py generates for the Act
     graph (Section_2 / SubSection_2_19AA / ...) via chunking.
     generate_entity_id() - imported directly, not re-implemented - so
     INTERPRETS edges land on real, already-ingested Act nodes rather than
     orphaned stand-ins, PROVIDED the Act has been ingested with matching
     ids (built from the same section/subsection/clause numbering
     convention either way).

KNOWN LIMITATIONS (being explicit rather than silently approximate):
  - assessee_arguments/revenue_arguments in the source JSON are a flat
    per-case list, not attached to a specific issue - HAS_ARGUMENT (whose
    ontology domain is Issue, not CaseLaw) is therefore created from EVERY
    issue this case raises to EVERY argument of the matching side, which
    over-attaches when a case has multiple issues. Precise per-issue
    argument attribution would require either a richer extraction prompt
    (issue-scoped arguments) or a second LLM pass - left as a follow-up.
  - CaseCitation nodes store the citation string(s) largely as given
    (reporter_name = raw string) rather than fully parsed into reporter/
    volume/page/year - citation formats vary too much across a century of
    reports (1970 AIR 691 vs 2025:MHC:1145 vs a bare ITA number) to parse
    reliably with a fixed regex; safer to keep the verbatim string than to
    guess wrong structure.
  - Judge/Bench are stored as a plain string property on CaseLaw
    (bench data property), not split into individual Judge nodes - doable
    later if judge-level analytics become a real need.
  - Very long orders (the corpus's Grasim_Industries file is ~117 pages,
    ~60K words) are sent as ONE completion call - every provider this
    project supports (Claude Haiku 4.5, Gemini 1.5 Flash, GPT-5 nano) has
    an input context window well above that in tokens, so this works for
    every file in the current corpus, but --max-chars exists as an
    explicit safety rail (with a loud warning, not a silent truncation)
    for a future document that doesn't fit.

Also emits Chunk records (see build_chunks_and_link() below) for each case's
facts/findings/arguments/legal-principles, in the same shape
chunking.py/02_extract_entities.py produce for the Act - so 03_build_graph.py
and 05_embed_chunks.py work against case law completely unmodified, and
case-law text becomes just as vector-searchable as the Act's provisions.

Usage:
    # Single case (writes _graph.json AND _chunks.json alongside --out):
    python 11_extract_case_law.py --md ../data/case_law_md/Reckitt_Benckiser_Ahmedabad.md \\
        --out ../data/case_law_graph/Reckitt_Benckiser_Ahmedabad_graph.json

    # Every case in a folder (writes <stem>_graph.json + <stem>_chunks.json per case):
    python 11_extract_case_law.py --dir ../data/case_law_md --out-dir ../data/case_law_graph

    # Then load into the same graph 03_build_graph.py already builds - note
    # --chunks is explicit here (unlike the Act, case law's chunks file
    # isn't literally named "chunks.json", so 03's default same-directory
    # lookup won't find it on its own):
    python 03_build_graph.py --in ../data/case_law_graph/Reckitt_Benckiser_Ahmedabad_graph.json \\
        --chunks ../data/case_law_graph/Reckitt_Benckiser_Ahmedabad_chunks.json --mode base

    # Then embed those chunks so they're vector-searchable:
    python 05_embed_chunks.py --chunks ../data/case_law_graph/Reckitt_Benckiser_Ahmedabad_chunks.json
"""
import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", ".."))

import chunking  # same directory (domain_packs/tax/converters/)
from framework.core.infra import llm_client
from framework.core.infra import step_logger
from framework.core.ingest import provenance
from framework.core.retrieve import observability

SCRIPT_DIR = Path(__file__).resolve().parent

# --- The extraction prompt (verbatim from the project's case-law extraction
# spec) ------------------------------------------------------------------

EXTRACTION_INSTRUCTIONS = """You are a tax law analyst extracting structured data from an Indian income-tax case judgment/order.

CRITICAL RULES:
- Read the FULL text provided — do not stop early
- Extract the ACTUAL reasoning from the judgment text, not summaries
- For 'findings.reasoning': extract the specific legal arguments the Tribunal relied on
  e.g. "The inter-unit balance between the treasury undertaking and the parent was treated
  as a liability; since this was not transferred to the resulting company, the condition
  under Section 2(19AA)(ii) was not satisfied."
- For 'assessee_arguments' and 'revenue_arguments': extract the ACTUAL arguments made
- issue_outcomes labels MUST exactly match issues array labels — copy them verbatim
- Distinguish 'precedents_overruled' from 'precedents_distinguished': a case is OVERRULED only
  when this judgment expressly states an earlier decision is wrong, no longer good law, or is
  reversed/set aside ON THE SUBSTANTIVE LEGAL QUESTION. A case that is merely "set aside and
  remanded/remitted" for fresh consideration WITHOUT the court ruling on the merits is NOT
  overruled — leave it out of 'precedents_overruled' (it still belongs in 'precedents_referred'
  or 'precedents_distinguished' if discussed). When in doubt, prefer 'precedents_distinguished'
  or 'precedents_referred' over 'precedents_overruled' — do not guess.
- 'precedents_reversed' is NARROWER than 'precedents_overruled': use it ONLY when THIS judgment
  is itself an appellate decision directly reversing the SAME case in the SAME proceeding on
  appeal (e.g. an ITAT order under appeal being reversed by the High Court in that very appeal) —
  a direct procedural reversal in the litigation's own chain, not a later, unrelated judgment
  disagreeing with a precedent's reasoning (that is 'precedents_overruled'). If this judgment is
  not itself an appeal of the named case, do not use 'precedents_reversed' even if the outcome
  was to disagree with it — use 'precedents_overruled' or 'precedents_distinguished' instead.
- 'precedents_approved' captures cases this judgment EXPRESSLY endorses, affirms, or approves the
  reasoning of (e.g. "we respectfully approve the view taken in X", "the reasoning in X is
  correct and is affirmed") — stronger than merely following or referring to a case. Do not use
  it for cases that are merely cited without an express approving statement — use
  'precedents_followed' or 'precedents_referred' for those. When in doubt, do not guess.

Return ONLY valid JSON, no markdown, no backticks.

{
  "citation": "Full case name exactly as it appears",
  "court": "e.g. Ahmedabad ITAT",
  "bench": "Names of judges/members",
  "date": "Date of judgment as written",
  "assessment_year": "e.g. 2011-12",
  "binding_scope": "itat_division_bench",
  "importance": "landmark / leading / persuasive / routine",
  "sections_interpreted": [
    {"number": "2(19AA)", "act": "Income-tax Act 1961"}
  ],
  "sections_applied": [
    {"number": "45", "act": "Income-tax Act 1961"}
  ],
  "sections_referred": [
    {"number": "14A", "act": "Income-tax Act 1961"}
  ],
  "issues": ["Exact label — MUST match issue_outcomes labels verbatim"],
  "questions_of_law": ["The exact legal questions framed by the court"],
  "facts": "3-4 sentence summary of key facts",
  "assessee_arguments": ["Actual arguments made by the assessee's counsel"],
  "revenue_arguments": ["Actual arguments made by the revenue's counsel"],
  "findings": [
    {
      "issue": "EXACT same label as in issues array",
      "holding": "What was decided",
      "reasoning": "The SPECIFIC legal reasoning — cite the actual conditions, provisions, and facts relied on"
    }
  ],
  "legal_principles": ["Generalisable principles this case establishes"],
  "precedents_followed": ["Case names followed"],
  "precedents_distinguished": ["Case names distinguished"],
  "precedents_referred": ["Case names referred to"],
  "precedents_overruled": ["Case names this judgment expressly overrules/reverses on the merits - see CRITICAL RULES above; leave empty if none"],
  "precedents_reversed": ["Case names THIS judgment directly reverses on appeal in the SAME proceeding - see CRITICAL RULES above; leave empty if none"],
  "precedents_approved": ["Case names THIS judgment expressly approves/affirms the reasoning of - see CRITICAL RULES above; leave empty if none"],
  "statutes_cited": [{"name": "Income-tax Act 1961", "section": "2(19AA)"}],
  "treaty_references": [{"treaty": "India-Mauritius DTAA", "article": "13"}],
  "keywords": ["5-10 search terms"],
  "issue_outcomes": [
    {"issue": "EXACT same label as in issues array", "winner": "assessee or revenue"}
  ],
  "final_outcome": "e.g. Appeal dismissed"
}

JUDGMENT TEXT (read fully):
{text}"""

DEFAULT_MAX_CHARS = 700_000  # generous - see module docstring's "very long orders" note
# Rich schema (issues/findings/arguments/citations/concepts) needs real
# headroom beyond llm_client's 1000 default. Raised from 6000 -> 12000:
# 6000 was cutting off mid-JSON for longer/more complex real orders (a
# multi-issue tribunal judgment easily produces several KB of structured
# output), which silently surfaced as "could not parse model output as
# JSON" with no indication WHY - see the diagnostic logging in process_one()
# below, which would show a JSON string ending abruptly (no closing brace)
# exactly when this was the cause.
EXTRACTION_MAX_TOKENS = 12_000


def call_extraction(judgment_text: str, max_chars: int, trace=None, name: str = "case_law_extraction") -> tuple[str, bool]:
    truncated = False
    if len(judgment_text) > max_chars:
        judgment_text = judgment_text[:max_chars]
        truncated = True
    prompt = EXTRACTION_INSTRUCTIONS.replace("{text}", judgment_text)
    raw = llm_client.complete(
        system="You are a precise legal-document data-extraction assistant. Follow the user's "
               "instructions exactly and return only the requested JSON.",
        user_content=prompt,
        max_tokens=EXTRACTION_MAX_TOKENS,
        trace=trace, name=name, purpose="ingestion",
    ).strip()
    return raw, truncated


def parse_llm_json(raw: str) -> dict | None:
    """Same defensive cleanup as 02_extract_entities.py's extract_semantic_layer()
    - strip a markdown fence if the model added one despite being told not
    to, then json.loads(). Returns None (not a crash) on failure, exactly
    like 02, so one bad case in a batch doesn't kill the whole run."""
    cleaned = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    if not cleaned:
        return None
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


# --- Section-citation resolution: map "2(19AA)" -> chunking.py's own
# Section_2 / SubSection_2_19AA id convention, so INTERPRETS edges land on
# the SAME nodes the Act's own graph uses. ----------------------------------
#
# 2026-08-27: DEFAULT_CITED_ACT_NAMESPACE - every judgment in the currently-
# ingested case-law corpus was decided under the Income-tax Act, 1961, NOT
# the brand-new Income-tax Act, 2025 that also gets ingested (as "the Act").
# A section NUMBER means something different under each Act - e.g. old-Act
# Section 90 (DTAA/treaty override, the section AT ISSUE in Union of India v.
# Azadi Bachao Andolan) was renumbered to new-Act Section 159; new-Act
# Section 90 is an unrelated capital-gains definitional provision - see
# Ontology_Legal_Review_2026-08-27.md for the full analysis. Before this,
# resolve_section_citation() had zero Act-awareness and would silently wire
# a case's INTERPRETS edge to whatever node happened to share that number in
# the CURRENTLY-ingested Act - wrong more often than not for a treaty/GAAR-
# heavy corpus like this one. Resolving against the OLD Act's namespace by
# default is the correct behavior for THIS corpus; if case law decided under
# the 2025 Act (or some other Act) is ever ingested, override via
# --cited-act-namespace rather than changing this default, since old-Act
# case law will keep being the common case for the foreseeable future.
DEFAULT_CITED_ACT_NAMESPACE = "ITA1961"

SECTION_CITATION_RE = re.compile(r'^\s*(\d+[A-Z]{0,3})((?:\s*\([^)]+\)){0,3})\s*$', re.IGNORECASE)


def resolve_section_citation(number: str, namespace: str | None = None) -> tuple[str | None, dict]:
    """Returns (target_entity_id, parsed_parts). parsed_parts has
    section/subsection/clause/sub_clause, matching chunking.py's own
    generate_entity_id()/generate_full_citation() signature - reused
    directly (not re-implemented) for guaranteed id consistency with
    whatever chunking.py already built for the Act.

    `namespace` - which Act's id-namespace to resolve this citation
    against (see DEFAULT_CITED_ACT_NAMESPACE above for why this defaults
    to the OLD Act for this corpus, not whatever Act happens to be
    currently-ingested)."""
    m = SECTION_CITATION_RE.match(number.strip())
    if not m:
        return None, {}
    section = m.group(1)
    parts = re.findall(r'\(([^)]+)\)', m.group(2) or "")
    subsection = parts[0] if len(parts) > 0 else None
    clause = parts[1] if len(parts) > 1 else None
    sub_clause = parts[2] if len(parts) > 2 else None
    entity_id = chunking.generate_entity_id(section, subsection, clause, sub_clause, namespace=namespace)
    return entity_id, {
        "section": section, "subsection": subsection, "clause": clause, "sub_clause": sub_clause,
    }


# --- JSON -> graph entities/relationships -----------------------------------

def _slug(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')[:60]


# forum_level - a deterministic, non-LLM classification of `court` (free text
# like "Ahmedabad ITAT", "Bombay High Court", "Supreme Court of India", "AAR")
# into a normalized tier, so precedential-value questions ("what did the
# Tribunal hold" vs "what did the High Court hold") can filter on ONE
# consistent property instead of matching every possible free-text court name
# variant. Order matters - check the more specific "Supreme Court" before the
# generic "Court" substring various tiers all contain.
_FORUM_LEVEL_PATTERNS = [
    (re.compile(r'supreme court', re.IGNORECASE), "Supreme Court"),
    (re.compile(r'high court', re.IGNORECASE), "High Court"),
    (re.compile(r'\bitat\b|income.?tax appellate tribunal|\btribunal\b', re.IGNORECASE), "Tribunal"),
    (re.compile(r'\baar\b|authority for advance ruling', re.IGNORECASE), "AAR"),
    (re.compile(r'commissioner.*appeals|\bcit\(a\)\b', re.IGNORECASE), "Commissioner (Appeals)"),
]


def classify_forum_level(court: str) -> str:
    if not court:
        return ""
    for pattern, level in _FORUM_LEVEL_PATTERNS:
        if pattern.search(court):
            return level
    return ""


# jurisdiction_scope (added 2026-08-27): the `jurisdiction` property on
# CaseLaw was being set to the raw `court` string verbatim (e.g. "Bombay
# High Court") - identical to the `court` property itself, so it carried no
# extra information despite the name suggesting a distinct concept (WHERE a
# ruling binds, not WHO issued it). A Bombay High Court ruling binds courts/
# authorities within Maharashtra & Goa; it's persuasive-only elsewhere - a
# real distinction for a legal research tool deciding how strongly to rely
# on a holding for an assessee outside that territory. Deterministic
# (forum_level + a High Court name -> territory lookup), no extra LLM call,
# same pattern as classify_forum_level(). Left "" (unset) rather than
# guessing when the court name doesn't match a known High Court - better to
# have no value than a wrong one for something this legally consequential.
_HIGH_COURT_JURISDICTION = [
    (re.compile(r'bombay', re.IGNORECASE), "Maharashtra & Goa (Bombay High Court)"),
    (re.compile(r'delhi', re.IGNORECASE), "NCT of Delhi (Delhi High Court)"),
    (re.compile(r'madras', re.IGNORECASE), "Tamil Nadu & Puducherry (Madras High Court)"),
    (re.compile(r'calcutta', re.IGNORECASE), "West Bengal & Andaman and Nicobar Islands (Calcutta High Court)"),
    (re.compile(r'karnataka', re.IGNORECASE), "Karnataka (Karnataka High Court)"),
    (re.compile(r'kerala', re.IGNORECASE), "Kerala & Lakshadweep (Kerala High Court)"),
    (re.compile(r'gujarat', re.IGNORECASE), "Gujarat (Gujarat High Court)"),
    (re.compile(r'punjab.*haryana', re.IGNORECASE), "Punjab, Haryana & Chandigarh (Punjab & Haryana High Court)"),
    (re.compile(r'allahabad', re.IGNORECASE), "Uttar Pradesh (Allahabad High Court)"),
    (re.compile(r'andhra pradesh|telangana', re.IGNORECASE), "Andhra Pradesh & Telangana High Court"),
    (re.compile(r'rajasthan', re.IGNORECASE), "Rajasthan (Rajasthan High Court)"),
    (re.compile(r'madhya pradesh', re.IGNORECASE), "Madhya Pradesh (Madhya Pradesh High Court)"),
    (re.compile(r'gauhati', re.IGNORECASE), "Assam, Nagaland, Mizoram & Arunachal Pradesh (Gauhati High Court)"),
]


def derive_jurisdiction_scope(court: str, forum_level: str) -> str:
    if forum_level == "Supreme Court":
        return "All India"
    if forum_level == "High Court":
        for pattern, territory in _HIGH_COURT_JURISDICTION:
            if pattern.search(court or ""):
                return territory
        return ""  # unrecognized High Court name - leave unset rather than guess
    # Tribunal/AAR/Commissioner(Appeals): binding scope is typically limited
    # to the specific bench/applicant, not a clean territory mapping the way
    # High Courts have - left unset (forum_level + court together already
    # tell a retrieval consumer as much as can be said deterministically).
    return ""


# stance / stanceBasis (added 2026-08-28, see Graph_First_Retrieval_Design.md
# Section 2.1): whether this case's outcome, taken as a whole, favors the
# taxpayer/assessee or the Revenue - the property the "for vs. against"
# case-law retrieval fix filters on directly instead of hoping vector
# similarity happens to surface both sides.
#
# DELIBERATELY derived from `issue_outcomes[].winner` - a field the
# extraction prompt already produces and that CRITICAL RULES above already
# holds to a tight, verbatim-matched standard - rather than asking the model
# for a fresh "stance" judgment call. Same "derive from already-extracted
# structured fields, don't have the model self-report a second, possibly
# inconsistent opinion" convention as classify_forum_level()/
# derive_jurisdiction_scope() just above. Left "unknown" (not guessed) when
# no issue_outcomes with a recognized winner exist - matches this file's
# existing "no value beats a wrong one" convention (see
# derive_jurisdiction_scope()'s docstring).
def derive_stance(extraction: dict) -> tuple[str, str]:
    issue_outcomes = extraction.get("issue_outcomes", []) or []
    winners = [(o or {}).get("winner", "").strip().lower() for o in issue_outcomes]
    winners = [w for w in winners if w in ("assessee", "revenue")]
    if not winners:
        return "unknown", "No issue_outcomes with a recognized winner ('assessee'/'revenue') were extracted."
    n_assessee = winners.count("assessee")
    n_revenue = winners.count("revenue")
    if n_assessee and n_revenue:
        stance = "mixed"
    elif n_assessee:
        stance = "taxpayer_favorable"
    else:
        stance = "revenue_favorable"
    basis = (f"{n_assessee} issue(s) decided for assessee, {n_revenue} for revenue "
             f"(of {len(winners)} issue(s) with a recognized winner).")
    return stance, basis


# retrievalSummary / retrievalKeywords (added 2026-08-28, see
# Graph_First_Retrieval_Design.md Section 2.3): the text embedded into the
# new CaseLaw-level metadata vector index used to find ENTRY POINT nodes for
# graph traversal (see framework/core/retrieve/case_law_entry_points.py) -
# distinct from the existing Chunk-level embedding used by the general
# hybrid pipeline.
#
# Composed DETERMINISTICALLY from fields the extraction prompt already
# produced (citation/court/each finding's issue+holding/final_outcome for
# the summary; keywords/section numbers/legal_principles for the keywords) -
# deliberately NOT a fresh LLM-authored abstract. A thin or generic
# LLM-written summary would degrade entry-point recall below what today's
# full chunk-text embedding gives (see the evaluation in
# Graph_First_Retrieval_Design.md of the metadata-only retrieval design) -
# reusing already-extracted, already-reviewed structured fields avoids that
# risk entirely rather than trading it for a new one.
def derive_retrieval_summary(extraction: dict, citation: str) -> str:
    parts = []
    court = (extraction.get("court") or "").strip()
    if citation or court:
        parts.append(f"{citation} ({court})." if court else f"{citation}.")
    for finding in extraction.get("findings", []) or []:
        issue = (finding or {}).get("issue", "").strip()
        holding = (finding or {}).get("holding", "").strip()
        if issue and holding:
            parts.append(f"{issue}: {holding}")
        elif holding:
            parts.append(holding)
    outcome = (extraction.get("final_outcome") or "").strip()
    if outcome:
        parts.append(f"Final outcome: {outcome}.")
    return " ".join(p for p in parts if p).strip()


def derive_retrieval_keywords(extraction: dict) -> str:
    keywords: list[str] = []

    def add(term: str):
        term = (term or "").strip()
        if term and term.lower() not in (k.lower() for k in keywords):
            keywords.append(term)

    for kw in extraction.get("keywords", []) or []:
        add(kw)
    for field in ("sections_interpreted", "sections_applied", "sections_referred"):
        for entry in extraction.get(field, []) or []:
            if isinstance(entry, dict):
                add(entry.get("number", ""))
    for s in extraction.get("statutes_cited", []) or []:
        if isinstance(s, dict) and s.get("section"):
            add(str(s["section"]))
    for principle in extraction.get("legal_principles", []) or []:
        add(principle)
    return ", ".join(keywords)


def build_graph_from_extraction(extraction: dict, source_file: str, file_type: str = "caselaw_pdf",
                                 doc_meta: dict | None = None, source_md_path: str | None = None) -> dict:
    entities: list[dict] = []
    relationships: list[dict] = []
    ambiguities: list[dict] = []

    citation = (extraction.get("citation") or Path(source_file).stem).strip()
    case_id = f"CaseLaw_{_slug(citation)}"
    run_timestamp = datetime.now(timezone.utc).isoformat()

    # Document-level provenance (document_id/document_name/source_type/
    # authority_level/jurisdiction) - passed through from pipeline.py's
    # doc_meta (same convention extract_entities.py's attach_pipeline_
    # metadata() already uses for the Act path). Previously this script
    # never received or set these fields at all, so every CaseLaw/Issue/
    # Conclusion/Argument/LegalConcept entity - and every Chunk linked to
    # one - had no document_name for GraphRAGRetriever.get_document_details()
    # to resolve, and citations in generated answers fell back to raw
    # internal chunk_id strings instead of "<document> - Page X". Defaults
    # below mirror pipeline.py's FILE_TYPES["caselaw_pdf"]["defaults"] so a
    # direct CLI invocation (no doc_meta) still gets sensible values instead
    # of blank ones.
    doc_meta = doc_meta or {}
    doc_id = doc_meta.get("document_id") or case_id
    doc_name = doc_meta.get("document_name") or citation or source_file
    source_type = doc_meta.get("source_type") or "CaseLaw"
    authority_level = doc_meta.get("authority_level") or "Judicial"
    doc_jurisdiction = doc_meta.get("jurisdiction") or "India"
    # Which Act's id-namespace this judgment's section citations resolve
    # against - see DEFAULT_CITED_ACT_NAMESPACE's docstring above.
    cited_act_namespace = doc_meta.get("cited_act_namespace") or DEFAULT_CITED_ACT_NAMESPACE
    _forum_level = classify_forum_level(extraction.get("court", ""))
    _stance, _stance_basis = derive_stance(extraction)

    case_props = {
        "case_name": citation,
        "full_citation": citation,
        "hasText": extraction.get("facts", ""),
        "bench": extraction.get("bench", ""),
        "judgment_date": extraction.get("date", ""),
        # A case's holding is citable/applicable FROM its judgment date -
        # mapped onto the same effective_from the Act/generic-document
        # paths use, so temporal/"as of <date>" questions and
        # apply_metadata_filters()'s as_of parameter also work against case
        # law, not just statutory provisions. Left unset (not "") when the
        # LLM found no date, matching apply_metadata_filters()'s existing
        # "unset = no known constraint" semantics.
        **({"effective_from": extraction["date"].strip()} if extraction.get("date", "").strip() else {}),
        "assessment_year": extraction.get("assessment_year", ""),
        "jurisdiction": extraction.get("court", ""),
        "court": extraction.get("court", ""),
        "precedential_value": extraction.get("importance", ""),
        "binding_scope": extraction.get("binding_scope", ""),
        "holding": extraction.get("final_outcome", ""),
        "final_outcome": extraction.get("final_outcome", ""),
        "keywords": extraction.get("keywords", []),
        "questions_of_law": extraction.get("questions_of_law", []),
        "statutes_cited": extraction.get("statutes_cited", []),
        # --- retrieval metadata (reuses fields the extraction prompt already
        # produces - no extra LLM calls needed) ---
        # legal_concepts: reuse the existing "keywords" search-term list rather
        # than asking the model to generate a second, near-duplicate topical
        # tag list - keeps this consistent with the Act/generic paths' field
        # name so a question can filter on ONE property across all file types.
        "legal_concepts": extraction.get("keywords", []),
        # cases_cited: precedents this judgment discusses, regardless of how it
        # treated them - covers "analyse with reference to Grasim Industries
        # and Reckitt Benckiser Healthcare India" style questions.
        "cases_cited": sorted(set(
            extraction.get("precedents_followed", [])
            + extraction.get("precedents_distinguished", [])
            + extraction.get("precedents_referred", [])
            + extraction.get("precedents_overruled", [])
            + extraction.get("precedents_reversed", [])
            + extraction.get("precedents_approved", [])
        )),
        # precedents_overruled: a DISTINCT property (not folded only into the
        # undifferentiated cases_cited above) so "which decisions has this
        # case overruled" questions can filter on this property directly,
        # without a Cypher query having to somehow separate "overruled" back
        # out of a merged followed+distinguished+referred+overruled list.
        # Also backed by a real OVERRULES/OVERRULED_BY graph relationship
        # below (the ontology already declares this pair; it was simply
        # never populated - see the stub-node relationship loop).
        "precedents_overruled": sorted(set(extraction.get("precedents_overruled", []) or [])),
        # precedents_reversed / precedents_approved: 2026-09-03 NEW, same
        # distinct-property rationale as precedents_overruled above - backed
        # by real REVERSES/REVERSED_BY (already declared in the ontology but
        # never populated, same gap as OVERRULES was) and the newly-added
        # APPROVES/APPROVED_BY graph relationships - see the stub-node loop.
        "precedents_reversed": sorted(set(extraction.get("precedents_reversed", []) or [])),
        "precedents_approved": sorted(set(extraction.get("precedents_approved", []) or [])),
        # other_act_references: any statute cited under sections_interpreted/
        # applied/referred whose "act" isn't an Income-tax Act - e.g. Companies
        # Act 2013 Section 233 fast-track-demerger questions.
        "other_act_references": sorted(set(
            f"{s['act']} Section {s['number']}"
            for s in (
                extraction.get("sections_interpreted", [])
                + extraction.get("sections_applied", [])
                + extraction.get("sections_referred", [])
            )
            if s.get("act") and s.get("number") and "income-tax act" not in s["act"].lower()
        )),
        "treaty_references": sorted(set(
            f"{t['treaty']} Article {t['article']}"
            for t in extraction.get("treaty_references", [])
            if t.get("treaty") and t.get("article")
        )),
        "forum_level": _forum_level,
        "jurisdiction_scope": derive_jurisdiction_scope(extraction.get("court", ""), _forum_level),
        # stance/stanceBasis/retrievalSummary/retrievalKeywords - see the
        # derive_*() functions above and Graph_First_Retrieval_Design.md
        # Section 2. All deterministically composed from fields already in
        # `extraction`, not new LLM calls.
        "stance": _stance,
        "stanceBasis": _stance_basis,
        "retrievalSummary": derive_retrieval_summary(extraction, citation),
        "retrievalKeywords": derive_retrieval_keywords(extraction),
        "confidence": 0.8,  # LLM-extracted, not deterministically parsed - see chunking.py's Chunk for the contrast
        "extraction_notes": f"Extracted from {source_file} via 11_extract_case_law.py",
        "source_document": source_file,
        "ingestion_date": run_timestamp,
        "node_id": case_id,
        "content_hash": hashlib.sha256(citation.encode("utf-8")).hexdigest()[:16],
    }
    entities.append({"id": case_id, "type": "CaseLaw", "properties": case_props})

    # --- sections_interpreted / sections_applied / sections_referred:
    # all map to INTERPRETS (CaseLaw's only ontology-defined relationship to
    # a statutory provision), disambiguated by a `relation_type` property -
    # the ontology has no separate "applies"/"refers" property for CaseLaw
    # -> Section, so this reuses the one that exists rather than inventing
    # an undeclared relationship type.
    for field, relation_type in [
        ("sections_interpreted", "interpreted"),
        ("sections_applied", "applied"),
        ("sections_referred", "referred"),
    ]:
        for entry in extraction.get(field, []) or []:
            number = (entry or {}).get("number", "") if isinstance(entry, dict) else str(entry)
            if not number:
                continue
            target_id, parts = resolve_section_citation(number, namespace=cited_act_namespace)
            if not target_id:
                ambiguities.append({
                    "type": "unresolved_section_citation", "case": case_id,
                    "raw": number, "field": field,
                })
                continue
            relationships.append({
                "type": "INTERPRETS", "source": case_id, "target": target_id,
                "properties": {"relation_type": relation_type, "act": (entry or {}).get("act", "") if isinstance(entry, dict) else "",
                               "cited_as": number},
            })

    # --- issues / issue_outcomes / findings -> Issue + Conclusion entities
    #
    # Every Issue node built below also gets an INTERPRETS edge back to this
    # CaseLaw (same relationship type used for sections_interpreted/applied/
    # referred above - the ontology's INTERPRETS/INTERPRETED_BY pair already
    # declares CaseLaw -> Issue as a valid target, disambiguated from the
    # StatutoryProvision case by the target node's own label, same as
    # relation_type disambiguates "interpreted"/"applied"/"referred").
    # Without this edge an Issue was only reachable by string-matching the
    # "_to_<case_id>" suffix in its id - no actual graph path connected it
    # to the case that raised it, so KG Explorer / doubleClick hierarchy
    # expansion / any Cypher traversal starting from a CaseLaw could never
    # find its own Issues. Fixes a real ontology-vs-ingestion gap found
    # while reviewing the schema.
    issue_label_to_id = {}
    for idx, label in enumerate(extraction.get("issues", []) or [], start=1):
        issue_id = f"Issue_{idx}_to_{case_id}"
        issue_label_to_id[label] = issue_id
        entities.append({"id": issue_id, "type": "Issue", "properties": {
            "hasText": label, "full_citation": f"Issue {idx} in {citation}",
            "confidence": 0.8, "extraction_notes": "",
        }})
        relationships.append({
            "type": "INTERPRETS", "source": case_id, "target": issue_id,
            "properties": {"relation_type": "raises_issue"},
        })

    outcome_by_label = {
        (o or {}).get("issue", ""): (o or {}).get("winner", "")
        for o in (extraction.get("issue_outcomes", []) or [])
    }
    for idx, finding in enumerate(extraction.get("findings", []) or [], start=1):
        label = (finding or {}).get("issue", "")
        issue_id = issue_label_to_id.get(label)
        if issue_id is None:
            ambiguities.append({
                "type": "finding_issue_label_mismatch", "case": case_id,
                "note": f"findings[{idx}].issue={label!r} does not match any label in issues[]",
            })
            issue_id = f"Issue_{_slug(label) or idx}_to_{case_id}"
            entities.append({"id": issue_id, "type": "Issue", "properties": {
                "hasText": label, "full_citation": f"Issue (unmatched) in {citation}",
                "confidence": 0.5, "extraction_notes": "Label did not match issues[] verbatim - see ambiguities.",
            }})
            # Same INTERPRETS-back-to-case edge as the issues[] loop above -
            # this Issue was built here (not there) only because its label
            # didn't match verbatim, it's no less a real issue of the case.
            relationships.append({
                "type": "INTERPRETS", "source": case_id, "target": issue_id,
                "properties": {"relation_type": "raises_issue"},
            })
        conclusion_id = f"Conclusion_{idx}_to_{issue_id}"
        entities.append({"id": conclusion_id, "type": "Conclusion", "properties": {
            "hasText": finding.get("reasoning", ""), "holding": finding.get("holding", ""),
            "winner": outcome_by_label.get(label, ""),
            "full_citation": f"Finding on {label!r} in {citation}",
            "confidence": 0.8, "extraction_notes": "",
        }})
        relationships.append({"type": "HAS_CONCLUSION", "source": issue_id, "target": conclusion_id, "properties": {}})

    # --- assessee_arguments / revenue_arguments -> Argument entities,
    # attached to every issue this case raises (see module docstring's
    # "Known limitations" - the source JSON doesn't scope arguments to a
    # specific issue).
    for field, arg_type, side in [
        ("assessee_arguments", "TaxpayerArgument", "assessee"),
        ("revenue_arguments", "RevenueArgument", "revenue"),
    ]:
        for idx, arg_text in enumerate(extraction.get(field, []) or [], start=1):
            arg_id = f"Argument_{side}_{idx}_to_{case_id}"
            entities.append({"id": arg_id, "type": arg_type, "properties": {
                "hasText": arg_text, "full_citation": f"{side.title()} argument {idx} in {citation}",
                "confidence": 0.8, "extraction_notes": "",
            }})
            for issue_id in issue_label_to_id.values():
                relationships.append({"type": "HAS_ARGUMENT", "source": issue_id, "target": arg_id, "properties": {}})

    # --- precedents_followed / distinguished / referred -> stub CaseLaw
    # nodes (idempotent, same pattern as chunking.py's _ensure_amending_act)
    # + the matching case-to-case relationship.
    for field, rel_type in [
        ("precedents_followed", "FOLLOWS"),
        ("precedents_distinguished", "DISTINGUISHES_CASE"),
        ("precedents_referred", "RELIES_ON_CASE"),
        # OVERRULES/OVERRULED_BY is already declared in the ontology
        # (CaseLaw -> CaseLaw, same as FOLLOWS/FOLLOWED_BY) but was never
        # populated - this was a pure extraction/ingestion gap, not a
        # missing ontology definition. Enables "which High Court decisions
        # have been overruled and by what authority" style aggregate
        # queries via a reverse (n)-[:OVERRULES]->(m) traversal instead of
        # relying on the model to infer the relationship from free-text
        # cases_cited entries at answer-generation time.
        ("precedents_overruled", "OVERRULES"),
        # 2026-09-03 NEW - REVERSES/REVERSED_BY was already declared in the
        # ontology (same unpopulated-gap situation OVERRULES was in) and
        # APPROVES/APPROVED_BY was newly added to schema.json alongside this
        # change. See EXTRACTION_INSTRUCTIONS above for how each is scoped
        # narrower than precedents_overruled/precedents_followed respectively.
        ("precedents_reversed", "REVERSES"),
        ("precedents_approved", "APPROVES"),
    ]:
        for name in extraction.get(field, []) or []:
            if not name:
                continue
            other_id = f"CaseLaw_{_slug(name)}"
            # 2026-10-14: provenance.mark_stub() - cited_reference + an explicit
            # empty sourceMdPath (the stamping loop below would otherwise give
            # this stub THIS judgment's .md path via setdefault). graph_loader
            # never lets this stub overwrite an already-ingested judgment.
            entities.append({"id": other_id, "type": "CaseLaw", "properties": provenance.mark_stub({
                "case_name": name, "full_citation": name,
                "confidence": 0.3,  # stub - not yet ingested from its own judgment text
                "extraction_notes": f"Stub created from a precedent reference in {citation}; "
                                     f"will be filled in properly if/when this case is itself ingested.",
            }, cited_in=doc_id)})
            relationships.append({"type": rel_type, "source": case_id, "target": other_id, "properties": {}})
            # 2026-08-31: NEW - reify OVERRULES as a dated LegalEvent too (see
            # chunking.build_legal_event()'s docstring) - the event date is
            # THIS case's own judgment date (extraction["date"]), since this
            # case is the INSTRUMENT that did the overruling; `other_id` (the
            # older case) is the SUBJECT. OVERRULES and (2026-09-03, NEW)
            # REVERSES both get this - both are dated status-change events in
            # the LegalEvent class's own sense (amendment/overruling/reversal/
            # supersession/withdrawal/stay). FOLLOWS/DISTINGUISHES_CASE/
            # RELIES_ON_CASE/APPROVES are NOT reified this way - they record
            # how this judgment treated a precedent's reasoning, not a dated
            # change in that precedent's own legal status.
            if rel_type in ("OVERRULES", "REVERSES"):
                event_entity, event_rels = chunking.build_legal_event(
                    event_type="overruling" if rel_type == "OVERRULES" else "reversal",
                    event_date=extraction.get("date", ""),
                    instrument_id=case_id, instrument_label=citation,
                    subject_id=other_id, subject_label=name,
                )
                entities.append(event_entity)
                relationships.extend(event_rels)

    # --- legal_principles -> LegalConcept via INSTANTIATES_CONCEPT
    for idx, principle in enumerate(extraction.get("legal_principles", []) or [], start=1):
        concept_id = f"LegalConcept_{idx}_{_slug(principle)}"
        entities.append({"id": concept_id, "type": "LegalConcept", "properties": {
            "hasText": principle, "full_citation": principle[:80],
            "confidence": 0.7, "extraction_notes": "",
        }})
        relationships.append({"type": "INSTANTIATES_CONCEPT", "source": case_id, "target": concept_id, "properties": {}})

    # --- CaseCitation (verbatim - see module docstring on why this isn't
    # parsed further into reporter/volume/page/year).
    if citation:
        cit_id = f"CaseCitation_{_slug(citation)}"
        entities.append({"id": cit_id, "type": "CaseCitation", "properties": {
            "reporter_name": citation, "full_citation": citation,
            "confidence": 0.6, "extraction_notes": "Stored verbatim - see script docstring.",
        }})
        relationships.append({"type": "HAS_CITATION", "source": case_id, "target": cit_id, "properties": {}})

    # Stamp file_type + document-level provenance onto EVERY entity this run
    # produced (not just the CaseLaw node itself) - lets the "Filter by file
    # type" control (graphrag.py's apply_metadata_filters) narrow retrieval
    # to just case law, and lets GraphRAGRetriever.get_document_details()
    # resolve document_name/page for ANY chunk, regardless of which entity
    # type that chunk's entity_id points at (build_chunks_and_link() below
    # links "case_facts" chunks to the CaseLaw node itself, but "case_
    # finding"/"case_argument"/"legal_principle" chunks - the majority of a
    # judgment's embeddable text - link to Conclusion/Argument/LegalConcept
    # nodes instead, which would otherwise have no document_name of their
    # own for that lookup to find). setdefault throughout so a more specific
    # value already set (e.g. CaseLaw's own "jurisdiction": court) always
    # wins over these document-level fallbacks.
    for e in entities:
        e["properties"].setdefault("file_type", file_type)
        e["properties"].setdefault("document_id", doc_id)
        e["properties"].setdefault("document_name", doc_name)
        e["properties"].setdefault("source_type", source_type)
        e["properties"].setdefault("authority_level", authority_level)
        e["properties"].setdefault("jurisdiction", doc_jurisdiction)
        # 2026-08-28: NEW. See md_hydration.py / Semantic_Graph_Retrieval_
        # Architecture.md - lets retrieval read this case's FULL judgment
        # text live from its converted .md file at query time (case-law .md
        # files run well under 1MB in this corpus - see that doc for real
        # sizes - so reading the whole file per matched case is cheap and
        # sidesteps the fact that a Conclusion/Argument node's stored text is
        # an LLM-composed paraphrase, not a verbatim positional slice, so no
        # line_start/line_end makes sense for it the way it does for a
        # deterministically-parsed Act Section).
        e["properties"].setdefault("sourceMdPath", source_md_path or "")
        # 2026-10-14: everything not explicitly marked as a stub above was
        # extracted from THIS judgment's own text.
        e["properties"].setdefault("provenance", provenance.INGESTED_DOCUMENT)

    chunks = build_chunks_and_link(entities, case_id, citation, file_type)

    return {"entities": entities, "relationships": relationships, "ambiguities": ambiguities,
            "case_id": case_id, "chunks": chunks}


# --- Chunk records: case law's equivalent of chunking.py's Chunk objects ----
# chunking.py (the Act) produces embeddable text chunks as a first-class part
# of parsing the document's own structure. Case-law extraction above had NO
# equivalent - narrative text (facts, each finding's reasoning, each
# argument, each legal principle) lands only as a NODE PROPERTY
# (hasText/holding), with no Chunk node, no hasSourceChunk edge, and nothing
# for 05_embed_chunks.py to embed. That meant vector search could never find
# case-law text at all - a real gap, not a missing flag. This function closes
# it by deriving one Chunk record per meaningful piece of narrative text from
# the entities already built above, in the EXACT record shape
# 05_embed_chunks.py already expects (chunk_id/text/short_title/chunk_type/
# full_citation/entity_id) - so that script and 03_build_graph.py's
# load_chunk_nodes_and_links() work completely unmodified against case law
# too, no new plumbing needed in either.

def build_chunks_and_link(entities: list[dict], case_id: str, citation: str,
                           file_type: str = "caselaw_pdf") -> list[dict]:
    """
    Mutates `entities` IN PLACE to stamp `source_chunk_ids` onto each
    grounded entity - the same property 03_build_graph.py already reads to
    build hasSourceChunk edges for the Act, so no new linking logic is
    needed there either; it Just Works once these chunk records exist.

    chunk_id is prefixed `caselaw_<case_id>_<n>`, NOT a bare sequence
    number - this matters for correctness, not just style: 03_build_graph.py
    MERGEs Chunk nodes by chunk_id ALONE, with no per-document scoping, and
    the Act's own chunk_ids (from chunking.py/02_extract_entities.py) are
    plain integers as strings ("1".."4076"ish). A case-law chunk numbered
    "5" would silently MERGE onto the Act's unrelated chunk "5" - two
    completely different pieces of text landing on one Chunk node - if not
    namespaced. The `caselaw_<case_id>_` prefix guarantees no collision with
    the Act's chunk_ids, and no collision between two different cases either.
    """
    chunks: list[dict] = []
    n = 0

    def add_chunk(entity: dict, text: str, chunk_type: str, short_title: str):
        nonlocal n
        text = (text or "").strip()
        if not text:
            return
        n += 1
        chunk_id = f"caselaw_{case_id}_{n}"
        chunks.append({
            "chunk_id": chunk_id,
            "text": text,
            "short_title": short_title,
            "chunk_type": chunk_type,
            "element_type": chunk_type,
            "full_citation": f"{citation} — {short_title}",
            "entity_id": entity["id"],
            "file_type": file_type,
        })
        # Comma-separated, same convention chunking.py/02_extract_entities.py
        # already use for this property - almost always a single id here
        # (one chunk per entity), but appended rather than overwritten in
        # case a future change ever attaches more than one.
        existing = entity["properties"].get("source_chunk_ids", "")
        ids = [c for c in existing.split(",") if c] if existing else []
        ids.append(chunk_id)
        entity["properties"]["source_chunk_ids"] = ",".join(ids)

    for e in entities:
        if e["type"] == "CaseLaw" and e["id"] == case_id:
            add_chunk(e, e["properties"].get("hasText", ""), "case_facts", "Facts")
        elif e["type"] == "Conclusion":
            holding = e["properties"].get("holding", "")
            reasoning = e["properties"].get("hasText", "")
            combined = f"{holding}. {reasoning}" if holding else reasoning
            add_chunk(e, combined, "case_finding", e["properties"].get("full_citation", "Finding"))
        elif e["type"] in ("TaxpayerArgument", "RevenueArgument"):
            add_chunk(e, e["properties"].get("hasText", ""), "case_argument",
                      e["properties"].get("full_citation", "Argument"))
        elif e["type"] == "LegalConcept":
            add_chunk(e, e["properties"].get("hasText", ""), "legal_principle", "Legal principle")
        # Issue nodes deliberately excluded - hasText there is just the short
        # issue LABEL (already echoed inside each finding's short_title/
        # full_citation above), too short to be a useful independent
        # embedding target.

    return chunks


def process_one(md_path: Path, max_chars: int, debug: bool, trace=None,
                 file_type: str = "caselaw_pdf", doc_meta: dict | None = None) -> tuple[dict | None, dict | None, dict]:
    """Returns (raw_extraction_json, graph_dict, stats)."""
    text = md_path.read_text(encoding="utf-8")
    stats = {"source_chars": len(text)}

    with step_logger.Step(f"Extracting structured data from {md_path.name} via LLM"):
        raw, truncated = call_extraction(text, max_chars, trace=trace, name=f"case_law_extraction_{md_path.stem}")
        stats["truncated"] = truncated
        if truncated:
            step_logger.log(f"  WARNING: document exceeded --max-chars ({max_chars}) and was truncated - "
                             f"extraction may be incomplete. Raise --max-chars if your provider's context "
                             f"window allows it.")
        extraction = parse_llm_json(raw)
        if extraction is None:
            # Always logged via step_logger (not gated behind --debug) - this
            # is captured into the webapp's ingest-job log regardless, and
            # was previously the ONLY thing visible there on a parse
            # failure ("skipping this file", no detail), which is exactly
            # useless for actually diagnosing why. The most common real
            # cause is the model's JSON getting cut off mid-object because
            # EXTRACTION_MAX_TOKENS was too small for a long/complex
            # judgment - the ending snippet below makes that immediately
            # obvious (a truncated JSON string ends abruptly, not with a
            # closing brace).
            step_logger.log(f"  WARNING: could not parse model output as JSON (raw output was "
                             f"{len(raw)} chars) - skipping this file.")
            step_logger.log(f"  ----- start of raw output -----\n{raw[:600]}")
            step_logger.log(f"  ----- end of raw output -----\n{raw[-600:]}")
            if debug:
                print("----- RAW MODEL OUTPUT (first 1500 chars) -----")
                print(raw[:1500])
                print("----- END RAW OUTPUT -----")
            return None, None, stats

    with step_logger.Step("Mapping extraction onto graph entities/relationships"):
        graph = build_graph_from_extraction(extraction, md_path.name, file_type=file_type, doc_meta=doc_meta,
                                             source_md_path=str(md_path.resolve()))
        stats["entities"] = len(graph["entities"])
        stats["relationships"] = len(graph["relationships"])
        stats["ambiguities"] = len(graph["ambiguities"])
        stats["chunks"] = len(graph["chunks"])
        step_logger.log(f"  {stats['entities']} entities, {stats['relationships']} relationships, "
                         f"{stats['chunks']} chunks, {stats['ambiguities']} ambiguities flagged")

    return extraction, graph, stats


def validate_against_schema(graph: dict, schema: dict) -> list[str]:
    """Lightweight sanity check (mirrors the checks run at the end of the
    v3 ontology migration) - not a hard failure, just a printed warning for
    any entity type or relationship type this script produced that the
    ontology doesn't actually declare, so a naming drift is caught
    immediately rather than silently failing at 03_build_graph.py time."""
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
    parser.add_argument("--md", type=Path, default=None, help="Single case-law .md file")
    parser.add_argument("--out", type=Path, default=None, help="Output graph JSON path (with --md)")
    parser.add_argument("--dir", type=Path, default=None, help="Process every .md file in this folder")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output folder for graph JSON files (with --dir)")
    parser.add_argument("--raw-out-dir", type=Path, default=None,
                         help="Optional: also write the raw LLM extraction JSON here, one file per case "
                              "(for audit/debug - defaults to alongside the graph JSON if --dir is used)")
    parser.add_argument("--chunks-out", type=Path, default=None,
                         help="Output path for this case's embeddable Chunk records (with --md). Feed this "
                              "to 05_embed_chunks.py's --chunks and to 03_build_graph.py's --chunks so the "
                              "case's facts/findings/arguments/legal-principles become embedded, retrievable "
                              "Chunk nodes just like the Act's provisions. In --dir mode this defaults "
                              "automatically to <out-dir>/<stem>_chunks.json per case - no flag needed.")
    parser.add_argument("--schema", type=Path, default=SCRIPT_DIR.parent / "ontology" / "schema.json")
    parser.add_argument("--file-type", default="caselaw_pdf",
                         help="Stamped onto every entity/chunk this run produces as a `file_type` "
                              "property, so the webapp's 'Filter by file type' control can narrow "
                              "retrieval to it later.")
    # document-id/-name/source-type/authority-level/jurisdiction - same flag
    # names + purpose as extract_entities.py's (act path) and extract_generic.py's
    # equivalents: stamped onto every entity this run produces so citations in
    # generated answers can show a real document name + the retriever's
    # get_document_details() can resolve provenance for every case-law chunk
    # (see build_graph_from_extraction()'s doc_meta handling for why this was
    # previously missing here specifically). Defaults match pipeline.py's
    # FILE_TYPES["caselaw_pdf"]["defaults"] for a direct/manual CLI run.
    parser.add_argument("--document-id", default=None,
                         help="Defaults to this case's own CaseLaw node id if omitted.")
    parser.add_argument("--document-name", default=None,
                         help="Defaults to the extracted case citation if omitted.")
    parser.add_argument("--source-type", default="CaseLaw")
    parser.add_argument("--authority-level", default="Judicial")
    parser.add_argument("--jurisdiction", default="India")
    parser.add_argument("--cited-act-namespace", default=None,
                         help=f"Which Act's id-namespace this batch's section citations resolve against "
                              f"(see DEFAULT_CITED_ACT_NAMESPACE={DEFAULT_CITED_ACT_NAMESPACE!r} in this file's "
                              f"source for why it defaults to the OLD Act, not whichever Act is currently "
                              f"ingested). Override only if ingesting case law decided under a different Act.")
    parser.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    schema = None
    if args.schema.exists():
        with open(args.schema, "r", encoding="utf-8") as f:
            schema = json.load(f)

    # One Langfuse trace for this whole run (single file or a --dir batch) -
    # every case's extraction call nests under it, so a batch run shows up
    # as ONE trace containing all its LLM calls in Langfuse, not one
    # disconnected trace per case. No-op if LANGFUSE_* isn't set in .env.
    llm_trace = observability.start_trace(
        "extract_case_law_run", {"md": str(args.md) if args.md else None, "dir": str(args.dir) if args.dir else None}
    )

    # Only include document_id/document_name when explicitly passed - leaving
    # them unset lets build_graph_from_extraction() default document_id to
    # this case's own id and document_name to its extracted citation (the
    # right per-file behavior in --dir/batch mode; --document-id/--document-
    # name are mainly meant for the --md single-file, pipeline.py-driven path).
    doc_meta = {"source_type": args.source_type, "authority_level": args.authority_level,
                "jurisdiction": args.jurisdiction}
    if args.document_id:
        doc_meta["document_id"] = args.document_id
    if args.document_name:
        doc_meta["document_name"] = args.document_name
    if args.cited_act_namespace:
        doc_meta["cited_act_namespace"] = args.cited_act_namespace

    def handle(md_path: Path, out_path: Path, raw_out_path: Path | None, chunks_out_path: Path):
        extraction, graph, stats = process_one(md_path, args.max_chars, args.debug, trace=llm_trace,
                                                file_type=args.file_type, doc_meta=doc_meta)
        if graph is None:
            return
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"entities": graph["entities"], "relationships": graph["relationships"]}, f, indent=2, ensure_ascii=False)
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
        md_files = sorted(args.dir.glob("*.md"))
        if not md_files:
            raise SystemExit(f"No .md files found in {args.dir}")
        step_logger.log(f"Found {len(md_files)} case-law .md file(s) to process")
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
