"""
DomainConfig instance for the tax domain pack - every value here was pulled
DIRECTLY out of a specific tax_kg_poc module during this migration (see each
field's comment for the exact source). This is the object that turns
framework/core/ingest + framework/core/retrieve + framework/webapp into
"the Income Tax Act knowledge graph app" - swap this file (and prompts/,
ontology/, converters/) for a different domain and the same framework code
runs unchanged.
"""
import re

from framework.contracts.domain_config import CitationRules, DomainConfig
from framework.core.retrieve.category_retrieval import CategorySpec
from framework.core.ingest.entity_resolution import KeyProperties  # 2026-10-14: CaseLaw key normalizer
from domain_packs.tax.case_names import normalize_case_name, same_designations  # 2026-10-14: see RESOLUTION_KEY_PROPERTIES["CaseLaw"]

# NOTE on ontology version: domain_packs/tax/ontology/schema.json is generated
# from New_tax_ontology_v5.owl (../../../ontology/New_tax_ontology_v5.owl in
# the original tax_kg_poc tree - a much larger schema than v3/v4: 164 classes
# covering transfer pricing, litigation, firm knowledge management, etc.).
# v5 does NOT declare SubClause/AmendingAct/DefinedTerm as formal classes
# (v3/v4 did) - scripts/chunking.py and extract_entities.py's LLM prompt
# still mint nodes under those labels, same as before. This is intentionally
# tolerated, not a bug: Neo4j doesn't enforce the ontology schema, and this
# project already has established, deliberate non-ontology labels (Chunk,
# AuditRun, PiiAccessLog) for exactly this reason - a label absent from the
# OWL file is not the same as a label that's wrong.

# --- citation parsing (was webapp/retrieval.py's CITATION_RE / CITATION_QUERY_RE) ---
# "Section 55(2)(b)(iii)" anywhere in a free-text question.
_CITATION_FREE_TEXT_RE = re.compile(
    r"\bSection\s+(\d+[A-Z]{0,3})"       # Section 55
    r"((?:\s*\(\w{1,4}\))*)",            # optional (2)(b)(iii)... suffix
    re.IGNORECASE,
)
# Same grammar, anchored, "Section" optional - for the citation search box
# ("12A", "12A(1)", "Section 12A(1)" all resolve the same way).
_CITATION_QUERY_RE = re.compile(
    r"^\s*(?:section\s+)?(\d+[A-Z]{0,3})((?:\s*\(\w{1,4}\))*)\s*$",
    re.IGNORECASE,
)


def _build_citation(section: str, suffix: str) -> str:
    """Must match scripts/chunking.py's generate_full_citation() output
    format exactly, byte-for-byte - that's what's stored on Chunk.full_citation."""
    clean_suffix = re.sub(r"\s+", "", suffix or "")
    return f"Section {section.upper()}{clean_suffix}"


CITATION_RULES = CitationRules(
    free_text_pattern=_CITATION_FREE_TEXT_RE,
    query_pattern=_CITATION_QUERY_RE,
    build_citation=_build_citation,
)

# --- structural labels (was scripts/03_build_graph.py's STRUCTURAL_LABELS) ---
# GenericDocument/DocumentSection (from converters/extract_generic.py) are
# included too - their ids are just as deterministic
# (GenericDocument_<doc_id> / DocumentSection_<doc_id>_<n>) as the Act's,
# so cross-run fuzzy entity resolution should be skipped for them for the
# same reason.
#
# 2026-10-14 (phase0/review-integrity): read this set as "EXACT-ID IDENTITY
# labels" - every label whose id is built deterministically from the
# authority's own coordinates, so that a different id IS a different entity.
# Added after review/phase0 showed cross-run fuzzy resolution (id-fallback
# keys, no resolution_key_properties entry) auto-merging DIFFERENT entities
# (scores from entity_resolution.similarity on the real id shapes):
#   FirmOpinion        FirmOpinion_<opinion_id slug>      OP_01 vs OP_02      96.6
#   Rule               ITR1962__Rule_<n>[_<sub>...]       Rule_10 vs Rule_11  95.2
#   Article            Article_<n>_TaxTreaty_<country>    12 vs 13            96.9
#   TaxTreaty          TaxTreaty_<country slug>           Austria/Australia   96.4
#   TreatyNotification TreatyNotification_<c>_<i>_<no.>   _1_ vs _2_          96.2
#   LegalEvent         LegalEvent_<type>_<instr>_<subj>   differs by 1 digit  98.8
#   Act                <act namespace> (ITA1961 ...)      (root node, never fuzzy-joined)
# Each of these ids is minted identically by the real converter and by
# extract_opinion.py's cited_reference stubs (same f-strings - see
# extract_opinion.build_graph_from_opinion), so a stub still joins its real
# authority by exact id; it just can never be re-pointed at a NEIGHBOURING one.
# CBDTCircular is deliberately NOT here: its real-document id is a slug of
# the LLM-read number ("8/2026" -> CBDTCircular_8_2026) while stubs use
# CBDTCircular_8_of_2026, so it resolves on a canonicalised circular_number
# instead (RESOLUTION_KEY_PROPERTIES below) - with the numeric veto.
STRUCTURAL_LABELS = frozenset({
    "Section", "SubSection", "Clause", "SubClause", "Proviso", "Explanation",
    "GenericDocument", "DocumentSection",
    "FirmOpinion", "Rule", "Article", "TaxTreaty", "TreatyNotification", "LegalEvent", "Act",
})

# --- Chunk node metadata allowlists ------------------------------------------
# Was scripts/03_build_graph.py's CHUNK_METADATA_KEYS (graph_loader - excludes
# content/embedding, only 05/embedder writes those).
CHUNK_METADATA_KEYS = (
    "full_citation", "short_title", "chunk_type", "element_type",
    "section", "subsection", "clause", "sub_clause",
    "has_proviso", "has_explanation", "has_redacted_content",
    "content_length_tokens", "related_sections", "cross_references",
    "amendment_info", "line_start", "line_end", "chapter", "page_number", "file_type",
    # Retrieval pre-filter metadata (see attach_pipeline_metadata() /
    # EXTRACTION_SYSTEM_PROMPT's "RETRIEVAL METADATA" section in
    # extract_entities.py, and the equivalent additions in extract_case_law.py/
    # extract_generic.py) - lets a question narrow to the right nodes by legal
    # topic/cross-reference BEFORE vector search, instead of relying on
    # semantic similarity alone. Present on the ENTITY node unfiltered
    # regardless of this list (graph_loader's `SET n += $props`) - these keys
    # only control what also gets mirrored onto the separate lightweight
    # Chunk node used for citation/embedding.
    "legal_concepts", "statutes_cited", "other_act_references", "treaty_references",
    "cases_cited", "forum_level", "defines_terms",
)
# Was scripts/05_embed_chunks.py's CHUNK_METADATA_KEYS (embedder - additional
# fields alongside the content/embedding it always sets).
EMBED_CHUNK_METADATA_KEYS = (
    "full_citation", "short_title", "chunk_type", "element_type",
    "section", "subsection", "clause", "sub_clause", "entity_id",
    "has_proviso", "has_explanation", "has_redacted_content", "chapter",
    "page_number", "file_type",
    "legal_concepts", "statutes_cited", "other_act_references", "treaty_references",
    "cases_cited", "forum_level", "defines_terms",
)

# --- entity resolution (was scripts/entity_resolution.py's RESOLUTION_KEY_PROPERTIES) ---
#
# CRITICAL: every entity TYPE extracted by any converter (see chunking.py's
# entity emitters, extract_entities.py, extract_case_law.py) MUST have an
# entry here, or graph_loader.py's find_cross_graph_match() silently falls
# back to entity_resolution.resolution_key()'s default: normalize(f"{type}
# {fallback_id}") - i.e. the entity's OWN ID text becomes the fuzzy-match
# key. That's fine for types whose id embeds no risk of near-duplicate
# strings (e.g. a truly unique slug), but it's actively DANGEROUS for any
# type whose id is "mostly-identical text + a small varying index/suffix" -
# rapidfuzz's token_sort_ratio scores such ids as 95%+ similar to each
# other (only one short token differs among a dozen+ shared ones), which
# clears HIGH_SIMILARITY and triggers a confident AUTO-MERGE onto a
# COMPLETELY DIFFERENT real-world entity.
#
# This is exactly what was happening to Issue/Conclusion/TaxpayerArgument/
# RevenueArgument (case-law sub-entities, ids like
# "Issue_1_to_CaseLaw_<case>" / "Issue_8_to_CaseLaw_<case>" - identical
# except for the leading index) before these four were added below: ANY
# second ingestion of a case-law document (a re-run, a retry, or simply
# ingesting the same case a second time under a new doc_id) would silently
# cross-merge this run's Issue/Conclusion/Argument entities onto a PRIOR
# run's entities of the same type and case - overwriting real content with
# the wrong issue's content, and leaving most of the fresh extraction
# unreachable under its own id (confirmed via
# framework.cli.compare_graph_json against a real re-ingested case: 63 of
# 106 nodes and 442 of 484 relationships reported "missing" even though
# ingestion had just run, because they'd been silently remapped onto stale
# nodes from an earlier run instead of created under their own ids).
#
# Fix: give these types a resolution key built from actual SUBSTANTIVE
# content (hasText/holding), not the numerically-indexed id - two
# genuinely different issues/arguments now score far apart (their content
# differs), while the SAME real issue re-extracted in a later run (same
# text) still correctly auto-merges. full_citation is included too since it
# embeds the parent case name, giving extra protection against an
# accidental cross-CASE match (a similar-sounding issue in an unrelated
# case). LegalConcept/CaseCitation are lower-risk (their ids already embed
# distinguishing content, not just a bare index) but are listed too for the
# same reason: never leave a real extracted type to the id-based fallback.
RESOLUTION_KEY_PROPERTIES = {
    "Section": ["section_no"],
    "SubSection": ["section_no", "sub_section_no"],
    "Clause": ["section_no", "sub_section_no", "clause_id"],
    "SubClause": ["section_no", "sub_section_no", "clause_id", "sub_clause_id"],
    "Proviso": ["section_no", "sub_section_no", "clause_id"],
    "Explanation": ["section_no", "sub_section_no", "clause_id"],
    # 2026-10-14 (phase0/followups): same property, now compared through
    # case_names.normalize_case_name() (party names only; reporter citations,
    # years, court suffixes stripped; Pvt Ltd/Private Limited, CIT/
    # Commissioner of Income Tax, v./vs, M/s unified) so a cited-case stub
    # ("... Pvt Ltd v. CIT, (2021) 432 ITR 471 (SC)") resolves onto the
    # ingested judgment ("... Private Limited v. The Commissioner of Income
    # Tax, (2021) 2 SCC 1") - they scored ~75 before, below the 95 merge bar.
    # Key only: node ids are unchanged. Still a list (KeyProperties subclasses it).
    # same_designations vetoes a merge across different revenue officers
    # ("X v. ITO" vs "X v. CIT"), which the fuzzy ratio alone scores ~97.
    "CaseLaw": KeyProperties(["case_name"], normalizer=normalize_case_name, compatible=same_designations),
    "AmendingAct": ["amendingActNo", "amendingActYear"],
    "DefinedTerm": ["id"],
    "Issue": ["full_citation", "hasText"],
    "Conclusion": ["full_citation", "holding", "hasText"],
    "TaxpayerArgument": ["full_citation", "hasText"],
    "RevenueArgument": ["full_citation", "hasText"],
    "LegalConcept": ["full_citation", "hasText"],
    "CaseCitation": ["full_citation", "reporter_name"],
}

# 2026-10-14 (phase0/review-integrity): CBDT circular identity. Circulars had
# no entry above, so cross-run matching compared id text and an opinion's stub
# CBDTCircular_8_of_2026 auto-merged (97.1) onto the real CBDTCircular_3_of_2026,
# re-pointing the opinion's RELIES_ON_AUTHORITY edge at the wrong circular. A
# circular IS its number/year, so key on circular_number, canonicalised so the
# real document's LLM-read "8/2026" (id CBDTCircular_8_2026) and a stub's
# "8 of 2026" / "Circular No. 08/2026" produce the same key, and rely on
# key_guard()'s numeric veto (entity_resolution.same_number_tokens) so 8 can
# never join 3. Applies to all three circular labels extract_circular.py emits.
_CIRCULAR_NO_YEAR_RE = re.compile(r"(\d+)\s*(?:/|-|\bof\b)\s*((?:19|20)\d{2})\b", re.IGNORECASE)


def canonical_circular_number(text: str) -> str:
    """'Circular No. 08/2026' / '8 of 2026' / '8-2026' -> '8 of 2026'. Text
    without a recognisable number/year pair is returned unchanged (the generic
    normalize() + numeric veto still apply) rather than guessed at."""
    m = _CIRCULAR_NO_YEAR_RE.search(text or "")
    return f"{int(m.group(1))} of {m.group(2)}" if m else text


_CIRCULAR_KEY = KeyProperties(["circular_number"], normalizer=canonical_circular_number)
RESOLUTION_KEY_PROPERTIES.update({
    "CBDTCircular": _CIRCULAR_KEY,
    "ExplanatoryCircular": _CIRCULAR_KEY,
    "ExtensionCircular": _CIRCULAR_KEY,
})

# --- PII (was webapp/pii.py's ALWAYS_MASKED_FIELDS / REGEX_DETECTORS) --------
ALWAYS_MASKED_FIELDS = {
    "CaseLaw": ["case_name", "bench"],
    # 2026-10-14: + client_name/counterparty_name - converters/extract_opinion.py
    # (the first real FirmOpinion producer) stores the real party names ONLY in
    # these properties (text properties/chunks are pseudonymised at ingestion),
    # so masking them here is what keeps client identity out of the UI/answers.
    "FirmOpinion": ["partner_name", "client_industry", "client_name", "counterparty_name"],
}

PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")                       # Indian PAN: AAAAA9999A
AADHAAR_RE = re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b")             # Indian Aadhaar: 12 digits
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
PHONE_RE = re.compile(r"\b(?:\+?91[\s-]?)?[6-9]\d{9}\b")                 # Indian mobile numbers

REGEX_DETECTORS = (
    ("PAN", PAN_RE),
    ("AADHAAR", AADHAAR_RE),
    ("EMAIL", EMAIL_RE),
    ("PHONE", PHONE_RE),
)

# --- post-ingestion sanity checks (was scripts/04_query_graph.py's steps 3/4) ---
SAMPLE_RELATIONSHIP_CHECKS = (
    ("Section", "REFERS_TO", "Section", "Sample: Sections and what they refer to"),
    ("CaseLaw", "INTERPRETS", "Section", "Sample: Case law interpreting sections"),
)

# --- Tax Knowledge & Analysis Framework categories ---------------------------
# 2026-08-26: NEW. Added per user requirement: every /api/ask response must
# be organized under these six fixed categories, and retrieval must actually
# ATTEMPT every category on every question (graph first, vector-search
# fallback over the chunks if the graph has nothing typed for that category)
# rather than only relying on the general-purpose hybrid retrieval already
# run by GraphRAGRetriever.retrieve(). See
# framework/core/retrieve/category_retrieval.py for the generic mechanics
# this spec list drives.
#
# node_labels below mix REAL labels already produced by this pack's
# converters (Section/SubSection/Clause/SubClause/Proviso/Explanation/
# DefinedTerm/AmendingAct for the Act path, CaseLaw for the case-law path -
# confirmed against actual ingested data/graphs/*.json at the time this was
# written) with ASPIRATIONAL labels declared in ontology/schema.json but not
# yet produced by any converter (Circular/Notification/Rule/TaxTreaty and
# their subclasses, FirmOpinion, the ClientEngagementDocument family) - this
# is intentional, not a bug: today those categories will correctly report
# "not found" (nothing of that type has been ingested), and will start
# surfacing real content automatically the moment a converter/ingestion path
# for that document type is added, with no further code change needed here.
#
# text_properties are SCALAR-only per category_retrieval.CategorySpec's
# contract - for FirmOpinion/ClientEngagementDocument (nothing ingested yet)
# these are best-effort guesses following this pack's own established
# snake_case property-naming convention (see CaseLaw's real properties above
# and ALWAYS_MASKED_FIELDS's "partner_name"/"client_industry" below) rather
# than ontology/schema.json's OWL-style camelCase - update these once a real
# converter for that document type exists and its actual property names are
# known.
TAX_CATEGORY_SPECS = (
    CategorySpec(
        key="fact_sheet", label="Fact Sheet",
        description="Internal facts, facts of the matter, and the tax position/query to be determined - "
                     "drawn from the question itself, not retrieved from the graph.",
        synthesis_only=True,
    ),
    CategorySpec(
        key="law_regulatory_framework", label="Law & Regulatory Framework",
        description="Income Tax Act, Notifications, Circulars, Rules, Amended IT Act Notifications/"
                     "Circulars/Rules, and Tax Treaty provisions.",
        node_labels=(
            "Act", "Section", "SubSection", "Clause", "SubClause", "Chapter", "Schedule",
            "Proviso", "Explanation", "DefinedTerm", "DefinitionSection", "TransitionalProvision",
            "SunsetClause", "Annexure", "AmendingAct",
            "Circular", "CBDTCircular", "ExplanatoryCircular", "ExtensionCircular",
            "Notification", "EffectiveDateNotification", "SafeHarbourNotification", "TreatyNotification",
            "Rule", "IncomeTaxRules", "ComputationRules", "ReportingRules",
            "Order", "AdministrativeOrder", "Section119Order",
            "Instruction", "AssessmentInstruction", "TransferPricingInstruction",
            "TaxTreaty", "Article", "Protocol", "Paragraph", "Reservation", "MLIProvision",
        ),
        text_properties=("hasText", "full_citation", "marginal_heading", "notificationNumber", "short_title"),
        vector_query_prefix="Indian Income Tax Act provisions, CBDT circulars, notifications, rules, "
                             "or tax treaty articles relevant to: ",
    ),
    CategorySpec(
        key="case_law_hierarchy", label="Case Law Hierarchy",
        description="Favourable/unfavourable case law, Tribunal/High Court/Supreme Court decisions, "
                     "jurisdictional case law, and case law predating the relevant change in law.",
        node_labels=("CaseLaw",),
        text_properties=("case_name", "full_citation", "hasText", "holding", "final_outcome"),
        vector_query_prefix="Judicial case law (Tribunal, High Court, or Supreme Court decisions) relevant to: ",
        # 2026-08-28: NEW. See Graph_First_Retrieval_Design.md. `instructions`
        # is a direct response to a real evaluated answer under-covering and
        # mischaracterizing the Revenue-favorable side of a "for and against"
        # case-law question - retrieval now also enforces balance explicitly
        # (see category_retrieval.py's _case_law_hierarchy_retrieval()), but
        # generation still needs to be told not to undo that by dropping or
        # downplaying one side anyway.
        instructions=(
            "Search the evidence below for BOTH taxpayer-favorable AND Revenue-favorable authorities before "
            "writing - each case below is tagged with its stance where known. If the question asks for 'for "
            "and against' or otherwise invites comparison, present both sides explicitly, even if one side's "
            "evidence is thinner; do not silently drop it. Do not characterize a case as 'insufficient "
            "context' or reduce its holding to a merely procedural point unless the evidence below actually "
            "supports that - check the full text provided first."
        ),
        # Case-law Chunk text (see extract_case_law.py's build_chunks_and_link())
        # is already a complete composed unit (an issue's full holding+
        # reasoning, a full argument, a full legal principle) - NOT a raw
        # slice of a converted PDF the way Act/Circular/Treaty chunks are.
        # The framework default 800/2500-char caps were truncating these
        # more aggressively than necessary and were a plausible contributor
        # to at least one real "insufficient context" mischaracterization -
        # raised here specifically rather than for every category.
        max_chars_per_chunk=2000,
        max_chars_per_category=6000,
    ),
    CategorySpec(
        key="opinion_expert_views", label="Opinion & Expert Views",
        description="KPMG opinions - latest, and prior to the relevant change in law.",
        node_labels=("FirmOpinion",),
        text_properties=("opinion_conclusion", "analysis", "facts", "questions", "reliance_clauses"),
        vector_query_prefix="Prior KPMG tax opinions or expert technical positions relevant to: ",
    ),
    CategorySpec(
        key="client_specific_data", label="Client Specific Data",
        description="Past contracts, past income tax litigation documents, past counsel opinions, "
                     "and VAT Infoline/attached client data.",
        node_labels=("ClientEngagementDocument", "TaxMemo", "RestructuringProposal",
                      "DueDiligenceReport", "LitigationSubmission"),
        text_properties=("hasText", "full_citation"),
        vector_query_prefix="Client-specific past contracts, litigation documents, counsel opinions, "
                             "or VAT Infoline data relevant to: ",
    ),
    CategorySpec(
        key="final_output_impact_analysis", label="Final Output & Impact Analysis",
        description="Whether the KPMG opinion or the tax position has changed because of a change in "
                     "law or a new case law (change in judicial interpretation) - synthesized from the "
                     "categories above, not retrieved.",
        synthesis_only=True,
    ),
)

DOMAIN_CONFIG = DomainConfig(
    domain_name="tax",
    structural_labels=STRUCTURAL_LABELS,
    chunk_metadata_keys=CHUNK_METADATA_KEYS,
    embed_chunk_metadata_keys=EMBED_CHUNK_METADATA_KEYS,
    resolution_key_properties=RESOLUTION_KEY_PROPERTIES,
    citation_rules=CITATION_RULES,
    # intent_retrieval_config / default_retrieval_config / valid_intents /
    # fallback_intent all left at framework defaults - tax's original
    # webapp/retrieval.py values were IDENTICAL to the framework defaults
    # (EXACT_LOOKUP/RELATIONAL/CONCEPTUAL with the same vector_top_k/mode),
    # so there's nothing to override here.
    pii_always_masked_fields=ALWAYS_MASKED_FIELDS,
    pii_regex_detectors=REGEX_DETECTORS,
    sample_relationship_checks=SAMPLE_RELATIONSHIP_CHECKS,
    # embedding_text_fn left as None - framework's default_embedding_text()
    # (short_title + content) is byte-for-byte what 05_embed_chunks.py did.
)
