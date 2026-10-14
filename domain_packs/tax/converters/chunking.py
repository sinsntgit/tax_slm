"""
Hierarchical + semantic chunker for the Income Tax Act (and similarly
structured legislative markdown), implementing the "Act Document Chunking
Strategy" spec: Section -> Subsection -> Clause -> Sub-clause, with
Provisos/Explanations attached to their parent element, deterministic
amendment-history and cross-reference extraction, and citation-precise
metadata on every chunk.

WHY THIS EXISTS: 02_extract_entities.py used to chunk purely by markdown
heading boundaries + a hard character cap, then ask the LLM to GUESS
section_no/sub_section_no/clause_id/citation for every entity from raw
text. That's both imprecise (the LLM sometimes gets citations wrong) and
wasteful (regex can get these exactly right, for free, before any LLM call
happens at all). This module does that deterministic parsing once, so:
  - Section/SubSection/Clause/SubClause/Proviso/Explanation entities and
    their structural relationships (PART_OF, HAS_PROVISO, HAS_EXPLANATION,
    REFERS_TO, AMENDED_BY) are built directly from the parse - NO LLM call
    needed for structure, and zero risk of a hallucinated citation.
  - The LLM (in 02_extract_entities.py) is only asked to fill in what
    genuinely needs judgment: DefinedTerm extraction, Illustration /
    Memorandum detection, status inference, confidence scoring - using
    the already-known citations/metadata as ground truth context instead
    of re-deriving them.

TUNED TO income_tax_act_1961_v2.md'S ACTUAL FORMAT, which differs from the
chunking-strategy spec's illustrative regexes in several ways discovered by
inspecting the real file directly (not guessed):
  - Section boundaries are `#### Section <num>. <title>.` lines. EVERY
    section - including lettered insertions like "5A", "80CCD" - carries
    this heading (890 occurrences); a redundant `### Section <num>.`
    heading exists ONLY for the 376 original numbered sections, so `####`
    is used as the sole, complete boundary marker.
  - Amendments are NOT the bracketed-tag style the spec's example regex
    assumes (`[Ins. by Act No. X of Y]`). They're footnote-style: an
    inline marker like `6[twenty-four]` appears in the running text, and
    the footnote definition (`6.Sub. for "thirty-six" by the Act No. 15
    of 2024, w.r.e.f. 23-7-2024.`) appears later, INSIDE the flow of the
    document (not collected at the end) - often literally splitting one
    paragraph into two pieces around the footnote-definition line(s).
    Footnote-definition lines are stripped from chunk content and indexed
    separately by marker id; inline markers are resolved to their
    footnote's amendment info and replaced with just the bracketed text
    (or the literal "[***]" marker, for genuinely redacted/omitted text).
  - Schedules (`## THE FIRST SCHEDULE` etc.) sometimes reuse "Section N."
    -style headings for unrelated internal content (confirmed: 139 such
    headings appear AFTER the first schedule boundary, vs 751 genuine
    section headings before it) - so section parsing stops at the first
    schedule heading; schedules are chunked as their own
    chunk_type="schedule" units, linked back to the section(s) that
    reference them via "[See section N]" / "[See sections N, M, ...]".
  - A handful of section numbers repeat even within the main body (e.g.
    two different "Section 11" headings, one genuine, one clearly a
    mis-parsed heading inside embedded table/notification text) - not
    fully avoidable via regex alone from this source file. Handled by
    keeping the FIRST occurrence as canonical and flagging subsequent
    ones (disambiguated id, listed in the validation report) rather than
    silently dropping or silently merging them.

Entities/relationships returned by this module are already in the exact
{"id", "type", "properties"} / {"type", "source", "target", "properties"}
shape used everywhere else in this pipeline (02's LLM output, 03's loader),
so no format translation is needed at the call sites.
"""
import re
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    import tiktoken
    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None


def token_count(text: str) -> int:
    """Approximate token count - used only for chunk-size DECISIONS (which
    tier of the hierarchy to split at), not for billing, so an approximate
    encoding is fine regardless of which LLM provider is actually configured."""
    if not text:
        return 0
    if _ENCODING is not None:
        try:
            return len(_ENCODING.encode(text))
        except Exception:
            pass
    return max(1, len(text) // 4)  # rough fallback: ~4 chars/token


# --- Size guidelines (tokens), per the chunking-strategy spec -------------
SIZE_GUIDELINES = {
    "section":     {"min": 100, "target": 500, "max": 1000},
    "subsection":  {"min": 75,  "target": 400, "max": 800},
    "clause":      {"min": 50,  "target": 300, "max": 800},
    "sub_clause":  {"min": 30,  "target": 200, "max": 500},
    "proviso":     {"min": 20,  "target": 200, "max": 800},
    "explanation": {"min": 50,  "target": 400, "max": 1200},
    "schedule":    {"min": 20,  "target": 400, "max": 1500},
}


def decide_chunk_size(element_type: str, content_tokens: int) -> tuple[str, str]:
    guide = SIZE_GUIDELINES.get(element_type)
    if guide is None:
        return "unknown", f"no size guideline for element_type={element_type}"
    if content_tokens < guide["min"]:
        return "too_small", "consider combining with adjacent elements"
    if content_tokens <= guide["max"]:
        return "good", "chunk size is acceptable"
    return "too_large", "chunk exceeds max; consider splitting further"


# --- Regexes tuned to income_tax_act_1961_v2.md ----------------------------

SECTION_HEADING_RE = re.compile(r'^####\s+Section\s+([0-9]+[A-Z]{0,3})\.\s*(.*)$')
CHAPTER_HEADING_RE = re.compile(r'^##\s+(?:Chapter|CHAPTER)\s+([IVXLCDM]+(?:-[A-Z])?)\s*[:\-]{1,2}\s*(.*)$', re.IGNORECASE)
SCHEDULE_HEADING_RE = re.compile(r'^##\s+THE\s+(.+?)\s+SCHEDULE\s*$', re.IGNORECASE)

# Level 2: subsection - digit(s) optionally followed by ONE uppercase letter, e.g. (1), (1A), (42A)
SUBSECTION_RE = re.compile(r'^\((\d+[A-Z]?)\)\s?(.*)$')
# Level 3: clause - lowercase letters only, e.g. (a), (aa), (ac)
CLAUSE_RE = re.compile(r'^\(([a-z]{1,3})\)\s?(.*)$')
# Level 4: sub-clause - lowercase roman numerals, e.g. (i), (ii), (iii), (iv)
SUB_CLAUSE_RE = re.compile(r'^\(([ivxlcdm]{1,6})\)\s?(.*)$', re.IGNORECASE)

PROVISO_START_RE = re.compile(r'^\s*Provided\s+(?:that|further|also)\b', re.IGNORECASE)
EXPLANATION_START_RE = re.compile(r'^\s*Explanation\s*(\d*)\s*[.—-]{1,2}\s*(.*)$')

# Footnote DEFINITION line, e.g. "6.Sub. for “thirty-six” by the Act No. 15 of 2024, w.r.e.f. 23-7-2024."
# or "9a.Omitt., ibid."
# BUG FIX (found while building 09_pdf_to_md_actdocs.py - a real footnote
# line straight from the source PDF, "3. Ins. by the Act No. 15 of 2024,
# w.e.f. 1-10-2024.", failed to match at all). The action alternation used
# to embed the trailing period INSIDE "Ins\." / "Sub\." themselves and then
# required \b immediately after - but \b only fires at a word/non-word
# transition, and the character right after that embedded period is always
# another non-word character (a space, in every real occurrence), so \b can
# never succeed there. "Omitt\.?" happened to still work by backtracking off
# its OPTIONAL period to land \b on the word/non-word letter boundary
# instead - which is exactly why this was never caught: every previously
# tested case happened to be "Omitt."/"Omitted"/"Added"/"Amended", never the
# two most common action words "Ins."/"Sub." Fixed by moving the trailing
# period OUT of the alternation (all action words listed bare) and checking
# \b right after the bare word - where the boundary always genuinely exists
# - with the period, if present, consumed separately afterward via `\.?`.
FOOTNOTE_DEF_RE = re.compile(
    r'^(?P<marker>\d+[a-z]?)\.\s*(?P<action>Ins|Sub|Omitt|Omitted|Added|Amended)\b\.?'
    r'(?P<rest>.*)$'
)
FOOTNOTE_ACTNO_RE = re.compile(r'Act\s*No\.?\s*(\d+)\s+of\s+(\d{4})', re.IGNORECASE)
# BUG FIX (found via a real end-to-end test): this only matched "w.r.e.f."
# (WITH RETROSPECTIVE EFFECT FROM) - the rarer variant, present in this
# regex's own illustrative comment above. It silently missed the much more
# common plain "w.e.f." (WITH EFFECT FROM), which is what every one of the
# real amendment footnotes actually in income_tax_act_1961_v2.md uses (e.g.
# "Omitt. by the Act No. 15 of 2024, w.e.f. 1-4-2025.") - so effective_from
# came out None on every single amendment record ever produced, 29/29 at
# last count, despite the raw text always having a clear date right there.
# "r." is now optional so both "w.e.f." and "w.r.e.f." match the same way.
FOOTNOTE_DATE_RE = re.compile(r'w\.\s*(?:r\.\s*)?e\.f\.?\s*([\d]{1,2}-[\d]{1,2}-[\d]{4})', re.IGNORECASE)

# Inline marker referencing a footnote, e.g. "6[twenty-four]" or "9a[***]"
INLINE_MARKER_RE = re.compile(r'(\d+[a-z]?)\[([^\]]*)\]')

REDACTED_RE = re.compile(r'\[\*\*\*\]')

SEE_SECTION_RE = re.compile(r'\[See\s+sections?\s+([^\]]+)\]', re.IGNORECASE)
CROSS_REF_SECTION_RE = re.compile(r'\bsection\s+(\d+[A-Z]{0,3})\b', re.IGNORECASE)
CROSS_REF_SUBSECTION_RE = re.compile(
    r'\bsub-section\s*\(([^)]+)\)\s*(?:of\s+section\s+(\d+[A-Z]{0,3}))?', re.IGNORECASE
)
CROSS_REF_CLAUSE_RE = re.compile(r'\bclause\s*\(([^)]+)\)', re.IGNORECASE)


ELEMENT_TYPE_TO_CLASS = {
    "section": "Section", "subsection": "SubSection", "clause": "Clause", "sub_clause": "SubClause",
}

ACTION_TYPE_MAP = {
    "ins.": "insertion", "sub.": "substitution", "omitt.": "omission",
    "omitt": "omission", "omitted": "omission", "added": "insertion",
    "amended": "amendment",
}


# --- Data model --------------------------------------------------------

@dataclass
class Chunk:
    section: str
    subsection: Optional[str] = None
    clause: Optional[str] = None
    sub_clause: Optional[str] = None

    full_citation: str = ""
    short_title: str = ""

    chunk_type: str = "provision"   # provision | proviso | explanation | schedule
    element_type: str = "section"   # section | subsection | clause | sub_clause | proviso | explanation | schedule

    entity_id: str = ""             # canonical KG entity id, e.g. "Section_55", "Clause_55_2_b"
    parent_entity_id: Optional[str] = None

    content: str = ""
    content_length_tokens: int = 0

    related_sections: list = field(default_factory=list)
    cross_references: list = field(default_factory=list)

    has_proviso: bool = False
    has_explanation: bool = False

    amendment_info: Optional[dict] = None
    has_redacted_content: bool = False

    line_start: int = 0
    line_end: int = 0

    page_number: Optional[int] = None

    chapter: Optional[str] = None

    ambiguities: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# --- Text normalization / footnote extraction ---------------------------

def _extract_footnote_index(lines: list[str]) -> tuple[list[str], dict[str, dict]]:
    """Scans every line for a footnote-DEFINITION (not the inline marker),
    removes those lines from the flow (they're not part of the actual legal
    text - they're editorial amendment notes interleaved mid-paragraph), and
    returns (cleaned_lines, footnote_index) where footnote_index maps
    marker id -> {"type", "act_number", "year", "effective_from", "raw"}."""
    footnote_index: dict[str, dict] = {}
    cleaned: list[str] = []
    for line in lines:
        m = FOOTNOTE_DEF_RE.match(line.strip())
        if m:
            marker = m.group("marker")
            action = m.group("action").lower().rstrip(".")
            rest = m.group("rest") or ""
            actno_m = FOOTNOTE_ACTNO_RE.search(rest)
            date_m = FOOTNOTE_DATE_RE.search(rest)
            entry = {
                "type": ACTION_TYPE_MAP.get(action + ".", ACTION_TYPE_MAP.get(action, "amendment")),
                "act_number": int(actno_m.group(1)) if actno_m else None,
                "year": int(actno_m.group(2)) if actno_m else None,
                "effective_from": _normalize_date(date_m.group(1)) if date_m else None,
                "raw": line.strip(),
            }
            footnote_index[marker] = entry
            continue  # drop this line from content
        cleaned.append(line)
    return cleaned, footnote_index


def _normalize_date(d: str) -> Optional[str]:
    """'23-7-2024' -> '2024-07-23'. Returns None (and lets the caller flag
    an ambiguity) rather than raising, if the format is off - malformed
    dates should be reviewable, not a crash."""
    try:
        day, month, year = d.split("-")
        return f"{year}-{int(month):02d}-{int(day):02d}"
    except Exception:
        return None


def _strip_inline_markers(text: str, footnote_index: dict[str, dict]) -> tuple[str, list[dict], bool]:
    """Replaces every inline marker (e.g. "6[twenty-four]") with just its
    bracketed content (the CURRENT/amended wording), collecting the
    amendment_info of every marker referenced so it can be attached to
    whichever chunk contains it. A marker whose bracket content is "***"
    (or empty) is genuinely redacted/omitted text - kept as a literal
    "[***]" marker in the content (per the spec's Case 3: never delete
    redacted-content markers) rather than stripped to nothing."""
    amendments: list[dict] = []
    redacted = False

    def _sub(m: re.Match) -> str:
        nonlocal redacted
        marker, inner = m.group(1), m.group(2)
        if marker in footnote_index:
            amendments.append({"marker": marker, **footnote_index[marker]})
        if not inner.strip() or inner.strip() == "***":
            redacted = True
            return "[***]"
        return inner

    cleaned = INLINE_MARKER_RE.sub(_sub, text)
    cleaned = re.sub(r'\[\*\*\*\]', '[***]', cleaned)  # normalize spacing variants, no-op if already clean
    return cleaned, amendments, redacted


# --- Cross-reference extraction -----------------------------------------

def extract_cross_references(text: str, self_section: str) -> list[dict]:
    refs: list[dict] = []
    seen = set()
    for m in CROSS_REF_SUBSECTION_RE.finditer(text):
        sub, target = m.group(1), m.group(2)
        if target and target != self_section and (target, sub) not in seen:
            seen.add((target, sub))
            refs.append({"text": m.group(0), "target_section": target, "target_subsection": sub})
    for m in CROSS_REF_SECTION_RE.finditer(text):
        target = m.group(1)
        if target != self_section and (target, None) not in seen:
            seen.add((target, None))
            refs.append({"text": m.group(0), "target_section": target})
    return refs


def related_sections_from_refs(cross_references: list[dict]) -> list[str]:
    seen = []
    for ref in cross_references:
        t = ref.get("target_section")
        if t and t not in seen:
            seen.append(t)
    return seen


# --- Citation / id generation --------------------------------------------

def generate_full_citation(section: str, subsection=None, clause=None, sub_clause=None) -> str:
    citation = f"Section {section}"
    if subsection:
        citation += f"({subsection})"
    if clause:
        citation += f"({clause})"
    if sub_clause:
        citation += f"({sub_clause})"
    return citation


def _slug(*parts: Optional[str]) -> str:
    return "_".join(p for p in parts if p)


def generate_entity_id(section: str, subsection=None, clause=None, sub_clause=None, namespace: Optional[str] = None) -> str:
    """`namespace` (added 2026-08-27): a section NUMBER alone is not a
    stable graph identity across more than one Act - "Section 90" means
    something completely different under the Income-tax Act, 1961 (treaty
    override) than under the Income-tax Act, 2025 (a capital-gains
    definition; the 1961 treaty-override provision was renumbered to
    Section 159). Before this, generate_entity_id() had zero Act-awareness,
    so ingesting a second Act would silently collide node ids with the
    first Act's sections of the same number - confirmed as a live risk,
    not theoretical, against the real ingested corpus (see
    Ontology_Legal_Review_2026-08-27.md). `namespace` should be the same
    short, stable per-Act id used as --document-id when ingesting that
    Act (e.g. "ITA2025", "ITA1961") - callers that don't pass one keep the
    prior unnamespaced ids (single-Act graphs still work unchanged)."""
    if sub_clause:
        base = f"SubClause_{_slug(section, subsection, clause, sub_clause)}"
    elif clause:
        base = f"Clause_{_slug(section, subsection, clause)}"
    elif subsection:
        base = f"SubSection_{_slug(section, subsection)}"
    else:
        base = f"Section_{section}"
    return f"{namespace}__{base}" if namespace else base


def build_legal_event(event_type: str, event_date: str, instrument_id: str, instrument_label: str,
                       subject_id: str, subject_label: str, confidence: float = 0.8) -> tuple[dict, list[dict]]:
    """2026-08-31: NEW - reifies a dated legal event (see schema.json's
    LegalEvent class docstring) between an INSTRUMENT (the document that
    caused the event - e.g. the later case that overrules, or the newer
    circular that supersedes/withdraws) and a SUBJECT (the thing acted upon
    - the older case/circular). The ontology already declared LegalEvent +
    EVENT_SUBJECT/EVENT_INSTRUMENT + eventType/eventDate; this was simply
    never populated by any converter - see extract_case_law.py's OVERRULES
    loop and extract_circular.py's SUPERSEDES/WITHDRAWS loop, which now both
    call this right where they already build the plain relationship, so no
    new LLM call is needed - the event date is always the INSTRUMENT's own
    already-extracted date (a case's judgment_date, a circular's
    date_issued).

    Deterministic id (LegalEvent_<type>_<instrument_id>_<subject_id>) so
    re-running extraction on the same document MERGEs onto the same
    LegalEvent node instead of duplicating it - same idempotent-by-id
    convention graph_loader.py's load_entities()/load_relationships() rely
    on for every other entity/relationship this pipeline produces.

    event_date may be "" (LLM found no date on the instrument) - left
    UNSET on the node rather than written as an empty string, matching the
    effective_from "unset = no known constraint" convention already used
    elsewhere in these converters (apply_metadata_filters()'s as_of param).

    Returns (event_entity_dict, [EVENT_SUBJECT_rel, EVENT_INSTRUMENT_rel]) -
    caller appends both onto its own entities/relationships lists, exactly
    like every other stub-node helper in this pipeline."""
    event_id = f"LegalEvent_{event_type}_{instrument_id}_{subject_id}"
    props = {
        "eventType": event_type,
        "full_citation": f"{event_type.title()} of {subject_label} by {instrument_label}",
        "confidence": confidence,
        "extraction_notes": f"Reified from a {event_type} relationship between "
                             f"{instrument_id} (instrument) and {subject_id} (subject).",
    }
    if (event_date or "").strip():
        props["eventDate"] = event_date.strip()
    entity = {"id": event_id, "type": "LegalEvent", "properties": props}
    relationships = [
        {"type": "EVENT_SUBJECT", "source": event_id, "target": subject_id, "properties": {}},
        {"type": "EVENT_INSTRUMENT", "source": event_id, "target": instrument_id, "properties": {}},
    ]
    return entity, relationships


def make_short_title(text: str, max_len: int = 90) -> str:
    t = re.sub(r'\s+', ' ', text).strip()
    return (t[:max_len].rsplit(" ", 1)[0] + "...") if len(t) > max_len else t


# --- Line-scope helpers ----------------------------------------------------

def _split_marked_blocks(lines: list[tuple[int, str]], pattern: re.Pattern) -> list[tuple[str, str, list[tuple[int, str]]]]:
    """Generic splitter: given (line_no, text) pairs for one scope, finds
    every line that starts a new marked block (subsection/clause/sub-clause)
    per `pattern`, and returns [(marker, first_line_remainder, block_lines)].
    Content BEFORE the first marker (a lead-in sentence, e.g. Section 2's
    "In this Act, unless the context otherwise requires,--") is returned
    separately as a block with marker=None.

    IMPORTANT: a marker value already used as a block-starter in THIS split
    is never treated as a new block-starter again - it's folded into the
    CURRENT block's content instead. This source's markdown has no
    indentation to distinguish nesting depth, so a clause's own internal
    numbered sub-points (e.g. clause (b) containing its own "(1)", "(2)"
    list) look byte-for-byte identical to a genuine top-level subsection
    marker. Real legal drafting never reuses a marker at the SAME level
    within one section/subsection/clause, so "first occurrence wins, repeats
    are nested content" is a safe, well-justified heuristic here - without
    it, a nested "(1)"/"(2)" list inside a clause gets mis-read as the
    section's real subsections 1 and 2 reappearing, duplicating them."""
    blocks: list[tuple[Optional[str], str, list[tuple[int, str]]]] = []
    seen_markers: set[str] = set()
    current_marker = None
    current_first = ""
    current_lines: list[tuple[int, str]] = []

    def flush():
        if current_lines or current_marker is not None:
            blocks.append((current_marker, current_first, current_lines))

    for line_no, text in lines:
        m = pattern.match(text)
        if m and m.group(1) not in seen_markers:
            flush()
            current_marker = m.group(1)
            seen_markers.add(current_marker)
            current_first = m.group(2) if m.lastindex and m.lastindex >= 2 else ""
            current_lines = [(line_no, current_first)] if current_first else []
        else:
            current_lines.append((line_no, text))
    flush()
    return blocks


def _join(lines: list[tuple[int, str]]) -> str:
    return "\n".join(t for _, t in lines).strip()


def _line_range(lines: list[tuple[int, str]], fallback: int) -> tuple[int, int]:
    if not lines:
        return fallback, fallback
    return lines[0][0], lines[-1][0]


# --- Proviso / explanation extraction (leaf-level) -------------------------

def _peel_provisos_and_explanations(lines: list[tuple[int, str]]) -> tuple[list[tuple[int, str]], list[list[tuple[int, str]]], list[tuple[str, list[tuple[int, str]]]]]:
    """Splits a leaf element's lines into (main_lines, proviso_blocks,
    explanation_blocks). A leaf can have MULTIPLE provisos ("Provided
    that... Provided further that...") and multiple numbered explanations
    (Explanation 1, Explanation 2, ...) - each becomes its own attached
    child entity rather than being flattened into one field."""
    main: list[tuple[int, str]] = []
    provisos: list[list[tuple[int, str]]] = []
    explanations: list[tuple[str, list[tuple[int, str]]]] = []

    mode = "main"
    current: list[tuple[int, str]] = []
    current_expl_num = ""

    def flush():
        nonlocal current
        if mode == "proviso" and current:
            provisos.append(current)
        elif mode == "explanation" and current:
            explanations.append((current_expl_num, current))
        current = []

    for line_no, text in lines:
        if PROVISO_START_RE.match(text):
            flush()
            mode = "proviso"
            current = [(line_no, text)]
            continue
        expl_m = EXPLANATION_START_RE.match(text)
        if expl_m:
            flush()
            mode = "explanation"
            current_expl_num = expl_m.group(1) or ""
            current = [(line_no, expl_m.group(2))] if expl_m.group(2) else []
            continue
        if mode == "main":
            main.append((line_no, text))
        else:
            current.append((line_no, text))
    flush()
    return main, provisos, explanations


# --- Amendment -> entity-level property mapping -----------------------------

def _amendment_entity_props(amendment: Optional[dict]) -> dict:
    """
    Turns one parsed amendment/footnote dict (see _extract_footnote_index)
    into the entity-level properties describing it - with semantics that
    depend on the amendment TYPE, not one-size-fits-all "always set
    effective_from" (which was backwards for omissions):

      - insertion/substitution/amendment: the recorded date is when the
        CURRENT text (this entity's hasText, right now) BECAME effective.
        status="in force", effective_from=that date, effective_to left
        unset (still open-ended as of this document snapshot).
      - omission: the recorded date is when this text STOPPED applying.
        chunking.py doesn't delete the old (pre-omission) text from
        hasText - the source Act keeps it printed for reference - so
        stamping that date as "effective_from" would say the exact
        opposite of what happened. status="omitted", effective_to=that
        date, effective_from left unset (there's no way to know from a
        single footnote when this text ORIGINALLY started applying -
        would need to chain back through whatever amendment came before
        it at the same position, which isn't tracked here).

    Returns {} if there's no usable date (nothing to add - the entity just
    won't have effective_from/effective_to/status set at all, which
    apply_metadata_filters() already treats as "no known constraint",
    never as "excluded").
    """
    if not amendment:
        return {}
    props = {}
    if amendment.get("act_number"):
        props["amendingActNo"] = amendment["act_number"]
    if amendment.get("year"):
        props["amendingActYear"] = amendment["year"]
    props["amendmentNature"] = amendment.get("type")
    date = amendment.get("effective_from")
    if amendment.get("type") == "omission":
        # status is knowable ("omitted") even without a date (some footnotes
        # are just "9a.Omitt., ibid." with no date/act of their own - the
        # ACTION alone is still useful metadata even when the date isn't).
        props["status"] = "omitted"
        if date:
            props["effective_to"] = date
    elif date:
        props["status"] = "in force"
        props["effective_from"] = date
    return props


# --- Core recursive builder -------------------------------------------------

class _ChunkBuilder:
    """Accumulates entities/relationships/chunks while walking the
    Section -> Subsection -> Clause -> Sub-clause hierarchy. Kept as a
    small stateful class (not free functions passing four lists around)
    purely to keep chunk_act_document() readable."""

    def __init__(self, footnote_index: dict[str, dict], namespace: Optional[str] = None):
        self.footnote_index = footnote_index
        self.namespace = namespace  # see generate_entity_id()'s docstring - Act-scoped id prefix
        self.entities: dict[str, dict] = {}
        self.relationships: list[dict] = []
        self.chunks: list[Chunk] = []
        self.ambiguities: list[dict] = []

    def _eid(self, section, subsection=None, clause=None, sub_clause=None) -> str:
        """generate_entity_id() pre-bound to this builder's namespace - use
        this everywhere inside the class instead of calling
        generate_entity_id() directly, so every id this builder mints
        (including cross-reference targets, which are always WITHIN the
        same Act) is consistently namespaced."""
        return generate_entity_id(section, subsection, clause, sub_clause, namespace=self.namespace)

    def add_entity(self, entity_id: str, etype: str, properties: dict):
        if entity_id in self.entities:
            # Duplicate id (e.g. a mis-parsed second "Section 11" heading
            # elsewhere in the source) - keep the first, flag for review
            # rather than silently overwriting real content.
            self.ambiguities.append({
                "type": "duplicate_entity_id", "entity_id": entity_id,
                "note": "kept first occurrence; second occurrence's content was discarded - review source.",
            })
            return entity_id
        self.entities[entity_id] = {"id": entity_id, "type": etype, "properties": properties}
        return entity_id

    def add_relationship(self, rel_type: str, source: str, target: str, properties: Optional[dict] = None):
        self.relationships.append({"type": rel_type, "source": source, "target": target, "properties": properties or {}})

    def _ensure_amending_act(self, act_number: Optional[int], year: Optional[int]) -> Optional[str]:
        """
        Creates (or reuses, if a previously-processed provision already
        created it) the AmendingAct node for a given act number/year, per
        the ontology's AmendingAct class + `AMENDED_BY` object property
        (StatutoryInstrument -> AmendingAct - see New_tax_ontology_v2.owl).
        Many provisions typically share the SAME amending act (e.g. the
        Finance Act, 2024 amends dozens of sections at once), so this is
        deliberately idempotent - reusing the id rather than routing
        through add_entity(), which would otherwise flag every re-use as a
        false "duplicate_entity_id" ambiguity (that check exists for
        genuine parsing collisions, not intentional sharing of one real
        Act across many provisions it amended).

        Returns None if there isn't enough information (act number AND
        year) to identify a specific Act - callers should skip creating the
        AMENDED_BY edge in that case rather than inventing a partial id.
        """
        if not act_number or not year:
            return None
        act_id = f"AmendingAct_{act_number}_{year}"
        if act_id not in self.entities:
            self.entities[act_id] = {
                "id": act_id, "type": "AmendingAct",
                "properties": {
                    "amendingActNo": act_number, "amendingActYear": year,
                    "full_citation": f"Act No. {act_number} of {year}",
                    "confidence": 1.0, "extraction_notes": "",
                    # 2026-10-14: provenance (framework/core/ingest/provenance.py) -
                    # this node exists only because an amendment footnote CITED the
                    # amending Act; its own text is never ingested here. Literal
                    # strings (not an import) to keep chunking.py dependency-free.
                    # sourceMdPath "" so attach_pipeline_metadata()'s setdefault
                    # can't point it at the Act's .md (wrong text for this node).
                    "provenance": "cited_reference", "sourceMdPath": "",
                },
            }
        return act_id

    def _attach_children(self, *, entity_id: str, citation: str, section, subsection, clause, sub_clause,
                          proviso_blocks, explanation_blocks, chapter: Optional[str], fallback_line: int) -> tuple[bool, bool]:
        """Creates Proviso/Explanation child entities + HAS_PROVISO/
        HAS_EXPLANATION edges + their own fine-grained Chunk records, and
        attaches them to `entity_id`. Shared by every scope level (a
        section, subsection, or clause can each have its own attached
        provisos/explanations - not just leaves), which matters because
        peeling now happens BEFORE the leaf-vs-split decision (see
        build_scope): a clause's Explanation is removed from consideration
        before we ever look for further clause/sub-clause structure in it,
        rather than only being peeled off once we already know we're at a
        leaf. Returns (has_proviso, has_explanation)."""
        for i, block in enumerate(proviso_blocks, start=1):
            p_text_raw = _join(block)
            p_text, p_amendments, p_redacted = _strip_inline_markers(p_text_raw, self.footnote_index)
            p_id = f"Proviso_{i}_to_{entity_id}" if len(proviso_blocks) > 1 else f"Proviso_to_{entity_id}"
            p_props = {
                "hasText": p_text, "full_citation": f"Proviso to {citation}",
                "confidence": 1.0, "extraction_notes": "",
            }
            if p_redacted:
                p_props["content_status"] = "deleted_or_redacted"
            if p_amendments:
                p_props.update(_amendment_entity_props(p_amendments[0]))
            self.add_entity(p_id, "Proviso", p_props)
            self.add_relationship("HAS_PROVISO", entity_id, p_id)
            if p_amendments:
                act_id = self._ensure_amending_act(p_amendments[0].get("act_number"), p_amendments[0].get("year"))
                if act_id:
                    self.add_relationship("AMENDED_BY", p_id, act_id)
            ls, le = _line_range(block, fallback_line)
            self.chunks.append(Chunk(
                section=section, subsection=subsection, clause=clause, sub_clause=sub_clause,
                full_citation=f"Proviso to {citation}", short_title=make_short_title(p_text),
                chunk_type="proviso", element_type="proviso", entity_id=p_id, parent_entity_id=entity_id,
                content=p_text, content_length_tokens=token_count(p_text),
                amendment_info=p_amendments[0] if p_amendments else None,
                has_redacted_content=p_redacted, line_start=ls, line_end=le, chapter=chapter,
            ))

        for expl_num, block in explanation_blocks:
            e_text_raw = _join(block)
            e_text, e_amendments, e_redacted = _strip_inline_markers(e_text_raw, self.footnote_index)
            suffix = f"_{expl_num}" if expl_num else ""
            e_id = f"Explanation{suffix}_to_{entity_id}"
            e_props = {
                "hasText": e_text, "full_citation": f"Explanation{(' ' + expl_num) if expl_num else ''} to {citation}",
                "confidence": 1.0, "extraction_notes": "",
            }
            if e_redacted:
                e_props["content_status"] = "deleted_or_redacted"
            if e_amendments:
                e_props.update(_amendment_entity_props(e_amendments[0]))
            self.add_entity(e_id, "Explanation", e_props)
            self.add_relationship("HAS_EXPLANATION", entity_id, e_id)
            if e_amendments:
                act_id = self._ensure_amending_act(e_amendments[0].get("act_number"), e_amendments[0].get("year"))
                if act_id:
                    self.add_relationship("AMENDED_BY", e_id, act_id)
            ls, le = _line_range(block, fallback_line)
            self.chunks.append(Chunk(
                section=section, subsection=subsection, clause=clause, sub_clause=sub_clause,
                full_citation=f"Explanation{(' ' + expl_num) if expl_num else ''} to {citation}",
                short_title=make_short_title(e_text),
                chunk_type="explanation", element_type="explanation", entity_id=e_id, parent_entity_id=entity_id,
                content=e_text, content_length_tokens=token_count(e_text),
                amendment_info=e_amendments[0] if e_amendments else None,
                has_redacted_content=e_redacted, line_start=ls, line_end=le, chapter=chapter,
            ))
        return bool(proviso_blocks), bool(explanation_blocks)

    def build_leaf(self, *, section: str, subsection, clause, sub_clause,
                   main_lines: list[tuple[int, str]], element_type: str,
                   parent_entity_id: Optional[str], chapter: Optional[str],
                   fallback_line: int, has_proviso: bool, has_explanation: bool):
        """A leaf is whatever level the decision tree decided NOT to split
        further. `main_lines` has ALREADY had any provisos/explanations
        peeled out and attached by the caller (build_scope) - this only
        builds the entity + Chunk for the remaining main content."""
        entity_id = self._eid(section, subsection, clause, sub_clause)
        citation = generate_full_citation(section, subsection, clause, sub_clause)

        raw_text = _join(main_lines)
        clean_text, amendments, redacted = _strip_inline_markers(raw_text, self.footnote_index)
        line_start, line_end = _line_range(main_lines, fallback_line)

        cross_refs = extract_cross_references(clean_text, section)
        related = related_sections_from_refs(cross_refs)

        props = {
            "hasText": clean_text,
            "full_citation": citation,
            "confidence": 1.0,       # deterministically parsed, not LLM-inferred
            "extraction_notes": "",
        }
        if section:
            props["section_no"] = section
        if subsection:
            props["sub_section_no"] = subsection
        if clause:
            props["clause_id"] = clause
        if sub_clause:
            props["sub_clause_id"] = sub_clause
        if redacted:
            props["content_status"] = "deleted_or_redacted"
        if amendments:
            props.update(_amendment_entity_props(amendments[0]))  # primary amendment for entity-level properties

        self.add_entity(entity_id, ELEMENT_TYPE_TO_CLASS[element_type], props)
        if parent_entity_id:
            self.add_relationship("PART_OF", entity_id, parent_entity_id)
        if amendments:
            act_id = self._ensure_amending_act(amendments[0].get("act_number"), amendments[0].get("year"))
            if act_id:
                self.add_relationship("AMENDED_BY", entity_id, act_id)
        for ref in cross_refs:
            # Same-Act cross-reference (a Section referring to another
            # Section of the SAME statute) - always namespaced to this
            # builder's own Act, never the caller's concern.
            target_id = self._eid(ref['target_section'])
            self.add_relationship("REFERS_TO", entity_id, target_id, {"cross_ref_text": ref["text"]})

        self.chunks.append(Chunk(
            section=section, subsection=subsection, clause=clause, sub_clause=sub_clause,
            full_citation=citation, short_title=make_short_title(clean_text),
            chunk_type="provision", element_type=element_type,
            entity_id=entity_id, parent_entity_id=parent_entity_id,
            content=clean_text, content_length_tokens=token_count(clean_text),
            related_sections=related, cross_references=cross_refs,
            has_proviso=has_proviso, has_explanation=has_explanation,
            amendment_info=amendments[0] if amendments else None,
            has_redacted_content=redacted, line_start=line_start, line_end=line_end, chapter=chapter,
        ))
        return entity_id

    def _finalize_leaf(self, *, section, subsection, clause, sub_clause, lines, element_type,
                        parent_entity_id, chapter, fallback_line):
        """Peels provisos/explanations from an ALREADY-DECIDED leaf's lines,
        attaches them, and builds the leaf entity + Chunk. This is the only
        place peeling happens - see build_scope's docstring for why peeling
        any earlier (before a scope is confirmed to be a leaf) is unsafe on
        this source."""
        main_lines, proviso_blocks, explanation_blocks = _peel_provisos_and_explanations(lines)
        entity_id = self._eid(section, subsection, clause, sub_clause)
        citation = generate_full_citation(section, subsection, clause, sub_clause)
        has_proviso, has_explanation = self._attach_children(
            entity_id=entity_id, citation=citation, section=section, subsection=subsection,
            clause=clause, sub_clause=sub_clause, proviso_blocks=proviso_blocks, explanation_blocks=explanation_blocks,
            chapter=chapter, fallback_line=fallback_line,
        )
        self.build_leaf(
            section=section, subsection=subsection, clause=clause, sub_clause=sub_clause,
            main_lines=main_lines, element_type=element_type, parent_entity_id=parent_entity_id,
            chapter=chapter, fallback_line=fallback_line,
            has_proviso=has_proviso, has_explanation=has_explanation,
        )

    def build_scope(self, *, section: str, subsection, clause, lines: list[tuple[int, str]],
                     element_type: str, parent_entity_id: Optional[str], chapter: Optional[str],
                     fallback_line: int):
        """Decision-tree step for one scope (a whole section, or one
        subsection, or one clause): tries to split into the NEXT level's
        marker pattern FIRST (on the raw, un-peeled lines), and only peels
        provisos/explanations once a scope is confirmed to be a leaf (no
        further genuine structure, or small enough to stop).

        Peeling has to happen this late, not before splitting, because
        peeling swallows EVERYTHING from the first "Provided that"/
        "Explanation" trigger to the end of whatever lines it's given (this
        source has no reliable structural cue for where a proviso/
        explanation actually ENDS other than "the scope ends") - peeling a
        whole SECTION's raw text before it's been split into subsections
        would swallow every subsection after the first explanation found
        anywhere in the section, not just that one subsection's own
        explanation.

        Splitting-first has the opposite failure mode instead: an
        Explanation can contain its OWN internal (i)/(ii)/(a)/(b)
        enumeration, which - looked at with no context - matches the exact
        same CLAUSE_RE/SUB_CLAUSE_RE patterns used for this scope's
        genuine next-level structure. Guarded against below: if the content
        BEFORE the first matched next-level marker already contains a
        Provided/Explanation trigger, the "next-level structure" found
        after it is actually nested inside that proviso/explanation, not
        real sibling structure of this scope - the split is discarded and
        the whole scope is treated as one leaf instead (to be peeled
        normally, keeping that nested enumeration as plain content of
        whichever proviso/explanation it belongs to)."""
        guide = SIZE_GUIDELINES.get(element_type, SIZE_GUIDELINES["clause"])
        tokens = token_count(_join(lines))

        if element_type == "section":
            next_pattern, next_type = SUBSECTION_RE, "subsection"
        elif element_type == "subsection":
            next_pattern, next_type = CLAUSE_RE, "clause"
        elif element_type == "clause":
            next_pattern, next_type = SUB_CLAUSE_RE, "sub_clause"
        else:
            next_pattern, next_type = None, None

        real_blocks: list = []
        lead_in_block = None
        if next_pattern is not None and tokens > guide["max"]:
            blocks = _split_marked_blocks(lines, next_pattern)
            lead_in_block = next((b for b in blocks if b[0] is None), None)
            candidate_blocks = [b for b in blocks if b[0] is not None]
            lead_in_lines = lead_in_block[2] if lead_in_block else []
            lead_has_trigger = any(
                PROVISO_START_RE.match(t) or EXPLANATION_START_RE.match(t) for _, t in lead_in_lines
            )
            if candidate_blocks and not lead_has_trigger:
                real_blocks = candidate_blocks

        if not real_blocks:
            if tokens > guide["max"] and next_pattern is not None:
                self.ambiguities.append({
                    "type": "oversized_leaf", "citation": generate_full_citation(section, subsection, clause),
                    "tokens": tokens, "note": "exceeds recommended size with no further (non-nested) "
                                               "structure detected - kept as one leaf",
                })
            self._finalize_leaf(
                section=section, subsection=subsection, clause=clause, sub_clause=None,
                lines=lines, element_type=element_type, parent_entity_id=parent_entity_id,
                chapter=chapter, fallback_line=fallback_line,
            )
            return

        # Entity for THIS scope still needs to exist (e.g. Section 55 itself,
        # even though its content lives in its subsections) so the hierarchy
        # is fully connected. Its own "content" is just the lead-in text
        # (before the first marked sub-block) - peeled the same way a leaf
        # would be, since a lead-in can itself carry a proviso/explanation
        # that applies to the WHOLE scope rather than any one child.
        # 2026-10-14: BUGFIX - was generate_entity_id(...) WITHOUT the builder's
        # namespace, so with a namespace (every Act ingestion since 2026-08-27)
        # each SPLIT Section/SubSection/Clause wrapper was minted as e.g.
        # "Section_2" while its leaves were "ITA2025__Clause_2_1_a" - the
        # wrappers collided across Acts (exactly what namespacing exists to
        # prevent), PART_OF edges pointed at the un-namespaced id, and
        # chunk_act_document()'s builder._eid(number) lookup never found the
        # wrapper to set its marginal_heading. Caught by tests/test_chunking.py.
        this_entity_id = self._eid(section, subsection, clause)
        citation = generate_full_citation(section, subsection, clause)
        lead_main, lead_provisos, lead_explanations = _peel_provisos_and_explanations(
            lead_in_block[2] if lead_in_block else []
        )
        lead_has_proviso, lead_has_explanation = self._attach_children(
            entity_id=this_entity_id, citation=citation, section=section, subsection=subsection,
            clause=clause, sub_clause=None, proviso_blocks=lead_provisos, explanation_blocks=lead_explanations,
            chapter=chapter, fallback_line=fallback_line,
        )
        clean_lead, lead_amendments, lead_redacted = _strip_inline_markers(_join(lead_main), self.footnote_index)
        wrapper_props = {
            "hasText": clean_lead, "full_citation": citation, "confidence": 1.0, "extraction_notes": "",
        }
        if lead_redacted:
            wrapper_props["content_status"] = "deleted_or_redacted"
        if lead_amendments:
            wrapper_props.update(_amendment_entity_props(lead_amendments[0]))
        self.add_entity(this_entity_id, ELEMENT_TYPE_TO_CLASS[element_type], wrapper_props)
        if parent_entity_id:
            self.add_relationship("PART_OF", this_entity_id, parent_entity_id)
        if lead_amendments:
            act_id = self._ensure_amending_act(lead_amendments[0].get("act_number"), lead_amendments[0].get("year"))
            if act_id:
                self.add_relationship("AMENDED_BY", this_entity_id, act_id)

        for marker, first_rest, block_lines in real_blocks:
            # block_lines already INCLUDES the marker line's own remainder
            # text (first_rest) as its first element - _split_marked_blocks
            # seeds it there. Do not prepend it again here.
            full_block = block_lines
            if next_type == "subsection":
                self.build_scope(section=section, subsection=marker, clause=None, lines=full_block,
                                  element_type="subsection", parent_entity_id=this_entity_id,
                                  chapter=chapter, fallback_line=fallback_line)
            elif next_type == "clause":
                self.build_scope(section=section, subsection=subsection, clause=marker, lines=full_block,
                                  element_type="clause", parent_entity_id=this_entity_id,
                                  chapter=chapter, fallback_line=fallback_line)
            else:  # sub_clause - leaf, no further recursion (Level 4 is our practical depth limit)
                self._finalize_leaf(
                    section=section, subsection=subsection, clause=clause, sub_clause=marker,
                    lines=full_block, element_type="sub_clause", parent_entity_id=this_entity_id,
                    chapter=chapter, fallback_line=fallback_line,
                )


# --- Top-level parsing: chapters, sections, schedules -----------------------

def _find_schedule_boundary(numbered_lines: list[tuple[int, str]]) -> int:
    for line_no, text in numbered_lines:
        if SCHEDULE_HEADING_RE.match(text):
            return line_no
    return numbered_lines[-1][0] + 1 if numbered_lines else 0


def _parse_schedules(numbered_lines: list[tuple[int, str]], start_at: int) -> list[Chunk]:
    schedule_starts = [(ln, m.group(1)) for ln, t in numbered_lines if ln >= start_at
                        for m in [SCHEDULE_HEADING_RE.match(t)] if m]
    chunks: list[Chunk] = []
    by_line = dict(numbered_lines)
    for idx, (start_ln, name) in enumerate(schedule_starts):
        end_ln = schedule_starts[idx + 1][0] - 1 if idx + 1 < len(schedule_starts) else numbered_lines[-1][0]
        body_lines = [(ln, t) for ln, t in numbered_lines if start_ln < ln <= end_ln]
        body_text = _join(body_lines)
        see_m = SEE_SECTION_RE.search(body_text)
        related = re.findall(r'\d+[A-Z]{0,3}', see_m.group(1)) if see_m else []
        entity_id = f"Schedule_{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}"
        chunks.append(Chunk(
            section="", full_citation=f"Schedule {name}", short_title=make_short_title(body_text, 80),
            chunk_type="schedule", element_type="schedule", entity_id=entity_id,
            content=body_text[:20000],  # schedules can be large tables; cap rather than embed megabytes raw
            content_length_tokens=token_count(body_text[:20000]),
            related_sections=related, line_start=start_ln, line_end=end_ln,
        ))
    return chunks


def page_for_line(line_no: int, page_boundaries: list) -> Optional[int]:
    """Resolves a markdown line number to its source PDF page, given the
    (output_line_count, page) boundary markers pdf_to_md_act.py's
    classify_and_render() records as it renders (one marker per source-page
    transition it consumes). page_boundaries must be sorted ascending by
    line count (true by construction - recorded in render order)."""
    if not page_boundaries:
        return None
    page = None
    for boundary_line, boundary_page in page_boundaries:
        if boundary_line > line_no:
            break
        page = boundary_page
    return page


def apply_page_numbers(chunks: list, page_boundaries: list) -> None:
    """Mutates each Chunk in-place, setting .page_number from its
    line_start via page_for_line(). No-op if page_boundaries is empty
    (e.g. sidecar not found - page_number just stays None)."""
    if not page_boundaries:
        return
    for c in chunks:
        c.page_number = page_for_line(c.line_start, page_boundaries)


def chunk_act_document(raw_text: str, namespace: Optional[str] = None, act_props: Optional[dict] = None) -> dict:
    """
    Main entry point. Returns:
      {
        "entities": [...],        # ready for 03_build_graph.py, unchanged format
        "relationships": [...],
        "chunks": [Chunk, ...],   # fine-grained, for chunks.json / pgvector / citation lookup
        "footnote_index": {...},
        "ambiguities": [...],     # flagged items for manual review
      }

    `namespace` (added 2026-08-27): short, stable per-Act id (e.g.
    "ITA2025", "ITA1961" - same value extract_entities.py passes as
    --document-id) used to prefix every entity id this call mints, so two
    different Acts can be ingested into the same graph without their
    Section/SubSection/Clause ids colliding - see generate_entity_id()'s
    docstring for why this matters (a bare section NUMBER is not a stable
    identity across Acts). Omit it and this behaves exactly as before
    (unnamespaced ids) - safe for a single-Act graph.

    `act_props` (added 2026-08-27): if given ALONGSIDE `namespace`, an Act
    node is created with id=`namespace` and these properties (actName/
    actYear/actNumber - see schema.json's Act class), and every top-level
    Section gets a PART_OF edge to it (previously top-level Sections had
    NO parent edge at all - Act was a declared-but-never-populated class).
    Omit it (namespace only, no act_props) to get namespaced ids WITHOUT
    an Act node - e.g. for a quick re-chunk where the caller manages the
    Act node itself.
    """
    lines = raw_text.split("\n")
    cleaned_lines, footnote_index = _extract_footnote_index(lines)
    numbered_lines = list(enumerate(cleaned_lines))

    schedule_boundary = _find_schedule_boundary(numbered_lines)
    main_body = [(ln, t) for ln, t in numbered_lines if ln < schedule_boundary]

    # Locate chapter boundaries (for the `chapter` metadata field only).
    chapter_marks = [(ln, m.group(1), m.group(2)) for ln, t in main_body
                      for m in [CHAPTER_HEADING_RE.match(t)] if m]

    def chapter_for_line(ln: int) -> Optional[str]:
        current = None
        for cln, num, title in chapter_marks:
            if cln <= ln:
                current = f"Chapter {num}" + (f": {title.strip()}" if title.strip() else "")
            else:
                break
        return current

    section_starts = [(ln, m.group(1), m.group(2)) for ln, t in main_body
                       for m in [SECTION_HEADING_RE.match(t)] if m]

    builder = _ChunkBuilder(footnote_index, namespace=namespace)

    act_entity_id = None
    if namespace and act_props:
        builder.add_entity(namespace, "Act", dict(act_props))
        act_entity_id = namespace

    for idx, (start_ln, number, title) in enumerate(section_starts):
        end_ln = section_starts[idx + 1][0] - 1 if idx + 1 < len(section_starts) else schedule_boundary - 1
        # 2026-10-14: BUGFIX - a "## CHAPTER ..." heading sits between the
        # last section of one chapter and the first of the next, i.e. INSIDE
        # the preceding section's line range, so it was being glued onto the
        # text of that section's last leaf/proviso/explanation (hasText and
        # Chunk content both ended in "\n## CHAPTER II -- ..."). Chapter
        # headings are already captured separately via chapter_for_line().
        body = [(ln, t) for ln, t in main_body if start_ln < ln <= end_ln and not CHAPTER_HEADING_RE.match(t)]
        # First body line is glued to the section number, e.g. "55. (1) For..."
        # or "2.In this Act,...". Strip that leading "N." prefix so the rest
        # parses like any other line (a lead-in sentence, or an immediate
        # subsection marker).
        if body:
            first_ln, first_text = body[0]
            stripped = re.sub(rf'^\s*{re.escape(number)}\.\s*', '', first_text, count=1)
            body[0] = (first_ln, stripped)

        builder.build_scope(
            # parent_entity_id=act_entity_id (was always None before) - gives
            # every top-level Section a real PART_OF edge to its Act node
            # when one was created above; still None (no parent) if the
            # caller didn't pass act_props, unchanged from prior behavior.
            section=number, subsection=None, clause=None, lines=body,
            element_type="section", parent_entity_id=act_entity_id,
            chapter=chapter_for_line(start_ln), fallback_line=start_ln,
        )
        # entity_id for the section itself always exists via build_scope's
        # lead-in handling OR build_leaf if it stayed a single chunk; set
        # short_title on it from the heading (more reliable than any
        # auto-derived lead-in snippet).
        sec_id = builder._eid(number)
        if sec_id in builder.entities:
            builder.entities[sec_id]["properties"].setdefault("marginal_heading", title.strip())

    schedule_chunks = _parse_schedules(numbered_lines, schedule_boundary)
    for sc in schedule_chunks:
        for target in sc.related_sections:
            builder.add_relationship("CROSS_REFERS_TO_ALLIED_LAW", builder._eid(target), sc.entity_id)
    builder.chunks.extend(schedule_chunks)

    return {
        "entities": list(builder.entities.values()),
        "relationships": builder.relationships,
        "chunks": builder.chunks,
        "footnote_index": footnote_index,
        "ambiguities": builder.ambiguities,
    }


# --- Batching fine-grained chunks for LLM calls -----------------------------

def group_chunks_for_llm(chunks: list[Chunk], max_chars: int = 60_000) -> list[list[Chunk]]:
    """Groups SEQUENTIAL leaf chunks (in document order) up to max_chars per
    group, for a single LLM call - chunks.json/pgvector still store each
    fine-grained chunk individually for citation-precise retrieval regardless
    of how they were batched here.

    Groups MAY span a Section boundary (changed from the original version,
    which forced a new group at every Section change regardless of size -
    for a ~855-section Act that meant ~1 group per section no matter how
    high max_chars was set, e.g. still 856 groups at max_chars=100,000 vs.
    the intended handful. Confirmed empirically before making this change:
    at max_chars=60,000 with the boundary forced, group count barely moved;
    with it removed, 4,014 non-schedule chunks pack into ~45 groups - a
    ~22x reduction in LLM calls for the full Act, which is what actually
    fixes multi-hour/timeout extraction runs). This is safe for extraction
    quality because format_chunk_group_for_prompt() (extract_entities.py)
    tags every chunk in the prompt with its own `[id=... | citation=...]`
    header regardless of which group it landed in - the LLM attributes each
    finding to its source chunk from that tag, not from section grouping -
    so packing multiple sections into one call changes call COUNT, not
    extraction correctness. See extract_entities.py's --max-group-chars and
    --max-output-tokens for how the two ends of this trade-off (fewer/bigger
    calls vs. each call's output token budget) are kept in sync."""
    groups: list[list[Chunk]] = []
    current: list[Chunk] = []
    current_chars = 0

    for c in chunks:
        c_chars = len(c.content)
        if current and current_chars + c_chars > max_chars:
            groups.append(current)
            current, current_chars = [], 0
        current.append(c)
        current_chars += c_chars

    if current:
        groups.append(current)
    return groups
