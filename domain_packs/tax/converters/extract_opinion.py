"""
Firm opinion structured extraction - 2026-10-14: NEW (phase0/opinions).

The FirmOpinion counterpart to extract_case_law.py (judgments) and
extract_circular.py (CBDT circulars). Firm opinions are the customer's OWN
proprietary knowledge and the centre of the "Tax SLM" product: a formal
opinion letter (facts -> issue -> law -> precedents -> analysis -> conclusion
-> caveats -> reliance restriction), usually with YAML-style frontmatter
(opinion_id, title, client, counterparty, category, income_tax_sections,
treaty, case_law_cited, confidence_level, partner, firm, date).

WHY DETERMINISTIC (NO LLM CALL): unlike a judgment or a circular, an opinion
in this format has a fixed, machine-readable structure - frontmatter fields
plus numbered "## N. <Heading>" body sections whose headings are a stable
vocabulary ("Statement of Facts", "Issue for Consideration", "Analysis",
"Conclusion / Opinion", ...). Per this project's "deterministic derivation
first" rule (see CLAUDE.md and extract_case_law.py's derive_*() helpers),
everything here is parsed, not model-judged. A PDF/TXT opinion WITHOUT
frontmatter still works: every frontmatter field has a body fallback (the
"Date:" line, "Subject:" heading, "To," addressee, the section 4/5 bullet
lists, "**Confidence level:**", the signature block). An optional LLM
enrichment pass is out of scope for Phase 0.

WHAT THIS SCRIPT PRODUCES (same entities/relationships + chunks JSON shape as
every other extract_*.py, so graph_loader.py/embedder.py need no changes):
  * ONE FirmOpinion node. Text properties use the names the tax pack's
    retrieval actually reads (config.py's opinion_expert_views CategorySpec:
    opinion_conclusion / analysis / facts / questions / reliance_clauses) and
    the names its PII masker masks (ALWAYS_MASKED_FIELDS["FirmOpinion"]:
    partner_name / client_name / counterparty_name / client_industry) - the
    config's snake_case convention, NOT schema.json's OWL camelCase
    (opinionConclusion/relianceClauses/partner) where the two disagree,
    because a property name nothing reads is worse than an off-ontology one.
  * One Chunk per substantive body section (facts, issue, legal framework,
    precedents, analysis, conclusion + any unrecognized heading; the
    engagement boilerplate stays node-only), all linked to the FirmOpinion node via
    source_chunk_ids/entity_id - opinion_agent.py finds opinions by Chunk
    vector search -> Chunk.entity_id -> label == "FirmOpinion".
  * cited_reference STUB nodes (framework/core/ingest/provenance.py) for every
    authority the opinion cites, with ids that JOIN to the real ingested
    authority:
      - Act sections -> extract_case_law.resolve_section_citation() (imported,
        not re-implemented) -> chunking.generate_entity_id() ids, namespaced
        to the Act the opinion is written under (ITA1961 default, ITA2025 if
        the opinion's own text cites only the 2025 Act; doc_meta
        cited_act_namespace overrides). "Section 9(1)(vii) & Explanation 2"
        -> Clause stub + Explanation_2_to_<clause id> stub (chunking.py's own
        Explanation id scheme); "2(29A) / 2(42A)" -> two stubs;
        "115JB(2A)-(2C)" -> three; "Section 80-IAC" -> 80IAC; "(as amended)"
        dropped; another Act ("Section 165, Finance Act, 2016") is NOT forced
        onto an Income-tax Act id - kept as other_act_references.
      - cases -> CaseLaw_<extract_case_law._slug(name)>, the exact scheme
        extract_case_law.py uses for cited precedents (and CaseLaw's
        case_name fuzzy resolution in graph_loader applies on top).
      - treaties -> TaxTreaty_<slug(country)> / Article_<n>_<treaty id>, the
        exact scheme extract_treaty.py mints for the real treaty.
      - CBDT circulars -> CBDTCircular_<n>_of_<yyyy>; Income-tax Rules ->
        ITR1962__Rule_<n> (no Rules converter exists yet - see the report's
        open questions).
    Relationships use the ontology's FirmOpinion object properties:
    RELIES_ON_PROVISION (sections, explanations, rules, treaty articles),
    RELIES_ON_CASE (cases), RELIES_ON_AUTHORITY (circulars) - with an
    `authority_kind` edge property, since the ontology's ranges don't
    literally cover Rule/Article/Circular (see validate notes in the report).
  * PROVENANCE on everything: the FirmOpinion node and every edge it asserts
    are ingested_document; every stub (and the stub-to-stub Article PART_OF
    TaxTreaty edge) is cited_reference. graph_loader.py guarantees a stub
    never overwrites a real ingested node and a later real ingestion upgrades
    the stub.

PII (client confidentiality): the real client / counterparty / partner names
live ONLY in client_name / counterparty_name / partner_name (masked at display
by config.ALWAYS_MASKED_FIELDS). Every text property and every Chunk text has
those names pseudonymised to [CLIENT] / [COUNTERPARTY] / [PARTNER] at
ingestion (--no-redact-parties to disable), because chunk text is shown to
the answer model and to other users researching OTHER clients' matters. The
source .md on disk is untouched.

Usage:
    python extract_opinion.py --md ../../../tests/fixtures/opinions/OP-01_withholding_tax_fts_section195.md \\
        --out ../data/opinion_graph/OP-01_graph.json
    python extract_opinion.py --dir ../../../tests/fixtures/opinions --out-dir ../data/opinion_graph
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
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

import chunking  # same directory (domain_packs/tax/converters/)
import extract_case_law  # same directory - resolve_section_citation()/_slug() reused for id parity
import extract_entities  # same directory - attach_pipeline_metadata()/compute_content_hash()
from framework.core.infra import step_logger
from framework.core.ingest import provenance

SCRIPT_DIR = Path(__file__).resolve().parent

FILE_TYPE = "firm_opinion"
DEFAULT_DOC_META = {"source_type": "FirmOpinion", "authority_level": "Firm", "jurisdiction": "India"}
# Income-tax Rules, 1962 id namespace for Rule stubs - no Rules converter
# exists yet, so this is the scheme a future one should adopt (open question).
RULES_NAMESPACE = "ITR1962"
STUB_CONFIDENCE = 0.3  # same value extract_case_law.py/extract_circular.py give stubs


# --- frontmatter ------------------------------------------------------------
# PyYAML is not in requirements.txt and the format here is a tiny, fixed
# subset (scalars, quoted strings, booleans, `[]`, block lists), so a small
# deterministic parser beats adding a dependency.

def _yaml_scalar(raw: str):
    v = raw.strip()
    if not v:
        return ""
    if (v[0] == v[-1]) and v[0] in "\"'" and len(v) >= 2:
        return v[1:-1]
    low = v.lower()
    if low in ("true", "yes"):
        return True
    if low in ("false", "no"):
        return False
    if low in ("null", "~"):
        return None
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [_yaml_scalar(x) for x in re.split(r",(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)", inner)] if inner else []
    return v


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Returns (frontmatter_dict, body). No frontmatter -> ({}, text)."""
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or lines[i].strip() != "---":
        return {}, text
    end = next((j for j in range(i + 1, len(lines)) if lines[j].strip() == "---"), None)
    if end is None:
        return {}, text
    meta: dict = {}
    current_list_key = None
    for line in lines[i + 1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        m_item = re.match(r"^\s+-\s+(.*)$", line)
        if m_item and current_list_key:
            meta[current_list_key].append(_yaml_scalar(m_item.group(1)))
            continue
        m_kv = re.match(r"^([A-Za-z_][\w-]*)\s*:\s*(.*)$", line)
        if not m_kv:
            continue
        key, value = m_kv.group(1), m_kv.group(2)
        if value.strip() == "":
            meta[key] = []
            current_list_key = key
        else:
            meta[key] = _yaml_scalar(value)
            current_list_key = None
    return meta, "\n".join(lines[end + 1:])


# --- body sections ----------------------------------------------------------
# Heading -> property. ORDER MATTERS: "Purpose and Basis of this Opinion" and
# "Conclusion / Opinion" both contain "Opinion"; "Restriction on Use and
# Reliance" must win over anything else. First match wins.
SECTION_KEY_PATTERNS = [
    ("purpose_and_basis", re.compile(r"\bpurpose\b|\bbasis\b|\bscope\b", re.I)),
    ("reliance_clauses", re.compile(r"\brestriction\b|\breliance\b", re.I)),
    ("limitations", re.compile(r"\bcaveats?\b|\bassumptions?\b|\blimitations?\b", re.I)),
    ("precedents_considered", re.compile(r"\bprecedents?\b|\bcase law\b|\bjudicial\b", re.I)),
    ("legal_framework", re.compile(r"legal framework|relevant (?:law|provisions?)|statutory framework", re.I)),
    ("facts", re.compile(r"\bfacts?\b|\bbackground\b", re.I)),
    ("questions", re.compile(r"\bissues?\b|\bquestions?\b", re.I)),
    ("opinion_conclusion", re.compile(r"\bconclusion\b|\bopinion\b|\brecommendation\b", re.I)),
    ("analysis", re.compile(r"\banalysis\b|\bdiscussion\b", re.I)),
]

BOILERPLATE_SECTION_KEYS = frozenset({"purpose_and_basis", "limitations", "reliance_clauses"})

_SECTION_HEADING_RE = re.compile(r"^##\s+(?:(\d+)[.)]\s*)?(.+?)\s*#*\s*$")
_CONFIDENCE_LINE_RE = re.compile(r"^\s*\*{0,2}\s*Confidence level\s*:?\s*\*{0,2}\s*:?\s*(.+?)\s*$", re.I)
_BULLET_RE = re.compile(r"^\s*[-*•]\s+(.*\S)\s*$")
_RULE_LINE_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")


def classify_heading(heading: str) -> str | None:
    for key, pattern in SECTION_KEY_PATTERNS:
        if pattern.search(heading):
            return key
    return None


def split_body_sections(body: str) -> list[dict]:
    """[{"number", "heading", "key", "text"}] for every level-2 heading. A
    section ends at the next level-2 heading OR the first horizontal rule
    (the signature block / footer always sits after a `---`)."""
    sections: list[dict] = []
    current = None
    stopped = False
    for line in body.splitlines():
        m = _SECTION_HEADING_RE.match(line)
        if m and not line.startswith("###"):
            current = {"number": m.group(1), "heading": m.group(2).strip(), "lines": []}
            current["key"] = classify_heading(current["heading"])
            sections.append(current)
            stopped = False
            continue
        if current is None or stopped:
            continue
        if _RULE_LINE_RE.match(line):
            stopped = True
            continue
        current["lines"].append(line)
    for s in sections:
        s["text"] = "\n".join(s.pop("lines")).strip()
    return sections


def _bullets(text: str) -> list[str]:
    return [m.group(1).strip() for m in map(_BULLET_RE.match, (text or "").splitlines()) if m]


# --- header/signature fallbacks (PDF/TXT opinions without frontmatter) -------

def _header_fallbacks(body: str) -> dict:
    out: dict = {}
    m = re.search(r"^\s*\**Date\**\s*:\s*\**\s*(.+?)\s*$", body, re.M | re.I)
    if m:
        out["date"] = m.group(1).strip()
    m = re.search(r"^\s*#*\s*\**Sub(?:ject)?\**\s*[:.-]\s*(.+?)\s*\**\s*$", body, re.M | re.I)
    if m:
        out["title"] = m.group(1).strip()
    m = re.search(r"^\s*To,?\s*$\n^\s*(\S.*?)\s*$", body, re.M)
    if m and not m.group(1).startswith("["):
        out["client"] = m.group(1).strip()
    m = re.search(r"\*\*For\s+(.+?)\*\*", body)
    if m:
        out["firm"] = m.group(1).strip()
    m = re.search(r"^\s*(\S[^\n]*?)\s*\n\s*\**Partner\**\s*$", body, re.M)
    if m:
        out["partner"] = m.group(1).strip()
    m = re.search(r"\b(OP-\d+)\b", body)
    if m:
        out["opinion_id"] = m.group(1)
    return out


# --- derived fields ---------------------------------------------------------

_DATE_FORMATS = ("%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y",
                 "%d %B, %Y")


def parse_date_iso(raw: str) -> str:
    """ISO yyyy-mm-dd, or "" when the format isn't recognized (leave unset
    rather than guess - apply_metadata_filters() compares effective_from as
    an ISO string)."""
    s = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", (raw or "").strip())
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return ""


# Normalized band from the free-text confidence statement - the EARLIEST
# matching phrase wins ("Moderate; fairly arguable if ..." -> moderate,
# "High degree of comfort on characterisation; ... scrutiny" -> high).
_CONFIDENCE_BANDS = [
    ("more_likely_than_not", re.compile(r"more likely than not|\breasonable\b", re.I)),
    ("fairly_arguable", re.compile(r"fairly arguable|\barguable\b", re.I)),
    ("moderate", re.compile(r"\bmoderate\b", re.I)),
    ("should", re.compile(r"^\s*should\b", re.I)),
    ("will", re.compile(r"^\s*will\b", re.I)),
    ("high", re.compile(r"\bhigh\b|\bstrong\b", re.I)),
    ("low", re.compile(r"\blow\b|\bweak\b|\bat risk\b", re.I)),
]


def derive_confidence_band(confidence_level: str) -> str:
    best = None
    for band, pattern in _CONFIDENCE_BANDS:
        m = pattern.search(confidence_level or "")
        if m and (best is None or m.start() < best[0]):
            best = (m.start(), band)
    return best[1] if best else ""


# --- party-name pseudonymisation (see module docstring "PII") -----------------

_CORPORATE_SUFFIX_RE = re.compile(r"\b(?:Ltd|Limited|LLP|Inc|Corp|Corporation|Pte|GmbH|B\.?V|LLC|PLC|Co\.|Company|"
                                  r"Holdings|Trust|Society|Partners)\b", re.I)
_TRAILING_SUFFIX_RE = re.compile(r"(?:\s+(?:Private|Pvt\.?|Pte\.?|Public|Ltd\.?|Limited|LLP|Inc\.?|Corp\.?|"
                                 r"Corporation|GmbH|B\.?V\.?|LLC|PLC|Co\.?))+\s*$", re.I)
_NOT_A_NAME_RE = re.compile(r"^\s*(?:not applicable|n/?a|none|nil|unknown)\b", re.I)
_GENERIC_FIRST_WORDS = {"the", "india", "indian", "global", "international", "national", "new", "first",
                        "united", "general", "resident", "non-resident", "unrelated", "not", "a", "an"}


# 2026-10-14: review/phase0 PII bug - a one-word party ("Infosys", "Zomato")
# has no corporate suffix and < 2 capitalised words, so it was never treated
# as a name and never pseudonymised. A single capitalised token is now a name
# unless it is one of these generic descriptors (a description like
# "Individual" must not turn every "individual" in the text into [COUNTERPARTY]).
_SINGLE_WORD_DESCRIPTORS = _GENERIC_FIRST_WORDS | {
    "individual", "company", "companies", "firm", "client", "assessee", "taxpayer", "resident",
    "nonresident", "partnership", "trust", "government", "foreign", "parent", "subsidiary",
    "buyer", "seller", "investor", "investors", "employee", "employees", "employer", "vendor",
    "customer", "lender", "borrower", "shareholder", "shareholders", "promoter", "promoters",
    "huf", "aop", "boi", "llp", "various", "multiple", "others", "self"}


def looks_like_party_name(name: str) -> bool:
    base = re.sub(r"\([^)]*\)", "", name or "").strip()
    if not base or _NOT_A_NAME_RE.match(base):
        return False
    if _CORPORATE_SUFFIX_RE.search(base):
        return True
    words = base.split()
    if len(words) == 1:
        w = words[0].strip(",.")
        return (len(w) >= 3 and w[:1].isupper() and re.fullmatch(r"[A-Za-z][A-Za-z&.'-]*", w) is not None
                and w.lower() not in _SINGLE_WORD_DESCRIPTORS)
    return sum(1 for w in words if w[:1].isupper()) >= 2


def party_variants(name: str, organisation: bool = True) -> list[str]:
    if not looks_like_party_name(name):
        return []
    name = name.strip()
    variants = {name}
    base = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip().rstrip(",")
    if base:
        variants.add(base)
        base_no_country = re.sub(r",\s*[A-Z][A-Za-z ]+$", "", base).strip()
        if base_no_country:
            variants.add(base_no_country)
        # "Amalgam Labs Limited" is usually referred to as "Amalgam Labs" later on
        short = _TRAILING_SUFFIX_RE.sub("", base_no_country or base).strip().rstrip(",")
        if organisation and short and " " in short and short != base:
            variants.add(short)
        first = base.split()[0].strip(",.")
        if organisation and len(first) >= 4 and first[:1].isupper() and first.lower() not in _GENERIC_FIRST_WORDS:
            variants.add(first)
    return sorted(variants, key=len, reverse=True)


def _full_name_forms(name: str) -> set[str]:
    """The party's name exactly as given (with/without a trailing parenthetical),
    lower-cased - as opposed to DERIVED short forms such as its first word."""
    name = (name or "").strip()
    base = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip().rstrip(",")
    return {f.lower() for f in (name, base) if f}


def build_redactions(client: str, counterparty: str, partner: str, protected: str = "") -> list[tuple[str, str]]:
    """`protected`: text of the cited authorities - a single-word short form
    (e.g. "Apollo") that also occurs in a cited case name is NOT used as a
    variant, so pseudonymising the client never mangles an authority.

    2026-10-14: that check is now case-insensitive, and it never drops the
    party's OWN full name - a one-word client "Infosys" that also occurs in a
    cited "Infosys BPO Ltd v. CIT" would otherwise never be pseudonymised at
    all. redact(..., protected=protected_phrases(...)) instead leaves the
    cited-authority text itself intact for single-word variants."""
    pairs = [(v, "[CLIENT]") for v in party_variants(client)]
    pairs += [(v, "[COUNTERPARTY]") for v in party_variants(counterparty)]
    pairs += [(v, "[PARTNER]") for v in party_variants(partner, organisation=False)]
    full = _full_name_forms(client) | _full_name_forms(counterparty) | _full_name_forms(partner)
    pairs = [(v, t) for v, t in pairs
             if " " in v or v.lower() in full
             or not re.search(r"(?<!\w)" + re.escape(v) + r"(?!\w)", protected or "", re.I)]
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


def protected_phrases(parsed: dict) -> list[str]:
    """Cited-authority strings redact() must leave intact: each raw citation,
    plus a case's bare name ("Apollo Tyres Ltd v. CIT") as it usually appears
    in running text."""
    phrases = set()
    for raw in (parsed.get("cases_cited_raw") or []) + (parsed.get("sections_cited_raw") or []) \
            + (parsed.get("treaty_raw") or []):
        raw = (raw or "").strip()
        if not raw:
            continue
        phrases.add(raw)
        bare = re.split(r"\s*[,(\[]", raw, maxsplit=1)[0].strip()
        if bare and bare != raw and _CASE_RE.search(bare):
            phrases.add(bare)
    return sorted(phrases, key=len, reverse=True)


def redact(text: str, redactions: list[tuple[str, str]], protected=()) -> str:
    """2026-10-14: case-insensitive (review/phase0 PII bug: "ZOMATO LIMITED" /
    "INFOSYS" survived a case-sensitive replace); word-boundary safe as before.
    Multi-word variants are replaced everywhere (unchanged behaviour); single-
    word variants everywhere EXCEPT inside the `protected` cited-authority
    phrases, so "Infosys BPO Ltd v. CIT" stays a usable citation."""
    if not text or not redactions:
        return text

    def sub(t: str, variant: str, token: str) -> str:
        return re.sub(r"(?<![\w\[])" + re.escape(variant) + r"(?!\w)", token, t, flags=re.I)

    for variant, token in redactions:
        if " " in variant:
            text = sub(text, variant, token)
    single = [p for p in redactions if " " not in p[0]]
    if not single:
        return text
    held: list[str] = []

    def hold(m):
        held.append(m.group(0))
        return f"{len(held) - 1}"

    for phrase in protected or ():
        if phrase:
            text = re.sub(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", hold, text, flags=re.I)
    for variant, token in single:
        text = sub(text, variant, token)
    return re.sub("(\\d+)", lambda m: held[int(m.group(1))], text)


# --- cited-authority classification + id resolution ---------------------------

_CASE_RE = re.compile(r"\sv(?:s)?\.?\s", re.I)
_CIRCULAR_RE = re.compile(r"Circular\s+No\.?\s*(\d+)\s*(?:/|of)\s*(\d{4})", re.I)
_TREATY_RE = re.compile(r"India\s*[-–—]\s*(.+?)\s+(DTAA|Tax Treaty|Treaty|Convention|Agreement)\b"
                        r"(?:.*?\bArticle\s+(\d+[A-Z]?)(?:\s*\(([^)]*)\))?)?", re.I)
_OTHER_ACT_RE = re.compile(r",\s*((?:[A-Z][\w.()-]*\s+)*Act),?\s*(\d{4})")
_MARKER_RE = re.compile(r"^(?:\d{1,3}[A-Za-z]{0,4}|[a-z]{1,5})$")
_ACT_YEAR_RE = re.compile(r"income.?tax act,?\s*(1961|2025)", re.I)


def _keep_marker_parens(text: str) -> str:
    """Drops parentheticals that aren't citation markers - "(as amended)"
    goes, "(viib)"/"(2A)"/"(1C)" stay."""
    return re.sub(r"\s*\(([^)]*)\)", lambda m: f"({m.group(1).strip()})" if _MARKER_RE.match(m.group(1).strip()) else "",
                  text).strip()


def split_section_citation(raw: str) -> tuple[list[str], str | None, str | None]:
    """'Section 9(1)(vii) & Explanation 2' -> (['9(1)(vii)'], '2', None).
    Returns (citation_numbers, explanation_no, other_act). other_act is set
    (and numbers empty) when the section belongs to a different Act."""
    s = (raw or "").strip()
    explanation = None
    m = re.match(r"^\s*Explanation\s+(\w+)\s+(?:to|of)\s+(.*)$", s, re.I)
    if m:
        explanation, s = m.group(1), m.group(2)
    else:
        m = re.search(r"\s*(?:&|,|\band\b)\s*Explanation\s+(\w+)\s*$", s, re.I)
        if m:
            explanation, s = m.group(1), s[:m.start()]
    s = re.sub(r"^\s*(?:Sections?|Sec\.?|S\.)\s+", "", s, flags=re.I)
    m = _OTHER_ACT_RE.search(s)
    if m and not re.search(r"income.?tax", m.group(1), re.I):
        return [], explanation, f"{m.group(1)}, {m.group(2)}"
    s = re.sub(r",?\s*(?:of the\s+)?Income.?tax Act,?\s*\d{4}", "", s, flags=re.I)
    s = _keep_marker_parens(s)
    numbers: list[str] = []
    base_section = None
    for piece in [p.strip() for p in s.split("/") if p.strip()]:
        piece = re.sub(r"^(\d+)\s*-\s*([A-Z]+)", r"\1\2", piece)  # "80-IAC" -> "80IAC"
        if piece.startswith("(") and base_section:
            piece = base_section + piece
        rng = re.match(r"^(.*?)\((\d+)([A-Z])\)\s*-\s*\((\d+)([A-Z])\)$", piece)
        if rng and rng.group(2) == rng.group(4) and rng.group(3) <= rng.group(5):
            for c in range(ord(rng.group(3)), ord(rng.group(5)) + 1):
                numbers.append(f"{rng.group(1)}({rng.group(2)}{chr(c)})")
        else:
            numbers.append(piece.replace(" ", ""))
        mb = re.match(r"^(\d+[A-Z]{0,4})", numbers[-1]) if numbers else None
        if mb:
            base_section = mb.group(1)
    return numbers, explanation, None


def classify_citation(raw: str) -> str:
    s = (raw or "").strip()
    if not s or _NOT_A_NAME_RE.match(s):
        return "skip"
    if _CIRCULAR_RE.search(s):
        return "circular"
    if re.match(r"^\s*(?:Explanation\s+\w+\s+(?:to|of)\s+)?(?:Sections?|Sec\.?)\s+\d", s, re.I):
        return "section"
    if re.match(r"^\s*Rules?\s+\d", s, re.I):
        return "rule"
    if _TREATY_RE.search(s):
        return "treaty"
    if _CASE_RE.search(s):
        return "case"
    return "other"


def detect_act_namespace(body: str, override: str | None = None) -> str:
    """ITA1961 unless the opinion's own text cites ONLY the 2025 Act."""
    if override:
        return override
    years = {m.group(1) for m in _ACT_YEAR_RE.finditer(body or "")}
    if years == {"2025"}:
        return "ITA2025"
    return extract_case_law.DEFAULT_CITED_ACT_NAMESPACE


def _treaty_slug(text: str) -> str:
    # Identical to extract_treaty.py's _slug() (not imported only because
    # that module pulls the treaty LLM prompt machinery in at import time).
    return re.sub(r"[^A-Za-z0-9]+", "_", text or "").strip("_")[:60]


def _dedupe(items: list[str]) -> list[str]:
    out, seen = [], set()
    for it in items:
        key = re.sub(r"\s+", " ", str(it or "")).strip().lower()
        if key and key not in seen:
            seen.add(key)
            out.append(str(it).strip())
    return out


def _as_list(value) -> list[str]:
    if value is None or value is False:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)] if str(value).strip() else []


# --- parse -----------------------------------------------------------------

def parse_opinion(text: str, source_file: str = "") -> dict:
    """Pure, deterministic structural parse of one opinion document."""
    fm, body = parse_frontmatter(text)
    fallback = _header_fallbacks(body)
    sections = split_body_sections(body)
    by_key: dict[str, str] = {}
    unrecognized = []
    for s in sections:
        if s["key"] and s["key"] not in by_key:
            by_key[s["key"]] = s["text"]
        elif not s["key"]:
            unrecognized.append(s["heading"])

    conclusion_text = by_key.get("opinion_conclusion", "")
    confidence_from_body = ""
    kept = []
    for line in conclusion_text.splitlines():
        m = _CONFIDENCE_LINE_RE.match(line)
        if m and not confidence_from_body:
            confidence_from_body = m.group(1).strip().strip("*").strip()
            continue
        kept.append(line)
    if "opinion_conclusion" in by_key:
        by_key["opinion_conclusion"] = "\n".join(kept).strip()
    if not confidence_from_body:
        m = re.search(r"\*\*Confidence level:?\*\*:?\s*(.+)", body, re.I)
        confidence_from_body = m.group(1).strip() if m else ""

    framework_bullets = _bullets(by_key.get("legal_framework", ""))
    treaty_from_body = []
    statutory_from_body = []
    for b in framework_bullets:
        m = re.match(r"^\*\*Treaty provisions?:?\*\*:?\s*(.+)$", b, re.I)
        (treaty_from_body if m else statutory_from_body).append(m.group(1).strip() if m else b)

    def pick(key):
        v = fm.get(key)
        return v if v not in (None, "", []) else fallback.get(key, "")

    return {
        "frontmatter": fm,
        "has_frontmatter": bool(fm),
        "opinion_id": str(pick("opinion_id") or Path(source_file).stem or "").strip(),
        "title": str(pick("title") or "").strip(),
        "client": str(pick("client") or "").strip(),
        "counterparty": str(fm.get("counterparty") or "").strip(),
        "category": str(fm.get("category") or "").strip(),
        "partner": str(pick("partner") or "").strip(),
        "firm": str(pick("firm") or "").strip(),
        "date": str(pick("date") or "").strip(),
        "confidence_level": str(fm.get("confidence_level") or confidence_from_body or "").strip(),
        "fictitious": bool(fm.get("fictitious")) or "FICTITIOUS" in body,
        "confidential": bool(re.search(r"PRIVATE\s*&\s*CONFIDENTIAL", body, re.I)),
        "sections_cited_raw": _dedupe(_as_list(fm.get("income_tax_sections")) + statutory_from_body),
        "cases_cited_raw": _dedupe(_as_list(fm.get("case_law_cited")) + _bullets(by_key.get("precedents_considered", ""))),
        "treaty_raw": _dedupe(_as_list(fm.get("treaty")) + treaty_from_body),
        "supersedes": _as_list(fm.get("supersedes")),
        "superseded_by": _as_list(fm.get("superseded_by")),
        "sections": sections,
        "text_by_key": by_key,
        "unrecognized_headings": unrecognized,
        "body": body,
    }


# --- graph build -----------------------------------------------------------

def _stub(entity_id: str, label: str, props: dict, cited_in: str, run_timestamp: str) -> dict:
    base = {
        "confidence": STUB_CONFIDENCE,
        "extraction_notes": f"Stub created from a citation in firm opinion {cited_in}; not verified against "
                             f"the authority's own text - filled in properly if/when it is itself ingested.",
        "jurisdiction": "India",
        "ingestion_date": run_timestamp,
        "node_id": entity_id,
    }
    base.update(props)
    return {"id": entity_id, "type": label, "properties": provenance.mark_stub(base, cited_in=cited_in)}


def build_graph_from_opinion(parsed: dict, source_file: str, doc_meta: dict | None = None,
                             source_md_path: str | None = None, file_type: str = FILE_TYPE,
                             redact_parties: bool = True) -> dict:
    doc_meta = {**DEFAULT_DOC_META, **(doc_meta or {})}
    run_timestamp = datetime.now(timezone.utc).isoformat()
    entities: list[dict] = []
    relationships: list[dict] = []
    ambiguities: list[dict] = []
    stub_ids: set[str] = set()

    opinion_key = parsed["opinion_id"] or Path(source_file).stem
    opinion_node_id = f"FirmOpinion_{extract_case_law._slug(opinion_key)}"
    doc_id = doc_meta.get("document_id") or f"opinion_{extract_case_law._slug(opinion_key)}"
    title = parsed["title"] or opinion_key
    protected = " | ".join(parsed["cases_cited_raw"] + parsed["sections_cited_raw"] + parsed["treaty_raw"])
    redactions = build_redactions(parsed["client"], parsed["counterparty"], parsed["partner"],
                                  protected=protected) if redact_parties else []
    keep = protected_phrases(parsed)
    text = {k: redact(v, redactions, keep) for k, v in parsed["text_by_key"].items()}
    doc_meta.setdefault("document_id", doc_id)
    # 2026-10-14: phase0/review-pii - document_name is copied onto the node and
    # every chunk and shown in source labels to the answer model and the UI
    # (get_document_details), so it must carry the REDACTED title, not the raw one.
    doc_meta.setdefault("document_name", f"{opinion_key} - {redact(title, redactions, keep)}"
                        if parsed["title"] else opinion_key)
    doc_meta.setdefault("file_type", file_type)
    doc_meta.setdefault("source_md_path", source_md_path or "")
    namespace = detect_act_namespace(parsed["body"], doc_meta.get("cited_act_namespace"))

    def add_stub(stub: dict):
        if stub["id"] not in stub_ids:
            stub_ids.add(stub["id"])
            entities.append(stub)

    def rel(rel_type: str, target: str, prov: str = provenance.INGESTED_DOCUMENT, source: str = opinion_node_id, **props):
        relationships.append({"type": rel_type, "source": source, "target": target,
                              "properties": {"provenance": prov, **{k: v for k, v in props.items() if v not in (None, "")}}})

    # --- statutory provisions (+ anything mis-filed in either list) ----------
    other_act_refs, other_authorities, sections_cited, rules_cited, circulars_cited = [], [], [], [], []
    case_names: list[str] = []
    treaty_refs: list[str] = []
    all_cited = [(c, "sections") for c in parsed["sections_cited_raw"]] + \
                [(c, "cases") for c in parsed["cases_cited_raw"]] + \
                [(c, "treaty") for c in parsed["treaty_raw"]]
    for raw, field in all_cited:
        kind = classify_citation(raw)
        if kind == "skip":
            continue
        if kind == "section":
            numbers, explanation, other_act = split_section_citation(raw)
            if other_act:
                other_act_refs.append(raw)
                ambiguities.append({"type": "other_act_section", "raw": raw, "act": other_act,
                                    "note": "not an Income-tax Act provision - no stub minted"})
                continue
            sections_cited.append(raw)
            for i, number in enumerate(numbers):
                target_id, parts = extract_case_law.resolve_section_citation(number, namespace=namespace)
                if not target_id:
                    ambiguities.append({"type": "unresolved_section_citation", "raw": raw, "parsed": number})
                    continue
                label = target_id.split("__", 1)[-1].split("_", 1)[0]
                add_stub(_stub(target_id, label, {
                    "full_citation": chunking.generate_full_citation(parts["section"], parts["subsection"],
                                                                     parts["clause"], parts["sub_clause"]),
                    "section": parts["section"], "subsection": parts["subsection"],
                    "clause": parts["clause"], "sub_clause": parts["sub_clause"],
                    "act_namespace": namespace, "cited_as": raw,
                }, doc_id, run_timestamp))
                rel("RELIES_ON_PROVISION", target_id, authority_kind="statutory_provision", cited_as=raw,
                    act_namespace=namespace)
                if explanation and i == 0:
                    expl_id = f"Explanation_{explanation}_to_{target_id}"
                    add_stub(_stub(expl_id, "Explanation", {
                        "full_citation": f"Explanation {explanation} to "
                                         + chunking.generate_full_citation(parts["section"], parts["subsection"],
                                                                           parts["clause"], parts["sub_clause"]),
                        "act_namespace": namespace, "cited_as": raw,
                    }, doc_id, run_timestamp))
                    rel("RELIES_ON_PROVISION", expl_id, authority_kind="statutory_explanation", cited_as=raw,
                        act_namespace=namespace)
        elif kind == "rule":
            rules_cited.append(raw)
            numbers, _expl, _other = split_section_citation(re.sub(r"^\s*Rules?\s+", "Section ", raw, flags=re.I))
            for number in numbers:
                m = extract_case_law.SECTION_CITATION_RE.match(number)
                if not m:
                    ambiguities.append({"type": "unresolved_rule_citation", "raw": raw, "parsed": number})
                    continue
                parts = [m.group(1)] + re.findall(r"\(([^)]+)\)", m.group(2) or "")
                rule_id = f"{RULES_NAMESPACE}__Rule_{'_'.join(parts)}"
                add_stub(_stub(rule_id, "Rule", {
                    "full_citation": f"Rule {number} of the Income-tax Rules, 1962",
                    "rule_number": number, "cited_as": raw,
                }, doc_id, run_timestamp))
                rel("RELIES_ON_PROVISION", rule_id, authority_kind="rule", cited_as=raw)
        elif kind == "circular":
            m = _CIRCULAR_RE.search(raw)
            number, year = str(int(m.group(1))), m.group(2)
            circ_id = f"CBDTCircular_{number}_of_{year}"
            circulars_cited.append(raw)
            add_stub(_stub(circ_id, "CBDTCircular", {
                "full_citation": f"CBDT Circular No. {number}/{year}", "circular_number": f"{number} of {year}",
                "circular_year": int(year), "cited_as": raw,
            }, doc_id, run_timestamp))
            rel("RELIES_ON_AUTHORITY", circ_id, authority_kind="circular", cited_as=raw)
        elif kind == "treaty":
            m = _TREATY_RE.search(raw)
            country, treaty_type = m.group(1).strip(), m.group(2).upper() if m.group(2).lower() == "dtaa" else m.group(2)
            treaty_id = f"TaxTreaty_{_treaty_slug(country)}"
            add_stub(_stub(treaty_id, "TaxTreaty", {
                "full_citation": f"India-{country} {treaty_type}", "treatyCountry": country,
                "treaty_type": treaty_type, "cited_as": raw,
            }, doc_id, run_timestamp))
            article = m.group(3)
            if article:
                art_id = f"Article_{article}_{treaty_id}"
                add_stub(_stub(art_id, "Article", {
                    "full_citation": f"Article {article}" + (f" ({m.group(4)})" if m.group(4) else "")
                                     + f" of India-{country} {treaty_type}",
                    "article_number": article, "article_title": m.group(4) or "", "cited_as": raw,
                }, doc_id, run_timestamp))
                # Structural edge between two stubs, inferred from the citation
                # string alone -> cited_reference (graph_loader never lets it
                # overwrite the real treaty's own PART_OF edge).
                rel("PART_OF", treaty_id, prov=provenance.CITED_REFERENCE, source=art_id)
                rel("RELIES_ON_PROVISION", art_id, authority_kind="treaty_article", cited_as=raw)
                treaty_refs.append(f"India-{country} {treaty_type} Article {article}")
            else:
                rel("RELIES_ON_PROVISION", treaty_id, authority_kind="treaty", cited_as=raw)
                treaty_refs.append(f"India-{country} {treaty_type}")
        elif kind == "case":
            case_names.append(raw)
            case_id = f"CaseLaw_{extract_case_law._slug(raw)}"
            add_stub(_stub(case_id, "CaseLaw", {"case_name": raw, "full_citation": raw}, doc_id, run_timestamp))
            rel("RELIES_ON_CASE", case_id, cited_as=raw)
        else:
            other_authorities.append(raw)
            ambiguities.append({"type": "unclassified_authority", "raw": raw, "field": field,
                                "note": "kept as other_authorities_cited - no id scheme to resolve it onto"})

    # --- opinion-to-opinion supersession (optional frontmatter keys) ---------
    for field, forward in (("supersedes", True), ("superseded_by", False)):
        for other in parsed[field]:
            other_id = f"FirmOpinion_{extract_case_law._slug(other)}"
            add_stub(_stub(other_id, "FirmOpinion", {"opinion_id": other, "full_citation": other}, doc_id, run_timestamp))
            newer, older = (opinion_node_id, other_id) if forward else (other_id, opinion_node_id)
            rel("SUPERSEDES", older, source=newer)
            rel("SUPERSEDED_BY", newer, source=older)  # opinion_agent.py queries this direction

    date_iso = parse_date_iso(parsed["date"])
    category_parts = [c.strip() for c in re.split(r"\s*/\s*", parsed["category"]) if c.strip()]
    opinion_props = {
        "opinion_id": parsed["opinion_id"],
        "title": redact(title, redactions, keep),
        "full_citation": f"{parsed['opinion_id']} - {redact(title, redactions, keep)}" if parsed["opinion_id"] else redact(title, redactions, keep),
        "category": parsed["category"],
        "legal_concepts": category_parts,
        "opinion_date": parsed["date"],
        **({"date": date_iso, "effective_from": date_iso} if date_iso else {}),
        "confidence_level": parsed["confidence_level"],
        "confidence_band": derive_confidence_band(parsed["confidence_level"]),
        # Real names ONLY here - masked at display via config.ALWAYS_MASKED_FIELDS.
        "client_name": parsed["client"],
        "counterparty_name": parsed["counterparty"] if looks_like_party_name(parsed["counterparty"]) else "",
        "counterparty_description": "" if looks_like_party_name(parsed["counterparty"]) else parsed["counterparty"],
        "partner_name": parsed["partner"],
        "firm_name": parsed["firm"],
        "purpose_and_basis": text.get("purpose_and_basis", ""),
        "facts": text.get("facts", ""),
        "questions": text.get("questions", ""),
        "legal_framework": text.get("legal_framework", ""),
        "precedents_considered": text.get("precedents_considered", ""),
        "analysis": text.get("analysis", ""),
        "opinion_conclusion": text.get("opinion_conclusion", ""),
        "limitations": text.get("limitations", ""),
        "reliance_clauses": text.get("reliance_clauses", ""),
        "hasText": text.get("opinion_conclusion", "") or text.get("analysis", ""),
        "sections_cited": sections_cited,
        "rules_cited": rules_cited,
        "circulars_cited": circulars_cited,
        "cases_cited": case_names,
        "treaty_references": treaty_refs,
        "other_act_references": other_act_refs,
        "other_authorities_cited": other_authorities,
        "statute_regime": namespace,
        "opinion_jurisdiction": "India",
        "confidentiality_classification": "Private & Confidential" if parsed["confidential"] else "",
        "is_fictitious": parsed["fictitious"],
        "parties_redacted_in_text": bool(redactions),
        "provenance": provenance.INGESTED_DOCUMENT,
        "confidence": 1.0,  # deterministic parse, not LLM-inferred (cf. CaseLaw's 0.8)
        "extraction_notes": f"Deterministically parsed from {source_file} via extract_opinion.py"
                            + ("" if parsed["has_frontmatter"] else " (no frontmatter - body fallbacks used)"),
        "source_document": source_file,
        "content_hash": extract_entities.compute_content_hash(parsed["body"]),
    }
    if parsed["unrecognized_headings"]:
        ambiguities.append({"type": "unrecognized_section_heading", "headings": parsed["unrecognized_headings"]})
    for key in ("facts", "questions", "analysis", "opinion_conclusion"):
        if not opinion_props[key]:
            ambiguities.append({"type": "missing_section", "section": key})
    opinion_entity = {"id": opinion_node_id, "type": "FirmOpinion", "properties": opinion_props}
    extract_entities.attach_pipeline_metadata(opinion_entity, doc_meta, run_timestamp, None)
    entities.insert(0, opinion_entity)

    # --- chunks: one per SUBSTANTIVE body section, all owned by the FirmOpinion
    # node. The engagement boilerplate (purpose/basis, caveats, reliance
    # restriction) is near-identical across every opinion a firm issues - as
    # chunks it would crowd opinion_agent's top-k vector search with 1 copy per
    # opinion of the same paragraph - so it stays on the node only
    # (reliance_clauses/limitations/purpose_and_basis properties).
    chunks = []
    for n, s in enumerate(parsed["sections"], start=1):
        if s["key"] in BOILERPLATE_SECTION_KEYS:
            continue
        body_text = redact(s["text"], redactions, keep)
        if s["key"] == "opinion_conclusion":
            body_text = text.get("opinion_conclusion", body_text)
            if parsed["confidence_level"]:
                body_text = f"{body_text}\n\nConfidence level: {parsed['confidence_level']}".strip()
        if not body_text.strip():
            continue
        chunk_id = f"opinion_{opinion_node_id}_{n}"
        chunks.append({
            "chunk_id": chunk_id, "text": body_text,
            "short_title": f"{parsed['opinion_id'] or opinion_key} - {s['heading']}",
            "chunk_type": "opinion_section", "element_type": s["key"] or "opinion_section",
            "full_citation": f"{opinion_props['full_citation']} — {s['heading']}",
            "entity_id": opinion_node_id, "file_type": doc_meta.get("file_type", file_type),
            "cases_cited": case_names, "treaty_references": treaty_refs,
            "other_act_references": other_act_refs, "legal_concepts": category_parts,
        })
    opinion_props["source_chunk_ids"] = ",".join(c["chunk_id"] for c in chunks)

    return {"entities": entities, "relationships": relationships, "ambiguities": ambiguities,
            "opinion_id": opinion_node_id, "chunks": chunks}


def validate_against_schema(graph: dict, schema: dict) -> list[str]:
    """Same check as every other converter's validate_against_schema()."""
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


def process_one(md_path: Path, doc_meta: dict | None = None, file_type: str = FILE_TYPE,
                redact_parties: bool = True) -> tuple[dict, dict, dict]:
    text = md_path.read_text(encoding="utf-8", errors="replace")
    with step_logger.Step(f"Parsing firm opinion {md_path.name} (deterministic, no LLM)"):
        parsed = parse_opinion(text, md_path.name)
        graph = build_graph_from_opinion(parsed, md_path.name, doc_meta=doc_meta,
                                         source_md_path=str(md_path.resolve()), file_type=file_type,
                                         redact_parties=redact_parties)
        stats = {"entities": len(graph["entities"]), "relationships": len(graph["relationships"]),
                 "chunks": len(graph["chunks"]), "ambiguities": len(graph["ambiguities"]),
                 "stubs": sum(1 for e in graph["entities"]
                              if e["properties"].get("provenance") == provenance.CITED_REFERENCE)}
        step_logger.log(f"  {stats['entities']} entities ({stats['stubs']} cited_reference stubs), "
                         f"{stats['relationships']} relationships, {stats['chunks']} chunks, "
                         f"{stats['ambiguities']} ambiguities flagged")
    return parsed, graph, stats


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--md", type=Path, default=None, help="Single opinion .md/.txt file")
    parser.add_argument("--out", type=Path, default=None, help="Output graph JSON path (with --md)")
    parser.add_argument("--dir", type=Path, default=None, help="Process every .md/.txt file in this folder")
    parser.add_argument("--out-dir", type=Path, default=None, help="Output folder (with --dir)")
    parser.add_argument("--chunks-out", type=Path, default=None)
    parser.add_argument("--schema", type=Path, default=SCRIPT_DIR.parent / "ontology" / "schema.json")
    parser.add_argument("--file-type", default=FILE_TYPE)
    parser.add_argument("--document-id", default=None)
    parser.add_argument("--document-name", default=None)
    parser.add_argument("--source-type", default=DEFAULT_DOC_META["source_type"])
    parser.add_argument("--authority-level", default=DEFAULT_DOC_META["authority_level"])
    parser.add_argument("--jurisdiction", default=DEFAULT_DOC_META["jurisdiction"])
    parser.add_argument("--cited-act-namespace", default=None,
                        help="Force the Act id-namespace section citations resolve against (default: ITA1961, "
                             "or ITA2025 when the opinion's text cites only the Income-tax Act, 2025).")
    parser.add_argument("--no-redact-parties", action="store_true",
                        help="Keep client/counterparty/partner names in chunk/text properties (see docstring PII).")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    schema = None
    if args.schema.exists():
        with open(args.schema, "r", encoding="utf-8") as f:
            schema = json.load(f)

    def handle(md_path: Path, out_path: Path, chunks_out_path: Path, single: bool):
        doc_meta = {"source_type": args.source_type, "authority_level": args.authority_level,
                    "jurisdiction": args.jurisdiction, "file_type": args.file_type}
        if single and args.document_id:
            doc_meta["document_id"] = args.document_id
        if single and args.document_name:
            doc_meta["document_name"] = args.document_name
        if args.cited_act_namespace:
            doc_meta["cited_act_namespace"] = args.cited_act_namespace
        _parsed, graph, _stats = process_one(md_path, doc_meta, args.file_type, not args.no_redact_parties)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"entities": graph["entities"], "relationships": graph["relationships"]}, f,
                      indent=2, ensure_ascii=False)
        chunks_out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_out_path, "w", encoding="utf-8") as f:
            json.dump(graph["chunks"], f, indent=2, ensure_ascii=False)
        for a in graph["ambiguities"]:
            step_logger.log(f"  ambiguity: {a}")
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
        files = sorted(args.dir.glob("*.md")) + sorted(args.dir.glob("*.txt"))
        if not files:
            raise SystemExit(f"No .md/.txt files found in {args.dir}")
        for md_path in files:
            try:
                handle(md_path, args.out_dir / f"{md_path.stem}_graph.json",
                       args.out_dir / f"{md_path.stem}_chunks.json", single=False)
            except Exception as e:  # one bad opinion never stops the batch
                step_logger.log(f"  ERROR on {md_path.name}: {e} - skipping")
    elif args.md:
        if not args.out:
            raise SystemExit("--out is required with --md")
        handle(args.md, args.out, args.chunks_out or args.out.parent / f"{args.md.stem}_chunks.json", single=True)
    else:
        raise SystemExit("Provide either --md/--out or --dir/--out-dir")


if __name__ == "__main__":
    main()
