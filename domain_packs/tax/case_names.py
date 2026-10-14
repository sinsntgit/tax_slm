"""
Deterministic case-name normalizer for CaseLaw entity resolution.

2026-10-14 (phase0/followups): NEW. The opinion and case-law converters create
cited-case stubs as CaseLaw_<slug(name as cited)> with case_name = the raw
citation, e.g. "Engineering Analysis Centre of Excellence Pvt Ltd v. CIT,
(2021) 432 ITR 471 (SC)", while the ingested judgment carries the name from
its own citation, "Engineering Analysis Centre of Excellence Private Limited v.
The Commissioner of Income Tax, (2021) 2 SCC 1". The CaseLaw resolution key
used to be just normalize(case_name), which scored those two at ~75 - below
entity_resolution.HIGH_SIMILARITY (95) - so the stub never joined the real
judgment and the opinion's RELIES_ON_CASE edge pointed at an empty stub.

normalize_case_name() reduces a citation to the PARTY NAMES only, with the
usual Indian tax-reporting variants unified, so both sides produce the same
resolution key. It is used ONLY for the fuzzy-comparison key (registered on
RESOLUTION_KEY_PROPERTIES["CaseLaw"] via entity_resolution.KeyProperties) -
node ids are never changed. Deterministic (no LLM), per this project's
convention. Deliberately conservative: it only folds spellings of the SAME
designation (CIT <-> Commissioner of Income Tax, UOI <-> Union of India);
different revenue officers (CIT vs ITO vs DCIT) stay different, and the
distinguishing party name is untouched, so CIT v. Samsung Electronics and
CIT v. ZTE Corporation stay far apart.

Steps, in order:
  1. Unicode/quote/dash clean-up; "(P)"/"(Pvt)" before Ltd -> "Pvt".
  2. Drop every (...) and [...] group: years "(2021)", court suffixes "(SC)",
     "(Del HC, 2020)", "(I)", editorial notes "(overruled on ...)".
  3. Cut the reporter-citation tail: at the first comma whose segment
     contains a digit (", 432 ITR 471", ", ITA Nos. 87-89/1992",
     ", 22.06.2005"), at "<vol> <REPORTER> <page>", at "<year> SCC ...",
     or at a trailing bare year.
  4. Unify the versus marker (v., vs., versus) and split into parties.
  5. Per party: M/s, leading "The", "In Re"; revenue designations
     (Commissioner of Income Tax -> cit, Pr. CIT -> pcit, Deputy CIT -> dcit,
     ...); a revenue respondent's ", <location/charge>" suffix
     ("CIT, West Bengal-VII" -> "cit"); company forms (Private Limited / Pvt.
     Ltd. / P. Ltd / Limited -> ltd, Company -> co, Corporation -> corp,
     Incorporated -> inc), "&" -> and.
  6. Lowercase, drop punctuation, collapse runs of single letters
     ("A.P." -> "ap", "B.V." -> "bv") and whitespace.

same_designations() is the matching KeyProperties.compatible veto: the fuzzy
ratio alone scores "Agro Portfolio Ltd v ITO" vs "... v CIT" at ~97 (one short
token differs), so keys naming different revenue officers never merge.
"""
from __future__ import annotations

import re
import unicodedata

_REPORTERS = (r"ITR|SCC|SCR|CTR|ITD|TTJ|SOT|DTR|Taxman|taxmann(?:\.com)?|AIR|BCR|ELT|STC|TLR|ITR\s*\(Trib\)|"
              r"SCC\s+OnLine|SCC\s+Online|ITXA|ITA|F\.\s*\dd|F\.\s*Supp|U\.?S\.?|All\s*ER|WLR|ECR")

_VERSUS_RE = re.compile(r"\s+(?:v|vs|versus)\.?\s+", re.IGNORECASE)

# Spelled-out revenue/government designations -> one canonical token.
# Longest/most specific first (Deputy/Assistant/Principal before plain CIT).
_DESIGNATIONS = [
    (r"(?:principal|pr\.?)\s+(?:commissioner|commr\.?)\s+of\s+income[\s-]*tax|pr\.?\s*cit|pcit", "pcit"),
    (r"(?:deputy|dy\.?)\s+(?:commissioner|commr\.?)\s+of\s+income[\s-]*tax|(?:deputy|dy\.?)\s+cit|dcit", "dcit"),
    (r"(?:assistant|asst\.?)\s+(?:commissioner|commr\.?)\s+of\s+income[\s-]*tax|(?:assistant|asst\.?)\s+cit|acit", "acit"),
    (r"(?:additional|addl\.?)\s+(?:commissioner|commr\.?)\s+of\s+income[\s-]*tax|(?:additional|addl\.?)\s+cit", "addl cit"),
    (r"(?:commissioner|commr\.?)\s+of\s+income[\s-]*tax|cit", "cit"),
    (r"(?:deputy|dy\.?)\s+director\s+of\s+income[\s-]*tax|ddit", "ddit"),
    (r"(?:assistant|asst\.?)\s+director\s+of\s+income[\s-]*tax|adit", "adit"),
    (r"director\s+of\s+income[\s-]*tax|dit", "dit"),
    (r"income[\s-]*tax\s+officer|ito", "ito"),
    (r"union\s+of\s+india|u\.?\s*o\.?\s*i\.?", "uoi"),
    (r"central\s+board\s+of\s+direct\s+taxes|cbdt", "cbdt"),
]
_DESIGNATION_RES = [(re.compile(rf"\b(?:{pat})\b\.?", re.IGNORECASE), tok) for pat, tok in _DESIGNATIONS]
_REVENUE_HEAD_RE = re.compile(r"^(addl cit|pcit|dcit|acit|cit|ddit|adit|dit|ito|uoi|cbdt)\b")

_COMPANY_FORMS = [
    # (?<!\.) so the "P." of an initialism ("Transmission Corporation of A.P.
    # Ltd.") is not read as "Pvt".
    (r"(?<!\.)\b(?:private|pvt\.?|p\.?)\s*(?:limited|ltd\.?)", "ltd"),
    (r"\blimited\b|\bltd\b\.?", "ltd"),
    (r"\bcompany\b|\bco\b\.?", "co"),
    (r"\bcorporation\b|\bcorpn\b\.?|\bcorp\b\.?", "corp"),
    (r"\bincorporated\b|\binc\b\.?", "inc"),
]
_COMPANY_RES = [(re.compile(p, re.IGNORECASE), t) for p, t in _COMPANY_FORMS]


def _clean(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text))
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"[‐-―]", "-", t)
    return re.sub(r"\s+", " ", t).strip().strip('"').strip()


def _cut_citation_tail(t: str) -> str:
    patterns = [
        r",\s*(?=[^,]*\d)",                                                 # ", 432 ITR 471" / ", 22.06.2005"
        rf"\s+\d[\d\-/\.]*\s+(?:{_REPORTERS})\b",                           # "432 ITR 471" without a comma
        rf"\s+(?:19|20)\d{{2}}\s+(?:\(\d+\)\s+)?(?:{_REPORTERS})\b",       # "2009 SCC OnLine Del 835"
        r"\s+(?:19|20)\d{2}\s*$",                                           # trailing bare year
    ]
    cut = len(t)
    for p in patterns:
        m = re.search(p, t, flags=re.IGNORECASE)
        if m and m.start() > 0:
            cut = min(cut, m.start())
    return t[:cut]


def _party(p: str, collapse_revenue: bool = True) -> str:
    p = p.strip(" ,.;:")
    p = re.sub(r"^\s*m/s\.?\s*", "", p, flags=re.IGNORECASE)
    p = re.sub(r"\bin\s+re\b\.?:?", " ", p, flags=re.IGNORECASE)
    p = re.sub(r"^\s*the\s+", "", p, flags=re.IGNORECASE)
    for rx, tok in _DESIGNATION_RES:
        p = rx.sub(f" {tok} ", p)
    p = re.sub(r"\s+", " ", p).strip(" ,.")
    # A revenue party's charge/location ("CIT, West Bengal-VII", "Pr. CIT-2,
    # Mumbai", "Deputy CIT, Special Bench ITAT Delhi", "ITO Ward 1(2)") is the
    # officer's posting, not a different case - keep only the designation.
    # Only for revenue parties: a private party's trailing words/commas are
    # part of its name ("Dassault Systems, K.K.").
    # Never for a single-party string ("CBDT Circular No. 7/2017" must not
    # collapse to "cbdt" and merge with every other CBDT document).
    m = _REVENUE_HEAD_RE.match(p.lower()) if collapse_revenue else None
    if m:
        p = m.group(1)
    p = p.replace("&", " and ")
    for rx, tok in _COMPANY_RES:
        p = rx.sub(f" {tok} ", p)
    p = p.lower()
    p = re.sub(r"[^\w\s]", " ", p)
    p = re.sub(r"_", " ", p)
    # collapse runs of single letters left by initials: "a p" -> "ap", "b v" -> "bv"
    p = re.sub(r"\b(?:[a-z]\s+)+[a-z]\b", lambda m: m.group(0).replace(" ", ""), p)
    p = re.sub(r"\s+", " ", p).strip()
    p = re.sub(r"^the\s+", "", p)
    return p


_DESIGNATION_TOKENS = {"pcit", "dcit", "acit", "addl", "cit", "ddit", "adit", "dit", "ito", "uoi", "cbdt"}


def same_designations(key_a: str, key_b: str) -> bool:
    """KeyProperties.compatible veto for CaseLaw: two keys whose revenue/
    government designations differ ("... v ito" vs "... v cit", "v dcit" vs
    "v cit") are never the same case, however high the fuzzy score - those
    are different proceedings (an ITAT appeal against the AO vs the HC appeal
    by the CIT). A side that names no designation at all doesn't veto (the
    fuzzy score decides, as before)."""
    a = set(str(key_a).split()) & _DESIGNATION_TOKENS
    b = set(str(key_b).split()) & _DESIGNATION_TOKENS
    return not a or not b or a == b


def normalize_case_name(name: str) -> str:
    """Party-name-only, variant-unified form of a case name/citation - see the
    module docstring. Returns "" for empty input."""
    if not name:
        return ""
    t = _clean(name)
    t = re.sub(r"\(\s*(?:p|pvt|p\.)\s*\)\s*\.?", " Pvt ", t, flags=re.IGNORECASE)
    # drop (...) and [...] groups (repeat for nesting)
    for _ in range(3):
        t = re.sub(r"\([^()]*\)|\[[^\[\]]*\]", " ", t)
    t = re.sub(r"[()\[\]]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    t = _cut_citation_tail(t)
    parties = _VERSUS_RE.split(t)
    parts = [_party(p, collapse_revenue=len(parties) >= 2) for p in parties]
    return " v ".join(p for p in parts if p)
