"""
Per-item scoring of one saved /api/ask response against one gold item.

2026-10-14: NEW. Every metric here is DETERMINISTIC (regex/string logic over
the saved response) - no LLM call - so a run can be re-scored offline any
number of times (`python -m eval.score <run_dir>`) when a metric is fixed or
a gold item is corrected, without paying for the answers again. The only
model-judged numbers (judge dimensions) are read from what the API itself
returned (grounding.answer_quality, and /api/ask/evaluate when --judge).

Response shape relied on (framework/webapp/app.py::_api_ask_impl):
  answer                     - final text, incl. the server-appended
                               "DISCLAIMER" block and, when citation_ok was
                               False, a "(Note: ...)" caution line
  source_chunks[]            - {chunk_id, via, preview(<=200 chars),
                               rerank_score, document{document_name,
                               chunk_citation, entity_citation, page, ...},
                               parent_context{...}}
  chunks_used[]              - full chunk text; only when the request sent
                               include_full_chunks=true (the runner always does)
  grounding.citation_check   - grounding.verify_citations() result
  grounding.referred_sources_check, grounding.answer_quality
  intent, answer_mode, retrieval_mode, from_cache
"""
from __future__ import annotations

import re
import unicodedata

from framework.core.retrieve import grounding

# --------------------------------------------------------------------------
# Text normalisation
# --------------------------------------------------------------------------

_DISCLAIMER_RE = re.compile(r"\n\s*DISCLAIMER\s*\n", re.MULTILINE)
_CAUTION_NOTE = "(Note: this answer should have cited specific sources but didn't"


def strip_server_additions(answer: str) -> str:
    """Removes what the SERVER appends after generation (fixed disclaimer,
    citation-caution note) so metrics judge only what the model wrote - e.g.
    a must_not_claim pattern must never fire on the boilerplate disclaimer."""
    text = answer or ""
    matches = list(_DISCLAIMER_RE.finditer(text))
    if matches:
        text = text[:matches[-1].start()]
    idx = text.find(_CAUTION_NOTE)
    if idx != -1:
        text = text[:idx]
    return text.rstrip()


def normalize(text: str) -> str:
    """Lower-cases and canonicalises punctuation that legal citations vary
    on: curly quotes, en/em dashes, the section sign, underscores (document
    names are often file stems like 'Engineering_Analysis_Centre_...'),
    and a hyphen between a section number and its letter suffix
    ('80-IAC' == '80IAC', '194-IC' == '194IC')."""
    t = unicodedata.normalize("NFKC", text or "")
    t = (t.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
          .replace("–", "-").replace("—", "-").replace("§", " section ").replace("_", " "))
    t = t.lower()
    t = re.sub(r"(\d)\s*-\s*([a-z])", r"\1\2", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


# --------------------------------------------------------------------------
# Authority matching (section-number aware)
# --------------------------------------------------------------------------

# Left/right "word" boundaries for an alias: the neighbouring character must
# not be a letter/digit. This is what makes matching section-number aware:
# "Section 195" does NOT match "Section 195A" or "Section 1951", and
# "9(1)(vi)" does NOT match "9(1)(vii)" (the alias ends in ')' and the text
# has 'i' there) - while "Section 195" still matches "Section 195(1)",
# since citing a sub-provision is citing the section.
_LB = r"(?<![a-z0-9])"
_RB = r"(?![a-z0-9])"
_SECTION_PREFIX = r"(?:sections?|secs?\.?|s\.|u/s\.?|section)\s*"


def _alias_regex(alias: str) -> str:
    norm = normalize(alias)
    body = re.escape(norm).replace(r"\ ", r"\s*")
    # "Pvt. Ltd." / "Pvt Ltd" / "v." / "v" - a literal '.' in the alias is optional in the text.
    body = body.replace(r"\.", r"\.?")
    return _LB + body + _RB


def compile_authority(auth: dict) -> list[re.Pattern]:
    patterns = [_alias_regex(a) for a in auth.get("aliases", []) if a.strip()]
    section = auth.get("section")
    if section:
        sec = re.escape(normalize(section)).replace(r"\ ", r"\s*")
        patterns.append(_LB + _SECTION_PREFIX + sec + _RB)
    return [re.compile(p) for p in patterns]


def find_authority(auth: dict, normalized_text: str) -> str | None:
    """Returns the matched snippet (normalised) or None."""
    for rx in compile_authority(auth):
        m = rx.search(normalized_text)
        if m:
            return m.group(0)
    return None


# --------------------------------------------------------------------------
# What the system actually retrieved (the "cited source documents")
# --------------------------------------------------------------------------

def _strings_in(obj, out: list[str], depth: int = 0):
    if depth > 4:
        return
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            _strings_in(v, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _strings_in(v, out, depth + 1)


def _strip_ext(name: str) -> str:
    return re.sub(r"\.(txt|pdf|md|docx?|json)$", "", name.strip(), flags=re.IGNORECASE)


def known_sources(response: dict) -> dict:
    """Everything retrieval handed the generator, from the response:
    chunk ids, document names (normalised, extension-stripped), known pages
    per document, a doc_details map in grounding.py's shape, and one big
    normalised text blob (names + citations + previews + full chunk text +
    parent-section text) used for 'is this authority in the retrieved
    sources' and the case-mention heuristic."""
    chunk_ids: set[str] = set()
    doc_names: dict[str, str] = {}          # normalised -> original
    pages: dict[str, set[int]] = {}
    doc_details: dict[str, dict] = {}
    texts: list[str] = []

    def add_doc(chunk_id, name, page):
        if chunk_id is not None:
            chunk_ids.add(str(chunk_id))
        if name:
            key = normalize(_strip_ext(str(name)))
            doc_names.setdefault(key, str(name))
            if page not in (None, ""):
                try:
                    pages.setdefault(key, set()).add(int(page))
                except (TypeError, ValueError):
                    pass

    for ref in response.get("source_chunks") or []:
        if not isinstance(ref, dict):
            continue
        doc = ref.get("document") or {}
        add_doc(ref.get("chunk_id"), doc.get("document_name"), doc.get("page"))
        if doc:
            doc_details[str(ref.get("chunk_id"))] = doc
        _strings_in(ref.get("preview"), texts)
        _strings_in(doc, texts)
        _strings_in(ref.get("parent_context"), texts)
    for c in response.get("chunks_used") or []:
        if not isinstance(c, dict):
            continue
        add_doc(c.get("chunk_id"), c.get("document_name"), c.get("page"))
        if c.get("document_name") and str(c.get("chunk_id")) not in doc_details:
            doc_details[str(c.get("chunk_id"))] = {"document_name": c.get("document_name"), "page": c.get("page")}
        _strings_in(c.get("text"), texts)
        _strings_in(c.get("document_name"), texts)
    # Graph nodes returned by NL2Cypher are evidence too (case names/citations live there).
    for n in response.get("nodes") or []:
        if isinstance(n, dict):
            _strings_in(n.get("properties") or n.get("label") or n.get("caption"), texts)
    return {
        "chunk_ids": chunk_ids,
        "doc_names": doc_names,
        "pages": pages,
        "doc_details": doc_details,
        "text": normalize("\n".join(texts)),
    }


# --------------------------------------------------------------------------
# Citation extraction + fabricated-citation check
# --------------------------------------------------------------------------

# v6 prompt's mandated inline style: "<document name> - Page <N>" where the
# document name is the source-file label (ends in .txt/.pdf/.md). The lazy
# prefix starts at the beginning of the line/after the previous citation, so
# the captured string can carry leading sentence text - matching below is
# containment-based, which tolerates that.
_FILE_CITATION_RE = re.compile(
    r"([^\n\[\]]{3,220}?\.(?:txt|pdf|md|docx?))(?![a-z0-9])(?:\s*[-–—,]\s*(?:page|p\.)\s*(\d+))?",
    re.IGNORECASE)
_BRACKET_CITATION_RE = re.compile(r"\[\s*document\s*:\s*([^\],\]]+)(?:,\s*p(?:age|\.)\s*(\d+))?", re.IGNORECASE)
_CHUNK_ID_RE = re.compile(r"chunk_id\s*=\s*([A-Za-z0-9_.:-]+)", re.IGNORECASE)
_MIN_NAME_LEN = 8


def _display(candidate: str) -> str:
    c = candidate.strip()
    # Drop leading sentence text: keep what follows the last ". " / bullet before the file name.
    # A sentence boundary is punctuation after a real word (3+ lower-case
    # letters) or a number - NOT after "v." / "Ltd." / "Pvt." inside a case name.
    parts = re.split(r"(?:(?:(?<=[a-z]{3}[.;:!?])|(?<=[\d)][.;]))\s+(?=[A-Z0-9\[(]))|^\s*[*\-•]+\s*"
                     r"|\*\*[^*]{1,40}\*\*\s*", c)
    tail = parts[-1].strip() if parts and parts[-1].strip() else c
    tail = tail.strip("*_` ")
    return tail[-160:]


def _match_doc(candidate: str, doc_names: dict[str, str]) -> str | None:
    cand = normalize(_strip_ext(candidate.strip().strip("*_`")))
    for key in doc_names:
        if len(key) >= _MIN_NAME_LEN and key in cand:
            return key
        if len(cand) >= _MIN_NAME_LEN and cand in key:
            return key
    return None


def check_citations(body: str, sources: dict) -> dict:
    """Finds every citation the answer makes and checks it against what was
    actually retrieved. A citation is FABRICATED when it names a document
    (or chunk_id) that is not among the retrieved sources. A citation to a
    real document with a page number that none of that document's retrieved
    chunks carry is reported separately as page_unverified (the prompt
    forbids inventing pages, but a parent-section page can legitimately
    differ - so it is a review signal, not counted as fabricated)."""
    citations = []
    for m in _FILE_CITATION_RE.finditer(body):
        citations.append({"kind": "document", "raw": m.group(1), "page": m.group(2)})
    for m in _BRACKET_CITATION_RE.finditer(body):
        citations.append({"kind": "bracket", "raw": m.group(1), "page": m.group(2)})
    for m in _CHUNK_ID_RE.finditer(body):
        citations.append({"kind": "chunk_id", "raw": m.group(1).rstrip(".:-"), "page": None})

    fabricated, page_unverified, checked = [], [], []
    for c in citations:
        entry = {"kind": c["kind"], "citation": _display(c["raw"]) if c["kind"] != "chunk_id" else f"chunk_id={c['raw']}",
                 "page": int(c["page"]) if c["page"] else None}
        if c["kind"] == "chunk_id":
            entry["verified"] = c["raw"] in sources["chunk_ids"]
        else:
            key = _match_doc(c["raw"], sources["doc_names"])
            entry["verified"] = key is not None
            if key is not None and entry["page"] is not None:
                known = sources["pages"].get(key) or set()
                entry["page_verified"] = (entry["page"] in known) if known else None
                if known and entry["page"] not in known:
                    page_unverified.append(f"{entry['citation']} (page {entry['page']}; retrieved pages {sorted(known)})")
        if not entry["verified"]:
            fabricated.append(entry["citation"])
        checked.append(entry)

    # Dedupe while preserving order - the same fake source cited 3 times is one fabrication.
    # Containment, not equality: the same source can be captured with more or
    # less leading sentence text depending on where it is cited.
    seen, uniq = [], []
    for f in sorted(fabricated, key=len):
        k = normalize(_strip_ext(f))
        if not any(s in k or k in s for s in seen):
            seen.append(k)
            uniq.append(f)
    uniq.sort(key=fabricated.index)
    return {"citations": checked, "citation_count": len(checked), "fabricated": uniq,
            "fabricated_count": len(uniq), "page_unverified": page_unverified}


# Case-name heuristic: "<Party> v. <Party>" / "vs" mentions in the body whose
# lead party name does not appear anywhere in the retrieved material. This
# catches an invented authority cited WITHOUT a file-style citation (which
# check_citations cannot see). Heuristic + informational only - never gated.
_CASE_RE = re.compile(
    r"\b([A-Z][A-Za-z&.'-]+(?:\s+(?:of|&|and|[A-Z(][A-Za-z&.'()-]*)){0,6})\s+v(?:s)?\.?\s+"
    r"((?:the\s+)?[A-Z][A-Za-z&.'-]+(?:\s+[A-Z(][A-Za-z&.'()-]*){0,5})")
_GENERIC_PARTY_WORDS = {
    "cit", "commissioner", "income", "tax", "union", "india", "uoi", "state", "acit", "dcit", "ito", "dit",
    "the", "of", "and", "&", "m/s", "m/s.", "pvt", "ltd", "private", "limited", "p", "(p)", "co", "co.",
    "inc", "corporation", "company", "deputy", "assistant", "additional", "principal", "pr", "exemptions",
    "(exemptions)", "authority", "director", "in", "re", "revenue", "assessee", "chief", "officer",
}


def unsupported_case_mentions(body: str, sources: dict) -> list[str]:
    out = []
    for m in _CASE_RE.finditer(body):
        parties = (m.group(1) + " " + m.group(2))
        words = [w.strip(".,()'").lower() for w in parties.split()]
        distinctive = [w for w in words if len(w) > 3 and w not in _GENERIC_PARTY_WORDS]
        if not distinctive:
            continue
        if not any(re.search(_LB + re.escape(w) + _RB, sources["text"]) for w in distinctive[:3]):
            mention = m.group(0).strip()
            if mention not in out:
                out.append(mention)
    return out


# --------------------------------------------------------------------------
# a/b/c/d format compliance (answer_generation v6, research mode)
# --------------------------------------------------------------------------

_HEADINGS = [
    ("a", r"executive\s+summary"),
    ("b", r"detailed\s+analysis"),
    ("c", r"conclusion"),
    ("d", r"referred\s+sources"),
]
_REFERRED_CATEGORIES = {
    "acts_rules": r"acts\s*(?:&|and)\s*rules",
    "dtaa": r"dtaa",
    "case_law": r"case\s+law",
    "circulars": r"circulars?",
    "fema_rbi": r"fema",
    "web_research": r"web\s+research",
}


def _heading_regex(letter: str, title: str) -> re.Pattern:
    # Line-start, tolerant of markdown decoration: "## a. Executive Summary",
    # "**a. Executive Summary - Core...**", "a) Executive summary".
    return re.compile(rf"^[ \t>#*_]*{letter}[.)][ \t*_]*{title}", re.IGNORECASE | re.MULTILINE)


def check_format(body: str, answer_mode: str) -> dict:
    """v6's Section 6 mandates the exact a./b./c./d. skeleton for research
    answers ('do not rename, reorder, or omit'). `compliant` = all four
    headings present at line start, in order. The litigation-risk bullet and
    the six Referred Sources category labels are reported as `strict_*`
    detail (v6 Section 7 allows dropping the litigation-risk bullet for a
    question with no litigation dimension, so it is not part of the gate).
    lookup/opinion modes use different prompts - not applicable there."""
    if answer_mode != "research":
        return {"applicable": False, "compliant": None}
    positions = {}
    for letter, title in _HEADINGS:
        m = _heading_regex(letter, title).search(body)
        positions[letter] = m.start() if m else None
    present = [k for k, v in positions.items() if v is not None]
    missing = [k for k, v in positions.items() if v is None]
    ordered = all(positions[a] < positions[b] for a, b in zip("abc", "bcd")
                  if positions[a] is not None and positions[b] is not None)
    compliant = not missing and ordered

    litigation = None
    if positions["a"] is not None:
        end = positions["b"] if positions["b"] is not None else len(body)
        litigation = bool(re.search(r"litigation\s+risk", body[positions["a"]:end], re.IGNORECASE))
    categories_missing = list(_REFERRED_CATEGORIES)
    if positions["d"] is not None:
        d_section = body[positions["d"]:]
        categories_missing = [k for k, rx in _REFERRED_CATEGORIES.items()
                              if not re.search(rx, d_section, re.IGNORECASE)]
    return {
        "applicable": True,
        "compliant": compliant,
        "headings_present": present,
        "headings_missing": missing,
        "in_order": ordered,
        "strict_litigation_risk_bullet": litigation,
        "strict_referred_categories_missing": categories_missing,
    }


# --------------------------------------------------------------------------
# must_not_claim + expected_key_points
# --------------------------------------------------------------------------

_NEGATIONS = {"not", "no", "never", "neither", "nor", "without", "cannot", "none", "unless", "rejected",
              "rejecting", "erroneous", "incorrect", "wrongly"}


def _is_negated(body: str, start: int) -> bool:
    # 2026-10-14: only the words BEFORE the match are checked, never the match
    # itself - a pattern may legitimately contain a negation as part of the
    # wrong claim (e.g. "not includible in book profit", "GAAR will not apply").
    sent_start = max(body.rfind(ch, 0, start) for ch in ".!?\n") + 1
    before = re.findall(r"[\w']+", body[sent_start:start])[-6:]
    for w in before:
        lw = w.lower()
        if lw in _NEGATIONS or lw.endswith("n't"):
            return True
    return False


def check_must_not_claim(body: str, claims: list[dict]) -> list[dict]:
    """Pattern screen for claims a correct answer must NOT make. With
    negation_guard (default on), a match whose 6 preceding words in the same
    sentence contain a negation ('not', "isn't", 'without', ...) is
    ignored - so "does NOT constitute royalty" never fires a "constitutes?
    royalty" pattern. Still a heuristic: an attributed claim
    ("the Revenue argued it is royalty") can fire - every violation carries
    its snippet so a reviewer can dismiss it."""
    violations = []
    for claim in claims or []:
        guard = claim.get("negation_guard", True)
        for pat in claim["patterns"]:
            for m in re.finditer(pat, body, re.IGNORECASE):
                if guard and _is_negated(body, m.start()):
                    continue
                lo, hi = max(0, m.start() - 80), min(len(body), m.end() + 80)
                violations.append({"claim": claim["claim"], "pattern": pat,
                                   "snippet": body[lo:hi].replace("\n", " ")})
                break
            else:
                continue
            break  # one violation per claim is enough
    return violations


def check_key_points(body: str, points: list[dict]) -> dict:
    scored = [p for p in points or [] if p.get("match_any")]
    hits = []
    for p in scored:
        hit = any(re.search(rx, body, re.IGNORECASE) for rx in p["match_any"])
        hits.append({"point": p["point"], "matched": hit})
    return {
        "scored": len(scored),
        "matched": sum(1 for h in hits if h["matched"]),
        "coverage": (sum(1 for h in hits if h["matched"]) / len(scored)) if scored else None,
        "missed": [h["point"] for h in hits if not h["matched"]],
    }


# --------------------------------------------------------------------------
# Judge scores (read from the API, never computed here)
# --------------------------------------------------------------------------

INTERNAL_JUDGE_DIMS = ("overall", "groundedness", "completeness", "specificity",
                       "statutory_fidelity", "holding_discipline", "authority_hierarchy")


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")


def extract_judge_scores(response: dict, judge_record: dict | None) -> dict:
    """Flat {"<source>.<dimension>": score} map. Sources:
      answer_quality.*  - grounding.answer_quality (answer_quality.judge_answer,
                          0-1, runs inside every /api/ask call)
      evaluation.*      - /api/ask/evaluate's grounded scorecard (0-10), --judge
      general_knowledge.overall - /api/ask/evaluate's no-context scorecard (0-10)
      auto_evaluation.* - only if the request asked for auto_evaluate
    Unavailable judges contribute nothing (never a 0) - a failed judge call
    is not a bad answer (same rule as answer_quality.judge_answer)."""
    scores: dict[str, float] = {}
    aq = ((response or {}).get("grounding") or {}).get("answer_quality") or {}
    if aq.get("available"):
        for d in INTERNAL_JUDGE_DIMS:
            if isinstance(aq.get(d), (int, float)):
                scores[f"answer_quality.{d}"] = float(aq[d])

    def scorecard(prefix: str, ev: dict | None):
        if not isinstance(ev, dict) or not ev.get("available"):
            return
        if isinstance(ev.get("overall"), (int, float)):
            scores[f"{prefix}.overall"] = float(ev["overall"])
        for dim in ev.get("dimensions") or []:
            if isinstance(dim, dict) and isinstance(dim.get("score"), (int, float)):
                scores[f"{prefix}.{_slug(dim.get('name'))}"] = float(dim["score"])

    scorecard("auto_evaluation", (response or {}).get("auto_evaluation"))
    jr = (judge_record or {}).get("response") if (judge_record or {}).get("status") == "ok" else None
    if isinstance(jr, dict):
        scorecard("evaluation", jr.get("evaluation"))
        gk = jr.get("general_knowledge_evaluation")
        if isinstance(gk, dict) and gk.get("available") and isinstance(gk.get("overall"), (int, float)):
            scores["general_knowledge.overall"] = float(gk["overall"])
    return scores


# --------------------------------------------------------------------------
# One item
# --------------------------------------------------------------------------

def score_item(gold: dict, record: dict, *, authority_item_threshold: float = 0.85) -> dict:
    """Scores one saved record (see eval/run.py for its shape). An errored
    record yields status='error' and no answer metrics - errors are counted
    in error_rate, never silently scored as zeros."""
    base = {
        "id": gold["id"],
        "category": gold["category"],
        "answer_mode": record.get("effective_answer_mode") or gold["answer_mode"],
        "retrieval_mode": record.get("effective_retrieval_mode"),
        "verified": gold["verified"],
        "status": record.get("status", "missing"),
        "latency_s": record.get("latency_s"),
        "attempts": record.get("attempts"),
    }
    if base["status"] != "ok":
        base["error"] = record.get("error")
        return base

    response = record.get("response") or {}
    raw_answer = response.get("answer") or ""
    body = strip_server_additions(raw_answer)
    sources = known_sources(response)
    norm_body = normalize(body)

    # Authorities
    def auth_rows(auths):
        rows = []
        for a in auths or []:
            in_answer = find_authority(a, norm_body)
            in_sources = find_authority(a, sources["text"])
            rows.append({"id": a["id"], "label": a["label"], "in_answer": in_answer is not None,
                         "in_sources": in_sources is not None, "answer_match": in_answer})
        return rows

    req = auth_rows(gold.get("required_authorities"))
    opt = auth_rows(gold.get("optional_authorities"))
    n_req = len(req)
    recall_answer = (sum(r["in_answer"] for r in req) / n_req) if n_req else None
    recall_sources = (sum(r["in_sources"] for r in req) / n_req) if n_req else None
    supported = (sum(r["in_answer"] and r["in_sources"] for r in req) / n_req) if n_req else None

    cit = check_citations(body, sources)
    referred = grounding.check_referred_sources(body, sources["doc_details"] or None)
    server_citation_check = ((response.get("grounding") or {}).get("citation_check")) or {}
    if "citation_ok" in server_citation_check:
        citation_ok, citation_ok_source = bool(server_citation_check["citation_ok"]), "server"
    else:
        local = grounding.verify_citations(body, response.get("chunks_used") or response.get("source_chunks") or [],
                                           [], doc_details=sources["doc_details"] or None)
        citation_ok, citation_ok_source = bool(local["citation_ok"]), "recomputed"

    base.update({
        "intent": response.get("intent"),
        "from_cache": bool(response.get("from_cache")),
        "no_context": not (response.get("source_chunks") or response.get("chunks_used")),
        "answer_chars": len(body),
        "server_elapsed_s": response.get("fast_ask_elapsed_seconds"),
        "authorities": {
            "required": n_req,
            "cited_in_answer": sum(r["in_answer"] for r in req),
            "found_in_sources": sum(r["in_sources"] for r in req),
            "recall_answer": recall_answer,
            "recall_sources": recall_sources,
            "recall_answer_and_sources": supported,
            "meets_threshold": (recall_answer >= authority_item_threshold) if recall_answer is not None else None,
            "missing_in_answer": [r["label"] for r in req if not r["in_answer"]],
            "missing_in_sources": [r["label"] for r in req if not r["in_sources"]],
            "optional_cited": [r["label"] for r in opt if r["in_answer"]],
        },
        "citations": {
            "count": cit["citation_count"],
            "fabricated": cit["fabricated"],
            "fabricated_count": cit["fabricated_count"],
            "page_unverified": cit["page_unverified"],
            "referred_sources_unverified": referred.get("unverified_sources", []),
            "server_referred_sources_unverified":
                ((response.get("grounding") or {}).get("referred_sources_check") or {}).get("unverified_sources", []),
            "unsupported_case_mentions": unsupported_case_mentions(body, sources),
            "citation_ok": citation_ok,
            "citation_ok_source": citation_ok_source,
        },
        "format": check_format(body, base["answer_mode"]),
        "must_not_claim": check_must_not_claim(body, gold.get("must_not_claim")),
        "key_points": check_key_points(body, gold.get("expected_key_points")),
        "judge": extract_judge_scores(response, record.get("judge")),
    })
    return base
