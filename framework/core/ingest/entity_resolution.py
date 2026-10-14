"""
Entity resolution: deterministic normalization + fuzzy matching - GENERALIZED
from tax_kg_poc/scripts/entity_resolution.py. The only domain-specific piece
of that module was the RESOLUTION_KEY_PROPERTIES dict (which properties
characterize "same real-world entity" per ontology class, e.g. Section's
section_no) - that now comes in as a parameter (DomainConfig.
resolution_key_properties) instead of a hardcoded module constant. Every
other line of logic (thresholds, normalize(), fuzzy scoring, batch dedup) is
unchanged.

Used by:
  - framework/core/ingest/graph_loader.py -> cross-run duplicate check against
    nodes already in the graph from a PREVIOUS ingestion run.
  - a domain pack's own extraction pipeline (e.g. domain_packs/tax/converters/
    extract_entities.py) -> resolve_within_batch() for intra-run dedup.
"""
import re

try:
    from rapidfuzz import fuzz
except ImportError as e:
    raise ImportError(
        "framework/core/ingest/entity_resolution.py needs the 'rapidfuzz' package, but it isn't "
        "importable in THIS Python environment. This almost always means rapidfuzz is installed "
        "somewhere OTHER than the interpreter actually running this script.\n\n"
        "To fix:\n"
        "  1. Confirm which interpreter is running this script:\n"
        "       python -c \"import sys; print(sys.executable)\"\n"
        "  2. Install into THAT SAME interpreter (activate your venv first):\n"
        "       python -m pip install -r requirements.txt\n"
        "  3. Re-run using that same 'python'.\n"
    ) from e

HIGH_SIMILARITY = 95    # >= this: confident auto-merge
MEDIUM_SIMILARITY = 85  # >= this (but < HIGH): flag as a possible duplicate, don't merge


def normalize(text: str) -> str:
    """Lowercase, strip punctuation/whitespace noise, collapse common
    abbreviations (Sec./Section, No./Number) so trivially-different strings
    compare equal. Kept generic - "Sec./Section" and "No./Number" are common
    enough abbreviation patterns across many document domains, not uniquely
    tax-specific, so this stays in the framework rather than moving to
    DomainConfig."""
    if not text:
        return ""
    t = str(text).strip().lower()
    t = re.sub(r"\bsec\.?\b", "section", t)
    t = re.sub(r"\bno\.?\b", "number", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


class KeyProperties(list):
    """2026-10-14 (phase0/followups): NEW - a resolution_key_properties entry
    that also carries a domain-specific `normalizer(str) -> str`, applied to
    the joined property values BEFORE the generic normalize() - e.g. the tax
    pack's case-name normalizer, which strips reporter citations/court
    suffixes and unifies "Pvt Ltd"/"Private Limited", "CIT"/"Commissioner of
    Income Tax", so a cited-case stub and the ingested judgment produce the
    same key. A list subclass on purpose: every existing consumer that treats
    the entry as a plain list of property names keeps working unchanged, and
    no resolution_key() call site needs a new argument. Affects only the
    fuzzy-comparison KEY, never a node id. A normalizer that raises or
    returns "" falls back to the raw text."""

    def __init__(self, props, normalizer=None, compatible=None):
        super().__init__(props)
        self.normalizer = normalizer
        # Optional `compatible(key_a, key_b) -> bool` veto, checked BEFORE the
        # fuzzy score: fuzzy ratios on long keys can clear HIGH_SIMILARITY
        # even when one short, legally decisive token differs (e.g. the
        # respondent "ito" vs "cit" in otherwise identical case names). False
        # means "never the same entity", whatever the score.
        self.compatible = compatible


_NUMBER_TOKEN_RE = re.compile(r"\d+")


def same_number_tokens(key_a: str, key_b: str) -> bool:
    """2026-10-14 (phase0/review-integrity): NEW - True iff both keys contain
    the same multiset of digit runs. In this corpus a differing NUMBER is the
    one token that distinguishes two different authorities whose citations are
    otherwise identical text: CBDT Circular 8 vs 3 of 2026, Article 12 vs 13
    of a DTAA, Rule 10 vs 11, Act No. 15 vs 16 of 2024, (2021) 432 ITR 471 vs
    417, opinion OP-01 vs OP-02. token_sort_ratio scores every such pair
    95-99 (one short token differs among many shared ones), i.e. an
    AUTO-MERGE onto the wrong authority (review/phase0 tests). Multiset, not
    sequence, to match token_sort_ratio's order-insensitivity. Cost of a
    false veto (the same authority written "1,00,000" vs "100000"): a missed
    merge, two nodes - the "leave unset rather than guess" side."""
    return sorted(_NUMBER_TOKEN_RE.findall(key_a or "")) == sorted(_NUMBER_TOKEN_RE.findall(key_b or ""))


def key_guard(entity_type: str, resolution_key_properties: dict | None):
    """The compatibility veto applied before fuzzy scoring for this type.

    2026-10-14: was just the type's KeyProperties.compatible (or None).
    2026-10-14 (phase0/review-integrity): now ALWAYS returns a veto - the
    numeric-token check (same_number_tokens) for every type, AND-ed with the
    type's own KeyProperties.compatible when it has one. Applied wherever
    key_guard() is (graph_loader cross-run matching, resolve_within_batch,
    merge_duplicate_caselaw clustering), so no fuzzy path can join two
    citations that differ only by a number."""
    compatible = getattr((resolution_key_properties or {}).get(entity_type), "compatible", None)
    if compatible is None:
        return same_number_tokens
    return lambda a, b: same_number_tokens(a, b) and compatible(a, b)


def has_resolution_key(entity_type: str, properties: dict,
                       resolution_key_properties: dict | None = None) -> bool:
    """2026-10-14 (phase0/review-integrity): NEW - True iff resolution_key()
    would build this entity's key from its configured properties, False when
    it falls back to the entity's own id text. graph_loader uses this to
    refuse to AUTO-MERGE on an id-fallback key (see find_cross_graph_match)."""
    props = (resolution_key_properties or {}).get(entity_type)
    return bool(props) and any(properties.get(p) for p in props)


def resolution_key(entity_type: str, properties: dict, fallback_id: str,
                    resolution_key_properties: dict | None = None) -> str:
    """Builds the normalized string used for fuzzy comparison. `resolution_key_properties`
    is DomainConfig.resolution_key_properties (entity_type -> [prop names]);
    falls back to the entity's own id if the type isn't listed."""
    props = (resolution_key_properties or {}).get(entity_type)
    if props:
        parts = [str(properties.get(p, "")) for p in props if properties.get(p)]
        if parts:
            text = " ".join(parts)
            normalizer = getattr(props, "normalizer", None)
            if normalizer is not None:
                try:
                    text = normalizer(text) or text
                except Exception:
                    pass
            return normalize(f"{entity_type} " + text)
    return normalize(f"{entity_type} {fallback_id}")


def similarity(key_a: str, key_b: str) -> float:
    if not key_a or not key_b:
        return 0.0
    return fuzz.token_sort_ratio(key_a, key_b)


def best_match(key: str, candidates: list[tuple[str, str]], compatible=None) -> tuple[str | None, float]:
    best_id, best_score = None, 0.0
    for cand_id, cand_key in candidates:
        # 2026-10-14: optional per-type veto - see KeyProperties.compatible.
        if compatible is not None and not compatible(key, cand_key):
            continue
        score = similarity(key, cand_key)
        if score > best_score:
            best_id, best_score = cand_id, score
    return best_id, best_score


def resolve_within_batch(entities: list[dict],
                          resolution_key_properties: dict | None = None
                          ) -> tuple[list[dict], dict[str, str], list[dict]]:
    """Deterministic + fuzzy dedup WITHIN one extraction run's entity list.
    Entities are only ever compared against others of the SAME type. Returns
    (resolved_entities, alias_map, review_flags) - see the original
    tax_kg_poc/scripts/entity_resolution.py for the full behavioral
    docstring (unchanged here)."""
    by_type: dict[str, list[dict]] = {}
    for ent in entities:
        by_type.setdefault(ent["type"], []).append(ent)

    kept: dict[str, dict] = {}
    alias_map: dict[str, str] = {}
    review_flags: list[dict] = []

    for etype, group in by_type.items():
        resolved: list[tuple[str, dict]] = []

        for ent in group:
            key = resolution_key(etype, ent.get("properties", {}), ent["id"], resolution_key_properties)
            best_id_key = [(e["id"], k) for k, e in resolved]
            best_id, best_score = best_match(key, best_id_key,
                                             compatible=key_guard(etype, resolution_key_properties))
            best_entity = next((e for k, e in resolved if e["id"] == best_id), None) if best_id else None

            if best_entity is not None and best_score >= HIGH_SIMILARITY and ent["id"] != best_entity["id"]:
                for k, v in ent.get("properties", {}).items():
                    if v and not best_entity["properties"].get(k):
                        best_entity["properties"][k] = v
                existing_aliases = best_entity["properties"].get("also_known_as_ids", "")
                alias_set = set(existing_aliases.split(",")) if existing_aliases else set()
                alias_set.add(ent["id"])
                best_entity["properties"]["also_known_as_ids"] = ",".join(sorted(alias_set))
                alias_map[ent["id"]] = best_entity["id"]

            elif best_entity is not None and MEDIUM_SIMILARITY <= best_score < HIGH_SIMILARITY:
                ent.setdefault("properties", {})
                ent["properties"]["possibleDuplicateOf"] = best_entity["id"]
                review_flags.append({
                    "kept_id": best_entity["id"], "candidate_id": ent["id"], "score": round(best_score, 1),
                })
                resolved.append((key, ent))
                kept[ent["id"]] = ent

            else:
                resolved.append((key, ent))
                kept[ent["id"]] = ent

    return list(kept.values()), alias_map, review_flags
