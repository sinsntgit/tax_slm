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


def resolution_key(entity_type: str, properties: dict, fallback_id: str,
                    resolution_key_properties: dict | None = None) -> str:
    """Builds the normalized string used for fuzzy comparison. `resolution_key_properties`
    is DomainConfig.resolution_key_properties (entity_type -> [prop names]);
    falls back to the entity's own id if the type isn't listed."""
    props = (resolution_key_properties or {}).get(entity_type)
    if props:
        parts = [str(properties.get(p, "")) for p in props if properties.get(p)]
        if parts:
            return normalize(f"{entity_type} " + " ".join(parts))
    return normalize(f"{entity_type} {fallback_id}")


def similarity(key_a: str, key_b: str) -> float:
    if not key_a or not key_b:
        return 0.0
    return fuzz.token_sort_ratio(key_a, key_b)


def best_match(key: str, candidates: list[tuple[str, str]]) -> tuple[str | None, float]:
    best_id, best_score = None, 0.0
    for cand_id, cand_key in candidates:
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
            best_id, best_score = best_match(key, best_id_key)
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
