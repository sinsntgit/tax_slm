"""
INTENT-BASED GRAPH-VS-RECALL ROUTING - 2026-09-03: NEW, per explicit request
"a few strategies where you can based on intent decide to go graph route or
RAG route", raised alongside a "plain RAG gives better answers" report.

This is a SEPARATE, additional axis layered on top of intent.py's existing
EXACT_LOOKUP/RELATIONAL/CONCEPTUAL classification (an LLM call deciding how
MUCH vector search to blend in - see graphrag.DEFAULT_INTENT_RETRIEVAL_
CONFIG) - that axis is a blend-RATIO knob, not a real either/or router, and
doesn't decide whether it's worth the extra Neo4j round-trip cost of
graph-traversal expansion (see graphrag.GraphRAGRetriever.
expand_via_related_entities()) for THIS specific question.

Cheap, deterministic (no extra LLM call) keyword match - same reasoning and
same hot-reloadable/Settings-editable pattern as comparative_response.py:
zero added latency/cost per question, at the cost of occasional under/over-
triggering on unusual phrasing, which the editable trigger list exists to
let you correct without a redeploy.

route(question) returns "lineage" or "recall":
  - "lineage": the question is asking about a relationship BETWEEN sources
    (overruled/superseded/still good law/amended/status of/which court) -
    graphrag.retrieve() runs the vector-seeded graph-expansion step for
    these, walking OVERRULES/SUPERSEDED_BY/AMENDED_BY/etc. edges one hop
    from both the NL2Cypher-matched entities AND the top vector hits, so an
    answer can surface a directly-connected authority (e.g. the Supreme
    Court ruling that overruled a vector-matched High Court case) even when
    its TEXT doesn't lexically/semantically resemble the question.
  - "recall": everything else - definitional/explanatory questions gain
    little from a 1-hop relationship walk, so the expansion step is skipped
    to avoid its extra Neo4j round-trip for no benefit; retrieval relies on
    NL2Cypher + broad vector recall as it already does.
"""
import json
import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "retrieval_router_config.json"

# Phrasing patterns that mark a question as asking about a relationship
# BETWEEN authorities (precedent lineage, supersession, amendment status)
# rather than asking what the law/position IS. Lowercased substring match
# against the (also lowercased) question text.
DEFAULT_LINEAGE_KEYWORDS = [
    "overrule", "overruled", "overrules", "overruling",
    "still good law", "good law", "no longer good law",
    "supersede", "superseded", "supersedes", "supersession",
    "withdraw", "withdrawn", "withdraws",
    "amend", "amended", "amendment",
    "distinguish", "distinguished",
    "status of", "current status", "as it stands",
    "which court", "later decision", "subsequent decision", "subsequent ruling",
    "followed by", "resolved by", "resolved the conflict",
    "affirmed", "reversed", "set aside",
    "post-judgment", "after the judgment", "after the ruling",
]

DEFAULT_CONFIG = {
    "enabled": True,
    "lineage_keywords": DEFAULT_LINEAGE_KEYWORDS,
    # Expansion tuning - how many of the top vector hits / NL2Cypher-matched
    # entities feed the graph-expansion step as seeds, and how many
    # one-hop neighbors it's allowed to pull in total per question. Kept
    # here (not hardcoded in graphrag.py) so both the trigger keywords AND
    # the expansion's cost/breadth are tunable from the same Settings card.
    "expansion_seed_top_k": 5,
    "expansion_max_neighbors": 15,
}


def _config_path() -> Path:
    override = os.environ.get("RETRIEVAL_ROUTER_CONFIG_PATH", "").strip()
    return Path(override) if override else DEFAULT_CONFIG_PATH


def load_config() -> dict:
    """Re-reads the config file from disk every call (cheap - it's a small
    JSON file). Falls back to DEFAULT_CONFIG - whole file missing, invalid
    JSON, or not a dict - so a bad hand-edit can never break /api/ask, it
    just silently reverts to the default for that request. Same pattern as
    response_config.load_response_config()."""
    path = _config_path()
    if not path.exists():
        return dict(DEFAULT_CONFIG)
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            return dict(DEFAULT_CONFIG)
    except (OSError, ValueError):
        return dict(DEFAULT_CONFIG)
    merged = dict(DEFAULT_CONFIG)
    merged.update({k: v for k, v in cfg.items() if k in DEFAULT_CONFIG})
    return merged


def save_config(cfg: dict) -> dict:
    """Validates + writes cfg to disk. Returns the normalized dict actually
    written (unknown keys dropped; empty/out-of-range values fall back to
    the default for that one key rather than rejecting the whole save, or -
    for lineage_keywords specifically - falling back to the full default
    list, so a save with a blank textarea can't wipe the feature out
    entirely by accident, same convention comparative_response.py uses)."""
    normalized = dict(DEFAULT_CONFIG)
    if isinstance(cfg.get("enabled"), bool):
        normalized["enabled"] = cfg["enabled"]
    if isinstance(cfg.get("lineage_keywords"), list):
        cleaned = [str(k).strip().lower() for k in cfg["lineage_keywords"] if str(k).strip()]
        if cleaned:
            normalized["lineage_keywords"] = cleaned
    for key, lo, hi in (("expansion_seed_top_k", 1, 25), ("expansion_max_neighbors", 1, 100)):
        val = cfg.get(key)
        if isinstance(val, (int, float)) and not isinstance(val, bool) and lo <= int(val) <= hi:
            normalized[key] = int(val)
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
    return normalized


def route(question: str, cfg: dict | None = None) -> str:
    """Returns "lineage" or "recall" - see module docstring. `cfg` can be
    passed in to avoid a redundant disk read when the caller already loaded
    it (graphrag.retrieve() does, to also read the expansion tuning keys)."""
    cfg = cfg if cfg is not None else load_config()
    if not cfg.get("enabled", True):
        return "recall"
    q = (question or "").lower()
    if any(kw in q for kw in (cfg.get("lineage_keywords") or []) if kw):
        return "lineage"
    return "recall"
