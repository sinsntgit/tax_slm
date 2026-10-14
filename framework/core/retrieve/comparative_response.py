"""
COMPARATIVE / FOR-AGAINST RESPONSE STRUCTURE OVERRIDE - 2026-08-29: NEW, per
explicit request: "for the questions related to favourable/unfavorable or
for/against or comparison or hold/withhold, use the following structure for
response... Add any specific additional items in the structure that is
relevant... you may want to add this to the settings page."

Forces a dedicated, user-editable answer structure onto questions that ask
which side prevails (favourable/unfavourable, for/against, hold/withhold,
"which view", etc.), REGARDLESS of whatever the active answer_generation
prompt's own default structure is - unlike response_config.py's generic
structure directive (skipped entirely once a prompt "defines its own
structure" - see app.py's `prompt_defines_own_structure`), this one is
appended on top of ANY prompt whenever the question itself calls for a
comparative analysis, because a for/against breakdown is a property of the
QUESTION, not a property of which prompt version happens to be active.

Detection is a cheap, deterministic keyword/phrase match against the
question text (see `TRIGGER_KEYWORDS` in the config) rather than an extra
LLM classification call - no added latency/cost per question, consistent
with this project's other cheap-heuristic-with-a-known-limit approach (e.g.
graphrag.py's normalize_citation_query()). It will occasionally under- or
over-trigger on unusual phrasing; the trigger list itself is user-editable
via the Settings page for exactly that reason.

Hot-reloadable, same pattern as response_config.py: load_config() re-reads
the JSON file from disk on every /api/ask call.
"""
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger("comparative_response")

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "comparative_response_config.json"

# The user's 6 required sections, PLUS additional items judged relevant to a
# thorough favourable/unfavourable-type tax analysis (statutory grounding,
# administrative guidance, and a practical takeaway) - all fully editable
# from the Settings page, so trim/reorder freely if these extras aren't
# wanted. See this module's docstring for the explicit request this answers.
DEFAULT_SECTIONS = [
    "Short answer / current position",
    "Applicable statutory provisions relied upon (Section/Rule cited by each side)",
    "Cases supporting — historical Revenue line",
    "Cases against — taxpayer line",
    "CBDT Circulars/administrative guidance relevant to the issue (if any, and whether still in force)",
    "Supreme Court resolution case law(s) — for and against, mentioned separately",
    "Status of each earlier authority post the resolving judgment (affirmed / distinguished / overruled / no longer good law)",
    "Practical implication for the taxpayer",
    "One-line conclusion",
]

# Phrasing patterns that mark a question as wanting a favourable/unfavourable,
# for/against, comparative, or hold/withhold-type analysis. Lowercased
# substring match against the (also lowercased) question text.
DEFAULT_TRIGGER_KEYWORDS = [
    "favourable", "favorable", "unfavourable", "unfavorable",
    "for and against", "for or against", "arguments for and against",
    "pros and cons", "supporting and opposing", "in favour of", "in favor of",
    "hold or withhold", "withhold or hold", "compare", "comparison",
    "which view", "prevailing view", "settled position", "revenue view",
    "taxpayer view", "assessee view", "pro-revenue", "pro-taxpayer",
    "both sides", "either side", "hold good", "still good law",
]

DEFAULT_CONFIG = {
    "enabled": True,
    "sections": DEFAULT_SECTIONS,
    "trigger_keywords": DEFAULT_TRIGGER_KEYWORDS,
}


def _config_path() -> Path:
    override = os.environ.get("COMPARATIVE_RESPONSE_CONFIG_PATH", "").strip()
    return Path(override) if override else DEFAULT_CONFIG_PATH


def load_config() -> dict:
    """Re-reads the config file from disk every call (cheap - it's a small
    JSON file). Falls back to DEFAULT_CONFIG - whole file missing, invalid
    JSON, or not a dict - so a bad hand-edit can never break /api/ask, it
    just silently reverts to the default structure for that request. Same
    pattern as response_config.load_response_config()."""
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
    written (unknown keys dropped, missing/empty keys fall back to
    DEFAULT_CONFIG so a save with a blank textarea can't wipe the feature
    out entirely by accident)."""
    normalized = dict(DEFAULT_CONFIG)
    if isinstance(cfg.get("enabled"), bool):
        normalized["enabled"] = cfg["enabled"]
    if isinstance(cfg.get("sections"), list):
        cleaned = [str(s).strip() for s in cfg["sections"] if str(s).strip()]
        if cleaned:
            normalized["sections"] = cleaned
    if isinstance(cfg.get("trigger_keywords"), list):
        cleaned = [str(k).strip().lower() for k in cfg["trigger_keywords"] if str(k).strip()]
        if cleaned:
            normalized["trigger_keywords"] = cleaned
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
    return normalized


def is_comparative_question(question: str, cfg: dict | None = None) -> bool:
    """True if `question` looks like it's asking for a favourable/
    unfavourable, for/against, comparison, or hold/withhold-type analysis -
    see this module's docstring for why this is a cheap keyword match rather
    than an LLM call. `cfg` can be passed in to avoid a redundant disk read
    when the caller already loaded it for build_structure_directive()."""
    cfg = cfg if cfg is not None else load_config()
    if not cfg.get("enabled", True):
        return False
    q = (question or "").lower()
    return any(kw in q for kw in (cfg.get("trigger_keywords") or []) if kw)


def build_structure_directive(cfg: dict | None = None) -> str:
    """Renders cfg into a directive block to APPEND to whatever
    answer_generation system prompt is already in use (built-in or a
    Settings-page override) - explicitly states it takes precedence over
    any other structure instruction for THIS question, since a comparative
    question's structure need is independent of which prompt version is
    active (see module docstring)."""
    cfg = cfg if cfg is not None else load_config()
    lines = [
        "\n\nRESPONSE STRUCTURE OVERRIDE (user-configured): this question asks for a "
        "favourable/unfavourable, for/against, comparative, or hold/withhold-type analysis. "
        "REGARDLESS of any other response-structure instruction elsewhere in this prompt, "
        "organize THIS answer using EXACTLY the following sections, in this order, as clear "
        "numbered headings:",
    ]
    for i, s in enumerate(cfg.get("sections") or DEFAULT_SECTIONS, 1):
        lines.append(f"{i}. {s}")
    lines.append(
        "\nFor the 'Cases supporting'/'Cases against' sections, cite only the actual retrieved "
        "case(s) by name and citation - do not invent a case to fill a side; if one side "
        "genuinely has no supporting authority in the retrieved material, say so plainly rather "
        "than leaving the section blank or fabricating an entry. For the Supreme Court "
        "resolution section, name any Supreme Court ruling that actually settled a conflict "
        "between the two lines above and state which line it favoured - if no Supreme Court "
        "ruling on point was retrieved, say so explicitly rather than omitting the section. For "
        "the 'status of each earlier authority' section, state for each cited ruling above "
        "whether it is still good law, distinguished, or overruled following any later ruling - "
        "based only on what the retrieved material actually shows, never on general/training "
        "knowledge (see this prompt's own anti-hallucination rules - they still apply here)."
    )
    lines.append(
        "\nUse these EXACT section headings (numbered, as shown above) as the top-level headings "
        "of your answer, even if other instructions elsewhere in this prompt describe a different "
        "structure (e.g. a Fact Sheet / Law & Regulatory Framework / Case Law Hierarchy layout) - "
        "for THIS question, the numbered list above replaces that layout entirely, it does not "
        "sit alongside it."
    )
    return "\n".join(lines)


def _section_key_phrase(section_text: str, max_words: int = 3) -> str:
    """A short, matchable prefix of a configured section name - e.g.
    "Cases supporting — historical Revenue line" -> "cases supporting" -
    used by check_structure_compliance() to substring-match against the
    model's actual headings without requiring an exact reproduction of the
    full (often long, parenthetical-laden) configured section text."""
    s = (section_text or "").strip()
    for sep in (" (", " — ", " – ", " - ", " / ", ":"):
        idx = s.find(sep)
        if idx > 0:
            s = s[:idx]
            break
    words = s.split()
    return " ".join(words[:max_words]).lower().strip(" .,:;-")


def check_structure_compliance(answer_text: str, cfg: dict | None = None) -> dict:
    """2026-08-29: NEW, per explicit report that the model 'is not following
    the structure' - a best-effort, NON-BLOCKING check (same honesty-over-
    silent-failure pattern as grounding.check_referred_sources()) for
    whether the generated answer actually contains each configured
    section's heading, so a compliance failure is VISIBLE (in the Evidence
    coverage panel + server log) instead of only suspected. A short
    key-phrase substring match, not a strict format check - the model is
    free to phrase/number headings differently as long as the recognizable
    heading text appears somewhere in the answer.
    Returns {"checked", "sections_expected", "sections_found",
    "missing_sections", "compliant"}."""
    cfg = cfg if cfg is not None else load_config()
    sections = cfg.get("sections") or DEFAULT_SECTIONS
    answer_lower = (answer_text or "").lower()
    missing = []
    for s in sections:
        phrase = _section_key_phrase(s)
        if phrase and phrase not in answer_lower:
            missing.append(s)
    found = len(sections) - len(missing)
    result = {
        "checked": True,
        "sections_expected": len(sections),
        "sections_found": found,
        "missing_sections": missing,
        "compliant": len(missing) == 0,
    }
    if missing:
        logger.warning(
            f"[comparative_response] Answer did not follow the configured comparative structure - "
            f"{found}/{len(sections)} section(s) found, missing: {missing}"
        )
    return result
