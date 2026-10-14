"""
User-editable OVERRIDE for the answer_generation SYSTEM PROMPT itself (the
full instruction template, not just the response-structure directive that
response_config.py controls). Same hot-reload pattern as response_config.py
and llm_client's refresh_model_config(): a small JSON file on disk, re-read
on every /api/ask call, so an edit made via the Settings page's prompt
editor takes effect on the very next question - no restart needed.

Domain-agnostic: any domain pack's webapp gets this for free. When no
override is saved (file missing, empty, or explicitly cleared), app.py falls
back to the domain pack's own PROMPT_PACK.get_prompt("answer_generation")
template - so this is purely additive, an empty/missing override file
behaves exactly as if this module didn't exist.
"""
import json
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("prompt_config")

# Repo root (this file lives at framework/core/retrieve/prompt_config.py -
# three parents up is the repo root, same depth as .env/.env.example).
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "prompt_config.json"


def _config_path() -> Path:
    override = os.environ.get("PROMPT_CONFIG_PATH", "").strip()
    return Path(override) if override else DEFAULT_CONFIG_PATH


def load_prompt_override() -> dict:
    """Re-reads the override file from disk every call (cheap - it's a
    small JSON file). Returns {"enabled": bool, "template": str}. Falls back
    to {"enabled": False, "template": ""} - file missing, invalid JSON, or
    not a dict - so a bad hand-edit can never break /api/ask, it just
    silently reverts to the domain pack's built-in prompt for that request."""
    path = _config_path()
    if not path.exists():
        return {"enabled": False, "template": ""}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        if not isinstance(cfg, dict):
            return {"enabled": False, "template": ""}
    except (OSError, ValueError):
        return {"enabled": False, "template": ""}
    return {
        "enabled": bool(cfg.get("enabled", False)),
        "template": str(cfg.get("template") or ""),
    }


def save_prompt_override(cfg: dict) -> dict:
    """Validates + writes cfg to disk. Returns the normalized dict actually
    written. `enabled=True` with a blank/whitespace-only template is
    normalized to `enabled=False` - an override that resolves to nothing
    should behave exactly like no override, not like an empty system
    prompt being sent to the LLM."""
    template = str(cfg.get("template") or "").strip()
    # 2026-10-13: NEW - a template that fails validate_template() (see its
    # docstring) is still SAVED as a draft, so the user never loses text they
    # typed, but is forced to enabled=False - the same "an override that
    # can't work should behave exactly like no override" rule already applied
    # to a blank template just above. `problems` is returned (not written to
    # disk) so the Settings page can tell the user WHY it wasn't enabled.
    problems = validate_template(template) if template else []
    normalized = {
        "enabled": bool(cfg.get("enabled", False)) and bool(template) and not problems,
        "template": template,
    }
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
    return {**normalized, "problems": problems}


def resolve_answer_system_prompt(default_template: str) -> tuple[str, bool]:
    """Returns (template_to_use, is_override). Callers (app.py) use this
    instead of reaching into load_prompt_override() directly, so the
    fallback-to-default logic lives in exactly one place."""
    override = load_prompt_override()
    if override["enabled"] and override["template"]:
        # 2026-10-13: BUGFIX - an override that fails validate_template() is
        # now IGNORED here (falls back to the domain pack's built-in prompt)
        # instead of being sent to the LLM as-is. Reported case: the saved
        # prompt_config.json held the full source of a Python prompts module
        # (three templates - LOOKUP/RESEARCH/OPINION_PROMPT - as Python string
        # assignments with unfilled {context}/{question}/{source_discipline}
        # placeholders), so the answer model was being handed a Python file as
        # its system prompt with no indication of which of the three templates
        # to follow, while the built-in answer_generation v6 (including its
        # 3A "graph rows are verified facts" fix) was silently bypassed. This
        # check runs on every call (not just at save time) because a file
        # saved BEFORE validation existed - like that one - never went
        # through save_prompt_override()'s check.
        problems = validate_template(override["template"])
        if problems:
            logger.warning(f"[prompt_config] Saved answer-prompt override is enabled but unusable "
                           f"({'; '.join(problems)}) - ignoring it and using the built-in prompt.")
            return default_template, False
        return override["template"], True
    return default_template, False


# 2026-10-13: NEW - see validate_template()'s docstring.
# A line that assigns a triple-quoted string to an UPPER_CASE name, e.g.
#   RESEARCH_PROMPT = """
# - the signature of a pasted Python prompts module rather than prompt text.
_PY_ASSIGNMENT_RE = re.compile(r"^\s*[A-Z][A-Z0-9_]*\s*=\s*(?:[rRfFbBuU]{0,2})(?:\"\"\"|\'\'\')", re.MULTILINE)
# str.format()-style placeholders: a lowercase identifier in SINGLE braces,
# e.g. {context}. Deliberately does NOT match JSON examples like {"a": 1} or
# doubled {{escaped}} braces, which legitimate prompts can contain.
_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{([a-z_][a-z0-9_]*)\}(?!\})")


def validate_template(template: str) -> list[str]:
    """Returns a list of human-readable problems that make `template`
    unusable AS A SYSTEM PROMPT ([] = fine). The override is sent to the
    model verbatim - nothing ever calls .format() on it - so:
      - Python source (e.g. `RESEARCH_PROMPT = \"\"\"...`) means the model
        receives code, possibly several alternative prompts, rather than one
        set of instructions.
      - A {placeholder} would reach the model as literal braces; the actual
        question/context are supplied separately in the user turn, so a
        template expecting them to be substituted into itself will never see
        them filled in.
    Deterministic, no LLM call - same convention as the rest of this
    codebase for checks that can be decided from the text alone."""
    problems = []
    assignments = _PY_ASSIGNMENT_RE.findall(template or "")
    if assignments:
        problems.append(f"looks like Python source code ({len(assignments)} triple-quoted string "
                        f"assignment(s), e.g. NAME = \"\"\"...) rather than plain prompt text")
    placeholders = sorted(set(_PLACEHOLDER_RE.findall(template or "")))
    if placeholders:
        shown = ", ".join("{" + p + "}" for p in placeholders[:6])
        more = f" (+{len(placeholders) - 6} more)" if len(placeholders) > 6 else ""
        problems.append(f"contains unfilled placeholder(s) {shown}{more} - the override is sent to the "
                        f"model verbatim; the question and retrieved context are supplied separately")
    return problems
