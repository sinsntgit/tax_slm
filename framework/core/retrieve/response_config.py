"""
User-editable configuration for HOW the LLM structures its /api/ask answer
(framework/webapp/app.py's answer_generation call) - what sections to
include, any extra instructions, and tone. Domain-agnostic: any domain pack's
webapp gets this for free.

Hot-reloadable, same pattern as framework.core.infra.llm_client's
refresh_model_config() (re-read .env on every call): load_response_config()
re-reads the JSON config file from disk on every /api/ask request, so a user
edit (via the UI's Response Format panel, or hand-editing the file) takes
effect on the very next question - no restart needed.
"""
import json
import os
from pathlib import Path

# Repo root (this file lives at framework/core/retrieve/response_config.py -
# three parents up is the repo root, same depth as .env/.env.example).
_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = _REPO_ROOT / "response_config.json"

DEFAULT_CONFIG = {
    "sections": [
        "Intent (what the question is asking for)",
        "Direct Answer",
        "Citations (document name, section/structural citation, page number)",
    ],
    "instructions": "",
    "tone": "concise, plain-language",
}


def _config_path() -> Path:
    override = os.environ.get("RESPONSE_CONFIG_PATH", "").strip()
    return Path(override) if override else DEFAULT_CONFIG_PATH


def load_response_config() -> dict:
    """Re-reads the config file from disk every call (cheap - it's a small
    JSON file). Falls back to DEFAULT_CONFIG - whole file missing, invalid
    JSON, or not a dict - so a bad hand-edit can never break /api/ask, it
    just silently reverts to the default structure for that request."""
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


def save_response_config(cfg: dict) -> dict:
    """Validates + writes cfg to disk. Returns the normalized dict actually
    written (unknown keys dropped, missing keys filled from DEFAULT_CONFIG)."""
    normalized = dict(DEFAULT_CONFIG)
    if isinstance(cfg.get("sections"), list):
        normalized["sections"] = [str(s) for s in cfg["sections"] if str(s).strip()]
    if isinstance(cfg.get("instructions"), str):
        normalized["instructions"] = cfg["instructions"]
    if isinstance(cfg.get("tone"), str) and cfg["tone"].strip():
        normalized["tone"] = cfg["tone"]
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalized, f, indent=2)
    return normalized


def build_structure_directive(cfg: dict) -> str:
    """Renders cfg into a directive block to append to the answer_generation
    system prompt, so the response structure and the retrieval/citation
    approach the base prompt already establishes both still apply -
    this only shapes HOW the answer is organized, on top of that."""
    lines = ["\n\nRESPONSE STRUCTURE (user-configured - organize your answer using exactly "
              "these sections, in this order, as clear headings or labeled paragraphs):"]
    for i, s in enumerate(cfg.get("sections") or DEFAULT_CONFIG["sections"], 1):
        lines.append(f"{i}. {s}")
    if cfg.get("instructions"):
        lines.append(f"\nAdditional instructions from the user: {cfg['instructions']}")
    if cfg.get("tone"):
        lines.append(f"Tone: {cfg['tone']}")
    lines.append(
        "\nRegardless of the above, always: (a) state the detected intent given to you "
        "separately in the context - do not invent or reclassify it; (b) for every citation, "
        "give the document name, the precise structural citation (e.g. Section/SubSection/"
        "Clause, or the equivalent paragraph/part for non-Act documents), and the page number "
        "when it's present in the source metadata given to you - omit only the page number if "
        "it isn't available, never fabricate one."
    )
    return "\n".join(lines)
