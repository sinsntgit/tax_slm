"""2026-10-15: NEW - scripts/export_retrieval_prompts.py must resolve every
call site's prompt (name + pinned version) against the live PromptPack, so a
renamed/removed prompt fails CI instead of silently dropping out of the
developer reference (docs/RETRIEVAL_PROMPTS.md)."""
import importlib.util
from pathlib import Path

from domain_packs.tax.prompts.tax_prompts import PROMPT_PACK

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "export_retrieval_prompts.py"


def _load():
    spec = importlib.util.spec_from_file_location("export_retrieval_prompts", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_call_site_resolves_and_is_exported():
    mod = _load()
    text = mod.build()
    for c in mod.CALLS:
        pv = PROMPT_PACK.get_prompt(c["name"], version=c["version"])
        assert f"## {c['name']} {pv.version}" in text
        assert pv.template.strip()[:80] in text
    assert "## Inline prompts in code" in text and "JUDGE" not in text[:200]


def test_fences_are_balanced():
    text = _load().build()
    opens = [l for l in text.splitlines() if l.startswith("```") and l.endswith("text")]
    assert opens, "expected fenced prompt blocks"
