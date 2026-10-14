"""2026-10-14: NEW - the hot-reloaded, Settings-page-editable config modules:
prompt_config.py (answer prompt override + validate_template), retrieval_router.py
(lineage vs recall routing) and comparative_response.py (for/against structure)."""
import json

import pytest

from framework.core.retrieve import comparative_response as cmp
from framework.core.retrieve import prompt_config as pc
from framework.core.retrieve import retrieval_router as rr

REPO_ROOT = pc._REPO_ROOT


# --- prompt_config.validate_template -------------------------------------------

def test_validate_template_accepts_plain_prompt_with_json_and_escaped_braces():
    t = ('You are a tax research assistant. Return JSON like {"answer": "...", "sources": []}. '
         'Literal {{context}} braces are escaped. Cite [Document: X, p. 1].')
    assert pc.validate_template(t) == []
    assert pc.validate_template("") == []


def test_validate_template_flags_python_source():
    t = 'RESEARCH_PROMPT = """\nDo research.\n"""\nLOOKUP_PROMPT = r\'\'\'\nx\n\'\'\''
    problems = pc.validate_template(t)
    assert len(problems) == 1 and "2 triple-quoted string assignment(s)" in problems[0]


def test_validate_template_flags_placeholders_and_caps_listing():
    t = " ".join("{p%d}" % i for i in range(8)) + " {context}"
    [problem] = pc.validate_template(t)
    assert "{context}" in problem and "(+3 more)" in problem


def test_real_repo_prompt_config_is_rejected():
    """The committed prompt_config.json holds a pasted Python prompts module
    (the 2026-10-13 incident) - it must be detected as unusable."""
    cfg = json.loads((REPO_ROOT / "prompt_config.json").read_text(encoding="utf-8"))
    problems = pc.validate_template(cfg["template"])
    assert len(problems) == 2
    assert "Python source" in problems[0] and "{context}" in problems[1]


def test_real_repo_prompt_config_is_ignored_by_resolver(monkeypatch):
    monkeypatch.delenv("PROMPT_CONFIG_PATH", raising=False)
    assert pc.resolve_answer_system_prompt("BUILT-IN") == ("BUILT-IN", False)


def test_every_builtin_answer_prompt_passes_validation():
    from domain_packs.tax.prompts.tax_prompts import PROMPT_PACK
    names = [n for n in PROMPT_PACK.repo if n.startswith("answer_generation")]
    assert {"answer_generation", "answer_generation_lookup", "answer_generation_opinion"} <= set(names)
    for name in names:
        for v in PROMPT_PACK.repo[name]:
            assert pc.validate_template(v.template) == [], f"{name} {v.version}"


# --- prompt_config load/save/resolve ----------------------------------------------

@pytest.fixture
def prompt_path(tmp_path, monkeypatch):
    p = tmp_path / "prompt_config.json"
    monkeypatch.setenv("PROMPT_CONFIG_PATH", str(p))
    return p


def test_load_override_defaults(prompt_path):
    assert pc.load_prompt_override() == {"enabled": False, "template": ""}
    prompt_path.write_text("not json", encoding="utf-8")
    assert pc.load_prompt_override() == {"enabled": False, "template": ""}
    prompt_path.write_text('"a string"', encoding="utf-8")
    assert pc.load_prompt_override() == {"enabled": False, "template": ""}


def test_save_override_valid_enabled_and_resolved(prompt_path):
    out = pc.save_prompt_override({"enabled": True, "template": "  Be precise.  "})
    assert out == {"enabled": True, "template": "Be precise.", "problems": []}
    assert json.loads(prompt_path.read_text(encoding="utf-8")) == {"enabled": True, "template": "Be precise."}
    assert pc.resolve_answer_system_prompt("DEFAULT") == ("Be precise.", True)


def test_save_override_invalid_is_kept_as_disabled_draft(prompt_path):
    out = pc.save_prompt_override({"enabled": True, "template": "Answer {question} using {context}"})
    assert out["enabled"] is False and out["template"] == "Answer {question} using {context}"
    assert len(out["problems"]) == 1
    assert "problems" not in json.loads(prompt_path.read_text(encoding="utf-8"))
    assert pc.resolve_answer_system_prompt("DEFAULT") == ("DEFAULT", False)


def test_save_blank_template_disables(prompt_path):
    assert pc.save_prompt_override({"enabled": True, "template": "   "})["enabled"] is False


def test_resolver_ignores_invalid_file_saved_before_validation_existed(prompt_path):
    prompt_path.write_text(json.dumps({"enabled": True, "template": "X = '''y'''"}), encoding="utf-8")
    assert pc.resolve_answer_system_prompt("DEFAULT") == ("DEFAULT", False)
    prompt_path.write_text(json.dumps({"enabled": False, "template": "fine"}), encoding="utf-8")
    assert pc.resolve_answer_system_prompt("DEFAULT") == ("DEFAULT", False)


# --- retrieval_router ------------------------------------------------------------

@pytest.mark.parametrize("q,expected", [
    ("Has Samsung Electronics been OVERRULED?", "lineage"),
    ("Is CIT v. Samsung still good law after Engineering Analysis?", "lineage"),
    ("What is the current status of Circular 7/2021", "lineage"),
    ("Define royalty under section 9(1)(vi)", "recall"),
    ("", "recall"),
    (None, "recall"),
])
def test_route_with_default_config(q, expected):
    assert rr.route(q, dict(rr.DEFAULT_CONFIG)) == expected


def test_route_disabled_and_custom_keywords():
    assert rr.route("overruled?", {**rr.DEFAULT_CONFIG, "enabled": False}) == "recall"
    assert rr.route("what about x", {"enabled": True, "lineage_keywords": ["", "about"]}) == "lineage"
    assert rr.route("what about x", {"enabled": True, "lineage_keywords": None}) == "recall"


def test_router_load_and_save(tmp_path, monkeypatch):
    p = tmp_path / "rr.json"
    monkeypatch.setenv("RETRIEVAL_ROUTER_CONFIG_PATH", str(p))
    assert rr.load_config() == rr.DEFAULT_CONFIG
    out = rr.save_config({"enabled": False, "lineage_keywords": [" Overrule ", " ", 5],
                          "expansion_seed_top_k": 99, "expansion_max_neighbors": 30.7, "x": 1})
    assert out == {"enabled": False, "lineage_keywords": ["overrule", "5"],
                   "expansion_seed_top_k": rr.DEFAULT_CONFIG["expansion_seed_top_k"], "expansion_max_neighbors": 30}
    assert rr.load_config() == out
    assert rr.route("was it overruled", None) == "recall"  # disabled config re-read from disk
    # blank keyword list never wipes the feature out
    assert rr.save_config({"lineage_keywords": ["  "]})["lineage_keywords"] == rr.DEFAULT_LINEAGE_KEYWORDS


def test_repo_router_config_loads(monkeypatch):
    monkeypatch.delenv("RETRIEVAL_ROUTER_CONFIG_PATH", raising=False)
    cfg = rr.load_config()
    assert isinstance(cfg["enabled"], bool) and cfg["lineage_keywords"]


# --- comparative_response -----------------------------------------------------------

@pytest.mark.parametrize("q,expected", [
    ("Give me arguments for and against treating this as royalty", True),
    ("What is the FAVOURABLE view for the assessee?", True),
    ("Compare the two High Court rulings", True),
    ("What is the TDS rate on rent?", False),
    (None, False),
])
def test_is_comparative_question(q, expected):
    assert cmp.is_comparative_question(q, dict(cmp.DEFAULT_CONFIG)) is expected


def test_is_comparative_disabled():
    assert cmp.is_comparative_question("pros and cons", {**cmp.DEFAULT_CONFIG, "enabled": False}) is False


def test_build_structure_directive_numbers_sections():
    d = cmp.build_structure_directive({"sections": ["Alpha", "Beta (detail)"]})
    assert "RESPONSE STRUCTURE OVERRIDE" in d
    assert "\n1. Alpha\n2. Beta (detail)\n" in d
    default = cmp.build_structure_directive({"sections": []})
    assert f"{len(cmp.DEFAULT_SECTIONS)}. {cmp.DEFAULT_SECTIONS[-1]}" in default


@pytest.mark.parametrize("section,phrase", [
    ("Short answer / current position", "short answer"),
    ("Cases supporting — historical Revenue line", "cases supporting"),
    ("Applicable statutory provisions relied upon (Section/Rule cited by each side)",
     "applicable statutory provisions"),
    ("One-line conclusion", "one-line conclusion"),
    ("", ""),
])
def test_section_key_phrase(section, phrase):
    assert cmp._section_key_phrase(section) == phrase


def test_check_structure_compliance():
    cfg = {"sections": ["Short answer / current position", "Cases against — taxpayer line", "One-line conclusion"]}
    good = "## 1. Short Answer\n...\n## 2. Cases against\n...\n## 3. One-line conclusion\n..."
    out = cmp.check_structure_compliance(good, cfg)
    assert out == {"checked": True, "sections_expected": 3, "sections_found": 3,
                   "missing_sections": [], "compliant": True}
    bad = cmp.check_structure_compliance("## Short answer only", cfg)
    assert bad["compliant"] is False and bad["sections_found"] == 1
    assert bad["missing_sections"] == cfg["sections"][1:]


def test_every_default_section_is_detectable_when_echoed_verbatim():
    answer = "\n".join(f"## {i}. {s}" for i, s in enumerate(cmp.DEFAULT_SECTIONS, 1))
    assert cmp.check_structure_compliance(answer, dict(cmp.DEFAULT_CONFIG))["compliant"] is True


def test_comparative_save_and_load(tmp_path, monkeypatch):
    p = tmp_path / "c.json"
    monkeypatch.setenv("COMPARATIVE_RESPONSE_CONFIG_PATH", str(p))
    assert cmp.load_config() == cmp.DEFAULT_CONFIG
    out = cmp.save_config({"enabled": "yes", "sections": ["  A ", ""], "trigger_keywords": [" VS "]})
    assert out == {"enabled": True, "sections": ["A"], "trigger_keywords": ["vs"]}
    assert cmp.is_comparative_question("India vs UK treaty") is True
    p.write_text("{broken", encoding="utf-8")
    assert cmp.load_config() == cmp.DEFAULT_CONFIG
