"""
Export every prompt the app sends to an LLM at QUESTION time (retrieval and
answering - not ingestion/extraction) into one Markdown reference,
docs/RETRIEVAL_PROMPTS.md, for developers and reviewers.

2026-10-15: NEW - per request to "extract all the prompts used for retrieval"
to share with a developer. Generated from the live PromptPack + the inline
prompt strings in code (not copied by hand) so it can't drift: re-run after any
prompt change.

    python scripts/export_retrieval_prompts.py            # writes docs/RETRIEVAL_PROMPTS.md
    python scripts/export_retrieval_prompts.py --out x.md

The call-site table below (CALLS) is maintained by hand - it records facts the
PromptPack itself doesn't know (which version a call site pins, max_tokens,
which retrieval mode triggers it). If you add/move an LLM call, update it.
"""
import argparse
import datetime as _dt
import os
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Offline-safe: the export only needs prompt text, never models. Install the same
# sentence_transformers stand-in the test suite uses if the real one is absent.
try:  # pragma: no cover - environment dependent
    import sentence_transformers  # noqa: F401
except Exception:  # noqa: BLE001
    import types
    _st = types.ModuleType("sentence_transformers")
    _st.SentenceTransformer = object
    _st.CrossEncoder = object
    sys.modules["sentence_transformers"] = _st

from domain_packs.tax.prompts.tax_prompts import PROMPT_PACK  # noqa: E402

# name -> call-site facts. `version`: None = latest registered (the default
# get_prompt() behaviour), else the version that call site pins.
# `formatted`: placeholders filled by .format() at runtime (None = sent verbatim).
CALLS = [
    # --- every /api/ask -----------------------------------------------------
    dict(name="intent_detection", version=None, call="intent_detection", max_tokens=20,
         where="framework/core/retrieve/intent.py: detect_intent()",
         when="Every /api/ask except fast mode. User turn = the question.",
         formatted=None),
    dict(name="nl2cypher", version=None, call="nl2cypher", max_tokens="NL2CYPHER_MAX_TOKENS (default 1024)",
         where="framework/core/retrieve/graphrag.py: GraphRAGRetriever.nl_to_cypher()",
         when="KG + RAG, Semantic Graph, Agentic (CYPHER tool), Structured Research (Law/Case Law stages). "
              "Not Simple RAG or fast mode. User turn = the question (or sub-query).",
         formatted="{labels}, {rel_types} = live Neo4j labels / relationship types"),
    dict(name="answer_generation", version=None, call="answer_generation (retry: answer_generation_retry)",
         max_tokens=5000,
         where="framework/webapp/app.py: _api_ask_impl() -> _generate()",
         when="Answer mode = Research (default). A saved Settings-page override replaces it if it passes "
              "prompt_config.validate_template(). User turn = assembled context (see 'User-turn assembly').",
         formatted=None),
    dict(name="answer_generation_lookup", version=None, call="answer_generation", max_tokens=5000,
         where="framework/webapp/app.py: _api_ask_impl() -> _generate()",
         when="Answer mode = Lookup.", formatted=None),
    dict(name="answer_generation_opinion", version=None, call="answer_generation", max_tokens=5000,
         where="framework/webapp/app.py: _api_ask_impl() -> _generate()",
         when="Answer mode = Client Opinion.", formatted=None),
    dict(name="answer_disclaimer", version=None, call="(not an LLM call)", max_tokens="-",
         where="framework/webapp/app.py: _api_ask_impl()",
         when="Appended verbatim after every generated answer.", formatted=None),
    # --- agentic ------------------------------------------------------------
    dict(name="retrieval_plan", version="v1", call="retrieval_plan", max_tokens=600,
         where="framework/core/retrieve/retrieval_planner.py: plan_retrieval()",
         when="Agentic mode, each of up to 3 iterations.",
         formatted="{labels}, {rel_types}, {prior_attempts}, {question}"),
    dict(name="evidence_sufficiency", version="v1", call="evidence_sufficiency", max_tokens=150,
         where="framework/core/retrieve/evidence_sufficiency.py: check_sufficiency()",
         when="Agentic mode, after each iteration.", formatted="{question}, {evidence_summary}"),
    # --- structured research --------------------------------------------------
    dict(name="fact_sheet_extraction", version=None, call="fact_sheet_extraction", max_tokens=600,
         where="framework/core/retrieve/fact_sheet_agent.py: build_fact_sheet()",
         when="Structured Research, stage 1.", formatted="{attached_file_context}, {question}"),
    dict(name="case_law_classification", version="v1", call="case_law_classification", max_tokens=1500,
         where="framework/core/retrieve/case_law_agent.py: retrieve_case_law()",
         when="Structured Research, Case Law stage.", formatted="{tax_position}, {candidate_cases}"),
    dict(name="tribunal_bucketing", version="v1", call="tribunal_bucketing", max_tokens=1500,
         where="framework/core/retrieve/tribunal_case_law_agent.py: retrieve_tribunal_case_law()",
         when="Structured Research, Tribunal Case Law stage.",
         formatted="{matter_jurisdiction}, {change_in_law_date}, {candidate_cases}"),
    dict(name="hierarchical_synthesis", version="v1", call="hierarchical_synthesis", max_tokens=3000,
         where="framework/webapp/app.py: _run_unstructured_agent() (structured_research branch)",
         when="Structured Research, final answer.",
         formatted="{facts}, {tax_position}, {law_findings}, {case_law_findings}, {tribunal_findings}, "
                   "{opinion_findings}, {question}"),
    # --- attached spreadsheet ---------------------------------------------------
    dict(name="agent_routing", version="v2", call="agent_routing", max_tokens=20,
         where="framework/webapp/app.py: _classify_agent_routing()",
         when="Only when a .xlsx/.csv/.json file is attached to the chat.",
         formatted="{filename}, {schema_summary}"),
    dict(name="structured_data_query", version="v1", call="structured_data_query", max_tokens=800,
         where="framework/webapp/app.py: _generate_structured_query_code()",
         when="Attached file routed STRUCTURED_ONLY or BOTH. Output = pandas code run in a sandbox.",
         formatted="{schema_summary}, {question}, {retry_context}"),
    dict(name="agent_merge", version="v1", call="agent_merge", max_tokens=2000,
         where="framework/webapp/app.py: _run_agentic_ask()",
         when="Attached file routed BOTH.",
         formatted="{unstructured_agent_answer}, {structured_agent_answer}, {question}"),
    # --- evaluation -------------------------------------------------------------
    dict(name="answer_evaluation", version="v2", call="answer_evaluation", max_tokens=1200,
         where="framework/webapp/app.py: _api_ask_impl() (auto_evaluate)",
         when="'Auto-evaluate' toggle on. v1 (below) is used by the Evaluate button and quality-checked/fast retry loops.",
         formatted=None),
    dict(name="answer_evaluation", version="v1", call="answer_evaluation", max_tokens=1200,
         where="framework/webapp/app.py: /api/ask/evaluate, _run_quality_checked_ask_job(), _run_fast_quality_checked_ask()",
         when="Evaluate button; quality-checked and fast retry loops.", formatted=None),
    dict(name="general_knowledge_evaluation", version="v1", call="answer_evaluation_general_knowledge",
         max_tokens=1200,
         where="framework/core/retrieve/answer_evaluation.py: evaluate_answer(no_context_mode=True)",
         when="Alongside answer_evaluation whenever it runs; graded with NO retrieved context.",
         formatted=None),
    # --- opinions workspace -------------------------------------------------------
    dict(name="document_qa", version=None, call="document_qa", max_tokens=2000,
         where="framework/core/retrieve/opinion_workspace.py: ask()",
         when="Generate Opinions screen, Q&A over uploaded matter documents.",
         formatted="{document_context}, {kg_context_block}, {question}"),
    dict(name="opinion_generation", version=None, call="opinion_generation", max_tokens=4000,
         where="framework/core/retrieve/opinion_workspace.py: generate_opinion()",
         when="Generate Opinions screen, opinion draft.",
         formatted="{matter_context}, {document_context}, {kg_context_block} (+ others in template)"),
]


def _git(*args):
    try:
        return subprocess.check_output(["git", *args], cwd=REPO, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _fence(text: str) -> str:
    # Use a fence longer than any backtick run inside the prompt.
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    f = "`" * max(3, longest + 1)
    return f"{f}text\n{text.rstrip()}\n{f}"


def _anchor(s: str) -> str:
    return re.sub(r"[^a-z0-9-]", "", s.lower().replace(" ", "-"))


def _inline_prompts():
    """Prompt text that lives in code rather than the PromptPack."""
    from framework.core.retrieve import answer_quality, category_retrieval, comparative_response, response_config
    from domain_packs.tax.config import TAX_CATEGORY_SPECS

    items = []
    items.append(("answer_quality_judge (system prompt)",
                  "framework/core/retrieve/answer_quality.py: JUDGE_SYSTEM_PROMPT - call name answer_quality_judge, "
                  "max_tokens 500. Runs after every non-fast /api/ask; a score below 0.6 triggers one wider-context retry.",
                  answer_quality.JUDGE_SYSTEM_PROMPT))

    cmp_cfg = comparative_response.load_config()
    items.append(("Comparative / for-against structure directive",
                  "framework/core/retrieve/comparative_response.py: build_structure_directive() - APPENDED to the "
                  "answer system prompt when the question matches the trigger keywords "
                  f"(currently: {', '.join(cmp_cfg.get('trigger_keywords', [])[:12])}...). Editable on the Settings page; "
                  "rendered here from the current config.",
                  comparative_response.build_structure_directive(cmp_cfg)))

    items.append(("Response-format directive",
                  "framework/core/retrieve/response_config.py: build_structure_directive() - appended ONLY for "
                  "answer_generation v1-v3. Not used with the current v7 (which defines its own structure). "
                  "Rendered from the current response_config.json.",
                  response_config.build_structure_directive(response_config.load_response_config())))

    demo = [category_retrieval._result_from_spec(s, status="synthesis" if s.synthesis_only else "not_found",
                                                  note="(example status)")
            for s in TAX_CATEGORY_SPECS]
    items.append(("Category coverage block (user turn)",
                  "framework/core/retrieve/category_retrieval.py: format_category_context() - inserted into the "
                  "answer user turn. Shown with every category as NOT FOUND / SYNTHESIS for illustration; at runtime "
                  "each FOUND category carries its retrieved chunk text. Per-category INSTRUCTIONS come from "
                  "domain_packs/tax/config.py TAX_CATEGORY_SPECS.",
                  category_retrieval.format_category_context(demo)))

    prefixes = "\n".join(f"{s.key}: {s.vector_query_prefix!r}" for s in TAX_CATEGORY_SPECS if s.vector_query_prefix)
    items.append(("Category vector-search query prefixes (embedding queries, not LLM prompts)",
                  "domain_packs/tax/config.py TAX_CATEGORY_SPECS.vector_query_prefix - prepended to the question "
                  "for each category's vector fallback search.", prefixes))

    items.append(("Quality-gate retry instruction (user turn)",
                  "framework/webapp/app.py: _api_ask_impl() - added to the user turn of answer_generation_retry "
                  "when the judge scores the first answer below 0.6. {gap_text} = the judge's gaps.",
                  "IMPORTANT: A prior draft of this answer was reviewed and found lacking ({gap_text}). "
                  "You now have a WIDER set of source chunks below than before - use them to give a more "
                  "complete, detailed, precisely-cited answer. Do not repeat the same gaps."))

    items.append(("Evaluator-feedback retry instruction (user turn)",
                  "framework/webapp/app.py: _build_eval_retry_instructions() - used by the quality-checked and fast "
                  "retry loops; merges issues from both evaluators.",
                  "IMPORTANT: An independent reviewer scored your previous attempt at this exact question "
                  "{score}/10 (target: {target_score}/10). Specific issues raised: {issues}. "
                  "Address these directly in this revised answer - do not repeat the same gaps."))
    return items


USER_TURN = """The answer model's USER turn is assembled in framework/webapp/app.py
_api_ask_impl() -> _build_context_parts(), in this order (parts joined by blank lines):

1. "Question: <question>"
2. "Detected intent: <EXACT_LOOKUP | RELATIONAL | CONCEPTUAL | None in fast mode>"
3. Extra instructions, if any (retry instructions - see Inline prompts)
4. The category coverage block (see Inline prompts)
5. "Graph query results (JSON):" + Cypher rows (bulky fields capped per property,
   repeated nodes de-duplicated - grounding.graph_rows_to_context_json, 16,000-char backstop)
6. "Retrieved source text chunks (reranked by relevance):" - each chunk as
   "[chunk_id=<id> | <document name>, p. <page> | provenance=<...>]" + up to 1,500 chars of text
7. "Full parent-section context for the chunks above ..." - each parent Section up to 3,000 chars

All of it passes through PII masking before it reaches the model."""


def build() -> str:
    head = _git("rev-parse", "--short", "HEAD")
    today = _dt.date.today().isoformat()
    out = []
    out.append("# Retrieval & answering prompts - Tax KG / Tax SLM\n")
    out.append(f"Generated {today} from commit `{head}` by `scripts/export_retrieval_prompts.py`. "
               "Do not edit by hand - change the prompt in code and re-run the script.\n")
    out.append("""**Scope:** every prompt sent to an LLM while answering a question (retrieval, answer generation,
evaluation, agents). Ingestion/extraction prompts (Act, case law, circular, treaty extraction) are excluded.

**How versions work:** prompts live in `domain_packs/tax/prompts/tax_prompts.py` as a list of
`PromptVersion`s per name. A call site gets the **latest** version unless it pins one
(`get_prompt(name, version="v1")`). Old versions are kept for traceability - only the versions
marked *active* below are sent. Every call goes through `framework/core/infra/llm_client.complete()`;
the "call name" column is what `LLM_ROUTE_<CALL_NAME>` routing and the call log use.

**Placeholders:** templates marked "formatted" have `{placeholders}` filled with `.format()` at
runtime (`{{`/`}}` in those templates are literal braces). Templates marked "verbatim" are sent as-is.
""")

    out.append("## Which prompts run in each mode\n")
    out.append("""| Mode / feature | Prompts (in order) |
|---|---|
| KG + RAG (hybrid), Semantic Graph | intent_detection -> nl2cypher -> answer_generation* -> answer_quality_judge (-> answer_generation_retry if score < 0.6) |
| Simple RAG (vector_only) | intent_detection -> answer_generation* -> answer_quality_judge |
| Fast mode (/api/ask/fast) | answer_generation* -> answer_evaluation v1 + general_knowledge_evaluation (concurrent), up to 2 attempts |
| Agentic | retrieval_plan -> (nl2cypher if CYPHER chosen) -> evidence_sufficiency, up to 3 rounds -> intent_detection -> answer_generation* -> answer_quality_judge |
| Structured Research | fact_sheet_extraction -> nl2cypher (x6 Law sub-queries + case law) -> case_law_classification + tribunal_bucketing (parallel) -> hierarchical_synthesis |
| Attached spreadsheet | agent_routing -> structured_data_query (+ agent_merge when BOTH) |
| Auto-evaluate toggle | + answer_evaluation v2 + general_knowledge_evaluation |
| Evaluate button / quality-checked retry | answer_evaluation v1 + general_knowledge_evaluation |
| Generate Opinions screen | document_qa, opinion_generation |

\\* answer_generation = `answer_generation` (Research), `answer_generation_lookup` (Lookup) or
`answer_generation_opinion` (Client Opinion), plus the comparative directive when triggered and
the fixed disclaimer appended after the answer.
""")

    out.append("## Contents\n")
    for c in CALLS:
        v = c["version"] or PROMPT_PACK.get_prompt(c["name"]).version
        out.append(f"- [{c['name']} {v}](#{_anchor(c['name'] + '-' + v)})")
    out.append("- [Inline prompts in code](#inline-prompts-in-code)")
    out.append("- [User-turn assembly for the answer model](#user-turn-assembly-for-the-answer-model)")
    out.append("- [Appendix: version history](#appendix-version-history)\n")

    for c in CALLS:
        pv = PROMPT_PACK.get_prompt(c["name"], version=c["version"])
        latest = PROMPT_PACK.get_prompt(c["name"]).version
        pin = "latest" if c["version"] is None else f"pinned to {c['version']} (latest registered: {latest})"
        out.append(f"## {c['name']} {pv.version}\n")
        out.append("| | |\n|---|---|")
        out.append(f"| Active version | {pv.version} - {pin}, approved {pv.approved_at} |")
        out.append(f"| Call name | `{c['call']}` |")
        out.append(f"| Called from | `{c['where']}` |")
        out.append(f"| When | {c['when']} |")
        out.append(f"| max_tokens | {c['max_tokens']} |")
        out.append(f"| Placeholders | {'formatted: ' + c['formatted'] if c['formatted'] else 'verbatim (no .format())'} |\n")
        if pv.notes:
            out.append(f"**Why this version:** {pv.notes}\n")
        out.append(_fence(pv.template) + "\n")

    out.append("## Inline prompts in code\n")
    for title, where, text in _inline_prompts():
        out.append(f"### {title}\n")
        out.append(f"{where}\n")
        out.append(_fence(text) + "\n")

    out.append("## User-turn assembly for the answer model\n")
    out.append(_fence(USER_TURN) + "\n")

    out.append("## Appendix: version history\n")
    out.append("All registered versions of the prompts above (only the active ones are sent).\n")
    out.append("| Prompt | Version | Approved | Notes |\n|---|---|---|---|")
    names = []
    for c in CALLS:
        if c["name"] not in names:
            names.append(c["name"])
    for n in names:
        for pv in PROMPT_PACK.repo[n]:
            note = (pv.notes or "").replace("|", "\\|").replace("\n", " ")
            if len(note) > 220:
                note = note[:217] + "..."
            out.append(f"| {n} | {pv.version} | {pv.approved_at} | {note} |")
    out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(REPO / "docs" / "RETRIEVAL_PROMPTS.md"))
    args = ap.parse_args()
    text = build()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"Wrote {args.out} ({len(text):,} chars)")


if __name__ == "__main__":
    main()
