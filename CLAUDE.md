# Tax Knowledge Graph POC - Project Guide for Claude Code

## What this is

A Neo4j-backed Tax Knowledge Graph + GraphRAG retrieval system for Indian income-tax research (Acts, Case Law, CBDT Circulars, International Agreements/DTAAs), serving a KPMG-style 6-section research answer format via a Flask webapp. Ingestion (PDF -> extraction -> graph load -> embedding) and retrieval (hybrid graph+vector, category-based coverage, reranking) are both first-class, heavily-iterated subsystems - read before assuming either is simple.

## Structure

- `framework/` - domain-agnostic core: `core/ingest/` (graph loader, embedder), `core/retrieve/` (graphrag.py hybrid retriever, reranker.py, category_retrieval.py, intent.py, answer_evaluation.py), `webapp/` (Flask app + static frontend), `cli/` (thin CLI wrappers).
- `domain_packs/tax/` - everything tax-specific: `ontology/schema.json` (the OWL-derived ontology - classes, object_properties, data_properties), `converters/` (per-document-type PDF->extraction->graph scripts: `extract_entities.py`/Act, `extract_case_law.py`, `extract_circular.py`, `extract_treaty.py`, plus `ingest_*.py` single-command orchestrators), `prompts/tax_prompts.py`, `config.py` (DOMAIN_CONFIG, TAX_CATEGORY_SPECS).
- Root-level `.md` files are real design/reference docs, not scratch notes - read `README.md` and `Modification_Guide.md` first, and `Graph_First_Retrieval_Design.md` / `Testing_Graph_First_Retrieval.md` for the most recent retrieval-architecture work (case-law stance-balanced retrieval).

## Engineering conventions already established here - follow, don't reinvent

- **Prefer deterministic derivation over a new LLM call whenever the graph/extraction already has what's needed.** See `extract_case_law.py`'s `classify_forum_level()`/`derive_jurisdiction_scope()`/`derive_stance()` - all derived from already-extracted structured fields, not fresh model judgment calls. Reach for this before adding another prompt.
- **Leave a field unset rather than guess wrong.** Repeated pattern across the codebase (see `derive_jurisdiction_scope()`'s docstring) - an empty/unset property is preferred over a plausible-but-wrong one, especially for anything legally consequential.
- **Every non-obvious change gets a comment explaining WHY, dated (`2026-08-27: ...`), not just what.** This is the dominant style throughout the codebase - match it. Comments frequently reference the specific bug report or user request that motivated the change.
- **Fallback tiers, not hard failures.** The reranker (`DEFAULT_FALLBACK_RELEVANCE`), category retrieval (graph-first vector-fallback), and the new stance-balanced case-law retrieval (falls through to the generic path if thin) all follow the same shape: try the precise path, degrade gracefully to a wider net, never just return nothing when something reasonable exists.
- **One bad item never stops a batch.** Upload/ingestion/extraction loops log-and-continue on a single failure rather than aborting the whole run.
- **Validate against `ontology/schema.json` after any extraction change** - every converter has a `validate_against_schema()` sanity check; keep using it.
- **Verify before declaring done**: `python -m py_compile` on every touched file, plus a mocked unit test exercising the actual logic (not just import-succeeds), before reporting a change complete. Real end-to-end verification (against a real sample PDF/question) beats a mock when one is available cheaply.

## Hard rules - do not violate

- **Never read or print the real `.env` file's contents/values.** Only check which keys are SET, e.g. `grep -oE "^[A-Z_]+=" .env | sort`, or print a value's length/prefix at most. Never print a full secret into a response, log, or commit.
- **This project's data/PDF corpus is large** (individual Act PDFs run 70-180MB). Be deliberate about full-file reads/greps over `data/rawdata/`.

## Version control status

2026-10-14: now a git repository (branch `main`), developed in a cloud workspace and pushed to a private GitHub repo; the owner pulls it to the Windows dev machine (D:\tax_kg_poc_revised) to run against Docker. `.gitignore` excludes `.env`, `venv/`, all of `data/` (corpus, uploads, graphs, vector stores) and `kpmg_tax_slm_documents_artifacts/` (client deliverables). Small public/fictitious samples used by tests live in `tests/fixtures/`. The pre-git change history lives in the dated code comments, not in git.

## Team workflow (multiple agents)

- One branch per workstream (`phase0/<area>`), merged into `main` only after the full test suite passes.
- Stay inside your workstream's files; if you must touch a shared file (`app.py`, `tax_prompts.py`, `pipeline.py`, `llm_client.py`, `schema.json`), make the smallest additive change and add NEW prompt versions rather than editing existing ones.
- Prompts: never edit a registered PromptVersion's text - append a new version with `notes` explaining why.
- Every change ships with tests under `tests/` that exercise the real logic.

## Testing this project

- `pytest` from the repo root (cloud venv: `/home/claude/venv/bin/python -m pytest`). `tests/conftest.py` installs offline stand-ins for `sentence_transformers` (no Hugging Face access in the cloud) - code must import and run with them.
- No Neo4j server in the cloud: unit tests mock the driver/session. Tests that need real Neo4j/LLM/.env are marked `@pytest.mark.live` and only run with `TAXKG_LIVE_TESTS=1` on the dev machine.
- `tests/test_smoke_imports.py` must always pass - it already caught an import-time NameError in `llm_client.py` (2026-10-14).
