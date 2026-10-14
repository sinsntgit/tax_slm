# tax_kg_poc_revised — Framework + Domain Packs

This is a restructured copy of `tax_kg_poc`, split into a domain-agnostic
**framework** (document conversion, graph ingestion, GraphRAG retrieval, the
webapp, and a CLI) and a **domain pack** (`domain_packs/tax/`) holding
everything specific to Indian income-tax law: the ontology, the Act/case-law
converters, and the prompt text.

**The original `tax_kg_poc` tree is untouched.** This folder is a fresh,
parallel build — nothing here reads from or writes to `../scripts`,
`../webapp`, `../ontology`, etc. Treat this as Phase 1 of the migration
described in-conversation: scaffold the framework/domain-pack split and move
today's tax pipeline onto it with the same behavior, before any further
generalization (a second domain pack) is attempted.

The tax domain pack runs on **ontology v5** (`New_tax_ontology_v5.owl`, 164
classes) — `domain_packs/tax/ontology/schema.json` is generated from it, not
from v3/v4. See the note at the top of `domain_packs/tax/config.py` for the
one behavioral consequence (v5 doesn't declare `SubClause`/`AmendingAct`/
`DefinedTerm` as formal classes — the pipeline still mints nodes under those
labels, same as the project's existing `Chunk`/`AuditRun` precedent).

## Ingestion UI

`framework/webapp/app.py` now has a full browser UI (`framework/webapp/static/index.html`)
with three tabs, on top of the graph/ask view from Phase 1:

- **Knowledge Graph & Ask** — the original pan/zoom/citation-search graph
  view plus a new **label filter** panel (checkboxes, top-left) to show/hide
  node types in the currently loaded view without re-querying. Ask responses
  now show, per cited chunk: the chunk text, its **document details**
  (document name/source type/authority level), its own citation, the
  **Cypher query** used as the relationship path, and any KG paths traversed.
- **Ingest Documents** — pick a file type (Income Tax Act PDF, Case Law PDF,
  or a Generic Document PDF/TXT/MD), upload, convert to Markdown (with a
  preview), fill in document metadata, and run ingestion. Extraction is
  automatically sized to the selected LLM's context window — a document that
  fits is sent as ONE extraction call; a longer one is split into as few
  token-budgeted calls as needed (see `domain_packs/tax/converters/extract_generic.py`'s
  `model_chunk_budget()`/`group_chunks_for_llm()` for the "generic" file
  type; the Act and case-law converters have their own established batching).
  Ingestion runs as a background job with a live log — the button returns
  immediately with a `job_id` you can poll (or just watch in the UI).
- **Validate Graph** — node/relationship counts, sample structural-relationship
  checks, an orphan-node check, and a direct citation lookup, all in one view
  (`framework/core/ingest/validate.py`, shared with `framework.cli.query_graph`).
- **Clear All Data** (top-right, every tab) — deletes every node/relationship
  in the connected Neo4j database and flushes the response cache, for a fresh
  ingestion start. Requires typing `DELETE ALL DATA` to confirm.

New file-type dispatch (requirement: "based on the file type, use the
appropriate code"):

| File type | Converter | Extractor |
|---|---|---|
| Income Tax Act (PDF) | `converters/pdf_to_md_act.py` | `converters/extract_entities.py` |
| Case Law (PDF) | `converters/pdf_to_md_caselaw.py` | `converters/extract_case_law.py` |
| Generic Document (PDF/TXT/MD) | `converters/pdf_to_md_generic.py` | `converters/extract_generic.py` |

`.docx` is explicitly not supported yet (would need a new dependency not in
`requirements.txt`) — convert to PDF or plain text first.

## Observability, model config, and themes

- **Langfuse tracing** — every LLM call anywhere in the app (ingestion
  extraction *and* the `/api/ask` retrieval pipeline) is logged to Langfuse
  as a "generation" (model, prompt, completion, token usage, latency),
  wired once in `framework/core/infra/llm_client.py:complete()` rather than
  at each of the 6 call sites — no-op automatically if `LANGFUSE_PUBLIC_KEY`/
  `LANGFUSE_SECRET_KEY` aren't set in `.env`. Each ingestion run (Act/case-law/
  generic) groups its LLM calls under one Langfuse trace per run;
  `/api/ask` groups intent detection, NL→Cypher, and answer generation under
  one trace per question. **Important:** `requirements.txt` pins
  `langfuse>=2.50.0,<3.0.0` — Langfuse v3/v4 removed the `.trace()`/`.span()`/
  `.generation()` methods this code (and the pre-existing `/api/ask` wiring)
  depends on, so an unpinned install would have silently produced zero traces.
- **Model name from `.env`, live at runtime** — `USE_MODEL` picks the
  provider (CLAUDE/GEMINI/GPT); `CLAUDE_MODEL_NAME`/`GEMINI_MODEL_NAME`/
  `GPT_MODEL_NAME` optionally override the exact model string per provider
  (falls back to `llm_client.MODEL_NAME_DEFAULTS` if unset). All of this is
  re-read from `.env` on **every** `complete()` call
  (`llm_client.refresh_model_config()`), so editing `.env` while the app or
  an ingestion run is already going — including switching provider — takes
  effect on the very next LLM call, no restart needed.
- **Theme switcher** — top-right of every tab: Dark / White / Beige / System
  (System follows the OS light/dark preference live). Every UI color is a
  CSS variable (`framework/webapp/static/index.html`'s `:root`/`[data-theme=...]`
  blocks); the choice persists in the browser's `localStorage` across
  reloads. The knowledge-graph view (vis-network, canvas-rendered, doesn't
  inherit CSS) re-reads the active theme's colors on every switch.
- **Separate LLM model for ingestion vs. retrieval** — `USE_MODEL_INGESTION`
  and `USE_MODEL_RETRIEVAL` in `.env` optionally override plain `USE_MODEL`
  for just that purpose (e.g. a cheap/fast model for bulk document
  extraction, a stronger one for user-facing `/api/ask` answers). Either can
  be left blank to fall back to `USE_MODEL`. Same live `.env` reload as
  everything else — `llm_client.complete(..., purpose="ingestion"|"retrieval")`
  re-resolves the provider/model on every call.
- **Parent-child context expansion in retrieval** — `/api/ask` already
  matches small, precise Chunks (Section/SubSection/Clause-level) for
  citation accuracy, but now also fetches each matched chunk's full parent
  Section (or `GenericDocument`, for non-Act sources) via the graph's
  `PART_OF` hierarchy (`GraphRAGRetriever.get_parent_section_context()`) and
  includes it as extra context for the answer LLM — comprehensive
  understanding without losing the precise citation. Shown in the UI as an
  expandable "Full parent section" block under each cited chunk.
- **Page-number citation** — Chunks now carry a `page_number` property when
  the source PDF conversion can determine it: the generic-document PDF path
  (`pdf_to_md_generic.py` + `extract_generic.py`) counts Poppler's
  `pdftotext -layout` page-separator characters that survive into the `.md`;
  the Act PDF path (`pdf_to_md_act.py` + `chunking.py` + `extract_entities.py`)
  writes a `<name>.pagemap.json` sidecar of page-boundary markers alongside
  the `.md` during conversion, which `extract_entities.py` consumes to tag
  each `Chunk`'s `line_start` with its source page. Surfaced in `/api/ask`
  responses as `source_chunks[].document.page` and shown in the UI as
  "p. N" next to each cited chunk. Case-law PDFs don't carry page numbers
  yet (`pdf_to_md_caselaw.py` discards page boundaries during header/footer
  stripping — out of scope for this pass).
- **Configurable response structure** — a "⚙ Response Format" button (top
  bar, next to the theme switcher) lets you edit what sections `/api/ask`
  answers are organized into (default: Intent, Direct Answer, Citations),
  plus free-text extra instructions and a tone hint. Saved to
  `response_config.json` at the repo root via `GET`/`POST
  /api/response-config`; re-read fresh on every question
  (`response_config.load_response_config()`), same hot-reload pattern as
  `.env`. Regardless of the configured structure, the answer always states
  the detected intent and, for every citation, the document name, the
  precise structural citation, and the page number when available.

## Evaluation

The gold-set evaluation harness and the release gates (frontier baseline compared with the Tax SLM) live in `eval/`. See [`eval/README.md`](eval/README.md).

## Layout

```
tax_kg_poc_revised/
  framework/                      # domain-agnostic — never imports domain_packs directly
    contracts/                    # DomainConfig, PromptPack, DocumentParser — the interfaces a
                                   # domain pack implements/supplies
    core/
      infra/                      # neo4j_client, audit_log, step_logger, llm_client
      convert/                    # poppler_utils (shared PDF-tooling helper)
      ingest/                     # graph_loader (was 03_build_graph.py), embedder (was
                                   # 05_embed_chunks.py), entity_resolution — all parameterized
                                   # by a DomainConfig instead of hardcoded tax constants
      retrieve/                   # graphrag (was webapp/retrieval.py), reranker, grounding,
                                   # cache, observability, intent, pii — same generalization
    webapp/                       # Flask app (was webapp/app.py), loads one domain pack at
                                   # startup via FRAMEWORK_DOMAIN / --domain
    cli/                          # ingest_graph / embed_chunks / query_graph — generic,
                                   # --domain-parameterized replacements for the old numbered
                                   # scripts 03/05/04
  domain_packs/
    tax/
      config.py                   # DomainConfig instance — citation regex, structural labels,
                                   # chunk metadata schema, PII rules, all pulled out of the
                                   # original tax_kg_poc modules (see comments in this file)
      pack.py                     # what framework/cli/pack_loader.py imports by name
      prompts/tax_prompts.py      # PromptPack instance — the exact prompt text from
                                   # webapp/prompts.py's nl2cypher/answer_generation/intent_detection
      ontology/schema.json        # copied from tax_kg_poc/ontology/schema.json
      converters/                 # ported from tax_kg_poc/scripts, import paths updated:
        chunking.py                  #   (unchanged — self-contained)
        pdf_to_md_act.py             #   was 09_pdf_to_md_actdocs.py
        pdf_to_md_caselaw.py         #   was 10_pdf_to_md_caselaw.py
        extract_entities.py          #   was 02_extract_entities.py
        extract_case_law.py          #   was 11_extract_case_law.py
  requirements.txt
  docker-compose.yml               # same Neo4j + Redis services as tax_kg_poc's
  .env.example                     # copy to .env and fill in — see comments in that file
```

## How the pieces fit together

A domain pack is "just" a `DomainConfig` + a `PromptPack` + some converter
scripts. `framework/cli/pack_loader.py` is the *only* place the framework
imports a domain pack, and it does so by name (`importlib.import_module`),
never a hardcoded `import domain_packs.tax...` anywhere else in
`framework/`. That's what makes adding a second domain pack (telecom,
retail, refinery, ...) a matter of writing a new `domain_packs/<name>/`
folder — nothing in `framework/` needs to change.

`framework/core/ingest/graph_loader.py` and `embedder.py` are the *same
logic* as `tax_kg_poc/scripts/03_build_graph.py` and `05_embed_chunks.py`
line-for-line, with the two genuinely domain-specific constants
(`STRUCTURAL_LABELS`, the Chunk-node metadata key allowlist) now read from
`DomainConfig` instead of being module-level constants. Same story for
`framework/core/retrieve/graphrag.py` vs. the old `webapp/retrieval.py` (the
citation regex and intent-retrieval policy now come from `DomainConfig`) and
`framework/core/retrieve/pii.py` vs. `webapp/pii.py` (the always-masked
fields and PAN/Aadhaar/email/phone regex detectors now come from
`DomainConfig` too, since those are India/tax-specific, not universal).

## Running it

All commands below assume you've `cd`'d into `tax_kg_poc_revised/` first
(so `framework` and `domain_packs` resolve as top-level packages) and have
a venv with `pip install -r requirements.txt` done, `.env` filled in from
`.env.example`, and `docker compose up -d` running.

```bash
cd tax_kg_poc_revised
python -m venv venv && venv\Scripts\activate   # or source venv/bin/activate on Linux/macOS
pip install -r requirements.txt
copy .env.example .env   # then fill in ANTHROPIC_API_KEY etc.
docker compose up -d
```

Mapping from the old numbered scripts to this structure:

| Old (`tax_kg_poc/scripts/...`)      | New                                                              |
|--------------------------------------|-------------------------------------------------------------------|
| `09_pdf_to_md_actdocs.py`            | `python domain_packs/tax/converters/pdf_to_md_act.py ...` (same args) |
| `10_pdf_to_md_caselaw.py`            | `python domain_packs/tax/converters/pdf_to_md_caselaw.py ...` (same args) |
| `02_extract_entities.py`             | `python domain_packs/tax/converters/extract_entities.py ...` (same args) |
| `11_extract_case_law.py`             | `python domain_packs/tax/converters/extract_case_law.py ...` (same args) |
| `03_build_graph.py`                  | `python -m framework.cli.ingest_graph --domain tax --in ... --mode base` |
| `05_embed_chunks.py`                 | `python -m framework.cli.embed_chunks --domain tax --chunks ...` |
| `04_query_graph.py`                  | `python -m framework.cli.query_graph --domain tax` |
| `webapp/app.py` (`python app.py`)    | `python -m framework.webapp.app --domain tax` (from `tax_kg_poc_revised/`) |

The four converter scripts keep their original multi-argument CLIs
unchanged — they're inherently domain-specific (an Act PDF converter and a
telecom runbook converter would never share a sensible generic CLI), so
Phase 1 relocates them rather than forcing them through a generic command.
The three graph-stage commands (`ingest_graph` / `embed_chunks` /
`query_graph`) operate on the *already-produced* entities/relationships/
chunks JSON regardless of domain, so a generic, `--domain`-parameterized CLI
genuinely applies there.

## What's faithfully ported vs. what's new

- **Byte-for-byte logic, just relocated + parameterized:** `neo4j_client.py`,
  `audit_log.py`, `step_logger.py`, `llm_client.py`, `poppler_utils.py`,
  `entity_resolution.py`, `graph_loader.py` (was `03_build_graph.py`),
  `embedder.py` (was `05_embed_chunks.py`), `graphrag.py` (was
  `webapp/retrieval.py`), `reranker.py`, `grounding.py`, `cache.py`,
  `observability.py`, `intent.py`, `pii.py`, `webapp/app.py`, `chunking.py`,
  `pdf_to_md_act.py` (was `09_...`), `pdf_to_md_caselaw.py` (was `10_...`),
  `extract_entities.py` (was `02_...`), `extract_case_law.py` (was `11_...`).
  All prompt *text* (`nl2cypher` v1–v4, `answer_generation` v1–v2,
  `intent_detection` v1) is copied verbatim into
  `domain_packs/tax/prompts/tax_prompts.py`.
- **New:** the contracts (`DomainConfig`/`PromptPack`/`DocumentParser`), the
  `--domain`-parameterized CLI (`framework/cli/*`), `pack_loader.py`, and
  `domain_packs/tax/config.py`/`pack.py` (the glue that supplies the
  framework with tax's specific rules).

## Known gaps / Phase 2 follow-ups

- `07_query_assessed_to_income_tax.py` and `08_diagnose_section_55.py` were
  not ported (they weren't part of the core pipeline in `tax_kg_poc` either
  — see that repo's `EXECUTION_STRUCTURE.md`).
- `06_pii_scan.py` (the standalone PII verification CLI) was not ported yet
  — `framework/core/retrieve/pii.py`'s `PiiMasker.scan_for_pii()` has the
  same detectors available; a thin CLI wrapper is a small follow-up.
- This was built and `py_compile`-checked in an environment with no live
  Neo4j/Redis server — it has not been run end-to-end against a real
  database. Run the pipeline table above in order against your own
  `docker compose up -d` before trusting it, the same disclosure
  `tax_kg_poc/EXECUTION_STRUCTURE.md` made for the Neo4j migration.
  Update (ingestion-UI pass): Flask and the `redis`/`neo4j` Python client
  libraries were installed in-session this round and genuinely exercised via
  a `app.test_client()` route battery covering `/api/file-types`,
  `/api/upload`, `/api/convert`, `/api/ingest/start`+`/api/ingest/status`,
  `/api/validate`, `/api/graph/clear`, and `/api/ask` — confirming correct
  status codes, request validation, and graceful `{"error": ...}` JSON on
  failure (one real bug, an unhandled exception in `/api/ask`, was found and
  fixed this way). What remains untested end-to-end in this sandbox: a live
  Neo4j *server* (client library only), a live Redis *server* (client
  library only, `REDIS_AVAILABLE` correctly resolves to `False` without
  one), `sentence-transformers` (install attempted, timed out — routes that
  need it were tested via a `sys.modules` stub), and outbound LLM API
  network access (blocked by an SSL restriction in this sandbox). None of
  these are code defects; each failure mode was verified to degrade
  gracefully rather than crash.
- The `DocumentParser` contract (`framework/contracts/document_parser.py`)
  is the *target* shape for a domain's converter, matching what
  `chunking.py`/`extract_entities.py` already produce today. They aren't
  formally wrapped in a `DocumentParser`-conforming class yet — that
  formalization is worth doing once a second domain pack's converter needs
  the polymorphism, not before.
- No second domain pack exists yet to prove the abstraction end-to-end —
  this is Phase 1 (scaffold + zero-behavior-change move), not Phase 3 (a
  second thin domain pack as proof), from the original migration plan.
