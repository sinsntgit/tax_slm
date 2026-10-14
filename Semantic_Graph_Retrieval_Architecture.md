# Semantic Graph Retrieval Architecture

**Status:** implemented, additive, opt-in via `retrieval_mode="semantic_graph"`.
**Date:** 2026-08-29.
**Supersedes:** `Vectorless_GraphRAG_Design.md` (the full-text-search-based "vectorless"
plan — `fulltext_search.py` was never built; this doc reflects what was actually
shipped instead). Complements, does not replace, `Graph_First_Retrieval_Design.md`
(the case-law stance-balance work, which is orthogonal and still active in all
retrieval modes).

## 1. What problem this solves

The original hybrid pipeline finds candidate text via ANN (approximate nearest
neighbour) search over pre-chunked, pre-embedded `Chunk.embedding` vectors stored
at ingestion time. That means: (a) every chunk's raw text is duplicated into the
graph as a `Chunk` node, and (b) the text an answer is built from is whatever was
sliced into that chunk at ingestion time — not necessarily the cleanest unit of
meaning for a given question.

The user asked for a design where the graph is used to find the *right entity*
first, and the actual answer text is read live from the source document, so:

- no need to store/maintain large `Chunk.embedding` ANN indexes as a primary
  retrieval mechanism,
- answer text always reflects the full, current converted document rather than a
  possibly-stale or awkwardly-cut-up stored chunk,
- entry-point *discovery* still needs a fast way to jump straight to the right
  node(s) for a natural-language question — hence one small vector index per
  entity type, over a short discriminating summary, not over full text.

## 1a. No vectors in Neo4j (2026-08-29, third pass)

**Explicit requirement, now enforced everywhere in this feature**: Neo4j
holds ONLY graph structure, metadata, and short text summaries - no
embedding vectors, anywhere, ever. This applies to all four semantic_graph
entry-point indexes (CaseLaw, Section, Circular, TaxTreaty). It does **not**
apply to the original Hybrid/Simple RAG pipeline's `Chunk.embedding` /
`chunk_embedding_index` (a separate, pre-existing mechanism, explicitly kept
out of scope for this change).

Embeddings for entry-point search now live in a small local flat file per
index (`framework/core/retrieve/local_vector_store.py`) - one `.npz` per
index name, under `VECTOR_STORE_DIR` (env override) or
`<repo_root>/data/vector_store/` by default. It's a brute-force numpy
cosine-similarity scan, not an ANN library - deliberate, since this
project's entry-point corpora are small (tens to low hundreds of nodes per
type), and it avoids adding a new heavyweight dependency (FAISS/Chroma/etc.)
for a POC.

`semantic_entry_points.py`'s engine now does exactly two things with Neo4j:
(1) reads each node's text properties to embed (`embed_nodes()` - read-only,
never writes a vector), and (2) ONE follow-up metadata lookup, by the
db_ids the LOCAL search already picked (`search_entry_points()` -
`MATCH (n) WHERE elementId(n) IN $db_ids`) - Neo4j never sees or stores a
vector at any point.

This also let `circular_entry_points.py` drop its earlier 3-separate-Neo4j-
index-per-subtype-label design (a workaround for Neo4j's single-label-per-
vector-index limitation) - a local file doesn't care about Neo4j labels, so
all 3 Circular subtypes (CBDTCircular/ExplanatoryCircular/ExtensionCircular)
now share ONE local store, with no merge-then-resort step needed at search
time. `case_law_entry_points.py` (previously a separate direct-Neo4j-vector-
index implementation, deliberately left alone when this pattern was first
generalized to Section) was also migrated onto the shared engine - required
now that storing any vector on a Neo4j property is prohibited outright. That
migration surfaced and fixed a latent bug: the old Cypher RETURN clause
never included `sourceMdPath`, so case-law hits were always silently
un-hydratable under semantic_graph mode - masked in earlier testing because
the relevant mocked test mocked `search_case_law_entry_points()` directly
rather than exercising its real Cypher.

## 2. Architecture in one paragraph

`retrieval_mode="semantic_graph"` keeps every graph-heavy part of the existing
hybrid pipeline unchanged — NL2Cypher, citation lookups, `PART_OF`/graph
traversal, the case-law stance-balanced path, parent-context expansion, the
BGE reranker. The ONE thing it replaces is the generic vector-fallback call
site (`vector_search_chunks()`, ANN over `Chunk.embedding`). In its place,
`semantic_entry_point_search()` runs a small per-entity-type embedding search
(currently: `CaseLaw.retrievalSummary`/`Section.hasText`+heading+citation) to
find candidate nodes, then reads the actual answer text live from the node's
`sourceMdPath` (the converted `.md` file), either the whole file or a precise
line range, and returns it in the exact same `{"chunk_id","text","similarity"}`
shape the rest of the pipeline (reranker, `_build_context_parts()`, citation
rendering) already expects — so nothing downstream needed to change.

## 3. Two-tier hydration split

Not every entity type is hydrated the same way, because they're extracted
differently:

- **Whole-file hydration** (CaseLaw, and — after 2026-08-29 — Circular,
  TaxTreaty): these are extracted via a single LLM call reading the whole
  document, and their stored text properties (holding, reasoning, `hasText`)
  are LLM-composed/paraphrased summaries, not verbatim positional slices. There
  is no valid `line_start:line_end` for them, so hydration reads the *entire*
  `.md` file. This is cheap: case-law/circular/treaty `.md` files run roughly
  9 KB–550 KB.
- **Line-range hydration** (StatutoryProvision / `Section`): extracted via
  `chunking.py`'s deterministic parser, which already tracks precise
  `line_start`/`line_end` per Section/SubSection. Hydration reads just that
  line range from the Act's `.md` file (the Act files run up to ~3 MB — too
  big to re-read whole-file on every query).

Both paths are implemented in one shared function,
`framework/core/retrieve/md_hydration.py::hydrate_text()`
(`line_start`/`line_end` both `None` → whole-file path; both set → line-range
path), fed by `hydrate_entity()` which reads `sourceMdPath`/`line_start`/
`line_end` straight off an entity's stored properties.

## 4. Component map

| Component | File | Role |
|---|---|---|
| `sourceMdPath` schema property | `domain_packs/tax/ontology/schema.json` | Domain: CaseLaw, StatutoryProvision, TaxTreaty, Circular. Absolute path to the converted `.md` file an entity was extracted from. |
| Stamping — CaseLaw | `domain_packs/tax/converters/extract_case_law.py` | `build_graph_from_extraction(..., source_md_path=...)`, stamped via `e["properties"].setdefault("sourceMdPath", ...)`. |
| Stamping — StatutoryProvision/Section | `domain_packs/tax/converters/extract_entities.py` | `doc_meta["source_md_path"]` → `attach_pipeline_metadata()` → `props.setdefault("sourceMdPath", ...)`. |
| Stamping — Circular | `domain_packs/tax/converters/extract_circular.py` | `make_doc_meta()` now sets `"source_md_path": str(md_path.resolve())`; flows through the same shared `extract_entities.attach_pipeline_metadata()` call already used here. |
| Stamping — TaxTreaty | `domain_packs/tax/converters/extract_treaty.py` | Same pattern as Circular, via the same shared `attach_pipeline_metadata()`. |
| Hydration engine | `framework/core/retrieve/md_hydration.py` | `hydrate_text(source_md_path, line_start, line_end)` → `(text, note)`; `hydrate_entity(entity_props)` convenience wrapper. Returns `("", note)` on any failure (missing path, missing file, out-of-range lines) rather than raising — callers skip that candidate and log the note. |
| Local vector store | `framework/core/retrieve/local_vector_store.py` | `save()`/`load()`/`merge_and_save()`/`search()` over a flat `.npz` file per index, under `VECTOR_STORE_DIR` (env override) or `<repo_root>/data/vector_store/`. Brute-force numpy cosine similarity — no Neo4j involvement at all. See §1a. |
| Generic entry-point search engine | `framework/core/retrieve/semantic_entry_points.py` | Reusable: `setup_entry_point_index()` (no-op, kept for call-site compatibility), `embed_nodes()` (reads Neo4j text, writes to the local store), `search_entry_points()` (searches the local store, then ONE Neo4j metadata lookup by db_id). Used by all four entry-point modules below, including `case_law_entry_points.py` (migrated onto this engine in the no-Neo4j-vectors pass — see §1a). |
| Section entry-point config | `framework/core/retrieve/section_entry_points.py` | Configures the generic engine for `Section` (embeds `full_citation` + `marginal_heading` + `hasText`, capped at 1500 chars). |
| Section embed CLI | `framework/cli/embed_sections.py` | `python -m framework.cli.embed_sections` — run after (re)ingesting the Act. |
| Case-law entry-point config | `framework/core/retrieve/case_law_entry_points.py` | Configures the generic engine for `CaseLaw` (embeds `retrievalSummary` + `retrievalKeywords`). Also used directly by `category_retrieval.py`'s stance-balanced path. |
| Case-law embed CLI | `framework/cli/embed_case_law.py` | `python -m framework.cli.embed_case_law` — run after (re)ingesting case law. |
| Circular entry-point config | `framework/core/retrieve/circular_entry_points.py` | Configures the generic engine for all 3 Circular subtype labels sharing ONE local store (`circular_retrieval_index`) — no per-label indexes/merge-sort needed now that Neo4j's single-label-per-vector-index limitation no longer applies. |
| Circular embed CLI | `framework/cli/embed_circulars.py` | `python -m framework.cli.embed_circulars` — run after (re)ingesting Circulars. |
| TaxTreaty entry-point config | `framework/core/retrieve/treaty_entry_points.py` | Single-label config (`TaxTreaty`), same pattern as `section_entry_points.py`. |
| TaxTreaty embed CLI | `framework/cli/embed_treaties.py` | `python -m framework.cli.embed_treaties` — run after (re)ingesting a treaty. |
| Retrieval wiring | `framework/core/retrieve/graphrag.py` | New `semantic_entry_point_search()` method (queries case-law + section entry points, hydrates each via `md_hydration`, skips any with empty text); `retrieve()` branches on `retrieval_mode == "semantic_graph"` to call it instead of `vector_search_chunks()`; citation/graph/`PART_OF` traversal paths unchanged. |
| Synthetic chunk_id resolution | `framework/core/retrieve/graphrag.py` | `get_document_details()` / `get_entity_labels_for_chunks()` extended to detect `mdhydrate_<elementId>`-prefixed ids (used for hydrated results, since they have no real `Chunk` node) and resolve them via direct `elementId()` lookup, merged with normal `Chunk`-based results. Without this, citations would show raw ids and category filtering would silently discard every semantic_graph candidate. |
| Category-level wiring | `framework/core/retrieve/category_retrieval.py` | Generic per-category vector-fallback call site branches the same way: `semantic_entry_point_search()` under `"semantic_graph"`, `vector_search_chunks()` otherwise. Same `{"chunk_id","text","similarity"}` shape either way, so entity-label filtering, reranking, and `format_category_context()` are unchanged. |
| Mode validation | `framework/webapp/app.py` | Three hardcoded `retrieval_mode not in (...)` gates extended to accept `"semantic_graph"` alongside `"hybrid"`/`"vector_only"`. |

## 5. What still uses the old `Chunk`-embedding ANN path

- `retrieval_mode="hybrid"` (default) and `"vector_only"` — unchanged, both
  still call `vector_search_chunks()`.
- `retrieval_mode="semantic_graph"` still uses `Chunk` nodes/embeddings
  nowhere in its own path, but does NOT delete or stop maintaining them —
  this is purely an additive new mode, existing modes and existing chunk
  embedding infra are untouched, per the original "additive new mode"
  rollout decision.

## 6. Operational steps to use semantic_graph mode

1. Re-run ingestion (or the relevant backfill) so `sourceMdPath` gets stamped
   on entities — already-ingested-before-2026-08-28 entities will not have it
   and will simply be skipped (empty-text, logged) by `hydrate_entity()`
   rather than erroring.
2. Build the entry-point indexes:
   - `python -m framework.cli.embed_case_law` (case law)
   - `python -m framework.cli.embed_sections` (Act/StatutoryProvision)
   - `python -m framework.cli.embed_circulars` (Circulars, all 3 subtypes)
   - `python -m framework.cli.embed_treaties` (TaxTreaty)
3. Call `/api/ask` (or the equivalent retrieval entry point) with
   `retrieval_mode="semantic_graph"`.

## 7. Known follow-up scope

**Done as of 2026-08-29 (second pass):**

- **Circular entry-point index** — `framework/core/retrieve/circular_entry_points.py`
  + `framework/cli/embed_circulars.py`. Circulars are ingested under THREE
  distinct Neo4j labels by subtype (`CBDTCircular`/`ExplanatoryCircular`/
  `ExtensionCircular` — see `extract_circular.py`'s `_SUBTYPE_TO_CLASS`), and
  `graph_loader.py` stamps only one label per node (no shared `Circular`
  ancestor label at ingestion time), so this builds ONE small vector index
  per subtype label (`circular_retrieval_index_<label>`) and merges +
  re-sorts results by similarity at search time, tolerating a not-yet-built
  index for a subtype with zero ingested circulars.
- **TaxTreaty entry-point index** — `framework/core/retrieve/treaty_entry_points.py`
  + `framework/cli/embed_treaties.py`. `TaxTreaty` is a single Neo4j label,
  so this follows `section_entry_points.py`'s simpler single-index pattern
  directly.
- **`graphrag.py` wiring** — `semantic_entry_point_search()` now queries all
  FOUR sources (case law, section, circular, treaty), each independently
  try/except-wrapped so one source failing (e.g. an index not yet built)
  never fails the others.
- **Ask screen UI** — a third "Semantic Graph" button was added to the
  `#ask-retrieval-mode` segmented control alongside "KG + RAG"/"Simple RAG"
  (`framework/webapp/static/index.html`); the control is fully data-driven
  off `data-mode`, so this needed no other JS changes. The Evaluation
  Feedback / chat-history mode label was also extended to render
  "Semantic Graph" instead of a blank label for these records.

**Done as of 2026-08-29 (third pass):**

- **`backfill_source_md_path.py`** (`domain_packs/tax/converters/`) — pure
  graph read/write, no LLM calls. Backfills `sourceMdPath` onto CaseLaw/
  Circular-subtype/TaxTreaty nodes via their existing `source_document`
  (bare filename) property, and onto Section nodes via a `document_id` ->
  Act-.md-filename resolution (explicit `--act-map`, or a year-substring
  auto-guess — verified correct against this project's actual two Acts).
  `--dry-run` first, same convention as `backfill_case_law_stance.py`. See
  `Testing_Semantic_Graph_Retrieval.md` for exact commands.
- **`Testing_Semantic_Graph_Retrieval.md`** — new companion testing guide
  (mirrors `Testing_Graph_First_Retrieval.md`'s structure), covering exactly
  which of the current corpus needs the backfill vs. is new-ingestion-ready,
  the embed CLI commands, and a direct (LLM-free) retrieval smoke test.

**Still open:**

- Circulars and TaxTreaty PDFs sitting in `data/rawdata/` have not been
  converted/ingested at all yet (no `.md` files, no graph entities) as of
  this writing — their FIRST ingestion will already include `sourceMdPath`
  (no backfill needed for them), but the backfill script's Circular/TaxTreaty
  branches are therefore currently untested against real data (only unit-
  tested against mocks) — worth a quick real check once at least one
  Circular/Treaty is ingested.

**Done as of 2026-08-29 (fourth pass) — real-request diagnosis + fixes:**

A live semantic_graph question surfaced three issues in one server log, all
now fixed:

- **Oversized Langfuse trace spans dropped.** `app.py`'s "bge_rerank"/
  "citation_grounding" spans passed the FULL candidate/top chunk dicts
  (including un-capped `"text"`) as trace input/output. Under semantic_graph
  mode a chunk's `"text"` can be an entire whole-file `md_hydration.py` read
  (up to `DEFAULT_WHOLE_FILE_MAX_CHARS` = 4M chars), and 30-39 such
  candidates summed to 95MB+ — well past Langfuse's per-item size ceiling,
  which silently dropped the whole span. Fixed with
  `observability.summarize_chunks_for_trace()`: keeps every key except
  `"text"` as-is, replaces it with a small preview + the original length.
  Purely a trace-payload change — does NOT touch what the LLM prompt sees
  (already separately capped at 1500 chars/chunk in `_build_context_parts`)
  or what the reranker scores on (`RERANK_TEXT_MAX_CHARS`, already 1500
  chars) — those were never the source of the size problem.
- **No console-visible record of which `.md` files were scanned.** The raw
  server log previously said nothing about file names — only the per-index
  try/except warnings. `graphrag.py`'s `semantic_entry_point_search()` now
  logs one `[graphrag] md file scan: READ/SKIP '<name>' - <path> (...)` line
  per hydration attempt plus a `md file scan SUMMARY` line, independent of
  (and in addition to) the UI evidence trail already built in a prior pass.
- **Circular/Treaty entry-point search failing with
  `db.index.vector.queryNodes: no such vector schema index`.** NOT a bug in
  current code — confirmed by re-reading `circular_entry_points.py`/
  `treaty_entry_points.py`/`semantic_entry_points.py`: none of them call
  Neo4j's native vector index anymore since the no-vectors-in-Neo4j
  migration (§1a). The error's 3-separate-per-subtype message format only
  exists in the PRE-migration code. Root cause: `app.py` runs with
  `use_reloader=False` (deliberate — see the comment at its `app.run()`
  call), so a long-lived Flask process keeps executing whatever code was in
  memory at process start and never picks up later file edits. Not
  something this codebase can fix for itself short of enabling the
  reloader (a deliberate tradeoff, left as-is) — requires an actual process
  restart after any retrieval-path code change.

Also new this pass, unrelated to the above but requested alongside it:

- **Data Explorer** (`framework/core/retrieve/data_explorer.py` +
  `/api/data-explorer/*` routes in `app.py` + a new "Data Explorer" screen
  in `index.html`) — a read-only browsing/search panel over what's actually
  stored: the "Knowledge Index" tab lists what's IN each of the 4 local
  vector stores (joined against Neo4j for display fields), "Metadata" lists
  any Neo4j label's full properties, "Chunks" lists `:Chunk` nodes — all
  three searchable across every property and paginated, with a "View full
  data" row-detail modal showing every property un-truncated. Distinct from
  KG Explorer (graph visualization) - this is a flat table/search view.

## 8. Verification performed

- `python3 -m py_compile` clean on `graphrag.py`, `category_retrieval.py`,
  `extract_circular.py`, `extract_treaty.py`.
- Mocked unit tests (sentence_transformers stubbed out, no live Neo4j/LLM
  needed):
  - `md_hydration.hydrate_text()`: whole-file read, line-range read, empty
    path, missing file, out-of-range lines — all correct.
  - `section_entry_points._embedding_text()`: field composition + 1500-char
    cap verified.
  - `graphrag.GraphRAGRetriever.semantic_entry_point_search()`: whole-file
    (case law) and line-range (Section) hydration both verified end-to-end
    against mocked entry-point results; empty-`sourceMdPath` entries
    correctly skipped rather than raising.
  - `graphrag.GraphRAGRetriever.get_document_details()` and
    `get_entity_labels_for_chunks()`: verified against a mixed list of real
    `Chunk.chunk_id`s and synthetic `mdhydrate_<id>`s in a single call — both
    resolve correctly.
  - `category_retrieval.run_category_retrieval()`: verified the generic
    vector-fallback branch calls `semantic_entry_point_search()` under
    `retrieval_mode="semantic_graph"` and `vector_search_chunks()` under
    `"hybrid"`, with the resulting candidates flowing correctly through
    entity-label filtering to a `"vector_fallback"` result either way.
