# Modification Guide: Neo4j Settings, Schema, Chunking, Ingestion

Reference for the four most common places this project needs to be changed. Each section says exactly which file to edit, what to change, and what happens downstream.

---

## 1. Neo4j Settings

Two layers: the **database itself** (either Neo4j Desktop or `docker-compose.yml` — pick one, they're interchangeable) and the **application's connection to it** (`.env`, read by `framework/core/infra/neo4j_client.py`). The app only ever talks to Neo4j over the bolt protocol via `.env`'s `NEO4J_URI` — it has no idea whether that URI points at a Docker container or a Neo4j Desktop instance, so switching between them is purely a `.env` change, no code change.

### 1a-i. Using Neo4j Desktop instead of Docker (recommended for local/KPMG dev machines without Docker)

Neo4j Desktop is the standalone GUI app (download from neo4j.com/download) — it manages local database instances for you, with no `docker-compose.yml` involved at all. Steps:

1. **Install Neo4j Desktop** and open it.
2. **Create a new Project**, then inside it **Add → Local DBMS**. Pick a name (e.g. `tax_kg`), a password, and a Neo4j version — use **5.24** or another 5.x release to match what `docker-compose.yml` uses (`neo4j:5.24-community`), since the app's Cypher (native vector index syntax, `CREATE VECTOR INDEX ...`) requires Neo4j 5.x+.
3. **Install the APOC plugin**: select the DBMS, go to the **Plugins** tab in the right-hand panel, find **APOC**, click **Install**. This is required — it's the same `NEO4J_PLUGINS: '["apoc"]'` dependency the Docker setup installs automatically; without it, some ingestion queries will fail.
4. **Start the DBMS** (the ▶ button). Desktop shows the Bolt port it's listening on — by default this is `bolt://localhost:7687`, but Desktop auto-increments the port if something else (like a Docker Neo4j) is already using 7687, so double-check the actual port shown in the DBMS's **Details** tab.
5. **Update `.env`** to match:
   ```
   NEO4J_URI=bolt://localhost:7687     # <- use the exact port Desktop shows, may not be 7687 if you also run Docker's Neo4j
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=<the password you set when creating the DBMS>
   NEO4J_DATABASE=neo4j
   ```
6. Skip `docker-compose.yml`'s `neo4j` service entirely (leave the `redis` service in Docker still — Desktop only replaces Neo4j, Redis has no Desktop equivalent). If you don't want Docker running at all, run `docker compose up -d redis` instead of the full `docker compose up -d`.

To open the database visually (equivalent of Neo4j Browser at `http://localhost:7474` in the Docker setup): click **Open** on the running DBMS in Desktop, which launches Neo4j Browser pointed at that instance automatically — same Cypher console, same experience, just launched from the desktop app instead of a browser URL.

Everything else in this guide (constraints/indexes auto-created at ingest time, vector index setup, etc. — see 1c and Section 2) behaves identically regardless of which of the two you use, since it's all driven by the same bolt connection.

### 1a-ii. Connection settings (host, port, credentials, timeouts) — applies to both Docker and Desktop

File: `.env` (create from `.env.example` if it doesn't exist yet).

```
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=<your password>
NEO4J_DATABASE=neo4j
NEO4J_CONNECT_TIMEOUT_SECONDS=10
NEO4J_STATEMENT_TIMEOUT_SECONDS=20
```

Change these values and restart the webapp — no code edit needed. `neo4j_client.py` reads them via `os.environ.get(...)` with the values above as fallback defaults if a key is missing entirely.

If you need to point at a different Neo4j instance (e.g. Neo4j Aura, a remote server), just change `NEO4J_URI` to that instance's bolt URL (e.g. `neo4j+s://xxxx.databases.neo4j.io`) and the matching `NEO4J_USER`/`NEO4J_PASSWORD`. `docker-compose.yml` becomes irrelevant in that case (you're not running your own container).

### 1b. The Neo4j container itself (memory, plugins, ports, password for local Docker)

File: `docker-compose.yml` (project root).

```yaml
services:
  neo4j:
    image: neo4j:5.24-community
    environment:
      NEO4J_AUTH: "neo4j/taxkg_neo4j_pw"        # <- change password here too, and match it in .env
      NEO4J_PLUGINS: '["apoc"]'
      NEO4J_server_memory_heap_initial__size: "512m"
      NEO4J_server_memory_heap_max__size: "1G"
    ports:
      - "7474:7474"   # Neo4j Browser (http://localhost:7474)
      - "7687:7687"   # Bolt driver protocol
```

Common changes:
- **More memory** (large graphs will need this): raise `NEO4J_server_memory_heap_max__size` (e.g. `"4G"`) and optionally add `NEO4J_server_memory_pagecache_size: "2G"`.
- **Change the password**: edit `NEO4J_AUTH` here AND `NEO4J_PASSWORD` in `.env` — they must match. If the container already has a data volume from a previous password, you'll need to `docker compose down -v` (wipes data) or use `cypher-shell`/Neo4j Browser to run `ALTER CURRENT USER SET PASSWORD` instead of just changing the env var.
- **Different port** (e.g. 7687 already in use): change the left side of the `ports:` mapping (e.g. `"7688:7687"`) and update `NEO4J_URI` in `.env` to match (`bolt://localhost:7688`).

After editing `docker-compose.yml`: `docker compose down && docker compose up -d`.

(This whole 1b section is Docker-only — if you're on Neo4j Desktop instead, the equivalent settings — memory, plugins, password, port — are all set through Desktop's UI as described in 1a-i, not through this file.)

### 1c. Vector index settings (dimensions, similarity function)

File: `framework/core/infra/neo4j_client.py`, function `ensure_vector_index()`. You normally don't touch this directly — it's called by `framework/core/ingest/embedder.py`'s `setup_vector_index()`, which passes `dimensions=embedding_dimension()` (auto-detected from whatever `EMBEDDING_MODEL_NAME` is configured — see Section 3) and `similarity_function="cosine"`.

If you need a different similarity function (e.g. `"euclidean"`), change the hardcoded `"cosine"` default in `ChunkEmbedder.setup_vector_index()` (`framework/core/ingest/embedder.py`). Note: `ensure_vector_index()` already detects a dimension mismatch and auto-drops/recreates the index — but changing similarity function on an existing index requires the same manual drop, since that check only looks at `vector.dimensions`, not `vector.similarity_function`. Add that check yourself if you expect to change this often.

---

## 2. Neo4j Schema Creation

There's no single "run this to create the schema" step — the schema is **implicit**: it's defined by (a) an ontology `.owl` file that gets converted to `domain_packs/tax/ontology/schema.json`, and (b) whatever labels/properties the extraction code actually emits at ingestion time. Neo4j itself is schema-optional (no CREATE TABLE step) — constraints/indexes are the only real "schema" objects that exist in the database.

### 2a. What schema.json actually is

File: `domain_packs/tax/ontology/schema.json`. This is a machine-readable export of the ontology (classes, object properties, data properties) used as **prompt context** for the LLM extraction step (`domain_packs/tax/converters/extract_entities.py` and friends pass a schema excerpt into the prompt so the LLM knows what entity types/relationships are valid to extract). It is NOT enforced by Neo4j — Neo4j will happily accept a label or relationship type that isn't in this file.

**Important caveat found during this audit**: the source `.owl` file this was generated from (`New_tax_ontology_v5.owl`) lives in the sibling `tax_kg_poc` project tree, not inside `tax_kg_poc_revised`. If you're sharing only `tax_kg_poc_revised`, the ontology source-of-truth isn't included — only its JSON export is. If KPMG needs to regenerate or extend the ontology, you'll either need to also share that `.owl` file (from the original `tax_kg_poc` folder) or hand-edit `schema.json` directly (it's just JSON — classes/properties can be added without the `.owl` roundtrip, since nothing currently regenerates `schema.json` automatically from an `.owl` file inside this repo).

To add a new entity type or relationship to the schema:
1. Add the class/property to `schema.json` directly (or regenerate from an updated `.owl`, if you have the OWL toolchain).
2. Reference it in the relevant converter's LLM extraction prompt (e.g. `domain_packs/tax/converters/extract_entities.py`'s system prompt) so the LLM actually starts extracting it — adding it to `schema.json` alone does nothing unless the prompt tells the LLM to use it.
3. If it's a new **structural** (deterministically-parsed, not LLM-inferred) type, you'd also need to touch `domain_packs/tax/converters/chunking.py` (Section 3 below).

### 2b. What actually gets created in Neo4j (constraints/indexes)

File: `framework/core/ingest/graph_loader.py`. Every time a new label is seen during ingestion, `GraphIngestionEngine.load_entities()` calls:

```python
neo4j_client.ensure_id_constraint(session, label)
```

which runs `CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE` — this is the only enforced "schema" in the database: every node label gets a uniqueness constraint on its `id` property, automatically, the first time that label is ingested. You never need to pre-declare labels.

Other indexes created automatically:
- `Chunk.chunk_id` — property index, via `neo4j_client.ensure_property_index()`.
- `Chunk.embedding` — vector index, via `embedder.py`'s `setup_vector_index()` (see 1c above).

To add a **new** constraint or index (e.g. a property index on `CaseLaw.case_name` for faster lookups), call the existing helpers in `neo4j_client.py` (`ensure_id_constraint`, `ensure_property_index`, `ensure_vector_index`) from wherever makes sense in the ingestion path — most naturally alongside the existing calls in `graph_loader.py` or `embedder.py`.

### 2c. Domain-level schema config (labels the framework treats specially)

File: `domain_packs/tax/config.py`. This is where you tell the framework which labels are "structural" (deterministically parsed, skip fuzzy cross-run entity resolution) vs. everything else, and which properties get stored where:

```python
STRUCTURAL_LABELS = frozenset({
    "Section", "SubSection", "Clause", "SubClause", "Proviso", "Explanation",
    "GenericDocument", "DocumentSection",
})

CHUNK_METADATA_KEYS = (...)        # properties mirrored onto Chunk nodes at graph-load time
EMBED_CHUNK_METADATA_KEYS = (...)  # properties mirrored onto Chunk nodes at embed time

RESOLUTION_KEY_PROPERTIES = {      # which properties define "same entity" for fuzzy matching
    "Section": ["section_no"],
    "CaseLaw": ["case_name"],
    ...
}
```

If you add a new entity type (say, `Circular`) that should be excluded from fuzzy cross-run resolution, add it to `STRUCTURAL_LABELS`. If it should participate in fuzzy resolution, instead add an entry to `RESOLUTION_KEY_PROPERTIES` describing which of its properties identify a duplicate.

---

## 3. Changing the Chunking Approach

There are **three separate chunking strategies**, one per document type, dispatched by `domain_packs/tax/pipeline.py`'s `FILE_TYPES` table:

| file_type | Chunker | Approach |
|---|---|---|
| `act_pdf` | `domain_packs/tax/converters/chunking.py` | Deterministic regex parse: Section → SubSection → Clause → SubClause, with Provisos/Explanations peeled off |
| `caselaw_pdf` | `domain_packs/tax/converters/extract_case_law.py` | One LLM call per judgment (no separate chunker — the whole judgment is one unit) |
| `generic` | `domain_packs/tax/converters/extract_generic.py` | Paragraph-level chunking, sized to the LLM's context window |

### 3a. Changing the Act chunking (chunking.py) — most likely area to modify

This is a **hand-tuned regex parser**, not a generic algorithm — it was built by inspecting the actual formatting of `income_tax_act_1961_v2.md`. If you ingest a differently-formatted Act document, the regexes at the top of the file are the first thing to check:

```python
SECTION_HEADING_RE = re.compile(r'^####\s+Section\s+([0-9]+[A-Z]{0,3})\.\s*(.*)$')
CHAPTER_HEADING_RE  = re.compile(r'^##\s+(?:Chapter|CHAPTER)\s+([IVXLCDM]+...)...')
SCHEDULE_HEADING_RE = re.compile(r'^##\s+THE\s+(.+?)\s+SCHEDULE\s*$', re.IGNORECASE)
SUBSECTION_RE = re.compile(r'^\((\d+[A-Z]?)\)\s?(.*)$')          # (1), (1A), (42A)
CLAUSE_RE     = re.compile(r'^\(([a-z]{1,3})\)\s?(.*)$')          # (a), (aa)
SUB_CLAUSE_RE = re.compile(r'^\(([ivxlcdm]{1,6})\)\s?(.*)$', re.IGNORECASE)  # (i), (ii)
```

If the source markdown uses a different heading marker (e.g. `### SECTION 5.` instead of `#### Section 5.`), edit `SECTION_HEADING_RE` to match. Everything downstream (entity IDs, citations, PART_OF hierarchy) derives from these patterns matching correctly — get this wrong and the whole document collapses into one oversized leaf chunk (you'll see this flagged in `ambiguities` as `"oversized_leaf"`).

To change **chunk size targets** (when a section is considered "too large" and split further):

```python
SIZE_GUIDELINES = {
    "section":     {"min": 100, "target": 500, "max": 1000},
    "subsection":  {"min": 75,  "target": 400, "max": 800},
    "clause":      {"min": 50,  "target": 300, "max": 800},
    ...
}
```

These are in **tokens** (via `tiktoken`'s `cl100k_base` encoding, used only as a sizing heuristic, not for LLM billing). Raise `max` if you want fewer, larger chunks; lower it for more granular chunks. This directly affects retrieval precision (smaller chunks = more precise citation but less surrounding context per chunk).

To change **amendment/footnote parsing** (how `w.e.f.`, Act numbers, insertion/substitution/omission are detected): edit `FOOTNOTE_DEF_RE`, `FOOTNOTE_ACTNO_RE`, `FOOTNOTE_DATE_RE`, `ACTION_TYPE_MAP` near the top of the file. These are tuned to the specific footnote phrasing found in the actual Act PDFs used so far (e.g. `"6.Sub. for "thirty-six" by the Act No. 15 of 2024, w.e.f. 23-7-2024."`) — a differently-worded source (e.g. a different country's tax code, or an Act PDF with amendment notes in a different citation style) will need these regexes adjusted.

To change **how many chunks get batched into one LLM extraction call**: `group_chunks_for_llm(chunks, max_chars=60_000)` at the bottom of the file. Raising `max_chars` means fewer, bigger LLM calls (faster overall, but larger single-call output risk); lowering it means more, smaller calls.

### 3b. Changing generic-document chunking

File: `domain_packs/tax/converters/extract_generic.py` — look for `MODEL_CONTEXT_WINDOWS` and `model_chunk_budget()`. This chunker sizes itself to whichever LLM provider is currently configured for ingestion (Claude/Gemini/GPT have different context windows), rather than using a fixed chunk size like the Act path.

### 3c. Adding a fourth document type / chunking strategy entirely

1. Write a new converter module in `domain_packs/tax/converters/` (a `pdf_to_md_X.py` + an `extract_X.py`, following the existing `act_pdf`/`caselaw_pdf`/`generic` pair pattern).
2. Register it in `domain_packs/tax/pipeline.py`'s `FILE_TYPES` dict (label, accepted extensions, description, default metadata) and in `convert_document()` / `run_extraction()`'s if/elif dispatch.
3. The UI's file-type dropdown (`framework/webapp/static/index.html`, Ingest tab) picks this up automatically from `FILE_TYPES` — no separate UI edit needed.

---

## 4. Changing the Ingestion Approach

"Ingestion" here spans four sequential stages, all orchestrated by `domain_packs/tax/pipeline.py`: **Convert → Extract → Load into Neo4j → Embed**. Each stage is independently swappable.

### 4a. Changing the ingestion MODE (fresh vs. amendment) — the most common tuning knob

This is a **runtime parameter**, not a code change — passed as `mode` to `pipeline_ops.ingest_graph_json()`, selectable in the UI's Ingest tab (or via `--mode` on the CLI ingestion scripts). Two values, handled in `framework/core/ingest/graph_loader.py`:

- **`mode="base"`**: any existing node with the same `id` is fully overwritten. Use for a first-time load or a deliberate full reload.
- **`mode="amendment"`**: compares `content_hash` on each incoming entity against what's already in Neo4j. If unchanged, no-op. If changed, the OLD version is cloned into a new node (`{id}__superseded__{old_hash}`), linked `OLD -[:SUPERSEDED_BY]-> NEW`, and the main node is updated to the new content. This is how "Income Tax Act 2025 supersedes 1961, section-by-section" is implemented — use the **same `document_id`/entity-id scheme** across both ingestions so entity IDs collide and get diffed, rather than creating two disconnected sets of nodes.

No code change needed to use either mode — this is purely a matter of picking the right one per upload. Code change would only be needed if you wanted a **third** mode (e.g. "always version, never overwrite even on first load") — that logic lives in `diff_and_prepare_update()` in `graph_loader.py`.

### 4b. Changing what metadata gets attached to entities/chunks during extraction

File: whichever converter's extraction prompt applies — e.g. `domain_packs/tax/converters/extract_entities.py`'s `EXTRACTION_SYSTEM_PROMPT` (Act path), `extract_case_law.py` (case law path), `extract_generic.py` (generic path). Each has a "RETRIEVAL METADATA" section in its system prompt instructing the LLM to populate fields like `legal_concepts`, `statutes_cited`, `cases_cited`, etc. Add a new field there, then also add its name to `CHUNK_METADATA_KEYS`/`EMBED_CHUNK_METADATA_KEYS` in `domain_packs/tax/config.py` (Section 2c above) — otherwise the field will exist on the Entity node but won't get mirrored onto the Chunk node used for retrieval filtering.

### 4c. Changing which LLM provider/model runs ingestion

File: `.env`.

```
USE_MODEL_INGESTION=CLAUDE            # or GEMINI, GPT — provider used for extraction calls
CLAUDE_MODEL_NAME_INGESTION=claude-haiku-4-5-20251001
```

This is deliberately separate from `USE_MODEL_RETRIEVAL` (used for `/api/ask`) — cheap/fast model for bulk ingestion, stronger model for user-facing answers. Changing this takes effect on the very next LLM call (no restart needed — `llm_client.refresh_model_config()` re-reads `.env` every call).

### 4d. Changing the embedding model used at ingestion time

File: `.env`, `EMBEDDING_MODEL_NAME` (currently `law-ai/InLegalBERT`). Changing this changes the vector dimension of NEW chunks going forward automatically (`embedder.py` auto-detects dimension from the model). If you change it on a database that **already has ingested data**, you must re-embed everything that was embedded under the old model — run:

```
python -m framework.cli.reembed_all_chunks --domain tax
```

This rewrites every existing `Chunk.embedding` at the new model's dimension, then rebuilds the vector index. Skipping this step leaves old chunks embedded at the wrong dimension, silently breaking vector search for them (or erroring outright if the index dimension no longer matches).

### 4e. Changing the ingestion pipeline's stage sequence itself

File: `domain_packs/tax/pipeline.py`, function `_run_ingestion_job()` (single-file) / `_run_batch_ingestion_job()` (batch). The four stages are explicit and sequential:

```python
md_path = convert_document(upload_path, file_type)              # [1/4]
graph_path, chunks_path = run_extraction(md_path, file_type, doc_meta)  # [2/4]
ingest_stats = pipeline_ops.ingest_graph_json(...)               # [3/4]
embed_stats = pipeline_ops.embed_chunks_json(...)                # [4/4]
```

To add a new stage (e.g. a post-ingestion validation or a notification step), insert it into this function — both the single-file and batch variants would need the same addition since batch doesn't call the single-file function directly (it inlines the same four calls for its own per-file try/except handling).

Conversion and extraction run as **subprocesses** of the same CLI scripts you could run by hand from a terminal (see each converter's own docstring for the exact command) — so the UI's "Run Ingestion" button and running the script directly always behave identically. Graph loading and embedding run in-process instead (no subprocess). If you need to debug a stage in isolation, you can always run its underlying script directly rather than going through the UI.

---

## Quick reference: file map

| Area | File(s) |
|---|---|
| Neo4j connection settings | `.env` |
| Neo4j container config | `docker-compose.yml` |
| Neo4j constraints/indexes | `framework/core/infra/neo4j_client.py` |
| Ontology / schema (LLM-facing) | `domain_packs/tax/ontology/schema.json` |
| Domain schema config (labels, resolution keys) | `domain_packs/tax/config.py` |
| Act chunking (regex parser) | `domain_packs/tax/converters/chunking.py` |
| Case-law chunking/extraction | `domain_packs/tax/converters/extract_case_law.py` |
| Generic-doc chunking/extraction | `domain_packs/tax/converters/extract_generic.py` |
| Ingestion orchestration (all 4 stages) | `domain_packs/tax/pipeline.py` |
| Graph-loading logic (base/amendment mode) | `framework/core/ingest/graph_loader.py` |
| Embedding logic | `framework/core/ingest/embedder.py`, `framework/cli/reembed_all_chunks.py` |
