# Testing the Semantic Graph Retrieval Mode

Companion to `Semantic_Graph_Retrieval_Architecture.md`. All commands below
run from `tax_kg_poc_revised` (repo root) with your normal `.env`/venv
active, unless a `cd` is shown.

## 0. Do you need to re-ingest everything? — short answer

**No.** `sourceMdPath` is stamped automatically the moment an entity is
(re-)ingested, because that's baked into the extraction converters now — you
never need to think about it for *new* ingestion.

The only thing that needs attention is entities that were **already in the
graph before 2026-08-29** (when this property was added) — those rows
exist without `sourceMdPath` and need it added, but that does **not** mean
re-running the full extraction pipeline (which would re-spend LLM calls, and
for the Act specifically has historically taken ~2 hours). Instead, run the
no-LLM backfill script in step 2.

As of this writing, based on what's actually in `data/converted_md/` and
`data/rawdata/`:

| Entity type | Already ingested? | What to do |
|---|---|---|
| CaseLaw (11 judgments) | Yes | Needs the backfill (step 2) |
| Section / StatutoryProvision (both Acts: 1961 + 2025) | Yes | Needs the backfill (step 2) |
| Circular (CBDT circulars in `data/rawdata/circulars/`) | **Not yet** — only raw PDFs exist, no `.md`/graph entities | Just ingest for the first time (step 4) — `sourceMdPath` comes for free, no backfill needed |
| TaxTreaty (agreements in `data/rawdata/international_agreements/`) | **Not yet** — same as Circular | Same — ingest fresh, no backfill needed |

If your setup differs (e.g. you've since ingested circulars/treaties, or
re-ingested case law), the backfill script is safe to run regardless — it
only touches nodes where `sourceMdPath` is still empty, so it's a no-op for
anything already stamped.

## 1. Confirm the schema loaded

```bash
python -c "
import json
schema = json.load(open('domain_packs/tax/ontology/schema.json'))
names = {p['name'] for p in schema['data_properties']}
print('sourceMdPath present:', 'sourceMdPath' in names)
"
```

## 2. Backfill sourceMdPath onto the existing CaseLaw + Section corpus

Dry run first (prints what would change, writes nothing):
```bash
cd domain_packs/tax/converters
python backfill_source_md_path.py --md-dir ../../../data/converted_md --dry-run
```
Check the printed lines:
- Every CaseLaw row should resolve to a real path under `data/converted_md/`
  (not a `SKIP ... does not exist` line — if you see one, that case's
  `source_document` filename doesn't match anything in `--md-dir`, which
  usually means it was ingested from a `.md` file that's since been moved
  or renamed).
- The Section summary line should show your two Acts' `document_id`s (e.g.
  `ITA1961`, `ITA2025`) each resolving to the right `.md` file via the
  year-substring guess. If a `document_id` doesn't auto-resolve, re-run with
  `--act-map "ITA1961=1b52526e_income_tax_act_1961.md,ITA2025=14456eeb_Income_Tax_Act_2025_as_amended_by_FA_Act_2026.md"`
  (substitute the actual filenames in your `data/converted_md/` if
  different).

If it looks right, write the changes:
```bash
python backfill_source_md_path.py --md-dir ../../../data/converted_md
```

## 3. Build the entry-point vector indexes

```bash
cd ../../..   # back to repo root
python -m framework.cli.embed_case_law
python -m framework.cli.embed_sections
```
Re-run either any time you re-ingest or backfill more of that entity type.

## 4. Ingest at least one Circular and one Treaty (first time, no backfill needed)

Using whichever converter/orchestrator you normally use, e.g.:
```bash
cd domain_packs/tax/converters
python extract_circular.py --md <path to a converted circular .md> --out <out.json> ...
python extract_treaty.py --md <path to a converted treaty .md> --out <out.json> ...
```
(Substitute your normal ingestion command/orchestrator and flags — the
point of this step is just to have at least one of each in the graph with
`sourceMdPath` already set, so step 5 below has something to find.)

Then build their entry-point indexes:
```bash
cd ../../..
python -m framework.cli.embed_circulars
python -m framework.cli.embed_treaties
```

## 5. Direct retrieval test (bypasses the LLM, cheapest way to check hydration)

```python
from framework.core.retrieve.graphrag import GraphRAGRetriever
from domain_packs.tax.config import DOMAIN_CONFIG

retriever = GraphRAGRetriever(DOMAIN_CONFIG, ...)   # match your app's actual construction args
results = retriever.semantic_entry_point_search("software payments taxable as royalties", top_k=10)
for r in results:
    print(r["chunk_id"], r["similarity"], "-", r["text"][:120].replace("\n", " "))
```
What "good" looks like:
- `chunk_id` values look like `mdhydrate_<neo4j-element-id>` (never a real
  `Chunk` id).
- `text` is real prose from the source document, not empty/truncated
  mid-sentence — for a CaseLaw/Circular/TaxTreaty hit this should read like
  the whole document; for a Section hit it should read like just that
  section's text.
- You should see hits from more than one entity type if your query touches
  more than one (e.g. this royalties example should surface both CaseLaw
  and Section hits, and a TaxTreaty hit once one is ingested).

If nothing comes back: check the embed CLIs actually ran (step 3/4) and that
the backfill (step 2) actually wrote changes (not just `--dry-run`).

## 6. Full end-to-end: ask a question through the app in Semantic Graph mode

On the Ask screen, click the new **"Semantic Graph"** button (next to "KG +
RAG"/"Simple RAG") and ask a real question, e.g. **"is software payment
taxable as royalty under Indian tax law"**.

Check:
- The answer's citations resolve to real document names/pages (not raw
  `chunk_id=...` text) — this exercises the synthetic-id resolution fix in
  `get_document_details()`/`get_entity_labels_for_chunks()`.
- The "Evidence Audit Trail" / retrieved-chunks view shows full-length text
  for CaseLaw/Circular/TaxTreaty hits (not the old ~800-2000 char truncated
  chunks) and a precise section-length excerpt for Section hits.
- Re-run the SAME question in "KG + RAG" mode and compare — both should
  reach a similar substantive answer (same underlying graph), but Semantic
  Graph's evidence text should read as more complete/verbatim.

## What would tell you this needs more work, not less

- Everything in step 5 comes back empty -> the embed CLIs didn't find any
  nodes with a populated `sourceMdPath`/text property to embed - re-check
  step 2 actually wrote (not `--dry-run`), and that `embed_nodes()`'s
  `where_not_empty_property` (`hasText` for Section/Circular/TaxTreaty,
  `retrievalSummary` for CaseLaw) is genuinely non-empty on your nodes.
- Step 5 returns results but `text` is empty or a `SKIP`-style short string
  -> hydration is failing - check the printed `sourceMdPath` value on that
  node actually exists as a file (`ls` it), and that it's readable
  (encoding/permissions).
- Section hits return the WRONG section's text -> check `line_start`/
  `line_end` on that Section node look sane (`MATCH (s:Section) WHERE
  s.node_id = '...' RETURN s.line_start, s.line_end`) - a bad line range
  from `chunking.py`'s parser would misfire here, not something this
  retrieval layer can detect on its own.
