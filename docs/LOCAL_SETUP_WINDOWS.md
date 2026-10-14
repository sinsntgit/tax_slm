# Running tax_slm on a Windows laptop

Written 2026-10-14 for moving from the old `D:\tax_kg_poc_revised` folder to the
`tax_slm` repo. Commands are for **Command Prompt** (cmd). The old folder is left
untouched as a backup.

## 0. Prerequisites

- Git, Python 3.11 or newer, Docker Desktop (running).
- Poppler for PDF conversion. Your existing `.env` already has `POPPLER_BIN_DIR`.

## 1. Get the code

```bat
cd /d D:\
git clone https://github.com/sinsntgit/tax_slm.git
cd /d D:\tax_slm
```

## 2. Bring over your settings and source documents (not the old derived data)

```bat
copy D:\tax_kg_poc_revised\.env D:\tax_slm\.env
robocopy D:\tax_kg_poc_revised\data\rawdata D:\tax_slm\data\rawdata /E
```

Do **not** copy `data\converted_md`, `data\graphs`, `data\vector_store` or
`data\uploads`. They were built with the old ID scheme and will be regenerated.

New optional `.env` keys are listed in `.env.example`: local model, per-call
routing, call logging and judge routing. Nothing new is required to run.

## 3. Python environment and offline tests

```bat
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
```

Expect about 658 passed, 4 skipped (live tests) and 1 xfailed (a known, pinned bug).
The first `pip install` downloads torch for sentence-transformers and takes a while.

## 4. Start Neo4j and Redis with an empty graph

`docker-compose.yml` uses fixed container names, so stop the old stack first.
`docker compose down` keeps its data volumes, so the old graph stays as a backup:

```bat
cd /d D:\tax_kg_poc_revised
docker compose down
cd /d D:\tax_slm
docker compose up -d
```

The new folder gets its own empty volumes (`tax_slm_*`), which gives you the clean
graph needed for the re-ingest. If you use Neo4j Desktop instead, create a new
empty DBMS and point `NEO4J_URI` at it.

Check connectivity (live tests run only when the flag is set):

```bat
set TAXKG_LIVE_TESTS=1
python -m pytest tests\live -k "neo4j or validate"
set TAXKG_LIVE_TESTS=
```

## 5. Run the app

```bat
python -m framework.webapp.app --domain tax
```

Open http://localhost:5001. Code changes need a restart (the reloader is off on purpose).

## 6. Five-minute smoke test (no LLM cost)

1. Ingestion tab → file type **Firm Opinion** → upload 2-3 files from
   `data\rawdata\opinions`. Opinion parsing has no LLM call, so this takes seconds.
2. Ask tab → e.g. *"What did the firm conclude on withholding tax for technical
   support fees paid to a US company?"*
3. Check the answer cites the opinion. Cited sections and cases show as "cited,
   not verified", because the Acts and judgments aren't loaded yet. Client names
   should appear as `[CLIENT]`.
4. When satisfied, use **Delete Neo4j Data** (type `DELETE ALL DATA`) and do the full
   ingest below.

## 7. Full ingest - order matters

Case law links to Act sections only if the Acts are already in the graph, and those
links are dropped, not retried. So ingest in this order:

| Step | Ingestion tab file type | Files | Document ID |
|---|---|---|---|
| 1 | Income Tax Act (PDF) | `data\rawdata\acts\income_tax_act_1961.pdf` | **`ITA1961`** (replace the pre-filled value) |
| 2 | Income Tax Act (PDF) | `data\rawdata\acts\Income_Tax_Act_2025...pdf` | **`ITA2025`** |
| 3 | Case Law (PDF), batch | `data\rawdata\caselaws\*` | leave as generated |
| 4 | CBDT Circular (PDF) | `data\rawdata\circulars\*` | leave as generated |
| 5 | International Agreement / DTAA | `data\rawdata\international_agreements\*` | leave as generated |
| 6 | Firm Opinion | `data\rawdata\opinions\*.md` | leave as generated |

- **Time:** each Act takes about 2 hours because of LLM extraction; the rest take minutes.
- **Re-uploads:** an identical file is now skipped automatically. Tick "Re-ingest
  anyway" only if you mean it.

Then, in a second terminal (venv active):

```bat
cd /d D:\tax_slm\domain_packs\tax\converters
python build_act_correspondence.py --old-namespace ITA1961 --new-namespace ITA2025 --dry-run
python build_act_correspondence.py --old-namespace ITA1961 --new-namespace ITA2025
cd /d D:\tax_slm
python -m framework.cli.embed_sections
python -m framework.cli.embed_case_law
python -m framework.cli.embed_circulars
python -m framework.cli.embed_treaties
```

Ingestion now refreshes these indexes itself; running them once here makes sure.

## 8. Check it works

- **Validate Graph tab:** node and relationship counts, plus orphans.
- **Ask tab:** try each retrieval mode (KG + RAG, Simple RAG, Semantic Graph,
  Agentic, Structured Research) on a few questions you know the answer to.
- **Live end-to-end test:**

```bat
set TAXKG_LIVE_TESTS=1
set TAXKG_LIVE_QUESTION=What did the Supreme Court hold in Engineering Analysis on software royalty?
python -m pytest tests\live
```

## 9. First evaluation baseline (frontier model)

With the app running:

```bat
python -m eval.goldset eval/gold/
python -m eval.run --gold eval/gold/ --target http://localhost:5001 --retrieval-mode hybrid --answer-mode research --out eval/runs/smoke --limit 3
python -m eval.run --gold eval/gold/ --target http://localhost:5001 --retrieval-mode hybrid --answer-mode research --judge --out eval/runs/baseline_frontier_hybrid
```

Read `eval\runs\baseline_frontier_hybrid\report.md`. The gold set is a draft, so the
results are labelled PROVISIONAL until tax professionals verify the questions.

## Updating later

```bat
cd /d D:\tax_slm
git pull
pip install -r requirements.txt -r requirements-dev.txt
```

Restart the app after pulling.
