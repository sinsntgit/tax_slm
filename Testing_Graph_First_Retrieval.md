# Testing the Graph-First Case-Law Retrieval Changes

Companion to Graph_First_Retrieval_Design.md. Everything below tests the implementation of that design's Phases 0-3 (stance property, entry-point index, stance-balanced retrieval, prompt instruction, sec 90(2) OVERRIDES) plus the pilot scope decision made during implementation (case-law hydration via raised truncation caps rather than raw-file offset math - see that file's Section 2.4 note).

All commands below run from `tax_kg_poc_revised/domain_packs/tax/converters` or the repo root as noted, with your normal `.env`/venv active.

## 1. Confirm the schema loaded

```bash
cd tax_kg_poc_revised
python -c "
import json
schema = json.load(open('domain_packs/tax/ontology/schema.json'))
names = {p['name'] for p in schema['data_properties']}
print('stance/stanceBasis/retrievalSummary/retrievalKeywords/overrideCondition present:',
      {'stance','stanceBasis','retrievalSummary','retrievalKeywords','overrideCondition'} <= names)
rel_names = {p['name'] for p in schema['object_properties']}
print('OVERRIDES/OVERRIDDEN_BY present:', {'OVERRIDES','OVERRIDDEN_BY'} <= rel_names)
"
```
Then open the app's **View Ontology** screen and confirm `stance`, `retrievalSummary`, `OVERRIDES` show up - this also exercises `/api/ontology`, catching any schema regression the command above wouldn't.

## 2. Backfill the existing case-law corpus

Dry run first (prints what would change, writes nothing):
```bash
cd domain_packs/tax/converters
python backfill_case_law_stance.py --dry-run
```
Look at the printed per-case stance lines. Specifically check that **CIT v. Samsung Electronics** and **CIT v. ZTE Corporation** (or whatever your ingested filenames map to) come back `revenue_favorable`, not `unknown` or blank - that's the exact mischaracterization this whole change targets. If either comes back `unknown`, the case's `issue_outcomes[].winner` was never populated at ingestion time - re-ingest that PDF (see extract_case_law.py) rather than expecting the backfill to fix it; the backfill can only derive from what's already in the graph.

If the dry run looks right, write the changes:
```bash
python backfill_case_law_stance.py
```

## 3. Build the entry-point vector index

```bash
cd tax_kg_poc_revised
python -m framework.cli.embed_case_law
```
This embeds every `CaseLaw` node's `retrievalSummary`/`retrievalKeywords` and creates the `case_law_retrieval_index` vector index. Re-run this any time you re-run the backfill or ingest new case law.

## 4. Direct retrieval test (bypasses the LLM, cheapest way to check the fix)

```python
from framework.core.infra import neo4j_client
from framework.core.retrieve import category_retrieval as cr
from domain_packs.tax.config import TAX_CATEGORY_SPECS
from framework.core.retrieve.graphrag import GraphRAGRetriever   # or however your app constructs it

retriever = GraphRAGRetriever()   # match your app's actual construction args
results = cr.run_category_retrieval(
    "give me for and against case laws for software taxable as royalties",
    retriever, TAX_CATEGORY_SPECS, reranker_module=None,
)
clh = [r for r in results if r.key == "case_law_hierarchy"][0]
print(clh.status, clh.note)
for c in clh.chunks:
    print("-", c["chunk_id"], c["text"][:100])
```
What "good" looks like: `status == "graph"` and `note` mentions a non-zero count on **both** `taxpayer-favorable` and `revenue-favorable`. If it instead falls to the generic path (`status` still `"graph"` or `"vector_fallback"` but `note` is empty/generic), that means fewer than 2 cases per side matched on topic keywords - check step 2's backfill actually ran, and that the case-law corpus has enough Revenue-favorable authorities on this topic to find (a genuine corpus-coverage question, separate from the retrieval logic).

## 5. Full end-to-end: re-ask the original question in the app

Re-ask exactly: **"give me for and against case laws for software taxable as royalties"**

Check the answer against the original complaint point by point:
- Is **CIT v. ZTE Corporation** now presented as a Revenue-favorable authority (not "insufficient context")?
- Is **CIT v. Samsung Electronics** presented with its actual holding (not reduced to "procedural")?
- Does the Case Law Hierarchy section show both a taxpayer-favorable table/list AND a Revenue-favorable one?
- Open the "Evidence Audit Trail" / retrieved-chunks view for this answer - are the case-law chunks now longer/more complete than before (no more mid-sentence truncation)?

Sunray Computers and Citrix Systems AAR will **still** be absent unless you've separately sourced and ingested those PDFs - that's the corpus-coverage gap from the original discussion, not something this retrieval change can fix. Confirm the answer no longer claims these don't exist though - it should be silent on them, not wrong about them.

## 6. Sanity-check the sec 90(2) OVERRIDES edge

Only meaningful once you've re-ingested (or ingested a new) treaty after this change:
```cypher
MATCH (t:TaxTreaty)-[o:OVERRIDES]->(a:Act)
RETURN t.treatyCountry, a.node_id, o.overrideCondition
```
Should return one row per ingested treaty, with a real `overrideCondition` string (not null/empty) on every row - a null condition on a real edge would itself be a data-quality bug per the schema comment.

## 7. Regression check on the other five categories

Ask a question that should hit `law_regulatory_framework` or `opinion_expert_views` only (e.g. a plain Section-lookup question) and confirm those categories still behave exactly as before - the case-law-specific change is scoped to only fire for `case_law_hierarchy`; nothing else should look different. The mocked test I ran during implementation confirms this in isolation, but a real question against your live graph is the real check.

## What would tell you this needs more work, not less

- Backfill leaves most cases as `unknown` -> extraction-quality issue (issue_outcomes not being populated), not a retrieval issue - look at re-ingestion, not this retrieval code.
- Stance-balanced path keeps falling through to the generic path for topics you know have real cases on both sides -> `min_per_stance=2` in `category_retrieval.py`'s `_case_law_hierarchy_retrieval()` may be too strict for a thin corpus - consider lowering it, or check whether `retrievalKeywords`/`retrievalSummary` text actually contains the terms your questions use.
- Answer still merges/confuses stance for a case you know is one-sided -> that's now a generation-prompt problem, not retrieval - the evidence is reaching the model correctly (verify with step 4) but it isn't using the stance instruction; worth revisiting the `case_law_hierarchy` `instructions` text in `domain_packs/tax/config.py`.
