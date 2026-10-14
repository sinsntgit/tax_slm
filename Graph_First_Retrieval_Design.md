# Graph-First Retrieval: Schema Additions & Rollout Plan

Design only — no code changes have been made yet. This follows on from the case-law imbalance discussion and evaluates a shift from chunk-embedding RAG to node-metadata-driven entry points + graph traversal + query-time full-text hydration.

## 1. Scope of this first cut

Piloted on the `case_law_hierarchy` category only (highest payoff, smallest blast radius), with the existing chunk/vector pipeline left running unchanged as a fallback until the new fields are proven reliably populated across the corpus. Not a wholesale migration of Act/Circular/Treaty retrieval in this pass.

## 2. Ontology additions (`schema.json`)

All new entity-level properties follow the project's existing camelCase datatype-property convention (see `bindingScope`, `effectiveFrom`, `ratioDecidendi`).

### 2.1 Stance (answers "for vs. against")

```json
{
  "name": "stance",
  "domain": ["CaseLaw"],
  "range": ["<class 'str'>"],
  "comment": [
    "Controlled vocabulary: 'taxpayer_favorable' | 'revenue_favorable' | 'mixed' | 'procedural'. ",
    "'procedural' is reserved for cases actually decided on a procedural point with no ",
    "substantive holding on the tax question - NOT a dumping ground for cases the extractor ",
    "found hard to classify. Scoped to the case's PRIMARY issue; a case with multiple issues ",
    "and split outcomes should be 'mixed', not force-fit to one side."
  ]
}
```

```json
{
  "name": "stanceConfidence",
  "domain": ["CaseLaw"],
  "range": ["<class 'str'>"],
  "comment": ["'high' | 'medium' | 'low' - lets retrieval/generation flag or down-weight a low-confidence stance instead of silently trusting it. Populated by the same extraction call that sets `stance`."]
}
```

Extraction must be required to justify `stance` from the case's own `holding`/`final_outcome` text, not infer it from the case name or outcome direction alone (this is exactly the class of error that mislabeled Samsung as "procedural").

### 2.2 Legal precedence / override hierarchy

This is the highest-risk piece flagged in the evaluation — modeled narrowly and conditionally, not as a blanket rule.

```json
{
  "name": "OVERRIDES",
  "domain": ["TaxTreaty", "StatutoryProvision", "Circular"],
  "range": ["TaxTreaty", "StatutoryProvision", "Circular"],
  "inverse": "OVERRIDDEN_BY",
  "functional": false
},
{
  "name": "OVERRIDDEN_BY",
  "domain": ["TaxTreaty", "StatutoryProvision", "Circular"],
  "range": ["TaxTreaty", "StatutoryProvision", "Circular"],
  "inverse": "OVERRIDES",
  "functional": false
}
```

```json
{
  "name": "overrideCondition",
  "domain": ["StatutoryProvision", "TaxTreaty", "Circular"],
  "range": ["<class 'str'>"],
  "comment": [
    "Free-text statement of WHEN the OVERRIDES edge applies - e.g. 'only if treaty provisions ",
    "are more beneficial to the assessee, per sec 90(2)' or 'subject to GAAR/PPT under the MLI'. ",
    "Mandatory whenever an OVERRIDES edge is created - an edge with no condition text should be ",
    "treated as a data-quality error, not an absolute rule, given how conditional real precedence is."
  ]
}
```

Recommendation: seed this relationship manually/deterministically for a small, well-known set of cases (treaty-vs-Act via sec 90(2), a handful of documented amendment-supersession chains) rather than having the LLM infer override edges freely at ingestion time — validate each seeded edge against a known outcome before trusting it in retrieval.

### 2.3 Retrieval entry-point metadata (replaces chunk-text embedding as the search surface)

```json
{
  "name": "retrievalSummary",
  "domain": ["CaseLaw", "StatutoryProvision", "TaxTreaty", "Circular", "FirmOpinion"],
  "range": ["<class 'str'>"],
  "comment": [
    "3-5 sentence LLM-generated abstract written FOR retrieval, not for display - must explicitly ",
    "name the substantive issue and outcome (e.g. 'holds software licence payments do NOT ",
    "constitute royalty absent transfer of copyright'), not just restate the case caption. This is ",
    "the primary signal the new entry-point vector index searches over, so a generic/thin summary ",
    "here directly degrades recall - treat generation quality here as retrieval-critical, not cosmetic."
  ]
}
```

```json
{
  "name": "retrievalKeywords",
  "domain": ["CaseLaw", "StatutoryProvision", "TaxTreaty", "Circular", "FirmOpinion"],
  "range": ["<class 'str'>"],
  "comment": ["Comma-separated list-as-string of the specific legal terms/issue phrases this entity is relevant to, independent of its own caption/title wording (e.g. a case never using the word 'royalty' verbatim should still list it here if that's the issue it decides)."]
}
```

### 2.4 Source-location pointers (chunk/document metadata, snake_case per existing `CHUNK_METADATA_KEYS` convention)

Extends the existing `page_number`/`line_start`/`line_end` fields already captured — not new infrastructure, a precision upgrade:

- `source_document_id` — already exists via document stamping (task history #210); confirm present on every entity type, not just CaseLaw.
- `char_offset_start`, `char_offset_end` — exact slice into the converted `.md`/`.txt` file for this entity's full text.
- `paragraph_id` — stable anchor if the converted file's line numbering ever shifts (e.g. re-conversion with a different tool version); offsets alone are fragile across re-ingestion.

These get set once at ingestion (chunking already computes something close to this) and read at query time to hydrate full text — no live PDF re-parsing per question.

## 3. Retrieval architecture changes

1. **Separate entity-metadata vector index**, built over `retrievalSummary` + `retrievalKeywords` (not chunk text). Distinct from the existing chunk-embedding index — both can coexist during the pilot.
2. **Entry-point retrieval**: for `case_law_hierarchy`, run the metadata-vector search, but do NOT rely on it alone for stance balance — additionally run two explicit Cypher pulls (`stance = 'taxpayer_favorable'`, `stance = 'revenue_favorable'`) over whatever's graph-reachable from the matched issue/topic, and merge. This is the concrete fix for the imbalance: balance is enforced by an explicit query, not hoped for from embedding similarity.
3. **Graph traversal expansion**: from entry-point nodes, walk `OVERRULES`/`OVERRULED_BY`/`OVERRIDES`/`OVERRIDDEN_BY`/`CORRESPONDS_TO` to pull in directly relevant connected authorities (e.g. a case reversed by Engineering Analysis should traversal-link to it automatically).
4. **Query-time text hydration**: for the final node set only, fetch text via the stored offset pointers (§2.4) — not the whole source file, not a pre-cut 1500-char chunk. Full holding/paragraph text, precisely bounded.
5. **Fallback safety net**: if the metadata-vector + stance-split Cypher pulls come back thin (below some minimum count per stance), fall back to the existing chunk-vector path for that category on that question — mirrors the reranker's existing `fallback_relevance` tier rather than failing silently or answering one-sided.

## 4. Phased rollout

| Phase | Scope | Exit criteria before moving on |
|---|---|---|
| 0 | Schema changes above land in `schema.json`, validated | `/api/ontology` still loads clean; no domain/range regressions |
| 1 | Backfill `stance`/`stanceConfidence`/`retrievalSummary`/`retrievalKeywords` onto existing CaseLaw corpus (one-time extraction pass, same pattern as the existing metadata backfill button) | Spot-check: every ingested case has a non-empty stance with justification traceable to its holding text |
| 2 | Build the entity-metadata vector index + entry-point retrieval function, `case_law_hierarchy` only | Re-run the "for and against software royalty" question; manually verify Samsung/ZTE/Engineering Analysis all surface with correct stance |
| 3 | Add stance-split Cypher pulls + traversal expansion + fallback tier | Same test question, confirm balance holds even if metadata-vector alone would've been one-sided |
| 4 | Query-time hydration via offset pointers, replacing the 1500-char chunk cap for this category | Confirm full holding text reaches the LLM, not a truncated fragment |
| 5 | Seed a small, hand-validated set of `OVERRIDES` edges (sec 90(2) treaty-vs-Act being the obvious first case) | Each edge checked against a known real outcome before trusting it in any answer |
| 6 | Decide whether to extend beyond `case_law_hierarchy` to other categories, based on Phase 1-5 results | — |

Ingestion coverage gaps (Sunray Computers, Citrix AAR) are tracked separately — sourcing those PDFs isn't blocked on any of the above and can happen in parallel.

## 5. Open items before implementation starts

- Confirm `source_document_id` is actually present on every entity type today, not just CaseLaw (needs a quick schema/data check).
- Decide backfill cost: re-running LLM extraction over the full existing case-law corpus for the four new fields — how many documents, roughly what LLM spend.
- Decide who validates the seeded `OVERRIDES` edges in Phase 5 (legal-domain review, not something to auto-generate and trust).
