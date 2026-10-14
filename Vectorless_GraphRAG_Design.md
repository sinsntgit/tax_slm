# Vectorless GraphRAG - Implementation Plan

Design only - no code changes yet. Scope per your answers: project-wide (not just case law), reranker stays, built as a new additive `retrieval_mode="vectorless"` alongside the existing `"hybrid"`/`"vector_only"` - nothing existing gets removed or changed in its current behavior.

## 1. What "vectorless" means here, precisely

Three call sites in the current pipeline do ANN (approximate-nearest-neighbor) vector search today:

| Call site | File | Index used |
|---|---|---|
| Main hybrid pipeline's semantic fallback | `graphrag.py`'s `vector_search_chunks()` | `chunk_embedding_index` (Chunk.embedding) |
| Per-category vector fallback | `category_retrieval.py`'s generic path, calls the same `vector_search_chunks()` | same index |
| Case-law entry points (added last session) | `case_law_entry_points.py`'s `search_case_law_entry_points()` | `case_law_retrieval_index` (CaseLaw.retrieval_embedding) |

"Vectorless" replaces all three, for `retrieval_mode="vectorless"` only, with **Neo4j native full-text search** (Lucene-backed, `db.index.fulltext.queryNodes`) - real BM25/TF-IDF-style lexical ranking with fuzzy-match and OR-of-terms support, not the existing crude `CONTAINS`-based keyword pre-filter (`category_retrieval._graph_keyword_search()`), which stays as a separate, even-cruder mechanism used elsewhere.

Everything else in the pipeline is **already vector-free and untouched**: NL2Cypher graph queries, citation-exact-match, `PART_OF`-hierarchy traversal (`get_chunk_ids_via_graph_traversal`, `get_parent_section_context`), the stance-split Cypher pulls added for case law, and the BGE cross-encoder reranker (per your answer - it scores candidate pairs directly, no index, so it isn't "vector search" in the ANN sense and stays in the pipeline regardless of mode).

**Why do this at all - the actual trade-off to go in eyes-open:**
- **Gain:** every retrieved chunk becomes justifiable by an exact keyword match or an explicit graph relationship, not an opaque cosine-similarity score - meaningful for a legal-research tool where "why did the system surface this" is a real question. Also drops the `sentence-transformers` runtime dependency and the embedding-freshness problem (re-embed after every ontology/model change) for this mode.
- **Cost:** lexical search has no semantic generalization. A question that says "royalty" against a chunk that only says "licence fee" will recall WORSE under vectorless than hybrid - this is inherent, not a bug to fix. Expect vectorless to underperform hybrid specifically on `CONCEPTUAL`-intent, paraphrased questions, and outperform or match it on `EXACT_LOOKUP`/`RELATIONAL` (citation-style) questions where lexical matching is already strong. Test both intent types, not just the case-law question, before deciding this becomes a default anywhere.

## 2. New infrastructure

### 2.1 `framework/core/retrieve/fulltext_search.py` (new file)

- `ensure_fulltext_indexes(session)` - creates:
  - `chunk_fulltext_index`: `CREATE FULLTEXT INDEX chunk_fulltext_index FOR (n:Chunk) ON EACH [n.content]` - the direct replacement for `chunk_embedding_index`, same node type, so every existing chunk becomes searchable with zero re-ingestion.
  - `case_law_fulltext_index`: `FOR (n:CaseLaw) ON EACH [n.retrievalSummary, n.retrievalKeywords, n.holding, n.case_name, n.full_citation]` - replaces `case_law_retrieval_index`.
  - One additional index per `CategorySpec` in `TAX_CATEGORY_SPECS` reusing its existing `node_labels`/`text_properties` (already-declared, domain-supplied metadata - no new config needed) - upgrades `category_retrieval._graph_keyword_search()`'s crude `CONTAINS` scan to real full-text ranking, which also fixes the plural/singular and non-discriminating-keyword brittleness flagged in the original case-law investigation, as a side benefit independent of `retrieval_mode`.
- `build_fulltext_query(keywords: list[str]) -> str` - turns `category_retrieval.extract_keywords()`'s output (already exists, already domain-agnostic, already used elsewhere) into a Lucene query string: OR-joined terms, each with `~1` fuzziness for typo tolerance, **with Lucene special-character escaping** (`+ - && || ! ( ) { } [ ] ^ " ~ * ? : \`) - this domain's questions are full of section-number tokens like `9(1)(vi)` that are syntactically meaningful to Lucene but semantically just a citation; get this wrong and every parenthesized citation silently breaks the query instead of matching literally.
- `search_chunks_fulltext(question: str, top_k: int) -> list[dict]` - returns the **exact same shape** `vector_search_chunks()` returns (`chunk_id`/`text`/a relevance score field) so every caller can branch on `retrieval_mode` without touching downstream merge/rerank logic.
- `search_case_law_entry_points_fulltext(question: str, top_k: int) -> list[dict]` - same shape as `case_law_entry_points.search_case_law_entry_points()`.

Neo4j full-text indexes are available in Community Edition (not an Enterprise-only feature) - worth a one-time check against whatever edition is actually running, but not expected to be a blocker.

### 2.2 No new Python dependencies

Native Neo4j feature - no `sentence-transformers`/model download needed for this mode specifically (still needed for `"hybrid"`/`"vector_only"`, which are untouched).

## 3. Per-file changes

**`framework/core/retrieve/graphrag.py`** - `retrieve()`: add a third branch alongside the existing `if retrieval_mode == "vector_only"` check - when `retrieval_mode == "vectorless"`, call `fulltext_search.search_chunks_fulltext()` instead of `vector_search_chunks()`, tag results `source="fulltext"` instead of `"vector"`. NL2Cypher/citation-match paths, `intent_retrieval_config`'s per-intent `top_k` sizing, and parent-section expansion (`get_parent_section_context` - currently only skipped for `"vector_only"`, correctly stays ENABLED for `"vectorless"` since this mode leans graph-heavy) all reused unchanged.

**`framework/core/retrieve/category_retrieval.py`** - `run_category_retrieval()`: thread `retrieval_mode` into the generic path's vector-fallback call (swap `retriever.vector_search_chunks()` for `fulltext_search.search_chunks_fulltext()` when mode is `"vectorless"`); update `_case_law_hierarchy_retrieval()` to call `search_case_law_entry_points_fulltext()` instead of the vector version under this mode - the stance-split Cypher pulls (already vector-free) need zero changes, so last session's case-law work mostly carries over as-is.

**`framework/webapp/app.py`** - `retrieval_mode` is already threaded end-to-end (`/api/ask` → `retriever.retrieve()` → `category_retrieval.run_category_retrieval()`); just extend whatever validates/enumerates the accepted values to include `"vectorless"`, and confirm the quality-checked retry job (`answer_evaluation.py`/background job) passes the mode through rather than hardcoding `"hybrid"`.

**`framework/webapp/static/index.html`** - Ask screen's retrieval-mode selector currently offers Hybrid / Simple RAG (`vector_only`) - add a third option, "Vectorless GraphRAG," wired to the same param. Purely additive UI, no existing option's behavior changes.

**Nothing removed:** `embedder.py`, `chunk_embedding_index`, `case_law_entry_points.py`, `EMBEDDING_MODEL_NAME` handling all stay exactly as they are, still driving `"hybrid"`/`"vector_only"`.

## 4. Phased rollout

| Phase | Scope | Exit criteria |
|---|---|---|
| 0 | Build `fulltext_search.py`, create indexes | Indexes show up in `SHOW INDEXES`; a hand-run `db.index.fulltext.queryNodes` call returns sane results against real data |
| 1 | Wire into `graphrag.py`'s main pipeline | Re-run a citation-style `EXACT_LOOKUP` question (e.g. "what does Section 9(1)(vi) say") under `vectorless` vs `hybrid` - results should be comparable, since lexical matching is already strong here |
| 2 | Wire into `category_retrieval.py`'s generic path + case-law stance-balanced path | Re-run the "for and against software royalty" question under `vectorless` - compare stance-balance quality against `hybrid`'s result from last session's testing |
| 3 | UI + quality-checked retry path | Confirm the Ask screen's mode selector round-trips correctly; confirm quality-checked retry doesn't silently fall back to hybrid |
| 4 | Comparative eval | Run a small fixed set of questions spanning all three intents (`EXACT_LOOKUP`/`RELATIONAL`/`CONCEPTUAL`) through both `hybrid` and `vectorless`, side by side - this is where you decide whether `vectorless` becomes a real option for users or stays an internal experiment. Expect the CONCEPTUAL-intent gap described in Section 1 to show up here; that's the real signal to evaluate, not a bug to chase. |

## 5. What I'd want confirmed before starting Phase 0

- Which Neo4j edition/version is actually running (Community full-text support assumed but not yet verified against your instance).
- Whether the per-category full-text indexes (Section 2.1's third bullet) should be built for **all six** categories now, or only for categories where you actually plan to compare `vectorless` against `hybrid` soon - building all six is cheap and I lean toward doing it now for consistency, but flagging it as a scope choice since it's extra index-maintenance surface for categories you may never route through `vectorless`.
