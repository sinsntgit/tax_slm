"""
GraphRAG retriever - GENERALIZED from tax_kg_poc/webapp/retrieval.py.

Two retrieval paths run for every question:
  1. GRAPH PATH: question -> Cypher (via the domain's nl2cypher prompt) ->
     Neo4j -> structured rows + matched nodes/edges.
  2. VECTOR PATH: question -> local embedding -> Neo4j's native vector index
     over Chunk.embedding -> similar chunks.
Both merge into one candidate chunk set for reranker.py/grounding.py to
refine further.

Everything that was tax-specific in the original module - the "Section
N(sub)(clause)" citation regex/normalizer, INTENT_RETRIEVAL_CONFIG - now
comes from DomainConfig (see framework/contracts/domain_config.py); the
nl2cypher prompt comes from the domain's PromptPack; PII masking goes
through a PiiMasker built from the same DomainConfig. Every other line -
graph traversal, vector search, PART_OF descendant expansion, metadata
filtering, the citation-fast-path/graph/vector merge order - is unchanged
from retrieval.py.
"""
import json
import logging
import os
import re

from dotenv import load_dotenv
from neo4j import Query
from neo4j.graph import Node, Relationship
from sentence_transformers import SentenceTransformer

from framework.core.infra import llm_client, neo4j_client
from framework.core.infra.dotenv_path import env_path as _env_path
from framework.core.retrieve.pii import PiiMasker
from framework.core.retrieve import retrieval_router

# Anchored to the repo root by file location, not process cwd - see
# framework.core.infra.dotenv_path's docstring.
load_dotenv(_env_path())
logger = logging.getLogger("graphrag")

EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
# Must match framework/core/ingest/embedder.py's VECTOR_INDEX_NAME.
VECTOR_INDEX_NAME = "chunk_embedding_index"
GRAPH_QUERY_TIMEOUT_SECONDS = int(os.environ.get("NEO4J_STATEMENT_TIMEOUT_SECONDS", "20"))

MAX_DESCENDANT_CHUNKS = 40  # cap per retrieve() call - see get_chunk_ids_via_graph_traversal()

# 2026-10-14: was a hardcoded max_tokens=300 in nl_to_cypher(). The nl2cypher
# prompt's own worked examples (OVERRULES-by-topic, the demerger metadata
# query, OPTIONAL MATCH correspondence lookups) already run 60-120 tokens, and
# a question combining a section + concept + case filter routinely needs more
# - at 300 the model's output could be cut mid-clause, and a cut query that
# happens to still parse (e.g. a WHERE that lost its last OR-branch) silently
# answers a DIFFERENT question. 1024 leaves ample headroom while still
# bounding cost; env-overridable like the other retrieval knobs here.
# Truncated output is additionally detected and dropped - see
# cypher_incompleteness_reason().
NL2CYPHER_MAX_TOKENS = int(os.environ.get("NL2CYPHER_MAX_TOKENS", "1024"))

# A Cypher query that ENDS on one of these keywords/operators was cut off
# mid-clause (nothing valid can follow e.g. a bare WHERE/AND or a trailing
# comma). Deliberately excludes '*' (`RETURN *` is valid) and END/closing
# brackets (bracket balance is checked separately).
_DANGLING_CYPHER_TAIL_RE = re.compile(
    r"(?:\b(?:WHERE|AND|OR|XOR|NOT|MATCH|OPTIONAL|WITH|RETURN|ORDER|BY|LIMIT|SKIP|CONTAINS|STARTS|ENDS|"
    r"IN|AS|UNWIND|DISTINCT|IS|CASE|WHEN|THEN|ELSE|UNION|CALL|YIELD)|[,.:=<>+\-/|])\s*$",
    re.IGNORECASE,
)
_CYPHER_BRACKET_PAIRS = {")": "(", "]": "[", "}": "{"}


def cypher_incompleteness_reason(cypher: str) -> str | None:
    """2026-10-14: NEW. Returns a short reason string if `cypher` looks
    TRUNCATED or structurally incomplete (so it must not be executed), else
    None. Deterministic, no Neo4j round-trip: unbalanced ()/[]/{} brackets,
    an unterminated string/backtick literal, no RETURN clause at all (a read
    query cannot end without one), or a dangling trailing keyword/operator
    (`... WHERE n.x =`, `... AND`, `RETURN n,`). "NO_QUERY"/empty input is not
    this function's concern - returns None for it. Not a full parser: a cut
    that happens to land on a clause boundary of a still-valid query cannot
    be detected from the text alone; the raised NL2CYPHER_MAX_TOKENS above
    is what makes that case rare."""
    text = (cypher or "").strip()
    if not text or text == "NO_QUERY":
        return None
    stack: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\" and quote != "`":
                i += 2
                continue
            if ch == quote:
                quote = None
        elif ch in ("'", '"', "`"):
            quote = ch
        elif ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != _CYPHER_BRACKET_PAIRS[ch]:
                return f"unbalanced '{ch}'"
            stack.pop()
        i += 1
    if quote:
        return f"unterminated {quote} literal"
    if stack:
        return f"unclosed '{stack[-1]}'"
    # Keyword checks run on the text with string literals blanked out, so a
    # literal like 'act no. 15 of 2024, and' can't fake a RETURN or a tail.
    no_literals = re.sub(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"|`[^`]*`", "''", text)
    if not re.search(r"\bRETURN\b", no_literals, re.IGNORECASE):
        return "no RETURN clause"
    if _DANGLING_CYPHER_TAIL_RE.search(no_literals):
        return "ends mid-clause"
    return None

# 2026-09-07: NEW - bounds for comprehensive_graph_retrieval()'s fixed,
# LLM-free traversal (see that method's docstring). HOPS=2 (not 6, like
# get_chunk_ids_via_graph_traversal()'s PART_OF descendant walk) because this
# traversal is UNRESTRICTED on relationship TYPE (every type in the schema,
# not just PART_OF), so hop count and result-set size compound multiplicatively
# in a densely-connected graph - 2 hops across all types already reaches
# everything within "cited by something this cites" distance, which is the
# same practical radius expand_via_related_entities()'s curated 1-hop walk
# covers, just without the relationship-type curation. MAX_ROWS is a safety
# cap on the raw (a, rel, b) triples returned, applied in Cypher via LIMIT
# (cheap even on a large match), not after fetching everything.
COMPREHENSIVE_TRAVERSAL_HOPS = int(os.environ.get("COMPREHENSIVE_TRAVERSAL_HOPS", "2"))
COMPREHENSIVE_TRAVERSAL_MAX_ROWS = int(os.environ.get("COMPREHENSIVE_TRAVERSAL_MAX_ROWS", "300"))

# 2026-09-03: NEW - relationship types worth walking for vector-seeded graph
# expansion (see GraphRAGRetriever.expand_via_related_entities()) - curated
# to edges that connect ONE authority to a DIFFERENT, substantively related
# authority (case overrules/follows/distinguishes another case, circular
# supersedes/withdraws another circular, provision amends/is overridden by
# another provision). Deliberately excludes structural/hierarchy edges
# (PART_OF, hasSourceChunk - already covered by get_chunk_ids_via_graph_
# traversal()) and this-entity's-own-sub-parts edges (HAS_ARGUMENT,
# HAS_CONCLUSION, INTERPRETS - these just pull in the SAME entity's own
# children, not a different related authority). Also excludes EVENT_SUBJECT/
# EVENT_INSTRUMENT (LegalEvent's own reification edges) - the LegalEvent
# node itself carries no text/hasSourceChunk, and the DIRECT relationship
# it reifies (e.g. OVERRULES) is already in this list, so walking through
# LegalEvent would be redundant for retrieval purposes (LegalEvent is for
# the Temporal Analysis display, not context expansion).
GRAPH_EXPANSION_REL_TYPES = [
    "OVERRULES", "OVERRULED_BY", "FOLLOWS", "FOLLOWED_BY",
    "DISTINGUISHES_CASE", "DISTINGUISHED_BY", "RELIED_ON_BY", "RELIES_ON",
    "SUPERSEDES", "SUPERSEDED_BY", "WITHDRAWS", "WITHDRAWN_BY",
    "AMENDS", "AMENDED_BY", "OVERRIDES", "OVERRIDDEN_BY",
]

# vector_top_k sizes the CANDIDATE POOL fed to the reranker, not the final
# answer set - reranker.rerank()'s own top_n/min_relevance decide what
# actually reaches the LLM. Raised from 5/5/15 -> 25/25/30: with the reranker
# now trying to return up to 25 chunks that clear an 85% relevance bar (see
# reranker.py), a pool of only 5 candidates could never supply that many even
# in the best case, and for EXACT_LOOKUP/RELATIONAL questions (the two
# intents graph search runs for first) the vector path is exactly the
# fallback/supplement that needs enough headroom to be useful.
DEFAULT_INTENT_RETRIEVAL_CONFIG = {
    "EXACT_LOOKUP": {"vector_top_k": 25, "vector_mode": "fallback_only"},
    "RELATIONAL": {"vector_top_k": 25, "vector_mode": "always"},
    "CONCEPTUAL": {"vector_top_k": 30, "vector_mode": "always"},
}

_embed_model = None


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        logger.info(f"[graphrag] Loading embedding model '{EMBEDDING_MODEL_NAME}' for query-time vector search...")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embed_model


# --- low-level Neo4j record -> plain-dict helpers ----------------------------

def vertex_to_node(n: Node) -> dict:
    props = dict(n)
    label = next(iter(n.labels), None)
    return {"db_id": n.element_id, "id": props.get("id", n.element_id), "label": label, "properties": props}


def edge_to_link(e: Relationship) -> dict:
    props = dict(e)
    start_id = e.start_node.element_id if e.start_node is not None else None
    end_id = e.end_node.element_id if e.end_node is not None else None
    return {"db_id": e.element_id, "type": e.type, "start_id": start_id, "end_id": end_id, "properties": props}


_SCHEMA_CACHE = None


def get_schema_summary() -> dict:
    global _SCHEMA_CACHE
    if _SCHEMA_CACHE is not None:
        return _SCHEMA_CACHE
    with neo4j_client.get_session() as session:
        labels = [r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label")]
        rel_types = [r["relationshipType"] for r in
                     session.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")]
    _SCHEMA_CACHE = {"node_labels": sorted(labels), "relationship_types": sorted(rel_types)}
    return _SCHEMA_CACHE


def invalidate_schema_cache() -> None:
    """Forces the next get_schema_summary() call to re-query Neo4j's actual
    label/relationship-type catalog, instead of returning whatever was
    cached at the first call. Without this, ingesting a genuinely NEW label
    (e.g. the first-ever CaseLaw document, or any future new file type/
    entity type a domain pack adds) would never appear in the UI's "Seed by
    label" dropdown until the process restarts, even though the data is
    already in the graph - db.labels()/db.relationshipTypes() are catalog
    lookups, not full scans, so re-querying them is cheap. Call this
    anywhere new labels may have just been written: after a single-document
    or batch ingestion job completes (domain_packs/tax/pipeline.py), and
    after a graph clear (framework/webapp/app.py's /api/graph/clear already
    does this directly for its own reason - emptying the label list)."""
    global _SCHEMA_CACHE
    _SCHEMA_CACHE = None


class GraphRAGRetriever:
    """Ties the generic retrieval mechanics above to one domain's
    DomainConfig + PromptPack. Construct once per process (it lazily loads
    the embedding model on first use)."""

    def __init__(self, domain_config, prompt_pack):
        self.domain_config = domain_config
        self.prompt_pack = prompt_pack
        self.pii = PiiMasker(domain_config)
        cfg = dict(DEFAULT_INTENT_RETRIEVAL_CONFIG)
        cfg.update(domain_config.intent_retrieval_config or {})
        self.intent_retrieval_config = cfg
        self.default_retrieval_config = domain_config.default_retrieval_config
        # 2026-09-07: NEW - the retrieval config fast_mode uses in place of an
        # actual classified intent (see retrieve()'s fast_mode docstring) -
        # the BROADEST config available (widest vector_top_k, vector_mode=
        # "always"), since fast_mode has no detect_intent() classification to
        # narrow it and is compensating for skipping both intent detection
        # AND per-question NL2Cypher with wider recall elsewhere. Reuses the
        # domain's own "CONCEPTUAL" entry (this project's broadest intent)
        # when the domain pack defines one, falling back to a hardcoded
        # equally-broad default for a domain pack that doesn't use that
        # exact intent name.
        self.CONCEPTUAL_STYLE_FALLBACK_CONFIG = cfg.get(
            "CONCEPTUAL", {"vector_top_k": 30, "vector_mode": "always"})

    # --- PATH 1: graph traversal ---------------------------------------------
    def nl_to_cypher(self, question: str, schema: dict, trace=None) -> str:
        prompt = self.prompt_pack.get_prompt("nl2cypher")
        system = prompt.template.format(
            labels=", ".join(schema["node_labels"]),
            rel_types=", ".join(schema["relationship_types"]),
        )
        text = llm_client.complete(system=system, user_content=question, max_tokens=NL2CYPHER_MAX_TOKENS,
                                    trace=trace, name="nl2cypher").strip()
        text = re.sub(r"^```(cypher)?|```$", "", text, flags=re.MULTILINE).strip()
        # 2026-10-14: a single trailing ';' is harmless to the model's intent
        # but the Python driver rejects it - strip rather than fail.
        text = re.sub(r";\s*$", "", text).strip()
        # 2026-10-14: NEW - a truncated/unbalanced query is treated exactly
        # like NO_QUERY (graph path skipped, vector/citation paths still run)
        # instead of being executed - see cypher_incompleteness_reason().
        problem = cypher_incompleteness_reason(text)
        if problem:
            logger.warning(f"[graphrag] nl2cypher output looks incomplete ({problem}) - treating as NO_QUERY. "
                           f"Output was: {text[:300]!r}")
            return "NO_QUERY"
        return text

    def run_graph_retrieval(self, question: str, trace=None) -> dict:
        schema = get_schema_summary()
        cypher = self.nl_to_cypher(question, schema, trace=trace)

        result = {"cypher": None, "plain_rows": [], "nodes": {}, "edges": []}
        if not cypher or cypher == "NO_QUERY":
            return result
        result["cypher"] = cypher

        try:
            with neo4j_client.get_session() as session:
                db_result = session.run(Query(cypher, timeout=GRAPH_QUERY_TIMEOUT_SECONDS))
                keys = db_result.keys()
                records = list(db_result)

            plain_rows = []
            for record in records:
                row_dict = {}
                for key in keys:
                    val = record[key]
                    if isinstance(val, Node):
                        n = vertex_to_node(val)
                        result["nodes"][n["db_id"]] = n
                        row_dict[key] = n["properties"]
                    elif isinstance(val, Relationship):
                        result["edges"].append(edge_to_link(val))
                        row_dict[key] = dict(val)
                    else:
                        row_dict[key] = val
                plain_rows.append(row_dict)
            result["plain_rows"] = plain_rows
        except Exception as e:
            logger.warning(f"[graphrag] Cypher execution failed, continuing with vector-only retrieval: {e}")

        return result

    # --- FAST-MODE GRAPH RETRIEVAL: fixed comprehensive query, no LLM ---------
    # 2026-09-07: NEW - per explicit request: skip per-question NL2Cypher (an
    # LLM round-trip - see nl_to_cypher() above) for latency-sensitive "fast"
    # requests, while still traversing "all nodes for all relationships" for
    # accuracy, via ONE fixed, hand-written Cypher template instead of a
    # model-generated one. Seeds from the ENTITIES behind the question's
    # vector-search hits (reusing resolve_chunk_ids_to_entity_db_ids(), which
    # already exists for vector-seeded graph expansion - see
    # expand_via_related_entities() above) rather than trying to extract
    # keywords/entities from the question text without an LLM. Traverses
    # EVERY relationship type in the schema (not the curated
    # GRAPH_EXPANSION_REL_TYPES subset expand_via_related_entities() uses),
    # both directions, up to COMPREHENSIVE_TRAVERSAL_HOPS hops - broader
    # coverage than the curated expansion path, by design, since this
    # REPLACES NL2Cypher's per-question flexibility rather than supplementing
    # it. Returns the exact same {cypher, plain_rows, nodes, edges} shape
    # run_graph_retrieval() does, so retrieve() below can use either
    # interchangeably with no downstream changes.
    def comprehensive_graph_retrieval(self, seed_chunk_ids: list[str], trace=None) -> dict:
        """No LLM call. `seed_chunk_ids` are typically the top vector-search
        hits for the question (see retrieve()'s fast_mode branch) - this
        resolves them to their underlying entities, then walks outward from
        those entities across every relationship type in the graph, up to
        COMPREHENSIVE_TRAVERSAL_HOPS hops, returning every node and
        relationship encountered along the way (capped by
        COMPREHENSIVE_TRAVERSAL_MAX_ROWS for safety on a densely-connected
        seed set - same "blunt cap, applied in Cypher not after fetching
        everything" reasoning as expand_via_related_entities()'s
        max_neighbors).

        Returns result["cypher"] = None deliberately (not the query text) -
        unlike run_graph_retrieval()'s NL2Cypher-generated query, this
        template is fixed/identical for every question, so surfacing it in
        the UI's "graph query" debug panel would be noise, not useful
        per-question information. Empty `seed_chunk_ids` short-circuits to
        the same empty-result shape run_graph_retrieval() returns for
        "NO_QUERY", so a caller with no vector hits yet (first call in the
        pipeline) degrades identically to the NL2Cypher path's own
        no-match case."""
        result = {"cypher": None, "plain_rows": [], "nodes": {}, "edges": []}
        if not seed_chunk_ids:
            return result

        try:
            seed_db_ids = self.resolve_chunk_ids_to_entity_db_ids(seed_chunk_ids)
        except Exception as e:
            logger.warning(f"[graphrag] comprehensive_graph_retrieval: seed resolution failed, "
                            f"skipping graph traversal for this fast-mode call: {e}")
            return result
        if not seed_db_ids:
            return result

        cypher = """
            MATCH (seed) WHERE elementId(seed) IN $seed_ids
            MATCH path = (seed)-[r*1..$hops]-(m)
            WITH seed, relationships(path) AS rels, nodes(path) AS path_nodes
            UNWIND range(0, size(rels)-1) AS i
            WITH path_nodes[i] AS a, rels[i] AS rel, path_nodes[i+1] AS b
            RETURN DISTINCT a, rel, b
            LIMIT $limit
        """.replace("$hops", str(COMPREHENSIVE_TRAVERSAL_HOPS))  # driver can't parametrize a variable-length bound

        try:
            with neo4j_client.get_session() as session:
                db_result = session.run(
                    Query(cypher, timeout=GRAPH_QUERY_TIMEOUT_SECONDS),
                    seed_ids=seed_db_ids, limit=COMPREHENSIVE_TRAVERSAL_MAX_ROWS,
                )
                records = list(db_result)

            plain_rows = []
            for record in records:
                row_dict = {}
                for key, val in (("a", record["a"]), ("rel", record["rel"]), ("b", record["b"])):
                    if isinstance(val, Node):
                        n = vertex_to_node(val)
                        result["nodes"][n["db_id"]] = n
                        row_dict[key] = n["properties"]
                    elif isinstance(val, Relationship):
                        result["edges"].append(edge_to_link(val))
                        row_dict[key] = dict(val)
                    else:
                        row_dict[key] = val
                plain_rows.append(row_dict)
            result["plain_rows"] = plain_rows
            logger.info(f"[graphrag] comprehensive_graph_retrieval: {len(seed_db_ids)} seed entity(ies) -> "
                        f"{len(result['nodes'])} node(s), {len(result['edges'])} relationship(s) "
                        f"(all types, {COMPREHENSIVE_TRAVERSAL_HOPS}-hop, no LLM call).")
        except Exception as e:
            logger.warning(f"[graphrag] comprehensive_graph_retrieval query failed, continuing with "
                            f"vector-only results for this fast-mode call: {e}")

        return result

    # --- DOCUMENT DETAILS (for the "what document did this come from" panel) ---
    def get_document_details(self, chunk_ids: list[str]) -> dict[str, dict]:
        """For each chunk_id, looks up its source entity (via the chunk_id ->
        entity_id property framework/core/ingest/embedder.py stores on every
        Chunk node) and returns that entity's document-level provenance
        fields (document_id/document_name/source_type/authority_level/
        jurisdiction) plus its own citation - what the UI shows as "document
        details" alongside each cited chunk in an answer. Returns
        {chunk_id: {...}}; a chunk_id with no resolvable entity/document
        maps to an empty dict rather than being omitted, so callers can
        always safely `.get(chunk_id, {})`."""
        if not chunk_ids:
            return {}
        chunk_ids = [str(c) for c in chunk_ids]
        # 2026-08-28: semantic_graph mode's md-hydrated candidates aren't
        # real Chunk nodes (see md_hydration.py / Semantic_Graph_Retrieval_
        # Architecture.md) - their synthetic chunk_id is
        # "mdhydrate_<elementId>", so they need a DIFFERENT lookup (straight
        # to the entity node by elementId, no Chunk hop) rather than falling
        # through the normal Chunk-based query below and silently resolving
        # to nothing (which would degrade every semantic_graph citation to a
        # bare "chunk_id=..." with no document name - see
        # app.py's _chunk_source_label()).
        real_chunk_ids = [c for c in chunk_ids if not c.startswith("mdhydrate_")]
        synthetic_db_ids = {c: c[len("mdhydrate_"):] for c in chunk_ids if c.startswith("mdhydrate_")}

        details = {}
        if synthetic_db_ids:
            with neo4j_client.get_session() as session:
                synth_rows = list(session.run(
                    """
                    UNWIND $pairs AS pair
                    MATCH (e) WHERE elementId(e) = pair.db_id
                    RETURN pair.chunk_id AS chunk_id, e.document_id AS document_id,
                           e.document_name AS document_name, e.source_type AS source_type,
                           e.authority_level AS authority_level, e.jurisdiction AS jurisdiction,
                           e.full_citation AS entity_citation, labels(e) AS entity_labels
                    """,
                    pairs=[{"chunk_id": cid, "db_id": db_id} for cid, db_id in synthetic_db_ids.items()],
                ))
            for r in synth_rows:
                details[str(r["chunk_id"])] = {
                    "chunk_citation": r["entity_citation"],
                    "page": None,  # md-hydrated text has no fixed page - it's a live line-range/whole-file read
                    "document_id": r["document_id"],
                    "document_name": r["document_name"],
                    "source_type": r["source_type"],
                    "authority_level": r["authority_level"],
                    "jurisdiction": r["jurisdiction"],
                    "entity_citation": r["entity_citation"],
                    "entity_type": (r["entity_labels"] or [None])[0],
                }

        if real_chunk_ids:
            with neo4j_client.get_session() as session:
                rows = list(session.run(
                    """
                    MATCH (c:Chunk) WHERE c.chunk_id IN $chunk_ids
                    OPTIONAL MATCH (e {id: c.entity_id})
                    RETURN c.chunk_id AS chunk_id, c.full_citation AS chunk_citation,
                           c.page_number AS page_number,
                           e.document_id AS document_id, e.document_name AS document_name,
                           e.source_type AS source_type, e.authority_level AS authority_level,
                           e.jurisdiction AS jurisdiction, e.full_citation AS entity_citation,
                           labels(e) AS entity_labels
                    """,
                    chunk_ids=real_chunk_ids,
                ))
            for r in rows:
                details[str(r["chunk_id"])] = {
                    "chunk_citation": r["chunk_citation"],
                    "page": r["page_number"],
                    "document_id": r["document_id"],
                    "document_name": r["document_name"],
                    "source_type": r["source_type"],
                    "authority_level": r["authority_level"],
                    "jurisdiction": r["jurisdiction"],
                    "entity_citation": r["entity_citation"],
                    "entity_type": (r["entity_labels"] or [None])[0],
                }
        return details

    def get_entity_labels_for_chunks(self, chunk_ids: list[str]) -> dict[str, list[str]]:
        """2026-08-26: NEW. For each chunk_id, returns the Neo4j label(s) of
        its owning entity (via the same c.entity_id -> entity.id join
        get_document_details() above already uses). Added for
        framework.core.retrieve.category_retrieval's vector-fallback path: a
        category's vector search returns the nearest chunks by pure semantic
        similarity regardless of what KIND of document they came from, so
        without this a "Case Law Hierarchy" vector fallback could just as
        easily surface an Income Tax Act clause - checking the owning
        entity's real label against the category's expected node_labels is
        what makes "vector fallback" mean "the closest matching content
        that's actually of this category," not "the closest matching
        content, period." A chunk_id with no resolvable entity maps to an
        empty list."""
        if not chunk_ids:
            return {}
        chunk_ids = [str(c) for c in chunk_ids]
        # 2026-08-28: same "mdhydrate_<elementId>" synthetic-id handling as
        # get_document_details() above - semantic_graph mode's candidates
        # aren't real Chunk nodes, so the normal Chunk-based query below
        # would resolve them to an empty label list, which would then make
        # category_retrieval.py's node_labels filter wrongly discard EVERY
        # semantic_graph candidate regardless of whether it's actually the
        # right kind of entity.
        real_chunk_ids = [c for c in chunk_ids if not c.startswith("mdhydrate_")]
        synthetic_db_ids = {c: c[len("mdhydrate_"):] for c in chunk_ids if c.startswith("mdhydrate_")}

        result: dict[str, list[str]] = {}
        if synthetic_db_ids:
            with neo4j_client.get_session() as session:
                synth_rows = list(session.run(
                    "UNWIND $pairs AS pair "
                    "MATCH (e) WHERE elementId(e) = pair.db_id "
                    "RETURN pair.chunk_id AS chunk_id, labels(e) AS entity_labels",
                    pairs=[{"chunk_id": cid, "db_id": db_id} for cid, db_id in synthetic_db_ids.items()],
                ))
            for r in synth_rows:
                result[str(r["chunk_id"])] = r["entity_labels"] or []

        if real_chunk_ids:
            with neo4j_client.get_session() as session:
                rows = list(session.run(
                    "MATCH (c:Chunk) WHERE c.chunk_id IN $chunk_ids "
                    "OPTIONAL MATCH (e {id: c.entity_id}) "
                    "RETURN c.chunk_id AS chunk_id, labels(e) AS entity_labels",
                    chunk_ids=real_chunk_ids,
                ))
            for r in rows:
                result[str(r["chunk_id"])] = r["entity_labels"] or []
        return result

    def get_chunk_properties_for_entity(self, entity_id: str) -> list[dict]:
        """Returns the full properties of every Chunk node linked to this
        entity (via the same c.entity_id == entity's own `id` property
        pattern get_document_details() above already uses) - i.e. the
        Chunk-node-only metadata (chunk_id, chunk_type, element_type,
        short_title, page_number, file_type, etc. - see domain_packs/tax/
        config.py's CHUNK_METADATA_KEYS) that does NOT get copied onto the
        entity's own node properties, so the UI's node-hover popup can show
        it ALONGSIDE the entity's own properties instead of only ever
        seeing whichever of these happened to also be stamped on the
        entity. Usually a single Chunk per entity - a list because
        `source_chunk_ids` is occasionally more than one."""
        if not entity_id:
            return []
        with neo4j_client.get_session() as session:
            rows = list(session.run(
                "MATCH (c:Chunk) WHERE c.entity_id = $entity_id RETURN c",
                entity_id=entity_id,
            ))
        return [dict(r["c"]) for r in rows]

    def get_entities_for_chunks(self, chunk_ids: list[str]) -> list[dict]:
        """Returns the entity node(s) linked to each of these chunk_ids, in
        the same {db_id, id, label, properties} shape vertex_to_node()
        produces for graph-matched nodes - so the webapp can show "the
        nodes actually used to answer this question" as a real graph view,
        not just the NL2Cypher-matched subgraph (retrieve()'s own "nodes"/
        "edges", which is empty whenever a question was answered purely via
        vector search - the common case for a broad/semantic question with
        no exact citation to match on)."""
        if not chunk_ids:
            return []
        chunk_ids = [str(c) for c in chunk_ids]
        with neo4j_client.get_session() as session:
            rows = list(session.run(
                "MATCH (c:Chunk) WHERE c.chunk_id IN $chunk_ids "
                "OPTIONAL MATCH (e {id: c.entity_id}) "
                "WITH DISTINCT e WHERE e IS NOT NULL RETURN e",
                chunk_ids=chunk_ids,
            ))
        return [vertex_to_node(r["e"]) for r in rows]

    def get_edges_among_nodes(self, node_db_ids: list[str]) -> list[dict]:
        """Returns every relationship directly connecting two nodes ALREADY
        in the given set (e.g. PART_OF between a Section and its
        SubSections). Needed because get_entities_for_chunks() above only
        ever returns bare NODES (vector search matches chunks, not
        relationships) - without this, a question answered mostly/entirely
        via vector search would show a graph view full of correctly-labeled
        but completely disconnected dots, even though the real PART_OF
        hierarchy between them exists in Neo4j (e.g. "what deductions are
        allowed in section 80C" pulling in Section_80C plus a dozen of its
        SubSections with no lines between them - reported directly by a
        user). This is a plain node-set intersection query, independent of
        whichever NL2Cypher query (if any) ran for this question."""
        if not node_db_ids or len(node_db_ids) < 2:
            return []
        with neo4j_client.get_session() as session:
            rows = list(session.run(
                "MATCH (a)-[r]-(b) WHERE elementId(a) IN $ids AND elementId(b) IN $ids "
                "AND elementId(a) < elementId(b) "
                "RETURN DISTINCT r",
                ids=node_db_ids,
            ))
        return [edge_to_link(r["r"]) for r in rows]

    # --- PARENT-CHILD CONTEXT EXPANSION ---------------------------------------
    # "Container" labels this hierarchy naturally bottoms out at when walking
    # PART_OF upward - Section for the Act (SubSection/Clause/SubClause/
    # Proviso/Explanation all eventually PART_OF a Section), GenericDocument
    # for the generic-document pipeline (DocumentSection -[:PART_OF]->
    # GenericDocument). Not domain-configurable (yet) - both come from
    # framework-level PART_OF wiring (chunking.py / extract_generic.py), not
    # anything a third domain pack would need to override to get sane
    # behavior; the topmost-ancestor fallback below covers any hierarchy
    # shape that doesn't use either label.
    PARENT_CONTAINER_LABELS = {"Section", "GenericDocument"}

    def get_parent_section_context(self, chunk_ids: list[str]) -> dict[str, dict]:
        """Parent-child chunking's retrieval-time half: the fine-grained
        Chunk (SubSection/Clause/SubClause/Proviso/Explanation - whatever
        chunking.py split at) stays the small, precise unit vector search
        and the reranker actually match on, but answer generation should
        see the FULL surrounding provision, not just an isolated clause
        ripped out of context. For each chunk_id, walks its source entity's
        PART_OF chain (up to 6 levels) and returns the nearest ancestor
        found that's labeled one of PARENT_CONTAINER_LABELS (falling back
        to the topmost ancestor in the chain if none match, so this still
        does something sensible for a hierarchy shape it doesn't know
        about) - that ancestor's OWN full `hasText` is the "parent block."
        Returns {} for a chunk_id whose entity has no PART_OF chain at all
        (already top-level, or an unlinked/legacy entity) - callers should
        treat a missing key as "no extra parent context beyond the chunk's
        own text," not an error."""
        if not chunk_ids:
            return {}
        chunk_ids = [str(c) for c in chunk_ids]
        result: dict[str, dict] = {}
        with neo4j_client.get_session() as session:
            ent_rows = list(session.run(
                "MATCH (c:Chunk) WHERE c.chunk_id IN $chunk_ids "
                "RETURN c.chunk_id AS chunk_id, c.entity_id AS entity_id",
                chunk_ids=chunk_ids,
            ))
            chunk_to_entity = {str(r["chunk_id"]): r["entity_id"] for r in ent_rows if r["entity_id"]}
            if not chunk_to_entity:
                return {}

            entity_to_chunks: dict[str, list[str]] = {}
            for cid, eid in chunk_to_entity.items():
                entity_to_chunks.setdefault(eid, []).append(cid)

            for entity_id in sorted(entity_to_chunks):
                rows = list(session.run(
                    """
                    MATCH (e {id: $eid})
                    OPTIONAL MATCH p = (e)-[:PART_OF*1..6]->(anc)
                    RETURN anc.id AS anc_id, anc.hasText AS anc_text,
                           anc.full_citation AS anc_citation, labels(anc) AS anc_labels,
                           length(p) AS depth
                    ORDER BY depth
                    """,
                    eid=entity_id,
                ))
                ancestors = [r for r in rows if r["anc_id"] is not None]
                if not ancestors:
                    continue  # no PART_OF chain - the chunk's own text is already the fullest available
                chosen = next(
                    (r for r in ancestors if set(r["anc_labels"] or []) & self.PARENT_CONTAINER_LABELS),
                    ancestors[-1],
                )
                for cid in entity_to_chunks[entity_id]:
                    result[cid] = {
                        "parent_id": chosen["anc_id"],
                        "parent_citation": chosen["anc_citation"],
                        "parent_label": (chosen["anc_labels"] or [None])[0],
                        "parent_text": chosen["anc_text"],
                    }
        return result

    # --- PATH 2: vector similarity search ------------------------------------
    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        if not chunk_ids:
            return []
        chunk_ids = [str(c) for c in chunk_ids]
        with neo4j_client.get_session() as session:
            rows = list(session.run(
                "MATCH (c:Chunk) WHERE c.chunk_id IN $chunk_ids RETURN c.chunk_id AS chunk_id, c.content AS text",
                chunk_ids=chunk_ids,
            ))
        # Sort key MUST be a consistent type across every row - chunk_id is
        # purely numeric for some ingestion paths (e.g. Act PDFs, "42") and
        # non-numeric for others (e.g. case-law chunks, "CaseName_para12").
        # A question that retrieves BOTH kinds in one call (as any question
        # touching both a Section and case law does) used to produce a
        # sort key list mixing int and str, which Python 3 refuses to
        # compare ('<' not supported between int and str), crashing every
        # such /api/ask call. Using a (is_non_numeric, value) tuple keeps
        # numeric chunk_ids sorted numerically among themselves, non-numeric
        # ones sorted alphabetically among themselves, and numeric IDs
        # ordered before non-numeric ones - all without ever comparing an
        # int to a str.
        rows.sort(key=lambda r: (
            (False, int(r["chunk_id"])) if str(r["chunk_id"]).isdigit()
            else (True, str(r["chunk_id"]))
        ))
        return [{"chunk_id": r["chunk_id"], "text": r["text"]} for r in rows]

    def vector_search_chunks(self, question: str, top_k: int = 10) -> list[dict]:
        embedding = _get_embed_model().encode(question, normalize_embeddings=True).tolist()
        with neo4j_client.get_session() as session:
            rows = list(session.run(
                """
                CALL db.index.vector.queryNodes($index_name, $top_k, $embedding)
                YIELD node, score
                RETURN node.chunk_id AS chunk_id, node.content AS text, score AS similarity
                ORDER BY score DESC, chunk_id ASC
                """,
                # 2026-09-05: NEW - explicit ORDER BY, per an explicit answer-
                # inconsistency investigation. db.index.vector.queryNodes()
                # (Neo4j's HNSW-based ANN index) yields rows in whatever order
                # its internal graph traversal happens to visit them - NOT
                # guaranteed stable across repeated identical calls, especially
                # for near-tied similarity scores. Without an explicit ORDER BY,
                # two calls for the literal same question could return the
                # same set of chunk_ids in a different order, which then
                # propagates into a different reranker tie-break winner (see
                # reranker.py's matching 2026-09-05 tie-break fix) and, from
                # there, a different final top_chunks / generated answer, even
                # though nothing about the QUESTION changed. `chunk_id ASC` as
                # the tie-break (not just `score DESC` alone) makes row order
                # for exactly-tied scores deterministic too, matching
                # reranker.py's own ascending-chunk_id tie-break convention.
                index_name=VECTOR_INDEX_NAME, top_k=top_k, embedding=embedding,
            ))
        return [{"chunk_id": r["chunk_id"], "text": r["text"], "similarity": float(r["similarity"])} for r in rows]

    def semantic_entry_point_search(self, question: str, top_k: int = 15,
                                     semantic_graph_trace: dict | None = None) -> list[dict]:
        """The "semantic_graph" retrieval mode's replacement for
        vector_search_chunks() - see Semantic_Graph_Retrieval_Architecture.md.
        Searches per-node semantic entry-point indexes (CaseLaw via
        case_law_entry_points.py, StatutoryProvision/Section via
        section_entry_points.py, Circular via circular_entry_points.py,
        TaxTreaty via treaty_entry_points.py) to find topically relevant
        NODES, then hydrates each one's actual text LIVE from its converted
        .md source file (md_hydration.py) instead of returning pre-stored
        Chunk text.

        Returns the SAME shape vector_search_chunks() does
        (chunk_id/text/similarity) so every downstream consumer (reranker,
        _build_context_parts, get_document_details - extended above to
        resolve these synthetic ids) needs no changes to handle either mode.
        chunk_id is synthesized as "mdhydrate_<elementId>" - never collides
        with a real Chunk's chunk_id, and carries the entity's elementId for
        get_document_details()'s synthetic-id lookup.

        `semantic_graph_trace` (2026-08-29: NEW, optional): a caller-owned
        dict with "files_used"/"files_not_found" list keys - if given, this
        method APPENDS to those lists (never replaces them), so a single
        trace dict can be shared across multiple calls (the main retrieve()
        pass plus category_retrieval.py's per-category fallback calls) to
        build ONE combined "which .md files did this whole answer touch"
        list for the UI's evidence trail. `None` (the default) skips this
        bookkeeping entirely - existing callers that don't care about it are
        unaffected."""
        # Imported here (not at module level) to avoid a hard import-time
        # dependency from graphrag.py (framework, domain-agnostic) onto
        # domain_packs/tax-specific modules for a mode most callers won't
        # use - mirrors how category_retrieval.py's case-law special case
        # also imports case_law_entry_points lazily, inside the function.
        from framework.core.retrieve import case_law_entry_points as clep
        from framework.core.retrieve import section_entry_points as sep
        from framework.core.retrieve import circular_entry_points as cep
        from framework.core.retrieve import treaty_entry_points as tep
        from framework.core.retrieve import md_hydration

        entry_points: list[dict] = []
        try:
            entry_points += clep.search_case_law_entry_points(question, top_k=top_k)
        except Exception as e:
            logger.warning(f"[graphrag] semantic_entry_point_search: case-law entry-point search failed: {e}")
        try:
            entry_points += sep.search_section_entry_points(question, top_k=top_k)
        except Exception as e:
            logger.warning(f"[graphrag] semantic_entry_point_search: section entry-point search failed: {e}")
        try:
            entry_points += cep.search_circular_entry_points(question, top_k=top_k)
        except Exception as e:
            logger.warning(f"[graphrag] semantic_entry_point_search: circular entry-point search failed: {e}")
        try:
            entry_points += tep.search_treaty_entry_points(question, top_k=top_k)
        except Exception as e:
            logger.warning(f"[graphrag] semantic_entry_point_search: treaty entry-point search failed: {e}")

        # 2026-08-29: NEW - console-visible file-scan log. Previously the ONLY
        # place a user could see which .md files this search actually touched
        # was the UI's evidence trail (semantic_graph_trace, appended below) -
        # the raw server/console log said nothing about file names at all,
        # only the 4 index-search try/excepts above. Explicit user request:
        # "list the files scanned in the log ... so that we can validate."
        # Each hydration attempt (hit or miss) now gets its own INFO line
        # here, PLUS one summary line at the end - so `grep "md file scan"` on
        # the server log alone is enough to validate a run without opening
        # the app.
        results = []
        hydrated_count = 0
        skipped_count = 0
        for ep in entry_points:
            name = ep.get("full_citation") or ep.get("case_name") or ep.get("document_id") or ep.get("db_id", "")
            source_md_path = ep.get("sourceMdPath", "")
            text, note = md_hydration.hydrate_text(
                source_md_path, ep.get("line_start"), ep.get("line_end"),
            )
            if not text:
                skipped_count += 1
                logger.info(f"[graphrag] md file scan: SKIP  '{name}' - "
                            f"{source_md_path or '(no sourceMdPath stored on this entity)'} - {note}")
                if semantic_graph_trace is not None:
                    semantic_graph_trace.setdefault("files_not_found", []).append({
                        "name": name,
                        "path": source_md_path or "(no sourceMdPath stored on this entity)",
                        "reason": note or "unknown",
                    })
                continue
            hydrated_count += 1
            line_start, line_end = ep.get("line_start"), ep.get("line_end")
            span = "whole file" if line_start is None or line_end is None else f"lines {line_start}-{line_end}"
            logger.info(f"[graphrag] md file scan: READ  '{name}' - {source_md_path} ({span}, "
                        f"{len(text):,} chars, similarity={float(ep.get('similarity', 0.0)):.3f})"
                        + (f" - {note}" if note else ""))
            if semantic_graph_trace is not None:
                semantic_graph_trace.setdefault("files_used", []).append({
                    "name": name,
                    "path": source_md_path,
                    "span": span,
                    "similarity": round(float(ep.get("similarity", 0.0)), 3),
                    "note": note,  # e.g. a truncation warning on an oversized whole-file read - "" when clean
                })
            results.append({
                "chunk_id": f"mdhydrate_{ep['db_id']}",
                "text": text,
                "similarity": float(ep.get("similarity", 0.0)),
            })
        logger.info(f"[graphrag] md file scan SUMMARY: {len(entry_points)} entry-point hit(s) -> "
                    f"{hydrated_count} file(s) read, {skipped_count} not found/skipped.")
        return results

    def get_chunk_ids_via_graph_traversal(self, node_db_ids: list[str]) -> list[str]:
        """Walks (entity)-[:hasSourceChunk]->(Chunk) directly, PLUS the
        PART_OF descendant hierarchy (child -[:PART_OF]-> parent) down to 6
        levels, so a "wrapper" entity with no Chunk of its own (e.g. a
        Section that only has subsections) still surfaces its descendants'
        chunks. Capped at MAX_DESCENDANT_CHUNKS. `PART_OF`/`hasSourceChunk`
        are framework-level edge names created by graph_loader.py - not
        domain-specific."""
        if not node_db_ids:
            return []

        with neo4j_client.get_session() as session:
            direct_rows = list(session.run(
                """
                MATCH (n)-[:hasSourceChunk]->(c:Chunk)
                WHERE elementId(n) IN $ids
                RETURN DISTINCT c.chunk_id AS chunk_id
                """,
                ids=node_db_ids,
            ))
            descendant_rows = list(session.run(
                """
                MATCH (n)<-[:PART_OF*1..6]-(d)-[:hasSourceChunk]->(c:Chunk)
                WHERE elementId(n) IN $ids
                RETURN DISTINCT c.chunk_id AS chunk_id
                LIMIT $limit
                """,
                ids=node_db_ids, limit=MAX_DESCENDANT_CHUNKS,
            ))

        ids = set()
        for row in list(direct_rows) + list(descendant_rows):
            val = row["chunk_id"]
            if val is not None:
                ids.add(str(val))
        return sorted(ids)

    def resolve_chunk_ids_to_entity_db_ids(self, chunk_ids: list[str]) -> list[str]:
        """2026-09-03: NEW - the entity-resolution half of vector-seeded
        graph expansion. Same c.entity_id -> entity.id join
        get_document_details()/get_entity_labels_for_chunks() already use,
        but returning the entity's elementId (for a follow-up Cypher MATCH
        by elementId, as expand_via_related_entities() does) rather than its
        display properties. "mdhydrate_"-prefixed synthetic ids (semantic_
        graph mode) are skipped - not applicable here since this is only
        called for retrieval_mode="hybrid" (see retrieve() below), which
        never produces synthetic chunk ids."""
        if not chunk_ids:
            return []
        with neo4j_client.get_session() as session:
            rows = list(session.run(
                """
                MATCH (c:Chunk) WHERE c.chunk_id IN $chunk_ids
                MATCH (e {id: c.entity_id})
                RETURN DISTINCT elementId(e) AS db_id
                """,
                chunk_ids=[str(c) for c in chunk_ids],
            ))
        return [r["db_id"] for r in rows if r["db_id"]]

    def expand_via_related_entities(self, node_db_ids: list[str], max_neighbors: int = 15) -> list[str]:
        """2026-09-03: NEW - "vector hit -> graph traversal" expansion (see
        Semantic_Graph_Retrieval_Architecture.md's sibling doc on this
        investigation, and retrieval_router.py's module docstring for why
        this only runs for questions the router classifies "lineage").
        Walks ONE hop from `node_db_ids` across GRAPH_EXPANSION_REL_TYPES -
        edges that connect an authority to a DIFFERENT, substantively
        related authority - and returns the neighboring entities' db_ids.
        NOT chunk ids: the caller (retrieve() below) feeds this into
        get_chunk_ids_via_graph_traversal() to actually get their text,
        exactly like it already does for NL2Cypher-matched entities, so a
        neighbor found this way is retrieved/reranked identically to any
        other candidate rather than force-included.

        `max_neighbors` caps total neighbors returned across ALL seeds
        combined (not per-seed) - a deliberately blunt cap so a
        heavily-cross-referenced case (dozens of citing/cited cases) can't
        blow up context size or latency; the LIMIT is applied in Cypher, not
        after fetching everything, so it's cheap even on a large seed set."""
        if not node_db_ids:
            return []
        with neo4j_client.get_session() as session:
            rows = list(session.run(
                """
                MATCH (n)-[r]-(m)
                WHERE elementId(n) IN $ids AND type(r) IN $rel_types
                RETURN DISTINCT elementId(m) AS db_id
                LIMIT $limit
                """,
                ids=node_db_ids, rel_types=GRAPH_EXPANSION_REL_TYPES, limit=max_neighbors,
            ))
        return [r["db_id"] for r in rows if r["db_id"]]

    def extract_source_chunk_ids_legacy(self, nodes: list[dict]) -> list[str]:
        """Legacy fallback for graphs built before Chunk nodes/hasSourceChunk
        edges existed - pulls source_chunk_ids straight off node properties."""
        ids = set()
        for n in nodes:
            raw = (n.get("properties") or {}).get("source_chunk_ids", "")
            if raw:
                for part in str(raw).split(","):
                    part = part.strip()
                    if part:
                        ids.add(part)
        return sorted(ids)

    # --- CITATION-PRECISE LOOKUP ----------------------------------------------
    def extract_citation(self, question: str) -> str | None:
        rules = self.domain_config.citation_rules
        if rules is None:
            return None
        m = rules.free_text_pattern.search(question)
        if not m:
            return None
        return rules.build_citation(m.group(1), m.group(2) or "")

    def normalize_citation_query(self, q: str) -> str | None:
        rules = self.domain_config.citation_rules
        if rules is None:
            return None
        m = rules.query_pattern.match(q)
        if not m:
            return None
        return rules.build_citation(m.group(1), m.group(2) or "")

    def retrieve_by_citation(self, citation: str) -> dict | None:
        with neo4j_client.get_session() as session:
            record = session.run(
                """
                MATCH (c:Chunk {full_citation: $citation})
                RETURN c.chunk_id AS chunk_id, c.content AS text, c.full_citation AS full_citation,
                       c.short_title AS short_title, c.chunk_type AS chunk_type,
                       c.element_type AS element_type, c.section AS section, c.subsection AS subsection,
                       c.clause AS clause, c.sub_clause AS sub_clause, c.entity_id AS entity_id,
                       c.has_proviso AS has_proviso, c.has_explanation AS has_explanation, c.chapter AS chapter
                LIMIT 1
                """,
                citation=citation,
            ).single()
        if not record:
            return None
        return dict(record)

    def find_cross_references(self, entity_id: str) -> dict:
        with neo4j_client.get_session() as session:
            outgoing = list(session.run(
                "MATCH (n {id: $entity_id})-[:REFERS_TO]->(t) RETURN t.id AS id, t.full_citation AS full_citation",
                entity_id=entity_id,
            ))
            incoming = list(session.run(
                "MATCH (s)-[:REFERS_TO]->(n {id: $entity_id}) RETURN s.id AS id, s.full_citation AS full_citation",
                entity_id=entity_id,
            ))
        return {
            "outgoing": [{"id": r["id"], "full_citation": r["full_citation"]} for r in outgoing],
            "incoming": [{"id": r["id"], "full_citation": r["full_citation"]} for r in incoming],
        }

    def build_cross_reference_graph(self, entity_id: str, depth: int = 2) -> dict:
        depth = int(depth)
        with neo4j_client.get_session() as session:
            rows = list(session.run(
                f"""
                MATCH (n {{id: $entity_id}})-[:REFERS_TO*1..{depth}]->(t)
                RETURN DISTINCT t.id AS id, t.full_citation AS full_citation
                """,
                entity_id=entity_id,
            ))
        nodes = [{"id": entity_id}] + [{"id": r["id"], "full_citation": r["full_citation"]} for r in rows]
        return {"nodes": nodes}

    def _parse_amendment_info(self, raw) -> dict | None:
        if raw is None:
            return None
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                return None
        if isinstance(raw, dict):
            return raw
        return None

    def get_amendment_history(self, entity_id: str) -> dict | None:
        with neo4j_client.get_session() as session:
            record = session.run(
                "MATCH (n {id: $entity_id}) RETURN n.full_citation AS full_citation, n.amendment_info AS amendment_info",
                entity_id=entity_id,
            ).single()
        if not record:
            return None
        return {
            "entity_id": entity_id,
            "full_citation": record["full_citation"],
            "amendment": self._parse_amendment_info(record["amendment_info"]),
        }

    def filter_by_effective_date(self, chunks: list[dict], as_of: str) -> list[dict]:
        def keep(chunk: dict) -> bool:
            amendment = self._parse_amendment_info(chunk.get("amendment_info"))
            if not amendment:
                return True
            effective_from = amendment.get("effective_from")
            if not effective_from:
                return True
            return effective_from <= as_of

        return [c for c in chunks if keep(c)]

    # --- METADATA LAYER --------------------------------------------------------
    def apply_metadata_filters(self, nodes: list[dict], filters: dict | None) -> list[dict]:
        if not filters:
            return nodes

        def keep(node: dict) -> bool:
            props = node.get("properties", {}) or {}
            for key in ("status", "jurisdiction", "authority_level", "file_type"):
                wanted = filters.get(key)
                actual = props.get(key)
                if wanted and actual and actual != wanted:
                    return False
            # List-valued retrieval metadata (legal_concepts, statutes_cited,
            # other_act_references, treaty_references, cases_cited,
            # defines_terms) - membership rather than equality, and
            # case-insensitive since these are LLM-extracted free text (e.g.
            # a question filter of "demerger" should match a stored
            # "Demerger" or "fast-track-demerger" tag list entry containing
            # it, not require an exact-case exact-string match).
            for key in ("legal_concepts", "statutes_cited", "other_act_references",
                        "treaty_references", "cases_cited", "defines_terms", "forum_level"):
                wanted = filters.get(key)
                if not wanted:
                    continue
                actual = props.get(key)
                if not actual:
                    return False
                wanted_l = wanted.lower()
                if isinstance(actual, list):
                    if not any(wanted_l in str(v).lower() for v in actual):
                        return False
                elif wanted_l not in str(actual).lower():
                    return False
            as_of = filters.get("as_of")
            if as_of:
                eff_from, eff_to = props.get("effective_from"), props.get("effective_to")
                if eff_from and eff_from > as_of:
                    return False
                if eff_to and eff_to < as_of:
                    return False
            return True

        return [n for n in nodes if keep(n)]

    # --- ORCHESTRATOR ------------------------------------------------------------
    def retrieve(self, question: str, intent: str | None = None, metadata_filters: dict | None = None,
                 trace=None, retrieval_mode: str = "hybrid", semantic_graph_trace: dict | None = None,
                 fast_mode: bool = False) -> dict:
        """`semantic_graph_trace` (2026-08-29: NEW, optional): pass a dict
        here (e.g. `{}`) to have it populated with "files_used"/
        "files_not_found" lists from this call's semantic_entry_point_search()
        - see that method's docstring. Pass the SAME dict into
        category_retrieval.run_category_retrieval()'s own
        `semantic_graph_trace` param to combine both into one list covering
        the whole /api/ask call, which is what app.py does. Ignored (no-op)
        outside retrieval_mode="semantic_graph".

        `fast_mode` (2026-09-07: NEW) - per explicit request for a low-latency
        path. Independent of `retrieval_mode` (works alongside "hybrid";
        meaningless combined with "vector_only", which already skips graph
        retrieval entirely, or "semantic_graph", which has its own
        LLM-free-by-design entry-point search - fast_mode is a no-op for
        either). When True:
          - `intent` is ignored entirely and `config` is forced to the
            broadest retrieval settings (same numbers as the domain's
            CONCEPTUAL intent config: widest vector_top_k, vector_mode=
            "always") - the caller is expected to have skipped
            detect_intent() upstream (app.py's fast path does exactly this)
            to remove that LLM round-trip, so there is no classified intent
            to key off here.
          - Graph retrieval runs comprehensive_graph_retrieval() (a FIXED
            Cypher template, no LLM) instead of run_graph_retrieval()
            (NL2Cypher, one LLM round-trip) - see that method's docstring.
            Since it seeds from vector-search hit chunks rather than the
            question text directly, vector search is run FIRST in this mode
            (reversing the normal graph-then-vector order below) so its top
            hits are available to seed the traversal.
          - Everything else (citation fast-path, PII masking, metadata
            filtering, candidate-chunk assembly, retrieval_scope reporting)
            is unchanged - fast_mode only changes WHERE the graph-traversal
            seed/query comes from, not the shape of what retrieve() returns.

        `retrieval_mode`:
          - "hybrid" (default) - today's behavior unchanged: NL2Cypher graph
            traversal + exact-citation lookup + vector search, merged.
          - "vector_only" - 2026-08-26: Skips NL2Cypher/graph traversal
            and the exact-citation shortcut entirely; answers come ONLY from
            vector_search_chunks(). Added so the Ask screen's "Simple RAG"
            mode can give an honest, graph-free baseline to compare answer
            quality against the normal "KG + RAG" hybrid path - the point is
            an apples-to-apples ablation, so nothing graph-derived (Cypher
            results, PART_OF-traversed chunks, exact citation matches) may
            leak in under this mode.
          - "semantic_graph" - 2026-08-28: NEW. See Semantic_Graph_Retrieval_
            Architecture.md. Keeps NL2Cypher/citation-match/PART_OF traversal
            exactly as "hybrid" does (this mode is graph-HEAVY, not
            graph-free), but replaces vector_search_chunks()'s ANN search
            over pre-stored Chunk.embedding with
            semantic_entry_point_search(): a small per-NODE summary-embedding
            index finds topically relevant entities, then their text is
            hydrated LIVE from the converted .md source file (md_hydration.py)
            rather than returned from a Chunk node captured at ingestion
            time."""
        # 2026-09-07: fast_mode forces the broadest retrieval config (no
        # classified `intent` to key off - the caller is expected to have
        # skipped detect_intent() upstream) - see this method's docstring.
        config = (
            self.CONCEPTUAL_STYLE_FALLBACK_CONFIG if fast_mode
            else self.intent_retrieval_config.get(intent, self.default_retrieval_config)
        )
        vector_only = retrieval_mode == "vector_only"
        semantic_graph = retrieval_mode == "semantic_graph"

        citation = None if vector_only else self.extract_citation(question)
        citation_chunk = self.retrieve_by_citation(citation) if citation else None
        if citation and not citation_chunk:
            logger.info(f"[graphrag] Question named citation {citation!r} but no chunk matched it exactly - "
                        f"falling back to graph/vector retrieval.")

        # 2026-09-07: NEW - fast_mode runs vector search FIRST (instead of
        # after graph retrieval, as the non-fast order below does) because
        # comprehensive_graph_retrieval() (the fixed, LLM-free Cypher
        # template - see that method's docstring) seeds its traversal from
        # vector-hit chunks' entities, not from the question text directly.
        # semantic_graph/vector_only ignore fast_mode entirely (see this
        # method's docstring) so they keep their existing order untouched.
        fast_vector_chunks: list[dict] | None = None
        if fast_mode and not vector_only and not semantic_graph:
            try:
                fast_vector_chunks = self.vector_search_chunks(question, top_k=config["vector_top_k"])
            except Exception as e:
                logger.warning(f"[graphrag] fast_mode: early vector search failed, "
                                f"comprehensive_graph_retrieval will have no seed chunks: {e}")
                fast_vector_chunks = []

        if vector_only:
            graph_result = {"cypher": None, "plain_rows": [], "nodes": {}, "edges": []}
        elif fast_mode and not semantic_graph:
            seed_chunk_ids = [c["chunk_id"] for c in (fast_vector_chunks or [])]
            graph_result = self.comprehensive_graph_retrieval(seed_chunk_ids, trace=trace)
        else:
            graph_result = self.run_graph_retrieval(question, trace=trace)
        matched_nodes = list(graph_result["nodes"].values())
        pre_filter_count = len(matched_nodes)
        matched_nodes = self.apply_metadata_filters(matched_nodes, metadata_filters)

        graph_chunk_ids = [] if vector_only else self.get_chunk_ids_via_graph_traversal(
            [n["db_id"] for n in matched_nodes])
        if not graph_chunk_ids and not vector_only:
            graph_chunk_ids = self.extract_source_chunk_ids_legacy(matched_nodes)

        graph_chunks = self.get_chunks_by_ids(graph_chunk_ids)

        graph_found_something = bool(graph_chunks or citation_chunk)

        run_vector = vector_only or (
            config["vector_mode"] == "always"
            or (config["vector_mode"] == "fallback_only" and not graph_found_something)
        )

        if fast_vector_chunks is not None:
            # Already fetched above (to seed the graph traversal) - reuse
            # rather than querying the vector index a second time for the
            # identical question/top_k.
            similar_chunks = fast_vector_chunks if run_vector else []
        elif run_vector and semantic_graph:
            try:
                similar_chunks = self.semantic_entry_point_search(
                    question, top_k=config["vector_top_k"], semantic_graph_trace=semantic_graph_trace)
            except Exception as e:
                # Best-effort, same "degrade, don't crash the whole answer"
                # reasoning as category_retrieval.py's case-law special case -
                # a bug in the new entry-point/hydration path shouldn't take
                # down /api/ask entirely when the graph/citation paths above
                # may already have found something useful.
                logger.warning(f"[graphrag] semantic_entry_point_search failed, continuing with "
                                f"graph/citation results only: {e}")
                similar_chunks = []
        elif run_vector:
            similar_chunks = self.vector_search_chunks(question, top_k=config["vector_top_k"])
        else:
            similar_chunks = []

        # 2026-09-03: NEW - "vector hit -> graph traversal" expansion (see
        # GRAPH_EXPANSION_REL_TYPES above + retrieval_router.py). Only for
        # "hybrid" - not vector_only (deliberately graph-free ablation
        # baseline, see that mode's own docstring) and not semantic_graph
        # (already graph-heavy via NL2Cypher/PART_OF, and its candidates are
        # md-hydrated synthetic ids with no real Chunk.entity_id to
        # resolve). Gated on retrieval_router.route() == "lineage" - a
        # 1-hop relationship walk buys little for a broad definitional
        # question, so skip its extra Neo4j round-trip otherwise. Also
        # skipped entirely under fast_mode - comprehensive_graph_retrieval()
        # above already does a broader (all-relationship-type, not curated
        # GRAPH_EXPANSION_REL_TYPES) traversal from the same vector seeds, so
        # this would be redundant extra latency for fast_mode callers.
        graph_expansion_chunks: list[dict] = []
        graph_expansion_route = None
        if not vector_only and not semantic_graph and not fast_mode:
            router_cfg = retrieval_router.load_config()
            graph_expansion_route = retrieval_router.route(question, router_cfg)
            if graph_expansion_route == "lineage":
                try:
                    seed_top_k = router_cfg["expansion_seed_top_k"]
                    vector_seed_chunk_ids = [
                        c["chunk_id"] for c in
                        sorted(similar_chunks, key=lambda c: c["similarity"] or 0, reverse=True)[:seed_top_k]
                    ]
                    expansion_seed_db_ids = list({
                        *[n["db_id"] for n in matched_nodes],
                        *self.resolve_chunk_ids_to_entity_db_ids(vector_seed_chunk_ids),
                    })
                    neighbor_db_ids = self.expand_via_related_entities(
                        expansion_seed_db_ids, max_neighbors=router_cfg["expansion_max_neighbors"])
                    expansion_chunk_ids = self.get_chunk_ids_via_graph_traversal(neighbor_db_ids)
                    graph_expansion_chunks = self.get_chunks_by_ids(expansion_chunk_ids)
                except Exception as e:
                    # Best-effort, same "degrade, don't crash the whole
                    # answer" reasoning as semantic_entry_point_search above -
                    # a bug in this new expansion step shouldn't take down
                    # /api/ask when graph/vector retrieval already succeeded.
                    logger.warning(f"[graphrag] graph-expansion step failed, continuing without it: {e}")
                    graph_expansion_chunks = []

        seen_ids = {c["chunk_id"] for c in graph_chunks}
        citation_candidate = []
        if citation_chunk and citation_chunk["chunk_id"] not in seen_ids:
            citation_candidate = [{
                "chunk_id": citation_chunk["chunk_id"], "text": citation_chunk["text"],
                "source": "citation", "similarity": None,
            }]
            seen_ids.add(citation_chunk["chunk_id"])

        expansion_candidates = []
        for c in graph_expansion_chunks:
            if c["chunk_id"] not in seen_ids:
                expansion_candidates.append({
                    "chunk_id": c["chunk_id"], "text": c["text"], "source": "graph_expansion", "similarity": None,
                })
                seen_ids.add(c["chunk_id"])

        vector_source_label = "semantic_entry_point" if semantic_graph else "vector"
        candidate_chunks = (
            citation_candidate
            + [{"chunk_id": c["chunk_id"], "text": c["text"], "source": "graph", "similarity": None} for c in graph_chunks]
            + expansion_candidates
            + [{"chunk_id": c["chunk_id"], "text": c["text"], "source": vector_source_label, "similarity": c["similarity"]}
               for c in similar_chunks if c["chunk_id"] not in seen_ids]
        )

        # --- PII MASKING (single enforcement point) ---------------------------
        context = f"question={question[:80]!r}"
        matched_nodes = self.pii.mask_nodes(matched_nodes, context=context)
        plain_rows_masked = self.pii.mask_plain_rows(graph_result["plain_rows"], context=context)
        candidate_chunks = self.pii.mask_chunks(candidate_chunks, context=context)

        return {
            "cypher": graph_result["cypher"],
            "plain_rows": plain_rows_masked,
            "nodes": matched_nodes,
            "edges": graph_result["edges"],
            "candidate_chunks": candidate_chunks,
            "retrieval_scope": {
                "intent": intent,
                "retrieval_mode": retrieval_mode,  # 2026-08-26: NEW - surfaced for UI/debugging transparency
                "vector_search_ran": run_vector,
                "vector_top_k_used": config["vector_top_k"] if run_vector else 0,
                "graph_found_something": graph_found_something,
                "metadata_filters_applied": metadata_filters or {},
                "nodes_filtered_out_by_metadata": pre_filter_count - len(matched_nodes),
                # 2026-08-29: NEW - only populated under retrieval_mode="semantic_graph"
                # (None otherwise). Caller (app.py) merges this with
                # category_retrieval.py's own semantic_entry_point_search()
                # calls (same shared dict, passed via semantic_graph_trace)
                # into one combined "files touched" list for the UI's
                # evidence trail - see Semantic_Graph_Retrieval_Architecture.md.
                "semantic_graph_files": semantic_graph_trace,
                # 2026-09-03: NEW - retrieval_router.py's decision for this
                # question ("lineage"/"recall", None outside hybrid mode) +
                # how many extra chunks the vector-seeded graph-expansion
                # step actually contributed - surfaced so the Evidence
                # coverage panel can show whether/why it ran, rather than
                # leaving it invisible. See retrieval_router.py + this
                # method's graph_expansion_* locals above.
                "graph_expansion_route": graph_expansion_route,
                "graph_expansion_chunk_count": len(expansion_candidates),
                # 2026-09-07: NEW - surfaces whether this call used the
                # fast_mode path (fixed comprehensive Cypher, no NL2Cypher/
                # detect_intent LLM calls) - see retrieve()'s fast_mode
                # docstring. False for every pre-existing caller.
                "fast_mode": fast_mode,
            },
        }
