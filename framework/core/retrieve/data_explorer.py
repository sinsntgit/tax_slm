"""
Neo4j / local-vector-store DATA EXPLORER - read-only browsing + search over
what's actually stored, backing the "View Neo4j Data" panel (Knowledge
Index / Metadata / Chunks tabs). 2026-08-29: NEW, per explicit request:
"add an option to view neo4j data: Knowledge Index, Metadata, Chunks, etc.
Create a very clear panel to view full data. Allow filter as well so i can
search for any knowledge index, filename etc."

Three views:
  - KNOWLEDGE INDEX: what's actually IN each of the 4 local semantic-entry-
    point vector stores (case_law_retrieval_index/section_retrieval_index/
    circular_retrieval_index/treaty_retrieval_index - see
    local_vector_store.py / Semantic_Graph_Retrieval_Architecture.md). The
    store itself only holds (db_id, vector) pairs, so this joins those
    db_ids against Neo4j for real display fields (citation, sourceMdPath,
    etc.) - the direct answer to "which files/entities does semantic_graph
    mode actually know about."
  - METADATA: any Neo4j node label's full property set, paginated and
    free-text searchable across every property. Full-scans up to
    MAX_METADATA_SCAN nodes of the requested label then filters in PYTHON
    rather than building a dynamic Cypher WHERE across unknown-typed
    properties (a node's own properties can be strings, ints, lists, etc. -
    casting arbitrary types to string in Cypher is fragile across Neo4j
    versions). Safe because these corpora are documented elsewhere
    (local_vector_store.py's module docstring) as "tens to low hundreds of
    nodes per entity type", not millions.
  - CHUNKS: :Chunk nodes specifically - can be far more numerous than entity
    nodes (one per document section/paragraph across the whole corpus), so
    search + pagination happen SERVER-SIDE in Cypher against a fixed set of
    known string properties (CHUNK_SEARCH_PROPERTIES), not a full scan.

Domain awareness: KNOWLEDGE_INDEXES below names this domain pack's 4
semantic_graph entry-point indexes and their backing Neo4j label(s) - kept
here (rather than importing search_*_entry_points from each domain module)
so this framework-level file doesn't need to call anything domain-specific,
mirroring graphrag.py's own "import only when actually needed" pattern for
the same reason. list_labels()/list_nodes()/list_chunks() are fully
domain-agnostic - they work over whatever labels are really in the
connected database, the same "enumerate what's there" approach
framework/core/ingest/reset.py's count_by_label() already uses.
"""
from framework.core.infra import neo4j_client
from framework.core.retrieve import local_vector_store

KNOWLEDGE_INDEXES = [
    {"index_name": "case_law_retrieval_index", "display_name": "Case Law", "labels": ["CaseLaw"]},
    {"index_name": "section_retrieval_index", "display_name": "Statutory Sections", "labels": ["Section"]},
    {"index_name": "circular_retrieval_index", "display_name": "Circulars",
     "labels": ["CBDTCircular", "ExplanatoryCircular", "ExtensionCircular"]},
    {"index_name": "treaty_retrieval_index", "display_name": "Tax Treaties", "labels": ["TaxTreaty"]},
]

MAX_METADATA_SCAN = 3000    # safety rail for the full-label-scan-then-filter approach (Metadata tab)
DEFAULT_PAGE_SIZE = 25
MAX_PAGE_SIZE = 200

CHUNK_SEARCH_PROPERTIES = (
    "chunk_id", "content", "document_id", "full_citation", "short_title",
    "section", "subsection", "file_type", "entity_id", "source_document", "sourceMdPath",
)


def _clamp_page_size(page_size) -> int:
    try:
        page_size = int(page_size or DEFAULT_PAGE_SIZE)
    except (TypeError, ValueError):
        page_size = DEFAULT_PAGE_SIZE
    return max(1, min(page_size, MAX_PAGE_SIZE))


def _clamp_page(page) -> int:
    try:
        return max(1, int(page or 1))
    except (TypeError, ValueError):
        return 1


def _matches_search(props: dict, search_lower: str, unsearchable=()) -> bool:
    """2026-10-14: phase0/review-pii - `unsearchable` (the PII masker's
    always-masked properties, e.g. FirmOpinion.client_name) are skipped:
    the row comes back masked, but MATCHING on them is an oracle - searching
    "Zomato" in the FirmOpinion tab would reveal which opinion is Zomato's."""
    if not search_lower:
        return True
    for k, v in props.items():
        if v is None or k in unsearchable:
            continue
        try:
            if search_lower in str(v).lower():
                return True
        except Exception:
            continue
    return False


def get_summary(session) -> dict:
    """Counts for both browsable surfaces: every Neo4j label present, and
    each of the 4 local knowledge-index stores (row count + whether the
    store file exists at all yet, e.g. before the matching embed_* CLI has
    ever been run)."""
    label_rows = list(session.run("CALL db.labels() YIELD label RETURN label ORDER BY label"))
    by_label = []
    for row in label_rows:
        label = row["label"]
        try:
            label_safe = neo4j_client.safe_identifier(label)
        except ValueError:
            continue
        count_row = session.run(f"MATCH (n:{label_safe}) RETURN count(n) AS c").single()
        by_label.append({"label": label, "count": count_row["c"] if count_row else 0})
    by_label.sort(key=lambda r: r["count"], reverse=True)

    indexes = []
    for spec in KNOWLEDGE_INDEXES:
        exists = local_vector_store.exists(spec["index_name"])
        indexes.append({
            "index_name": spec["index_name"], "display_name": spec["display_name"],
            "labels": spec["labels"], "exists": exists,
            "vector_count": local_vector_store.count(spec["index_name"]) if exists else 0,
            "store_path": local_vector_store.describe_path(spec["index_name"]),
        })
    return {"by_label": by_label, "knowledge_indexes": indexes}


def list_labels(session) -> list:
    return [r["label"] for r in session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")]


def list_nodes(session, label: str, search: str = "", page=1, page_size=DEFAULT_PAGE_SIZE,
               unsearchable_properties=()) -> dict:
    """Paginated, free-text-searchable property listing for ONE Neo4j label
    ("Metadata" tab) - see module docstring for why this filters in Python."""
    label_safe = neo4j_client.safe_identifier(label)
    rows = list(session.run(
        f"MATCH (n:{label_safe}) RETURN elementId(n) AS db_id, properties(n) AS props "
        f"ORDER BY db_id LIMIT {MAX_METADATA_SCAN}"
    ))
    search_lower = (search or "").strip().lower()
    filtered = [{"db_id": r["db_id"], "properties": dict(r["props"])}
                for r in rows if _matches_search(dict(r["props"]), search_lower, unsearchable_properties)]

    page = _clamp_page(page)
    page_size = _clamp_page_size(page_size)
    start = (page - 1) * page_size
    return {
        "label": label, "rows": filtered[start:start + page_size], "total": len(filtered),
        "page": page, "page_size": page_size,
        "scanned": len(rows), "scan_capped": len(rows) >= MAX_METADATA_SCAN,
    }


def list_chunks(session, search: str = "", page=1, page_size=DEFAULT_PAGE_SIZE) -> dict:
    """Paginated Chunk listing ("Chunks" tab) - search/pagination happen
    SERVER-SIDE (see module docstring) since Chunk nodes can be numerous."""
    search_clean = (search or "").strip()
    if search_clean:
        where_clauses = " OR ".join(
            f"toLower(coalesce(c.{neo4j_client.safe_identifier(p)}, '')) CONTAINS toLower($search)"
            for p in CHUNK_SEARCH_PROPERTIES
        )
        where = f"WHERE {where_clauses}"
    else:
        where = ""

    page = _clamp_page(page)
    page_size = _clamp_page_size(page_size)
    skip = (page - 1) * page_size

    total_row = session.run(f"MATCH (c:Chunk) {where} RETURN count(c) AS c", search=search_clean).single()
    total = total_row["c"] if total_row else 0

    rows = list(session.run(
        f"""
        MATCH (c:Chunk) {where}
        RETURN elementId(c) AS db_id, properties(c) AS props
        ORDER BY c.chunk_id
        SKIP $skip LIMIT $limit
        """,
        search=search_clean, skip=skip, limit=page_size,
    ))
    page_rows = []
    for r in rows:
        props = dict(r["props"])
        # content can be a whole chunk's text - trim what's shown inline in
        # the table row (full properties are still returned, just with an
        # added preview key) - same "don't ship the whole body for a list
        # view" reasoning as observability.summarize_chunks_for_trace().
        content = props.get("content")
        if isinstance(content, str) and len(content) > 400:
            props["content_preview"] = content[:400] + "…"
        page_rows.append({"db_id": r["db_id"], "properties": props})
    return {"rows": page_rows, "total": total, "page": page, "page_size": page_size}


def list_knowledge_index(session, index_name: str, search: str = "", page=1,
                          page_size=DEFAULT_PAGE_SIZE, unsearchable_properties=()) -> dict:
    """What's actually IN one local semantic-entry-point vector store
    ("Knowledge Index" tab) - db_ids from local_vector_store.list_all(),
    joined against Neo4j (label-agnostic, same `MATCH (n) WHERE
    elementId(n) IN $db_ids` pattern semantic_entry_points.py's
    search_entry_points() uses) for real display fields."""
    spec = next((s for s in KNOWLEDGE_INDEXES if s["index_name"] == index_name), None)
    if spec is None:
        raise ValueError(f"Unknown knowledge index: {index_name!r} - expected one of "
                          f"{[s['index_name'] for s in KNOWLEDGE_INDEXES]}")
    page = _clamp_page(page)
    page_size = _clamp_page_size(page_size)

    db_ids = local_vector_store.list_all(index_name)
    if not db_ids:
        return {"index_name": index_name, "display_name": spec["display_name"], "rows": [], "total": 0,
                "page": page, "page_size": page_size, "vector_count": 0}

    rows = list(session.run(
        "MATCH (n) WHERE elementId(n) IN $db_ids "
        "RETURN elementId(n) AS db_id, labels(n) AS labels, properties(n) AS props",
        db_ids=db_ids,
    ))
    search_lower = (search or "").strip().lower()
    filtered = [{"db_id": r["db_id"], "labels": r["labels"], "properties": dict(r["props"])}
                for r in rows if _matches_search(dict(r["props"]), search_lower, unsearchable_properties)]
    filtered.sort(key=lambda r: r["db_id"])

    start = (page - 1) * page_size
    return {
        "index_name": index_name, "display_name": spec["display_name"],
        "rows": filtered[start:start + page_size], "total": len(filtered),
        "page": page, "page_size": page_size, "vector_count": len(db_ids),
    }
