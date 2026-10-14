"""
Full-graph reset - the engine behind "Clear All Data" in the ingestion UI
(and available as a small CLI too). Domain-agnostic: it deletes every node/
relationship regardless of label, since a fresh start means a fresh start
for whichever domain pack is currently loaded - it does not attempt to
scope deletion by domain, because in this project's design ONE Neo4j
database instance holds ONE domain pack's data at a time (see
docker-compose.yml / .env's NEO4J_DATABASE).

Deliberately does NOT drop constraints/indexes (ensure_id_constraint /
ensure_vector_index etc.) - only data. Re-ingesting after a clear recreates
them idempotently (`IF NOT EXISTS`), and leaving them in place means a
clear-then-reingest cycle doesn't pay the index-build cost twice for no
reason.
"""
from framework.core.infra import neo4j_client, step_logger


def count_by_label(session) -> dict:
    """Real, itemized counts of what a full wipe would delete - backs the
    Delete Neo4j Data screen's Step 2 breakdown (and Settings/Dashboard's
    graph-size cards). Returns:
        {
          "by_label": [{"label": str, "count": int}, ...]  (all labels present, desc by count),
          "total_nodes": int,
          "total_relationships": int,
          "chunks_with_embedding": int,
        }
    Domain-agnostic on purpose (same reasoning as clear_all_data): it
    enumerates whatever labels actually exist in the currently connected
    database rather than assuming tax-specific label names, so this works
    unchanged for any domain pack."""
    label_rows = list(session.run("CALL db.labels() YIELD label RETURN label ORDER BY label"))
    by_label = []
    for row in label_rows:
        label = row["label"]
        try:
            label_safe = neo4j_client.safe_identifier(label)
        except ValueError:
            continue  # skip anything that isn't a safe Cypher identifier (shouldn't happen for a real label)
        count_row = session.run(f"MATCH (n:{label_safe}) RETURN count(n) AS c").single()
        by_label.append({"label": label, "count": count_row["c"] if count_row else 0})
    by_label.sort(key=lambda r: r["count"], reverse=True)

    total_nodes_row = session.run("MATCH (n) RETURN count(n) AS c").single()
    total_rels_row = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()
    embedded_row = session.run(
        "MATCH (c:Chunk) WHERE c.embedding IS NOT NULL RETURN count(c) AS c"
    ).single()

    return {
        "by_label": by_label,
        "total_nodes": total_nodes_row["c"] if total_nodes_row else 0,
        "total_relationships": total_rels_row["c"] if total_rels_row else 0,
        "chunks_with_embedding": embedded_row["c"] if embedded_row else 0,
    }


def clear_all_data(session, batch_size: int = 5000) -> dict:
    """Deletes every node (and its relationships) in batches, so a
    large graph doesn't attempt one giant transaction. Returns
    {"nodes_deleted": int, "batches": int}."""
    total_deleted = 0
    batches = 0
    while True:
        result = session.run(
            "MATCH (n) WITH n LIMIT $batch_size DETACH DELETE n",
            batch_size=batch_size,
        )
        summary = result.consume()
        deleted = summary.counters.nodes_deleted
        batches += 1
        total_deleted += deleted
        step_logger.log(f"  [reset] batch {batches}: deleted {deleted} node(s) (running total: {total_deleted})")
        if deleted == 0:
            break
    return {"nodes_deleted": total_deleted, "batches": batches}


# --- PER-DOCUMENT (selective) delete -----------------------------------------
# Backs the Delete Neo4j Data screen's "delete specific document(s)" option,
# alongside the full-wipe clear_all_data() above. Every entity this codebase's
# extraction converters produce (Act, case law, generic - see
# attach_pipeline_metadata()/build_graph_from_extraction()/extract_generic.py)
# is stamped with `document_id`/`document_name` at ingestion, so a document is
# identified the same way everywhere: by its document_id, not by node label
# (one document's entities span many different labels - CaseLaw, Issue,
# Conclusion, Argument, LegalConcept, ... for a single case-law PDF, for
# example). Chunk nodes deliberately do NOT carry document_id themselves (see
# config.py's CHUNK_METADATA_KEYS) - they're identified only via the
# hasSourceChunk edge from their owning entity - so a Chunk is deleted as an
# ORPHAN CLEANUP pass after its owning entities are gone, not by matching
# document_id on the Chunk itself. This also means a Chunk that (unusually)
# is still linked to a KEPT entity is correctly left alone.

def list_documents(session) -> list[dict]:
    """One row per distinct document_id currently in the graph - the Delete
    screen's document picker. entity_count is every node stamped with this
    document_id (regardless of label); chunk_count is the Chunk nodes
    reachable from those entities via hasSourceChunk, i.e. what a delete of
    this document would also remove. Returns entity_count-descending order,
    same convention as count_by_label()'s by_label list."""
    rows = list(session.run(
        """
        MATCH (n) WHERE n.document_id IS NOT NULL
        OPTIONAL MATCH (n)-[:hasSourceChunk]->(c:Chunk)
        RETURN n.document_id AS document_id,
               head(collect(DISTINCT n.document_name)) AS document_name,
               head(collect(DISTINCT n.source_type)) AS source_type,
               head(collect(DISTINCT n.authority_level)) AS authority_level,
               head(collect(DISTINCT n.file_type)) AS file_type,
               count(DISTINCT n) AS entity_count,
               count(DISTINCT c) AS chunk_count
        ORDER BY entity_count DESC
        """
    ))
    return [
        {
            "document_id": r["document_id"],
            "document_name": r["document_name"] or r["document_id"],
            "source_type": r["source_type"],
            "authority_level": r["authority_level"],
            "file_type": r["file_type"],
            "entity_count": r["entity_count"],
            "chunk_count": r["chunk_count"],
        }
        for r in rows
    ]


def backfill_case_law_document_metadata(session) -> dict:
    """One-time, NON-DESTRUCTIVE repair for CaseLaw graphs ingested BEFORE
    extract_case_law.py started stamping document_id/document_name/
    source_type/authority_level/jurisdiction (see that script's
    build_graph_from_extraction() docstring) - those older CaseLaw nodes,
    and everything extracted alongside them (Issue/Conclusion/Argument/
    LegalConcept/CaseCitation), have no document_id at all, so
    list_documents() above can't see them and the Delete screen's per-
    document picker silently omits every case law ingested before the fix.
    Only ADDS missing properties (`document_id IS NULL` guards every write),
    never touches a document that already has one - safe to run repeatedly
    or against a graph that's already fully backfilled (no-op then).

    Two passes:
      1. Real (non-stub) CaseLaw nodes - identified by having a
         `source_document` property, which only a genuine extraction run
         sets (see build_graph_from_extraction()'s case_props); the small
         placeholder CaseLaw nodes created for precedents referenced-but-
         not-yet-ingested (see the FOLLOWS/DISTINGUISHES_CASE/RELIES_ON_CASE/
         OVERRULES stub-node loop) deliberately do NOT get treated as their
         own "document" here - they aren't one, and giving each a document_id
         would flood the Delete picker with one fake entry per cited
         precedent instead of one entry per actually-ingested PDF.
      2. Every node reachable from a now-stamped CaseLaw via the
         WITHIN-THIS-CASE structural relationships only (INTERPRETS to an
         Issue, HAS_CONCLUSION, HAS_ARGUMENT, INSTANTIATES_CONCEPT,
         HAS_CITATION) inherits the same document fields. Deliberately
         EXCLUDES FOLLOWS/DISTINGUISHES_CASE/RELIES_ON_CASE/OVERRULES and
         INTERPRETS-to-Section/SubSection, which cross into a DIFFERENT
         case's or the Act's own nodes - those must keep their own identity,
         not inherit this case's.
    Returns {"case_law_nodes_stamped": int, "sub_entities_stamped": int}."""
    case_result = session.run(
        """
        MATCH (c:CaseLaw)
        WHERE c.document_id IS NULL AND c.source_document IS NOT NULL
        SET c.document_id = c.id,
            c.document_name = coalesce(c.full_citation, c.case_name, c.id),
            c.source_type = coalesce(c.source_type, 'CaseLaw'),
            c.authority_level = coalesce(c.authority_level, 'Judicial'),
            c.jurisdiction = coalesce(c.jurisdiction, c.court, 'India')
        RETURN count(c) AS n
        """
    )
    case_law_stamped = case_result.single()["n"]

    sub_result = session.run(
        """
        MATCH (c:CaseLaw)-[:INTERPRETS|HAS_CONCLUSION|HAS_ARGUMENT|INSTANTIATES_CONCEPT|HAS_CITATION*1..2]->(sub)
        WHERE c.document_id IS NOT NULL AND sub.document_id IS NULL
          AND NOT sub:CaseLaw AND NOT sub:Section AND NOT sub:SubSection
          AND NOT sub:Clause AND NOT sub:SubClause
        SET sub.document_id = c.document_id,
            sub.document_name = c.document_name,
            sub.source_type = c.source_type,
            sub.authority_level = c.authority_level,
            sub.jurisdiction = c.jurisdiction
        RETURN count(DISTINCT sub) AS n
        """
    )
    sub_entities_stamped = sub_result.single()["n"]

    step_logger.log(f"  [reset] backfill_case_law_document_metadata: stamped {case_law_stamped} "
                     f"CaseLaw node(s), {sub_entities_stamped} linked sub-entity node(s)")
    return {"case_law_nodes_stamped": case_law_stamped, "sub_entities_stamped": sub_entities_stamped}


def delete_documents(session, document_ids: list[str], batch_size: int = 5000) -> dict:
    """Deletes every entity stamped with one of the given document_ids, then
    sweeps up any Chunk node left with no remaining relationships at all
    (i.e. every entity that used to link to it is now gone) - see this
    section's docstring above for why Chunk cleanup can't just filter on
    document_id directly. Both passes are batched, same pattern as
    clear_all_data(). Returns {"entities_deleted": int, "chunks_deleted": int,
    "batches": int}."""
    if not document_ids:
        return {"entities_deleted": 0, "chunks_deleted": 0, "batches": 0}

    total_entities = 0
    batches = 0
    while True:
        result = session.run(
            """
            MATCH (n) WHERE n.document_id IN $document_ids
            WITH n LIMIT $batch_size
            DETACH DELETE n
            """,
            document_ids=document_ids, batch_size=batch_size,
        )
        summary = result.consume()
        deleted = summary.counters.nodes_deleted
        batches += 1
        total_entities += deleted
        step_logger.log(f"  [reset] delete_documents batch {batches}: deleted {deleted} entity node(s)")
        if deleted == 0:
            break

    total_chunks = 0
    while True:
        result = session.run(
            """
            MATCH (c:Chunk) WHERE NOT (c)--()
            WITH c LIMIT $batch_size
            DETACH DELETE c
            """,
            batch_size=batch_size,
        )
        summary = result.consume()
        deleted = summary.counters.nodes_deleted
        batches += 1
        total_chunks += deleted
        step_logger.log(f"  [reset] delete_documents orphan-chunk batch: deleted {deleted} chunk node(s)")
        if deleted == 0:
            break

    return {"entities_deleted": total_entities, "chunks_deleted": total_chunks, "batches": batches}
