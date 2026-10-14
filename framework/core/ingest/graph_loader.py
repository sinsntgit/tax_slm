"""
Graph ingestion engine - GENERALIZED from tax_kg_poc/scripts/03_build_graph.py.

Every function here is that script's logic verbatim, with the two genuinely
domain-specific constants (STRUCTURAL_LABELS, the Chunk-node metadata key
allowlist) turned into parameters sourced from a DomainConfig instead of
module-level constants - see framework/contracts/domain_config.py. Nothing
about MERGE-by-id loading, content-hash change detection, base/amendment
versioning, or Chunk-node/hasSourceChunk linking is tax-specific; none of
that changed.

Usage (see framework/cli/ingest_graph.py for the full CLI):
    from framework.core.ingest.graph_loader import GraphIngestionEngine
    engine = GraphIngestionEngine(domain_config)
    with neo4j_client.get_session() as session:
        engine.load_entities(session, data["entities"], mode="base")
        engine.load_chunk_nodes_and_links(session, data["entities"], chunk_records)
        engine.load_relationships(session, data["relationships"])
"""
import json

from framework.core.infra import neo4j_client, step_logger
from framework.core.ingest import entity_resolution

SUPERSEDED_BY_REL = "SUPERSEDED_BY"
HAS_SOURCE_CHUNK_REL = "hasSourceChunk"


def sanitize_props(props: dict) -> dict:
    """Neo4j property values must be primitives or arrays of a single
    primitive type - not nested maps, not mixed-type arrays. A list/dict
    value is JSON-serialized into a string property instead; a list of
    plain strings/numbers/bools passes through natively. None values are
    omitted entirely. Unchanged from 03_build_graph.py."""
    cleaned = {}
    for k, v in props.items():
        if v is None:
            continue
        if isinstance(v, dict):
            if not v:
                continue
            cleaned[k] = json.dumps(v, default=str)
        elif isinstance(v, list):
            if not v:
                continue
            # 2026-10-14: BUGFIX - also require ONE element type. The old check
            # passed mixed lists like ["9", 10] (LLM extractions mix numeric and
            # string section numbers) straight through, which Neo4j rejects
            # ("Collections containing mixed types can not be stored in
            # properties"), failing that node's whole write - the docstring
            # above already promised mixed arrays get JSON-serialized instead.
            if (all(isinstance(item, (str, int, float, bool)) for item in v)
                    and len({type(item) for item in v}) == 1):
                cleaned[k] = v
            else:
                cleaned[k] = json.dumps(v, default=str)
        else:
            cleaned[k] = v
    return cleaned


def get_existing_node_properties(session, label: str, entity_id: str) -> dict | None:
    label = neo4j_client.safe_identifier(label)
    result = session.run(
        f"MATCH (n:{label} {{id: $entity_id}}) RETURN n LIMIT 1",
        entity_id=entity_id,
    )
    record = result.single()
    if not record:
        return None
    return dict(record["n"])


def diff_and_prepare_update(existing_props: dict | None, new_props: dict, mode: str) -> dict:
    """Unchanged from 03_build_graph.py - see that module's history for the
    full behavioral docstring (base vs. amendment semantics, content_hash
    change detection, origin_* provenance preservation)."""
    if existing_props is None:
        return new_props

    if mode == "base":
        return new_props

    old_hash = existing_props.get("content_hash")
    new_hash = new_props.get("content_hash")

    if old_hash and new_hash and old_hash == new_hash:
        merged = dict(existing_props)
        merged["last_seen_document_id"] = new_props.get("document_id", existing_props.get("document_id"))
        merged["last_seen_ingestion_date"] = new_props.get("ingestion_date", existing_props.get("ingestion_date"))
        return merged

    merged = dict(new_props)
    merged["origin_document_id"] = existing_props.get("origin_document_id", existing_props.get("document_id"))
    merged["origin_ingestion_date"] = existing_props.get("origin_ingestion_date", existing_props.get("ingestion_date"))
    merged["last_amended_by_document_id"] = new_props.get("document_id")
    merged["last_amended_at"] = new_props.get("ingestion_date")
    merged["content_changed"] = True
    return merged


def clone_as_historical_node(session, label: str, current_id: str, old_props: dict):
    label = neo4j_client.safe_identifier(label)
    old_hash = old_props.get("content_hash", "unknown")
    historical_id = f"{current_id}__superseded__{old_hash}"

    historical_props = sanitize_props(dict(old_props))
    historical_props["id"] = historical_id
    historical_props["status"] = "superseded"

    session.run(
        f"""
        MERGE (h:{label} {{id: $historical_id}})
        SET h += $props
        """,
        historical_id=historical_id, props=historical_props,
    )
    neo4j_client.ensure_id_constraint(session, label)

    session.run(
        f"""
        MATCH (h:{label} {{id: $historical_id}}), (c:{label} {{id: $current_id}})
        MERGE (h)-[r:{SUPERSEDED_BY_REL}]->(c)
        """,
        historical_id=historical_id, current_id=current_id,
    )
    step_logger.log(f"  Versioned: {current_id} changed -> preserved old version as {historical_id}")


class GraphIngestionEngine:
    """Wraps the ingestion functions with a DomainConfig so callers don't
    have to thread structural_labels/chunk_metadata_keys/
    resolution_key_properties through every call manually."""

    def __init__(self, domain_config):
        self.domain_config = domain_config

    # --- CHUNK NODES ---------------------------------------------------------
    def load_chunk_nodes_and_links(self, session, entities: list[dict], chunk_records: list[dict] = None):
        """For every entity, reads its `source_chunk_ids` property, MERGEs a
        Chunk node per distinct id (with metadata from `chunk_records`
        limited to DomainConfig.chunk_metadata_keys), and links entity
        -[:hasSourceChunk]-> chunk. Idempotent, batched via UNWIND. Unchanged
        logic from 03_build_graph.py's load_chunk_nodes_and_links() - only
        CHUNK_METADATA_KEYS is now domain-supplied."""
        chunk_metadata_keys = self.domain_config.chunk_metadata_keys

        all_chunk_ids: set[str] = set()
        entity_to_chunks: dict[str, list[str]] = {}

        for ent in entities:
            raw = ent.get("properties", {}).get("source_chunk_ids", "")
            if not raw:
                continue
            chunk_ids = [c.strip() for c in raw.split(",") if c.strip()]
            entity_to_chunks[ent["id"]] = chunk_ids
            all_chunk_ids.update(chunk_ids)

        if not all_chunk_ids:
            step_logger.log("No source_chunk_ids found on any entity - skipping Chunk node creation.")
            return

        chunk_by_id = {str(c["chunk_id"]): c for c in (chunk_records or [])}

        chunk_rows = []
        for chunk_id in sorted(all_chunk_ids, key=lambda x: int(x) if x.isdigit() else 0):
            record = chunk_by_id.get(chunk_id, {})
            props = {"chunk_id": chunk_id}
            props.update({k: record[k] for k in chunk_metadata_keys if record.get(k) not in (None, "", [])})
            chunk_rows.append(sanitize_props(props))

        session.run(
            """
            UNWIND $rows AS row
            MERGE (c:Chunk {chunk_id: row.chunk_id})
            SET c += row
            """,
            rows=chunk_rows,
        )
        neo4j_client.ensure_property_index(session, "Chunk", "chunk_id")

        link_rows = [
            {"entity_id": entity_id, "chunk_id": chunk_id}
            for entity_id, chunk_ids in entity_to_chunks.items()
            for chunk_id in chunk_ids
        ]
        session.run(
            f"""
            UNWIND $rows AS row
            MATCH (n {{id: row.entity_id}}), (c:Chunk {{chunk_id: row.chunk_id}})
            MERGE (n)-[r:{HAS_SOURCE_CHUNK_REL}]->(c)
            """,
            rows=link_rows,
        )

        step_logger.log(f"Chunks: created/merged {len(all_chunk_ids)} Chunk nodes, "
                         f"{len(link_rows)} {HAS_SOURCE_CHUNK_REL} links.")

    # --- ENTITIES --------------------------------------------------------------
    def find_cross_graph_match(self, session, label: str, properties: dict, fallback_id: str):
        """Cross-run fuzzy entity resolution. Skipped for
        DomainConfig.structural_labels (deterministic, citation-precise ids -
        "no exact-id match" means "genuinely new", full stop)."""
        if label in self.domain_config.structural_labels:
            return None, 0.0
        key = entity_resolution.resolution_key(label, properties, fallback_id, self.domain_config.resolution_key_properties)
        if not key:
            return None, 0.0
        label_safe = neo4j_client.safe_identifier(label)
        try:
            result = session.run(f"MATCH (n:{label_safe}) RETURN n")
            records = list(result)
        except Exception:
            return None, 0.0

        candidates = []
        for record in records:
            existing_props = dict(record["n"]) or {}
            existing_id = existing_props.get("id")
            if not existing_id or existing_id == fallback_id:
                continue
            cand_key = entity_resolution.resolution_key(label, existing_props, existing_id, self.domain_config.resolution_key_properties)
            candidates.append((existing_id, cand_key))
        return entity_resolution.best_match(key, candidates)

    def load_entities(self, session, entities: list[dict], mode: str) -> tuple[set, dict[str, str]]:
        created_labels: set = set()
        resolved_id_map: dict[str, str] = {}
        n_created, n_updated, n_unchanged, n_versioned = 0, 0, 0, 0
        n_cross_merged, n_flagged = 0, 0

        total = len(entities)
        for i, ent in enumerate(entities):
            if total >= 50 and (i + 1) % 50 == 0:
                step_logger.log(f"  ...{i + 1}/{total} entities loaded so far")
            step_logger.vlog(f"  entity {i + 1}/{total}: {ent.get('type')} {ent.get('id')}")
            label = ent["type"]
            original_id = ent["id"]
            entity_id = original_id
            new_props = dict(ent.get("properties", {}))
            new_props["id"] = entity_id

            existing_props = get_existing_node_properties(session, label, entity_id)

            if existing_props is None:
                match_id, score = self.find_cross_graph_match(session, label, new_props, entity_id)
                if match_id and score >= entity_resolution.HIGH_SIMILARITY:
                    aliases = set(new_props.get("also_known_as_ids", "").split(",")) if new_props.get("also_known_as_ids") else set()
                    aliases.add(original_id)
                    new_props["also_known_as_ids"] = ",".join(sorted(a for a in aliases if a))
                    entity_id = match_id
                    new_props["id"] = entity_id
                    existing_props = get_existing_node_properties(session, label, entity_id)
                    resolved_id_map[original_id] = entity_id
                    n_cross_merged += 1
                elif match_id and score >= entity_resolution.MEDIUM_SIMILARITY:
                    new_props["possibleDuplicateOf"] = match_id
                    n_flagged += 1
                    n_created += 1
                else:
                    n_created += 1
            elif mode == "amendment":
                old_hash = existing_props.get("content_hash")
                new_hash = new_props.get("content_hash")
                if old_hash and new_hash and old_hash != new_hash:
                    clone_as_historical_node(session, label, entity_id, existing_props)
                    n_versioned += 1
                else:
                    n_unchanged += 1
            else:
                n_updated += 1

            final_props = sanitize_props(diff_and_prepare_update(existing_props, new_props, mode))
            label_safe = neo4j_client.safe_identifier(label)

            session.run(
                f"""
                MERGE (n:{label_safe} {{id: $entity_id}})
                SET n += $props
                """,
                entity_id=entity_id, props=final_props,
            )
            if label not in created_labels:
                neo4j_client.ensure_id_constraint(session, label)
                created_labels.add(label)

        step_logger.log(f"Entities: {n_created} newly created, {n_updated} overwritten (base mode), "
                         f"{n_versioned} amended-and-versioned, {n_unchanged} unchanged (amendment mode, same content_hash), "
                         f"{n_cross_merged} cross-run fuzzy-merged, {n_flagged} flagged as possible duplicates (possibleDuplicateOf).")
        return created_labels, resolved_id_map

    def load_relationships(self, session, relationships: list[dict]):
        # 2026-08-27: was a plain `MATCH (a {id: $source}), (b {id: $target})`
        # - a zero-row match (either endpoint doesn't exist) silently produces
        # NO edge and was never counted anywhere, so a citation that doesn't
        # resolve to a real node (e.g. a case-law section citation that can't
        # be matched under the currently-ingested Act's numbering - see the
        # cross-Act id-namespacing work this same date) vanished with zero
        # trace. Switched to OPTIONAL MATCH + FOREACH-guarded MERGE so a
        # missing endpoint is now detected and logged (with WHICH side was
        # missing) instead of just not happening.
        skipped = 0
        dropped_missing_endpoint = 0
        for rel in relationships:
            rel_type = neo4j_client.safe_identifier(rel["type"])
            source = rel["source"]
            target = rel["target"]
            props = sanitize_props(rel.get("properties", {}))
            try:
                result = session.run(
                    f"""
                    OPTIONAL MATCH (a {{id: $source}})
                    OPTIONAL MATCH (b {{id: $target}})
                    FOREACH (_ IN CASE WHEN a IS NOT NULL AND b IS NOT NULL THEN [1] ELSE [] END |
                        MERGE (a)-[r:{rel_type}]->(b)
                        SET r += $props
                    )
                    RETURN a IS NOT NULL AS source_found, b IS NOT NULL AS target_found
                    """,
                    source=source, target=target, props=props,
                )
                record = result.single()
                if record and not (record["source_found"] and record["target_found"]):
                    dropped_missing_endpoint += 1
                    missing = []
                    if not record["source_found"]:
                        missing.append(f"source id={source!r}")
                    if not record["target_found"]:
                        missing.append(f"target id={target!r}")
                    step_logger.log(f"  Dropped relationship {source} -[{rel_type}]-> {target}: "
                                     f"missing {', '.join(missing)} (no node with that id exists).")
            except Exception as e:
                skipped += 1
                step_logger.log(f"  Skipped relationship {source} -[{rel_type}]-> {target}: {e}")
        if skipped:
            step_logger.log(f"Skipped {skipped} relationships (query errors - see log above).")
        if dropped_missing_endpoint:
            step_logger.log(f"Dropped {dropped_missing_endpoint} relationships - missing endpoint node(s), "
                             f"see log lines above for which ones. A citation that doesn't resolve to any "
                             f"ingested node now shows up here instead of silently vanishing.")
