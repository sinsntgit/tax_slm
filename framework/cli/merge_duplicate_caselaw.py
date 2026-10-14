"""
One-off cleanup: merges duplicate CaseLaw nodes that represent the SAME
real-world judgment but ended up as separate Neo4j nodes - discovered while
investigating why an /api/ask question about the Reckitt Benckiser case
surfaced its "scrap sale income" finding instead of its "demerger" finding.

ROOT CAUSE (confirmed by inspecting this project's own data/graphs/*Reckitt*
files, not guessed): CaseLaw's id is built from a slug of the LLM-extracted
`citation` text (extract_case_law.py: `case_id = f"CaseLaw_{_slug(citation)}"`
). Two ingestion runs against the SAME PDF produced citation text differing
only by "Dy. Commissioner" vs "Deputy Commissioner" - enough to slug into two
different case_ids, so graph_loader.py's exact-id MERGE never even considers
them the same node. Its cross-graph fuzzy match (find_cross_graph_match())
SHOULD catch this via case_name similarity (it scores ~98/100 for this exact
pair - well above the auto-merge bar) - but that only merges the CaseLaw
node ITSELF. Every entity that run also created (Issue, Conclusion,
Argument, ...) has no entry in DomainConfig.resolution_key_properties, so
those never cross-match and each run leaves behind its OWN complete copy of
the case's Issue/Conclusion structure, some of it left pointing at a case_id
that may no longer resolve to any node at all. Net effect: a question about
the case can surface either copy's findings depending on which the graph/
vector path happens to reach, with no guarantee the more complete (or more
relevant) copy wins.

WHAT THIS SCRIPT DOES (dry-run by default - see --apply):
  1. Loads every CaseLaw node's case_name and clusters them using the SAME
     fuzzy-similarity function graph_loader.py already uses at ingest time
     (entity_resolution.similarity, token_sort_ratio), at --min-similarity
     (default 95 = entity_resolution.HIGH_SIMILARITY - the bar an ingest-
     time auto-merge would already have used).
  2. Within each cluster (2+ nodes judged the same case), picks a CANONICAL
     node - whichever currently has the most relationships (already the
     better-connected copy) - tied broken by the earliest ingestion_date.
  3. For every other (duplicate) node in the cluster: re-points every
     relationship touching it (either direction, any type) onto the
     canonical node instead, copies over any property the canonical is
     missing (same "fill gaps, never overwrite" rule graph_loader.py's own
     batch dedup uses), records the duplicate's id under
     also_known_as_ids on the canonical, then DETACH DELETEs the now-empty
     duplicate node.
  4. Runs the same INTERPRETS backfill as framework.cli.
     backfill_caselaw_issue_edges (imported, not reimplemented) across the
     WHOLE graph afterwards, so Issue/Conclusion/Argument nodes that were
     created under a losing duplicate's id - and therefore never linked to
     any CaseLaw at all - become reachable from the surviving node too.
     This last step is what actually fixes "the answer only reflects one
     issue of the case".

Usage:
    python -m framework.cli.merge_duplicate_caselaw --domain tax                    # dry run - report only, writes nothing
    python -m framework.cli.merge_duplicate_caselaw --domain tax --apply            # actually merge
    python -m framework.cli.merge_duplicate_caselaw --domain tax --min-similarity 90 --apply
"""
import argparse

from framework.cli.pack_loader import load_domain_pack
from framework.cli.backfill_caselaw_issue_edges import backfill as backfill_interprets_issue_edges
from framework.core.infra import neo4j_client, step_logger
from framework.core.ingest import entity_resolution


def _fetch_caselaw_nodes(session) -> list[dict]:
    rows = session.run(
        "MATCH (n:CaseLaw) RETURN n.id AS id, n.case_name AS case_name, "
        "n.ingestion_date AS ingestion_date"
    )
    return [dict(r) for r in rows]


def _cluster(nodes: list[dict], resolution_key_properties: dict, min_similarity: float) -> list[list[dict]]:
    """Union-find over pairwise case_name similarity - same key-building/
    scoring functions ingest-time cross-graph matching uses, just applied
    after the fact across every CaseLaw already in the graph instead of one
    new entity against the existing ones."""
    keyed = [
        (n, entity_resolution.resolution_key("CaseLaw", {"case_name": n["case_name"]}, n["id"], resolution_key_properties))
        for n in nodes
    ]
    parent = list(range(len(keyed)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(len(keyed)):
        for j in range(i + 1, len(keyed)):
            if entity_resolution.similarity(keyed[i][1], keyed[j][1]) >= min_similarity:
                union(i, j)

    groups: dict[int, list[dict]] = {}
    for i, (n, _key) in enumerate(keyed):
        groups.setdefault(find(i), []).append(n)
    return [g for g in groups.values() if len(g) >= 2]


def _degree(session, node_id: str) -> int:
    rec = session.run(
        "MATCH (n:CaseLaw {id: $id}) OPTIONAL MATCH (n)-[r]-() RETURN count(r) AS deg",
        id=node_id,
    ).single()
    return rec["deg"] if rec else 0


def _pick_canonical(session, cluster: list[dict]) -> dict:
    scored = [(n, _degree(session, n["id"])) for n in cluster]
    scored.sort(key=lambda t: (-t[1], t[0].get("ingestion_date") or "9999", t[0]["id"]))
    return scored[0][0]


def _merge_duplicate_into_canonical(session, dup_id: str, canon_id: str, dry_run: bool) -> dict:
    dup_props = (session.run("MATCH (n:CaseLaw {id: $id}) RETURN properties(n) AS p", id=dup_id).single() or {}).get("p") or {}
    canon_props = (session.run("MATCH (n:CaseLaw {id: $id}) RETURN properties(n) AS p", id=canon_id).single() or {}).get("p") or {}

    fill = {k: v for k, v in dup_props.items() if v and not canon_props.get(k) and k not in ("id",)}
    aliases = set((canon_props.get("also_known_as_ids") or "").split(",")) if canon_props.get("also_known_as_ids") else set()
    aliases.add(dup_id)
    fill["also_known_as_ids"] = ",".join(sorted(a for a in aliases if a))

    outgoing = list(session.run(
        "MATCH (n:CaseLaw {id: $id})-[r]->(b) RETURN type(r) AS rtype, properties(r) AS props, b.id AS other_id",
        id=dup_id,
    ))
    incoming = list(session.run(
        "MATCH (a)-[r]->(n:CaseLaw {id: $id}) RETURN type(r) AS rtype, properties(r) AS props, a.id AS other_id",
        id=dup_id,
    ))

    plan = {"dup_id": dup_id, "canon_id": canon_id, "properties_filled": sorted(fill.keys() - {"also_known_as_ids"}),
            "relationships_repointed": len(outgoing) + len(incoming)}

    if dry_run:
        return plan

    if fill:
        session.run("MATCH (n:CaseLaw {id: $id}) SET n += $props", id=canon_id, props=fill)

    for row in outgoing:
        if row["other_id"] == canon_id or row["other_id"] is None:
            continue
        rtype = neo4j_client.safe_identifier(row["rtype"])
        session.run(
            f"MATCH (a:CaseLaw {{id: $canon_id}}), (b {{id: $other_id}}) "
            f"MERGE (a)-[r:{rtype}]->(b) SET r += $props",
            canon_id=canon_id, other_id=row["other_id"], props=row["props"] or {},
        )
    for row in incoming:
        if row["other_id"] == canon_id or row["other_id"] is None:
            continue
        rtype = neo4j_client.safe_identifier(row["rtype"])
        session.run(
            f"MATCH (a {{id: $other_id}}), (b:CaseLaw {{id: $canon_id}}) "
            f"MERGE (a)-[r:{rtype}]->(b) SET r += $props",
            canon_id=canon_id, other_id=row["other_id"], props=row["props"] or {},
        )

    session.run("MATCH (n:CaseLaw {id: $id}) DETACH DELETE n", id=dup_id)
    return plan


def run(session, resolution_key_properties: dict, min_similarity: float, apply: bool) -> list[dict]:
    nodes = _fetch_caselaw_nodes(session)
    clusters = _cluster(nodes, resolution_key_properties, min_similarity)
    step_logger.log(f"Found {len(nodes)} CaseLaw node(s), {len(clusters)} duplicate cluster(s) "
                     f"at min_similarity={min_similarity}.")

    plans = []
    for cluster in clusters:
        canonical = _pick_canonical(session, cluster)
        step_logger.log(f"Cluster: {[n['id'] for n in cluster]}")
        step_logger.log(f"  canonical -> {canonical['id']}")
        for n in cluster:
            if n["id"] == canonical["id"]:
                continue
            plan = _merge_duplicate_into_canonical(session, n["id"], canonical["id"], dry_run=not apply)
            step_logger.log(f"  {'WOULD MERGE' if not apply else 'merged'} {n['id']} -> {canonical['id']} "
                             f"({plan['relationships_repointed']} relationship(s) repointed, "
                             f"properties filled: {plan['properties_filled'] or 'none'})")
            plans.append(plan)
    return plans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default=None)
    parser.add_argument("--min-similarity", type=float, default=entity_resolution.HIGH_SIMILARITY,
                         help="Fuzzy case_name similarity (0-100) required to treat two CaseLaw nodes as "
                              "the same case. Default matches the ingest-time auto-merge bar.")
    parser.add_argument("--apply", action="store_true", help="Actually merge. Without this, only reports the plan.")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    domain_config, _prompt_pack = load_domain_pack(args.domain)

    with step_logger.Step("Connecting to Neo4j"):
        neo4j_client.verify_connectivity()

    with neo4j_client.get_session() as session:
        with step_logger.Step("Finding and merging duplicate CaseLaw nodes"):
            plans = run(session, domain_config.resolution_key_properties, args.min_similarity, args.apply)

        if args.apply:
            with step_logger.Step("Backfilling CaseLaw -> Issue INTERPRETS edges across the whole graph"):
                created = backfill_interprets_issue_edges(session, dry_run=False)
            step_logger.log(f"Backfilled {created} INTERPRETS edge(s) after merge.")

    if not plans:
        step_logger.log("No duplicate CaseLaw clusters found - nothing to do.")
    elif not args.apply:
        step_logger.log(f"Dry run - {len(plans)} merge(s) would be made. Re-run with --apply to perform them.")
    else:
        step_logger.log(f"Done - {len(plans)} duplicate CaseLaw node(s) merged.")


if __name__ == "__main__":
    main()
