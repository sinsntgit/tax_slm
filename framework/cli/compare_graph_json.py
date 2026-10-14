"""
Read-only comparison: how much of a local data/graphs/*_graph.json
extraction (its entities[] + relationships[]) is actually present in the
live Neo4j graph, and where it differs. Answers "does what's in Neo4j match
what this extraction file says should be there?" without writing anything -
every query is a MATCH/OPTIONAL MATCH.

Built to answer the Reckitt Benckiser question directly (comparing BOTH
data/graphs/*Reckitt*_graph.json duplicate-run files against Neo4j - e.g.
to confirm what merge_duplicate_caselaw.py actually did), but works for any
*_graph.json under data/graphs/ via --glob/--file.

CaseLaw-aware: if merge_duplicate_caselaw.py has already run, a duplicate
CaseLaw node's id was DETACH DELETEd from Neo4j entirely - an exact-id miss
for a CaseLaw entity does NOT necessarily mean data loss. This script
resolves that case specifically: it loads every surviving CaseLaw node's
also_known_as_ids (set by the merge script) and, before reporting a CaseLaw
id (or any relationship touching one) as missing, checks whether it was
folded into a canonical node - reporting "merged into <canonical_id>"
instead of "missing" when so.

Usage:
    python -m framework.cli.compare_graph_json --glob "data/graphs/*Reckitt*_graph.json"
    python -m framework.cli.compare_graph_json --file data/graphs/Foo_graph.json --verbose
"""
import argparse
import glob
import json

from framework.core.infra import neo4j_client, step_logger

# Properties whose values are long free text (case facts, findings, ...) -
# still checked for equality, but truncated in printed diffs so a mismatch
# report doesn't dump paragraphs of text into the terminal.
_LONG_TEXT_PROPS = {"hasText", "holding", "case_name", "full_citation"}
_TRUNCATE_AT = 90


def _load_files(globs: list[str], files: list[str]) -> list[tuple[str, dict]]:
    paths = []
    for pat in globs:
        paths.extend(sorted(glob.glob(pat)))
    paths.extend(files)
    seen, out = set(), []
    for p in paths:
        if p in seen:
            continue
        seen.add(p)
        with open(p, encoding="utf-8") as f:
            out.append((p, json.load(f)))
    return out


def _build_caselaw_alias_map(session) -> dict[str, str]:
    """dup_id -> canonical_id, from every surviving CaseLaw's
    also_known_as_ids (comma-separated), as left by merge_duplicate_caselaw.py."""
    alias_map = {}
    for row in session.run("MATCH (n:CaseLaw) WHERE n.also_known_as_ids IS NOT NULL "
                            "RETURN n.id AS id, n.also_known_as_ids AS aka"):
        for dup_id in (row["aka"] or "").split(","):
            dup_id = dup_id.strip()
            if dup_id:
                alias_map[dup_id] = row["id"]
    return alias_map


def _resolve_id(node_id: str, alias_map: dict[str, str]) -> tuple[str, str | None]:
    """Returns (id_to_query_neo4j_with, note) - note is set only when the
    original id was translated to a merge survivor."""
    if node_id in alias_map:
        return alias_map[node_id], f"merged into {alias_map[node_id]}"
    return node_id, None


def _fetch_node(session, node_id: str):
    rec = session.run("MATCH (n {id: $id}) RETURN properties(n) AS p, labels(n) AS labels", id=node_id).single()
    return {"properties": rec["p"], "labels": rec["labels"]} if rec else None


def _fetch_rel(session, source: str, rtype: str, target: str):
    safe_type = neo4j_client.safe_identifier(rtype)
    rec = session.run(
        f"MATCH (a {{id: $source}})-[r:{safe_type}]->(b {{id: $target}}) RETURN properties(r) AS p LIMIT 1",
        source=source, target=target,
    ).single()
    return {"properties": rec["p"]} if rec else None


def _diff_props(json_props: dict, neo4j_props: dict) -> list[str]:
    diffs = []
    for k, jv in (json_props or {}).items():
        nv = (neo4j_props or {}).get(k)
        if str(jv) != str(nv):
            jv_s = str(jv)[:_TRUNCATE_AT] + ("…" if k in _LONG_TEXT_PROPS and len(str(jv)) > _TRUNCATE_AT else "")
            nv_s = str(nv)[:_TRUNCATE_AT] + ("…" if k in _LONG_TEXT_PROPS and len(str(nv)) > _TRUNCATE_AT else "")
            diffs.append(f"{k}: json={jv_s!r} neo4j={nv_s!r}")
    return diffs


def compare_file(session, path: str, graph: dict, alias_map: dict[str, str], verbose: bool) -> dict:
    entities = graph.get("entities", [])
    relationships = graph.get("relationships", [])

    node_found = node_missing = node_merged = node_mismatched = 0
    rel_found = rel_missing = 0
    lines = [f"\n=== {path} ({len(entities)} entities, {len(relationships)} relationships) ==="]

    id_translation: dict[str, str] = {}  # original json id -> id actually present in neo4j (post-merge aware)
    for e in entities:
        jid, jtype = e["id"], e["type"]
        query_id, note = _resolve_id(jid, alias_map) if jtype == "CaseLaw" else (jid, None)
        node = _fetch_node(session, query_id)
        if node is None:
            node_missing += 1
            lines.append(f"  MISSING  [{jtype}] {jid}")
            continue
        id_translation[jid] = query_id
        if note:
            node_merged += 1
            if verbose:
                lines.append(f"  MERGED   [{jtype}] {jid} -> {note}")
        diffs = _diff_props(e.get("properties", {}), node["properties"])
        if diffs:
            node_mismatched += 1
            lines.append(f"  DIFFERS  [{jtype}] {jid}" + (f"  ({note})" if note else ""))
            for d in diffs:
                lines.append(f"             {d}")
        else:
            node_found += 1
            if verbose and not note:
                lines.append(f"  OK       [{jtype}] {jid}")

    for r in relationships:
        src = id_translation.get(r["source"], r["source"])
        tgt = id_translation.get(r["target"], r["target"])
        rel = _fetch_rel(session, src, r["type"], tgt)
        if rel is None:
            rel_missing += 1
            lines.append(f"  MISSING REL  ({r['source']})-[:{r['type']}]->({r['target']})")
        else:
            rel_found += 1

    summary = {
        "path": path, "entities_total": len(entities),
        "nodes_found_exact": node_found, "nodes_merged": node_merged,
        "nodes_property_mismatch": node_mismatched, "nodes_missing": node_missing,
        "relationships_total": len(relationships),
        "relationships_found": rel_found, "relationships_missing": rel_missing,
    }
    lines.append(f"  -- nodes: {node_found} exact match, {node_merged} merged-but-present, "
                 f"{node_mismatched} present-with-differences, {node_missing} missing (of {len(entities)})")
    lines.append(f"  -- relationships: {rel_found} present, {rel_missing} missing (of {len(relationships)})")
    for line in lines:
        step_logger.log(line)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--glob", action="append", default=[],
                         help="Glob pattern for *_graph.json files to compare, e.g. "
                              "'data/graphs/*Reckitt*_graph.json'. Repeatable.")
    parser.add_argument("--file", action="append", default=[], help="Explicit *_graph.json path. Repeatable.")
    parser.add_argument("--verbose", action="store_true", help="Also print exact/merged matches, not just diffs.")
    args = parser.parse_args()
    step_logger.set_verbose(True)

    if not args.glob and not args.file:
        args.glob = ["data/graphs/*Reckitt*_graph.json"]

    files = _load_files(args.glob, args.file)
    if not files:
        step_logger.log(f"No files matched {args.glob + args.file} - nothing to compare.")
        return

    with step_logger.Step("Connecting to Neo4j"):
        neo4j_client.verify_connectivity()

    with neo4j_client.get_session() as session:
        with step_logger.Step("Building CaseLaw merge-alias map (also_known_as_ids)"):
            alias_map = _build_caselaw_alias_map(session)
        if alias_map:
            step_logger.log(f"{len(alias_map)} duplicate CaseLaw id(s) known to have been merged: "
                             f"{alias_map}")

        summaries = [compare_file(session, path, graph, alias_map, args.verbose) for path, graph in files]

    step_logger.log("\n=== OVERALL ===")
    for s in summaries:
        step_logger.log(f"{s['path']}: nodes {s['nodes_found_exact']}✓/{s['nodes_merged']}merged/"
                         f"{s['nodes_property_mismatch']}differ/{s['nodes_missing']}missing of {s['entities_total']}"
                         f"  |  rels {s['relationships_found']}✓/{s['relationships_missing']}missing of "
                         f"{s['relationships_total']}")


if __name__ == "__main__":
    main()
