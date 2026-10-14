"""
Read-only view of a domain pack's STATIC ontology definition (schema.json)
for the webapp's "View Ontology" screen - distinct from /api/schema
(graphrag.get_schema_summary()), which reports the LIVE Neo4j schema (what
labels/relationship types actually exist in the graph right now). This
module answers "what does the ontology DECLARE" (tree of classes + their
properties), not "what's actually been ingested" - useful precisely because
the two can and do diverge (e.g. CBDTCircular/TaxTreaty are fully declared
here but nothing has ingested one yet).

Two shapes are built from the same schema.json, since the webapp offers
both a tree view and a table view:
  - build_class_tree(schema): nested dict, rooted at a synthetic "Ontology"
    node, for a collapsible org-chart-style tree control. A class with
    MULTIPLE parents (a few exist, e.g. Conclusion) is nested under its
    FIRST declared parent only, to keep this a genuine tree (not a DAG) -
    its other parents are listed on the node as `also_parents` instead.
  - build_class_table(schema): flat list, one row per class, with its own
    (non-inherited) data properties and the object properties where it's
    the DOMAIN (the "this class HAS a relationship to X" direction) - for
    a sortable/filterable/collapsible table.
"""


def _own_data_properties(schema: dict, class_name: str) -> list[dict]:
    return sorted(
        (
            {"name": p["name"], "range": p.get("range", []), "comment": p.get("comment", [])}
            for p in schema.get("data_properties", [])
            if class_name in p.get("domain", [])
        ),
        key=lambda p: p["name"],
    )


def _own_object_properties(schema: dict, class_name: str) -> list[dict]:
    """Relationships where this class is the DOMAIN (source) - the
    direction that reads naturally as "this class HAS a ... TO ...". The
    inverse direction (this class as RANGE) is available on the target
    class's own row instead, rather than doubling every edge onto both
    ends - keeps the table one row per class, not one row per class per
    incoming edge."""
    return sorted(
        (
            {"name": p["name"], "range": p.get("range", []), "inverse": p.get("inverse"),
             "functional": p.get("functional", False), "comment": p.get("comment", [])}
            for p in schema.get("object_properties", [])
            if class_name in p.get("domain", [])
        ),
        key=lambda p: p["name"],
    )


def build_class_table(schema: dict) -> list[dict]:
    rows = []
    for c in schema.get("classes", []):
        name = c["name"]
        rows.append({
            "name": name,
            "parents": c.get("parents", []),
            "comment": c.get("comment", []),
            "data_properties": _own_data_properties(schema, name),
            "object_properties": _own_object_properties(schema, name),
        })
    rows.sort(key=lambda r: r["name"])
    return rows


def build_class_tree(schema: dict) -> dict:
    class_names = {c["name"] for c in schema.get("classes", [])}
    nodes: dict[str, dict] = {}
    for c in schema.get("classes", []):
        name = c["name"]
        parents = c.get("parents", []) or []
        primary_parent = next((p for p in parents if p in class_names), None)
        nodes[name] = {
            "name": name,
            "comment": c.get("comment", []),
            "primary_parent": primary_parent,
            "also_parents": [p for p in parents if p != primary_parent],
            "data_properties": _own_data_properties(schema, name),
            "object_properties": _own_object_properties(schema, name),
            "children": [],
        }

    root = {"name": "Ontology", "comment": [], "data_properties": [], "object_properties": [], "children": []}
    for name, node in nodes.items():
        parent = node["primary_parent"]
        if parent and parent in nodes:
            nodes[parent]["children"].append(node)
        else:
            # No declared parent, or parent isn't itself a declared class
            # (e.g. "Thing", the implicit OWL root) - attach directly under
            # the synthetic root so every class is reachable in the tree.
            root["children"].append(node)

    def _sort(node: dict):
        node["children"].sort(key=lambda n: n["name"])
        for child in node["children"]:
            _sort(child)

    _sort(root)
    return root


def build_ontology_view(schema: dict) -> dict:
    return {
        "tree": build_class_tree(schema),
        "table": build_class_table(schema),
        "object_property_count": len(schema.get("object_properties", [])),
        "data_property_count": len(schema.get("data_properties", [])),
        "class_count": len(schema.get("classes", [])),
    }
