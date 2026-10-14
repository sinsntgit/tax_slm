"""
2026-10-14: NEW (phase0/opinions) - provenance semantics of
framework/core/ingest/graph_loader.py, proven against a small in-memory fake
Neo4j session (no server in the cloud; tests/helpers didn't exist in this
worktree yet, so the fake lives here). The fake interprets exactly the Cypher
shapes graph_loader.py issues - including the ON CREATE SET vs SET choice for
relationships - so these tests exercise the loader's real decision logic.

Both directions the product needs:
  * a cited_reference stub NEVER overwrites/downgrades a node (or edge) that
    came from a real ingested document - base AND amendment mode;
  * ingesting the real document later UPGRADES the stub (properties replaced,
    provenance -> ingested_document, no stub residue, no "superseded" clone).
"""
import json
import re
import sys
from pathlib import Path

import pytest

from framework.core.ingest import provenance
from framework.core.ingest.graph_loader import GraphIngestionEngine
from domain_packs.tax.config import DOMAIN_CONFIG

REPO = Path(__file__).resolve().parents[1]
CONVERTERS = REPO / "domain_packs" / "tax" / "converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))


# --- minimal fake Neo4j session ----------------------------------------------

class _Result:
    def __init__(self, records):
        self._records = records

    def single(self):
        return self._records[0] if self._records else None

    def __iter__(self):
        return iter(self._records)


class FakeGraph:
    """Nodes keyed by id ({"labels": set, "props": dict}); edges keyed by
    (source, type, target) -> props. Unknown Cypher raises, so a loader change
    that issues a new query shape fails loudly here instead of passing."""

    def __init__(self):
        self.nodes: dict[str, dict] = {}
        self.edges: dict[tuple, dict] = {}
        self.chunks: dict[str, dict] = {}
        self.queries: list[str] = []

    # neo4j session API
    def run(self, query, **params):
        q = " ".join(str(query).split())
        self.queries.append(q)
        if q.startswith("CREATE CONSTRAINT") or q.startswith("CREATE INDEX"):
            return _Result([])
        m = re.fullmatch(r"MATCH \(n:(\w+) \{id: \$entity_id\}\) RETURN n LIMIT 1", q)
        if m:
            node = self.nodes.get(params["entity_id"])
            return _Result([{"n": dict(node["props"])}] if node and m.group(1) in node["labels"] else [])
        m = re.fullmatch(r"MATCH \(n:(\w+)\) RETURN n", q)
        if m:
            return _Result([{"n": dict(n["props"])} for n in self.nodes.values() if m.group(1) in n["labels"]])
        m = re.fullmatch(r"MERGE \(n:(\w+) \{id: \$entity_id\}\) SET n (\+?=) \$props", q)
        if m:
            node = self.nodes.setdefault(params["entity_id"], {"labels": set(), "props": {}})
            node["labels"].add(m.group(1))
            if m.group(2) == "=":
                node["props"] = dict(params["props"])
            else:
                node["props"].update(params["props"])
            return _Result([])
        m = re.fullmatch(r"MERGE \(h:(\w+) \{id: \$historical_id\}\) SET h \+= \$props", q)
        if m:
            node = self.nodes.setdefault(params["historical_id"], {"labels": set(), "props": {}})
            node["labels"].add(m.group(1))
            node["props"].update(params["props"])
            return _Result([])
        if q.startswith("MATCH (h:") and "SUPERSEDED_BY" in q:
            self.edges[(params["historical_id"], "SUPERSEDED_BY", params["current_id"])] = {}
            return _Result([])
        if q.startswith("UNWIND $rows AS row MERGE (c:Chunk"):
            for row in params["rows"]:
                self.chunks.setdefault(row["chunk_id"], {}).update(row)
            return _Result([])
        if q.startswith("UNWIND $rows AS row MATCH (n {id: row.entity_id})"):
            for row in params["rows"]:
                if row["entity_id"] in self.nodes and row["chunk_id"] in self.chunks:
                    self.edges[(row["entity_id"], "hasSourceChunk", row["chunk_id"])] = {}
            return _Result([])
        m = re.search(r"MERGE \(a\)-\[r:(\w+)\]->\(b\) (ON CREATE SET|SET) r \+= \$props", q)
        if m and q.startswith("OPTIONAL MATCH (a {id: $source})"):
            src, tgt = params["source"], params["target"]
            found_a, found_b = src in self.nodes, tgt in self.nodes
            if found_a and found_b:
                key = (src, m.group(1), tgt)
                if key not in self.edges:
                    self.edges[key] = dict(params["props"])
                elif m.group(2) == "SET":
                    self.edges[key].update(params["props"])
            return _Result([{"source_found": found_a, "target_found": found_b}])
        raise AssertionError(f"FakeGraph: unhandled Cypher: {q[:160]}")

    def props(self, node_id):
        return self.nodes[node_id]["props"]


@pytest.fixture
def graph():
    return FakeGraph()


@pytest.fixture
def engine():
    return GraphIngestionEngine(DOMAIN_CONFIG)


def _real_section(entity_id="ITA1961__Section_195", text="Any person responsible for paying to a non-resident..."):
    return {"id": entity_id, "type": "Section", "properties": {
        "hasText": text, "full_citation": "Section 195", "document_id": "ITA1961",
        "sourceMdPath": "/data/converted_md/ita1961.md", "content_hash": "realhash1", "confidence": 1.0,
    }}


def _stub_section(entity_id="ITA1961__Section_195"):
    return {"id": entity_id, "type": "Section", "properties": provenance.mark_stub({
        "full_citation": "Section 195", "cited_as": "Section 195", "confidence": 0.3,
        "extraction_notes": "Stub created from a citation in firm opinion OP-01",
    }, cited_in="opinion_OP_01")}


# --- defaults ---------------------------------------------------------------

def test_entity_and_relationship_without_provenance_default_to_ingested(graph, engine):
    engine.load_entities(graph, [_real_section(), _real_section("ITA1961__Section_196")], mode="base")
    engine.load_relationships(graph, [{"type": "REFERS_TO", "source": "ITA1961__Section_195",
                                       "target": "ITA1961__Section_196", "properties": {}}])
    assert graph.props("ITA1961__Section_195")["provenance"] == provenance.INGESTED_DOCUMENT
    assert graph.edges[("ITA1961__Section_195", "REFERS_TO", "ITA1961__Section_196")]["provenance"] == \
        provenance.INGESTED_DOCUMENT


def test_provenance_rank_and_normalize():
    assert provenance.normalize(None) == provenance.INGESTED_DOCUMENT
    assert provenance.rank("ingested_document") > provenance.rank("cited_reference") > provenance.rank("model_suggested")
    assert provenance.rank("something_unknown") < provenance.rank("model_suggested")
    props = provenance.mark_stub({"sourceMdPath": "/citing/doc.md"}, cited_in="X")
    assert props == {"sourceMdPath": "", "provenance": "cited_reference", "cited_in_document_id": "X"}


# --- direction 1: stub never overwrites a real node ------------------------------

@pytest.mark.parametrize("mode", ["base", "amendment"])
def test_stub_never_overwrites_or_downgrades_real_node(graph, engine, mode):
    engine.load_entities(graph, [_real_section()], mode="base")
    before = dict(graph.props("ITA1961__Section_195"))
    engine.load_entities(graph, [_stub_section()], mode=mode)
    after = graph.props("ITA1961__Section_195")
    assert after == before, "stub must not change ANY property of the real node"
    assert after["provenance"] == provenance.INGESTED_DOCUMENT
    assert after["sourceMdPath"] == "/data/converted_md/ita1961.md"
    assert "cited_in_document_id" not in after
    assert not any("__superseded__" in nid for nid in graph.nodes), "no historical clone for a stub write"


def test_legacy_node_without_provenance_is_protected_too(graph, engine):
    # A node loaded before 2026-10-14 has no provenance property at all.
    graph.nodes["ITA1961__Section_195"] = {"labels": {"Section"}, "props": {"id": "ITA1961__Section_195",
                                                                              "hasText": "legacy real text"}}
    engine.load_entities(graph, [_stub_section()], mode="base")
    assert graph.props("ITA1961__Section_195") == {"id": "ITA1961__Section_195", "hasText": "legacy real text"}


def test_cited_edge_never_downgrades_real_edge(graph, engine):
    engine.load_entities(graph, [
        {"id": "Article_12_TaxTreaty_USA", "type": "Article", "properties": {"hasText": "real article text"}},
        {"id": "TaxTreaty_USA", "type": "TaxTreaty", "properties": {"hasText": "real treaty"}},
    ], mode="base")
    engine.load_relationships(graph, [{"type": "PART_OF", "source": "Article_12_TaxTreaty_USA",
                                       "target": "TaxTreaty_USA", "properties": {"note": "real"}}])
    engine.load_relationships(graph, [{"type": "PART_OF", "source": "Article_12_TaxTreaty_USA",
                                       "target": "TaxTreaty_USA",
                                       "properties": {"provenance": "cited_reference", "note": "from citation"}}])
    edge = graph.edges[("Article_12_TaxTreaty_USA", "PART_OF", "TaxTreaty_USA")]
    assert edge == {"note": "real", "provenance": "ingested_document"}
    assert any("ON CREATE SET r += $props" in q for q in graph.queries)


# --- direction 2: real ingestion upgrades the stub --------------------------------

@pytest.mark.parametrize("mode", ["base", "amendment"])
def test_real_document_upgrades_stub(graph, engine, mode):
    engine.load_entities(graph, [_stub_section()], mode="base")
    assert graph.props("ITA1961__Section_195")["provenance"] == provenance.CITED_REFERENCE
    engine.load_entities(graph, [_real_section()], mode=mode)
    upgraded = graph.props("ITA1961__Section_195")
    assert upgraded["provenance"] == provenance.INGESTED_DOCUMENT
    assert upgraded["upgraded_from_provenance"] == provenance.CITED_REFERENCE
    assert upgraded["hasText"].startswith("Any person responsible")
    assert upgraded["sourceMdPath"] == "/data/converted_md/ita1961.md"
    assert upgraded["confidence"] == 1.0
    # stub residue is gone (SET n = $props, not +=)
    assert "cited_in_document_id" not in upgraded and "cited_as" not in upgraded
    assert not any("__superseded__" in nid for nid in graph.nodes), "a stub is not a prior version"


def test_real_edge_upgrades_cited_edge(graph, engine):
    engine.load_entities(graph, [
        {"id": "A", "type": "Article", "properties": {}}, {"id": "T", "type": "TaxTreaty", "properties": {}}], "base")
    engine.load_relationships(graph, [{"type": "PART_OF", "source": "A", "target": "T",
                                       "properties": {"provenance": "cited_reference"}}])
    engine.load_relationships(graph, [{"type": "PART_OF", "source": "A", "target": "T", "properties": {}}])
    assert graph.edges[("A", "PART_OF", "T")]["provenance"] == provenance.INGESTED_DOCUMENT


def test_stub_onto_stub_only_fills_gaps(graph, engine):
    first = _stub_section()
    first["properties"]["cited_as"] = "Section 195 (first citer)"
    engine.load_entities(graph, [first], mode="base")
    second = _stub_section()
    second["properties"]["cited_as"] = "Section 195 (second citer)"
    second["properties"]["extra_hint"] = "new key"
    engine.load_entities(graph, [second], mode="amendment")
    props = graph.props("ITA1961__Section_195")
    assert props["cited_as"] == "Section 195 (first citer)"
    assert props["extra_hint"] == "new key"
    assert props["provenance"] == provenance.CITED_REFERENCE
    assert "content_changed" not in props


def test_real_reingest_keeps_normal_amendment_versioning(graph, engine):
    engine.load_entities(graph, [_real_section()], mode="base")
    newer = _real_section(text="Amended text")
    newer["properties"]["content_hash"] = "realhash2"
    engine.load_entities(graph, [newer], mode="amendment")
    assert graph.props("ITA1961__Section_195")["hasText"] == "Amended text"
    assert "ITA1961__Section_195__superseded__realhash1" in graph.nodes


# --- end to end with real converter output -----------------------------------------

def _opinion_graph(fixtures_dir, prefix):
    import extract_opinion
    path = next((fixtures_dir / "opinions").glob(f"{prefix}*.md"))
    return extract_opinion.build_graph_from_opinion(extract_opinion.parse_opinion(path.read_text(), path.name), path.name)


def test_opinion_stubs_then_real_act_and_case(graph, engine, fixtures_dir):
    """OP-01 loaded first (all stubs), then the real Section 195 and a real
    judgment arrive: stubs upgrade; the opinion's RELIES_ON edges stay."""
    op = _opinion_graph(fixtures_dir, "OP-01")
    engine.load_entities(graph, op["entities"], mode="base")
    engine.load_chunk_nodes_and_links(graph, op["entities"], op["chunks"])
    engine.load_relationships(graph, op["relationships"])
    assert graph.props("ITA1961__Section_195")["provenance"] == provenance.CITED_REFERENCE
    assert ("FirmOpinion_OP_01", "RELIES_ON_PROVISION", "ITA1961__Section_195") in graph.edges
    assert len([e for e in graph.edges if e[1] == "hasSourceChunk"]) == len(op["chunks"])

    engine.load_entities(graph, [_real_section()], mode="amendment")
    assert graph.props("ITA1961__Section_195")["provenance"] == provenance.INGESTED_DOCUMENT
    assert ("FirmOpinion_OP_01", "RELIES_ON_PROVISION", "ITA1961__Section_195") in graph.edges

    # Re-ingesting the opinion afterwards must not downgrade the now-real node.
    engine.load_entities(graph, op["entities"], mode="base")
    assert graph.props("ITA1961__Section_195")["provenance"] == provenance.INGESTED_DOCUMENT
    assert graph.props("ITA1961__Section_195")["hasText"].startswith("Any person responsible")


def test_real_case_graph_then_opinion_stub_is_protected(graph, engine, fixtures_dir):
    """The real Engineering Analysis judgment graph (an extract_case_law.py
    fixture) is loaded first; OP-18 then cites the same case under a
    differently-worded name. Whether the stub fuzzy-resolves onto the real
    node or stays a separate stub, the real node must survive untouched."""
    case_graph = json.loads(next((fixtures_dir / "graphs").glob("325ebe7a_*_graph.json")).read_text())
    real = [e for e in case_graph["entities"] if e["properties"].get("confidence") == 0.8 and e["type"] == "CaseLaw"]
    engine.load_entities(graph, real, mode="base")
    real_id = real[0]["id"]
    before = dict(graph.props(real_id))

    op = _opinion_graph(fixtures_dir, "OP-18")
    engine.load_entities(graph, op["entities"], mode="base")
    assert graph.props(real_id) == before
    assert graph.props(real_id)["provenance"] == provenance.INGESTED_DOCUMENT
