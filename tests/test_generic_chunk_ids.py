"""
2026-10-14: NEW (phase0/followups) - extract_generic.py chunk ids are
namespaced per document (`<document_id>__<i>`), so two generic documents (or
a generic document and an un-namespaced Act run) can no longer MERGE onto
each other's Chunk nodes, which graph_loader MERGEs on chunk_id alone.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONVERTERS = REPO / "domain_packs" / "tax" / "converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

import extract_generic  # noqa: E402

TEXT_A = ("CBDT Circular on TDS.\n\n" + ("Paragraph about deduction of tax at source. " * 40) + "\n\n"
          + ("Second paragraph about the due date for deposit. " * 40))
TEXT_B = ("Firm memo on royalty.\n\n" + ("Discussion of software royalty under section 9. " * 40) + "\n\n"
          + ("Treaty position under Article 12 of the DTAA. " * 40))


def test_generic_chunk_id_shape_matches_act_namespacing():
    import extract_entities
    assert extract_generic.generic_chunk_id(3, "circ_23_2024") == "circ_23_2024__3"
    # same "<namespace>__<i>" shape as the Act helper
    assert extract_generic.generic_chunk_id(3, "X") == extract_entities.act_chunk_id(3, "X")
    # commas would break graph_loader's comma-joined source_chunk_ids
    assert "," not in extract_generic.generic_chunk_id(0, "memo,v2")
    with pytest.raises(ValueError):
        extract_generic.generic_chunk_id(0, "  ")


def _run_main(tmp_path, monkeypatch, name, text):
    md = tmp_path / f"{name}.md"
    md.write_text(text, encoding="utf-8")
    out = tmp_path / f"{name}_graph.json"
    chunks_out = tmp_path / f"{name}_chunks.json"
    monkeypatch.setattr(extract_generic.audit_log, "start_run", lambda **k: "run-1")
    monkeypatch.setattr(extract_generic.audit_log, "finish_run", lambda *a, **k: None)
    monkeypatch.setattr(extract_generic.observability, "start_trace", lambda *a, **k: None)
    monkeypatch.setattr(extract_generic.observability, "end_trace", lambda *a, **k: None)
    monkeypatch.setattr(extract_generic, "extract_document_temporal_metadata", lambda *a, **k: {})
    monkeypatch.setattr(sys, "argv", [
        "extract_generic.py", "--md", str(md), "--schema", str(REPO / "domain_packs/tax/ontology/schema.json"),
        "--out", str(out), "--chunks-out", str(chunks_out), "--skip-enrichment", "--skip-datasource",
        "--target-chunk-tokens", "200",
    ])
    extract_generic.main()
    return json.loads(out.read_text()), json.loads(chunks_out.read_text())


def test_two_generic_documents_get_disjoint_consistent_chunk_ids(tmp_path, monkeypatch):
    graph_a, chunks_a = _run_main(tmp_path, monkeypatch, "doc_a", TEXT_A)
    graph_b, chunks_b = _run_main(tmp_path, monkeypatch, "doc_b", TEXT_B)
    ids_a = [c["chunk_id"] for c in chunks_a]
    ids_b = [c["chunk_id"] for c in chunks_b]
    assert len(ids_a) >= 2 and len(ids_b) >= 2
    assert ids_a == [f"doc_a__{i}" for i in range(len(ids_a))]
    assert not set(ids_a) & set(ids_b)
    # the DocumentSection entities point at exactly those (namespaced) chunk ids,
    # so graph_loader's hasSourceChunk links + chunk metadata lookup line up
    for graph, chunks in ((graph_a, chunks_a), (graph_b, chunks_b)):
        by_entity = {c["entity_id"]: c["chunk_id"] for c in chunks}
        sections = [e for e in graph["entities"] if e["type"] == "DocumentSection"]
        assert sections
        for e in sections:
            assert e["properties"]["source_chunk_ids"] == by_entity[e["id"]]
            assert e["properties"]["chunk_id"] == by_entity[e["id"]]


def test_graph_loader_merges_distinct_chunk_nodes_for_two_documents(tmp_path, monkeypatch, fake_neo4j):
    from domain_packs.tax.config import DOMAIN_CONFIG
    from framework.core.infra import neo4j_client
    from framework.core.ingest.graph_loader import GraphIngestionEngine

    graph_a, chunks_a = _run_main(tmp_path, monkeypatch, "doc_a", TEXT_A)
    graph_b, chunks_b = _run_main(tmp_path, monkeypatch, "doc_b", TEXT_B)
    loader = GraphIngestionEngine(DOMAIN_CONFIG)
    merged_rows = []
    with neo4j_client.get_session() as session:
        for graph, chunks in ((graph_a, chunks_a), (graph_b, chunks_b)):
            loader.load_chunk_nodes_and_links(session, graph["entities"], chunks)
            merged_rows += fake_neo4j.last("MERGE (c:Chunk {chunk_id: row.chunk_id})").params["rows"]
    ids = [r["chunk_id"] for r in merged_rows]
    assert len(ids) == len(set(ids)) == len(chunks_a) + len(chunks_b)
    # metadata from chunks.json was found for every namespaced id (string lookup)
    assert all(r.get("full_citation") for r in merged_rows)
