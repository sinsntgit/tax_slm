"""
2026-10-14: NEW (phase0/opinions) - the wiring around extract_opinion.py:
pipeline.py dispatch, opinion_agent's chunk->FirmOpinion join, answer_generation
v7, provenance reaching the answer model, and provenance on the EXISTING
converters' stub nodes (case law precedents, superseded circulars, amending Acts).
"""
import sys
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONVERTERS = REPO / "domain_packs" / "tax" / "converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

import chunking  # noqa: E402
import extract_case_law  # noqa: E402
import extract_circular  # noqa: E402
import extract_entities  # noqa: E402
import extract_opinion  # noqa: E402
from framework.core.ingest import provenance  # noqa: E402


def _op_graph(prefix="OP-01"):
    path = next((REPO / "tests" / "fixtures" / "opinions").glob(f"{prefix}*.md"))
    return extract_opinion.build_graph_from_opinion(extract_opinion.parse_opinion(path.read_text(), path.name), path.name)


# --- pipeline.py --------------------------------------------------------------

def test_firm_opinion_file_type_registered():
    from domain_packs.tax import pipeline
    ft = pipeline.FILE_TYPES["firm_opinion"]
    assert set(ft["accept"].split(",")) == {".md", ".txt", ".pdf"}
    assert ft["defaults"]["source_type"] == "FirmOpinion"
    assert "firm_opinion" not in pipeline._ENTRY_POINT_REFRESH  # chunk vector index covers it


def test_convert_and_extract_dispatch(monkeypatch, tmp_path):
    from domain_packs.tax import pipeline
    calls = []

    def fake_run(cmd, timeout):
        calls.append(cmd)
        out = Path(cmd[cmd.index("--out") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("{}")
        return 0, []

    monkeypatch.setattr(pipeline, "_run_subprocess", fake_run)
    monkeypatch.setattr(pipeline, "GRAPH_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "MD_DIR", tmp_path)
    up = tmp_path / "abc_OP-01.md"
    up.write_text("x")
    pipeline.convert_document(up, "firm_opinion")
    assert calls[-1][1].endswith("pdf_to_md_generic.py")
    pipeline.run_extraction(up, "firm_opinion", {"document_id": "abc_OP-01", "cited_act_namespace": "ITA2025"})
    cmd = calls[-1]
    assert cmd[1].endswith("extract_opinion.py")
    assert cmd[cmd.index("--document-id") + 1] == "abc_OP-01"
    assert cmd[cmd.index("--file-type") + 1] == "firm_opinion"
    assert cmd[cmd.index("--cited-act-namespace") + 1] == "ITA2025"


# --- opinion retrieval join --------------------------------------------------------

def test_opinion_agent_finds_firm_opinion_via_chunk_entity_id(monkeypatch):
    from domain_packs.tax.config import EMBED_CHUNK_METADATA_KEYS
    from framework.core.retrieve import opinion_agent
    assert "entity_id" in EMBED_CHUNK_METADATA_KEYS  # embedder writes Chunk.entity_id
    graphs = [_op_graph("OP-01"), _op_graph("OP-18")]
    chunk_to_entity = {c["chunk_id"]: c["entity_id"] for g in graphs for c in g["chunks"]}
    nodes = {e["id"]: e for g in graphs for e in g["entities"]}

    class FakeRetriever:
        def vector_search_chunks(self, q, top_k=10):
            return [{"chunk_id": cid, "text": "", "similarity": 0.9} for cid in list(chunk_to_entity)[:top_k]]

        def get_entities_for_chunks(self, chunk_ids):  # mirrors graphrag's c.entity_id -> e.id join
            out = {}
            for cid in chunk_ids:
                e = nodes[chunk_to_entity[cid]]
                out[e["id"]] = {"db_id": e["id"], "id": e["id"], "label": e["type"], "properties": e["properties"]}
            return list(out.values())

    monkeypatch.setattr(opinion_agent, "_fetch_supersession_status", lambda ids: {})
    res = opinion_agent.retrieve_opinions(FakeRetriever(), {"tax_position": "withholding on FTS"})
    assert res["candidates_found"] == 2
    assert {o["id"] for o in res["latest"]} == {"FirmOpinion_OP_01", "FirmOpinion_OP_18"}
    assert res["latest"][0]["citation"].startswith("OP-")


# --- answer_generation v7 -------------------------------------------------------

def test_answer_generation_v7_is_v6_plus_provenance_section():
    from domain_packs.tax.prompts.tax_prompts import PROMPT_PACK, ANSWER_GENERATION_V7_PROVENANCE_SECTION
    v6 = PROMPT_PACK.get_prompt("answer_generation", "v6").template
    v7 = PROMPT_PACK.get_prompt("answer_generation", "v7").template
    assert PROMPT_PACK.get_prompt("answer_generation").template == v7  # latest
    assert "3B. PROVENANCE" not in v6 and "(v6)" in v6
    assert "# 3B. PROVENANCE OF GRAPH FACTS" in v7
    assert v7.index("# 3A.") < v7.index("# 3B.") < v7.index("# 4. USE ALL RELEVANT CONTEXT")
    assert "not verified against the authority's text" in v7
    assert v7.replace(ANSWER_GENERATION_V7_PROVENANCE_SECTION, "").replace("(v7)", "(v6)") == v6


def test_document_details_carry_provenance(monkeypatch):
    from framework.core.retrieve import graphrag

    class Rec(dict):
        pass

    class Sess:
        def run(self, q, **p):
            assert "e.provenance AS provenance" in q
            return [Rec(chunk_id="c1", chunk_citation="x", page_number=None, document_id="d", document_name="OP-01",
                        source_type="FirmOpinion", authority_level="Firm", jurisdiction="India",
                        entity_citation="OP-01", entity_labels=["FirmOpinion"], provenance="ingested_document")]

    @contextmanager
    def fake_session():
        yield Sess()

    monkeypatch.setattr(graphrag.neo4j_client, "get_session", fake_session)
    retr = graphrag.GraphRAGRetriever.__new__(graphrag.GraphRAGRetriever)
    d = graphrag.GraphRAGRetriever.get_document_details(retr, ["c1"])
    assert d["c1"]["provenance"] == "ingested_document" and d["c1"]["entity_type"] == "FirmOpinion"


def test_chunk_label_includes_provenance_in_app_source():
    src = (REPO / "framework" / "webapp" / "app.py").read_text(encoding="utf-8")
    start = src.index("def _chunk_source_label")
    body = src[start:src.index("def _build_context_parts")]
    assert "provenance=" in body and "firm_opinion" in body


# --- existing converters' stubs ------------------------------------------------------

def test_case_law_precedent_stubs_are_cited_reference():
    extraction = {"citation": "A v. B", "court": "Supreme Court of India", "date": "2021",
                  "precedents_followed": ["C v. D"], "precedents_overruled": ["E v. F"], "facts": "facts"}
    g = extract_case_law.build_graph_from_extraction(extraction, "a.md", source_md_path="/md/a.md")
    ents = {e["id"]: e["properties"] for e in g["entities"]}
    assert ents["CaseLaw_A_v_B"]["provenance"] == provenance.INGESTED_DOCUMENT
    assert ents["CaseLaw_A_v_B"]["sourceMdPath"] == "/md/a.md"
    for sid in ("CaseLaw_C_v_D", "CaseLaw_E_v_F"):
        assert ents[sid]["provenance"] == provenance.CITED_REFERENCE
        assert ents[sid]["sourceMdPath"] == ""
    assert all(p["provenance"] for p in ents.values())


def test_circular_supersedes_stub_is_cited_reference():
    extraction = {"circular_number": "3 of 2026", "subject": "s", "supersedes": ["Circular 1 of 2020"]}
    meta = {"document_id": "c3", "document_name": "c3", "source_type": "Circular", "authority_level": "Delegated",
            "jurisdiction": "India", "file_type": "circular_pdf", "source_md_path": "/md/c3.md"}
    g = extract_circular.build_graph_from_extraction(extraction, "text", "c3.md", meta, "ITA1961", "ITA2025")
    ents = {e["id"]: e["properties"] for e in g["entities"]}
    assert ents["CBDTCircular_3_of_2026"]["provenance"] == provenance.INGESTED_DOCUMENT
    stub = ents["CBDTCircular_Circular_1_of_2020"]
    assert stub["provenance"] == provenance.CITED_REFERENCE and stub["sourceMdPath"] == ""
    assert stub["cited_in_document_id"] == "c3"


def test_amending_act_stub_is_cited_reference():
    builder = chunking._ChunkBuilder({}, namespace="ITA1961")
    act_id = builder._ensure_amending_act("7", "2025")
    props = builder.entities[act_id]["properties"]
    assert props["provenance"] == provenance.CITED_REFERENCE and props["sourceMdPath"] == ""
    e = {"id": act_id, "properties": props}
    extract_entities.attach_pipeline_metadata(e, {"document_id": "ITA1961", "document_name": "Act", "source_type": "Act",
                                                  "authority_level": "Statute", "jurisdiction": "India",
                                                  "source_md_path": "/md/act.md"}, "now", None)
    assert props["provenance"] == provenance.CITED_REFERENCE and props["sourceMdPath"] == ""
    real = {"id": "x", "properties": {"hasText": "t"}}
    extract_entities.attach_pipeline_metadata(real, {"document_id": "d", "document_name": "n", "source_type": "Act",
                                                     "authority_level": "Statute", "jurisdiction": "India"}, "now", None)
    assert real["properties"]["provenance"] == provenance.INGESTED_DOCUMENT
