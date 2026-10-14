"""2026-10-14: NEW - duplicate upload/ingest prevention (domain_packs/tax/
ingest_dedup.py + pipeline.py hooks) and the merge_duplicate_caselaw.py
cleanup fixes. Neo4j is faked; extract_case_law.build_graph_from_extraction()
is the REAL converter, run twice with differently-worded LLM citations the
way the owner's Samsung/Vodafone duplicates arose."""
import json
import sys
from pathlib import Path

import pytest

from domain_packs.tax import ingest_dedup, pipeline
from framework.cli import merge_duplicate_caselaw as merge

CONVERTERS = Path(__file__).resolve().parents[1] / "domain_packs" / "tax" / "converters"


@pytest.fixture(scope="module")
def extract_case_law():
    sys.path.insert(0, str(CONVERTERS))
    try:
        import extract_case_law as mod
    finally:
        sys.path.remove(str(CONVERTERS))
    return mod


def _extraction(citation):
    return {
        "citation": citation, "court": "Karnataka High Court", "date": "2010-03-12",
        "facts": "Payments for shrink-wrapped software...", "final_outcome": "Appeal allowed for Revenue.",
        "issues": ["Whether software payment is royalty", "Whether TDS applies"],
        "issue_outcomes": [{"issue": "Whether software payment is royalty", "winner": "revenue"}],
        "findings": [{"issue": "Whether software payment is royalty", "holding": "Royalty", "reasoning": "Copyright..."},
                     {"issue": "Whether TDS applies", "holding": "Yes", "reasoning": "Section 195..."}],
        "assessee_arguments": ["It is a sale of goods"], "revenue_arguments": ["It is a licence"],
        "precedents_followed": ["Tata Consultancy Services v. State of AP"],
        "precedents_overruled": ["Some Older Tribunal Case"],
        "legal_principles": ["Licence of copyright is royalty"],
    }


# --- upload layer ---------------------------------------------------------------

def test_identical_bytes_reuse_one_upload(tmp_path):
    p1, i1 = ingest_dedup.save_upload_dedup(tmp_path, "Samsung.pdf", b"%PDF judgment bytes")
    p2, i2 = ingest_dedup.save_upload_dedup(tmp_path, "Samsung.pdf", b"%PDF judgment bytes")
    p3, i3 = ingest_dedup.save_upload_dedup(tmp_path, "Samsung (1).pdf", b"%PDF judgment bytes")
    assert p1 == p2 == p3 and len(list(tmp_path.iterdir())) == 1
    assert p1.name == f"{i1['content_sha256'][:8]}_Samsung.pdf"
    assert (i1["duplicate_upload"], i2["duplicate_upload"], i3["duplicate_upload"]) == (False, True, True)


def test_different_bytes_same_name_is_a_new_upload(tmp_path):
    p1, _ = ingest_dedup.save_upload_dedup(tmp_path, "order.pdf", b"version 1")
    p2, i2 = ingest_dedup.save_upload_dedup(tmp_path, "order.pdf", b"version 2")
    assert p1 != p2 and not i2["duplicate_upload"]


def test_legacy_uuid_prefixed_upload_is_recognised(tmp_path):
    legacy = tmp_path / "a1b2c3d4_Vodafone.pdf"  # pre-fix random prefix
    legacy.write_bytes(b"vodafone judgment")
    (tmp_path / "ffffffff_Vodafone.pdf").write_bytes(b"other bytes, same len!")  # same name, different bytes
    p, info = ingest_dedup.save_upload_dedup(tmp_path, "Vodafone.pdf", b"vodafone judgment")
    assert p == legacy and info["duplicate_upload"]


def test_pipeline_save_upload_is_content_addressed(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "UPLOAD_DIR", tmp_path)
    assert pipeline.save_upload("x.pdf", b"abc") == pipeline.save_upload("x.pdf", b"abc")


# --- graph layer ------------------------------------------------------------------

def test_stamp_skips_precedent_stubs(extract_case_law):
    g = extract_case_law.build_graph_from_extraction(_extraction("Samsung Electronics Co. Ltd. v. ITO"), "s.md")
    n = ingest_dedup.stamp_source_file_hash(g, "h" * 64)
    stubs = [e for e in g["entities"] if (e["properties"].get("extraction_notes") or "").startswith("Stub created")]
    assert stubs and all(ingest_dedup.SOURCE_FILE_HASH_PROP not in e["properties"] for e in stubs)
    assert n == len(g["entities"]) - len(stubs)


def test_reingest_with_reworded_citation_converges_on_existing_ids(extract_case_law):
    first = extract_case_law.build_graph_from_extraction(_extraction("Samsung Electronics Co. Ltd. v. ITO"), "s.md")
    second = extract_case_law.build_graph_from_extraction(
        _extraction("M/s Samsung Electronics Co Ltd vs Income Tax Officer"), "s.md")
    old_id, new_id = ingest_dedup.main_case_law_id(first), ingest_dedup.main_case_law_id(second)
    assert old_id != new_id  # the root cause: LLM wording changes the slugged id

    n = ingest_dedup.converge_case_law_ids(second, second["chunks"], new_id, old_id)
    assert n > 0
    case_scoped = lambda g: {e["id"] for e in g["entities"] if e["type"] not in ("CaseCitation",)}
    assert case_scoped(second) == case_scoped(first)
    assert {c["chunk_id"] for c in second["chunks"]} == {c["chunk_id"] for c in first["chunks"]}
    assert {(r["source"], r["type"], r["target"]) for r in second["relationships"] if r["type"] != "HAS_CITATION"} == \
           {(r["source"], r["type"], r["target"]) for r in first["relationships"] if r["type"] != "HAS_CITATION"}
    for e in second["entities"]:
        for cid in (e["properties"].get("source_chunk_ids") or "").split(","):
            assert new_id not in cid


def test_converge_never_touches_a_stub_sharing_the_prefix():
    g = {"entities": [
        {"id": "CaseLaw_Samsung", "type": "CaseLaw", "properties": {"node_id": "CaseLaw_Samsung"}},
        {"id": "CaseLaw_Samsung_Heavy_Industries", "type": "CaseLaw", "properties": {}},
    ], "relationships": [{"type": "FOLLOWS", "source": "CaseLaw_Samsung", "target": "CaseLaw_Samsung_Heavy_Industries"}]}
    ingest_dedup.converge_case_law_ids(g, [], "CaseLaw_Samsung", "CaseLaw_Samsung_Electronics_v_ITO")
    assert [e["id"] for e in g["entities"]] == ["CaseLaw_Samsung_Electronics_v_ITO", "CaseLaw_Samsung_Heavy_Industries"]
    assert g["relationships"][0]["target"] == "CaseLaw_Samsung_Heavy_Industries"


# --- ingestion job ------------------------------------------------------------------

@pytest.fixture
def job_env(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "MD_DIR", tmp_path)
    upload = tmp_path / "abcd1234_Samsung.pdf"
    upload.write_bytes(b"samsung pdf")
    calls = {"convert": 0, "loaded": None}

    def fake_convert(upload_path, file_type):
        calls["convert"] += 1
        md = tmp_path / f"{upload_path.stem}.md"
        md.write_text("judgment text")
        return md, {}, []

    monkeypatch.setattr(pipeline, "convert_document", fake_convert)
    monkeypatch.setattr(pipeline, "refresh_entry_point_index", lambda ft: "refreshed")
    monkeypatch.setattr(pipeline.graphrag_module, "invalidate_schema_cache", lambda: None)

    def fake_ingest(cfg, graph_path, chunks_path, mode):
        calls["loaded"] = json.loads(Path(graph_path).read_text())
        calls["chunks"] = json.loads(Path(chunks_path).read_text())
        return {"entities_in_file": len(calls["loaded"]["entities"]), "relationships_in_file": 0}

    monkeypatch.setattr(pipeline.pipeline_ops, "ingest_graph_json", fake_ingest)
    monkeypatch.setattr(pipeline.pipeline_ops, "embed_chunks_json", lambda cfg, p: {"chunks_embedded": 1})
    return tmp_path, upload, calls


def _run_job(upload, force=False):
    job_id = "t" + str(force)
    with pipeline._JOBS_LOCK:
        pipeline._JOBS[job_id] = {"status": "running", "log": [], "result": None, "error": None,
                                  "started_at": 0, "finished_at": None}
    pipeline._run_ingestion_job(job_id, upload, "caselaw_pdf", {"document_id": "Samsung"}, "base", force)
    return pipeline.get_job(job_id)


def test_duplicate_file_is_skipped_with_clear_log(job_env, monkeypatch):
    _tmp, upload, calls = job_env
    monkeypatch.setattr(ingest_dedup, "find_ingested_copies", lambda sha, md=None: [
        {"document_id": "Samsung", "document_name": "Samsung v ITO", "file_type": "caselaw_pdf", "nodes": 12}])
    job = _run_job(upload)
    assert job["status"] == "done" and job["result"]["skipped"] is True
    assert calls["convert"] == 0 and calls["loaded"] is None
    assert any(line.startswith("SKIPPED - this exact file") and "force_reingest" in line for line in job["log"])


def test_force_reingest_converges_and_stamps(job_env, monkeypatch, extract_case_law):
    tmp, upload, calls = job_env
    monkeypatch.setattr(ingest_dedup, "find_ingested_copies", lambda sha, md=None: [
        {"document_id": "Samsung", "document_name": "x", "file_type": "caselaw_pdf", "nodes": 12}])
    old = extract_case_law.build_graph_from_extraction(_extraction("Samsung Electronics Co. Ltd. v. ITO"), "s.md")
    old_id = ingest_dedup.main_case_law_id(old)
    monkeypatch.setattr(ingest_dedup, "find_case_law_ids_for_file", lambda sha, md=None: [old_id])

    def fake_extract(md_path, file_type, doc_meta):
        g = extract_case_law.build_graph_from_extraction(
            _extraction("M/s Samsung Electronics Co Ltd vs Income Tax Officer"), md_path.name, doc_meta=doc_meta)
        gp, cp = tmp / "g.json", tmp / "c.json"
        gp.write_text(json.dumps({"entities": g["entities"], "relationships": g["relationships"]}))
        cp.write_text(json.dumps(g["chunks"]))
        return gp, cp, {}, []

    monkeypatch.setattr(pipeline, "run_extraction", fake_extract)
    job = _run_job(upload, force=True)
    assert job["status"] == "done", job["log"]
    assert calls["convert"] == 1
    ids = {e["id"] for e in calls["loaded"]["entities"]}
    assert old_id in ids and ingest_dedup.main_case_law_id(calls["loaded"]) == old_id
    assert all(c["chunk_id"].startswith(f"caselaw_{old_id}_") for c in calls["chunks"])
    sha = ingest_dedup.sha256_of_file(upload)
    main = next(e for e in calls["loaded"]["entities"] if e["id"] == old_id)
    assert main["properties"][ingest_dedup.SOURCE_FILE_HASH_PROP] == sha
    assert any("Re-ingesting anyway" in line for line in job["log"])


def test_batch_counts_skipped(job_env, monkeypatch):
    _tmp, upload, calls = job_env
    monkeypatch.setattr(ingest_dedup, "find_ingested_copies", lambda sha, md=None: [{"document_id": "d", "nodes": 1}])
    with pipeline._JOBS_LOCK:
        pipeline._JOBS["b1"] = {"type": "batch", "status": "running", "log": [], "total": 1, "completed": 0,
                                "failed": 0, "skipped": 0, "current_file": None, "files": [], "result": None,
                                "error": None, "started_at": 0, "finished_at": None}
    pipeline._run_batch_ingestion_job("b1", [upload], "caselaw_pdf", {}, "base")
    job = pipeline.get_batch_job("b1")
    assert job["skipped"] == 1 and job["failed"] == 0 and job["files"][0]["status"] == "skipped"
    assert calls["convert"] == 0


def test_upload_route_reports_already_ingested(tmp_path, monkeypatch):
    from framework.webapp import app as webapp
    import io
    monkeypatch.setattr(pipeline, "UPLOAD_DIR", tmp_path)
    monkeypatch.setattr(ingest_dedup, "find_ingested_copies",
                        lambda sha, md=None: [{"document_id": "Samsung", "nodes": 3}])
    client = webapp.app.test_client()
    data = {"file": (io.BytesIO(b"same bytes"), "Samsung.pdf")}
    r1 = client.post("/api/upload", data=data, content_type="multipart/form-data").get_json()
    r2 = client.post("/api/upload", data={"file": (io.BytesIO(b"same bytes"), "Samsung.pdf")},
                     content_type="multipart/form-data").get_json()
    assert r1["upload_path"] == r2["upload_path"] and r2["duplicate_upload"] is True
    assert r2["already_ingested"][0]["document_id"] == "Samsung"


# --- merge_duplicate_caselaw.py ---------------------------------------------------

def test_cluster_joins_same_source_file_even_when_names_differ():
    nodes = [{"id": "CaseLaw_A", "case_name": "M/s Samsung Electronics Co. Ltd. v. ITO"},
             {"id": "CaseLaw_B", "case_name": "Samsung Electronics Co Ltd vs Income Tax Officer (2010) 320 ITR 209"},
             {"id": "CaseLaw_C", "case_name": "Vodafone International Holdings BV v Union of India"}]
    assert merge._cluster(nodes, {"CaseLaw": ["case_name"]}, 95) == []
    clusters = merge._cluster(nodes, {"CaseLaw": ["case_name"]}, 95,
                              {"CaseLaw_A": "md:x", "CaseLaw_B": "md:x", "CaseLaw_C": "md:y"})
    assert [sorted(n["id"] for n in c) for c in clusters] == [["CaseLaw_A", "CaseLaw_B"]]


def test_file_key_uses_hash_then_md_content(tmp_path):
    md1, md2 = tmp_path / "a1_S.md", tmp_path / "b2_S.md"
    md1.write_text("same converted text")
    md2.write_text("same converted text")
    k1 = merge._file_key({"node_id": "x", "sourceMdPath": str(md1)})
    assert k1 and k1 == merge._file_key({"node_id": "y", "sourceMdPath": str(md2)})
    assert merge._file_key({"node_id": "x", "source_file_sha256": "abc"}) == "file:abc"
    assert merge._file_key({"sourceMdPath": str(md1)}) is None  # precedent stub (no node_id)


class _Res:
    def __init__(self, rows):
        self.rows = rows

    def __iter__(self):
        return iter(self.rows)

    def single(self):
        return self.rows[0] if self.rows else None


class _MergeSession:
    def __init__(self):
        self.writes = []

    def run(self, q, **p):
        if "RETURN properties(n)" in q:
            return _Res([{"p": {"id": p["id"], "case_name": p["id"]}}])
        if "-[r]->(b)" in q and "RETURN type(r)" in q:  # outgoing
            return _Res([{"rtype": "hasSourceChunk", "props": {}, "other_eid": "4:c:1", "other_id": None},
                         {"rtype": "INTERPRETS", "props": {}, "other_eid": "4:s:9", "other_id": "ITA1961__Section_9"}])
        if "(b)-[r]->(n:CaseLaw" in q and "RETURN type(r)" in q:  # incoming
            return _Res([{"rtype": "FOLLOWS", "props": {}, "other_eid": "4:k:2", "other_id": "CaseLaw_Other"}])
        if "count(n) AS c" in q:
            return _Res([{"c": 7}])
        self.writes.append((q, p))
        return _Res([])


def test_merge_repoints_chunks_by_element_id():
    s = _MergeSession()
    plan = merge._merge_duplicate_into_canonical(s, "CaseLaw_B", "CaseLaw_A", dry_run=False)
    assert plan["relationships_repointed"] == 3
    repoints = [p for q, p in s.writes if "MERGE" in q]
    assert {p["other_eid"] for p in repoints} == {"4:c:1", "4:s:9", "4:k:2"}  # the Chunk edge is no longer lost
    assert all("elementId(b) = $other_eid" in q for q, _ in s.writes if "MERGE" in q)
    assert any("DETACH DELETE" in q and p.get("id") == "CaseLaw_B" for q, p in s.writes)


def test_drop_subtree_dry_run_writes_nothing():
    s = _MergeSession()
    plan = merge._merge_duplicate_into_canonical(s, "CaseLaw_B", "CaseLaw_A", dry_run=True, drop_subtree=True)
    assert plan["subtree_nodes_dropped"] == 7 and s.writes == []
