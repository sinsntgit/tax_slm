"""2026-10-14: NEW - md_hydration.py (live text from converted .md files) and
local_vector_store.py (flat-file .npz entry-point store, VECTOR_STORE_DIR)."""
import numpy as np
import pytest

from framework.core.retrieve import local_vector_store as lvs
from framework.core.retrieve import md_hydration as mh


# --- md_hydration ------------------------------------------------------------

@pytest.fixture
def md_file(tmp_path):
    p = tmp_path / "act.md"
    p.write_text("line1\nline2\nline3\nline4\n", encoding="utf-8")
    return p


def test_hydrate_missing_path_and_file(tmp_path):
    assert mh.hydrate_text("") == ("", "no sourceMdPath stored on this entity")
    text, note = mh.hydrate_text(str(tmp_path / "nope.md"))
    assert text == "" and note.startswith("source file not found")


def test_hydrate_whole_file_and_truncation(md_file):
    assert mh.hydrate_text(str(md_file)) == ("line1\nline2\nline3\nline4\n", "")
    text, note = mh.hydrate_text(str(md_file), whole_file_max_chars=7)
    assert text == "line1\nl" and "truncated" in note
    # only ONE of line_start/line_end given -> still whole-file
    assert mh.hydrate_text(str(md_file), line_start=2)[0].startswith("line1")


def test_hydrate_line_range_is_one_indexed_inclusive_and_clamped(md_file):
    assert mh.hydrate_text(str(md_file), 2, 3) == ("line2\nline3", "")
    assert mh.hydrate_text(str(md_file), 0, 1) == ("line1", "")          # start clamped to 1
    assert mh.hydrate_text(str(md_file), "3", "99")[0] == "line3\nline4\n"  # end clamped, str ints ok


@pytest.mark.parametrize("start,end", [(10, 12), (3, 2)])
def test_hydrate_out_of_range(md_file, start, end):
    text, note = mh.hydrate_text(str(md_file), start, end)
    assert text == "" and "out of range" in note


def test_hydrate_entity_dispatch(md_file):
    props = {"sourceMdPath": str(md_file), "line_start": 4, "line_end": 4}
    assert mh.hydrate_entity(props) == ("line4", "")
    assert mh.hydrate_entity({})[0] == ""


def test_hydrate_tolerates_bad_utf8(tmp_path):
    p = tmp_path / "bad.md"
    p.write_bytes(b"ok \xff\xfe end")
    text, note = mh.hydrate_text(str(p))
    assert text.startswith("ok ") and text.endswith(" end") and note == ""


def test_hydrate_real_converted_fixture(fixtures_dir):
    """A real converted case-law .md (whole-file shape, as CaseLaw nodes use)."""
    p = next((fixtures_dir / "converted_md").glob("*Engineering_Analysis*.md"))
    text, note = mh.hydrate_text(str(p))
    assert note == "" and len(text) > 1000 and "software" in text.lower()


# --- local_vector_store --------------------------------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path / "vs"))
    return lvs


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return (v / np.linalg.norm(v)).tolist()


def test_store_absent_behaviour(store):
    assert not store.exists("idx")
    assert store.load("idx") == (None, None)
    assert store.search("idx", [1.0, 0.0], 5) == []
    assert store.list_all("idx") == [] and store.count("idx") == 0
    assert store.describe_path("idx").endswith("idx.npz")


def test_save_load_search_ranking(store):
    store.save("idx", ["a", "b", "c"], [_unit([1, 0]), _unit([0, 1]), _unit([1, 1])])
    assert store.exists("idx") and store.count("idx") == 3
    ids, vecs = store.load("idx")
    assert ids == ["a", "b", "c"] and vecs.shape == (3, 2) and vecs.dtype == np.float32
    hits = store.search("idx", _unit([1, 0.1]), 2)
    assert [h[0] for h in hits] == ["a", "c"]
    assert hits[0][1] == pytest.approx(float(np.dot(_unit([1, 0]), _unit([1, 0.1]))), rel=1e-5)
    # top_k larger than the store -> everything, still sorted
    assert [h[0] for h in store.search("idx", _unit([0, 1]), 99)] == ["b", "c", "a"]


def test_merge_and_save_upserts(store):
    store.save("idx", ["a", "b"], [_unit([1, 0]), _unit([0, 1])])
    total = store.merge_and_save("idx", ["b", "c"], [_unit([1, 0]), _unit([1, 1])])
    assert total == 3 and store.list_all("idx") == ["a", "b", "c"]
    # "b" was REPLACED (now points along x), so it ties "a" for an x query
    top2 = {h[0] for h in store.search("idx", _unit([1, 0]), 2)}
    assert top2 == {"a", "b"}


def test_merge_into_empty_store(store):
    assert store.merge_and_save("new", ["x"], [_unit([1, 2])]) == 1
    assert store.list_all("new") == ["x"]


def test_embed_nodes_end_to_end_with_fake_neo4j(store, fake_neo4j):
    """semantic_entry_points.embed_nodes() -> local store (no vectors to Neo4j),
    then search_entry_points() looks up metadata only for the picked ids."""
    from framework.core.retrieve import semantic_entry_points as sep
    fake_neo4j.on("WHERE coalesce(n.retrievalSummary", [
        {"db_id": "4:c:1", "retrievalSummary": "software royalty copyright"},
        {"db_id": "4:c:2", "retrievalSummary": "agricultural land capital gains"},
    ])
    fake_neo4j.on("elementId(n) IN $db_ids", lambda q, p: [
        {"db_id": i, "case_name": f"case {i}"} for i in p["db_ids"]])
    n = sep.embed_nodes(fake_neo4j.session(), label="CaseLaw", index_name="cl_idx",
                        embedding_text_fn=lambda r: r["retrievalSummary"],
                        text_property_names=["retrievalSummary"], where_not_empty_property="retrievalSummary")
    assert n == 2 and store.count("cl_idx") == 2
    assert not any("SET" in q.text for q in fake_neo4j.queries)
    hits = sep.search_entry_points("cl_idx", "software royalty copyright", 1, {"case_name": "case_name"})
    assert hits[0]["db_id"] == "4:c:1" and hits[0]["case_name"] == "case 4:c:1"
    assert fake_neo4j.last("elementId(n) IN $db_ids").params["db_ids"] == ["4:c:1"]

