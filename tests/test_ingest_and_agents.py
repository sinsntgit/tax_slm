"""2026-10-14: NEW - entity_resolution.py, graph_loader.sanitize_props /
diff_and_prepare_update (+ a mocked-session load_entities pass),
embedder model loading, tribunal_case_law_agent._extract_change_in_law_date,
domain_packs/tax/pipeline.refresh_entry_point_index."""
import json

import pytest

from framework.core.ingest import entity_resolution as er
from framework.core.ingest import graph_loader as gl


# --- entity_resolution -------------------------------------------------------------

@pytest.mark.parametrize("raw,norm", [
    ("Sec. 9(1)(vi)", "section 9 1 vi"),
    ("  Circular No. 7/2021 ", "circular number 7 2021"),
    ("Section   80-IAC", "section 80 iac"),
    ("", ""), (None, ""),
])
def test_normalize(raw, norm):
    assert er.normalize(raw) == norm


def test_resolution_key_uses_domain_props_then_falls_back_to_id():
    keys = {"Section": ["section_no"]}
    assert er.resolution_key("Section", {"section_no": "9"}, "x", keys) == "section 9"
    assert er.resolution_key("Section", {}, "Section_9", keys) == "section section_9"
    assert er.resolution_key("Other", {"section_no": "9"}, "O_1", keys) == "other o_1"


def test_similarity_and_best_match():
    assert er.similarity("", "a") == 0.0
    assert er.similarity("section 9", "9 section") == 100
    best_id, score = er.best_match("concept royalty", [("a", "concept fees"), ("b", "concept royalties")])
    assert best_id == "b" and 85 <= score < 100


def test_resolve_within_batch_merges_flags_and_keeps():
    keys = {"LegalConcept": ["name"]}
    ents = [
        {"id": "LC_royalty", "type": "LegalConcept", "properties": {"name": "Royalty", "def": "d1"}},
        {"id": "LC_royalty_2", "type": "LegalConcept", "properties": {"name": "royalty.", "src": "s2"}},
        {"id": "LC_royalties", "type": "LegalConcept", "properties": {"name": "Royalties"}},  # ~90: flag
        {"id": "LC_tds", "type": "LegalConcept", "properties": {"name": "TDS"}},
        {"id": "Royalty", "type": "Issue", "properties": {"name": "Royalty"}},  # other type: never compared
    ]
    kept, alias, flags = er.resolve_within_batch(ents, keys)
    kept_ids = [e["id"] for e in kept]
    assert alias == {"LC_royalty_2": "LC_royalty"}
    assert "LC_royalty_2" not in kept_ids and {"LC_royalty", "LC_tds", "Royalty"} <= set(kept_ids)
    merged = next(e for e in kept if e["id"] == "LC_royalty")["properties"]
    assert merged["src"] == "s2" and merged["def"] == "d1" and merged["also_known_as_ids"] == "LC_royalty_2"
    # medium similarity (85-95) -> kept AND flagged for review, never merged
    flagged = next(e for e in kept if e["id"] == "LC_royalties")
    assert flagged["properties"]["possibleDuplicateOf"] == "LC_royalty"
    assert flags == [{"kept_id": "LC_royalty", "candidate_id": "LC_royalties", "score": 90.5}]


# --- graph_loader.sanitize_props ------------------------------------------------------

def test_sanitize_props_primitives_lists_and_nested():
    out = gl.sanitize_props({
        "none": None, "s": "x", "n": 1, "empty_list": [], "empty_dict": {},
        "strs": ["a", "b"], "nums": [1, 2], "dict": {"k": 1},
        "list_of_dicts": [{"name": "ITA", "section": "9"}],
    })
    assert out == {"s": "x", "n": 1, "strs": ["a", "b"], "nums": [1, 2], "dict": '{"k": 1}',
                   "list_of_dicts": '[{"name": "ITA", "section": "9"}]'}


@pytest.mark.parametrize("mixed", [["9", 10], [1, "a", True], [1.5, "x"]])
def test_sanitize_props_mixed_type_list_is_serialized(mixed):
    """Neo4j rejects heterogeneous arrays ("Collections containing mixed types
    can not be stored in properties") - the docstring promises these are
    JSON-serialized, so the whole node write doesn't fail."""
    out = gl.sanitize_props({"v": mixed})
    assert out["v"] == json.dumps(mixed)


def test_sanitize_props_homogeneous_lists_stay_native():
    assert gl.sanitize_props({"b": [True, False], "f": [0.5, 1.5]}) == {"b": [True, False], "f": [0.5, 1.5]}


# --- graph_loader.diff_and_prepare_update ------------------------------------------------

OLD = {"id": "S9", "content_hash": "h1", "document_id": "doc_2024", "ingestion_date": "2024-01-01", "hasText": "old"}


def test_diff_new_node_and_base_mode():
    new = {"content_hash": "h2", "hasText": "new"}
    assert gl.diff_and_prepare_update(None, new, "amendment") is new
    assert gl.diff_and_prepare_update(OLD, new, "base") is new


def test_diff_amendment_unchanged_content_keeps_old_and_records_last_seen():
    new = {"content_hash": "h1", "document_id": "doc_2025", "ingestion_date": "2025-01-01", "hasText": "old"}
    out = gl.diff_and_prepare_update(OLD, new, "amendment")
    assert out["hasText"] == "old" and out["document_id"] == "doc_2024"
    assert out["last_seen_document_id"] == "doc_2025" and out["last_seen_ingestion_date"] == "2025-01-01"
    assert "content_changed" not in out


def test_diff_amendment_changed_content_preserves_origin():
    new = {"content_hash": "h2", "document_id": "doc_2025", "ingestion_date": "2025-01-01", "hasText": "new"}
    out = gl.diff_and_prepare_update(OLD, new, "amendment")
    assert out["hasText"] == "new" and out["content_changed"] is True
    assert out["origin_document_id"] == "doc_2024" and out["origin_ingestion_date"] == "2024-01-01"
    assert out["last_amended_by_document_id"] == "doc_2025"
    # a second amendment keeps the ORIGINAL origin
    again = gl.diff_and_prepare_update(out, {"content_hash": "h3", "document_id": "doc_2026"}, "amendment")
    assert again["origin_document_id"] == "doc_2024"


def test_diff_missing_hash_counts_as_changed():
    out = gl.diff_and_prepare_update({"document_id": "a"}, {"document_id": "b"}, "amendment")
    assert out["content_changed"] is True and out["origin_document_id"] == "a"


def test_get_existing_node_properties_uses_safe_label(fake_neo4j):
    fake_neo4j.on("MATCH (n:Section {id: $entity_id})", [{"n": {"id": "S9", "x": 1}}])
    assert gl.get_existing_node_properties(fake_neo4j.session(), "Section", "S9") == {"id": "S9", "x": 1}
    assert fake_neo4j.last("MATCH (n:Section").params == {"entity_id": "S9"}
    assert gl.get_existing_node_properties(fake_neo4j.session(), "CaseLaw", "nope") is None


# --- embedder ----------------------------------------------------------------------------

def test_embedder_works_with_old_and_new_sentence_transformers_api(monkeypatch):
    from framework.core.ingest import embedder

    class OldST:  # sentence-transformers < 5: only get_sentence_embedding_dimension()
        def __init__(self, name):
            pass

        def get_sentence_embedding_dimension(self):
            return 7

        def encode(self, texts, **k):
            import numpy as np
            return np.ones((len(texts), 7))

    monkeypatch.setattr(embedder, "SentenceTransformer", OldST)
    monkeypatch.setattr(embedder, "_model", None)
    monkeypatch.setattr(embedder, "_embedding_dim", None)
    assert embedder.embedding_dimension() == 7
    assert embedder.embed_batch(["a", "b"]) == [[1.0] * 7] * 2


def test_embedder_default_embedding_text():
    from framework.core.ingest import embedder
    assert embedder.default_embedding_text({"short_title": " T ", "text": "body"}) == "T. body"
    assert embedder.default_embedding_text({"content": "c"}) == "c"


# --- tribunal_case_law_agent._extract_change_in_law_date ----------------------------------

def _date(**findings):
    from framework.core.retrieve.tribunal_case_law_agent import _extract_change_in_law_date
    return _extract_change_in_law_date(findings)


def test_change_in_law_date_from_nested_node_props_most_recent():
    rows = [{"provision": {"effective_from": "2012-04-01"}, "r": {}, "act": {"amendingActYear": 2012}},
            {"provision": {"effective_to": "2020-04-01"}},
            {"provision": {"effective_from": "not a date"}}]
    assert _date(amendment_rows=rows) == "2020-04-01"


def test_change_in_law_date_prefers_amendment_rows_over_plain_rows():
    assert _date(amendment_rows=[{"a": {"effective_from": "2001-01-01"}}],
                 plain_rows=[{"b": {"effective_from": "2024-01-01"}}]) == "2001-01-01"


def test_change_in_law_date_falls_back_to_plain_rows_and_years():
    assert _date(amendment_rows=[{"a": {"x": 1}}], plain_rows=[{"b": {"effective_date": "2019-09-20"}}]) == "2019-09-20"
    assert _date(amendment_rows=[{"act": {"amendingActYear": 2016}}, {"act": {"amendingActYear": "2020"}}]) == "2020"
    # a scalar-alias row (top-level key) also counts
    assert _date(plain_rows=[{"effective_from": "2015-06-01"}]) == "2015-06-01"


def test_change_in_law_date_empty_when_unknown():
    assert _date() == ""
    assert _date(amendment_rows=None, plain_rows=["not a dict", {"n": {"amendingActYear": "20xx"}}]) == ""


# --- pipeline.refresh_entry_point_index --------------------------------------------------

def test_refresh_entry_point_index_unknown_type():
    from domain_packs.tax import pipeline
    assert "nothing to refresh" in pipeline.refresh_entry_point_index("generic")


@pytest.mark.parametrize("file_type,module_name,fn_name", [
    ("act_pdf", "section_entry_points", "embed_section_nodes"),
    ("caselaw_pdf", "case_law_entry_points", "embed_case_law_nodes"),
    ("circular_pdf", "circular_entry_points", "embed_circular_nodes"),
    ("treaty_pdf", "treaty_entry_points", "embed_treaty_nodes"),
])
def test_refresh_entry_point_index_calls_matching_embedder(fake_neo4j, monkeypatch, file_type, module_name, fn_name):
    import importlib
    from domain_packs.tax import pipeline
    mod = importlib.import_module(f"framework.core.retrieve.{module_name}")
    seen = {}

    def fake_embed(session):
        seen["session"] = session
        return 3
    monkeypatch.setattr(mod, fn_name, fake_embed)
    msg = pipeline.refresh_entry_point_index(file_type)
    assert msg == f"Semantic-graph entry-point index refreshed ({module_name}): 3 node(s) embedded."
    assert seen["session"] is not None and fake_neo4j.sessions_opened == 1


def test_refresh_entry_point_index_never_raises(fake_neo4j, monkeypatch):
    from domain_packs.tax import pipeline
    from framework.core.retrieve import case_law_entry_points
    monkeypatch.setattr(case_law_entry_points, "embed_case_law_nodes",
                        lambda s: (_ for _ in ()).throw(RuntimeError("model missing")))
    msg = pipeline.refresh_entry_point_index("caselaw_pdf")
    assert msg.startswith("WARNING:") and "model missing" in msg and "ingestion itself succeeded" in msg


def test_refresh_entry_point_index_real_embed_into_temp_store(fake_neo4j, monkeypatch, tmp_path):
    """No embedder monkeypatch: the real embed_case_law_nodes() -> stub
    SentenceTransformer -> local_vector_store under a temp VECTOR_STORE_DIR."""
    from domain_packs.tax import pipeline
    from framework.core.retrieve import local_vector_store
    monkeypatch.setenv("VECTOR_STORE_DIR", str(tmp_path))
    fake_neo4j.on("MATCH (n:CaseLaw)", [
        {"db_id": "4:c:1", "retrievalSummary": "royalty software", "retrievalKeywords": "royalty",
         "case_name": "EA", "full_citation": "EA (2021)"}])
    msg = pipeline.refresh_entry_point_index("caselaw_pdf")
    assert "1 node(s) embedded" in msg, msg
    stores = list(tmp_path.glob("*.npz"))
    assert len(stores) == 1 and local_vector_store.count(stores[0].stem) == 1
