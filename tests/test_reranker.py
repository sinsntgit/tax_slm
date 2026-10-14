"""2026-10-14: NEW - framework/core/retrieve/reranker.py + reranker_config.py.
Uses the conftest CrossEncoder stub (word-overlap score: 10*overlap/len(q) - 5,
then sigmoid) and a temp reranker config via RERANKER_CONFIG_PATH."""
import json

import pytest

from framework.core.retrieve import reranker, reranker_config


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    """Writes a temp reranker config and points RERANKER_CONFIG_PATH at it."""
    path = tmp_path / "reranker_config.json"

    def write(**values):
        path.write_text(json.dumps(values), encoding="utf-8")
        return path

    monkeypatch.setenv("RERANKER_CONFIG_PATH", str(path))
    write(min_relevance=0.45, fallback_relevance=0.30, fallback_max_chunks=5, top_n=20)
    return write


class CountingEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        return [self.scores[text] for _q, text in pairs]


@pytest.fixture
def encoder(monkeypatch):
    def install(scores):
        enc = CountingEncoder(scores)
        monkeypatch.setattr(reranker, "_reranker", enc)
        return enc
    return install


def _chunks(*specs):
    return [{"chunk_id": cid, "text": text, **extra} for cid, text, extra in specs]


def test_sigmoid_bounds():
    assert reranker._sigmoid(0) == 0.5
    assert reranker._sigmoid(1000) == 1.0
    assert reranker._sigmoid(-1000) == 0.0


def test_empty_candidates():
    assert reranker.rerank("q", []) == []


def test_primary_tier_filters_sorts_and_caps(cfg, encoder):
    cfg(min_relevance=0.5, fallback_relevance=0.3, fallback_max_chunks=5, top_n=2)
    enc = encoder({"a": 3.0, "b": -3.0, "c": 1.0, "d": 2.0})
    chunks = _chunks(("1", "a", {}), ("2", "b", {}), ("3", "c", {}), ("4", "d", {}))
    out = reranker.rerank("q", chunks)
    assert [c["chunk_id"] for c in out] == ["1", "4"]           # top_n=2 of the 3 above 0.5
    assert all(c["relevance_tier"] == "primary" for c in out)
    assert out[0]["rerank_score"] == pytest.approx(reranker._sigmoid(3.0))
    assert out[0]["rerank_score_raw"] == 3.0
    assert enc.calls == 1


def test_fallback_tier_capped_by_fallback_max(cfg, encoder):
    cfg(min_relevance=0.9, fallback_relevance=0.3, fallback_max_chunks=2, top_n=10)
    encoder({"a": 0.0, "b": 0.5, "c": -0.2, "d": -5.0})  # sigmoid: .5 .62 .45 .007
    out = reranker.rerank("q", _chunks(("1", "a", {}), ("2", "b", {}), ("3", "c", {}), ("4", "d", {})))
    assert [c["chunk_id"] for c in out] == ["2", "1"]
    assert {c["relevance_tier"] for c in out} == {"fallback"}


def test_nothing_clears_fallback_returns_empty(cfg, encoder):
    cfg(min_relevance=0.9, fallback_relevance=0.8, fallback_max_chunks=2, top_n=10)
    encoder({"a": -2.0})
    assert reranker.rerank("q", _chunks(("1", "a", {}))) == []


def test_explicit_params_override_config(cfg, encoder):
    cfg(min_relevance=0.99, fallback_relevance=0.99, fallback_max_chunks=1, top_n=1)
    encoder({"a": 0.0, "b": 0.1})
    out = reranker.rerank("q", _chunks(("1", "a", {}), ("2", "b", {})), top_n=5, min_relevance=0.4,
                          fallback_relevance=0.1, fallback_max_chunks=5)
    assert [c["chunk_id"] for c in out] == ["2", "1"]


def test_deterministic_tie_break_regardless_of_input_order(cfg, encoder):
    encoder({"t": 2.0})
    a = _chunks(("10", "t", {}), ("2", "t", {}), ("abc", "t", {}), ("9", "t", {}))
    b = list(reversed(_chunks(("10", "t", {}), ("2", "t", {}), ("abc", "t", {}), ("9", "t", {}))))
    ids_a = [c["chunk_id"] for c in reranker.rerank("q", a)]
    ids_b = [c["chunk_id"] for c in reranker.rerank("q", b)]
    # non-numeric ids sort before numeric ones (reverse sort), numeric ids smallest-first
    assert ids_a == ids_b == ["abc", "2", "9", "10"]


def test_reuses_existing_scores_without_predict(cfg, monkeypatch):
    def boom():
        raise AssertionError("cross-encoder must not be loaded when every chunk is already scored")
    monkeypatch.setattr(reranker, "_get_reranker", boom)
    chunks = _chunks(("1", "x", {"rerank_score": 0.2}), ("2", "y", {"rerank_score": 0.9}))
    out = reranker.rerank("q", chunks, top_n=5, min_relevance=0.1)
    assert [c["chunk_id"] for c in out] == ["2", "1"]


def test_partial_scores_trigger_full_rescore(cfg, encoder):
    enc = encoder({"x": 1.0, "y": 2.0})
    chunks = _chunks(("1", "x", {"rerank_score": 0.01}), ("2", "y", {}))
    reranker.rerank("q", chunks)
    assert enc.calls == 1 and chunks[0]["rerank_score"] == pytest.approx(reranker._sigmoid(1.0))


def test_skip_rerank_orders_graph_first_then_similarity(cfg, monkeypatch):
    cfg(min_relevance=0.45, fallback_relevance=0.3, fallback_max_chunks=5, top_n=3)
    monkeypatch.setattr(reranker, "_get_reranker", lambda: (_ for _ in ()).throw(AssertionError("no model")))
    chunks = _chunks(("v1", "a", {"similarity": 0.2}), ("g1", "b", {"similarity": None}),
                     ("v2", "c", {"similarity": 0.9}), ("g2", "d", {}))
    out = reranker.rerank("q", chunks, skip_rerank=True)
    assert [c["chunk_id"] for c in out] == ["g1", "g2", "v2"]   # top_n=3 from config
    assert all(c["rerank_score"] is None and c["relevance_tier"] == "unranked" for c in out)


def test_rerank_after_skip_rerank_rescores_instead_of_crashing(cfg, encoder):
    """A list that went through skip_rerank carries rerank_score=None on every
    chunk; a later normal rerank() of the SAME list (e.g. app.py's quality-gate
    retry) must re-score rather than try to sort None against floats."""
    enc = encoder({"a": 2.0, "b": 1.0})
    chunks = _chunks(("1", "a", {}), ("2", "b", {}))
    reranker.rerank("q", chunks, skip_rerank=True, top_n=5)
    out = reranker.rerank("q", chunks, top_n=5, min_relevance=0.1)
    assert enc.calls == 1
    assert [c["chunk_id"] for c in out] == ["1", "2"]


def test_text_truncated_before_scoring(cfg, monkeypatch):
    seen = []

    class Spy:
        def predict(self, pairs):
            seen.extend(pairs)
            return [0.0 for _ in pairs]

    monkeypatch.setattr(reranker, "_reranker", Spy())
    reranker.rerank("q", [{"chunk_id": "1", "text": "z" * 5000}, {"chunk_id": "2", "text": None}])
    assert len(seen[0][1]) == reranker.RERANK_TEXT_MAX_CHARS
    assert seen[1][1] == ""


def test_real_stub_cross_encoder_end_to_end(cfg, monkeypatch):
    monkeypatch.setattr(reranker, "_reranker", None)
    q = "royalty on software payments"
    chunks = _chunks(("1", "payments for software are not royalty", {}), ("2", "agricultural income", {}))
    out = reranker.rerank(q, chunks)
    assert [c["chunk_id"] for c in out] == ["1"]


# --- reranker_config ---------------------------------------------------------

def test_load_config_missing_corrupt_and_partial(tmp_path, monkeypatch):
    path = tmp_path / "rc.json"
    monkeypatch.setenv("RERANKER_CONFIG_PATH", str(path))
    assert reranker_config.load_config() == reranker_config.DEFAULT_CONFIG
    path.write_text("{not json", encoding="utf-8")
    assert reranker_config.load_config() == reranker_config.DEFAULT_CONFIG
    path.write_text("[1, 2]", encoding="utf-8")
    assert reranker_config.load_config() == reranker_config.DEFAULT_CONFIG
    path.write_text(json.dumps({"top_n": 7, "junk": 1}), encoding="utf-8")
    cfg = reranker_config.load_config()
    assert cfg["top_n"] == 7 and "junk" not in cfg and cfg["min_relevance"] == 0.45


def test_save_config_normalizes_per_key(tmp_path, monkeypatch):
    path = tmp_path / "sub" / "rc.json"
    monkeypatch.setenv("RERANKER_CONFIG_PATH", str(path))
    out = reranker_config.save_config({"min_relevance": 1.5, "fallback_relevance": 0.2,
                                       "fallback_max_chunks": True, "top_n": 30, "extra": 1})
    assert out == {"min_relevance": 0.45, "fallback_relevance": 0.2, "fallback_max_chunks": 5, "top_n": 30}
    assert json.loads(path.read_text(encoding="utf-8")) == out


def test_repo_reranker_config_is_valid(monkeypatch):
    """The committed reranker_config.json must load to in-range values."""
    monkeypatch.delenv("RERANKER_CONFIG_PATH", raising=False)
    cfg = reranker_config.load_config()
    assert 0 <= cfg["fallback_relevance"] <= cfg["min_relevance"] <= 1
    assert 1 <= cfg["top_n"] <= 100 and 1 <= cfg["fallback_max_chunks"] <= 50
