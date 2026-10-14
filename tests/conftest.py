"""
Shared pytest setup. 2026-10-14: NEW - the cloud dev/CI environment cannot
download Hugging Face models (sentence-transformers / BGE reranker) and has
no Neo4j server, so heavy/networked dependencies are replaced with small
stand-ins HERE, once, before any project module is imported. Tests that need
the real thing are marked `@pytest.mark.live` and skipped unless
TAXKG_LIVE_TESTS=1 (run those on a machine with Docker + .env).
"""
import os
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

FIXTURES = REPO_ROOT / "tests" / "fixtures"


def _install_sentence_transformers_stub():
    """Deterministic fake embedder/cross-encoder - same call shapes as the
    real library, so graphrag/reranker/embedder import and run offline."""
    if "sentence_transformers" in sys.modules or os.environ.get("TAXKG_LIVE_TESTS") == "1":
        return
    import numpy as np

    class SentenceTransformer:
        def __init__(self, name, *a, **k):
            self.name = name

        def get_sentence_embedding_dimension(self):
            return 16

        # 2026-10-14: sentence-transformers >=5 renamed the method above to
        # get_embedding_dimension() (embedder.py calls the new name) - the
        # stub offers both so embedder._get_model() runs offline.
        get_embedding_dimension = get_sentence_embedding_dimension

        def encode(self, texts, normalize_embeddings=False, **k):
            single = isinstance(texts, str)
            items = [texts] if single else list(texts)
            out = []
            for t in items:
                v = np.zeros(16, dtype=np.float32)
                for i, ch in enumerate((t or "").lower()):
                    v[(ord(ch) + i) % 16] += 1.0
                n = np.linalg.norm(v) or 1.0
                out.append(v / n if normalize_embeddings else v)
            arr = np.array(out)
            return arr[0] if single else arr

    class CrossEncoder:
        def __init__(self, name, *a, **k):
            self.name = name

        def predict(self, pairs):
            # word-overlap score, roughly in the cross-encoder's logit range
            res = []
            for q, d in pairs:
                qs, ds = set(q.lower().split()), set((d or "").lower().split())
                res.append(10.0 * len(qs & ds) / (len(qs) or 1) - 5.0)
            return res

    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = SentenceTransformer
    mod.CrossEncoder = CrossEncoder
    sys.modules["sentence_transformers"] = mod


_install_sentence_transformers_stub()


def pytest_configure(config):
    config.addinivalue_line("markers", "live: needs real Neo4j/LLM/.env - skipped unless TAXKG_LIVE_TESTS=1")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("TAXKG_LIVE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="live test - set TAXKG_LIVE_TESTS=1 on a machine with Docker + .env")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def fixtures_dir():
    return FIXTURES


@pytest.fixture
def fake_neo4j(monkeypatch):
    """2026-10-14: NEW - a scripted, installed FakeNeo4j (see
    tests/helpers/neo4j_mock.py). Script it with fake_neo4j.on(substring, records)."""
    from tests.helpers.neo4j_mock import FakeNeo4j
    return FakeNeo4j().install(monkeypatch)
