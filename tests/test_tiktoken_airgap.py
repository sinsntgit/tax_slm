"""2026-10-14: NEW (phase0/followups) - chunking.py's tiktoken load is lazy,
bounded and fail-fast offline, with the len(text)//4 fallback (an air-gapped
customer machine must never hang on the BPE download). A fake `tiktoken`
module stands in for the real one - no network."""
import importlib
import sys
import time
import types
from pathlib import Path

import pytest

CONVERTERS = Path(__file__).resolve().parents[1] / "domain_packs" / "tax" / "converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))
chunking = importlib.import_module("chunking")

TEXT = "Section 195 requires deduction of tax at source on payments to non-residents."


class FakeEncoding:
    def encode(self, text):
        return text.split()   # deliberately != len(text)//4


def _fake_tiktoken(monkeypatch, behaviour):
    calls = []

    def get_encoding(name):
        calls.append(name)
        return behaviour(name)

    monkeypatch.setitem(sys.modules, "tiktoken", types.SimpleNamespace(get_encoding=get_encoding))
    return calls


@pytest.fixture
def fresh(monkeypatch, tmp_path):
    monkeypatch.setattr(chunking, "_ENCODING", chunking._NOT_LOADED)
    monkeypatch.setattr(chunking, "_ENCODING_STATUS", "not_loaded")
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", str(tmp_path / "tiktoken_cache"))
    monkeypatch.delenv("TIKTOKEN_OFFLINE", raising=False)
    monkeypatch.delenv("TIKTOKEN_LOAD_TIMEOUT_SECONDS", raising=False)
    return tmp_path


def test_import_does_not_load_tiktoken(monkeypatch):
    calls = _fake_tiktoken(monkeypatch, lambda n: FakeEncoding())
    monkeypatch.delitem(sys.modules, "chunking", raising=False)
    mod = importlib.import_module("chunking")
    try:
        assert calls == [] and mod._ENCODING is mod._NOT_LOADED
    finally:
        sys.modules["chunking"] = chunking   # keep the module other tests imported


def test_loaded_once_and_used(fresh, monkeypatch):
    calls = _fake_tiktoken(monkeypatch, lambda n: FakeEncoding())
    assert chunking.token_count(TEXT) == len(TEXT.split())
    assert chunking.token_count(TEXT) == len(TEXT.split())
    assert calls == ["cl100k_base"] and chunking._ENCODING_STATUS == "loaded"


def test_hanging_download_is_abandoned_quickly(fresh, monkeypatch, capsys):
    calls = _fake_tiktoken(monkeypatch, lambda n: time.sleep(5) or FakeEncoding())
    monkeypatch.setenv("TIKTOKEN_LOAD_TIMEOUT_SECONDS", "0.2")
    t0 = time.time()
    assert chunking.token_count(TEXT) == len(TEXT) // 4
    assert time.time() - t0 < 2
    assert "timed out" in chunking._ENCODING_STATUS
    assert "estimate" in capsys.readouterr().err
    # not retried on later calls (no second 0.2s wait, no second thread)
    chunking.token_count(TEXT)
    assert calls == ["cl100k_base"]


def test_offline_without_cache_never_calls_tiktoken(fresh, monkeypatch):
    calls = _fake_tiktoken(monkeypatch, lambda n: FakeEncoding())
    monkeypatch.setenv("TIKTOKEN_OFFLINE", "1")
    assert chunking.token_count(TEXT) == len(TEXT) // 4
    assert calls == [] and "offline" in chunking._ENCODING_STATUS


def test_offline_with_preseeded_cache_loads(fresh, monkeypatch):
    calls = _fake_tiktoken(monkeypatch, lambda n: FakeEncoding())
    monkeypatch.setenv("TIKTOKEN_OFFLINE", "1")
    cache_file = Path(chunking.tiktoken_cache_file())
    assert cache_file.parent == fresh / "tiktoken_cache"
    # tiktoken's own cache naming: sha1 of the blob URL
    assert cache_file.name == "9b5ad71b2ce5302211f9c61530b329a4922fc6a4"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_bytes(b"seeded")
    assert chunking.token_count(TEXT) == len(TEXT.split())
    assert calls == ["cl100k_base"]


def test_blank_cache_dir_is_treated_as_unset(fresh, monkeypatch):
    _fake_tiktoken(monkeypatch, lambda n: FakeEncoding())
    monkeypatch.setenv("TIKTOKEN_CACHE_DIR", "")   # tiktoken would read "" as "never cache"
    chunking.token_count(TEXT)
    assert "TIKTOKEN_CACHE_DIR" not in __import__("os").environ


def test_load_error_falls_back(fresh, monkeypatch):
    def boom(name):
        raise ConnectionError("blocked")
    _fake_tiktoken(monkeypatch, boom)
    assert chunking.token_count(TEXT) == len(TEXT) // 4
    assert "ConnectionError" in chunking._ENCODING_STATUS


def test_forced_fallback_still_works(monkeypatch):
    monkeypatch.setattr(chunking, "_ENCODING", None)
    assert chunking.token_count(TEXT) == len(TEXT) // 4
    assert chunking.token_count("") == 0
