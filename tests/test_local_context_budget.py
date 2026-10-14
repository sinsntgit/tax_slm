"""
2026-10-14: NEW (phase0/followups) - ingestion chunk-group sizing honours a
LOCAL model's configured context window (LOCAL_LLM_CONTEXT_LENGTH) instead of
falling back to 100k tokens, in both extract_generic.model_chunk_budget() and
pipeline.llm_chunk_info(). Isolated .env only - never touches the real one.
"""
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONVERTERS = REPO / "domain_packs" / "tax" / "converters"
if str(CONVERTERS) not in sys.path:
    sys.path.insert(0, str(CONVERTERS))

import extract_generic  # noqa: E402
from framework.core.infra import llm_client  # noqa: E402

_PREFIXES = ("USE_MODEL", "LLM_ROUTE_", "LOCAL_", "CLAUDE_", "GPT_", "GEMINI_")


@pytest.fixture
def env(tmp_path):
    saved = dict(os.environ)
    for k in list(os.environ):
        if k.startswith(_PREFIXES):
            del os.environ[k]
    env_file = tmp_path / "test.env"
    env_file.write_text("")
    os.environ["ENV_FILE_PATH"] = str(env_file)
    os.environ["WEBAPP_SETTINGS_PATH"] = str(tmp_path / "settings.json")

    def write(**values):
        env_file.write_text("".join(f"{k}={v}\n" for k, v in values.items()))
    yield write
    os.environ.clear()
    os.environ.update(saved)
    llm_client.refresh_model_config("retrieval")   # restore back-compat globals


def _pipeline_chunk_info():
    from domain_packs.tax import pipeline
    return pipeline.llm_chunk_info()


def test_api_provider_budget_unchanged(env):
    env(USE_MODEL="CLAUDE")
    assert extract_generic.model_chunk_budget() == 200_000 - 1_500 - 2_000
    info = _pipeline_chunk_info()
    assert info["context_window_tokens"] == 200_000
    assert info["extraction_call_budget_tokens"] == 200_000 - 1_500 - 2_000


def test_local_window_drives_both_budgets(env):
    env(USE_MODEL="CLAUDE", USE_MODEL_INGESTION="LOCAL", LOCAL_LLM_BASE_URL="http://x/v1",
        LOCAL_MODEL_NAME="tax-slm-7b", LOCAL_LLM_CONTEXT_LENGTH="8192")
    budget = extract_generic.model_chunk_budget()
    # far below the old 100k fallback, and a full prompt (system reserve + text
    # + output reserve) fits the window under llm_client's own 3-chars/token
    # estimate when the text is ~4 chars per tiktoken token
    assert 2_000 < budget < 8192 - 1_500 - 2_000
    text_chars = budget * 4
    est_prompt = (1_500 * 3 + text_chars) / 3 + 64
    assert est_prompt + 2_000 <= 8192 + 1
    info = _pipeline_chunk_info()
    assert info["provider"] == "LOCAL" and info["model"] == "tax-slm-7b"
    assert info["context_window_tokens"] == 8192
    assert info["extraction_call_budget_tokens"] == budget


def test_local_route_for_enrichment_calls_is_honoured(env):
    # only the enrichment call type is routed to the local model
    env(USE_MODEL="GPT", LLM_ROUTE_ENRICHMENT="LOCAL", LOCAL_LLM_BASE_URL="http://x/v1",
        LOCAL_LLM_CONTEXT_LENGTH="16384")
    assert extract_generic.model_chunk_budget() == llm_client.local_text_token_budget(16384, 1_500, 2_000)
    assert _pipeline_chunk_info()["context_window_tokens"] == 16384


def test_local_without_window_keeps_old_fallback(env):
    env(USE_MODEL="LOCAL", LOCAL_LLM_BASE_URL="http://x/v1")
    assert extract_generic.model_chunk_budget() == 100_000 - 1_500 - 2_000


def test_small_window_groups_split_document(env):
    env(USE_MODEL="LOCAL", LOCAL_LLM_BASE_URL="http://x/v1", LOCAL_LLM_CONTEXT_LENGTH="8192")
    para = "The assessee claimed deduction under section 80-IAC for the eligible start-up. " * 60
    chunks = [para] * 12
    groups = extract_generic.group_chunks_for_llm(chunks, extract_generic.model_chunk_budget())
    assert len(groups) > 1
    for g in groups:
        assert sum(extract_generic.chunking.token_count(chunks[i]) for i in g) <= extract_generic.model_chunk_budget()


def test_local_text_budget_floor():
    assert llm_client.local_text_token_budget(2_000, 1_500, 2_000) == 256
