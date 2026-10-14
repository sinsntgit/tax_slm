"""2026-10-14: NEW - every module imports offline (catches syntax errors and
import-time side effects that need a live service)."""
import importlib

import pytest

MODULES = [
    "framework.core.retrieve.graphrag", "framework.core.retrieve.reranker",
    "framework.core.retrieve.category_retrieval", "framework.core.retrieve.prompt_config",
    "framework.core.retrieve.tribunal_case_law_agent", "framework.core.retrieve.law_agent",
    "framework.core.infra.llm_client", "framework.core.ingest.graph_loader",
    "domain_packs.tax.config", "domain_packs.tax.prompts.tax_prompts", "domain_packs.tax.pipeline",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_webapp_imports(monkeypatch):
    # app.py touches Neo4j at import (best-effort, wrapped in try/except) - must still import.
    importlib.import_module("framework.webapp.app")
