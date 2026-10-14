"""
Registration point for the tax domain pack - the module
framework/cli/pack_loader.py's load_domain_pack("tax") imports.

Exposes:
  - DOMAIN_CONFIG - framework.contracts.domain_config.DomainConfig instance
    (see config.py for how every field was derived from the original
    tax_kg_poc modules).
  - PROMPT_PACK - framework.contracts.prompt_pack.PromptPack instance (see
    prompts/tax_prompts.py).
  - CONVERTERS - logical-name -> relative script path, for reference/CLI help
    only. Conversion/extraction scripts (PDF-to-Markdown, LLM entity
    extraction) keep their own multi-argument CLIs rather than being
    forwarded through a generic framework command - see this pack's
    converters/ folder and the root README for exact invocations.
"""
from domain_packs.tax import pipeline
from domain_packs.tax.config import DOMAIN_CONFIG, TAX_CATEGORY_SPECS
from domain_packs.tax.prompts.tax_prompts import PROMPT_PACK

CONVERTERS = {
    "pdf_to_md_act": "domain_packs/tax/converters/pdf_to_md_act.py",
    "pdf_to_md_caselaw": "domain_packs/tax/converters/pdf_to_md_caselaw.py",
    "extract_entities": "domain_packs/tax/converters/extract_entities.py",
    "extract_case_law": "domain_packs/tax/converters/extract_case_law.py",
    "pdf_to_md_generic": "domain_packs/tax/converters/pdf_to_md_generic.py",
    "extract_generic": "domain_packs/tax/converters/extract_generic.py",
}

# The ingestion-UI orchestrator (upload/convert/extract/ingest/embed,
# file-type dispatch, background job registry) - what framework/webapp/app.py's
# /api/upload, /api/convert, /api/ingest/* routes call. See pipeline.py.
PIPELINE = pipeline

# 2026-08-26: NEW. The Tax Knowledge & Analysis Framework's six response
# categories (see config.py's TAX_CATEGORY_SPECS docstring) - optional, like
# PIPELINE above, so framework/webapp/app.py stays domain-agnostic (it reads
# this via getattr(PACK, "CATEGORY_SPECS", None) rather than importing
# anything tax-specific directly).
CATEGORY_SPECS = TAX_CATEGORY_SPECS

__all__ = ["DOMAIN_CONFIG", "PROMPT_PACK", "CONVERTERS", "PIPELINE", "CATEGORY_SPECS"]
