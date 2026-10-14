"""Framework: the domain-agnostic engine (convert / ingest / retrieve / webapp / cli).

Nothing in this package may import from `domain_packs.*`. Domain packs import
FROM the framework and register themselves with it (see
framework/contracts/domain_config.py and framework/cli/pack_loader.py) - never
the other way around. This is the one architectural rule that keeps adding a
new domain (telecom, retail, refinery, ...) a matter of writing a new
domain_packs/<name>/ folder, not touching framework code.
"""
