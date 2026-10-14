"""
Dynamic domain-pack loader - the ONE place the framework "reaches into" a
domain_packs/<name>/ folder, and it does so by NAME only (importlib), never
a direct `import domain_packs.tax....` anywhere else in framework/. This is
what makes "add a new domain" a matter of dropping in domain_packs/<name>/
with a pack.py that exposes DOMAIN_CONFIG/PROMPT_PACK, not touching any
framework code.

Usage:
    from framework.cli.pack_loader import load_domain_pack, load_pack_module
    domain_config, prompt_pack = load_domain_pack("tax")   # or omit for FRAMEWORK_DOMAIN env / "tax" default
    pack_module = load_pack_module("tax")                  # the raw module, for pack-specific extras like PIPELINE
"""
import importlib
import os


def load_pack_module(name: str | None = None):
    """Returns the raw domain_packs.<name>.pack module. Resolution order:
    explicit `name` arg -> FRAMEWORK_DOMAIN env var -> "tax" (this project's
    only domain pack so far)."""
    name = name or os.environ.get("FRAMEWORK_DOMAIN", "tax")
    try:
        return importlib.import_module(f"domain_packs.{name}.pack")
    except ModuleNotFoundError as e:
        raise SystemExit(
            f"No domain pack named '{name}' (looked for domain_packs/{name}/pack.py). "
            f"Available packs are folders under domain_packs/ with a pack.py exposing "
            f"DOMAIN_CONFIG and PROMPT_PACK. Original error: {e}"
        ) from e


def load_domain_pack(name: str | None = None):
    """Returns (domain_config, prompt_pack) for the named domain pack - the
    two pieces every framework engine needs. Use load_pack_module() directly
    if you need pack-specific extras (e.g. tax's PIPELINE orchestrator)."""
    module = load_pack_module(name)
    domain_config = getattr(module, "DOMAIN_CONFIG", None)
    prompt_pack = getattr(module, "PROMPT_PACK", None)
    if domain_config is None or prompt_pack is None:
        raise SystemExit(
            f"{module.__name__} must define both DOMAIN_CONFIG (a "
            f"framework.contracts.domain_config.DomainConfig) and PROMPT_PACK "
            f"(a framework.contracts.prompt_pack.PromptPack)."
        )
    return domain_config, prompt_pack
