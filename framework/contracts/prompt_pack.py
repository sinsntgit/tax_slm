"""
PromptPack - the generic, versioned prompt-repository PATTERN, extracted from
webapp/prompts.py (which hardcoded the tax domain's actual prompt text
alongside the mechanism for versioning/serving it). The mechanism belongs in
the framework; the prompt TEXT belongs in the domain pack
(domain_packs/tax/prompts/tax_prompts.py builds a PromptPack instance from
the exact same PromptVersion entries prompts.py used to define at module
level).

Every prompt retrieval is logged so an answer can always be traced back to
the exact prompt version that produced it - unchanged behavior from
prompts.py's get_prompt(), just no longer tied to one fixed set of prompt
names/content.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger("prompt_management")


@dataclass
class PromptVersion:
    name: str
    version: str
    template: str
    approved_by: str
    approved_at: str
    notes: str = ""


@dataclass
class PromptPack:
    """Holds every versioned prompt for one domain. Construct with a dict of
    name -> [PromptVersion, ...] (last entry = current production version),
    same shape as prompts.py's old module-level _PROMPT_REPO."""

    domain_name: str
    repo: dict[str, list[PromptVersion]] = field(default_factory=dict)

    def get_prompt(self, name: str, version: str | None = None) -> PromptVersion:
        versions = self.repo.get(name)
        if not versions:
            raise KeyError(f"[{self.domain_name}] No prompt registered under name '{name}'")

        if version:
            match = next((v for v in versions if v.version == version), None)
            if not match:
                raise KeyError(f"[{self.domain_name}] Prompt '{name}' has no version '{version}'")
            chosen = match
        else:
            chosen = versions[-1]  # latest = last in the list = current production version

        logger.info(f"[prompt_management] domain={self.domain_name} serving prompt name={name} version={chosen.version}")
        return chosen

    def list_versions(self, name: str) -> list[str]:
        return [v.version for v in self.repo.get(name, [])]
