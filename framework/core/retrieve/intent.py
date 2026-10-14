"""
INTENT DETECTION - GENERALIZED from tax_kg_poc/webapp/intent.py. Classifies
a question into a retrieval-scoping intent before running GraphRAG. The
prompt now comes from the domain's PromptPack (not a module-level `from
prompts import get_prompt`), and the valid intent set / fallback intent come
from DomainConfig - a domain pack can keep the framework's default
EXACT_LOOKUP/RELATIONAL/CONCEPTUAL/OUT_OF_SCOPE categories or define its own.
"""
import logging

from framework.core.infra import llm_client

logger = logging.getLogger("intent_detection")


def detect_intent(question: str, domain_config, prompt_pack, trace=None) -> str:
    """Returns one of domain_config.valid_intents. Falls back to
    domain_config.fallback_intent if the model output doesn't cleanly match a
    known label, rather than crashing the pipeline over a classification
    hiccup. `trace` (optional) nests this call's Langfuse generation under
    the caller's existing trace - see llm_client.complete()."""
    prompt = prompt_pack.get_prompt("intent_detection")

    raw = llm_client.complete(system=prompt.template, user_content=question, max_tokens=20,
                               trace=trace, name="intent_detection").strip().upper()

    if raw in domain_config.valid_intents:
        logger.info(f"[intent_detection] '{question[:60]}' -> {raw}")
        return raw

    logger.warning(f"[intent_detection] Unrecognized model output '{raw}', "
                    f"defaulting to {domain_config.fallback_intent}")
    return domain_config.fallback_intent
