"""
DomainConfig - the one object that turns the framework's generic engines
(framework/core/ingest, framework/core/retrieve, framework/webapp) into a
working pipeline for a SPECIFIC domain (tax, telecom, retail, refinery, ...).

Every field here was pulled out of a hardcoded constant somewhere in the
original tax_kg_poc codebase during this migration - see the comment on each
field for exactly which script/module it came from and why it had to move.
A new domain pack (domain_packs/<name>/config.py) builds one instance of this
dataclass and the framework does the rest.

Design note: fields are deliberately concrete (compiled regexes, dicts,
tuples), not yet-more-abstract plugin interfaces - this project has exactly
one real domain pack (tax) so far, so this is calibrated to "generalize what
we've already proven needs generalizing," not to guess at requirements a
second or third domain pack hasn't actually presented yet. Expect to widen
this dataclass, not redesign it, when domain_packs/telecom or similar shows
up for real.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional, Pattern


@dataclass
class CitationRules:
    """Governs how free-text citations ('Section 55(2)(b)', '12A(1)') are
    recognized and normalized. Ported out of webapp/retrieval.py's
    CITATION_RE / CITATION_QUERY_RE / extract_citation() /
    normalize_citation_query(), which were hardcoded to the Income Tax Act's
    'Section N(sub)(clause)...' citation grammar. A different domain's
    'citation' concept (e.g. telecom's 'Circuit ID CKT-004512', retail's SKU
    codes) supplies its own patterns/builder here instead."""

    # Matches a citation anywhere inside a free-text question. group(1) must
    # be the primary identifier (e.g. section number), group(2) the raw
    # remainder (e.g. "(2)(b)") - see build_citation(), which receives
    # exactly these two positional groups (framework/core/retrieve/graphrag.py
    # calls `rules.build_citation(m.group(1), m.group(2) or "")`).
    free_text_pattern: Pattern
    # Same grammar, anchored to the WHOLE input, with the leading keyword
    # ("Section") optional - used for a citation search box where the user is
    # typing a bare identifier rather than a sentence.
    query_pattern: Pattern
    # Turns a regex match's two groups into the exact string stored on
    # Chunk.full_citation by the domain's converter (must match byte-for-byte,
    # since retrieval does an exact-match lookup against it).
    build_citation: Callable[[str, str], str]


@dataclass
class DomainConfig:
    domain_name: str

    # --- INGESTION (framework/core/ingest/graph_loader.py) ------------------
    # Entity labels whose ids are DETERMINISTIC (built from citation
    # coordinates by the domain's own converter, not fuzzy/LLM-derived) -
    # cross-run fuzzy entity-resolution is skipped for these labels because
    # "no exact-id match" already means "genuinely new", full stop. Ported
    # from 03_build_graph.py's STRUCTURAL_LABELS constant.
    structural_labels: frozenset[str] = field(default_factory=frozenset)

    # Chunk dataclass field names copied onto a Chunk node's properties by
    # graph_loader.load_chunk_nodes_and_links() - EXCLUDES "content"/"embedding"
    # (only the embedder writes those). Ported from 03_build_graph.py's
    # CHUNK_METADATA_KEYS.
    chunk_metadata_keys: tuple[str, ...] = ()

    # Same idea, for framework/core/ingest/embedder.py - INCLUDES "content"/
    # "embedding" implicitly (the embedder always sets those two); this list
    # is the ADDITIONAL metadata fields copied alongside them. Ported from
    # 05_embed_chunks.py's CHUNK_METADATA_KEYS.
    embed_chunk_metadata_keys: tuple[str, ...] = ()

    # Ontology class name -> which of its properties best characterize "is
    # this the same real-world entity", for cross-run fuzzy entity
    # resolution. Ported from scripts/entity_resolution.py's
    # RESOLUTION_KEY_PROPERTIES. A type not listed falls back to comparing
    # its own id.
    resolution_key_properties: dict = field(default_factory=dict)

    # --- RETRIEVAL (framework/core/retrieve/graphrag.py) ---------------------
    citation_rules: Optional[CitationRules] = None

    # Intent label -> {"vector_top_k": int, "vector_mode": "always"|"fallback_only"}.
    # Ported from webapp/retrieval.py's INTENT_RETRIEVAL_CONFIG. A domain
    # pack may override this entirely, or start from
    # framework.core.retrieve.graphrag.DEFAULT_INTENT_RETRIEVAL_CONFIG.
    intent_retrieval_config: dict = field(default_factory=dict)
    default_retrieval_config: dict = field(
        default_factory=lambda: {"vector_top_k": 10, "vector_mode": "always"}
    )

    # Valid intent labels for framework/core/retrieve/intent.py's classifier.
    # Ported from webapp/intent.py's VALID_INTENTS.
    valid_intents: frozenset[str] = field(
        default_factory=lambda: frozenset({"EXACT_LOOKUP", "RELATIONAL", "CONCEPTUAL", "OUT_OF_SCOPE"})
    )
    fallback_intent: str = "CONCEPTUAL"

    # --- PII (framework/core/retrieve/pii.py) --------------------------------
    # Ontology class name -> property names ALWAYS masked for that class,
    # regardless of content. Ported from webapp/pii.py's ALWAYS_MASKED_FIELDS.
    pii_always_masked_fields: dict = field(default_factory=dict)
    # Structural/administrative property names that must NEVER be masked
    # (join keys, provenance, filterable metadata). Ported from webapp/pii.py's
    # NEVER_MASK_PROPERTIES. Framework supplies a sensible cross-domain
    # default (see framework.core.retrieve.pii.DEFAULT_NEVER_MASK_PROPERTIES);
    # a domain pack extends it, doesn't replace it wholesale.
    pii_never_mask_properties: frozenset[str] = field(default_factory=frozenset)
    # [(label, compiled_regex), ...] applied to ANY free-text property/chunk
    # content regardless of class. Ported from webapp/pii.py's REGEX_DETECTORS
    # (PAN/Aadhaar/email/phone - India/tax-specific; a different domain/
    # jurisdiction supplies its own).
    pii_regex_detectors: tuple = ()

    # --- SANITY CHECKS (framework/cli/query_graph.py) -------------------------
    # [(source_label, relationship_type, target_label, description), ...] -
    # sample "does this structural relationship exist" checks run after
    # ingestion. Ported from 04_query_graph.py's steps 3 ("Sections and what
    # they refer to") and 4 ("Case law interpreting sections"), which
    # hardcoded Section/REFERS_TO/CaseLaw/INTERPRETS.
    sample_relationship_checks: tuple = ()

    # --- MISC -----------------------------------------------------------------
    # Embedding text builder: (chunk_record: dict) -> str. Ported from
    # 05_embed_chunks.py's embedding_text() (prepends short_title to content).
    embedding_text_fn: Optional[Callable[[dict], str]] = None
