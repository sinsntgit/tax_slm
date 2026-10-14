"""
DocumentParser contract - the shape a domain pack's document-to-graph
converter must produce so framework/core/ingest can load it with zero
domain-specific code.

This is the TARGET interface, matching the dict/dataclass shape
scripts/chunking.py's chunk_act_document() and scripts/02_extract_entities.py
already produce today (see domain_packs/tax/converters/ - Phase 1 keeps them
as plain functions returning this exact shape rather than forcing a class
hierarchy on them; Phase 2 can formalize them as ParserRegistry
implementations if a second domain pack's converter needs the polymorphism).

entities / relationships are already in the {"id", "type", "properties"} /
{"type", "source", "target", "properties"} shape framework/core/ingest/
graph_loader.py loads directly - no format translation at the call site,
matching every prior script in this project (02's LLM output, 03's loader).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ParsedChunk:
    """One fine-grained, embeddable/citable unit of source text. Mirrors
    scripts/chunking.py's Chunk dataclass field-for-field - kept generic here
    (no section/subsection/clause naming baked in) so a non-legislative
    domain pack (e.g. a telecom runbook, a retail policy doc) can reuse the
    same shape with domain-specific coordinate fields folded into `metadata`."""

    chunk_id: str
    entity_id: str
    text: str
    short_title: str = ""
    chunk_type: str = "provision"
    element_type: str = "section"
    full_citation: str = ""
    parent_entity_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedDocument:
    entities: list[dict]
    relationships: list[dict]
    chunks: list[ParsedChunk]
    ambiguities: list[dict] = field(default_factory=list)


class DocumentParser(Protocol):
    """Structural typing contract - a domain pack converter satisfies this by
    exposing a callable with this signature, no base-class inheritance
    required (matches how chunking.chunk_act_document() and
    02_extract_entities.py's pipeline already work)."""

    def __call__(self, raw_text: str, **kwargs: Any) -> ParsedDocument: ...
