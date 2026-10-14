"""
provenance.py - 2026-10-14: NEW (phase0/opinions workstream).

Every entity and relationship this project writes to the graph carries a
`provenance` property saying HOW the graph came to know about it. This is
product-critical for the "Tax SLM" product: the answer model must be able to
tell a fact that was read from an authority's OWN text apart from a node that
only exists because some other document cited it.

  ingested_document - extracted from the text of the document being ingested
                      (an Act section parsed from the Act, a CaseLaw node built
                      from that judgment, a FirmOpinion built from that
                      opinion). Also used for RELATIONSHIPS that the ingested
                      document itself asserts (e.g. "this opinion relies on
                      Section 195", "this case FOLLOWS X") - the edge is a
                      faithful record of what that document says, even when
                      its target is only a stub.
  cited_reference   - a STUB: a node (or an edge between two stubs) that exists
                      only because some document cited it. Its properties come
                      from the citing document's citation string, NOT from the
                      authority's own text - so nothing about its content has
                      been verified.
  model_suggested   - RESERVED for later (e.g. an LLM-proposed SIMILAR_TO edge).
                      Never produced today; ranked below cited_reference.

Load-time rules (enforced in graph_loader.py, see PROVENANCE_RANK):
  * an entity/relationship with no provenance defaults to ingested_document
    (every converter written before 2026-10-14 only ever emitted real content
    without the property, apart from the stubs it now marks explicitly);
  * a LOWER-ranked write never overwrites or downgrades an existing node of a
    higher rank - a cited_reference stub only fills in when the node does not
    exist yet;
  * a HIGHER-ranked write REPLACES a lower-ranked node outright (a real
    ingestion upgrades a stub), so stale stub-only properties don't linger.
"""

INGESTED_DOCUMENT = "ingested_document"
CITED_REFERENCE = "cited_reference"
MODEL_SUGGESTED = "model_suggested"

VALID_PROVENANCE = (INGESTED_DOCUMENT, CITED_REFERENCE, MODEL_SUGGESTED)

# Higher = more trustworthy. An unrecognized non-empty value ranks lowest
# (treated as untrusted rather than silently promoted to "verified").
PROVENANCE_RANK = {INGESTED_DOCUMENT: 2, CITED_REFERENCE: 1, MODEL_SUGGESTED: 0}


def normalize(value) -> str:
    """Missing/empty -> ingested_document (the load-time default)."""
    return value if value else INGESTED_DOCUMENT


def rank(value) -> int:
    return PROVENANCE_RANK.get(normalize(value), -1)


def mark_stub(props: dict, cited_in: str | None = None) -> dict:
    """Stamps stub provenance onto a properties dict IN PLACE (and returns it).

    Also pins sourceMdPath to "" explicitly: several converters stamp
    document-level metadata onto every entity they built via setdefault()
    (extract_entities.attach_pipeline_metadata, extract_case_law's stamping
    loop), which used to give a stub the CITING document's sourceMdPath -
    so md_hydration would have served the citing document's text as if it
    were the cited authority's own full text. An explicit "" survives
    setdefault()."""
    props["provenance"] = CITED_REFERENCE
    props["sourceMdPath"] = ""
    if cited_in:
        props.setdefault("cited_in_document_id", cited_in)
    return props
