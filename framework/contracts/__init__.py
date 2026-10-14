"""Contracts: the interfaces a domain pack implements/supplies so the
framework's engines (ingest, retrieve, webapp, cli) can run against ANY
domain without containing domain-specific logic themselves.

  - domain_config.DomainConfig  - the single object a domain pack builds and
    that every framework engine takes as a parameter. Holds citation parsing
    rules, structural labels, chunk metadata schema, retrieval policy, and
    PII rules for the domain.
  - prompt_pack.PromptPack      - the domain's versioned prompt repository
    (nl2cypher / answer_generation / intent_detection / extraction prompts).
  - document_parser.ParsedDocument / DocumentParser - the shape a domain's
    document-to-graph converter must produce (entities/relationships/chunks),
    so framework/core/ingest never has to know how a Section or a Cell Tower
    or a SKU got parsed - only that it got parsed into this shape.
"""
