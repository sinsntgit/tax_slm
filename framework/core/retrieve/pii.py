"""
PII IDENTIFICATION & MASKING - GENERALIZED from tax_kg_poc/webapp/pii.py.

The MECHANISM (stable pseudonymous tokens via HMAC, regex-sweep of free
text, per-class always-masked-field replacement, audit logging to Neo4j,
"ingest everything, mask on output" enforcement point) is domain-agnostic
and unchanged. The RULES (which fields are always masked, which regex
patterns detect PII, which properties are never masked) come from
DomainConfig - tax_kg_poc's India-specific PAN/Aadhaar detectors and
CaseLaw/FirmOpinion field list now live in domain_packs/tax/config.py
instead of being hardcoded here, so a different domain/jurisdiction (e.g. a
US-focused pack needing SSN detection instead) supplies its own.
"""
import hashlib
import hmac
import logging
import os
import re
from datetime import datetime, timezone

from dotenv import load_dotenv

from framework.core.infra import neo4j_client
from framework.core.infra.dotenv_path import env_path as _env_path

# Anchored to the repo root by file location, not process cwd - see
# framework.core.infra.dotenv_path's docstring.
load_dotenv(_env_path())
logger = logging.getLogger("pii")

PII_MASK_SALT = os.environ.get("PII_MASK_SALT", "dev-only-change-me-in-.env")

# Structural/administrative property names that are NEVER PII regardless of
# domain - join keys, provenance, filterable metadata used by the framework
# itself. A domain pack's DomainConfig.pii_never_mask_properties EXTENDS
# this set (via the union in _never_mask()), never replaces it.
FRAMEWORK_NEVER_MASK_PROPERTIES = {
    "id", "node_id", "chunk_id", "document_id", "content_hash", "source_chunk_ids",
    "also_known_as_ids", "possibleDuplicateOf", "ingestion_date", "status",
    "effective_from", "effective_to", "authority_level", "jurisdiction",
    "source_type", "confidence",
}


def _log_pii_access(entity_type: str | None, property_name: str | None, detection_method: str,
                     raw_value: str, context: str = ""):
    value_hash = hashlib.sha256((raw_value or "").encode("utf-8")).hexdigest()[:16]
    try:
        with neo4j_client.get_session() as session:
            neo4j_client.ensure_property_index(session, "PiiAccessLog", "entity_type")
            session.run(
                """
                CREATE (l:PiiAccessLog {
                    occurred_at: $occurred_at, entity_type: $entity_type, property_name: $property_name,
                    detection_method: $detection_method, value_hash: $value_hash, context: $context
                })
                """,
                occurred_at=datetime.now(timezone.utc).isoformat(),
                entity_type=entity_type, property_name=property_name,
                detection_method=detection_method, value_hash=value_hash,
                context=(context or "")[:500],
            )
    except Exception as e:
        logger.warning(f"[pii] WARNING: could not write PiiAccessLog entry ({e}). Masking still applied.")


def mask_value(raw_value: str, token_prefix: str) -> str:
    digest = hmac.new(PII_MASK_SALT.encode("utf-8"), str(raw_value).encode("utf-8"), hashlib.sha256).hexdigest()[:10]
    return f"[{token_prefix}_REDACTED_{digest}]"


class PiiMasker:
    """Bundles the masking mechanism with one domain's rules
    (ALWAYS_MASKED_FIELDS / regex detectors / never-mask set), so callers
    don't have to thread a DomainConfig through every function call."""

    def __init__(self, domain_config):
        self.always_masked_fields: dict = domain_config.pii_always_masked_fields or {}
        self.never_mask_properties = FRAMEWORK_NEVER_MASK_PROPERTIES | set(domain_config.pii_never_mask_properties or ())
        self.regex_detectors = list(domain_config.pii_regex_detectors or ())
        self._always_masked_property_names = {
            name for names in self.always_masked_fields.values() for name in names
        }

    def mask_free_text(self, text: str, entity_type: str | None = None, property_name: str | None = None,
                        context: str = "") -> tuple[str, list[str]]:
        if not text or not isinstance(text, str):
            return text, []

        found_types = []
        masked = text
        for label, pattern in self.regex_detectors:
            def _replace(m, label=label):
                found_types.append(label)
                _log_pii_access(entity_type, property_name, f"regex:{label}", m.group(0), context)
                return mask_value(m.group(0), label)
            masked = pattern.sub(_replace, masked)
        return masked, found_types

    def mask_node_properties(self, label: str, properties: dict, context: str = "") -> dict:
        if not properties:
            return properties

        masked_props = dict(properties)
        always_masked = self.always_masked_fields.get(label, [])

        for key, value in properties.items():
            if key in self.never_mask_properties:
                continue
            if not isinstance(value, str) or not value:
                continue

            if key in always_masked:
                masked_props[key] = mask_value(value, f"{label.upper()}_{key.upper()}")
                _log_pii_access(label, key, "always_masked_field", value, context)
            else:
                masked_text, _found = self.mask_free_text(value, entity_type=label, property_name=key, context=context)
                masked_props[key] = masked_text

        return masked_props

    def mask_node(self, node: dict, context: str = "") -> dict:
        if not node:
            return node
        masked = dict(node)
        masked["properties"] = self.mask_node_properties(node.get("label", ""), node.get("properties", {}), context)
        return masked

    def mask_nodes(self, nodes: list[dict], context: str = "") -> list[dict]:
        return [self.mask_node(n, context) for n in nodes]

    def mask_chunk_text(self, text: str, context: str = "") -> str:
        masked, _found = self.mask_free_text(text, entity_type="Chunk", property_name="content", context=context)
        return masked

    def mask_chunks(self, chunks: list[dict], context: str = "") -> list[dict]:
        masked_chunks = []
        for c in chunks:
            c2 = dict(c)
            c2["text"] = self.mask_chunk_text(c.get("text", ""), context=context)
            masked_chunks.append(c2)
        return masked_chunks

    def mask_plain_row_value(self, value, context: str = ""):
        if isinstance(value, dict):
            masked = {}
            for k, v in value.items():
                if k in self.never_mask_properties:
                    masked[k] = v
                elif k in self._always_masked_property_names and isinstance(v, str):
                    masked[k] = mask_value(v, k.upper())
                    _log_pii_access(None, k, "always_masked_field(plain_row)", v, context)
                elif isinstance(v, str):
                    masked[k], _ = self.mask_free_text(v, property_name=k, context=context)
                else:
                    masked[k] = v
            return masked
        if isinstance(value, str):
            masked_text, _ = self.mask_free_text(value, context=context)
            return masked_text
        return value

    def mask_plain_rows(self, rows: list[dict], context: str = "") -> list[dict]:
        return [{k: self.mask_plain_row_value(v, context) for k, v in row.items()} for row in rows]

    def scan_for_pii(self, value, _path: str = "$") -> list[dict]:
        """Independent, read-only verification scan - used by a domain
        pack's PII-scan CLI to confirm zero leakage using the EXACT SAME
        detectors as the masking path itself."""
        findings = []
        if isinstance(value, str):
            for label, pattern in self.regex_detectors:
                for m in pattern.finditer(value):
                    raw = m.group(0)
                    preview = raw[:2] + "***" + raw[-2:] if len(raw) > 4 else "***"
                    findings.append({"path": _path, "type": label, "match_preview": preview})
        elif isinstance(value, dict):
            for k, v in value.items():
                findings.extend(self.scan_for_pii(v, f"{_path}.{k}"))
        elif isinstance(value, list):
            for i, v in enumerate(value):
                findings.extend(self.scan_for_pii(v, f"{_path}[{i}]"))
        return findings
