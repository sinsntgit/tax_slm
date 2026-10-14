"""
Shared Neo4j driver helper - PROMOTED unchanged from
tax_kg_poc/scripts/neo4j_client.py. Every pipeline engine and the webapp,
for EVERY domain pack, imports this same module - the graph database
connection itself is not a domain concept.

Usage:
    from framework.core.infra import neo4j_client
    with neo4j_client.get_session() as session:
        session.run("MATCH (n) RETURN n LIMIT 5")
"""
import os
import re

from dotenv import load_dotenv
from neo4j import GraphDatabase

from framework.core.infra.dotenv_path import env_path as _env_path

# Anchored to the repo root by file location, not process cwd - see
# framework.core.infra.dotenv_path's docstring.
load_dotenv(_env_path())

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "taxkg_neo4j_pw")
NEO4J_DATABASE = os.environ.get("NEO4J_DATABASE", "neo4j")

NEO4J_CONNECT_TIMEOUT_SECONDS = int(os.environ.get("NEO4J_CONNECT_TIMEOUT_SECONDS", "10"))

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD),
            connection_timeout=NEO4J_CONNECT_TIMEOUT_SECONDS,
        )
    return _driver


def get_session():
    return get_driver().session(database=NEO4J_DATABASE)


def verify_connectivity():
    get_driver().verify_connectivity()


def close_driver():
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


_SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def safe_identifier(name: str) -> str:
    if not isinstance(name, str) or not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(f"Unsafe Cypher label/relationship-type/index identifier: {name!r}")
    return name


def ensure_id_constraint(session, label: str):
    label = safe_identifier(label)
    session.run(f"CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE")


def ensure_property_index(session, label: str, property_name: str, index_name: str | None = None):
    label = safe_identifier(label)
    property_name = safe_identifier(property_name)
    index_name = safe_identifier(index_name or f"idx_{label.lower()}_{property_name}")
    session.run(f"CREATE INDEX {index_name} IF NOT EXISTS FOR (n:{label}) ON (n.{property_name})")


def ensure_vector_index(session, index_name: str, label: str, property_name: str,
                         dimensions: int, similarity_function: str = "cosine"):
    """`CREATE VECTOR INDEX ... IF NOT EXISTS` is a no-op if an index with
    this name already exists - it does NOT update the dimensions, even if
    they no longer match the embedding model actually in use (e.g. switching
    EMBEDDING_MODEL_NAME from a 384-dim model to a 768-dim one). Silently
    leaving a stale-dimension index in place would either error on the next
    vector write/query or, worse, silently mix incompatible-dimension
    vectors under the same indexed property. So: check the EXISTING index's
    configured dimension first, and DROP + recreate if it doesn't match what
    the caller is asking for now - safe because the index is just a search
    structure over the `embedding` property, not the underlying data; nodes
    keep their (possibly stale-dimension, until re-embedded) `embedding`
    values regardless."""
    index_name = safe_identifier(index_name)
    label = safe_identifier(label)
    property_name = safe_identifier(property_name)

    existing = list(session.run(
        "SHOW INDEXES YIELD name, options WHERE name = $name RETURN options",
        name=index_name,
    ))
    if existing:
        try:
            existing_dims = existing[0]["options"]["indexConfig"]["vector.dimensions"]
        except (KeyError, TypeError):
            existing_dims = None
        if existing_dims is not None and int(existing_dims) != int(dimensions):
            session.run(f"DROP INDEX {index_name} IF EXISTS")

    session.run(
        f"""
        CREATE VECTOR INDEX {index_name} IF NOT EXISTS
        FOR (n:{label}) ON (n.{property_name})
        OPTIONS {{indexConfig: {{
            `vector.dimensions`: $dimensions,
            `vector.similarity_function`: $similarity_function
        }}}}
        """,
        dimensions=dimensions, similarity_function=similarity_function,
    )
