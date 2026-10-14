"""
Embeds Circular (CBDTCircular/ExplanatoryCircular/ExtensionCircular) nodes'
full_citation/subject/applicability_summary/hasText into the local
circular_retrieval_index vector store (see
framework/core/retrieve/circular_entry_points.py,
framework/core/retrieve/local_vector_store.py, and
Semantic_Graph_Retrieval_Architecture.md - NO vectors are written to Neo4j;
this only reads text from Neo4j and writes embeddings to a local file). Run
this any time after ingesting or re-ingesting Circulars.

Usage:
    python -m framework.cli.embed_circulars
"""
import argparse

from framework.core.infra import neo4j_client, step_logger
from framework.core.retrieve import circular_entry_points as cep


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    step_logger.set_verbose(args.verbose)

    with step_logger.Step("Connecting to Neo4j"):
        neo4j_client.verify_connectivity()

    with neo4j_client.get_session() as session:
        cep.setup_circular_vector_indexes(session)  # no-op, kept for compatibility - see that function's docstring
        cep.embed_circular_nodes(session, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
