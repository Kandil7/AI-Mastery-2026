"""
Script to ingest data into a vector database.

Usage:
    python scripts/ingest_to_vector_db.py --input data/processed/ --output db/
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Ingest data into vector database")
    parser.add_argument("--input", type=str, required=True, help="Input data directory")
    parser.add_argument("--output", type=str, required=True, help="Output database path")
    args = parser.parse_args()

    print(f"Ingesting data from {args.input} to {args.output}")
    # TODO: Implement ingestion logic


if __name__ == "__main__":
    main()
