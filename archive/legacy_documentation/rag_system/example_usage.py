"""
Example Usage Script for Arabic Islamic Literature RAG System

This script demonstrates how to use the RAG system to query
the Arabic Islamic literature corpus.

Usage:
    python example_usage.py
"""

import asyncio
import os
import sys

# Fix Windows console encoding for Arabic text
if sys.platform == "win32":
    import io

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.complete_pipeline import create_rag_pipeline, RAGConfig


async def main():
    """Main example function."""

    print("=" * 60)
    print("Arabic Islamic Literature RAG System - Example")
    print("=" * 60)

    # Create pipeline configuration
    config = RAGConfig(
        datasets_path="K:/learning/technical/ai-ml/AI-Mastery-2026/datasets",
        output_path="K:/learning/technical/ai-ml/AI-Mastery-2026/rag_system/data",
        # Use a smaller embedding model for faster processing
        embedding_model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        # Use mock LLM for testing (set OPENAI_API_KEY for real LLM)
        llm_provider="mock",
        llm_model="gpt-4o",
        # Processing settings
        chunk_size=512,
        chunk_overlap=50,
        # Retrieval settings
        retrieval_top_k=50,
        rerank_top_k=5,
    )

    # Create pipeline
    print("\n[1] Creating RAG pipeline...")
    pipeline = create_rag_pipeline(config)

    # Try to load existing index
    print("\n[2] Checking for existing index...")
    try:
        pipeline.load_indexes()

        # Debug: check what was loaded
        print(f"    [DEBUG] _indexed = {pipeline._indexed}")
        print(f"    [DEBUG] _chunks count = {len(pipeline._chunks)}")

        if pipeline._indexed:
            print("    [OK] Loaded existing index")
        else:
            print("    [X] Index file exists but not properly loaded - will reindex")
            print("\n[3] Indexing documents...")
            await pipeline.index_documents(limit=10)
            print("    [OK] Indexing complete")
            pipeline._save_indexes()
            print("    [OK] Indexes saved to disk")
    except Exception as e:
        print(f"    [X] No existing index: {e}")
        print("\n[3] Indexing documents...")

        # Index with limit for faster demo (remove limit for full indexing)
        await pipeline.index_documents(limit=10)
        print("    [OK] Indexing complete")

        # Save again to ensure all components are saved
        pipeline._save_indexes()
        print("    [OK] Indexes saved to disk")

    # Get stats
    print("\n[4] Pipeline Statistics:")
    stats = pipeline.get_stats()
    for key, value in stats.items():
        print(f"    {key}: {value}")

    # Get categories
    print("\n[5] Available Categories:")
    categories = pipeline.get_categories()
    for cat in categories[:10]:
        print(f"    - {cat}")
    if len(categories) > 10:
        print(f"    ... and {len(categories) - 10} more")

    # Example queries
    print("\n" + "=" * 60)
    print("Example Queries")
    print("=" * 60)

    queries = [
        "ما هو التوحيد في الإسلام؟",  # What is tawhid in Islam?
        "Explain the concept of Tawhid",  # English query
        "ما حكم الزكاة في الإسلام؟",  # What is the ruling on zakah?
        "Describe the pillars of Islam",  # English query
    ]

    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        try:
            result = await pipeline.query(query, top_k=3)

            print(f"Answer: {result.answer[:300]}...")
            print(f"\nSources:")
            for i, source in enumerate(result.sources, 1):
                print(f"  [{i}] {source['book_title']} - {source['author']}")
                print(f"      Category: {source['category']}")
                print(f"      Score: {source['score']:.3f}")

            print(f"\nLatency: {result.latency_ms:.2f}ms")
            print(f"Model: {result.model}")

        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


async def quick_demo():
    """Quick demo without indexing."""

    print("\nQuick Demo - Mock Response")
    print("-" * 40)

    # Create pipeline with mock LLM
    config = RAGConfig(llm_provider="mock")
    pipeline = create_rag_pipeline(config)

    # Just test the query (will use mock)
    try:
        result = await pipeline.query("ما هو مفهوم التوحيد؟", top_k=3)
        print(f"Query: {result.query}")
        print(f"Answer: {result.answer}")
        print(f"Latency: {result.latency_ms:.2f}ms")
    except Exception as e:
        print(f"Note: {e}")
        print("This is expected if no index is loaded.")


if __name__ == "__main__":
    # Run full example with indexing
    asyncio.run(main())
