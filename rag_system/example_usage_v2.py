"""
Enhanced Example Usage - Complete RAG Pipeline for Islamic Literature

This version properly handles:
1. Data cleaning and encoding
2. Category-aware chunking
3. BM25-only fallback (no torch required)
4. Complete processing pipeline
"""

import os
import sys
import asyncio
import json
import logging
from pathlib import Path

# Setup UTF-8 for Windows
if sys.platform == "win32":
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

DATASETS_PATH = "K:/learning/technical/ai-ml/AI-Mastery-2026/datasets"
OUTPUT_PATH = "K:/learning/technical/ai-ml/AI-Mastery-2026/rag_system/data"


# ============================================================================
# Main Functions
# ============================================================================


async def analyze_dataset():
    """Analyze the dataset to understand its structure."""

    logger.info("=" * 60)
    logger.info("STEP 1: Analyzing Dataset")
    logger.info("=" * 60)

    # Import enhanced ingestion
    from src.data.enhanced_ingestion import EnhancedIngestionPipeline

    pipeline = EnhancedIngestionPipeline(
        datasets_path=DATASETS_PATH,
        chunk_size=512,
        chunk_overlap=50,
    )

    # Load metadata
    metadata = await pipeline.load_metadata()
    logger.info(f"Loaded metadata: {metadata}")

    # Get files
    files = pipeline.get_book_files()
    logger.info(f"Found {len(files)} book files")

    # Show sample files
    logger.info("\n📚 Sample books:")
    for f in files[:5]:
        logger.info(f"  - ID {f['book_id']}: {f['title']}")
        logger.info(
            f"    Category: {f['category']}, Author: {f['author']}, Size: {f['size_mb']:.2f} MB"
        )

    # Show category distribution
    cat_counts = {}
    for f in files:
        cat = f["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    logger.info("\n📊 Category Distribution:")
    for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"  {cat}: {count} books")

    # Calculate total size
    total_size = sum(f["size_mb"] for f in files)
    logger.info(f"\n💾 Total dataset size: {total_size:.2f} MB")

    return pipeline


async def process_books(pipeline, limit: int = 100):
    """Process books and create chunks."""

    logger.info("\n" + "=" * 60)
    logger.info(f"STEP 2: Processing {limit} Books")
    logger.info("=" * 60)

    # Process books
    chunks = await pipeline.process_books(limit=limit)

    # Get stats
    stats = pipeline.get_stats()

    logger.info(f"\n✅ Processing Complete!")
    logger.info(f"  - Successfully processed: {stats['successfully_processed']} books")
    logger.info(f"  - Failed: {stats['failed']} books")
    logger.info(f"  - Total chunks: {stats['total_chunks']}")
    logger.info(f"  - Total words: {stats['total_words']:,}")
    logger.info(f"  - Encoding issues: {stats['encoding_issues']}")
    logger.info(f"  - Avg chunks/book: {stats['avg_chunks_per_book']:.1f}")
    logger.info(f"  - Avg words/chunk: {stats['avg_words_per_chunk']:.1f}")

    # Show category breakdown
    logger.info("\n📊 Chunks by Category:")
    cat_chunks = {}
    for chunk in chunks:
        cat = chunk["category"]
        cat_chunks[cat] = cat_chunks.get(cat, 0) + 1

    for cat, count in sorted(cat_chunks.items(), key=lambda x: -x[1])[:10]:
        logger.info(f"  {cat}: {count} chunks")

    # Save chunks to file
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_file = output_dir / f"chunks_{limit}.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total": len(chunks),
                "chunks": chunks[:100],  # Save first 100 for preview
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    logger.info(f"\n💾 Saved chunks preview to {chunks_file}")

    return chunks


async def test_retrieval_only(chunks):
    """Test BM25 retrieval without embeddings (fallback mode)."""

    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Testing BM25 Retrieval (No Embeddings Required)")
    logger.info("=" * 60)

    try:
        from src.retrieval.bm25_retriever import BM25Retriever

        # Create BM25 index
        retriever = BM25Retriever(k1=1.5, b=0.75)

        # Prepare documents
        documents = [
            {
                "id": chunk["chunk_id"],
                "content": chunk["content"],
                "book_title": chunk["book_title"],
                "author": chunk["author"],
                "category": chunk["category"],
            }
            for chunk in chunks
        ]

        logger.info(f"Indexing {len(documents)} documents...")
        retriever.index(documents)
        logger.info("Index created!")

        # Test queries
        test_queries = [
            "توحيد الله",
            "صحيح البخاري",
            "فقه الصلاة",
            "صفات الله",
        ]

        for query in test_queries:
            logger.info(f"\n🔍 Query: {query}")
            results = retriever.search(query, top_k=3)

            for i, result in enumerate(results, 1):
                logger.info(f"  {i}. [{result['category']}] {result['book_title']}")
                logger.info(f"     Score: {result['score']:.3f}")
                logger.info(f"     Preview: {result['content'][:100]}...")

        return retriever

    except Exception as e:
        logger.error(f"BM25 test failed: {e}")
        return None


async def main():
    """Main execution function."""

    logger.info("🚀 Islamic Literature RAG - Enhanced Pipeline")
    logger.info("=" * 60)

    try:
        # Step 1: Analyze dataset
        pipeline = await analyze_dataset()

        # Step 2: Process books (limit to manageable amount)
        chunks = await process_books(pipeline, limit=100)

        # Step 3: Test BM25 retrieval
        retriever = await test_retrieval_only(chunks)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📋 SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Dataset analyzed: {pipeline.stats.total_files} books")
        logger.info(f"✅ Chunks created: {len(chunks)}")
        logger.info(f"✅ BM25 retrieval: {'Working' if retriever else 'Failed'}")
        logger.info(f"⚠️  Semantic search: Requires PyTorch fix")
        logger.info("\nNext steps:")
        logger.info("  1. Fix PyTorch installation or use BM25-only mode")
        logger.info("  2. Process full dataset (8000+ books)")
        logger.info("  3. Build complete RAG pipeline")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
