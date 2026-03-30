"""
Example Usage V2 - Updated RAG Pipeline for Islamic Literature

This version uses the current architecture with:
1. Multi-source ingestion
2. Advanced chunking
3. BM25 retrieval
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

# Configuration
DATASETS_PATH = "K:/learning/technical/ai-ml/AI-Mastery-2026/datasets"
OUTPUT_PATH = "K:/learning/technical/ai-ml/AI-Mastery-2026/rag_system/data"


async def analyze_dataset():
    """Analyze the dataset to understand its structure."""

    logger.info("=" * 60)
    logger.info("STEP 1: Analyzing Dataset")
    logger.info("=" * 60)

    # Import metadata ingestion
    from src.data.multi_source_ingestion import MetadataIngestionPipeline

    metadata_path = os.path.join(DATASETS_PATH, "metadata")
    metadata_pipeline = MetadataIngestionPipeline(metadata_path)
    metadata = await metadata_pipeline.load_metadata()
    logger.info(f"Loaded metadata: {metadata}")

    # Show category distribution
    logger.info(f"\n📊 Total Books: {metadata['books']}")
    logger.info(f"📊 Total Authors: {metadata['authors']}")
    logger.info(f"📊 Total Categories: {metadata['categories']}")

    # Show categories
    if metadata_pipeline.categories:
        logger.info("\n📚 Categories:")
        for cat_id, cat in list(metadata_pipeline.categories.items())[:10]:
            logger.info(f"  - {cat.get('cat_name', 'Unknown')}")

    return metadata_pipeline


async def test_chunking():
    """Test advanced chunking."""

    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: Testing Advanced Chunking")
    logger.info("=" * 60)

    from src.processing.advanced_chunker import create_chunker

    # Create chunker
    chunker = create_chunker(strategy="recursive", chunk_size=512, chunk_overlap=50)

    # Sample text
    sample_text = """
    بسم الله الرحمن الرحيم
    هذا نص تجريبي لاختبار نظام التقطيع المتقدم.
    يتضمن النص فقرات متعددة لاختبار التقطيع العودي والتقطيع الدلالي.
    التقطيع الإسلامي يحافظ على حدود الآيات والأحاديث.
    """

    document = {
        "id": "test_001",
        "content": sample_text,
        "metadata": {"category": "test"},
    }

    chunks = chunker.chunk(document)
    logger.info(f"✅ Created {len(chunks)} chunks")

    for i, chunk in enumerate(chunks, 1):
        logger.info(f"  Chunk {i}: {chunk.word_count} words, {chunk.char_count} chars")

    return chunker


async def test_retrieval(chunks=None):
    """Test BM25 retrieval."""

    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: Testing BM25 Retrieval")
    logger.info("=" * 60)

    from src.retrieval.bm25_retriever import BM25Retriever

    # Create BM25 index
    retriever = BM25Retriever(k1=1.5, b=0.75)

    # Use provided chunks or create test documents
    if chunks:
        documents = [
            {
                "id": chunk["chunk_id"],
                "content": chunk["content"],
                "book_title": chunk.get("book_title", "Unknown"),
                "author": chunk.get("author", "Unknown"),
                "category": chunk.get("category", "Unknown"),
            }
            for chunk in chunks[:100]  # Limit for testing
        ]
    else:
        # Test documents
        documents = [
            {"id": "doc_1", "content": "التوحيد في الإسلام هو إفراد الله بالعبادة", "book_title": "التوحيد", "author": "ابن تيمية", "category": "العقيدة"},
            {"id": "doc_2", "content": "صحيح البخاري أصح كتب الحديث", "book_title": "صحيح البخاري", "author": "البخاري", "category": "الحديث"},
            {"id": "doc_3", "content": "الصلاة عماد الدين", "book_title": "الفقه", "author": "ابن قيم", "category": "الفقه"},
        ]

    logger.info(f"Indexing {len(documents)} documents...")
    retriever.index_documents(documents)
    logger.info("✅ Index created!")

    # Test queries
    test_queries = [
        "توحيد الله",
        "صحيح البخاري",
        "فقه الصلاة",
        "صفات الله",
    ]

    for query in test_queries:
        logger.info(f"\n🔍 Query: {query}")
        results = retriever.search(query, top_k=2)

        for i, result in enumerate(results, 1):
            logger.info(f"  {i}. [{result.get('category', 'N/A')}] {result.get('book_title', 'N/A')}")
            logger.info(f"     Score: {result.get('score', 0):.3f}")
            logger.info(f"     Preview: {result.get('content', '')[:100]}...")

    return retriever


async def test_full_pipeline():
    """Test complete RAG pipeline."""

    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: Testing Complete RAG Pipeline")
    logger.info("=" * 60)

    from src.pipeline.complete_pipeline import create_rag_pipeline, RAGConfig

    # Create config
    config = RAGConfig(
        datasets_path=DATASETS_PATH,
        output_path=OUTPUT_PATH,
        llm_provider="mock",  # Use mock for testing
        chunk_size=512,
        chunk_overlap=50,
    )

    # Create pipeline
    pipeline = create_rag_pipeline(config)

    # Try loading existing index
    try:
        pipeline.load_indexes()
        logger.info("✅ Loaded existing index")
    except Exception as e:
        logger.warning(f"No existing index: {e}")
        logger.info("Indexing sample documents...")
        await pipeline.index_documents(limit=10)
        logger.info("✅ Indexing complete")

    # Test query
    logger.info("\n🔍 Testing query...")
    result = await pipeline.query("ما هو التوحيد؟", top_k=3)

    logger.info(f"✅ Answer: {result.answer[:200]}...")
    logger.info(f"⏱️  Latency: {result.latency_ms:.2f}ms")
    logger.info(f"📚 Sources: {len(result.sources)}")

    return pipeline


async def main():
    """Main execution function."""

    logger.info("🚀 Islamic Literature RAG - Updated Pipeline")
    logger.info("=" * 60)

    try:
        # Step 1: Analyze dataset
        metadata = await analyze_dataset()

        # Step 2: Test chunking
        chunker = await test_chunking()

        # Step 3: Test retrieval
        retriever = await test_retrieval()

        # Step 4: Test full pipeline
        pipeline = await test_full_pipeline()

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("📋 SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✅ Dataset analyzed: {metadata.books} books")
        logger.info(f"✅ Chunking tested: recursive strategy")
        logger.info(f"✅ Retrieval tested: BM25")
        logger.info(f"✅ Full pipeline tested")

        logger.info("\nNext steps:")
        logger.info("  1. Index full dataset: await pipeline.index_documents()")
        logger.info("  2. Start API: uvicorn src.api.service:app --reload")
        logger.info("  3. See example_usage_complete.py for full demo")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
