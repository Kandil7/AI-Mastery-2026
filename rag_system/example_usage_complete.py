"""
Complete RAG System Demo - Islamic Literature 2026

This script demonstrates the complete RAG pipeline:
1. Data ingestion from multiple sources
2. Advanced chunking with Islamic text optimization
3. Multi-provider embedding generation
4. Vector storage with Qdrant/ChromaDB
5. Hybrid retrieval (semantic + BM25)
6. Query transformation
7. Cross-encoder reranking
8. LLM generation with citations
9. Islamic-specific evaluation
10. Cost tracking and monitoring

Usage:
    python example_usage_complete.py
"""

import os
import sys
import asyncio
import json
import time
from pathlib import Path
from typing import Dict, Any

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

# Configure UTF-8 for Windows
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")


# ============================================================================
# Configuration
# ============================================================================

CONFIG = {
    "datasets_path": "K:/learning/technical/ai-ml/AI-Mastery-2026/datasets",
    "output_path": "K:/learning/technical/ai-ml/AI-Mastery-2026/rag_system/data",
    "embedding_model": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    "llm_provider": "mock",  # Change to "openai" for real LLM
    "llm_model": "gpt-4o",
    "chunk_size": 512,
    "chunk_overlap": 50,
    "retrieval_top_k": 50,
    "rerank_top_k": 5,
    "vector_db_type": "memory",  # or "qdrant", "chroma"
}


# ============================================================================
# Demo Functions
# ============================================================================


async def demo_basic_query(pipeline):
    """Demonstrate basic RAG query."""

    print("\n" + "=" * 70)
    print("📝 DEMO 1: Basic RAG Query")
    print("=" * 70)

    queries = [
        "ما هو التوحيد في الإسلام؟",
        "Explain the concept of Tawhid",
        "ما حكم الزكاة؟",
    ]

    for query in queries:
        print(f"\n❓ Query: {query}")
        print("-" * 50)

        try:
            result = await pipeline.query(query, top_k=3)

            print(f"✅ Answer: {result.answer[:200]}...")
            print(f"⏱️  Latency: {result.latency_ms:.2f}ms")
            print(f"📚 Sources: {len(result.sources)}")

            for i, source in enumerate(result.sources[:2], 1):
                print(f"   [{i}] {source['book_title']} - {source['author']}")

        except Exception as e:
            print(f"❌ Error: {e}")


async def demo_domain_specialist(pipeline):
    """Demonstrate domain-specific queries."""

    print("\n" + "=" * 70)
    print("🕌 DEMO 2: Domain Specialists")
    print("=" * 70)

    from rag_system import create_islamic_rag

    rag = create_islamic_rag()
    await rag.initialize()

    # Tafsir query
    print("\n📖 Tafsir Specialist:")
    result = await rag.query_tafsir("ما تفسير سورة الإخلاص؟")
    print(f"   Domain: {result.get('domain_name', 'Unknown')}")
    print(f"   Answer: {result.get('answer', '')[:150]}...")

    # Hadith query
    print("\n📜 Hadith Specialist:")
    result = await rag.query_hadith("ما حديث إنما الأعمال بالنيات؟")
    print(f"   Domain: {result.get('domain_name', 'Unknown')}")
    print(f"   Answer: {result.get('answer', '')[:150]}...")

    # Fiqh query
    print("\n⚖️  Fiqh Specialist:")
    result = await rag.query_fiqh("ما شروط الصلاة؟")
    print(f"   Domain: {result.get('domain_name', 'Unknown')}")
    print(f"   Answer: {result.get('answer', '')[:150]}...")


async def demo_comparative_fiqh(pipeline):
    """Demonstrate comparative fiqh analysis."""

    print("\n" + "=" * 70)
    print("⚖️  DEMO 3: Comparative Fiqh")
    print("=" * 70)

    from rag_system import create_islamic_rag

    rag = create_islamic_rag()
    await rag.initialize()

    result = await rag.compare_madhhabs("ما حكم قراءة البسملة في الصلاة؟")

    print(f"\n📋 Question: {result.get('question', '')}")
    print(f"\n📊 Consensus: {result.get('consensus', {}).get('message', '')}")

    madhhab_results = result.get('madhhab_results', {})
    for madhhab, madhhab_result in madhhab_results.items():
        if 'error' not in madhhab_result:
            print(f"\n   {madhhab.upper()}:")
            print(f"   {madhhab_result.get('answer', '')[:100]}...")


async def demo_multi_hop_reasoning(pipeline):
    """Demonstrate multi-hop reasoning."""

    print("\n" + "=" * 70)
    print("🧠 DEMO 4: Multi-Hop Reasoning")
    print("=" * 70)

    from rag_system import create_islamic_rag

    rag = create_islamic_rag()
    await rag.initialize()

    result = await rag.reason_with_chain("ما الفرق بين التوحيد والشرع؟")

    print(f"\n❓ Question: {result.get('question', '')}")

    reasoning_chain = result.get('reasoning_chain', [])
    for step in reasoning_chain:
        print(f"\n   {step.get('title', 'Step')}:")
        print(f"   {step.get('content', '')[:150]}...")

    print(f"\n✅ Final Answer: {result.get('final_answer', '')[:200]}...")


async def demo_agents(pipeline):
    """Demonstrate agent system."""

    print("\n" + "=" * 70)
    print("🤖 DEMO 5: Agent System")
    print("=" * 70)

    from rag_system import create_islamic_rag

    rag = create_islamic_rag()
    await rag.initialize()

    # Researcher agent
    print("\n🔬 Researcher Agent:")
    result = await rag.ask_as_researcher("ما أدلة وجود الله؟")
    print(f"   Answer: {result.get('answer', '')[:150]}...")

    # Student agent
    print("\n🎓 Student Agent:")
    result = await rag.ask_as_student("الصلاة")
    print(f"   Lesson: {result.get('content', '')[:150]}...")

    # Fatwa agent (research, not real fatwa)
    print("\n📿 Fatwa Research Agent:")
    result = await rag.ask_fatwa("ما حكم الربا؟")
    print(f"   Answer: {result.get('answer', '')[:150]}...")
    if 'disclaimer' in result:
        print(f"   ⚠️  Disclaimer included")


async def demo_evaluation(pipeline):
    """Demonstrate evaluation system."""

    print("\n" + "=" * 70)
    print("📊 DEMO 6: Evaluation")
    print("=" * 70)

    from rag_system.src.evaluation.evaluator import RAGEvaluator, ArabicTestDataset

    evaluator = RAGEvaluator(pipeline=pipeline)

    # Get test samples
    samples = ArabicTestDataset.get_samples(language="arabic", difficulty="easy")

    print(f"\n📋 Evaluating on {len(samples[:3])} samples...")

    # Run evaluation
    results = await evaluator.evaluate_dataset(samples[:3], top_k=5)

    print(f"\n📈 Results:")
    print(f"   Retrieval:")
    print(f"      Precision@K: {results['retrieval']['precision_at_k']:.3f}")
    print(f"      Recall@K: {results['retrieval']['recall_at_k']:.3f}")
    print(f"      MRR: {results['retrieval']['mrr']:.3f}")
    print(f"      NDCG: {results['retrieval']['ndcg']:.3f}")

    if results['generation']:
        print(f"   Generation:")
        print(f"      Faithfulness: {results['generation']['faithfulness']:.3f}")
        print(f"      Relevance: {results['generation']['answer_relevance']:.3f}")

    print(f"   System:")
    print(f"      Avg Latency: {results['system']['avg_latency_ms']:.2f}ms")


async def demo_monitoring(pipeline):
    """Demonstrate monitoring system."""

    print("\n" + "=" * 70)
    print("📈 DEMO 7: Monitoring & Cost Tracking")
    print("=" * 70)

    from rag_system.src.monitoring.monitoring import get_monitor

    monitor = get_monitor()

    # Simulate queries
    for i in range(5):
        monitor.log_query(
            query=f"Test query {i}",
            latency_ms=100 + i * 10,
            tokens_input=50,
            tokens_output=100,
            retrieval_count=5,
            success=True,
        )

    # Get dashboard
    metrics = monitor.get_dashboard_metrics()

    print(f"\n💰 Cost Tracking:")
    print(f"   Daily Spent: ${metrics['cost']['daily']['spent']:.4f}")
    print(f"   Daily Budget: ${metrics['cost']['daily']['budget']:.2f}")
    print(f"   Remaining: ${metrics['cost']['daily']['remaining']:.4f}")

    print(f"\n📊 Query Stats (24h):")
    print(f"   Total Queries: {metrics['queries_last_24h']['query_count']}")
    print(f"   Avg Latency: {metrics['queries_last_24h']['avg_latency_ms']:.2f}ms")
    print(f"   Success Rate: {metrics['queries_last_24h']['success_rate']:.2%}")

    print(f"\n🔝 Top Queries:")
    for q in metrics['top_queries'][:3]:
        print(f"   - {q['query']}: {q['count']} times")


async def demo_indexing(pipeline):
    """Demonstrate indexing process."""

    print("\n" + "=" * 70)
    print("📚 DEMO 8: Document Indexing")
    print("=" * 70)

    from rag_system import create_islamic_rag

    rag = create_islamic_rag()
    await rag.initialize()

    # Index with limit
    print("\n📖 Indexing 10 books...")

    start_time = time.time()
    await rag.index_documents(limit=10)
    elapsed = time.time() - start_time

    print(f"✅ Indexing complete in {elapsed:.2f} seconds")

    # Get stats
    stats = rag.get_stats()
    print(f"\n📊 Statistics:")
    print(f"   Total Chunks: {stats.get('total_chunks', 0)}")
    print(f"   BM25 Documents: {stats.get('bm25_documents', 0)}")
    print(f"   Vector Count: {stats.get('vector_count', 0)}")


async def demo_query_transformation(pipeline):
    """Demonstrate query transformation."""

    print("\n" + "=" * 70)
    print("🔄 DEMO 9: Query Transformation")
    print("=" * 70)

    from rag_system.src.retrieval.query_transformer import create_query_transformer

    transformer = create_query_transformer(
        enable_hyde=True,
        enable_decomposition=True,
        enable_step_back=True,
    )

    query = "ما حكم الزكاة وكيفية حسابها؟"

    print(f"\n❓ Original: {query}")

    result = await transformer.transform(query)

    print(f"\n📊 Query Type: {result.query_type.value}")
    print(f"✏️  Rewritten: {result.rewritten_query}")
    print(f"🔀 Sub-queries: {len(result.sub_queries)}")

    for i, sub_q in enumerate(result.sub_queries[:3], 1):
        print(f"   {i}. {sub_q}")

    if result.hypothetical_document:
        print(f"\n📄 HyDE Document: {result.hypothetical_document[:150]}...")

    if result.metadata.get('step_back_query'):
        print(f"\n🔙 Step-back: {result.metadata['step_back_query']}")


async def demo_chunking_strategies(pipeline):
    """Demonstrate different chunking strategies."""

    print("\n" + "=" * 70)
    print("✂️  DEMO 10: Chunking Strategies")
    print("=" * 70)

    from rag_system.src.processing.advanced_chunker import (
        create_chunker,
        get_recommended_chunking,
    )

    # Sample text
    sample_text = """
بسم الله الرحمن الرحيم
هذا نص تجريبي يوضح استراتيجيات التقطيع المختلفة.
يتضمن النص فقرات متعددة لاختبار التقطيع العودي والتقطيع الدلالي.
التقطيع الإسلامي يحافظ على حدود الآيات والأحاديث.
"""

    strategies = ["recursive", "fixed", "semantic"]

    for strategy in strategies:
        print(f"\n📊 Strategy: {strategy}")

        chunker = create_chunker(strategy=strategy, chunk_size=100)

        chunks = chunker.chunk({
            "id": "test_001",
            "content": sample_text,
            "metadata": {},
        })

        print(f"   Chunks created: {len(chunks)}")
        for i, chunk in enumerate(chunks[:2], 1):
            print(f"   [{i}] {chunk.word_count} words, {chunk.char_count} chars")

    # Get recommended settings
    print(f"\n📋 Recommended Settings:")
    for category in ["quran", "hadith", "fiqh", "general"]:
        config = get_recommended_chunking(category)
        print(f"   {category}: strategy={config['strategy']}, size={config['chunk_size']}")


# ============================================================================
# Main Execution
# ============================================================================


async def run_all_demos():
    """Run all demonstrations."""

    print("\n" + "=" * 70)
    print("🚀 COMPLETE RAG SYSTEM DEMO - ISLAMIC LITERATURE 2026")
    print("=" * 70)

    # Create pipeline
    from rag_system.src.pipeline.complete_pipeline import create_rag_pipeline, RAGConfig

    config = RAGConfig(
        datasets_path=CONFIG["datasets_path"],
        output_path=CONFIG["output_path"],
        embedding_model=CONFIG["embedding_model"],
        llm_provider=CONFIG["llm_provider"],
        llm_model=CONFIG["llm_model"],
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        retrieval_top_k=CONFIG["retrieval_top_k"],
        rerank_top_k=CONFIG["rerank_top_k"],
        vector_db_type=CONFIG["vector_db_type"],
    )

    pipeline = create_rag_pipeline(config)

    # Try to load existing index
    print("\n📂 Loading existing index...")
    try:
        pipeline.load_indexes()
        print("✅ Index loaded successfully")
    except Exception as e:
        print(f"⚠️  No existing index: {e}")
        print("\n📚 Indexing sample documents...")
        await pipeline.index_documents(limit=10)
        pipeline._save_indexes()
        print("✅ Index created and saved")

    # Run demos
    demos = [
        ("Basic Query", demo_basic_query),
        ("Domain Specialists", demo_domain_specialist),
        ("Comparative Fiqh", demo_comparative_fiqh),
        ("Multi-Hop Reasoning", demo_multi_hop_reasoning),
        ("Agent System", demo_agents),
        ("Evaluation", demo_evaluation),
        ("Monitoring", demo_monitoring),
        ("Query Transformation", demo_query_transformation),
        ("Chunking Strategies", demo_chunking_strategies),
    ]

    for demo_name, demo_func in demos:
        try:
            await demo_func(pipeline)
        except Exception as e:
            print(f"\n❌ Error in {demo_name}: {e}")
            import traceback
            traceback.print_exc()

    # Final summary
    print("\n" + "=" * 70)
    print("📋 DEMO SUMMARY")
    print("=" * 70)

    stats = pipeline.get_stats()
    print(f"✅ Pipeline initialized")
    print(f"✅ Total chunks: {stats.get('total_chunks', 0)}")
    print(f"✅ Vector count: {stats.get('vector_count', 0)}")
    print(f"✅ BM25 documents: {stats.get('bm25_documents', 0)}")
    print(f"✅ All demos completed")

    print("\n" + "=" * 70)
    print("🎉 COMPLETE DEMO FINISHED")
    print("=" * 70)


def main():
    """Main entry point."""

    print("""
╔════════════════════════════════════════════════════════════════╗
║                                                                ║
║     Arabic Islamic Literature RAG System - Complete Demo      ║
║                                                                ║
║     Production-Grade RAG Pipeline for 8,425+ Islamic Books    ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
    """)

    asyncio.run(run_all_demos())


if __name__ == "__main__":
    main()
