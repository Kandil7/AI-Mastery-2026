"""
Comprehensive Test Suite for RAG System

Tests all components after architecture fixes.

Usage:
    python test_rag_system.py
"""

import sys
import asyncio
from pathlib import Path

# Add parent path and current directory
BASE_PATH = Path(__file__).parent
sys.path.insert(0, str(BASE_PATH))
sys.path.insert(0, str(BASE_PATH.parent))

# Set rag_system as current module
import os
os.chdir(str(BASE_PATH))

print("=" * 70)
print("RAG SYSTEM - COMPREHENSIVE TEST SUITE")
print("=" * 70)

# ============================================================================
# Test 1: Import Tests
# ============================================================================

print("\n[TEST 1/6] Testing imports...")

try:
    # Root imports
    from rag_system import (
        IslamicRAG,
        IslamicRAGConfig,
        create_islamic_rag,
        create_chunker,
        create_embedding_pipeline,
        create_vector_store,
        create_query_transformer,
        create_agent,
    )
    print("  ✅ Root imports successful")
    
    # Submodule imports
    from src.data import (
        MultiSourceIngestionPipeline,
        create_file_source,
    )
    print("  ✅ Data module imports successful")
    
    from src.processing import (
        AdvancedChunker,
        EmbeddingPipeline,
    )
    print("  ✅ Processing module imports successful")
    
    from src.retrieval import (
        HybridRetriever,
        VectorStore,
        QueryTransformer,
    )
    print("  ✅ Retrieval module imports successful")
    
    from src.generation import (
        LLMClient,
        RAGGenerator,
    )
    print("  ✅ Generation module imports successful")
    
    from src.specialists import (
        IslamicScholar,
        create_islamic_scholar,
    )
    print("  ✅ Specialists module imports successful")
    
    from src.agents import (
        IslamicRAGAgent,
        create_agent,
    )
    print("  ✅ Agents module imports successful")
    
    from src.evaluation import (
        RAGEvaluator,
        ArabicTestDataset,
    )
    print("  ✅ Evaluation module imports successful")
    
    from src.monitoring import (
        get_monitor,
        CostTracker,
    )
    print("  ✅ Monitoring module imports successful")
    
    from src.api import app
    print("  ✅ API module imports successful")
    
except ImportError as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 2: Configuration Tests
# ============================================================================

print("\n[TEST 2/6] Testing configuration...")

try:
    from rag_system import IslamicRAGConfig, RAGConfig as PipelineRAGConfig
    
    # Test IslamicRAGConfig
    config = IslamicRAGConfig()
    assert hasattr(config, 'datasets_path')
    assert hasattr(config, 'embedding_model')
    assert hasattr(config, 'llm_provider')
    print("  ✅ IslamicRAGConfig valid")
    
    # Test PipelineRAGConfig
    pipeline_config = PipelineRAGConfig()
    assert hasattr(pipeline_config, 'datasets_path')
    assert hasattr(pipeline_config, 'chunk_size')
    print("  ✅ PipelineRAGConfig valid")
    
except Exception as e:
    print(f"  ❌ Configuration test failed: {e}")
    sys.exit(1)

# ============================================================================
# Test 3: Component Instantiation Tests
# ============================================================================

print("\n[TEST 3/6] Testing component instantiation...")

try:
    # Test chunker
    chunker = create_chunker(strategy="recursive", chunk_size=512)
    print("  ✅ AdvancedChunker created")
    
    # Test embedding pipeline
    embedding_pipeline = create_embedding_pipeline(
        provider="sentence_transformers"
    )
    print("  ✅ EmbeddingPipeline created")
    
    # Test vector store
    vector_store = create_vector_store(
        store_type="memory",
        vector_size=768,
    )
    print("  ✅ VectorStore created")
    
    # Test query transformer
    transformer = create_query_transformer()
    print("  ✅ QueryTransformer created")
    
    # Test agent
    agent = create_agent("researcher")
    print("  ✅ IslamicRAGAgent created")
    
except Exception as e:
    print(f"  ❌ Component instantiation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 4: Chunking Test
# ============================================================================

print("\n[TEST 4/6] Testing chunking...")

try:
    sample_text = """
    بسم الله الرحمن الرحيم
    هذا نص تجريبي لاختبار نظام التقطيع.
    يتضمن النص فقرات متعددة.
    """
    
    document = {
        "id": "test_001",
        "content": sample_text,
        "metadata": {"category": "test"},
    }
    
    chunks = chunker.chunk(document)
    assert len(chunks) > 0
    assert hasattr(chunks[0], 'content')
    assert hasattr(chunks[0], 'chunk_id')
    print(f"  ✅ Chunking successful: {len(chunks)} chunks created")
    
except Exception as e:
    print(f"  ❌ Chunking test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 5: Vector Store Test
# ============================================================================

print("\n[TEST 5/6] Testing vector store...")

try:
    import numpy as np
    
    # Create test vectors
    test_ids = ["doc_1", "doc_2", "doc_3"]
    test_vectors = [np.random.randn(768).astype(np.float32) for _ in range(3)]
    test_payloads = [
        {"book_title": "Book 1", "content": "Content 1"},
        {"book_title": "Book 2", "content": "Content 2"},
        {"book_title": "Book 3", "content": "Content 3"},
    ]
    
    # Add vectors
    vector_store.add_vectors(test_ids, test_vectors, test_payloads)
    print(f"  ✅ Added {len(test_ids)} vectors")
    
    # Search
    query_vector = np.random.randn(768).astype(np.float32)
    results = vector_store.search(query_vector, top_k=2)
    
    assert len(results) > 0
    assert hasattr(results[0], 'score')
    assert hasattr(results[0], 'payload')
    print(f"  ✅ Search successful: {len(results)} results")
    
    # Get count
    count = vector_store.count()
    assert count == 3
    print(f"  ✅ Count correct: {count}")
    
except Exception as e:
    print(f"  ❌ Vector store test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Test 6: Query Transformation Test
# ============================================================================

print("\n[TEST 6/6] Testing query transformation...")

async def test_query_transformer():
    try:
        test_query = "ما حكم الزكاة؟"
        
        result = await transformer.transform(test_query)
        
        assert hasattr(result, 'original_query')
        assert hasattr(result, 'rewritten_query')
        assert hasattr(result, 'query_type')
        
        print(f"  ✅ Query transformation successful")
        print(f"     Original: {result.original_query}")
        print(f"     Type: {result.query_type.value}")
        
    except Exception as e:
        print(f"  ❌ Query transformation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Run async test
asyncio.run(test_query_transformer())

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✅")
print("=" * 70)
print("""
Test Results:
  ✅ Import tests (all modules)
  ✅ Configuration tests
  ✅ Component instantiation tests
  ✅ Chunking tests
  ✅ Vector store tests
  ✅ Query transformation tests

The RAG system architecture is correctly fixed and all components
are working properly.

Next Steps:
  1. Run example_usage_complete.py for full demo
  2. Index documents with: await rag.index_documents(limit=100)
  3. Start API server: uvicorn rag_system.src.api.service:app --reload
  4. Deploy to production
""")
