"""
Migration Verification Script
=============================

Tests all new imports and functionality after the src/ reorganization.

Run with:
    python scripts/verify_migration.py

Or:
    python -m src.verify_migration
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


def test_import(description: str, import_statement: str):
    """Test an import and print result."""
    try:
        exec(import_statement)
        print(f"✅ {description}")
        return True
    except Exception as e:
        print(f"❌ {description}")
        print(f"   Error: {e}")
        return False


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("AI-Mastery-2026 Migration Verification")
    print("=" * 60)
    print()

    results = []

    # Test 1: Main package imports
    print("1. Main Package Imports")
    print("-" * 40)
    results.append(test_import(
        "from src import rag",
        "from src import rag"
    ))
    results.append(test_import(
        "from src import agents",
        "from src import agents"
    ))
    results.append(test_import(
        "from src import embeddings",
        "from src import embeddings"
    ))
    results.append(test_import(
        "from src import vector_stores",
        "from src import vector_stores"
    ))
    results.append(test_import(
        "from src import production",
        "from src import production"
    ))
    print()

    # Test 2: Vector stores module
    print("2. Vector Stores Module")
    print("-" * 40)
    results.append(test_import(
        "from src.vector_stores import VectorStoreConfig",
        "from src.vector_stores import VectorStoreConfig"
    ))
    results.append(test_import(
        "from src.vector_stores import MemoryVectorStore",
        "from src.vector_stores import MemoryVectorStore"
    ))
    results.append(test_import(
        "from src.vector_stores import FAISSStore",
        "from src.vector_stores import FAISSStore"
    ))
    results.append(test_import(
        "from src.vector_stores.base import VectorStore",
        "from src.vector_stores.base import VectorStore"
    ))
    print()

    # Test 3: RAG retrieval module
    print("3. RAG Retrieval Module")
    print("-" * 40)
    results.append(test_import(
        "from src.rag.retrieval import BaseRetriever",
        "from src.rag.retrieval import BaseRetriever"
    ))
    results.append(test_import(
        "from src.rag.retrieval import SimilarityRetriever",
        "from src.rag.retrieval import SimilarityRetriever"
    ))
    results.append(test_import(
        "from src.rag.retrieval import HybridRetrieval",
        "from src.rag.retrieval import HybridRetrieval"
    ))
    results.append(test_import(
        "from src.rag.retrieval import MultiQueryRetriever",
        "from src.rag.retrieval import MultiQueryRetriever"
    ))
    results.append(test_import(
        "from src.rag.retrieval import HyDERetriever",
        "from src.rag.retrieval import HyDERetriever"
    ))
    print()

    # Test 4: RAG reranking module
    print("4. RAG Reranking Module")
    print("-" * 40)
    results.append(test_import(
        "from src.rag.reranking import BaseReranker",
        "from src.rag.reranking import BaseReranker"
    ))
    results.append(test_import(
        "from src.rag.reranking import CrossEncoderReranker",
        "from src.rag.reranking import CrossEncoderReranker"
    ))
    results.append(test_import(
        "from src.rag.reranking import LLMReranker",
        "from src.rag.reranking import LLMReranker"
    ))
    results.append(test_import(
        "from src.rag.reranking import DiversityReranker",
        "from src.rag.reranking import DiversityReranker"
    ))
    print()

    # Test 5: RAG main module
    print("5. RAG Main Module")
    print("-" * 40)
    results.append(test_import(
        "from src.rag import RAGPipeline",
        "from src.rag import RAGPipeline"
    ))
    results.append(test_import(
        "from src.rag import Document, DocumentChunk",
        "from src.rag import Document, DocumentChunk"
    ))
    results.append(test_import(
        "from src.rag.chunking import SemanticChunker",
        "from src.rag.chunking import SemanticChunker"
    ))
    results.append(test_import(
        "from src.rag.chunking import RecursiveChunker",
        "from src.rag.chunking import RecursiveChunker"
    ))
    print()

    # Test 6: Agents module
    print("6. Agents Module")
    print("-" * 40)
    results.append(test_import(
        "from src.agents import ReActAgent",
        "from src.agents import ReActAgent"
    ))
    results.append(test_import(
        "from src.agents import ToolRegistry",
        "from src.agents import ToolRegistry"
    ))
    print()

    # Test 7: Embeddings module
    print("7. Embeddings Module")
    print("-" * 40)
    results.append(test_import(
        "from src.embeddings import TextEmbedder",
        "from src.embeddings import TextEmbedder"
    ))
    results.append(test_import(
        "from src.embeddings import ImageEmbedder",
        "from src.embeddings import ImageEmbedder"
    ))
    print()

    # Test 8: Production module
    print("8. Production Module")
    print("-" * 40)
    results.append(test_import(
        "from src.production import FastAPIApp",
        "from src.production import FastAPIApp"
    ))
    results.append(test_import(
        "from src.production import SemanticCache",
        "from src.production import SemanticCache"
    ))
    print()

    # Test 9: Utils module
    print("9. Utils Module")
    print("-" * 40)
    results.append(test_import(
        "from src.utils.logging import get_logger",
        "from src.utils.logging import get_logger"
    ))
    results.append(test_import(
        "from src.utils.logging import log_performance",
        "from src.utils.logging import log_performance"
    ))
    results.append(test_import(
        "from src.utils.errors import AIError",
        "from src.utils.errors import AIError"
    ))
    print()

    # Test 10: Functionality tests
    print("10. Functionality Tests")
    print("-" * 40)

    # Test vector store functionality
    try:
        from src.vector_stores import MemoryVectorStore, VectorStoreConfig

        config = VectorStoreConfig(dim=3)
        store = MemoryVectorStore(config)
        store.initialize()

        vectors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        ids = ["a", "b", "c"]
        store.upsert(vectors, ids)

        results = store.search([1, 0, 0], top_k=2)

        assert len(results) == 2, f"Expected 2 results, got {len(results)}"
        assert results[0].id == "a", f"Expected first result to be 'a', got {results[0].id}"

        print("✅ MemoryVectorStore functionality works")
        results.append(True)
    except Exception as e:
        print(f"❌ MemoryVectorStore functionality failed")
        print(f"   Error: {e}")
        results.append(False)

    # Test retrieval functionality
    try:
        from src.rag.retrieval import SimilarityRetriever
        from src.vector_stores import MemoryVectorStore, VectorStoreConfig

        config = VectorStoreConfig(dim=3)
        store = MemoryVectorStore(config)
        store.initialize()
        store.upsert([[1, 0, 0]], ["test"])

        retriever = SimilarityRetriever(store, top_k=1)
        # Note: Full test would need content in metadata
        print("✅ SimilarityRetriever initializes correctly")
        results.append(True)
    except Exception as e:
        print(f"❌ SimilarityRetriever failed")
        print(f"   Error: {e}")
        results.append(False)

    print()

    # Summary
    print("=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)

    passed = sum(results)
    total = len(results)
    percentage = (passed / total) * 100 if total > 0 else 0

    print(f"Passed: {passed}/{total} ({percentage:.1f}%)")

    if percentage == 100:
        print("\n✅ ALL TESTS PASSED! Migration successful.")
        return 0
    else:
        print(f"\n⚠️ {total - passed} test(s) failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
