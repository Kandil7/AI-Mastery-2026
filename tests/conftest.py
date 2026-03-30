"""
Pytest Fixtures for AI-Mastery-2026
====================================

Reusable test fixtures for all test modules.

Usage:
------
    def test_something(sample_documents, vector_store):
        # Use fixtures
        pass

Fixture Categories:
-------------------
- Documents: Sample documents for testing
- Chunks: Pre-chunked documents
- Embeddings: Mock embedding models
- Vector Stores: In-memory vector stores
- RAG: Complete RAG pipeline components
- Config: Test configurations
- Temp directories: Temporary file storage
"""

import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import tempfile
import shutil

# Import test doubles
from unittest.mock import Mock, MagicMock


# ============================================================
# Document Fixtures
# ============================================================

@pytest.fixture
def sample_text() -> str:
    """Fixture providing sample text for testing."""
    return """
    Artificial intelligence is transforming the world.
    Machine learning is a subset of AI that enables systems to learn from data.
    Deep learning uses neural networks with multiple layers.
    Transformers have revolutionized natural language processing.
    Large language models like GPT-4 are powerful tools for text generation.
    Retrieval-Augmented Generation combines retrieval with generation.
    Vector databases store embeddings for efficient similarity search.
    Semantic search understands query intent rather than keyword matching.
    """ * 5


@pytest.fixture
def sample_documents() -> List[Dict[str, Any]]:
    """Fixture providing sample documents for testing."""
    return [
        {
            "id": f"doc_{i}",
            "content": f"This is test document {i} with some content for testing purposes. " * 10,
            "metadata": {
                "source": "test",
                "category": f"cat_{i % 3}",
                "created_at": datetime.utcnow().isoformat(),
            },
        }
        for i in range(10)
    ]


@pytest.fixture
def sample_document(sample_documents) -> Dict[str, Any]:
    """Fixture providing a single sample document."""
    return sample_documents[0]


@pytest.fixture
def long_document() -> Dict[str, Any]:
    """Fixture providing a long document for chunking tests."""
    content = " ".join([f"Paragraph {i}." for i in range(100)])
    return {
        "id": "long_doc",
        "content": content,
        "metadata": {"source": "test", "length": "long"},
    }


@pytest.fixture
def multi_language_documents() -> List[Dict[str, Any]]:
    """Fixture providing multi-language documents."""
    return [
        {
            "id": "doc_en",
            "content": "This is an English document about artificial intelligence.",
            "metadata": {"language": "en"},
        },
        {
            "id": "doc_es",
            "content": "Este es un documento en español sobre inteligencia artificial.",
            "metadata": {"language": "es"},
        },
        {
            "id": "doc_fr",
            "content": "Ceci est un document en français sur l'intelligence artificielle.",
            "metadata": {"language": "fr"},
        },
    ]


# ============================================================
# Chunking Fixtures
# ============================================================

@pytest.fixture
def sample_chunks(sample_documents) -> List[Dict[str, Any]]:
    """Fixture providing sample chunks from documents."""
    chunks = []
    chunk_id = 0
    
    for doc in sample_documents:
        content = doc["content"]
        chunk_size = 100
        
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i:i + chunk_size]
            chunks.append({
                "id": f"chunk_{chunk_id}",
                "content": chunk_content,
                "doc_id": doc["id"],
                "chunk_index": len([c for c in chunks if c["doc_id"] == doc["id"]]),
                "start_char": i,
                "end_char": min(i + chunk_size, len(content)),
                "metadata": doc["metadata"].copy(),
            })
            chunk_id += 1
    
    return chunks


@pytest.fixture
def chunk_sizes() -> List[int]:
    """Fixture providing various chunk sizes for testing."""
    return [64, 128, 256, 512, 1024]


@pytest.fixture
def overlap_sizes() -> List[int]:
    """Fixture providing various overlap sizes for testing."""
    return [0, 25, 50, 100]


# ============================================================
# Embedding Fixtures
# ============================================================

@pytest.fixture
def embedding_dim() -> int:
    """Fixture providing embedding dimension."""
    return 384


@pytest.fixture
def sample_embedding(embedding_dim) -> np.ndarray:
    """Fixture providing a sample embedding vector."""
    return np.random.randn(embedding_dim).astype(np.float32)


@pytest.fixture
def sample_embeddings(embedding_dim) -> np.ndarray:
    """Fixture providing sample embedding matrix."""
    return np.random.randn(10, embedding_dim).astype(np.float32)


@pytest.fixture
def dummy_embeddings_model(embedding_dim):
    """Fixture providing dummy embedding model for testing."""
    model = Mock()
    model.dim = embedding_dim
    model.encode = Mock(side_effect=lambda text: np.random.randn(embedding_dim).astype(np.float32))
    model.encode_batch = Mock(
        side_effect=lambda texts, **kwargs: np.random.randn(len(texts), embedding_dim).astype(np.float32)
    )
    return model


@pytest.fixture
def zero_embedding(embedding_dim) -> np.ndarray:
    """Fixture providing a zero embedding for edge case testing."""
    return np.zeros(embedding_dim, dtype=np.float32)


@pytest.fixture
def unit_embedding(embedding_dim) -> np.ndarray:
    """Fixture providing a unit embedding for normalization testing."""
    embedding = np.zeros(embedding_dim, dtype=np.float32)
    embedding[0] = 1.0
    return embedding


# ============================================================
# Vector Store Fixtures
# ============================================================

@pytest.fixture
def vector_store_config(embedding_dim) -> Dict[str, Any]:
    """Fixture providing vector store configuration."""
    return {
        "dim": embedding_dim,
        "index_type": "flat",
        "metric": "cosine",
    }


@pytest.fixture
def temp_vector_store_path(tmp_path: Path) -> Path:
    """Fixture providing temporary path for vector store."""
    return tmp_path / "vector_store"


@pytest.fixture
def indexed_embeddings(sample_embeddings) -> List[Dict[str, Any]]:
    """Fixture providing embeddings with IDs for indexing."""
    return [
        {
            "id": f"vec_{i}",
            "embedding": sample_embeddings[i],
            "metadata": {"index": i, "category": f"cat_{i % 3}"},
        }
        for i in range(len(sample_embeddings))
    ]


# ============================================================
# RAG Pipeline Fixtures
# ============================================================

@pytest.fixture
def rag_config() -> Dict[str, Any]:
    """Fixture providing RAG pipeline configuration."""
    return {
        "chunk_size": 512,
        "chunk_overlap": 50,
        "retrieval_top_k": 5,
        "rerank_top_k": 3,
        "similarity_threshold": 0.7,
        "cache_enabled": True,
        "cache_ttl_hours": 24,
    }


@pytest.fixture
def sample_query() -> str:
    """Fixture providing a sample query."""
    return "What is artificial intelligence?"


@pytest.fixture
def sample_queries() -> List[str]:
    """Fixture providing multiple sample queries."""
    return [
        "What is machine learning?",
        "How do transformers work?",
        "Explain retrieval-augmented generation",
        "What are vector databases?",
        "How to improve RAG performance?",
    ]


@pytest.fixture
def retrieval_results(sample_documents, sample_embeddings) -> List[Dict[str, Any]]:
    """Fixture providing mock retrieval results."""
    return [
        {
            "id": doc["id"],
            "content": doc["content"],
            "score": 0.9 - i * 0.1,
            "metadata": doc["metadata"],
            "embedding": sample_embeddings[i],
        }
        for i, doc in enumerate(sample_documents[:5])
    ]


# ============================================================
# Cache Fixtures
# ============================================================

@pytest.fixture
def cache_config() -> Dict[str, Any]:
    """Fixture providing cache configuration."""
    return {
        "enabled": True,
        "backend": "memory",
        "ttl_seconds": 3600,
        "max_size": 1000,
        "similarity_threshold": 0.95,
    }


@pytest.fixture
def cached_queries() -> List[Dict[str, Any]]:
    """Fixture providing cached query responses."""
    return [
        {
            "query": "What is AI?",
            "query_embedding": np.random.randn(384).astype(np.float32),
            "response": "AI stands for Artificial Intelligence...",
            "sources": [{"id": "doc_1", "content": "..."}],
            "created_at": datetime.utcnow(),
            "hit_count": 5,
        },
        {
            "query": "Explain machine learning",
            "query_embedding": np.random.randn(384).astype(np.float32),
            "response": "Machine learning is a subset of AI...",
            "sources": [{"id": "doc_2", "content": "..."}],
            "created_at": datetime.utcnow() - timedelta(hours=1),
            "hit_count": 3,
        },
    ]


# ============================================================
# Configuration Fixtures
# ============================================================

@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Fixture providing test configuration."""
    return {
        "environment": "test",
        "debug": True,
        "log_level": "DEBUG",
        "database": {
            "url": "sqlite:///test.db",
            "pool_size": 5,
        },
        "redis": {
            "url": "redis://localhost:6379/0",
        },
        "api": {
            "host": "localhost",
            "port": 8001,  # Use different port for tests
            "debug": True,
        },
    }


@pytest.fixture
def temp_env_vars(monkeypatch):
    """Fixture providing temporary environment variables."""
    def _set_env_vars(vars_dict: Dict[str, str]):
        for key, value in vars_dict.items():
            monkeypatch.setenv(key, value)
        return vars_dict
    return _set_env_vars


# ============================================================
# Temporary Directory Fixtures
# ============================================================

@pytest.fixture
def temp_directory(tmp_path: Path) -> Path:
    """Fixture providing temporary directory for file operations."""
    temp_dir = tmp_path / "test_data"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


@pytest.fixture
def temp_file(temp_directory: Path) -> Path:
    """Fixture providing temporary file."""
    temp_file = temp_directory / "test.txt"
    temp_file.write_text("Test content")
    return temp_file


@pytest.fixture
def temp_json_file(temp_directory: Path) -> Path:
    """Fixture providing temporary JSON file."""
    import json
    temp_file = temp_directory / "test.json"
    with open(temp_file, "w") as f:
        json.dump({"test": "data", "number": 42})
    return temp_file


@pytest.fixture
def temp_csv_file(temp_directory: Path) -> Path:
    """Fixture providing temporary CSV file."""
    temp_file = temp_directory / "test.csv"
    with open(temp_file, "w") as f:
        f.write("id,name,value\n")
        f.write("1,test1,100\n")
        f.write("2,test2,200\n")
    return temp_file


# ============================================================
# API Fixtures
# ============================================================

@pytest.fixture
def api_client():
    """Fixture providing test API client."""
    from fastapi.testclient import TestClient
    # Import your app here
    # from src.production.api import app
    # return TestClient(app)
    return None


@pytest.fixture
def auth_headers():
    """Fixture providing authentication headers."""
    return {
        "Authorization": "Bearer test_token",
        "Content-Type": "application/json",
    }


# ============================================================
# Performance Fixtures
# ============================================================

@pytest.fixture
def benchmark_iterations() -> int:
    """Fixture providing number of benchmark iterations."""
    return 100


@pytest.fixture
def benchmark_warmup() -> int:
    """Fixture providing number of warmup iterations."""
    return 10


@pytest.fixture
def performance_thresholds() -> Dict[str, float]:
    """Fixture providing performance thresholds."""
    return {
        "embedding_latency_ms": 100,
        "retrieval_latency_ms": 500,
        "generation_latency_ms": 5000,
        "total_latency_ms": 6000,
    }


# ============================================================
# Error Fixtures
# ============================================================

@pytest.fixture
def error_messages() -> Dict[str, str]:
    """Fixture providing common error messages."""
    return {
        "not_found": "Resource not found",
        "validation": "Validation error",
        "authentication": "Authentication required",
        "authorization": "Permission denied",
        "rate_limit": "Rate limit exceeded",
        "internal": "Internal server error",
    }


@pytest.fixture
def invalid_inputs() -> List[Dict[str, Any]]:
    """Fixture providing invalid inputs for error testing."""
    return [
        {"query": ""},  # Empty query
        {"query": None},  # None query
        {"top_k": -1},  # Negative top_k
        {"top_k": 1001},  # Too large top_k
        {"filters": "invalid"},  # Invalid filters type
    ]


# ============================================================
# Utility Fixtures
# ============================================================

@pytest.fixture
def random_seed() -> int:
    """Fixture providing random seed for reproducibility."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seed(random_seed):
    """Fixture setting random seed for all tests."""
    np.random.seed(random_seed)
    return random_seed


@pytest.fixture
def skip_slow_tests(request) -> bool:
    """Fixture for conditionally skipping slow tests."""
    return request.config.getoption("--run-slow")


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "e2e: marks tests as end-to-end tests")
    config.addinivalue_line("markers", "requires_api: marks tests that require API access")
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests",
    )
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests",
    )
