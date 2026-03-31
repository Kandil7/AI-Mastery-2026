"""
Test Fixtures and Utilities for AI-Mastery-2026
================================================

Common fixtures, mocks, and utilities for testing.

Usage:
    from tests.conftest import sample_document, mock_embedding_model
"""

import numpy as np
import pytest
from typing import Any, Dict, List, Optional
from pathlib import Path
import tempfile
import shutil


# ============================================================
# DOCUMENT FIXTURES
# ============================================================

@pytest.fixture
def sample_document():
    """Create a sample document for testing."""
    return {
        "id": "test-doc-001",
        "content": "This is a test document for AI-Mastery-2026 testing.",
        "metadata": {
            "source": "test",
            "created_at": "2026-03-31",
        }
    }


@pytest.fixture
def sample_documents():
    """Create multiple sample documents."""
    return [
        {
            "id": f"test-doc-{i:03d}",
            "content": f"Test document content number {i}. " * 10,
            "metadata": {"index": i}
        }
        for i in range(5)
    ]


@pytest.fixture
def sample_query():
    """Create a sample query."""
    return {
        "text": "What is AI-Mastery-2026?",
        "metadata": {}
    }


# ============================================================
# NUMPY ARRAY FIXTURES
# ============================================================

@pytest.fixture
def sample_embedding():
    """Create a sample embedding vector."""
    return np.random.randn(384).astype(np.float32)


@pytest.fixture
def sample_embeddings():
    """Create multiple sample embeddings."""
    return np.random.randn(10, 384).astype(np.float32)


@pytest.fixture
def sample_attention_matrix():
    """Create a sample attention matrix."""
    seq_len = 16
    matrix = np.random.randn(seq_len, seq_len).astype(np.float32)
    # Make it symmetric (like attention matrices often are)
    return (matrix + matrix.T) / 2


# ============================================================
# MODEL OUTPUT FIXTURES
# ============================================================

@pytest.fixture
def sample_model_output():
    """Create a sample model output."""
    return {
        "loss": 0.5,
        "logits": np.random.randn(10, 4),
        "predictions": [0, 1, 2, 3, 0, 1, 2, 3, 0, 1]
    }


@pytest.fixture
def sample_classification_data():
    """Create sample classification data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    n_classes = 4

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, n_classes, n_samples)

    return X, y


@pytest.fixture
def sample_regression_data():
    """Create sample regression data."""
    np.random.seed(42)
    n_samples = 100
    n_features = 5

    X = np.random.randn(n_samples, n_features)
    y = np.sum(X[:, :3], axis=1) + np.random.randn(n_samples) * 0.1

    return X, y


# ============================================================
# TEMPORARY DIRECTORY FIXTURES
# ============================================================

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath)


@pytest.fixture
def temp_file(temp_dir):
    """Create a temporary file for testing."""
    filepath = temp_dir / "test.txt"
    filepath.write_text("Test content")
    yield filepath


# ============================================================
# MOCK CLASSES
# ============================================================

class MockEmbeddingModel:
    """Mock embedding model for testing."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """Return random embeddings."""
        return np.random.randn(len(texts), self.dim).astype(np.float32)

    def __call__(self, texts: List[str]) -> np.ndarray:
        return self.encode(texts)


class MockVectorStore:
    """Mock vector store for testing."""

    def __init__(self):
        self.documents: Dict[str, Any] = {}
        self.embeddings: Dict[str, np.ndarray] = {}

    def add(self, ids: List[str], embeddings: np.ndarray, documents: List[Dict]):
        """Add documents to store."""
        for i, doc_id in enumerate(ids):
            self.documents[doc_id] = documents[i]
            self.embeddings[doc_id] = embeddings[i]

    def search(self, query_embedding: np.ndarray, k: int = 5):
        """Return random search results."""
        ids = list(self.documents.keys())[:k]
        scores = np.random.rand(k).astype(np.float32)
        return {
            "ids": ids,
            "scores": scores,
            "documents": [self.documents[id] for id in ids]
        }


class MockLLM:
    """Mock LLM for testing."""

    def generate(self, prompt: str, **kwargs) -> str:
        """Return a mock response."""
        return f"Mock response to: {prompt[:50]}..."

    def __call__(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)


@pytest.fixture
def mock_embedding_model():
    """Create a mock embedding model."""
    return MockEmbeddingModel()


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store."""
    return MockVectorStore()


@pytest.fixture
def mock_llm():
    """Create a mock LLM."""
    return MockLLM()


# ============================================================
# ASSERTION HELPERS
# ============================================================

def assert_arrays_close(actual: np.ndarray, expected: np.ndarray, rtol: float = 1e-5):
    """Assert two arrays are close."""
    np.testing.assert_allclose(actual, expected, rtol=rtol)


def assert_shape(array: np.ndarray, expected_shape: tuple):
    """Assert array has expected shape."""
    assert array.shape == expected_shape, f"Expected shape {expected_shape}, got {array.shape}"


def assert_valid_probability_distribution(probs: np.ndarray):
    """Assert array is a valid probability distribution."""
    assert np.all(probs >= 0), "Probabilities must be non-negative"
    np.testing.assert_allclose(np.sum(probs), 1.0, rtol=1e-5)


# ============================================================
# CONFIGURATION FIXTURES
# ============================================================

@pytest.fixture
def test_settings():
    """Create test settings."""
    from src.config import Settings, Environment

    return Settings(
        environment=Environment.TESTING,
        debug=True,
        batch_size=4,
    )


@pytest.fixture
def test_transformer_config():
    """Create test transformer config."""
    from src.config import TransformerConfig

    return TransformerConfig(
        hidden_dim=128,
        num_heads=4,
        num_layers=2,
        max_seq_len=64,
        vocab_size=1000,
    )


@pytest.fixture
def test_training_config():
    """Create test training config."""
    from src.config import TrainingConfig

    return TrainingConfig(
        learning_rate=1e-3,
        batch_size=4,
        num_epochs=1,
        warmup_steps=10,
    )
