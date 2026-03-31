# 💻 Code Organization Principles

**AI-Mastery-2026: Domain-Driven Design for Educational Code**

| Document Info | Details |
|---------------|---------|
| **Version** | 3.0 |
| **Date** | March 30, 2026 |
| **Status** | Standard |
| **Architecture Pattern** | Domain-Driven Design + Layered Architecture |

---

## 📋 Executive Summary

This document defines the **code organization principles** for AI-Mastery-2026, applying **domain-driven design** and **layered architecture** patterns to ensure:

- ✅ **Clear separation of concerns** across all code
- ✅ **Consistent import hierarchy** for all modules
- ✅ **Test organization** that mirrors production code
- ✅ **Example vs production code** separation
- ✅ **Dependency management** best practices

---

## 🏗️ Architectural Principles

### Principle 1: Layered Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                                    │
│  (Notebooks, Scripts, CLI, API Endpoints - User-facing interfaces)          │
├─────────────────────────────────────────────────────────────────────────────┤
│                         DOMAIN LAYER                                         │
│  (RAG, Agents, LLM - Business logic, domain-specific implementations)       │
├─────────────────────────────────────────────────────────────────────────────┤
│                         ML LAYER                                             │
│  (Classical ML, Deep Learning, Vision - ML algorithms and models)           │
├─────────────────────────────────────────────────────────────────────────────┤
│                         CORE LAYER                                           │
│  (Math, Probability, Optimization - Foundational utilities from scratch)    │
├─────────────────────────────────────────────────────────────────────────────┤
│                         UTILITIES LAYER (Cross-cutting)                      │
│  (Logging, Configuration, Error Handling, Types - Shared across all layers) │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Dependency Rule:** Dependencies point inward. Outer layers depend on inner layers, never vice versa.

### Principle 2: Domain-Driven Design

| Domain | Responsibility | Key Modules |
|--------|----------------|-------------|
| **Core** | Mathematical foundations | `math/`, `probability/`, `optimization/` |
| **ML** | Machine learning algorithms | `classical/`, `deep_learning/`, `vision/` |
| **LLM** | Language model architectures | `architecture/`, `training/`, `alignment/` |
| **RAG** | Retrieval-augmented generation | `chunking/`, `retrieval/`, `reranking/`, `vector_stores/` |
| **Agents** | AI agent systems | `core/`, `memory/`, `tools/`, `multi_agent/` |
| **Production** | Deployment and operations | `api/`, `monitoring/`, `deployment/`, `security/` |

### Principle 3: Single Responsibility

Each module should have **one clear responsibility**:

```python
# ✅ GOOD: Single responsibility
# src/rag/chunking/semantic.py
class SemanticChunker:
    """Chunks text based on semantic similarity."""
    pass

# ❌ BAD: Multiple responsibilities
# src/rag/chunking.py (contains chunking, embedding, and retrieval)
```

### Principle 4: Explicit Over Implicit

```python
# ✅ GOOD: Explicit imports
from src.rag.chunking.semantic import SemanticChunker
from src.rag.retrieval.dense import DenseRetriever

# ❌ BAD: Implicit/wildcard imports
from src.rag import *
from src.rag.chunking import *
```

---

## 📁 Source Code Structure

### Root src/ Organization

```
src/
├── __init__.py                          # Package initialization + public API
├── README.md                            # src/ overview
│
├── core/                                # CORE LAYER: Foundations
│   ├── __init__.py
│   ├── README.md                        # Module documentation
│   ├── math/
│   │   ├── __init__.py
│   │   ├── vectors.py                   # Vector operations from scratch
│   │   ├── matrices.py                  # Matrix operations from scratch
│   │   ├── calculus.py                  # Numerical calculus
│   │   └── decompositions.py            # SVD, QR, Cholesky
│   ├── probability/
│   │   ├── __init__.py
│   │   ├── distributions.py             # Probability distributions
│   │   ├── bayes.py                     # Bayes theorem
│   │   └── hypothesis_testing.py        # Statistical tests
│   └── optimization/
│       ├── __init__.py
│       ├── optimizers.py                # SGD, Adam, RMSprop
│       └── loss_functions.py            # Loss functions
│
├── ml/                                  # ML LAYER: Machine Learning
│   ├── __init__.py
│   ├── README.md
│   ├── classical/
│   │   ├── __init__.py
│   │   ├── linear_regression.py
│   │   ├── logistic_regression.py
│   │   ├── decision_trees.py
│   │   ├── random_forests.py
│   │   ├── svm.py
│   │   └── kmeans.py
│   ├── deep_learning/
│   │   ├── __init__.py
│   │   ├── layers.py                    # Dense, Conv2D, LSTM layers
│   │   ├── activations.py               # ReLU, Sigmoid, Softmax
│   │   ├── losses.py                    # MSE, CrossEntropy
│   │   ├── mlp.py                       # Multi-layer perceptron
│   │   ├── cnn.py                       # Convolutional networks
│   │   └── rnn.py                       # Recurrent networks
│   └── vision/
│       ├── __init__.py
│       ├── resnet.py                    # ResNet architecture
│       └── vit.py                       # Vision Transformer
│
├── llm/                                 # DOMAIN LAYER: LLM
│   ├── __init__.py
│   ├── README.md
│   ├── architecture/
│   │   ├── __init__.py
│   │   ├── attention.py                 # Multi-head attention
│   │   ├── transformer.py               # Transformer from scratch
│   │   ├── tokenization.py              # BPE, WordPiece
│   │   └── positional_encodings.py      # Sinusoidal, RoPE
│   ├── training/
│   │   ├── __init__.py
│   │   ├── pretraining.py               # Pre-training loops
│   │   ├── fine_tuning.py               # Full fine-tuning
│   │   ├── lora.py                      # LoRA adapters
│   │   └── qlora.py                     # QLoRA implementation
│   └── alignment/
│       ├── __init__.py
│       ├── rlhf.py                      # RLHF implementation
│       └── dpo.py                       # Direct Preference Optimization
│
├── rag/                                 # DOMAIN LAYER: RAG
│   ├── __init__.py
│   ├── README.md
│   ├── chunking/
│   │   ├── __init__.py
│   │   ├── base.py                      # BaseChunker ABC
│   │   ├── fixed_size.py
│   │   ├── recursive.py
│   │   ├── semantic.py
│   │   └── hierarchical.py
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── base.py                      # EmbeddingModel ABC
│   │   └── sentence_transformers.py
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── base.py                      # Retriever ABC
│   │   ├── dense.py
│   │   ├── sparse.py
│   │   └── hybrid.py
│   ├── reranking/
│   │   ├── __init__.py
│   │   ├── base.py                      # Reranker ABC
│   │   └── cross_encoder.py
│   ├── vector_stores/
│   │   ├── __init__.py
│   │   ├── base.py                      # VectorStore ABC
│   │   ├── faiss_store.py
│   │   ├── qdrant_store.py
│   │   └── chroma_store.py
│   └── pipeline/
│       ├── __init__.py
│       ├── base.py                      # RAGPipeline ABC
│       ├── standard.py
│       └── advanced.py
│
├── agents/                              # DOMAIN LAYER: Agents
│   ├── __init__.py
│   ├── README.md
│   ├── core/
│   │   ├── __init__.py
│   │   ├── base.py                      # BaseAgent ABC
│   │   ├── react.py                     # ReAct agent
│   │   └── planning.py                  # Planning agent
│   ├── memory/
│   │   ├── __init__.py
│   │   ├── short_term.py
│   │   └── long_term.py
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                      # BaseTool ABC
│   │   ├── search.py
│   │   └── code_interpreter.py
│   └── multi_agent/
│       ├── __init__.py
│       ├── coordinator.py
│       └── protocols.py
│
├── production/                          # APPLICATION LAYER: Production
│   ├── __init__.py
│   ├── README.md
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py                       # FastAPI application
│   │   ├── routes/
│   │   │   ├── __init__.py
│   │   │   ├── health.py
│   │   │   ├── rag.py
│   │   │   └── agents.py
│   │   ├── schemas/
│   │   │   ├── __init__.py
│   │   │   ├── requests.py
│   │   │   └── responses.py
│   │   └── middleware/
│   │       ├── __init__.py
│   │       ├── auth.py
│   │       └── rate_limit.py
│   ├── monitoring/
│   │   ├── __init__.py
│   │   ├── metrics.py                   # Prometheus metrics
│   │   ├── tracing.py                   # Distributed tracing
│   │   └── alerting.py                  # Alert rules
│   ├── deployment/
│   │   ├── __init__.py
│   │   ├── docker.py
│   │   ├── kubernetes.py
│   │   └── vllm.py
│   └── security/
│       ├── __init__.py
│       ├── auth.py                      # Authentication
│       ├── rate_limit.py                # Rate limiting
│       └── guardrails.py                # Content safety
│
├── utils/                               # UTILITIES LAYER: Cross-cutting
│   ├── __init__.py
│   ├── errors.py                        # Error hierarchy
│   ├── logging.py                       # Logging setup
│   ├── config.py                        # Configuration
│   └── types.py                         # Type definitions
│
└── data/                                # UTILITIES LAYER: Data
    ├── __init__.py
    ├── loading.py                       # Data loading
    ├── preprocessing.py                 # Data preprocessing
    └── versioning.py                    # Data versioning
```

---

## 📦 Import Hierarchy Rules

### Rule 1: Import Direction

```python
# ✅ CORRECT: Inner → Outer (allowed)
from src.core.math.vectors import Vector
from src.ml.classical.linear_regression import LinearRegression
from src.rag.pipeline.standard import StandardRAGPipeline

# ❌ WRONG: Outer → Inner (not allowed)
# from src.rag import Vector  # Vector is in core, not rag
```

### Rule 2: Import Specificity

```python
# ✅ PREFERRED: Specific imports
from src.rag.chunking.semantic import SemanticChunker
from src.rag.retrieval.hybrid import HybridRetriever

# ⚠️ ACCEPTABLE: Module-level imports (for public API)
from src.rag import SemanticChunker, HybridRetriever

# ❌ AVOID: Wildcard imports
from src.rag.chunking import *
```

### Rule 3: Circular Dependency Prevention

```python
# ✅ CORRECT: Extract shared code to common module
# src/rag/shared/types.py
class Document:
    pass

# src/rag/chunking/semantic.py
from src.rag.shared.types import Document

# src/rag/retrieval/dense.py
from src.rag.shared.types import Document

# ❌ WRONG: Direct cross-imports
# src/rag/chunking/semantic.py
from src.rag.retrieval.dense import DenseRetriever  # Creates cycle
```

### Rule 4: Public API Exposure

```python
# src/rag/__init__.py
"""RAG module public API."""

from src.rag.chunking.semantic import SemanticChunker
from src.rag.chunking.hierarchical import HierarchicalChunker
from src.rag.retrieval.dense import DenseRetriever
from src.rag.retrieval.hybrid import HybridRetriever
from src.rag.pipeline.standard import StandardRAGPipeline
from src.rag.vector_stores.faiss_store import FAISSStore

__all__ = [
    # Chunking
    'SemanticChunker',
    'HierarchicalChunker',
    
    # Retrieval
    'DenseRetriever',
    'HybridRetriever',
    
    # Pipeline
    'StandardRAGPipeline',
    
    # Vector Stores
    'FAISSStore',
]
```

---

## 🧪 Test Organization

### Test Directory Structure

```
tests/
├── __init__.py
├── README.md
├── conftest.py                          # Shared fixtures
│
├── unit/                                # Unit tests
│   ├── core/
│   │   ├── math/
│   │   │   ├── test_vectors.py
│   │   │   ├── test_matrices.py
│   │   │   └── test_calculus.py
│   │   ├── probability/
│   │   │   └── test_distributions.py
│   │   └── optimization/
│   │       └── test_optimizers.py
│   ├── ml/
│   │   ├── classical/
│   │   │   ├── test_linear_regression.py
│   │   │   └── test_decision_trees.py
│   │   └── deep_learning/
│   │       ├── test_layers.py
│   │       └── test_mlp.py
│   ├── llm/
│   │   ├── architecture/
│   │   │   ├── test_attention.py
│   │   │   └── test_transformer.py
│   │   └── training/
│   │       └── test_lora.py
│   ├── rag/
│   │   ├── chunking/
│   │   │   ├── test_semantic_chunker.py
│   │   │   └── test_hierarchical_chunker.py
│   │   ├── retrieval/
│   │   │   └── test_dense_retriever.py
│   │   └── pipeline/
│   │       └── test_standard_rag.py
│   └── production/
│       ├── api/
│       │   └── test_routes.py
│       └── monitoring/
│           └── test_metrics.py
│
├── integration/                         # Integration tests
│   ├── test_rag_pipeline.py
│   ├── test_agent_workflow.py
│   └── test_api_endpoints.py
│
├── e2e/                                 # End-to-end tests
│   ├── test_full_rag_flow.py
│   └── test_production_deployment.py
│
└── performance/                         # Performance tests
    ├── test_latency.py
    ├── test_throughput.py
    └── test_memory.py
```

### Test File Template

```python
"""Tests for [module name]."""

import pytest
import numpy as np

from src.[module].[ submodule] import [ClassName]


class Test[ClassName]:
    """Test suite for [ClassName]."""
    
    @pytest.fixture
    def instance(self):
        """Create test instance."""
        return [ClassName]()
    
    def test_initialization(self, instance):
        """Test that instance initializes correctly."""
        assert instance is not None
    
    def test_main_method(self, instance):
        """Test main method with typical input."""
        # Arrange
        input_data = [...]
        expected = [...]
        
        # Act
        result = instance.main_method(input_data)
        
        # Assert
        assert result == expected
    
    def test_edge_case_empty_input(self, instance):
        """Test behavior with empty input."""
        with pytest.raises(ValueError):
            instance.main_method([])
    
    def test_edge_case_large_input(self, instance):
        """Test behavior with large input."""
        large_input = [1] * 10000
        result = instance.main_method(large_input)
        assert len(result) > 0
```

### Test Markers

```python
# conftest.py
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "e2e: marks end-to-end tests")
    config.addinivalue_line("markers", "requires_gpu: marks tests requiring GPU")
    config.addinivalue_line("markers", "requires_api_key: marks tests requiring API key")

# Usage in test files
@pytest.mark.slow
def test_large_model_training():
    pass

@pytest.mark.integration
def test_rag_pipeline_integration():
    pass

@pytest.mark.requires_gpu
def test_gpu_acceleration():
    pass
```

---

## 📚 Example vs Production Code

### Separation Strategy

```
src/
├── [production code]              # Importable, tested, documented

examples/                          # Separate examples directory
├── README.md
├── core/
│   ├── vectors_example.py
│   └── matrices_example.py
├── ml/
│   ├── linear_regression_example.py
│   └── neural_network_example.py
└── rag/
    ├── basic_rag_example.py
    └── advanced_rag_example.py

notebooks/                         # Interactive examples
├── README.md
├── 01_mathematical_foundations/
├── 02_classical_ml/
└── 03_rag_systems/
```

### Example Code Guidelines

```python
# examples/rag/basic_rag_example.py
"""
Basic RAG Example

This example demonstrates how to create a simple RAG pipeline.
For production usage, see src/rag/pipeline/standard.py
"""

from src.rag.chunking.semantic import SemanticChunker
from src.rag.retrieval.dense import DenseRetriever
from src.rag.vector_stores.faiss_store import FAISSStore


def main():
    """Run basic RAG example."""
    # Sample documents
    documents = [
        "AI is transforming industries.",
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
    ]
    
    # Create components
    chunker = SemanticChunker(chunk_size=100)
    vector_store = FAISSStore(embedding_dim=384)
    retriever = DenseRetriever(vector_store, top_k=2)
    
    # Process documents
    chunks = chunker.chunk("\n".join(documents))
    vector_store.add(chunks)
    
    # Query
    results = retriever.retrieve("What is AI?")
    print(f"Found {len(results)} relevant chunks")


if __name__ == "__main__":
    main()
```

### Production Code Guidelines

```python
# src/rag/pipeline/standard.py
"""Standard RAG pipeline implementation for production use."""

from __future__ import annotations

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import logging

from src.rag.chunking.base import BaseChunker
from src.rag.retrieval.base import BaseRetriever
from src.rag.vector_stores.base import BaseVectorStore
from src.utils.logging import get_logger
from src.utils.errors import RAGPipelineError

logger = get_logger(__name__)


@dataclass
class RAGResult:
    """Result from RAG pipeline."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    latency_ms: float


class StandardRAGPipeline:
    """
    Standard RAG pipeline for production use.
    
    This pipeline orchestrates chunking, retrieval, and generation
    with proper error handling, logging, and metrics.
    
    Attributes:
        chunker: Document chunker
        retriever: Document retriever
        generator: Response generator
        
    Example:
        >>> pipeline = StandardRAGPipeline(chunker, retriever, generator)
        >>> result = pipeline.execute("What is AI?")
        >>> print(result.answer)
    """
    
    def __init__(
        self,
        chunker: BaseChunker,
        retriever: BaseRetriever,
        generator: Any,
    ) -> None:
        """
        Initialize RAG pipeline.
        
        Args:
            chunker: Document chunker
            retriever: Document retriever
            generator: Response generator
            
        Raises:
            ValueError: If any component is None
        """
        if chunker is None:
            raise ValueError("Chunker cannot be None")
        if retriever is None:
            raise ValueError("Retriever cannot be None")
        if generator is None:
            raise ValueError("Generator cannot be None")
            
        self._chunker = chunker
        self._retriever = retriever
        self._generator = generator
        
    def execute(
        self,
        query: str,
        top_k: int = 5,
        **kwargs: Any
    ) -> RAGResult:
        """
        Execute RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            **kwargs: Additional arguments
            
        Returns:
            RAGResult with answer and sources
            
        Raises:
            RAGPipelineError: If pipeline execution fails
        """
        try:
            # Retrieve relevant documents
            documents = self._retriever.retrieve(query, top_k=top_k)
            
            # Generate response
            answer = self._generator.generate(query, documents)
            
            return RAGResult(
                answer=answer,
                sources=[doc.metadata for doc in documents],
                confidence=self._calculate_confidence(documents),
                latency_ms=0.0,  # Would be calculated
            )
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            raise RAGPipelineError(f"Pipeline failed: {e}") from e
    
    def _calculate_confidence(self, documents: List[Any]) -> float:
        """Calculate confidence score."""
        pass
```

---

## 📋 Dependency Management

### Requirements Structure

```
requirements/
├── base.txt                         # Core dependencies
├── dev.txt                          # Development dependencies
├── llm.txt                          # LLM-specific dependencies
├── vector.txt                       # Vector database dependencies
├── prod.txt                         # Production dependencies
└── test.txt                         # Testing dependencies
```

### requirements/base.txt

```txt
# Core Dependencies
numpy>=1.24.0,<2.0.0
pandas>=2.0.0,<3.0.0
scikit-learn>=1.3.0,<2.0.0

# Deep Learning
torch>=2.0.0,<3.0.0
torchvision>=0.15.0,<1.0.0

# NLP
transformers>=4.30.0,<5.0.0
tokenizers>=0.13.0,<1.0.0
sentencepiece>=0.1.99,<1.0.0

# Vector Search
faiss-cpu>=1.7.4,<2.0.0
numpy>=1.24.0,<2.0.0

# Utilities
pydantic>=2.0.0,<3.0.0
pyyaml>=6.0.0,<7.0.0
tqdm>=4.65.0,<5.0.0
```

### requirements/dev.txt

```txt
-r base.txt

# Development
pre-commit>=3.3.0,<4.0.0
black>=23.0.0,<24.0.0
isort>=5.12.0,<6.0.0
flake8>=6.0.0,<7.0.0
mypy>=1.0.0,<2.0.0
pylint>=2.17.0,<3.0.0

# Jupyter
jupyter>=1.0.0,<2.0.0
ipykernel>=6.23.0,<7.0.0
nbconvert>=7.0.0,<8.0.0

# Documentation
mkdocs>=1.4.0,<2.0.0
mkdocs-material>=9.0.0,<10.0.0
```

### requirements/test.txt

```txt
-r base.txt

# Testing
pytest>=7.3.0,<8.0.0
pytest-cov>=4.0.0,<5.0.0
pytest-asyncio>=0.21.0,<1.0.0
pytest-mock>=3.10.0,<4.0.0
hypothesis>=6.75.0,<7.0.0

# Integration Testing
httpx>=0.24.0,<1.0.0
pytest-httpserver>=1.0.0,<2.0.0
```

### requirements/prod.txt

```txt
-r base.txt

# API
fastapi>=0.100.0,<1.0.0
uvicorn[standard]>=0.22.0,<1.0.0
python-multipart>=0.0.6,<1.0.0

# Security
python-jose[cryptography]>=3.3.0,<4.0.0
passlib[bcrypt]>=1.7.4,<2.0.0

# Monitoring
prometheus-client>=0.17.0,<1.0.0
opentelemetry-api>=1.18.0,<2.0.0
opentelemetry-sdk>=1.18.0,<2.0.0

# Caching
redis>=4.5.0,<5.0.0

# Deployment
gunicorn>=20.1.0,<21.0.0
```

### Dependency Rules

1. **Pin Major Versions:** `numpy>=1.24.0,<2.0.0`
2. **Separate Concerns:** Different files for different use cases
3. **Document Dependencies:** Explain why each dependency is needed
4. **Regular Updates:** Review and update monthly
5. **Security Scanning:** Run `pip-audit` in CI/CD

---

## ✅ Code Quality Checklist

### Module Quality

- [ ] **Single Responsibility:** One clear purpose
- [ ] **Type Hints:** 100% coverage on public APIs
- [ ] **Docstrings:** Google-style for all public classes/functions
- [ ] **Error Handling:** Appropriate exceptions with context
- [ ] **Logging:** Structured logging at appropriate levels
- [ ] **Tests:** Unit tests with >90% coverage
- [ ] **Examples:** Usage examples in docstrings

### Import Quality

- [ ] **Explicit Imports:** No wildcards
- [ ] **No Circular Dependencies:** Verified with import checks
- [ ] **Consistent Style:** Same pattern across all files
- [ ] **Public API:** `__all__` defined for all modules

### Test Quality

- [ ] **Structure Mirrors src/:** Easy to find tests
- [ ] **Fixtures:** Reusable fixtures in conftest.py
- [ ] **Markers:** Appropriate use of pytest markers
- [ ] **Coverage:** >90% for core modules

---

**Document Status:** ✅ **COMPLETE - Code Organization Standard**

**Next Document:** [STUDENT_EXPERIENCE_DESIGN.md](./STUDENT_EXPERIENCE_DESIGN.md)

---

*Document Version: 3.0 | Last Updated: March 30, 2026 | AI-Mastery-2026*
