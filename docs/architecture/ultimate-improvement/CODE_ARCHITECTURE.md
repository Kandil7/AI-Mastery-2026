# 💻 CODE ARCHITECTURE

**AI-Mastery-2026: Code Organization & Best Practices**

| Document Info | Details |
|---------------|---------|
| **Version** | 1.0 |
| **Date** | March 31, 2026 |
| **Status** | Code Architecture Specification |

---

## 📋 EXECUTIVE SUMMARY

### Code Organization Philosophy

AI-Mastery-2026 follows **Domain-Driven Design (DDD)** principles with a focus on:

- ✅ **Educational Clarity**: Code teaches through readability and structure
- ✅ **Production Quality**: All code meets industry standards
- ✅ **From-Scratch First**: Implement algorithms manually before using libraries
- ✅ **Progressive Complexity**: Simple → optimized → production

### Code Structure Overview

```
src/
├── core/           # From-scratch implementations (teaching focus)
├── ml/             # Machine learning algorithms
├── llm/            # LLM architecture and training
├── rag/            # RAG system components
├── agents/         # AI agent systems
├── production/     # Production infrastructure
├── utils/          # Shared utilities
└── data/           # Data pipelines
```

---

## 🏗️ DOMAIN-DRIVEN DESIGN APPLICATION

### Domain Boundaries

```
┌─────────────────────────────────────────────────────────────┐
│                      DOMAIN BOUNDARIES                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │    CORE      │    │     ML       │    │    LLM       │  │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤  │
│  │ Math ops     │    │ Classical    │    │ Architecture │  │
│  │ Probability  │    │ Deep Learning│    │ Training     │  │
│  │ Optimization │    │ Vision       │    │ Alignment    │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │
│  │    RAG       │    │   AGENTS     │    │  PRODUCTION  │  │
│  ├──────────────┤    ├──────────────┤    ├──────────────┤  │
│  │ Chunking     │    │ Core         │    │ API          │  │
│  │ Embeddings   │    │ Memory       │    │ Monitoring   │  │
│  │ Retrieval    │    │ Tools        │    │ Deployment   │  │
│  │ Vector Store │    │ Multi-agent  │    │ Security     │  │
│  └──────────────┘    └──────────────┘    └──────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Dependency Rules

**Rule 1: Lower-level domains don't depend on higher-level domains**

```python
# ✅ ALLOWED: core → ml (lower → higher)
from src.core.math.matrices import Matrix

class LinearRegression:
    def __init__(self):
        self.weights = Matrix.zeros(10, 1)

# ❌ FORBIDDEN: ml → core (creates circular dependency)
# from src.ml.classical import LinearRegression  # in core module
```

**Rule 2: Sibling domains communicate through interfaces**

```python
# ✅ ALLOWED: Use abstract base classes
from src.rag.embeddings.base import BaseEmbedding

class RAGPipeline:
    def __init__(self, embedding_model: BaseEmbedding):
        self.embedding_model = embedding_model

# ❌ FORBIDDEN: Direct concrete class dependency
# from src.rag.embeddings.sentence_transformers import SentenceTransformerEmbedding
```

---

## 📝 CODING STANDARDS

### Python Style Guide

**Base Standard**: PEP 8 + Google Python Style Guide

**Key Requirements**:

```python
# ✅ DO: Type hints on all public functions
def calculate_attention(
    query: np.ndarray,
    key: np.ndarray,
    value: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scale: float = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate scaled dot-product attention.
    
    Args:
        query: Query tensor of shape (batch, seq_len, d_k)
        key: Key tensor of shape (batch, seq_len, d_k)
        value: Value tensor of shape (batch, seq_len, d_v)
        mask: Optional mask tensor
        scale: Scaling factor (default: 1/sqrt(d_k))
    
    Returns:
        Tuple of (attention_output, attention_weights)
    """
    ...

# ✅ DO: Use dataclasses for data containers
@dataclass
class TrainingConfig:
    """Configuration for model training."""
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    weight_decay: float = 0.01
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.learning_rate <= 0:
            raise ValueError("Learning rate must be positive")

# ✅ DO: Use properties for computed attributes
class ModelMetrics:
    def __init__(self, predictions: np.ndarray, targets: np.ndarray):
        self.predictions = predictions
        self.targets = targets
    
    @property
    def accuracy(self) -> float:
        """Calculate accuracy."""
        return np.mean(self.predictions == self.targets)
    
    @property
    def mse(self) -> float:
        """Calculate mean squared error."""
        return np.mean((self.predictions - self.targets) ** 2)

# ❌ DON'T: Use bare except clauses
try:
    result = risky_operation()
except:  # BAD: Catches everything including KeyboardInterrupt
    pass

# ✅ DO: Catch specific exceptions
try:
    result = risky_operation()
except ValueError as e:
    logger.warning(f"Invalid value: {e}")
except TimeoutError as e:
    logger.error(f"Operation timed out: {e}")
```

### File Organization

```python
"""
Module docstring describing the module's purpose.
"""

# 1. Standard library imports
import math
import os
from typing import List, Optional, Tuple

# 2. Third-party imports
import numpy as np
import torch
import torch.nn as nn

# 3. Local application imports
from src.core.math.vectors import Vector
from src.utils.logging import get_logger

# 4. Module constants
DEFAULT_LEARNING_RATE = 1e-4
MAX_ITERATIONS = 1000

# 5. Module-level variables (if needed)
logger = get_logger(__name__)

# 6. Classes and functions
class MyClass:
    """Class docstring."""
    pass


def my_function():
    """Function docstring."""
    pass
```

---

## 🧪 TESTING STANDARDS

### Test Organization

```
tests/
├── unit/                 # Unit tests (test individual components)
│   ├── core/
│   │   ├── test_vectors.py
│   │   └── test_matrices.py
│   ├── ml/
│   │   ├── test_classical.py
│   │   └── test_deep_learning.py
│   └── llm/
│       ├── test_attention.py
│       └── test_transformer.py
├── integration/          # Integration tests (test component interactions)
│   ├── test_rag_pipeline.py
│   └── test_agent_system.py
├── e2e/                  # End-to-end tests (test complete workflows)
│   ├── test_learning_path.py
│   └── test_deployment.py
└── fixtures/             # Test fixtures and mock data
    ├── sample_data/
    └── mock_models/
```

### Test Writing Standards

```python
"""
Test module for attention mechanisms.
"""

import numpy as np
import pytest

from src.llm.architecture.attention import (
    scaled_dot_product_attention,
    multi_head_attention,
    MultiHeadAttention
)


class TestScaledDotProductAttention:
    """Tests for scaled dot-product attention."""
    
    def test_output_shape(self):
        """Test that output has correct shape."""
        # Arrange
        batch_size = 2
        seq_len = 10
        d_k = 64
        d_v = 64
        
        query = np.random.randn(batch_size, seq_len, d_k)
        key = np.random.randn(batch_size, seq_len, d_k)
        value = np.random.randn(batch_size, seq_len, d_v)
        
        # Act
        output, weights = scaled_dot_product_attention(query, key, value)
        
        # Assert
        assert output.shape == (batch_size, seq_len, d_v)
        assert weights.shape == (batch_size, seq_len, seq_len)
    
    def test_attention_weights_sum_to_one(self):
        """Test that attention weights form valid probability distribution."""
        # Arrange
        query = np.random.randn(1, 5, 32)
        key = np.random.randn(1, 5, 32)
        value = np.random.randn(1, 5, 64)
        
        # Act
        _, weights = scaled_dot_product_attention(query, key, value)
        
        # Assert
        weights_sum = np.sum(weights, axis=-1)
        assert np.allclose(weights_sum, 1.0, atol=1e-6)
    
    def test_masked_attention(self):
        """Test that masking prevents attending to future positions."""
        # Arrange
        batch_size = 1
        seq_len = 5
        d_k = 32
        d_v = 64
        
        query = np.random.randn(batch_size, seq_len, d_k)
        key = np.random.randn(batch_size, seq_len, d_k)
        value = np.random.randn(batch_size, seq_len, d_v)
        
        # Create causal mask (upper triangular)
        mask = np.triu(np.ones((seq_len, seq_len)), k=1) * -1e9
        
        # Act
        output, weights = scaled_dot_product_attention(query, key, value, mask=mask)
        
        # Assert
        # Masked positions should have near-zero attention
        assert np.all(weights[0, 0, 1:] < 1e-6)  # First position can't attend to future


class TestMultiHeadAttention:
    """Tests for multi-head attention."""
    
    def test_multi_head_output(self):
        """Test multi-head attention output shape."""
        # Arrange
        batch_size = 2
        seq_len = 10
        d_model = 512
        num_heads = 8
        
        x = np.random.randn(batch_size, seq_len, d_model)
        
        # Act
        output, weights = multi_head_attention(
            x, x, x,
            num_heads=num_heads
        )
        
        # Assert
        assert output.shape == (batch_size, seq_len, d_model)
    
    def test_gradient_flow(self):
        """Test that gradients flow through attention."""
        # Arrange
        attention = MultiHeadAttention(d_model=128, num_heads=4)
        x = torch.randn(2, 10, 128, requires_grad=True)
        
        # Act
        output = attention(x, x, x)
        loss = output.sum()
        loss.backward()
        
        # Assert
        assert x.grad is not None
        assert x.grad.shape == x.shape
        assert not torch.isnan(x.grad).any()


@pytest.fixture
def sample_attention_input():
    """Fixture providing sample attention input."""
    return {
        'query': np.random.randn(2, 10, 64),
        'key': np.random.randn(2, 10, 64),
        'value': np.random.randn(2, 10, 128)
    }


def test_with_fixture(sample_attention_input):
    """Test using fixture."""
    output, _ = scaled_dot_product_attention(
        sample_attention_input['query'],
        sample_attention_input['key'],
        sample_attention_input['value']
    )
    assert output.shape == (2, 10, 128)
```

### Test Coverage Requirements

| Component Type | Minimum Coverage | Critical Coverage |
|----------------|------------------|-------------------|
| **Core algorithms** | 95% | 100% (math must be correct) |
| **ML models** | 90% | 95% (forward + backward pass) |
| **Utilities** | 85% | 90% (error handling) |
| **API endpoints** | 90% | 95% (all routes) |
| **Integration tests** | N/A | All critical paths |

---

## 📦 IMPORT HIERARCHY

### Import Rules

**Rule 1: Public API through `__init__.py`**

```python
# src/llm/architecture/__init__.py
"""LLM Architecture Module - Public API."""

from .attention import (
    scaled_dot_product_attention,
    MultiHeadAttention,
)
from .transformer import (
    TransformerEncoder,
    TransformerDecoder,
    Transformer,
)
from .tokenization import (
    Tokenizer,
    BPETokenizer,
)

__all__ = [
    # Attention
    'scaled_dot_product_attention',
    'MultiHeadAttention',
    # Transformer
    'TransformerEncoder',
    'TransformerDecoder',
    'Transformer',
    # Tokenization
    'Tokenizer',
    'BPETokenizer',
]
```

**Rule 2: Use relative imports within packages**

```python
# ✅ DO: Relative imports for sibling modules
from .attention import MultiHeadAttention
from ..utils.normalization import LayerNorm

# ❌ DON'T: Absolute imports for internal modules
from src.llm.architecture.attention import MultiHeadAttention
```

**Rule 3: Avoid circular imports**

```python
# ❌ DON'T: Circular import
# file_a.py
from file_b import function_b

def function_a():
    return function_b()

# file_b.py
from file_a import function_a  # CIRCULAR!

# ✅ DO: Restructure to avoid circular dependency
# Move shared functionality to separate module
# Use dependency injection
# Import inside function (last resort)
```

---

## 🔧 VERSIONING STRATEGY

### Semantic Versioning

```
MAJOR.MINOR.PATCH

Examples:
1.0.0  - Initial release
1.1.0  - New features (backward compatible)
1.1.1  - Bug fixes
2.0.0  - Breaking changes
```

### Version Numbering Rules

| Change Type | Version Bump | Examples |
|-------------|--------------|----------|
| **Breaking API change** | MAJOR | Removing public function, changing signature |
| **New feature** | MINOR | Adding new module, new public class |
| **Bug fix** | PATCH | Fixing incorrect calculation, performance fix |
| **Documentation** | PATCH | Fixing typos, adding examples |
| **Internal refactoring** | PATCH | No API changes |

### Deprecation Policy

```python
import warnings
from typing import Optional


def old_function(x: np.ndarray) -> np.ndarray:
    """
    Deprecated: Use `new_function` instead.
    
    This function will be removed in version 3.0.0.
    """
    warnings.warn(
        "old_function is deprecated and will be removed in version 3.0.0. "
        "Use new_function instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return _old_implementation(x)


def new_function(x: np.ndarray) -> np.ndarray:
    """New implementation with better performance."""
    return _new_implementation(x)
```

**Deprecation Timeline**:
- **Minor version**: Deprecation warning added
- **Next major version**: Function removed (minimum 6 months notice)

---

## 📊 CODE QUALITY METRICS

### Quality Gates

| Metric | Target | Enforcement |
|--------|--------|-------------|
| **Test coverage** | 90%+ | CI/CD blocking |
| **Type hint coverage** | 95%+ | mypy strict mode |
| **Docstring coverage** | 90%+ | CI/CD warning |
| **Linting errors** | 0 | CI/CD blocking |
| **Code duplication** | <5% | CI/CD warning |
| **Cyclomatic complexity** | <15 avg | Code review |
| **Function length** | <50 lines | Code review |
| **File length** | <500 lines | Code review |

### CI/CD Quality Checks

```yaml
# .github/workflows/quality-gates.yml
name: Quality Gates

on: [push, pull_request]

jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      
      - name: Lint (flake8)
        run: flake8 src/ --max-line-length=100
      
      - name: Type check (mypy)
        run: mypy src/ --strict
      
      - name: Test (pytest)
        run: pytest tests/ --cov=src/ --cov-report=xml --cov-fail-under=90
      
      - name: Code duplication (pylint)
        run: pylint src/ --disable=all --enable=duplicate-code
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

---

## 📝 DOCUMENT HISTORY

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | March 31, 2026 | AI Engineering Tech Lead | Initial code architecture |

---

## 🔗 RELATED DOCUMENTS

This document is part of the **Ultimate Repository Improvement** series:

1. ✅ [ULTIMATE_REPOSITORY_VISION.md](./ULTIMATE_REPOSITORY_VISION.md)
2. ✅ [DEFINITIVE_DIRECTORY_STRUCTURE.md](./DEFINITIVE_DIRECTORY_STRUCTURE.md)
3. ✅ [CURRICULUM_ARCHITECTURE.md](./CURRICULUM_ARCHITECTURE.md)
4. ✅ **CODE_ARCHITECTURE.md** (this document)
5. 📋 [DOCUMENTATION_ARCHITECTURE.md](./DOCUMENTATION_ARCHITECTURE.md)
6. 🎓 [STUDENT_JOURNEY_DESIGN.md](./STUDENT_JOURNEY_DESIGN.md)
7. 👥 [CONTRIBUTOR_ECOSYSTEM.md](./CONTRIBUTOR_ECOSYSTEM.md)
8. 🏢 [INDUSTRY_INTEGRATION_HUB.md](./INDUSTRY_INTEGRATION_HUB.md)
9. ⚡ [SCALABILITY_AND_PERFORMANCE.md](./SCALABILITY_AND_PERFORMANCE.md)
10. 🔄 [MIGRATION_MASTERPLAN.md](./MIGRATION_MASTERPLAN.md)
11. 📖 [QUICK_REFERENCE_COMPENDIUM.md](./QUICK_REFERENCE_COMPENDIUM.md)
12. 📅 [IMPLEMENTATION_ROADMAP_2026.md](./IMPLEMENTATION_ROADMAP_2026.md)

---

<div align="center">

**💻 Code architecture defined. Next: Documentation system.**

[Next: DOCUMENTATION_ARCHITECTURE.md](./DOCUMENTATION_ARCHITECTURE.md)

</div>
