# Developer Documentation

<div align="center">

![Contributors](https://img.shields.io/badge/contributors-50+-orange.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)
![Last Updated](https://img.shields.io/badge/updated-March%2028%2C%202026-blue.svg)

**Guidelines and resources for contributing to AI-Mastery-2026**

[Architecture](#-architecture) • [Code Style](#-code-style) • [Testing](#-testing) • [Contributing](#-contributing)

</div>

---

## 🏗️ Architecture Overview

### Project Structure

```
AI-Mastery-2026/
├── src/                          # Source code
│   ├── core/                     # Pure Python implementations
│   │   ├── linear_algebra.py     # Matrix operations
│   │   ├── calculus.py           # Derivatives, integration
│   │   └── optimization.py       # Gradient descent variants
│   ├── ml/                       # Machine learning
│   │   ├── classical.py          # Decision trees, SVM, etc.
│   │   ├── deep_learning.py      # Neural networks
│   │   └── vision.py             # CNN, ResNet
│   ├── llm/                      # LLM engineering
│   │   ├── transformer.py        # Transformer architecture
│   │   ├── attention.py          # Attention mechanisms
│   │   └── finetuning.py         # LoRA, QLoRA
│   ├── rag_specialized/          # RAG implementations
│   │   ├── adaptive_multimodal/  # Multi-modal RAG
│   │   ├── temporal_aware/       # Time-aware RAG
│   │   └── advanced_rag.py       # Advanced patterns
│   ├── agents/                   # AI agents
│   │   ├── agent_base.py         # Base agent class
│   │   ├── tools.py              # Tool implementations
│   │   └── orchestration.py      # Multi-agent coordination
│   └── production/               # Production code
│       ├── api.py                # FastAPI endpoints
│       ├── monitoring.py         # Metrics, logging
│       └── security.py           # Auth, PII detection
├── tests/                        # Test suite
│   ├── test_core.py
│   ├── test_ml.py
│   ├── test_llm.py
│   └── test_rag.py
├── notebooks/                    # Jupyter notebooks
│   ├── 01_mathematical_foundations/
│   ├── 02_classical_ml/
│   ├── 03_deep_learning/
│   ├── 04_llm/
│   └── RAG/
├── docs/                         # Documentation
│   ├── guides/                   # User guides
│   ├── api/                      # API reference
│   ├── tutorials/                # Tutorials
│   ├── kb/                       # Knowledge base
│   ├── faq/                      # FAQ
│   └── reference/                # Technical reference
├── scripts/                      # Utility scripts
│   ├── setup.py
│   ├── diagnose.py
│   └── optimize-docs.py
└── config/                       # Configuration files
    ├── logging.yaml
    └── model_config.yaml
```

### Design Principles

1. **White-Box First:** Implement from scratch before using libraries
2. **Mathematical Rigor:** Derive and document formulas
3. **Production Ready:** All code must be deployable
4. **Test Coverage:** Minimum 90% coverage
5. **Documentation:** Code is incomplete without docs

---

## 📝 Code Style Guide

### Python Style

Follow [PEP 8](https://pep8.org/) with these project-specific conventions:

#### Naming Conventions

```python
# Classes: PascalCase
class DecisionTreeClassifier:
    pass

# Functions: snake_case
def calculate_gradient_descent():
    pass

# Constants: UPPER_SNAKE_CASE
MAX_ITERATIONS = 1000
LEARNING_RATE = 0.01

# Private methods: _prefix
def _internal_helper():
    pass

# Module-level private: __prefix
__all__ = ["public_api"]
```

#### Type Hints

Always use type hints:

```python
from typing import List, Dict, Optional, Union, Tuple
import numpy as np

def process_data(
    data: List[Dict[str, float]],
    normalize: bool = True,
    threshold: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Process input data.
    
    Args:
        data: List of feature dictionaries
        normalize: Whether to normalize features
        threshold: Optional threshold for filtering
    
    Returns:
        Tuple of processed array and metadata
    """
    pass
```

#### Docstrings

Use Google-style docstrings:

```python
class RAGSystem:
    """Retrieval-Augmented Generation system.
    
    Combines document retrieval with text generation to provide
    accurate, context-aware responses.
    
    Attributes:
        retriever: Document retrieval component
        generator: Text generation component
        embedding_model: Model for creating embeddings
    
    Example:
        >>> rag = RAGSystem()
        >>> rag.add_documents(docs)
        >>> result = rag.query("What is AI?")
        >>> print(result.answer)
    """
    
    def query(self, query: str, top_k: int = 5) -> RAGResult:
        """Query the RAG system.
        
        Args:
            query: User's question or search query
            top_k: Number of documents to retrieve
        
        Returns:
            RAGResult containing answer and source documents
        
        Raises:
            ValueError: If query is empty
            RuntimeError: If retrieval fails
        
        Example:
            >>> result = rag.query("What is machine learning?")
            >>> print(result.answer)
        """
        pass
```

#### Error Handling

```python
# Use specific exceptions
def load_model(path: str) -> Model:
    """Load model from file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    try:
        model = torch.load(path)
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}") from e
    
    return model

# Use context managers for resources
def process_file(path: str):
    """Process file with proper resource management."""
    with open(path, 'r') as f:
        data = f.read()
    # File automatically closed
```

### Code Organization

#### Module Structure

```python
"""
Module docstring describing purpose.

Example:
    >>> from src.ml import DecisionTreeClassifier
    >>> model = DecisionTreeClassifier()
"""

# Standard library imports
import os
import sys
from typing import List, Dict

# Third-party imports
import numpy as np
import torch

# Local imports
from src.core.linear_algebra import matrix_multiply
from src.utils.logging import get_logger

# Constants
DEFAULT_MAX_DEPTH = 10
MIN_SAMPLES_SPLIT = 2

# Classes
class DecisionTreeClassifier:
    """Decision tree implementation."""
    pass

# Functions
def train_model(data: np.ndarray) -> DecisionTreeClassifier:
    """Train a decision tree model."""
    pass

# Main block for testing
if __name__ == "__main__":
    # Test code
    pass
```

---

## 🧪 Testing Guide

### Test Structure

```python
"""Tests for RAG system."""

import pytest
import numpy as np
from src.rag import RAGSystem, RAGResult

class TestRAGSystem:
    """Test cases for RAGSystem."""
    
    @pytest.fixture
    def rag_system(self):
        """Create RAG system for testing."""
        return RAGSystem(
            embedding_model="test-model",
            vector_db="memory"
        )
    
    def test_add_documents(self, rag_system):
        """Test adding documents to RAG system."""
        docs = [
            {"id": "1", "content": "Test content"},
            {"id": "2", "content": "More content"}
        ]
        
        rag_system.add_documents(docs)
        
        # Verify documents were added
        assert len(rag_system.vector_db) == 2
    
    def test_query_returns_result(self, rag_system):
        """Test querying returns valid result."""
        # Setup
        rag_system.add_documents([
            {"id": "1", "content": "Machine learning is AI subset"}
        ])
        
        # Execute
        result = rag_system.query("What is ML?")
        
        # Verify
        assert isinstance(result, RAGResult)
        assert result.answer is not None
        assert len(result.sources) > 0
    
    def test_query_empty_raises_error(self, rag_system):
        """Test that empty query raises error."""
        with pytest.raises(ValueError, match="Query cannot be empty"):
            rag_system.query("")
    
    @pytest.mark.parametrize("query,expected_keywords", [
        ("machine learning", ["AI", "learning"]),
        ("deep learning", ["neural", "network"]),
    ])
    def test_query_relevance(self, rag_system, query, expected_keywords):
        """Test query returns relevant results."""
        # Setup with known documents
        rag_system.add_documents([
            {"id": "1", "content": "Machine learning is a subset of AI"},
            {"id": "2", "content": "Deep learning uses neural networks"}
        ])
        
        # Execute
        result = rag_system.query(query)
        
        # Verify relevance
        assert any(kw in result.answer for kw in expected_keywords)
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_rag.py -v

# Run specific test class
pytest tests/test_rag.py::TestRAGSystem -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run fast tests only (no slow integration tests)
pytest tests/ -v -m "not slow"

# Run tests matching pattern
pytest tests/ -k "test_query" -v
```

### Test Categories

```python
# Unit tests (fast, isolated)
@pytest.mark.unit
def test_matrix_multiplication():
    pass

# Integration tests (slower, multiple components)
@pytest.mark.integration
def test_rag_pipeline():
    pass

# Slow tests (very slow, external services)
@pytest.mark.slow
def test_large_model_training():
    pass

# Skip in CI
@pytest.mark.skipif(os.getenv("CI"), reason="Slow test")
def test_very_slow_operation():
    pass
```

---

## 🤝 Contributing Guidelines

### How to Contribute

#### 1. Fork and Clone

```bash
# Fork on GitHub, then clone
git clone https://github.com/YOUR_USERNAME/AI-Mastery-2026.git
cd AI-Mastery-2026

# Add upstream remote
git remote add upstream https://github.com/Kandil7/AI-Mastery-2026.git
```

#### 2. Create Branch

```bash
# Sync with upstream
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/my-feature
```

#### 3. Make Changes

```bash
# Make your changes
# Follow code style guide
# Add tests
# Update documentation
```

#### 4. Run Checks

```bash
# Run tests
pytest tests/ -v

# Run linter
ruff check src/

# Run type checker
mypy src/

# Check formatting
black --check src/

# Run all checks
make all
```

#### 5. Commit Changes

```bash
# Stage changes
git add src/my_feature.py
git add tests/test_my_feature.py

# Commit with conventional commits
git commit -m "feat: add new RAG retrieval method

- Implement hybrid search combining dense and sparse retrieval
- Add re-ranking with cross-encoder
- Include comprehensive tests

Closes #123"
```

#### 6. Push and Create PR

```bash
# Push to your fork
git push origin feature/my-feature

# Create PR on GitHub
# Fill out PR template
# Wait for review
```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting)
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(rag): add hybrid search implementation

fix(llm): resolve memory leak in transformer

docs: update API documentation

test(rag): add integration tests for retrieval
```

### Pull Request Template

```markdown
## Description

Brief description of changes

## Type of Change

- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update

## Testing

- [ ] I have added tests that prove my fix/feature works
- [ ] All tests pass locally
- [ ] I have updated documentation accordingly

## Checklist

- [ ] My code follows the style guidelines
- [ ] I have performed a self-review
- [ ] I have commented my code
- [ ] I have updated the changelog

## Related Issues

Closes #123
```

---

## 📊 Performance Guidelines

### Optimization Tips

#### Memory Efficiency

```python
# ❌ Bad: Loads entire file
def load_large_file(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()  # Loads all into memory
    return lines

# ✅ Good: Generator
def load_large_file(path: str):
    with open(path, 'r') as f:
        for line in f:
            yield line  # Streams line by line

# Use numpy efficiently
# ❌ Bad
result = []
for i in range(len(arr)):
    result.append(arr[i] * 2)
result = np.array(result)

# ✅ Good
result = arr * 2  # Vectorized operation
```

#### Computation Efficiency

```python
# ❌ Bad: Python loops
def sum_squares(numbers: List[float]) -> float:
    total = 0.0
    for n in numbers:
        total += n * n
    return total

# ✅ Good: NumPy
def sum_squares(numbers: np.ndarray) -> float:
    return np.sum(numbers ** 2)

# ❌ Bad: Repeated computation
for i in range(1000):
    result = expensive_function(x) * i

# ✅ Good: Cache computation
cached = expensive_function(x)
for i in range(1000):
    result = cached * i
```

#### Batch Processing

```python
# ❌ Bad: One at a time
for doc in documents:
    embedding = model.encode(doc)
    vector_db.add(embedding)

# ✅ Good: Batch processing
embeddings = model.encode(documents, batch_size=32)
vector_db.add_batch(embeddings)
```

---

## 📚 Documentation Guidelines

### Writing Documentation

#### Structure

```markdown
# Feature Name

## Overview

Brief description of what this feature does.

## Installation

```bash
pip install package
```

## Quick Start

```python
from package import Feature

feature = Feature()
result = feature.run()
```

## API Reference

### Class Name

Description of the class.

#### Parameters

- `param1` (type): Description
- `param2` (type): Description

#### Methods

##### method_name

Description of method.

**Example:**
```python
obj.method_name()
```

## Advanced Usage

### Use Case 1

Explanation and example.

### Use Case 2

Explanation and example.

## Troubleshooting

### Common Issues

**Problem:** Error message

**Solution:** Steps to fix

## Related Resources

- [Link 1](url)
- [Link 2](url)
```

### Documentation Checklist

- [ ] Clear title and overview
- [ ] Installation instructions
- [ ] Quick start example
- [ ] API reference with parameters
- [ ] Advanced usage examples
- [ ] Troubleshooting section
- [ ] Related resources
- [ ] Code examples are tested
- [ ] Links are valid

---

## 🔧 Development Tools

### Required Tools

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Or use make
make dev-install
```

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run manually
pre-commit run --all-files
```

**Pre-commit checks:**
- Black (formatting)
- Ruff (linting)
- MyPy (type checking)
- Secret detection

### IDE Configuration

#### VS Code Settings

```json
{
    "python.defaultInterpreterPath": ".venv/Scripts/python.exe",
    "python.linting.enabled": true,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter",
        "editor.tabSize": 4
    }
}
```

---

## 📞 Getting Help

### Resources

- [Architecture Docs](architecture.md) - System design
- [Code Style](#code-style-guide) - Coding standards
- [Testing Guide](#testing) - Testing best practices
- [FAQ](../faq/) - Common questions

### Communication

- **GitHub Issues:** Bug reports and feature requests
- **GitHub Discussions:** Questions and ideas
- **Discord:** Real-time chat with contributors

---

**Last Updated:** March 28, 2026  
**Contributors:** 50+  
**Open Issues:** See [GitHub Issues](https://github.com/Kandil7/AI-Mastery-2026/issues)
