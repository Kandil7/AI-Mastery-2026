# Migration Guide

**Version:** 0.1.0  
**Date:** March 31, 2026

This guide helps you migrate your code to the restructured AI-Mastery-2026 codebase (Phases 1-3).

---

## 📋 Overview

The repository underwent significant restructuring to improve maintainability and clarity. This guide covers:

1. [Module Renames](#module-renames)
2. [Import Path Changes](#import-path-changes)
3. [New Modules](#new-modules)
4. [Deprecated Patterns](#deprecated-patterns)
5. [Quick Reference](#quick-reference)

---

## 🔀 Module Renames

### Classical ML Module

| Old Path | New Path | Reason |
|----------|----------|--------|
| `src.ml.classical_base` | `src.ml.classical_scratch` | More descriptive name |

**Update your imports:**
```python
# ❌ Old
from src.ml.classical_base import LinearRegressionScratch

# ✅ New
from src.ml.classical_scratch import LinearRegressionScratch
```

### Deep Learning Module

| Old Path | New Path | Reason |
|----------|----------|--------|
| `src.ml.deep_learning_base` | `src.ml.neural_networks_scratch` | Avoids conflict with `deep_learning/` directory |

**Update your imports:**
```python
# ❌ Old
from src.ml.deep_learning_base import NeuralNetwork, Layer

# ✅ New
from src.ml.neural_networks_scratch import NeuralNetwork, Layer
```

---

## 📦 Import Path Changes

### Core Module

The `src.core` module now has comprehensive `__all__` exports. Direct imports still work:

```python
# These still work ✅
from src.core.optimization import Adam
from src.core.time_series import ExtendedKalmanFilter
from src.core.mcmc import metropolis_hastings

# Now also available via main module ✅
from src.core import Adam, ExtendedKalmanFilter, metropolis_hastings
```

### ML Module

```python
# ❌ Old (still works but deprecated)
from src.ml.deep_learning import NeuralNetwork

# ✅ New
from src.ml import NeuralNetwork
from src.ml.neural_networks_scratch import NeuralNetwork
```

### LLM Module

```python
# ✅ Now available (was missing before)
from src.llm import MultiHeadAttention, BERT, GPT2, FineTuner, LLMEvaluator
```

### RAG Module

```python
# ✅ Now available (placeholder implementations)
from src.rag import Document, RAGPipeline, DocumentChunk
from src.rag import BaseChunker, SemanticChunker
```

### Agents Module

```python
# ✅ Simplified imports
from src.agents import SupportAgent, MultiAgent, Agent, ReActAgent, Chain
```

---

## 🆕 New Modules

### Configuration Module (`src.config`)

Centralized configuration management:

```python
from src.config import (
    Settings, get_settings,
    ModelConfig, TransformerConfig, TrainingConfig,
    RAGConfig, DataConfig
)

# Get global settings
settings = get_settings()
print(f"Environment: {settings.environment}")

# Create model config
config = TransformerConfig(hidden_dim=768, num_heads=12)
```

### Types Module (`src.types`)

Shared type definitions:

```python
from src.types import (
    DocumentProtocol,
    EmbeddingVector,
    ModelOutput,
    RAGOutput,
    Trainable,
    Saveable,
)

def process(doc: DocumentProtocol) -> EmbeddingVector:
    ...
```

---

## ⚠️ Deprecated Patterns

### Wildcard Imports

```python
# ❌ Deprecated
from src.ml import *

# ✅ Use explicit imports
from src.ml import NeuralNetwork, DecisionTree
```

### Missing `__all__` Exports

All modules now have `__all__` declarations. Relying on implicit exports may break in future versions.

### Legacy RAG Imports

```python
# ❌ Removed (archived)
from src.rag.legacy_rag import RAGPipeline

# ✅ Use new implementation
from src.rag import RAGPipeline
```

---

## 📖 Quick Reference

### Import Changes Summary

| What You Had | What to Use Now |
|--------------|-----------------|
| `from src.ml.classical_base import ...` | `from src.ml.classical_scratch import ...` |
| `from src.ml.deep_learning_base import ...` | `from src.ml.neural_networks_scratch import ...` |
| `from src.rag.legacy_* import ...` | `from src.rag import ...` |
| N/A | `from src.config import ...` |
| N/A | `from src.types import ...` |

### New Available Exports

**src.core:**
```python
from src.core import (
    # Optimization
    Adam, GradientDescent, Momentum, RMSprop,
    # Time Series
    ExtendedKalmanFilter, UnscentedKalmanFilter, ParticleFilter,
    # MCMC
    metropolis_hastings, HamiltonianMonteCarlo, nuts_sampler,
    # And 80+ more exports
)
```

**src.ml:**
```python
from src.ml import (
    # Classical
    LinearRegression, LogisticRegression, DecisionTree, RandomForest, SVM,
    # Deep Learning
    NeuralNetwork, Dense, Conv2D, LSTM,
    # Vision
    ResNet18, ResidualBlock,
    # GNN
    GraphSAGELayer, GNNRecommender,
)
```

**src.llm:**
```python
from src.llm import (
    MultiHeadAttention, BERT, GPT2,
    FineTuner, LLMEvaluator,
    # And 50+ more exports
)
```

---

## 🔧 Migration Script

For automated migration, use the provided script:

```bash
# Dry run (show changes)
python scripts/migrate_imports.py --dry-run

# Apply changes
python scripts/migrate_imports.py
```

---

## ✅ Verification

After migration, verify your code works:

```bash
# Run import tests
python -c "from src.core import Adam; from src.ml import NeuralNetwork; print('OK')"

# Run your test suite
pytest tests/ -v
```

---

## 📞 Need Help?

- **Issues:** [GitHub Issues](https://github.com/Kandil7/AI-Mastery-2026/issues)
- **Discussions:** [GitHub Discussions](https://github.com/Kandil7/AI-Mastery-2026/discussions)
- **Documentation:** [docs/README.md](docs/README.md)

---

**Last Updated:** March 31, 2026  
**Migration Deadline:** April 30, 2026 (legacy imports will be removed in v0.2.0)
