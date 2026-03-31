# ADR-003: Type Definitions and Shared Types

**Status:** Accepted
**Date:** 2026-03-31
**Authors:** Kandil7

---

## Context

The AI-Mastery-2026 codebase uses many common types:
- Document and chunk types
- Embedding vectors and matrices
- Similarity scores and distances
- Model outputs and predictions
- Protocol definitions for components

These types are currently:
1. Scattered across multiple modules
2. Inconsistently named
3. Often redefined in each module
4. Missing from type hints in many places

This leads to:
- Type incompatibilities between modules
- Confusion about which type to use
- Duplicate type definitions
- Harder refactoring

## Decision

We propose a centralized type definitions module:

### New `src/types/` Module

```
src/types/
└── __init__.py    # All shared type definitions
```

### Type Categories

1. **Re-exported Types** from `src.core.utils.types`:
   - Basic types (TextLike, PathLike, IDType)
   - Protocols (DocumentProtocol, ChunkProtocol)
   - Generic types (Result, PaginatedResult)
   - Pattern types (Strategy, Repository)

2. **NumPy Array Types**:
   - `EmbeddingVector` - 1D array of floats
   - `EmbeddingMatrix` - 2D array of floats
   - `AttentionMatrix` - 2D attention weights
   - `Logits` - Model output logits

3. **Tensor Types** (conditional on PyTorch):
   - `Tensor` - torch.Tensor or np.ndarray fallback
   - `OptionalTensor` - Optional tensor type

4. **Model Output Types**:
   - `ModelOutput` - Base class for outputs
   - `TransformerOutput` - Transformer-specific output
   - `RAGOutput` - RAG system output

5. **Evaluation Types**:
   - `MetricResult` - Single metric result
   - `EvaluationResult` - Full evaluation results

6. **Protocol Definitions**:
   - `Embeddable` - Objects that can be embedded
   - `Trainable` - Trainable models
   - `Saveable` - Saveable/loadable objects

7. **Type Aliases**:
   - `DocumentId`, `ChunkId`
   - `QueryText`, `QueryEmbedding`
   - `RetrievalScore`, `RerankingScore`
   - `LearningRate`, `LossValue`, `Gradient`

### Usage Pattern

```python
from src.types import (
    DocumentProtocol,
    EmbeddingVector,
    SimilarityScore,
    Trainable,
    ModelOutput,
    RAGOutput,
)

def embed_document(doc: DocumentProtocol) -> EmbeddingVector:
    ...

def train_model(model: Trainable, X: np.ndarray, y: np.ndarray) -> None:
    ...

def compute_similarity(
    v1: EmbeddingVector,
    v2: EmbeddingVector
) -> SimilarityScore:
    ...
```

## Consequences

### Positive Consequences

- **Consistent types** - Single source of truth for types
- **Better IDE support** - Centralized type definitions
- **Easier refactoring** - Change type in one place
- **Clearer interfaces** - Self-documenting type hints
- **Protocol-based design** - Duck typing with type safety

### Negative Consequences

- **Migration effort** - Existing code needs type updates
- **Additional imports** - Need to import from src.types
- **Circular dependency risk** - Must avoid importing from modules

### Neutral Consequences

- **Re-export pattern** - Some types re-exported from core.utils.types
- **Conditional types** - Tensor types depend on PyTorch availability

## Alternatives Considered

### Alternative 1: Keep Types Scattered

- **Description:** Each module defines its own types
- **Pros:** No migration, module isolation
- **Cons:** Duplication, inconsistency, harder refactoring
- **Why not chosen:** Technical debt would grow

### Alternative 2: Use External Type Library

- **Description:** Use a library like `typing-extensions`
- **Pros:** Well-maintained, standard types
- **Cons:** Doesn't cover domain-specific types
- **Why not chosen:** Still need custom types for AI domain

### Alternative 3: Centralized Types Module (CHOSEN)

- **Description:** Single module for all shared types
- **Pros:** Consistency, easier refactoring, clear interfaces
- **Cons:** Migration effort, potential circular imports
- **Why chosen:** Best balance of benefits vs effort

## References

- [Types Implementation](../../src/types/__init__.py)
- [Core Utils Types](../../src/core/utils/types.py)
- [PEP 544 - Protocols](https://peps.python.org/pep-0544/)

---

**Related ADRs:** [ADR-001](adr-001-project-structure.md)
