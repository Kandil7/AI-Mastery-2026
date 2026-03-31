# ADR-001: Project Structure and Module Organization

**Status:** Accepted
**Date:** 2026-03-31
**Authors:** Kandil7

---

## Context

The AI-Mastery-2026 project has grown to include multiple domains:
- Core mathematics (linear algebra, calculus, optimization, statistics)
- Classical and deep learning ML
- LLM engineering (transformers, attention, fine-tuning)
- RAG systems (chunking, retrieval, reranking)
- AI agents (orchestration, multi-agent systems)
- Production infrastructure (API, caching, monitoring)

As the codebase expanded, several structural issues emerged:
1. Naming conflicts between files and directories (e.g., `classical.py` vs `classical/`)
2. Inconsistent import patterns across modules
3. Missing `__all__` exports in many modules
4. Legacy code mixed with production code
5. Empty placeholder directories suggesting incomplete features

## Decision

We propose the following structural organization:

### Module Naming Convention

1. **Directories over files** for complex modules with multiple components
2. **Descriptive suffixes** for from-scratch implementations:
   - `classical_scratch.py` - From-scratch classical ML
   - `neural_networks_scratch.py` - From-scratch neural networks
3. **Clear separation** between reference implementations and optimized versions

### Import Standardization

1. **Absolute imports** for cross-package imports: `from src.module import ...`
2. **Relative imports** within packages: `from .submodule import ...`
3. **No wildcard imports**: Always use explicit imports
4. **`__all__` exports** in all `__init__.py` files

### Directory Structure

```
src/
├── core/              # Mathematical foundations
│   ├── __init__.py
│   ├── integration.py
│   ├── optimization.py
│   ├── mcmc.py
│   └── ...
├── ml/                # Machine learning
│   ├── __init__.py
│   ├── classical/     # Organized classical ML
│   ├── deep_learning/ # Organized deep learning
│   ├── classical_scratch.py    # From-scratch implementations
│   ├── neural_networks_scratch.py
│   ├── vision.py
│   └── gnn_recommender.py
├── llm/               # LLM engineering
├── rag/               # RAG systems
├── agents/            # AI agents
├── production/        # Production infrastructure
├── config/            # Centralized configuration (NEW)
└── types/             # Shared type definitions (NEW)
```

### Legacy Code Management

1. **Archive directory** for deprecated code
2. **Clear migration notes** in archive README
3. **No imports from archive** in production code

## Consequences

### Positive Consequences

- **Clearer module boundaries** - Easier to understand what belongs where
- **Better IDE support** - Proper `__all__` enables autocomplete
- **Reduced naming conflicts** - Descriptive names prevent confusion
- **Cleaner codebase** - Legacy code properly archived
- **Consistent patterns** - Standardized imports reduce cognitive load

### Negative Consequences

- **Migration effort** - Existing imports need updating
- **Learning curve** - New contributors need to learn structure
- **File renames** - Some files renamed (breaking for existing users)

### Neutral Consequences

- **Slightly more files** - Separate scratch implementations
- **Additional documentation** - ADRs and structure docs needed

## Alternatives Considered

### Alternative 1: Keep Current Structure

- **Description:** Maintain existing file organization
- **Pros:** No migration needed, backward compatible
- **Cons:** Growing technical debt, continued confusion
- **Why not chosen:** Technical debt would compound over time

### Alternative 2: Complete Restructure

- **Description:** Full reorganization with new package hierarchy
- **Pros:** Clean slate, optimal structure
- **Cons:** Breaking changes, significant migration effort
- **Why not chosen:** Too disruptive for active development

### Alternative 3: Gradual Refactoring (CHOSEN)

- **Description:** Incremental improvements with clear phases
- **Pros:** Manageable changes, can rollback, less risky
- **Cons:** Takes longer, temporary inconsistencies
- **Why chosen:** Balances improvement with stability

## References

- [Phase 1 Restructuring Plan](../README.md)
- [Phase 2 Implementation Notes](../implementation-notes.md)
- [Contributing Guide](../../CONTRIBUTING.md)

---

**Related ADRs:** None (first ADR)
