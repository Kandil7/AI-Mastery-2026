# ADR-002: Configuration Management Strategy

**Status:** Accepted
**Date:** 2026-03-31
**Authors:** Kandil7

---

## Context

The AI-Mastery-2026 project has configuration scattered across:
- Environment variables (`.env` files)
- Hard-coded constants in modules
- Configuration dictionaries in various functions
- YAML/JSON config files (inconsistent usage)

This leads to:
1. Difficulty changing settings across environments
2. Hard-to-test code with hard-coded values
3. Inconsistent configuration patterns
4. No centralized validation of settings

## Decision

We propose a centralized configuration system:

### New `src/config/` Module

```
src/config/
├── __init__.py          # Exports all config classes
├── settings.py          # Global application settings
├── model_config.py      # Model-specific configurations
└── data_config.py       # Data pipeline configurations
```

### Configuration Classes

1. **Settings** - Global application settings
   - Environment (dev/test/prod)
   - Paths (data, models, cache)
   - API settings (host, port)
   - Performance settings (batch size, workers)
   - Feature flags

2. **ModelConfig** - Base model configuration
   - Model type, dimensions, layers
   - Dropout, activation functions

3. **TransformerConfig** - Transformer-specific settings
   - Number of heads, FF dimension
   - Sequence length, vocabulary size

4. **TrainingConfig** - Training hyperparameters
   - Learning rate, optimizer
   - Batch size, epochs
   - Early stopping

5. **RAGConfig** - RAG system settings
   - Chunk size, overlap
   - Top-k retrieval
   - Embedding model

### Usage Pattern

```python
from src.config import get_settings, TransformerConfig, TrainingConfig

# Get global settings
settings = get_settings()
print(f"Environment: {settings.environment}")

# Create model config
model_config = TransformerConfig(
    hidden_dim=768,
    num_heads=12,
    num_layers=12
)

# Create training config
train_config = TrainingConfig(
    learning_rate=1e-4,
    batch_size=32,
    num_epochs=10
)
```

### Environment Variable Support

All settings can be overridden via environment variables:
```bash
export ENVIRONMENT=production
export API_PORT=8080
export BATCH_SIZE=64
export MODEL_DIR=/path/to/models
```

## Consequences

### Positive Consequences

- **Centralized configuration** - Single source of truth
- **Type safety** - Config classes with type hints
- **Validation** - `__post_init__` validation
- **Testability** - Easy to override settings in tests
- **Documentation** - Self-documenting config classes

### Negative Consequences

- **Migration effort** - Existing code needs updating
- **Additional dependency** - Requires python-dotenv (optional)
- **Learning curve** - New pattern for contributors

### Neutral Consequences

- **Slightly more code** - Config classes add lines
- **Singleton pattern** - Global settings instance

## Alternatives Considered

### Alternative 1: Continue with Environment Variables Only

- **Description:** Use only environment variables
- **Pros:** Simple, no additional code
- **Cons:** No validation, no type safety, hard to document
- **Why not chosen:** Doesn't scale for complex configurations

### Alternative 2: YAML Configuration Files

- **Description:** Store all config in YAML files
- **Pros:** Human-readable, version-controllable
- **Cons:** No type safety, runtime errors possible
- **Why not chosen:** Less IDE support, harder to validate

### Alternative 3: Pydantic Settings (CHOSEN with dataclasses)

- **Description:** Use Pydantic or dataclasses for config
- **Pros:** Type safety, validation, IDE support
- **Cons:** Additional dependency (Pydantic)
- **Why chosen:** We use dataclasses (stdlib) for zero dependencies

## References

- [Settings Implementation](../../src/config/settings.py)
- [Model Config Implementation](../../src/config/model_config.py)
- [12-Factor App Config](https://12factor.net/config)

---

**Related ADRs:** [ADR-001](adr-001-project-structure.md)
