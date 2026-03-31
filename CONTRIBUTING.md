# Contributing to AI-Mastery-2026

Thank you for your interest in contributing to AI-Mastery-2026! This document provides guidelines and instructions for contributing to the project.

<div align="center">

[Code of Conduct](#code-of-conduct) • [Getting Started](#getting-started) • [Development Workflow](#development-workflow) • [Coding Standards](#coding-standards) • [Pull Requests](#pull-requests)

</div>

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Setting Up Development Environment](#setting-up-development-environment)
- [Development Workflow](#development-workflow)
  - [Branch Naming](#branch-naming)
  - [Commit Messages](#commit-messages)
  - [Testing](#testing)
- [Coding Standards](#coding-standards)
  - [Python Style Guide](#python-style-guide)
  - [Type Hints](#type-hints)
  - [Documentation](#documentation)
  - [Import Organization](#import-organization)
- [Pull Requests](#pull-requests)
  - [PR Template](#pr-template)
  - [Review Process](#review-process)
- [Architecture Decision Records (ADRs)](#architecture-decision-records-adrs)
- [Questions?](#questions)

---

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them learn
- Focus on constructive feedback
- Maintain a professional environment

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- pip or conda/mamba
- (Optional) Docker for containerized development

### Setting Up Development Environment

1. **Fork and clone the repository:**
   ```bash
   git clone https://github.com/your-username/AI-Mastery-2026.git
   cd AI-Mastery-2026
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   # Or with conda
   conda create -n ai-mastery python=3.10
   conda activate ai-mastery
   ```

3. **Install development dependencies:**
   ```bash
   pip install -e ".[dev]"
   # Or using make
   make install-dev
   ```

4. **Set up pre-commit hooks:**
   ```bash
   pre-commit install
   ```

5. **Verify installation:**
   ```bash
   pytest tests/unit -v --tb=short
   ```

---

## Development Workflow

### Branch Naming

Use descriptive branch names following this convention:

```
<type>/<short-description>

Examples:
- feature/add-transformer-encoder
- fix/memory-leak-rag-pipeline
- docs/update-api-reference
- refactor/simplify-attention-mechanism
- test/add-benchmark-tests
```

**Branch Types:**
- `feature` - New functionality
- `fix` - Bug fixes
- `docs` - Documentation changes
- `refactor` - Code refactoring (no functional changes)
- `test` - Adding or updating tests
- `chore` - Maintenance tasks

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Examples:**
```
feat(llm): add RoPE positional encoding

- Implement rotary positional embeddings
- Add tests for RoPE attention
- Update transformer documentation

Closes #123
```

```
fix(rag): resolve memory leak in vector store

- Properly release FAISS index references
- Add cleanup method to base vector store

Fixes #456
```

**Commit Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `style` - Formatting (no code change)
- `refactor` - Code refactoring
- `test` - Tests
- `chore` - Maintenance

### Testing

1. **Run unit tests:**
   ```bash
   pytest tests/unit -v
   ```

2. **Run integration tests:**
   ```bash
   pytest tests/integration -v
   ```

3. **Run with coverage:**
   ```bash
   pytest --cov=src --cov-report=html
   ```

4. **Run specific test:**
   ```bash
   pytest tests/unit/core/test_optimization.py -v
   ```

**Test Guidelines:**
- Write tests for new features
- Maintain >90% code coverage for new code
- Use descriptive test names: `test_<function>_<scenario>_<expected>`
- Mock external dependencies

---

## Coding Standards

### Python Style Guide

- **Line length:** 100 characters maximum
- **Indentation:** 4 spaces (no tabs)
- **Quotes:** Double quotes for strings
- **Naming conventions:**
  - Classes: `PascalCase` (e.g., `TransformerEncoder`)
  - Functions/Methods: `snake_case` (e.g., `scaled_dot_product_attention`)
  - Constants: `UPPER_CASE` (e.g., `MAX_SEQUENCE_LENGTH`)
  - Private methods: `_leading_underscore`

**Code Formatting:**
```bash
# Format code
black src/ tests/

# Sort imports
isort src/ tests/

# Lint code
flake8 src/ tests/
```

### Type Hints

Use type hints for all function signatures:

```python
from typing import List, Dict, Optional, Tuple, Union

def process_tokens(
    tokens: List[str],
    max_length: int = 512,
    padding: bool = True
) -> Tuple[List[int], List[int]]:
    """Process tokens and return IDs and attention mask."""
    ...
```

**Type Hint Guidelines:**
- Use `Optional[T]` instead of `Union[T, None]`
- Use `List[T]`, `Dict[K, V]` instead of `list`, `dict`
- Add return type annotations
- Use `Any` sparingly and with justification

### Documentation

**Module Docstrings:**
```python
"""
Module Name
===========

Brief description of module functionality.

Components:
- Component1: Description
- Component2: Description

Example:
    >>> from src.module import Component
    >>> component = Component()
    >>> result = component.process(data)
"""
```

**Function Docstrings:**
```python
def function_name(arg1: Type1, arg2: Type2) -> ReturnType:
    """
    Brief description of function.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ValueError: When invalid input provided
        TypeError: When wrong type provided

    Example:
        >>> result = function_name(value1, value2)
    """
```

### Import Organization

**Order of imports:**
1. Standard library imports
2. Third-party imports
3. Local application imports

**Within each section:**
- Alphabetical order
- Group related imports together

```python
# Standard library
import math
from typing import List, Optional

# Third-party
import numpy as np
import torch
import torch.nn as nn

# Local application
from src.core.optimization import Adam
from src.ml.deep_learning import Layer
```

**Import Guidelines:**
- Use absolute imports for cross-package imports: `from src.module import ...`
- Use relative imports within packages: `from .submodule import ...`
- Avoid wildcard imports: `from module import *`
- Add `__all__` to all `__init__.py` files

---

## Pull Requests

### PR Template

When creating a pull request, use the following template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change that fixes an issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to change)
- [ ] Documentation update

## Motivation and Context
Why is this change needed? What problem does it solve?

## How Has This Been Tested?
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing performed

Test details:
- Test case 1: Description
- Test case 2: Description

## Checklist
- [ ] Code follows project style guidelines
- [ ] Type hints added for new functions
- [ ] Documentation updated
- [ ] Tests pass locally
- [ ] No new warnings introduced

## Screenshots (if applicable)
Add screenshots for UI changes

## Related Issues
Closes #123
Related to #456
```

### Review Process

1. **Submit PR** with completed template
2. **Automated checks** run (CI/CD)
3. **Code review** by maintainers
4. **Address feedback** and update PR
5. **Approval** and merge

**Review Guidelines:**
- Be constructive and respectful
- Focus on code quality, not personal preference
- Suggest improvements, not just criticisms
- Approve when ready, request changes when needed

---

## Architecture Decision Records (ADRs)

For significant architectural changes, create an ADR:

1. Create file: `docs/architecture/decisions/adr-NNN-short-title.md`
2. Use ADR template:
   ```markdown
   # ADR NNN: Title

   ## Status
   Proposed | Accepted | Deprecated | Superseded

   ## Context
   What is the issue that we're seeing?

   ## Decision
   What is the change that we're proposing?

   ## Consequences
   What becomes easier or more difficult?
   ```

3. Reference ADR in PR description

---

## Questions?

- **General questions:** Open a [Discussion](https://github.com/Kandil7/AI-Mastery-2026/discussions)
- **Bug reports:** Open an [Issue](https://github.com/Kandil7/AI-Mastery-2026/issues)
- **Code review questions:** Ask in your PR
- **Direct contact:** medokandeal7@gmail.com

---

## Thank You!

Your contributions make AI-Mastery-2026 better for everyone. We appreciate your time and effort!

<div align="center">

**Last Updated:** March 31, 2026

[Back to Top](#contributing-to-ai-mastery-2026)

</div>
