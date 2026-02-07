# Python Packaging and Project Structure

## Introduction

This guide explains Python packaging basics and how project structure influences imports, testing, and deployment. It connects these ideas to the RAG Engine Mini repository layout.

## Learning Objectives

By the end of this guide, you will be able to:
- Understand packages vs modules
- Use `__init__.py` correctly
- Choose import paths that scale
- Structure projects for testing and tooling
- Avoid common packaging pitfalls

---

## 1. Modules vs Packages

- **Module**: a single `.py` file
- **Package**: a directory containing `__init__.py`

Example:

```
src/
  core/
    __init__.py
    config.py
```

`core` is a package, `config.py` is a module.

---

## 2. Why `__init__.py` Matters

`__init__.py` tells Python to treat a directory as a package.

```python
# src/core/__init__.py
from .config import load_settings

__all__ = ["load_settings"]
```

This allows:

```python
from src.core import load_settings
```

---

## 3. Import Paths and Tools

Common patterns:
- Use absolute imports from the `src` root
- Keep `PYTHONPATH` consistent in tooling
- Avoid relative imports across boundaries

Example:

```python
from src.application.services import QueryHistoryService
```

---

## 4. Layout in This Repo

Key folders:
- `src/`: application code
- `tests/`: mirrors `src` structure
- `docs/`: learning content and guides
- `notebooks/`: learning notebooks

This layout supports tooling like pytest, mypy, and black.

---

## 5. Common Pitfalls

Avoid:
- Running modules directly with relative imports (causes import errors)
- Missing `__init__.py` in packages
- Duplicating package names
- Importing from `tests/` in production code

---

## 6. Recommended Workflow

```bash
# run tests
make test

# run lint
make lint

# run app
make run
```

---

## Summary

Good structure reduces import errors and simplifies testing and deployment. Keep packages explicit, use absolute imports, and align tests with application modules.

---

## Additional Resources

- Packaging guide: https://packaging.python.org/en/latest/
- Python modules: https://docs.python.org/3/tutorial/modules.html
