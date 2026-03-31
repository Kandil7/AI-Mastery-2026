# Python Imports and Modules Guide

## Introduction

This guide explains Python's import system, how to resolve circular import dependencies, and best practices for organizing modules.

## Learning Objectives

By the end of this guide, you will understand:
- **How Python's import system works**
- **What causes circular import errors**
- **How to resolve circular dependencies**
- **When to use TYPE_CHECKING for forward references**
- **Module organization best practices**
- **Relative vs absolute imports**

---

## Python Import System Overview

### How Imports Work

When Python executes an `import` statement:

```
1. Check if module is in sys.modules
   ↓ (if not found)
2. Search for module in sys.path
   ↓ (if found)
3. Execute module code and store in sys.modules
   ↓
4. Make module available to importing code
```

### Import Types

```python
# 1. Module import
import math
import numpy as np

# 2. From import
from datetime import datetime
from typing import List, Dict

# 3. Star import (avoid in production)
from os.path import *

# 4. Relative import
from . import local_module
from ..parent_module import ParentClass
```

---

## Circular Import Problem

### What is Circular Import?

A **circular import** occurs when:
- Module A imports Module B
- Module B imports Module A
- Either directly or transitively (A → B → C → A)

### Why Circular Imports Fail

```python
# models.py
from services import UserService  # Imports services

class User:
    pass


# services.py
from models import User  # Imports models
import models  # Circular!

class UserService:
    def create_user(self):
        return User()
```

When `models.py` is imported:
1. Python starts executing `models.py`
2. Sees `from services import UserService`
3. Starts executing `services.py`
4. Sees `from models import User`
5. But `models.py` hasn't finished importing yet!
6. `User` doesn't exist → ImportError or AttributeError

---

## Resolving Circular Imports

### Solution 1: Import Inside Functions

Import at function call time instead of module level:

```python
# models.py
class User:
    def get_service(self):
        # Import inside method
        from services import UserService
        return UserService()

# services.py
class UserService:
    def create_user(self):
        # Import inside method
        from models import User
        return User()
```

**Benefits:**
- Breaks circular dependency
- Import only runs when needed
- Defers import until method execution

**Drawbacks:**
- Slightly slower (imports on every call)
- Can be repetitive if called frequently

### Solution 2: Import at End of Module

Place import after all classes are defined:

```python
# models.py
class User:
    pass

# Import at end, after User is defined
from services import UserService


# services.py
class UserService:
    pass

# Import at end, after UserService is defined
from models import User
```

**Benefits:**
- Classes defined before import
- Simple and clear
- Works for many cases

**Drawbacks:**
- Doesn't work for type hints
- Still circular at runtime

### Solution 3: Use TYPE_CHECKING

Use `typing.TYPE_CHECKING` for type hints only:

```python
# models.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from services import UserService

class User:
    def get_service(self) -> "UserService":
        # Import for runtime
        from services import UserService
        return UserService()

# services.py
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models import User

class UserService:
    def create_user(self) -> User:
        # Import for runtime
        from models import User
        return User()
```

**Benefits:**
- Type checking works (mypy, IDEs)
- No circular import at runtime
- Clean separation

**Drawbacks:**
- Still need runtime import in methods
- More verbose

### Solution 4: Refactor to Remove Cycle

Best solution: restructure code to remove circular dependency:

```python
# Before: Circular
models.py ← services.py ← models.py

# After: No cycle
models.py
services.py
common.py  # Shared utilities

models.py ← common.py
services.py ← common.py
```

**Example Refactor:**

```python
# common.py - Shared interface
from abc import ABC, abstractmethod

class IUserRepository(ABC):
    @abstractmethod
    def save(self, user): pass

# models.py
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from common import IUserRepository

class User:
    # No direct dependency on services
    pass

# services.py
from common import IUserRepository
from models import User

class UserService:
    def __init__(self, repo: IUserRepository):
        self.repo = repo  # Dependency injection!

    def create_user(self) -> User:
        user = User()
        return self.repo.save(user)
```

**Benefits:**
- Clean architecture
- Testable (inject mocks)
- No circular dependencies
- Better separation of concerns

**Drawbacks:**
- Requires code restructuring
- More initial work

---

## Absolute vs Relative Imports

### Absolute Imports (Recommended)

```python
# From project root
from src.application.services import SearchService
from src.domain.entities import Document
from src.adapters.persistence.postgres import PostgresDocumentRepo
```

**Benefits:**
- Clear and explicit
- Works from anywhere
- IDE autocomplete works better
- Refactoring tool-friendly

### Relative Imports

```python
# From same directory
from .search_service import SearchService

# From parent directory
from ..domain.entities import Document

# From sibling package
from ..adapters.persistence.postgres import PostgresDocumentRepo
```

**When to Use:**
- Within a package's own modules
- When package will be moved/relocated
- Avoid repeating long paths in large packages

**When to Avoid:**
- Crossing package boundaries
- When absolute import is clearer
- In application entry points (main.py)

---

## Module Organization Best Practices

### Recommended Structure

```
src/
├── domain/           # Pure business logic, no external deps
│   ├── __init__.py
│   ├── entities.py    # Domain models
│   └── errors.py     # Domain errors
├── application/       # Use cases, no framework deps
│   ├── __init__.py
│   ├── services/      # Application services
│   └── use_cases/    # Business workflows
├── adapters/         # External integrations
│   ├── __init__.py
│   ├── persistence/   # Database adapters
│   ├── llm/          # LLM integrations
│   └── api/          # HTTP clients
├── api/             # HTTP layer
│   ├── __init__.py
│   └── v1/          # API routes
└── core/            # Shared utilities
    ├── __init__.py
    ├── config.py      # Configuration
    └── logging.py     # Logging setup
```

### Import Rules

1. **Domain layer**: No imports from adapters/api
```python
# OK: domain → domain
from src.domain.entities import Document

# BAD: domain → adapters
from src.adapters.persistence import PostgresRepo  # Circular!
```

2. **Application layer**: Import from domain, not adapters
```python
# OK: application → domain
from src.domain.entities import Document

# BAD: application → adapters
from src.adapters.persistence import PostgresRepo
```

3. **Adapter layer**: Can import from domain, but not application
```python
# OK: adapters → domain
from src.domain.entities import Document

# BAD: adapters → application
from src.application.services import DocumentService
```

4. **API layer**: Can import from all layers except API
```python
# OK: api → application
from src.application.use_cases import AskQuestion

# BAD: api → api (circular!)
from src.api.v2 import SomeEndpoint
```

---

## Forward References

### When You Need Forward References

Sometimes you need to reference a class before it's defined:

```python
class Node:
    def __init__(self):
        self.children = []  # Type: List[Node]

    def add_child(self, child: Node):  # Circular reference!
        self.children.append(child)
```

### Solution: String Type Hints

```python
# Use string for forward reference
class Node:
    def __init__(self):
        self.children = []  # type: List[Node]

    def add_child(self, child: 'Node'):  # String annotation!
        self.children.append(child)

# Or use from __future__ annotations
from __future__ import annotations

class Node:
    def __init__(self):
        self.children: list[Node] = []

    def add_child(self, child: Node) -> None:
        self.children.append(child)
```

---

## Import Performance

### What Gets Cached

Python caches imports in `sys.modules`:

```python
import sys

# First import: executes module
import my_module
print("Imported:", my_module in sys.modules)

# Second import: uses cache
import my_module
print("Still cached:", my_module in sys.modules)
```

### When to Use Lazy Imports

```python
# BAD: Always imports heavy library
import matplotlib.pyplot as plt
import torch
import tensorflow as tf

def process_data(data):
    # Libraries loaded even if not used!
    return data.mean()

# GOOD: Lazy import
def process_data(data):
    import numpy as np  # Only loads when needed
    return np.mean(data)

def plot_results(results):
    import matplotlib.pyplot as plt  # Only loads for plotting
    plt.plot(results)
```

---

## Common Import Errors

### 1. ModuleNotFoundError

```python
# Error
import nonexistent_module

# Fix
from . import my_local_module  # Use relative import
import numpy  # Install missing package
```

### 2. ImportError

```python
# Error (wrong name)
from my_module import wrong_name

# Fix
from my_module import correct_name
```

### 3. AttributeError (Circular Import)

```python
# Error
from models import User
# Service defined but User doesn't exist yet!

# Fix: Refactor or use TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from models import User
```

---

## Checklist: Import Review

Before committing code, check:

- [ ] No circular imports (A → B → A)
- [ ] Type hints use strings or TYPE_CHECKING for forward refs
- [ ] Heavy libraries imported inside functions
- [ ] Absolute imports for cross-package references
- [ ] Relative imports only within same package
- [ ] Domain layer doesn't depend on adapters/api
- [ ] No star imports (`from module import *`)

---

## Summary

### Key Takeaways:

1. **Circular imports** cause runtime errors
2. **Import inside functions** breaks circular dependencies
3. **TYPE_CHECKING** enables type hints without runtime imports
4. **Refactor** to remove cycles is best long-term solution
5. **Absolute imports** are clearer and more maintainable
6. **Lazy imports** improve startup performance
7. **String type hints** enable forward references

### Best Practices:

- ✅ Import inside functions to break circular deps
- ✅ Use TYPE_CHECKING for type hints only
- ✅ Prefer absolute imports over relative
- ✅ Use relative imports only within package
- ✅ Lazy import heavy/optional libraries
- ✅ Refactor to remove circular dependencies

### Anti-Patterns:

- ❌ Circular imports between layers
- ❌ Star imports (`from x import *`)
- ❌ Importing heavy libraries at module level
- ❌ Relative imports across packages
- ❌ Domain depending on adapters

---

## Additional Resources

- **Python Import System**: https://docs.python.org/3/reference/import.html
- **PEP 484**: TYPE_CHECKING constant
- **Type Hints**: https://docs.python.org/3/library/typing.html
- **Circular Dependency**: Martin Fowler's articles
