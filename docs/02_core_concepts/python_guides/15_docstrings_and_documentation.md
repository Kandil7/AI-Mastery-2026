# Python Documentation and Docstrings

## Introduction

This guide explains how to document Python code with docstrings and how to keep project documentation usable for teammates. Good docs reduce onboarding time and bugs.

## Learning Objectives

By the end of this guide, you will be able to:
- Write clear docstrings for functions and classes
- Use type hints alongside docstrings
- Structure documentation for modules
- Avoid common documentation anti-patterns

---

## 1. Function Docstrings

```python

def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        Sum of a and b.
    """
    return a + b
```

---

## 2. Class Docstrings

```python
class DocumentService:
    """Service for document operations.

    Handles validation and persistence logic.
    """
```

---

## 3. Module Docstrings

```python
"""Utilities for file ingestion.

Includes streaming readers and safe writers.
"""
```

---

## 4. Keep Docs Close to Code

- Public functions should have docstrings
- Keep docs updated when behavior changes
- Prefer short, precise descriptions

---

## 5. Anti-Patterns

Avoid:
- Restating obvious code
- Outdated docs
- Huge docstrings with implementation details

---

## Summary

Documentation should be concise and accurate. Use docstrings to describe intent, inputs, outputs, and side effects.

---

## Additional Resources

- PEP 257: https://peps.python.org/pep-0257/
