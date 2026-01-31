# Python Data Validation Patterns

## Introduction

This guide explains practical data validation patterns in Python, from simple checks to Pydantic models. Validation keeps data safe at system boundaries.

## Learning Objectives

By the end of this guide, you will be able to:
- Validate input data with simple checks
- Use Pydantic for structured validation
- Separate validation from business logic
- Handle validation errors cleanly

---

## 1. Simple Validation

```python
from typing import Any


def validate_document(data: dict[str, Any]) -> None:
    if "id" not in data:
        raise ValueError("Missing id")
    if "content" not in data:
        raise ValueError("Missing content")
```

---

## 2. Pydantic Models

```python
from pydantic import BaseModel

class DocumentCreate(BaseModel):
    title: str
    content: str
```

---

## 3. Keep Validation at Boundaries

- API input validation
- File ingestion validation
- External API responses

---

## Summary

Validation protects your system from bad data. Use simple checks for internal code and Pydantic models for external inputs.

---

## Additional Resources

- Pydantic: https://docs.pydantic.dev/
