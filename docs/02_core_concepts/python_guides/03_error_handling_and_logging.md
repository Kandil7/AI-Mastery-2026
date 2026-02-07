# Python Error Handling and Logging

## Introduction

This guide focuses on robust error handling and logging. These are essential for production services like the RAG Engine Mini API and workers.

## Learning Objectives

By the end of this guide, you will be able to:
- Use try/except/finally safely
- Raise and define custom exceptions
- Build context managers for cleanup
- Configure and use logging consistently
- Map exceptions to API responses
- Avoid common error-handling pitfalls

---

## 1. Exceptions Basics

### try/except/finally

```python
from typing import Any

def parse_int(value: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"Invalid integer: {value}") from exc


def safe_divide(a: float, b: float) -> float:
    try:
        return a / b
    except ZeroDivisionError:
        return 0.0
    finally:
        # Always runs
        pass
```

### Catching Specific Exceptions

```python
try:
    data = int("42")
except ValueError:
    data = 0
```

Avoid catching `Exception` unless you re-raise or log and handle.

---

## 2. Custom Exceptions

Define domain-specific exceptions so callers can handle them precisely.

```python
class DocumentNotFoundError(RuntimeError):
    def __init__(self, doc_id: str) -> None:
        super().__init__(f"Document not found: {doc_id}")
        self.doc_id = doc_id
```

Use these in application services:

```python
def get_document(doc_id: str) -> "Document":
    doc = repo.get(doc_id)
    if doc is None:
        raise DocumentNotFoundError(doc_id)
    return doc
```

---

## 3. Exception Mapping in APIs

In FastAPI, map domain exceptions to HTTP errors.

```python
from fastapi import HTTPException

try:
    doc = service.get_document(doc_id)
except DocumentNotFoundError as exc:
    raise HTTPException(status_code=404, detail=str(exc))
```

This keeps domain logic clean and API concerns isolated.

---

## 4. Context Managers

Use context managers for resources that need cleanup.

```python
from contextlib import contextmanager
from typing import Iterator

@contextmanager
def open_file(path: str) -> Iterator[str]:
    handle = open(path, "r", encoding="utf-8")
    try:
        yield handle.read()
    finally:
        handle.close()
```

### Using with

```python
with open("README.md", "r", encoding="utf-8") as handle:
    content = handle.read()
```

---

## 5. Logging Basics

### Standard Logger

```python
import logging

logger = logging.getLogger(__name__)

logger.info("Service started")
logger.warning("Low disk space")
logger.error("Upload failed", exc_info=True)
```

### Minimal Configuration

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s"
)
```

---

## 6. Logging in this Project

Recommended patterns:
- Use `logging.getLogger(__name__)` per module
- Log errors with `exc_info=True`
- Avoid logging secrets or raw document content
- Include request IDs if available

Example:

```python
logger = logging.getLogger(__name__)

try:
    result = use_case.execute(payload)
    logger.info("document.uploaded", extra={"doc_id": result.id})
except Exception:
    logger.exception("document.upload.failed")
    raise
```

---

## 7. Anti-Patterns

Avoid:
- Swallowing exceptions without logging
- Catching too broad exceptions
- Re-raising without context
- Logging full payloads with PII

---

## 8. Checklist

Before you commit:
- Exceptions are specific and meaningful
- API layer maps domain errors to HTTP errors
- Cleanup is guaranteed with context managers
- Logs are informative and safe

---

## Summary

Good error handling and logging are essential to stable services. Use custom exceptions to keep intent clear, map errors at API boundaries, and log what matters without leaking sensitive data.

---

## Additional Resources

- Python exceptions: https://docs.python.org/3/tutorial/errors.html
- Python logging: https://docs.python.org/3/library/logging.html
