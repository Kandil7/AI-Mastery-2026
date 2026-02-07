# Python Typing and Async Guide

## Introduction

This guide covers practical Python typing and async patterns used in this codebase. It connects core language features to the RAG Engine Mini architecture so you can read, extend, and test the code confidently.

## Learning Objectives

By the end of this guide, you will be able to:
- Use type hints consistently in functions and classes
- Model repository and service contracts with Protocols
- Choose between dataclasses and Pydantic models
- Understand async execution, awaitables, and event loops
- Apply safe async patterns in FastAPI and Celery contexts
- Avoid common typing and async pitfalls

---

## 1. Why Types Matter in This Project

The project uses:
- Clean architecture layers (domain, application, adapters, api)
- Explicit interfaces for repositories and services
- Asynchronous I/O for APIs and file storage
- Celery tasks for background processing

Type hints make these boundaries explicit and reduce integration bugs.

---

## 2. Typing Fundamentals

### Function Signatures

```python
from typing import Optional

def normalize_query(text: str, limit: int = 128) -> str:
    cleaned = text.strip()
    return cleaned[:limit]


def find_user(user_id: str) -> Optional[dict[str, str]]:
    if user_id == "missing":
        return None
    return {"id": user_id, "name": "Alice"}
```

### Union and Optional

```python
from typing import Union

# Optional[T] == Union[T, None]
Result = Union[str, None]
```

### Collections

```python
from typing import Iterable

def chunk_ids(ids: Iterable[str], size: int) -> list[list[str]]:
    batch: list[str] = []
    batches: list[list[str]] = []
    for item in ids:
        batch.append(item)
        if len(batch) == size:
            batches.append(batch)
            batch = []
    if batch:
        batches.append(batch)
    return batches
```

---

## 3. Protocols for Interfaces

Protocols define behavior without inheritance. This is ideal for repositories and adapters.

```python
from typing import Protocol

class DocumentRepo(Protocol):
    def save(self, doc: "Document") -> "Document":
        ...

    def get(self, doc_id: str) -> "Document | None":
        ...
```

You can implement this in Postgres, Redis, or in-memory without coupling.

---

## 4. Dataclasses vs Pydantic

### Dataclasses

Good for pure domain models, no validation side effects.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Document:
    id: str
    title: str
    content: str
```

### Pydantic

Best for API boundaries and validated input.

```python
from pydantic import BaseModel

class DocumentCreate(BaseModel):
    title: str
    content: str
```

Rule of thumb:
- Domain: dataclasses
- API input/output: Pydantic models

---

## 5. Type Narrowing and Guards

```python
from typing import Any


def is_document(obj: Any) -> bool:
    return isinstance(obj, dict) and "id" in obj and "content" in obj
```

Type checkers can infer a narrower type inside conditionals.

---

## 6. Async Basics

### Synchronous vs Asynchronous

- Synchronous: one task at a time, blocking I/O
- Asynchronous: switch tasks when waiting on I/O

```python
import asyncio

async def fetch_doc(doc_id: str) -> str:
    await asyncio.sleep(0.1)  # simulate I/O
    return f"doc:{doc_id}"

async def main() -> None:
    result = await fetch_doc("1")
    print(result)

asyncio.run(main())
```

### Awaitables

An awaitable is a coroutine, task, or future that can be awaited.

---

## 7. Async Patterns in FastAPI

FastAPI endpoints can be async. Use async for I/O bound calls.

```python
from fastapi import APIRouter

router = APIRouter()

@router.get("/status")
async def status() -> dict[str, str]:
    return {"status": "ok"}
```

If your code is CPU heavy, keep it sync or offload to a worker.

---

## 8. Async in Celery Tasks

Celery tasks are synchronous by default. If you need async code:

- Use an internal async runner
- Avoid awaiting directly in the task body

```python
import asyncio
from celery import shared_task

async def _async_job(doc_id: str) -> str:
    await asyncio.sleep(0.1)
    return doc_id

@shared_task
def process_doc(doc_id: str) -> str:
    return asyncio.run(_async_job(doc_id))
```

---

## 9. Pitfalls and Best Practices

Common issues:
- Awaiting inside a non-async function
- Calling asyncio.run when an event loop is already running
- Returning non-serializable objects from Celery tasks
- Using blocking I/O inside async endpoints

Best practices:
- Keep interfaces typed and explicit
- Use Protocols for adapters
- Keep async boundaries clear: API layer can be async, tasks are sync
- Prefer small async helpers to isolate I/O

---

## 10. Applied to RAG Engine Mini

Where you will use these concepts:
- `UploadDocumentUseCase.execute_stream` uses async file I/O
- `routes_documents.py` uses async endpoints for uploads
- `tasks.py` wraps async operations for Celery workers
- `DocumentRepo` implementations use typed interfaces

Example: Protocol for a file store

```python
from typing import Protocol, BinaryIO

class FileStore(Protocol):
    def save_stream(self, name: str, data: BinaryIO) -> str:
        ...
```

---

## 11. Checklist

Before you commit:
- All functions have type hints
- Interfaces use Protocols where appropriate
- Async code is only in async functions
- Celery tasks stay synchronous
- Data models are clear: dataclasses or Pydantic

---

## Summary

Typing and async are core to clean architecture in this repo. Use type hints to make dependencies explicit, and use async for I/O in the API layer while keeping Celery tasks sync and minimal.

---

## Additional Resources

- Python typing: https://docs.python.org/3/library/typing.html
- PEP 484: https://peps.python.org/pep-0484/
- Asyncio: https://docs.python.org/3/library/asyncio.html
