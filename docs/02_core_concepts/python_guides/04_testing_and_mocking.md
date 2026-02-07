# Python Testing and Mocking

## Introduction

This guide explains how to write reliable tests with pytest and how to mock dependencies. It focuses on patterns used in the RAG Engine Mini codebase (services, repositories, adapters).

## Learning Objectives

By the end of this guide, you will be able to:
- Write unit tests with pytest
- Use fixtures and parametrization
- Mock external dependencies cleanly
- Test async functions
- Structure tests to mirror modules

---

## 1. Why Tests Matter

In this project, tests validate:
- Domain logic (pure functions)
- Application services and use cases
- Adapter behavior (databases, filestores)
- API endpoints

Tests should be small, fast, and deterministic.

---

## 2. Pytest Basics

```python
# tests/test_math.py

def add(a: int, b: int) -> int:
    return a + b


def test_add() -> None:
    assert add(2, 3) == 5
```

Run:

```
pytest tests/ -v
```

---

## 3. Fixtures

Fixtures let you share setup logic across tests.

```python
import pytest

@pytest.fixture()

def sample_document() -> dict[str, str]:
    return {"id": "1", "title": "Hello", "content": "World"}


def test_document_title(sample_document: dict[str, str]) -> None:
    assert sample_document["title"] == "Hello"
```

---

## 4. Parametrization

Test multiple cases in one test function.

```python
import pytest

@pytest.mark.parametrize("value,expected", [("1", 1), ("42", 42)])

def test_parse_int(value: str, expected: int) -> None:
    assert int(value) == expected
```

---

## 5. Mocking Dependencies

Use `unittest.mock` to isolate code from external systems.

```python
from unittest.mock import Mock

class Repo:
    def get(self, doc_id: str) -> dict[str, str] | None:
        raise NotImplementedError


def get_title(repo: Repo, doc_id: str) -> str:
    doc = repo.get(doc_id)
    if doc is None:
        raise ValueError("missing")
    return doc["title"]


def test_get_title_with_mock() -> None:
    repo = Mock(spec=Repo)
    repo.get.return_value = {"id": "1", "title": "Test"}
    assert get_title(repo, "1") == "Test"
```

---

## 6. Testing Async Code

Use `pytest.mark.asyncio` for async functions.

```python
import pytest

async def async_add(a: int, b: int) -> int:
    return a + b

@pytest.mark.asyncio
async def test_async_add() -> None:
    result = await async_add(2, 3)
    assert result == 5
```

---

## 7. Testing Services with Fakes

Prefer fakes over heavy mocks when possible.

```python
class FakeRepo:
    def __init__(self) -> None:
        self.items: dict[str, str] = {"1": "doc"}

    def get(self, doc_id: str) -> str | None:
        return self.items.get(doc_id)


def test_get_with_fake() -> None:
    repo = FakeRepo()
    assert repo.get("1") == "doc"
```

---

## 8. Test Structure in This Repo

Mirror module structure:

```
src/application/services/query_history_service.py
tests/application/services/test_query_history_service.py
```

Keep test names descriptive and focused on behavior.

---

## 9. Checklist

Before you commit:
- Tests cover success and failure paths
- External dependencies are mocked or faked
- Async tests use proper markers
- Test files mirror the `src/` structure

---

## Summary

Testing is about confidence. Use pytest fixtures and parametrization for clarity, and mock or fake dependencies so tests stay fast and reliable.

---

## Additional Resources

- Pytest docs: https://docs.pytest.org/en/latest/
- unittest.mock: https://docs.python.org/3/library/unittest.mock.html
