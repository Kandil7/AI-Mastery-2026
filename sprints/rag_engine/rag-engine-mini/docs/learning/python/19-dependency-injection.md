# Python Dependency Injection Basics

## Introduction

This guide explains dependency injection (DI) and why it matters for testability and clean architecture. It connects to how services and repositories are wired in this project.

## Learning Objectives

By the end of this guide, you will be able to:
- Explain dependency injection and inversion of control
- Inject repositories and services into use cases
- Replace dependencies with fakes in tests
- Avoid tight coupling between layers

---

## 1. What is Dependency Injection?

Dependency injection means passing dependencies into a class rather than creating them inside it.

```python
class Repo:
    def get(self, doc_id: str) -> str:
        return doc_id

class Service:
    def __init__(self, repo: Repo) -> None:
        self.repo = repo
```

---

## 2. Why It Helps

- Easier unit testing
- Less coupling between layers
- Cleaner architecture

---

## 3. Example with Fake Repository

```python
class FakeRepo:
    def __init__(self) -> None:
        self.data = {"1": "doc"}

    def get(self, doc_id: str) -> str | None:
        return self.data.get(doc_id)

class Service:
    def __init__(self, repo: FakeRepo) -> None:
        self.repo = repo

    def get_doc(self, doc_id: str) -> str:
        doc = self.repo.get(doc_id)
        if doc is None:
            raise ValueError("missing")
        return doc
```

---

## 4. Applying to This Project

- Use cases depend on interfaces, not concrete adapters
- Repositories are injected at bootstrap
- Tests use fakes or mocks to isolate behavior

---

## Summary

Dependency injection makes systems easier to test and evolve. Prefer passing dependencies into constructors instead of creating them internally.

---

## Additional Resources

- Clean Architecture by Robert C. Martin
