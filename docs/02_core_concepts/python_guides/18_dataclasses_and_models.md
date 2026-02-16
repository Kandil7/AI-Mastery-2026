# Python Data Classes and Models

## Introduction

This guide explains when to use dataclasses and how they relate to Pydantic models. It focuses on keeping domain models simple and API models validated.

## Learning Objectives

By the end of this guide, you will be able to:
- Create dataclasses for domain entities
- Compare dataclasses and Pydantic models
- Use frozen dataclasses for immutability
- Keep validation at API boundaries

---

## 1. Basic Dataclass

```python
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    title: str
    content: str
```

---

## 2. Frozen Dataclass

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class User:
    id: str
    name: str
```

---

## 3. Dataclass vs Pydantic

- Dataclasses: lightweight, no validation by default
- Pydantic: validation and parsing

Use Pydantic for API input/output. Use dataclasses for domain logic.

---

## Summary

Dataclasses are ideal for domain entities. Pydantic models enforce validation at system boundaries.

---

## Additional Resources

- dataclasses: https://docs.python.org/3/library/dataclasses.html
- pydantic: https://docs.pydantic.dev/
