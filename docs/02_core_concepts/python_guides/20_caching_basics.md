# Python Caching Basics

## Introduction

This guide covers caching fundamentals in Python and when to use them. Caching can improve performance and reduce redundant work in retrieval, embeddings, and API responses.

## Learning Objectives

By the end of this guide, you will be able to:
- Explain what caching is and why it helps
- Use functools.lru_cache for simple caching
- Understand cache invalidation basics
- Decide when caching is appropriate

---

## 1. Simple Caching with lru_cache

```python
from functools import lru_cache

@lru_cache(maxsize=128)

def compute(x: int) -> int:
    return x * x
```

---

## 2. Cache Size and Eviction

`maxsize` controls how many entries are kept. Older entries are evicted first.

---

## 3. Cache Invalidation

Caches can become stale. Common approaches:
- Time-based expiration
- Manual invalidation
- Versioned keys

---

## 4. When to Cache

Use caching for:
- Expensive computations
- Repeated queries
- External calls with stable results

Avoid caching for data that changes frequently.

---

## Summary

Caching improves performance when used carefully. Keep cache scope small and be intentional about invalidation.

---

## Additional Resources

- functools.lru_cache: https://docs.python.org/3/library/functools.html#functools.lru_cache
