# Python Performance Profiling

## Introduction

This guide introduces basic profiling and performance measurement techniques in Python. It helps you identify bottlenecks in data processing and API workloads.

## Learning Objectives

By the end of this guide, you will be able to:
- Measure execution time
- Use cProfile for function-level profiling
- Interpret profiling output
- Apply simple optimizations safely

---

## 1. Timing with time.perf_counter

```python
import time

start = time.perf_counter()
# work
end = time.perf_counter()
print(f"elapsed: {end - start:.6f}s")
```

---

## 2. Function Timing Helper

```python
import time
from typing import Callable, TypeVar

T = TypeVar("T")

def time_it(fn: Callable[[], T]) -> tuple[T, float]:
    start = time.perf_counter()
    result = fn()
    end = time.perf_counter()
    return result, end - start
```

---

## 3. cProfile Basics

```python
import cProfile

def work() -> int:
    total = 0
    for i in range(10000):
        total += i
    return total

cProfile.run("work()")
```

---

## 4. Interpreting Results

Look for:
- High cumulative time
- Hot functions called frequently

---

## 5. Safe Optimization

- Optimize only after measuring
- Prefer algorithmic improvements
- Avoid premature micro-optimizations

---

## Summary

Profiling helps you focus on real bottlenecks. Use timing and cProfile to measure, then optimize with evidence.

---

## Additional Resources

- cProfile: https://docs.python.org/3/library/profile.html
