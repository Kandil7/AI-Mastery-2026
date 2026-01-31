# Python Concurrency Basics (Threads and Processes)

## Introduction

This guide introduces concurrency in Python using threads and processes. It helps you decide when to use threading vs multiprocessing and how to avoid common pitfalls in service code.

## Learning Objectives

By the end of this guide, you will be able to:
- Understand the difference between concurrency and parallelism
- Use `threading` for I/O-bound tasks
- Use `multiprocessing` for CPU-bound tasks
- Avoid race conditions with locks
- Recognize when async is a better fit

---

## 1. Concurrency vs Parallelism

- **Concurrency**: multiple tasks progress by interleaving
- **Parallelism**: tasks run at the same time on multiple CPU cores

Python threads are good for I/O-bound workloads due to the GIL.

---

## 2. Threading for I/O-Bound Work

```python
import threading
import time


def fetch(name: str) -> None:
    time.sleep(0.1)
    print(f"done: {name}")

threads = [threading.Thread(target=fetch, args=(f"job-{i}",)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()
```

---

## 3. Multiprocessing for CPU-Bound Work

```python
from multiprocessing import Pool

def cpu_task(x: int) -> int:
    return x * x

with Pool(processes=4) as pool:
    results = pool.map(cpu_task, [1, 2, 3, 4])

print(results)
```

---

## 4. Avoiding Race Conditions

```python
import threading

counter = 0
lock = threading.Lock()


def increment() -> None:
    global counter
    with lock:
        counter += 1

threads = [threading.Thread(target=increment) for _ in range(100)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter)
```

---

## 5. When to Use Async Instead

- Use **async** for high-concurrency I/O (APIs, file streams)
- Use **threads** for blocking libraries
- Use **processes** for CPU-heavy computations

---

## 6. Checklist

Before you commit:
- Use threads for I/O, processes for CPU
- Protect shared state with locks
- Prefer async for high fan-out I/O

---

## Summary

Concurrency choices depend on workload type. Threads help with I/O, processes help with CPU, and async fits event-driven services.

---

## Additional Resources

- threading: https://docs.python.org/3/library/threading.html
- multiprocessing: https://docs.python.org/3/library/multiprocessing.html
