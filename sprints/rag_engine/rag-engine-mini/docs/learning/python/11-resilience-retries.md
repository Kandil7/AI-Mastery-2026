# Python Resilience: Retries, Timeouts, and Backoff

## Introduction

This guide covers resilience patterns for external calls: timeouts, retries, and backoff strategies. These are critical for APIs, storage, and LLM integrations in production systems.

## Learning Objectives

By the end of this guide, you will be able to:
- Set timeouts for external calls
- Implement retry loops with backoff
- Distinguish retryable vs non-retryable errors
- Avoid retry storms
- Apply resilience patterns safely

---

## 1. Timeouts First

Always set timeouts for external calls to avoid hanging requests.

```python
import requests

response = requests.get("https://example.com", timeout=5)
```

---

## 2. Simple Retry Loop

```python
import time


def retry(operation, attempts: int = 3, delay: float = 0.5):
    last_error = None
    for _ in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            time.sleep(delay)
    raise last_error
```

---

## 3. Exponential Backoff

```python
import time


def retry_backoff(operation, attempts: int = 3, base_delay: float = 0.5):
    last_error = None
    for attempt in range(attempts):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            sleep = base_delay * (2 ** attempt)
            time.sleep(sleep)
    raise last_error
```

---

## 4. Retryable vs Non-Retryable

Not all errors should be retried.

```python
class RetryableError(RuntimeError):
    pass

class NonRetryableError(RuntimeError):
    pass


def risky() -> str:
    raise RetryableError("temporary")
```

Only retry known transient errors (timeouts, rate limits, 5xx).

---

## 5. Avoid Retry Storms

- Use jitter (random delay)
- Cap maximum delay
- Limit total attempts
- Circuit break after repeated failures

---

## 6. Checklist

Before you commit:
- Timeouts are defined for external calls
- Retries are bounded
- Retryable errors are explicit
- Backoff uses jitter for safety

---

## Summary

Resilience patterns prevent cascading failures. Use timeouts, retry with backoff, and never retry blindly.

---

## Additional Resources

- Requests timeout: https://requests.readthedocs.io/en/latest/user/quickstart/#timeouts
- Resilience patterns: https://learn.microsoft.com/azure/architecture/patterns/
