# Python Logging in Production

## Introduction

This guide focuses on production logging practices: structured logs, log levels, and correlation IDs. It helps you make logs useful for debugging and monitoring.

## Learning Objectives

By the end of this guide, you will be able to:
- Use log levels correctly
- Emit structured logs
- Include correlation IDs
- Avoid noisy or sensitive logs

---

## 1. Log Levels

```python
import logging

logger = logging.getLogger(__name__)
logger.debug("debug")
logger.info("info")
logger.warning("warning")
logger.error("error")
```

---

## 2. Structured Logs

```python
logger.info("document.uploaded", extra={"doc_id": "123", "size": 1024})
```

---

## 3. Correlation IDs

Pass a request ID through layers and include it in logs.

```python
request_id = "req-123"
logger.info("request.start", extra={"request_id": request_id})
```

---

## 4. Avoid Sensitive Data

Do not log:
- passwords
- tokens
- full document contents

---

## Summary

Production logs should be structured, minimal, and safe. Use levels and correlation IDs to make troubleshooting easier.

---

## Additional Resources

- Python logging: https://docs.python.org/3/library/logging.html
