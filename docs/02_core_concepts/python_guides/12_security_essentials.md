# Python Security Essentials

## Introduction

This guide covers practical security basics for Python services: input validation, secrets handling, safe logging, and dependency hygiene. It aligns with common risks in API and ingestion systems.

## Learning Objectives

By the end of this guide, you will be able to:
- Validate inputs at boundaries
- Avoid logging secrets and PII
- Handle secrets safely
- Understand common injection risks
- Keep dependencies secure

---

## 1. Validate Inputs

Always validate user input at the API boundary.

```python
from pydantic import BaseModel

class DocumentCreate(BaseModel):
    title: str
    content: str
```

---

## 2. Avoid Logging Secrets

```python
import logging

logger = logging.getLogger(__name__)
logger.info("user.login", extra={"user_id": "123"})
# Do not log passwords, tokens, or full documents.
```

---

## 3. Secrets Handling

- Use environment variables
- Never commit secrets to git
- Rotate keys regularly

```python
import os

api_key = os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("Missing API_KEY")
```

---

## 4. Injection Risks

Common injection vectors:
- SQL injection
- Command injection
- Path traversal

Avoid string concatenation for queries.

---

## 5. Dependency Hygiene

- Pin dependencies
- Run security scanners
- Update regularly

---

## 6. Checklist

Before you commit:
- Inputs validated with schema
- Logs are safe
- Secrets only in env vars
- Dependencies pinned

---

## Summary

Security is about safe defaults. Validate inputs, protect secrets, and keep dependencies current.

---

## Additional Resources

- OWASP Top 10: https://owasp.org/www-project-top-ten/
- Python security: https://docs.python.org/3/library/security_warnings.html
