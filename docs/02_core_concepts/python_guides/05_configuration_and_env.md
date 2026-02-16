# Python Configuration and Environment Management

## Introduction

This guide covers practical configuration patterns used in production services: environment variables, .env files, default values, and structured settings. It connects directly to how the RAG Engine Mini loads settings.

## Learning Objectives

By the end of this guide, you will be able to:
- Read configuration from environment variables
- Use .env files safely in development
- Provide defaults and validate required settings
- Structure settings for clarity and reuse
- Avoid common configuration mistakes

---

## 1. Why Configuration Matters

Production systems run in multiple environments:
- Local development
- CI testing
- Staging
- Production

Hardcoding values causes failures when moving between environments. Config should be external and explicit.

---

## 2. Reading Environment Variables

Use `os.getenv` for optional values and `os.environ` for required values.

```python
import os

# Optional with default
api_url = os.getenv("API_URL", "http://localhost:8000")

# Required
try:
    secret_key = os.environ["SECRET_KEY"]
except KeyError as exc:
    raise RuntimeError("Missing SECRET_KEY") from exc
```

---

## 3. .env Files (Development Only)

Use `.env` for local development, never for production secrets.

Example `.env`:

```
API_URL=http://localhost:8000
SECRET_KEY=dev-secret
LOG_LEVEL=INFO
```

Load it with `python-dotenv`:

```python
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env into os.environ
```

---

## 4. Typed Settings Classes

Create a single settings object to keep config organized.

```python
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    api_url: str
    log_level: str


def load_settings() -> Settings:
    return Settings(
        api_url=os.getenv("API_URL", "http://localhost:8000"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
```

---

## 5. Required vs Optional Configuration

Be strict about what is required:

```python
import os

def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required env var: {name}")
    return value

secret = require_env("SECRET_KEY")
```

---

## 6. Configuration for External Services

Typical integrations:
- Database URLs
- File storage credentials
- API keys

Example:

```python
s3_bucket = os.getenv("S3_BUCKET")
if s3_bucket is None:
    raise RuntimeError("Missing S3_BUCKET")
```

Keep these in env vars, not code.

---

## 7. Anti-Patterns

Avoid:
- Hardcoded secrets
- Defaulting sensitive values silently
- Mixing config access all over the codebase
- Logging full secrets

---

## 8. Checklist

Before you commit:
- Required settings are validated
- Optional settings have safe defaults
- No secrets checked into git
- Configuration lives in one module

---

## Summary

Configuration should be explicit, secure, and easy to change. Centralize settings, load environment variables, and validate what the system needs to start.

---

## Additional Resources

- os.environ: https://docs.python.org/3/library/os.html#os.environ
- python-dotenv: https://pypi.org/project/python-dotenv/
