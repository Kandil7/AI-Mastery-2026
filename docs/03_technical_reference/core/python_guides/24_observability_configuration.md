# Python Configuration for Observability

## Introduction

This guide explains how to configure logging, metrics, and tracing in Python services. It emphasizes consistent configuration and environment-driven settings.

## Learning Objectives

By the end of this guide, you will be able to:
- Centralize logging configuration
- Enable metrics exporters safely
- Use environment variables for observability settings
- Avoid noisy instrumentation

---

## 1. Centralized Logging Config

```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
```

---

## 2. Metrics Configuration

Expose metrics on a dedicated endpoint (Prometheus pattern).

```python
# Example: start metrics server
# from prometheus_client import start_http_server
# start_http_server(8001)
```

---

## 3. Environment Driven Settings

```python
import os

log_level = os.getenv("LOG_LEVEL", "INFO")
metrics_port = int(os.getenv("METRICS_PORT", "8001"))
```

---

## 4. Best Practices

- Avoid high-cardinality labels
- Keep metric names consistent
- Sample tracing if needed

---

## Summary

Observability configuration should be centralized and environment-driven. This keeps logs and metrics consistent across environments.

---

## Additional Resources

- Prometheus client: https://github.com/prometheus/client_python
