# Monitoring & Error Tracking: Complete Guide

## Table of Contents
1. [Structured Logging](#structured-logging)
2. [Prometheus Dashboards](#prometheus-dashboards)
3. [Alerting Rules](#alerting-rules)
4. [Error Tracking with Sentry](#error-tracking-with-sentry)
5. [Log Aggregation with Loki](#log-aggregation-with-loki)

---

## Structured Logging

### What is Structured Logging?

**Structured logging** outputs logs as machine-readable key-value pairs (usually JSON). Unlike traditional string logs, structured logs are:

- **Queryable**: Filter by field values
- **Parseable**: No regex parsing needed
- **Indexable**: Fast search at scale

### Traditional vs Structured

```python
# ❌ Traditional: Hard to query
print(f"User {user_id} logged in at {timestamp} from {ip}")

# ✅ Structured: Easy to query
logger.info(
    "user_login",
    user_id=user_id,
    timestamp=timestamp,
    ip=ip,
)
```

### Query Examples

```sql
-- Traditional: Need regex
SELECT * FROM logs WHERE message LIKE 'User % logged in'

-- Structured: Simple filter
SELECT * FROM logs WHERE event = 'user_login' AND user_id = '123'
```

### Arabic
**التسجيل المنظم**: إخراج السجلات كأزواج قيمة-مفتاح قابلة للقراءة آلياً

---

## Structlog Configuration

### Basic Setup

```python
import structlog

# Configure for production
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.JSONRenderer(),  # JSON output
    ],
    context_class=dict,
)

logger = structlog.get_logger(__name__)
```

### Adding Context

```python
# Global context (all logs)
logger = logger.bind(user_id="123", tenant_id="456")

# Local context (single log)
logger.info(
    "processing_request",
    path="/api/v1/ask",
    method="POST",
)
```

### Context Variables (Distributed Tracing)

```python
from contextvars import ContextVar

REQUEST_ID: ContextVar[str] = ContextVar("request_id")

# Set in middleware
request_id = REQUEST_ID.set(generate_id())

# Use in logging
logger.info(
    "api_request",
    request_id=REQUEST_ID.get(),
)
```

---

## Prometheus Dashboards

### Dashboard Architecture

```
┌─────────────────────────────────────────────────────────┐
│ RAG Engine Dashboard                                │
├─────────────────────────────────────────────────────────┤
│ Request Rate  │  Latency     │  Error Rate       │
│ 120/min       │  P95: 1.2s   │  0.5%           │
├─────────────────────────────────────────────────────────┤
│ Token Usage                                   │
│ ┌─────────────────────────────────────────────┐     │
│ │ Prompt:  600K/hour                      │     │
│ │ Complete: 300K/hour                     │     │
│ │ Total: 900K/hour                        │     │
│ └─────────────────────────────────────────────┘     │
├─────────────────────────────────────────────────────────┤
│ Latency Distribution (P50, P95, P99)              │
│ ┌─────────────────────────────────────────────┐     │
│ │                                         │     │
│ │   P95 ────────                         │     │
│ │   P99 ───────────                       │     │
│ │   P50 ───                               │     │
│ │                                         │     │
│ └─────────────────────────────────────────────┘     │
├─────────────────────────────────────────────────────────┤
│ Retrieval Score Distribution                        │
│ ┌─────────────────────────────────────────────┐     │
│ │ ████░░░░░░░░░░░ (0.7-0.8)            │     │
│ │ ███████████░░░░░ (0.8-0.9)              │     │
│ │ ████████████████░ (0.9-1.0)               │     │
│ └─────────────────────────────────────────────┘     │
├─────────────────────────────────────────────────────────┤
│ Cache Hit Rate                                     │
│ ┌─────────────────────────────────────────────┐     │
│ │ 75% ────────                             │     │
│ │ 50% ────                                 │     │
│ └─────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

### Key Panels

1. **Request Rate**: Requests per minute
   ```promql
   rate(rag_api_requests_total[5m]) * 60
   ```

2. **Latency (P95, P99)**: Response time percentiles
   ```promql
   histogram_quantile(0.95, rate(rag_api_request_duration_seconds_bucket[5m]))
   ```

3. **Error Rate**: Percentage of failed requests
   ```promql
   rate(rag_api_requests_total{status=~"5.."}[5m]) / rate(rag_api_requests_total[5m])
   ```

4. **Token Usage**: LLM token consumption
   ```promql
   rate(rag_llm_tokens_total[5m])
   ```

5. **Cache Hit Rate**: Embedding cache effectiveness
   ```promql
   rate(rag_embedding_cache_total{result="hit"}[5m]) / rate(rag_embedding_cache_total[5m])
   ```

---

## Alerting Rules

### Alert Anatomy

```yaml
- alert: HighErrorRate
  expr: rate(rag_api_requests_total{status=~"5.."}[5m]) > 0.05
  for: 5m
  labels:
    severity: warning
  annotations:
    summary: "High error rate > 5% for 5 minutes"
    description: "Current error rate is {{ $value }}"
```

### Alert Levels

| Severity | Threshold | Action |
|-----------|-----------|---------|
| **warning** | 5% error rate, P95 > 2s | Investigate within 30 min |
| **critical** | 10% error rate, P95 > 5s | Page on-call immediately |

### Alert Groups

```yaml
groups:
  - name: rag_api_alerts       # API-level alerts
  - name: rag_retrieval_alerts  # Retrieval alerts
  - name: rag_cache_alerts       # Cache alerts
  - name: rag_llm_alerts        # LLM alerts
  - name: rag_database_alerts    # Database alerts
```

### Alert Examples

```yaml
# High API error rate
- alert: HighAPIErrorRate
  expr: |
    rate(rag_api_requests_total{status=~"5.."}[5m]) / 
    rate(rag_api_requests_total[5m]) > 0.05
  for: 5m

# Low cache hit rate
- alert: LowCacheHitRate
  expr: |
    rate(rag_embedding_cache_total{result="hit"}[5m]) / 
    rate(rag_embedding_cache_total[5m]) < 0.3
  for: 10m

# High database latency
- alert: HighDBLatency
  expr: |
    histogram_quantile(0.95, rate(pg_stat_statements_duration_seconds_bucket[5m])) > 0.5
  for: 5m
```

---

## Error Tracking with Sentry

### What is Sentry?

**Sentry** is an error tracking platform that:
- Captures exceptions with full stack traces
- Aggregates similar errors (issues)
- Provides release and deployment tracking
- Offers user feedback collection

### Setup

```python
import sentry_sdk

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    environment="production",
    release="rag-engine@1.0.0",
    sample_rate=1.0,  # Capture 100% of errors
    traces_sample_rate=0.1,  # 10% of transactions
)
```

### Capturing Errors

```python
# Automatic (FastAPI integration)
@app.get("/ask")
def ask(question: str):
    result = process_question(question)  # Auto-captured if exception

# Manual
try:
    risky_operation()
except Exception as e:
    sentry_sdk.capture_exception(e)
```

### Adding Context

```python
# User context
sentry_sdk.set_user({
    "id": "user-123",
    "email": "user@example.com",
    "username": "testuser",
})

# Request context
sentry_sdk.set_tag("request_id", "req-456")
sentry_sdk.set_tag("tenant_id", "tenant-789")

# Custom context
sentry_sdk.set_context("rag_pipeline", {
    "question": "What is RAG?",
    "retrieval_count": 8,
    "llm_model": "gpt-4",
})
```

### Breadcrumbs

**Breadcrumbs** show what happened before an error:

```python
# Add breadcrumb (captured if error occurs)
sentry_sdk.add_breadcrumb(
    message="User clicked search button",
    category="ui",
    level="info",
    data={"page": "/search"},
)

sentry_sdk.add_breadcrumb(
    message="API call started",
    category="api",
    level="info",
    data={"endpoint": "/api/v1/ask"},
)
```

### Performance Monitoring

```python
# Track function performance
with sentry_sdk.start_transaction("vector_search", op="search") as transaction:
    results = vector_search(embedding)
    transaction.set_data("results_count", len(results))
    transaction.set_status("ok")
```

---

## Log Aggregation with Loki

### What is Loki?

**Loki** is a horizontally-scalable log aggregation system:
- Labels-based indexing (like Prometheus)
- Efficient storage (no full-text indexing)
- Integration with Grafana

### Log Format

```json
{
  "time": "2024-01-30T10:23:45.123Z",
  "level": "info",
  "message": "user_login",
  "request_id": "req-123",
  "tenant_id": "tenant-456",
  "user_id": "user-789",
  "ip": "192.168.1.1"
}
```

### LogQL Queries

```logql
# Find logs for specific tenant
{tenant_id="tenant-456"}

# Find errors
{level="error"}

# Find slow requests
{path="/api/v1/ask"} | json | duration_ms > 1000

# Filter by multiple labels
{level="error", service="rag_api"}

# Calculate error rate
rate({level="error"}[5m])
```

### Loki + Prometheus + Grafana

```
┌─────────────────────────────────────────────────────┐
│ Grafana (Visualization)                       │
│                                              │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Metrics  │  │  Logs    │  │ Traces │ │
│  │ (Prom)   │  │  (Loki)  │  │(Jaeger)│ │
│  └──────────┘  └──────────┘  └────────┘ │
└─────────────────────────────────────────────────────┘

Query across all three:
"Show me requests with P95 > 2s, errors, and slow traces"
```

---

## Summary

| Component | Purpose | Tool |
|-----------|----------|-------|
| **Structured Logging** | Queryable, parseable logs | Structlog |
| **Metrics** | Numerical data over time | Prometheus |
| **Dashboards** | Visualize metrics | Grafana |
| **Alerting** | Proactive monitoring | Alertmanager |
| **Error Tracking** | Exception tracking | Sentry |
| **Log Aggregation** | Centralize logs | Loki |

### Key Takeaways

1. **Structured logs** are queryable and parseable
2. **Dashboards** provide at-a-glance visibility
3. **Alerts** prevent problems before they impact users
4. **Sentry** captures exceptions with full context
5. **Loki** centralizes logs with label-based queries

---

## Further Reading

- [Structlog Documentation](https://www.structlog.org/)
- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [Grafana Dashboards](https://grafana.com/grafana/dashboards/)
- [Sentry Documentation](https://docs.sentry.io/)
- [Loki Documentation](https://grafana.com/docs/loki/latest/)
