# Observability: Complete Guide

## Table of Contents
1. [What is Observability?](#what-is-observability)
2. [Why Observability Matters](#why-observability-matters)
3. [The Three Pillars](#the-three-pillars)
4. [Prometheus Metrics](#prometheus-metrics)
5. [Metrics in RAG Pipeline](#metrics-in-rag-pipeline)
6. [OpenTelemetry Tracing](#opentelemetry-tracing)
7. [Structured Logging](#structured-logging)
8. [Alerting & Dashboards](#alerting--dashboards)
9. [Implementation in RAG Engine](#implementation-in-rag-engine)

---

## What is Observability?

**Observability** is the ability to understand a system's internal state by examining its external outputs. Unlike monitoring (which tells you "something is wrong"), observability tells you "why it's wrong".

### Definition
Observability = **Monitoring** + **Context** + **Insight**

**Arabic (العربية)**:
المراقبة (Observability) هي قدرة فهم الحالة الداخلية للنظام من خلال فحص مخرجاته الخارجية. على عكس المراقبة التي تخبرك بوجود مشكلة، المراقبة تخبرك سبب المشكلة.

### Key Concepts

| Concept | Description | Example |
|---------|-------------|---------|
| **Metric** | Numerical measurement over time | Request latency, error rate |
| **Log** | Discrete event record | "User logged in at 10:23" |
| **Trace** | Causal chain of events across services | Request → API → DB → LLM |
| **Span** | Single unit of work in a trace | Embedding generation |

---

## Why Observability Matters

### 1. Debugging Production Issues
```python
# Without observability:
# "Error 500 occurred at 3:45 PM"
# ❌ What happened? Why?

# With observability:
# "Error 500 at 3:45 PM: LLM timeout (30s) after vector search returned 0 results"
# ✅ Clear: Vector search failed → No context → LLM timeout
```

### 2. Performance Optimization
```python
# Identify bottlenecks
- Embedding: 200ms (✅ OK)
- Vector Search: 50ms (✅ OK)
- Reranking: 500ms (❌ SLOW)
- LLM Generation: 1500ms (⚠️ Can optimize)
```

### 3. Cost Management
```python
# Track token usage
Total tokens: 1,000,000
- Prompt tokens: 600,000 ($0.60 @ $1/1M)
- Completion tokens: 400,000 ($4.00 @ $10/1M)
Total cost: $4.60/month
```

### 4. User Experience
```python
# Monitor SLA compliance
P95 latency: 2.3s (Target: <2s) ❌
P99 latency: 5.7s (Target: <5s) ✅
Error rate: 0.8% (Target: <1%) ✅
```

### Arabic
1. **تصحيح مشاكل الإنتاج**: فهم سبب الأخطاء
2. **تحسين الأداء**: تحديد النقاط البطيئة
3. **إدارة التكاليف**: تتبع استخدام الرموز
4. **تجربة المستخدم**: مراقبة الامتثال لـ SLA

---

## The Three Pillars

### Pillar 1: Metrics
**What**: Numerical data over time
**Examples**: Latency, throughput, error rate

```python
from prometheus_client import Counter, Histogram

# Counter: Monotonically increasing
REQUEST_COUNT = Counter("requests_total", "Total requests")

# Histogram: Distribution of values
LATENCY = Histogram("request_duration_seconds", "Request latency")

# Use them
REQUEST_COUNT.inc()  # Increment counter
LATENCY.observe(0.5)  # Record observation
```

### Pillar 2: Logs
**What**: Structured event records
**Examples**: "User logged in", "API error 500"

```python
import structlog

logger = structlog.get_logger()
logger.info("user_login", user_id="123", ip="192.168.1.1")
```

### Pillar 3: Traces
**What**: Causal chain of operations
**Examples**: Request flow across services

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)
with tracer.start_as_current_span("rag_pipeline"):
    # All work here is traced
    pass
```

### Arabic
1. **المقاييس (Metrics)**: بيانات رقمية زمنية
2. **السجلات (Logs)**: سجلات أحداث منظمة
3. **التتبع (Traces)**: سلسلة سببية للعمليات

---

## Prometheus Metrics

### Metric Types

#### 1. Counter
**Monotonically increasing** - never decreases

```python
from prometheus_client import Counter

TOKENS_USED = Counter(
    "llm_tokens_total",
    "Total tokens consumed by LLM",
    ["model", "type"]  # Labels for dimensions
)

# Usage
TOKENS_USED.labels(model="gpt-4", type="prompt").inc(150)
TOKENS_USED.labels(model="gpt-4", type="completion").inc(300)
```

**Use for**: Request counts, token usage, cache hits

---

#### 2. Gauge
**Can go up or down** - current value

```python
from prometheus_client import Gauge

ACTIVE_CONNECTIONS = Gauge(
    "db_connections_active",
    "Currently active database connections"
)

# Usage
ACTIVE_CONNECTIONS.set(15)  # Set to 15
ACTIVE_CONNECTIONS.inc()    # Increment to 16
ACTIVE_CONNECTIONS.dec()    # Decrement to 15
```

**Use for**: Queue sizes, memory usage, active connections

---

#### 3. Histogram
**Distribution** - counts observations in buckets

```python
from prometheus_client import Histogram

REQUEST_LATENCY = Histogram(
    "api_request_duration_seconds",
    "Request latency",
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

# Usage
REQUEST_LATENCY.observe(0.7)  # 0.7s request
```

**Automatic metrics created**:
- `api_request_duration_seconds_bucket`: Count per bucket
- `api_request_duration_seconds_sum`: Sum of all observations
- `api_request_duration_seconds_count`: Total count

**Use for**: Latencies, request sizes, response times

---

#### 4. Summary
**Similar to histogram** - quantile-based

```python
from prometheus_client import Summary

LATENCY_SUMMARY = Summary(
    "api_latency_summary",
    "Request latency summary"
)

# Usage
LATENCY_SUMMARY.observe(0.7)
```

**Automatic metrics created**:
- `api_latency_summary_sum`
- `api_latency_summary_count`
- `api_latency_summary` (quantiles: 0.5, 0.9, 0.99)

---

### Metric Naming

**Rules**:
- Use `snake_case`
- Include units (seconds, bytes, total)
- Prefix with domain (e.g., `rag_`)

```python
# ✅ Good
rag_llm_tokens_total
rag_api_request_duration_seconds
rag_cache_hit_ratio

# ❌ Bad
llmTokens
requestTime
cacheHit
```

---

### Labels

**Tags** that add dimensions to metrics:

```python
CACHE_HITS = Counter(
    "cache_hits_total",
    "Total cache hits",
    ["result", "tier"]  # Labels
)

# Different combinations create separate series
CACHE_HITS.labels(result="hit", tier="redis").inc()
CACHE_HITS.labels(result="miss", tier="redis").inc()
CACHE_HITS.labels(result="hit", tier="memory").inc()
```

**Best practices**:
- Keep cardinality low (< 100 unique values per label)
- Avoid high-cardinality labels (user_id, request_id)

```python
# ❌ Bad: High cardinality
CACHE_HITS.labels(user_id="12345", result="hit").inc()

# ✅ Good: Low cardinality
CACHE_HITS.labels(tier="redis", result="hit").inc()
```

---

## Metrics in RAG Pipeline

### Architecture Overview

```
User Request
    │
    ├─► API Layer (track request count, latency)
    │
    ├─► Embedding Generation (track cache hit/miss)
    │
    ├─► Vector Search (track retrieval scores, time)
    │
    ├─► Reranking (track time, score distribution)
    │
    └─► LLM Generation (track token usage, time)
```

---

### Key Metrics to Track

#### 1. API Metrics
```python
from prometheus_client import Counter, Histogram

API_REQUEST_COUNT = Counter(
    "rag_api_requests_total",
    "Total API requests",
    ["method", "endpoint", "status"]
)

API_LATENCY = Histogram(
    "rag_api_request_duration_seconds",
    "API request latency",
    ["method", "endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
)
```

#### 2. Retrieval Metrics
```python
RETRIEVAL_SCORE = Histogram(
    "rag_retrieval_score",
    "Retrieval similarity scores",
    ["method"],  # vector, keyword, hybrid
    buckets=[0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]
)

RETRIEVAL_TIME = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval operation time",
    ["operation"],  # embed, vector_search, keyword_search, rerank
    buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
)
```

#### 3. LLM Metrics
```python
TOKEN_USAGE = Counter(
    "rag_llm_tokens_total",
    "Total tokens consumed",
    ["model", "type"]  # prompt, completion
)

LLM_LATENCY = Histogram(
    "rag_llm_generation_duration_seconds",
    "LLM generation time",
    ["model"],
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
)
```

#### 4. Cache Metrics
```python
CACHE_HIT = Counter(
    "rag_cache_operations_total",
    "Cache operations",
    ["cache_type", "result"]  # embedding, hit/miss
)

CACHE_LATENCY = Histogram(
    "rag_cache_duration_seconds",
    "Cache operation time",
    ["cache_type", "operation"],
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1]
)
```

---

### Instrumentation Examples

#### Decorator-Based Approach

```python
from functools import wraps
import time

def track_latency(metric: Histogram, labels: dict):
    """Decorator to track function latency."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                metric.labels(**labels).observe(time.time() - start)
                return result
            except Exception:
                metric.labels(**labels).observe(time.time() - start)
                raise
        return wrapper
    return decorator

# Usage
@track_latency(
    API_LATENCY,
    labels={"method": "POST", "endpoint": "/ask"}
)
def ask_question(request):
    # Implementation
    pass
```

---

#### Manual Instrumentation

```python
def ask_question(request):
    # Track request count
    API_REQUEST_COUNT.labels(
        method="POST",
        endpoint="/ask",
        status="200"
    ).inc()
    
    start_time = time.time()
    
    # Embedding
    embedding_start = time.time()
    embedding = get_embedding(request.question)
    EMBEDDING_TIME.observe(time.time() - embedding_start)
    
    # Track cache
    if hasattr(embedding, '_was_cached'):
        CACHE_HIT.labels(cache_type="embedding", result="hit").inc()
    else:
        CACHE_HIT.labels(cache_type="embedding", result="miss").inc()
    
    # Vector search
    vector_start = time.time()
    results = vector_search(embedding)
    VECTOR_SEARCH_TIME.observe(time.time() - vector_start)
    
    # Track scores
    for result in results:
        RETRIEVAL_SCORE.labels(method="vector").observe(result.score)
    
    # LLM generation
    llm_start = time.time()
    answer = llm.generate(results)
    LLM_LATENCY.observe(time.time() - llm_start)
    
    # Track tokens
    TOKEN_USAGE.labels(model="gpt-4", type="prompt").inc(len(results) * 100)
    TOKEN_USAGE.labels(model="gpt-4", type="completion").inc(len(answer) // 4)
    
    # Track overall latency
    API_LATENCY.labels(method="POST", endpoint="/ask").observe(time.time() - start_time)
    
    return answer
```

---

## OpenTelemetry Tracing

### What is Distributed Tracing?

Tracing shows the **entire journey** of a request through multiple services:

```
┌─────────────────────────────────────────────────────────┐
│ User Request                                             │
│ ┌─────────────┐  ┌─────────────┐  ┌─────────────┐       │
│ │ API         │  │ Embedding   │  │ Vector      │       │
│ │ (50ms)      │─►│ Service     │─►│ Store       │       │
│ └─────────────┘  │ (200ms)     │  │ (50ms)      │       │
│                 └─────────────┘  └─────────────┘       │
│                        │                │               │
│                        ▼                ▼               │
│                 ┌─────────────┐  ┌─────────────┐       │
│                 │ LLM         │  │ Reranker    │       │
│                 │ (1500ms)    │  │ (500ms)     │       │
│                 └─────────────┘  └─────────────┘       │
└─────────────────────────────────────────────────────────┘
Total: 2300ms
```

### Basic Tracing with OpenTelemetry

```python
from opentelemetry import trace
from opentelemetry.instrumentation.flask import FlaskInstrumentor

# Setup tracer
tracer = trace.get_tracer(__name__)

# Instrument Flask
app = Flask(__name__)
FlaskInstrumentor().instrument_app(app)

@app.route("/ask")
def ask():
    with tracer.start_as_current_span("rag_pipeline"):
        with tracer.start_as_current_span("embedding"):
            embedding = get_embedding(request.question)
        
        with tracer.start_as_current_span("vector_search"):
            results = vector_search(embedding)
        
        with tracer.start_as_current_span("llm_generation"):
            answer = llm.generate(results)
        
        return {"answer": answer}
```

### Span Attributes

```python
with tracer.start_as_current_span("vector_search") as span:
    span.set_attribute("tenant_id", request.tenant_id)
    span.set_attribute("k", request.k)
    span.set_attribute("results_count", len(results))
    span.set_attribute("top_score", results[0].score)
    
    results = vector_search(embedding)
```

### Span Events

```python
with tracer.start_as_current_span("llm_generation") as span:
    span.add_event("llm_call_started", {"model": "gpt-4"})
    
    answer = llm.generate(results)
    
    span.add_event("llm_call_completed", {
        "prompt_tokens": 500,
        "completion_tokens": 300,
        "total_time_ms": 1500
    })
```

---

## Structured Logging

### Why Structured Logging?

```python
# ❌ Unstructured (hard to parse)
print(f"User {user_id} logged in from {ip}")

# ✅ Structured (easy to query)
logger.info("user_login", user_id=user_id, ip=ip)
```

### Using Structlog

```python
import structlog

# Configure
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)

logger = structlog.get_logger()

# Usage
logger.info(
    "rag_request",
    tenant_id="tenant_123",
    question="What is RAG?",
    retrieval_count=8,
    generation_time_ms=1500,
    answer_length=250
)
```

### Log Levels

| Level | Usage | Example |
|-------|-------|---------|
| `DEBUG` | Detailed diagnostics | "Cache key: abc123" |
| `INFO` | Normal operations | "Request processed" |
| `WARNING` | Unexpected but OK | "Cache miss, fallback used" |
| `ERROR` | Errors that don't crash | "LLM timeout, retrying" |
| `CRITICAL` | System-failing errors | "Database connection lost" |

---

## Alerting & Dashboards

### Prometheus Alert Rules

```yaml
# alerts.yml
groups:
  - name: rag_alerts
    interval: 30s
    rules:
      # High error rate
      - alert: HighErrorRate
        expr: rate(rag_api_requests_total{status=~"5.."}[5m]) > 0.05
        for: 5m
        annotations:
          summary: "Error rate > 5% for 5 minutes"
      
      # Slow responses
      - alert: HighLatency
        expr: histogram_quantile(0.95, rate(rag_api_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        annotations:
          summary: "P95 latency > 2s for 5 minutes"
      
      # Low cache hit rate
      - alert: LowCacheHitRate
        expr: rate(rag_cache_operations_total{result="hit"}[5m]) / rate(rag_cache_operations_total[5m]) < 0.5
        for: 10m
        annotations:
          summary: "Cache hit rate < 50%"
```

### Grafana Dashboard

**Key Panels**:

1. **Request Rate**
   ```promql
   rate(rag_api_requests_total[5m])
   ```

2. **P95 Latency**
   ```promql
   histogram_quantile(0.95, rate(rag_api_request_duration_seconds_bucket[5m]))
   ```

3. **Error Rate**
   ```promql
   rate(rag_api_requests_total{status=~"5.."}[5m]) / rate(rag_api_requests_total[5m])
   ```

4. **Token Usage**
   ```promql
   rate(rag_llm_tokens_total[1h])
   ```

5. **Cache Hit Ratio**
   ```promql
   rate(rag_cache_operations_total{result="hit"}[5m]) / rate(rag_cache_operations_total[5m])
   ```

---

## Implementation in RAG Engine

### File Structure

```
src/core/
├── observability.py          # Prometheus metrics & decorators
└── tracing.py                # OpenTelemetry setup

src/application/use_cases/
└── ask_question_hybrid.py    # Instrumented with metrics

src/api/
└── middleware/
    ├── metrics.py            # Request tracking middleware
    └── logging.py            # Structured logging middleware
```

### Key Metrics Tracked

1. **API Layer**: Request count, latency, status codes
2. **Embedding**: Cache hit/miss, embedding time
3. **Retrieval**: Scores (vector/keyword), search time
4. **Reranking**: Rerank time, score distribution
5. **LLM**: Token usage, generation time

---

## Summary

| Concept | Tool | Use Case |
|---------|------|----------|
| **Metrics** | Prometheus | Numerical data over time |
| **Logs** | Structlog | Structured event records |
| **Traces** | OpenTelemetry | Request flow across services |
| **Alerts** | Alertmanager | Automated notifications |
| **Dashboards** | Grafana | Visualization |

### Key Takeaways

1. **Metrics tell you "what happened"** - counters, gauges, histograms
2. **Traces tell you "how it happened"** - causal chains
3. **Logs tell you "why it happened"** - detailed context
4. **Alerts prevent problems** - proactive monitoring
5. **Dashboards provide visibility** - at-a-glance health

---

## Further Reading

- [Prometheus Best Practices](https://prometheus.io/docs/practices/)
- [OpenTelemetry Python](https://opentelemetry.io/docs/instrumentation/python/)
- [Structlog Documentation](https://www.structlog.org/)
- [Grafana Dashboards](https://grafana.com/grafana/dashboards/)
- `notebooks/learning/04-observability/metrics-basics.ipynb` - Interactive notebook
