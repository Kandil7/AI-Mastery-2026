# OpenTelemetry Tracing: Complete Guide

## Table of Contents
1. [What is Distributed Tracing?](#what-is-distributed-tracing)
2. [Why Tracing Matters](#why-tracing-matters)
3. [OpenTelemetry Basics](#opentelemetry-basics)
4. [Tracing Concepts](#tracing-concepts)
5. [Instrumentation Patterns](#instrumentation-patterns)
6. [RAG Pipeline Tracing](#rag-pipeline-tracing)
7. [Analyzing Traces](#analyzing-traces)
8. [Best Practices](#best-practices)

---

## What is Distributed Tracing?

**Distributed tracing** is a method of tracking requests as they flow through distributed systems. It provides visibility into the entire lifecycle of a request across multiple services.

### Definition
Distributed tracing = **Request tracking** + **Causal relationships** + **Cross-service context**

**Arabic (العربية)**:
التتبع الموزع هو طريقة لتتبع الطلبات أثناء تدفقها عبر الأنظمة الموزعة. يوفر رؤية شاملة لدورة حياة الطلب عبر خدمات متعددة.

### Visual Example

```
Client Request
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ API Gateway (50ms)                                        │
│ └─► Authentication Service (20ms)                          │
│ └─► Rate Limiting (5ms)                                   │
└────────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│ RAG Service (2000ms)                                       │
│ └─► Embedding Service (200ms)                             │
│ │   └─► Cache Check (5ms)                                 │
│ │   └─► OpenAI API (195ms)                                │
│ └─► Vector Store (50ms)                                   │
│ └─► Reranking Service (500ms)                             │
│ └─► LLM Service (1250ms)                                  │
└────────────────────────────────────────────────────────────┘
    │
    ▼
Response
```

### Key Components

| Component | Description | Example |
|-----------|-------------|---------|
| **Trace** | Complete journey of a request | User login → API → DB |
| **Span** | Single unit of work | DB query, HTTP call |
| **Trace ID** | Unique identifier for entire trace | `abc123...` |
| **Span ID** | Unique identifier for single span | `def456...` |
| **Parent ID** | Links child spans to parent | Links DB query to API call |

---

## Why Tracing Matters

### 1. Debugging Production Issues

```python
# Without tracing:
# "Error: Request timeout after 30s"
# ❌ Where did it timeout? Which service?

# With tracing:
# "Error: Request timeout after 30s"
# Trace shows:
#   - API Gateway: 50ms (✅ OK)
#   - Embedding Service: 200ms (✅ OK)
#   - Vector Search: 50ms (✅ OK)
#   - Reranking: 500ms (✅ OK)
#   - LLM Generation: 27.5s (❌ TIMEOUT)
# ✅ LLM is the bottleneck - investigate LLM API
```

### 2. Performance Optimization

```python
# Analyze trace waterfall
Total: 2000ms
- Embedding: 200ms (10%)
- Vector Search: 50ms (2.5%)
- Reranking: 500ms (25%)
- LLM Generation: 1250ms (62.5%)

Optimization opportunities:
1. Reranking: 500ms → 250ms (cross-encoder → bi-encoder)
2. LLM: 1250ms → 1000ms (cache, streaming)
Potential savings: 750ms (37.5%)
```

### 3. Service Dependency Mapping

```python
# Trace data shows:
RAG Service depends on:
- Embedding Service (99.9% of requests)
- Vector Store (99.9% of requests)
- Reranking Service (95% of requests)
- LLM Service (100% of requests)

Impact analysis:
- If Vector Store is down → All requests fail
- If Reranking is slow → 95% of requests affected
```

### 4. Root Cause Analysis

```python
# Error trace:
RAG Pipeline
└─► LLM Generation
    └─► HTTP POST /v1/completions
        └─► Error: 429 Rate Limit Exceeded

Root cause: LLM API rate limit exceeded
Solution: Implement caching, reduce concurrent requests
```

### Arabic
1. **تصحيح مشاكل الإنتاج**: تحديد مكان حدوث المهلة
2. **تحسين الأداء**: تحديد فرص التحسين
3. **تعيين التبعيات**: فهم اعتماديات الخدمة
4. **تحليل السبب الجذري**: إصلاح المشاكل في المصدر

---

## OpenTelemetry Basics

### What is OpenTelemetry?

OpenTelemetry (OTel) is a **vendor-neutral** standard for observability data (metrics, logs, traces). It provides:
- Automatic instrumentation libraries
- Manual instrumentation APIs
- Vendor-agnostic exporters

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Vendor-neutral** | Works with Jaeger, Zipkin, Grafana Tempo, etc. |
| **Automatic** | Auto-instruments popular libraries (FastAPI, SQLAlchemy) |
| **Standardized** | Same API across languages (Python, Go, Java, etc.) |
| **Extensible** | Custom exporters, processors, samplers |

### Installation

```bash
pip install opentelemetry-api
pip install opentelemetry-sdk
pip install opentelemetry-exporter-jaeger
pip install opentelemetry-exporter-otlp
pip install opentelemetry-instrumentation-fastapi
pip install opentelemetry-instrumentation-sqlalchemy
```

---

## Tracing Concepts

### Trace Hierarchy

```
Trace (root)
├─► Span 1 (parent)
│   ├─► Span 2 (child of 1)
│   │   └─► Span 3 (child of 2)
│   └─► Span 4 (child of 1)
└─► Span 5 (sibling of 1)
```

### Span Lifecycle

```python
# 1. Create span
span = tracer.start_span("operation_name")

# 2. Add attributes (metadata)
span.set_attribute("user_id", "123")

# 3. Add events (timestamps)
span.add_event("cache_hit", {"key": "abc123"})

# 4. Set status
span.set_status(StatusCode.OK)

# 5. End span
span.end()
```

### Context Propagation

```python
# Automatically handled by OTel
# Headers added to HTTP requests:
# Traceparent: 00-abc123-def456-01
# Tracestate: key1=value1,key2=value2
```

---

## Instrumentation Patterns

### 1. Manual Instrumentation

```python
from opentelemetry import trace

tracer = trace.get_tracer(__name__)

def process_request(request):
    with tracer.start_as_current_span("process_request") as span:
        span.set_attribute("request_id", request.id)
        
        # Do work
        result = do_something()
        
        span.set_attribute("result_count", len(result))
        return result
```

### 2. Decorator-Based

```python
from functools import wraps

def traced(operation_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with tracer.start_as_current_span(operation_name) as span:
                span.set_attribute("function", func.__name__)
                try:
                    result = func(*args, **kwargs)
                    span.set_status(StatusCode.OK)
                    return result
                except Exception as e:
                    span.set_status(StatusCode.ERROR, str(e))
                    span.record_exception(e)
                    raise
        return wrapper
    return decorator

# Usage
@traced("embedding_generation")
def generate_embedding(text):
    # Implementation
    pass
```

### 3. Context Manager

```python
class TraceContext:
    def __init__(self, span_name: str):
        self.span_name = span_name
        self.span = None
    
    def __enter__(self):
        self.span = tracer.start_as_current_span(self.span_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.span.set_status(StatusCode.ERROR, str(exc_val))
            self.span.record_exception(exc_val)
        else:
            self.span.set_status(StatusCode.OK)
        self.span.end()

# Usage
with TraceContext("vector_search"):
    results = vector_search(embedding)
```

### 4. Automatic Instrumentation

```python
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Automatically traces all FastAPI endpoints
app = FastAPI()
FastAPIInstrumentor.instrument_app(app)

@app.get("/ask")
def ask(question: str):
    # This is automatically traced
    return {"answer": "..."}

# Traces include:
# - Request method, path, status code
# - Request/response timing
# - Error handling
```

---

## RAG Pipeline Tracing

### Full Pipeline Trace

```python
class RAGTracer:
    """Specialized tracer for RAG operations."""
    
    def trace_pipeline(self, tenant_id: str, question: str):
        """Trace complete RAG pipeline."""
        with tracer.start_as_current_span("rag_pipeline") as span:
            span.set_attribute("tenant_id", tenant_id)
            span.set_attribute("question", question)
            
            # Sub-spans
            embedding = self.trace_embedding(question)
            vector_results = self.trace_vector_search(embedding)
            reranked = self.trace_rerank(vector_results)
            answer = self.trace_llm_generation(question, reranked)
            
            return answer
    
    def trace_embedding(self, text: str):
        """Trace embedding generation."""
        with tracer.start_as_current_span("embedding_generation") as span:
            span.set_attribute("text_length", len(text))
            span.set_attribute("model", "text-embedding-ada-002")
            
            # Sub-span: cache check
            with tracer.start_as_current_span("cache_check"):
                cached = check_cache(text)
                span.set_attribute("cached", cached)
            
            if not cached:
                with tracer.start_as_current_span("openai_api_call"):
                    embedding = call_openai(text)
                    save_cache(text, embedding)
            
            return embedding
```

### Span Attributes

```python
# Standard attributes
span.set_attribute("user.id", "123")
span.set_attribute("http.method", "POST")
span.set_attribute("http.status_code", 200)

# Custom attributes for RAG
span.set_attribute("tenant_id", "tenant_123")
span.set_attribute("question", "What is RAG?")
span.set_attribute("k", 10)
span.set_attribute("results_count", 8)
span.set_attribute("top_score", 0.95)
span.set_attribute("model", "gpt-4")
span.set_attribute("prompt_tokens", 500)
span.set_attribute("completion_tokens", 300)
```

### Span Events

```python
# Timestamped events within a span
with tracer.start_as_current_span("vector_search") as span:
    span.add_event("search_started", {"k": 10, "tenant": "123"})
    
    results = vector_search(embedding)
    
    span.add_event("search_completed", {
        "results_count": len(results),
        "top_score": results[0].score
    })
    
    span.add_event("cache_hit", {"key": "embedding:123"})
```

### Error Handling

```python
with tracer.start_as_current_span("llm_generation") as span:
    try:
        answer = llm.generate(prompt)
        span.set_status(StatusCode.OK)
    except RateLimitError as e:
        span.set_status(StatusCode.ERROR, "Rate limit exceeded")
        span.set_attribute("error.type", "RateLimitError")
        span.record_exception(e)
        raise
```

---

## Analyzing Traces

### Jaeger UI

```bash
# Start Jaeger locally
docker run -d -p 16686:16686 -p 14268:14268 jaegertracing/all-in-one:latest

# View traces at: http://localhost:16686
```

### Querying Traces

```
# Find slow requests
Operation: rag_pipeline
Duration: > 5000ms

# Find errors
Operation: *
Status: ERROR

# Find specific tenant
Tags: tenant_id = "tenant_123"

# Find cache misses
Operation: cache_check
Tags: cached = "false"
```

### Trace Analysis

```python
# Analyze trace data
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

class TraceAnalyzer:
    def __init__(self):
        self.spans = []
    
    def on_end(self, span):
        self.spans.append(span)
    
    def analyze(self):
        # Find slowest operation
        slowest = max(self.spans, key=lambda s: s.end_time - s.start_time)
        
        # Find errors
        errors = [s for s in self.spans if s.status.is_error]
        
        # Calculate averages
        avg_latency = sum(s.end_time - s.start_time for s in self.spans) / len(self.spans)
        
        return {
            "slowest": slowest.name,
            "errors": len(errors),
            "avg_latency_ms": avg_latency * 1000,
        }
```

---

## Best Practices

### 1. Sampling

```python
# Don't trace 100% of requests (too expensive)
# Use sampling instead

from opentelemetry.sdk.trace import sampling

# Sample 10% of traces
sampler = sampling.TraceIdRatioBased(0.1)

provider = TracerProvider(sampler=sampler)
```

### 2. Span Naming

```python
# ✅ Good: Specific, actionable
"vector_search"
"embedding_generation"
"llm_api_call"

# ❌ Bad: Vague
"search"
"generate"
"call"
```

### 3. Attribute Cardinality

```python
# ✅ Good: Low cardinality
span.set_attribute("model", "gpt-4")  # Few values
span.set_attribute("cache_tier", "redis")

# ❌ Bad: High cardinality (exceeds limits)
span.set_attribute("user_id", "12345")  # Too many unique values
span.set_attribute("request_id", "abc-def-ghi")
```

### 4. Span Duration

```python
# Don't create spans for < 1ms operations
# They add overhead without value

# ✅ Good: Meaningful duration
with tracer.start_as_current_span("llm_generation"):  # 1500ms
    answer = llm.generate(prompt)

# ❌ Bad: Too granular
with tracer.start_as_current_span("string_concat"):  # 0.1ms
    text = text1 + text2
```

### 5. Error Context

```python
# Always add context to errors
with tracer.start_as_current_span("operation") as span:
    try:
        result = risky_operation()
    except Exception as e:
        span.set_attribute("error.message", str(e))
        span.set_attribute("error.type", type(e).__name__)
        span.set_attribute("error.stack_trace", traceback.format_exc())
        span.record_exception(e)
        raise
```

---

## Summary

| Concept | Description | Example |
|---------|-------------|---------|
| **Trace** | Complete request journey | User → API → DB → LLM |
| **Span** | Single unit of work | DB query, HTTP call |
| **Trace ID** | Unique identifier for trace | `abc123...` |
| **Span ID** | Unique identifier for span | `def456...` |
| **Sampling** | Trace percentage of requests | 10% of requests |
| **Exporters** | Send traces to backend | Jaeger, Grafana Tempo |

### Key Takeaways

1. **Tracing provides end-to-end visibility** across services
2. **Use automatic instrumentation** where possible (FastAPI, SQLAlchemy)
3. **Manual instrumentation** adds business context
4. **Sampling reduces overhead** while maintaining value
5. **Analyze traces** to optimize performance and debug issues

---

## Further Reading

- [OpenTelemetry Python Documentation](https://opentelemetry.io/docs/instrumentation/python/)
- [Jaeger Documentation](https://www.jaegertracing.io/docs/)
- [Distributed Tracing Best Practices](https://www.lightstep.com/blog/distributed-tracing-best-practices/)
- `notebooks/learning/04-observability/tracing-basics.ipynb` - Interactive notebook
