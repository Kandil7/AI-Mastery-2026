# Evaluation & Monitoring Best Practices in RAG Systems

## Table of Contents
1. [Introduction](#introduction)
2. [Evaluation Framework](#evaluation-framework)
3. [Key Metrics for RAG Systems](#key-metrics-for-rag-systems)
4. [Monitoring Architecture](#monitoring-architecture)
5. [Alerting and Anomaly Detection](#alerting-and-anomaly-detection)
6. [Dashboard Design](#dashboard-design)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting Common Issues](#troubleshooting-common-issues)
9. [Best Practices](#best-practices)

---

## Introduction

Evaluation and monitoring are critical components of any production RAG system. Unlike traditional machine learning models, RAG systems have multiple interconnected components that must be monitored individually and collectively. This document outlines best practices for implementing a comprehensive evaluation and monitoring solution.

### Why Monitoring Matters in RAG Systems

- **Complex Interactions**: RAG systems have multiple components (retriever, generator, embeddings) that interact in complex ways
- **Drift Detection**: Both the underlying data and model components can drift over time
- **User Experience**: Metrics must correlate with user satisfaction and business outcomes
- **Cost Management**: Monitoring helps optimize resource usage and manage operational costs

---

## Evaluation Framework

### Components of a RAG Evaluation System

1. **Offline Evaluation**: Performed on static datasets before deployment
2. **Online Evaluation**: Measured during live system operation
3. **Human Evaluation**: Qualitative assessments of system performance
4. **A/B Testing**: Comparing different system configurations

### Implementation in RAG Engine

The RAG Engine Mini implements evaluation through multiple layers:

```python
# From src/workers/tasks.py - Indexing Metrics
CELERY_TASK_COUNT.labels(task="index_document", status="success").inc()
CELERY_TASK_DURATION.labels(task="index_document").observe(time.time() - start_time)
```

```python
# From src/core/observability.py - Custom metrics
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry.instrumentation.grpc import GrpcInstrumentorServer
```

### Synthetic Test Generation

For continuous evaluation, RAG systems should generate synthetic test cases:

```python
# Conceptual implementation
def generate_synthetic_tests(documents, n_tests=100):
    """
    Generate synthetic Q&A pairs from documents for evaluation
    """
    tests = []
    for _ in range(n_tests):
        # Select random document
        doc = random.choice(documents)
        
        # Generate question about content
        question = llm.generate(f"Generate a question about: {doc[:500]}")
        
        # Generate expected answer
        answer = llm.generate(f"Answer this question based on: {doc}\nQuestion: {question}")
        
        tests.append({
            "question": question,
            "context": doc,
            "expected_answer": answer
        })
    
    return tests
```

---

## Key Metrics for RAG Systems

### Retrieval Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Recall@K | `|relevant retrieved| / |all relevant|` | Measure of completeness |
| Precision@K | `|relevant retrieved| / |total retrieved|` | Measure of precision |
| MRR | `1/n * Σ(1/rank_i)` | Ranking quality |
| NDCG | Normalized DCG calculation | Ranked list quality |

### Generation Metrics

| Metric | Description | Use Case |
|--------|-------------|----------|
| BLEU | n-gram overlap with reference | Translation tasks |
| ROUGE | Overlap of n-grams, word sequences | Summarization |
| METEOR | Harmonic mean of precision/recall | General text generation |
| BERTScore | BERT-based similarity | Semantic similarity |

### RAG-Specific Metrics

| Metric | Definition | Calculation |
|--------|------------|-------------|
| Faithfulness | How factually consistent is the answer to the context? | LLM-based evaluation |
| Context Relevance | How relevant is the retrieved context? | Overlap between query and context |
| Answer Relevance | How relevant is the answer to the query? | Semantic similarity measures |
| Groundedness | How much is the answer based on provided context? | Attribution analysis |

### System Metrics

| Category | Metric | Measurement |
|----------|--------|-------------|
| Latency | Query response time | Timer histogram |
| Throughput | Queries per second | Counter rate |
| Resource | CPU, memory usage | Gauge values |
| Error Rate | Failed queries | Error counter |

---

## Monitoring Architecture

### Observability Stack Components

The RAG Engine Mini implements a comprehensive observability stack:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│  Prometheus     │───▶│   Alertmanager  │
│   Instrumentation│    │   Metrics      │    │   Notifications │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ OpenTelemetry   │───▶│   Grafana       │    │   Sentry        │
│   Traces/Logs   │    │   Dashboards    │    │   Error Tracking│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Implementation in Code

```python
# From src/core/observability.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Set up tracing
resource = Resource(attributes={
    SERVICE_NAME: "rag-engine"
})

tracer_provider = TracerProvider(resource=resource)
otel_endpoint = os.getenv("OTEL_ENDPOINT", "http://localhost:4317")
span_processor = BatchSpanProcessor(
    OTLPSpanExporter(endpoint=otel_endpoint, insecure=True)
)
tracer_provider.add_span_processor(span_processor)
trace.set_tracer_provider(tracer_provider)
```

### Custom Metrics

```python
# From src/workers/tasks.py
from src.core.observability import CELERY_TASK_COUNT, CELERY_TASK_DURATION

@celery_app.task(name="index_document", ...)
def index_document(self, *, tenant_id: str, document_id: str):
    start_time = time.time()
    try:
        # ... processing logic ...
        
        CELERY_TASK_COUNT.labels(task="index_document", status="success").inc()
        CELERY_TASK_DURATION.labels(task="index_document").observe(
            time.time() - start_time
        )
        return {"ok": True, ...}
    except Exception as e:
        CELERY_TASK_COUNT.labels(task="index_document", status="failure").inc()
        CELERY_TASK_DURATION.labels(task="index_document").observe(
            time.time() - start_time
        )
        raise
```

---

## Alerting and Anomaly Detection

### Threshold-Based Alerts

| Metric | Warning Threshold | Critical Threshold | Reason |
|--------|-------------------|-------------------|---------|
| P95 Latency | >2s | >5s | User experience impact |
| Error Rate | >1% | >5% | System reliability |
| Success Rate | <95% | <90% | Availability |
| FAITHFULNESS | <0.7 | <0.5 | Quality degradation |

### Anomaly Detection Implementation

```python
# Conceptual implementation
class AnomalyDetector:
    def __init__(self, window_size=100, sensitivity=2.0):
        self.window_size = window_size
        self.sensitivity = sensitivity
        self.history = []
    
    def check_anomaly(self, current_value):
        self.history.append(current_value)
        if len(self.history) > self.window_size:
            self.history.pop(0)
        
        if len(self.history) < 10:  # Need minimum data
            return False
            
        mean = np.mean(self.history[:-1])  # Exclude current value
        std = np.std(self.history[:-1])
        
        if std == 0:  # Prevent division by zero
            return False
            
        z_score = abs(current_value - mean) / std
        return z_score > self.sensitivity
```

### Alert Configuration (Prometheus)

```yaml
# From config/prometheus/alerts.yml
groups:
- name: rag-engine-alerts
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, rag_request_duration_seconds_bucket) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High latency detected"
      description: "P95 latency is above 2s for more than 5 minutes"

  - alert: HighErrorRate
    expr: rate(rag_request_errors_total[5m]) / rate(rag_requests_total[5m]) > 0.01
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is above 1% for more than 2 minutes"
```

---

## Dashboard Design

### Essential Dashboard Panels

1. **System Health Overview**
   - Request rate and success rate
   - Error rate and latency percentiles
   - Active users and concurrent requests

2. **Retrieval Performance**
   - Retrieval latency and recall@k
   - Top queries and their performance
   - Cache hit rates

3. **Generation Quality**
   - Answer relevance scores
   - Faithfulness metrics
   - User satisfaction ratings

4. **Resource Utilization**
   - CPU, memory, disk usage
   - Database connection pools
   - Vector database performance

### Grafana Panel Examples

```json
{
  "title": "RAG System Performance",
  "panels": [
    {
      "title": "Request Rate",
      "type": "graph",
      "targets": [
        {
          "expr": "rate(http_requests_total[5m])",
          "legendFormat": "{{method}} {{route}}"
        }
      ]
    },
    {
      "title": "Latency Percentiles",
      "type": "graph",
      "targets": [
        {
          "expr": "histogram_quantile(0.5, http_request_duration_seconds_bucket)",
          "legendFormat": "p50"
        },
        {
          "expr": "histogram_quantile(0.95, http_request_duration_seconds_bucket)",
          "legendFormat": "p95"
        },
        {
          "expr": "histogram_quantile(0.99, http_request_duration_seconds_bucket)",
          "legendFormat": "p99"
        }
      ]
    },
    {
      "title": "System Resources",
      "type": "timeseries",
      "targets": [
        {
          "expr": "process_cpu_seconds_total",
          "legendFormat": "CPU"
        },
        {
          "expr": "process_resident_memory_bytes",
          "legendFormat": "Memory"
        }
      ]
    }
  ]
}
```

---

## Performance Optimization

### Monitoring for Performance

1. **Identify Bottlenecks**
   - High latency components
   - Resource contention
   - Inefficient queries

2. **Optimize Resource Usage**
   - Adjust caching strategies
   - Optimize embedding batch sizes
   - Tune database connection pools

3. **Scale Responsively**
   - Auto-scaling based on metrics
   - Circuit breaker patterns
   - Load shedding for overload protection

### Example Performance Monitoring

```python
# From src/api/v1/routes_documents.py
@router.post("/upload", ...)
async def upload_document(...):
    start_time = time.time()
    
    try:
        # Track processing time
        result = await use_case.execute(...)
        
        # Record metrics
        UPLOAD_DURATION.observe(time.time() - start_time)
        UPLOAD_COUNT.labels(status="success").inc()
        
        return result
    except Exception as e:
        UPLOAD_COUNT.labels(status="error").inc()
        logger.error(f"Upload failed: {str(e)}")
        raise
```

---

## Troubleshooting Common Issues

### Issue: High Latency
**Symptoms:**
- Slow response times
- P95 latency > threshold

**Diagnosis Steps:**
1. Check component-specific latencies (retrieval vs generation)
2. Monitor resource utilization
3. Review embedding batch sizes
4. Check database/vectordb performance

**Solutions:**
- Optimize retrieval algorithms
- Increase embedding batch sizes
- Improve caching strategies
- Scale infrastructure

### Issue: Quality Degradation
**Symptoms:**
- Decreasing faithfulness scores
- User complaints about accuracy

**Diagnosis Steps:**
1. Review recent document changes
2. Check embedding model performance
3. Analyze retrieval quality
4. Verify prompt engineering

**Solutions:**
- Retrain embeddings if needed
- Adjust retrieval parameters
- Improve prompt engineering
- Revisit document processing

### Issue: High Error Rates
**Symptoms:**
- Increased error counter
- Decreased success rate

**Diagnosis Steps:**
1. Check error logs for patterns
2. Monitor resource exhaustion
3. Review third-party service health
4. Check for traffic spikes

**Solutions:**
- Implement retry logic
- Add circuit breakers
- Scale infrastructure
- Improve error handling

---

## Best Practices

### 1. Implement Comprehensive Observability
- Instrument all critical paths
- Use distributed tracing
- Correlate metrics, logs, and traces
- Establish baseline performance

### 2. Set Meaningful Thresholds
- Define SLIs/SLOs based on user experience
- Set alerts for early warning
- Regularly review and adjust thresholds
- Consider different user segments

### 3. Monitor Data Quality
- Track document ingestion quality
- Monitor for data drift
- Check for content freshness
- Validate retrieval relevance

### 4. Implement Feedback Loops
- Collect user feedback signals
- Track click-through rates
- Monitor query-answer pairs
- Use feedback for model improvement

### 5. Plan for Scalability
- Monitor resource utilization trends
- Set up predictive scaling
- Implement load shedding
- Design for graceful degradation

### 6. Secure Your Monitoring
- Protect monitoring endpoints
- Encrypt sensitive data
- Implement access controls
- Audit monitoring access

---

This comprehensive guide covers the essential aspects of evaluating and monitoring RAG systems. Proper implementation of these practices ensures reliable, high-quality RAG system operation in production environments.