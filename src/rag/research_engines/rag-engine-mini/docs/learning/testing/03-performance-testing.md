# Performance Testing Guide

## Overview

Performance testing is essential for ensuring your RAG Engine can handle production workloads efficiently. This guide covers load testing, stress testing, benchmarking, and capacity planning.

## Table of Contents

1. [Types of Performance Testing](#types-of-performance-testing)
2. [Tools and Setup](#tools-and-setup)
3. [Load Testing with Locust](#load-testing-with-locust)
4. [Benchmarking with pytest](#benchmarking-with-pytest)
5. [Key Metrics](#key-metrics)
6. [Performance Targets](#performance-targets)
7. [Best Practices](#best-practices)

## Types of Performance Testing

### 1. Load Testing
Tests system behavior under expected load conditions.

**Purpose:**
- Validate response times under normal traffic
- Identify bottlenecks
- Verify system stability

**Example Scenarios:**
- 100 concurrent users asking questions
- 1000 search requests per minute
- 50 document uploads per hour

### 2. Stress Testing
Tests system behavior beyond normal capacity.

**Purpose:**
- Find breaking points
- Test recovery mechanisms
- Identify resource limits

**Example Scenarios:**
- Sudden traffic spikes (10x normal load)
- Sustained high load for extended periods
- Resource exhaustion scenarios

### 3. Spike Testing
Tests system reaction to sudden load increases.

**Purpose:**
- Test auto-scaling capabilities
- Verify circuit breakers
- Check rate limiting

### 4. Endurance Testing
Tests system stability over extended periods.

**Purpose:**
- Detect memory leaks
- Monitor resource degradation
- Verify long-term stability

### 5. Benchmark Testing
Provides baseline performance metrics.

**Purpose:**
- Track performance over time
- Compare against SLAs
- Identify regressions

## Tools and Setup

### Required Dependencies

```bash
pip install locust pytest-benchmark aiohttp psutil
```

### Project Structure

```
tests/performance/
├── locustfile.py           # Locust load testing configuration
├── test_load_api.py        # pytest performance tests
├── test_rag_pipeline.py    # RAG-specific performance tests
└── conftest.py            # Performance test fixtures
```

## Load Testing with Locust

### What is Locust?

Locust is an open-source load testing tool that uses Python code to define user behavior. Unlike GUI-based tools, Locust allows you to:
- Define complex user scenarios in Python
- Run distributed load tests across multiple machines
- Monitor results in real-time via web UI
- Export results for analysis

### Basic Locustfile Structure

```python
from locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(1, 5)  # Wait 1-5 seconds between tasks
    
    @task
    def my_task(self):
        self.client.get("/endpoint")
```

### Running Locust

```bash
# Start Locust web interface
locust -f tests/performance/locustfile.py --host=http://localhost:8000

# Headless mode (for CI/CD)
locust -f tests/performance/locustfile.py \
    --host=http://localhost:8000 \
    --headless \
    -u 100 \
    -r 10 \
    --run-time 5m \
    --csv=results
```

### Parameters Explained

- `-u 100`: Number of concurrent users
- `-r 10`: Spawn rate (users per second)
- `--run-time 5m`: Test duration
- `--csv=results`: Export results to CSV

### User Classes

#### Regular User (`RAGAPIUser`)

Simulates typical API usage:
- Authenticates with JWT
- Asks questions (weight: 3)
- Searches documents (weight: 2)
- Lists documents (weight: 2)
- Creates chat sessions (weight: 1)

```python
class RAGAPIUser(HttpUser):
    wait_time = between(1, 5)
    
    @task(3)
    def ask_question(self):
        self.client.post("/api/v1/ask", ...)
```

#### Read-Only User (`RAGReadOnlyUser`)

Simulates monitoring/health checks:
- No authentication required
- High frequency health checks
- Minimal resource usage

#### Heavy User (`RAGHeavyUser`)

Simulates power users:
- Complex multi-step operations
- Bulk operations
- Higher resource consumption

### Custom Load Shapes

Locust supports custom load patterns:

```python
class SpikeLoadShape:
    """Creates traffic spikes every 60 seconds."""
    
    def tick(self):
        run_time = time.time()
        if run_time % 60 < 10:
            return 200, 50  # Spike: 200 users
        else:
            return 50, 10   # Normal: 50 users
```

### Analyzing Locust Results

The Locust web UI provides:
- **Statistics Table**: Request counts, failures, percentiles
- **Charts**: RPS, response times, user count over time
- **Failures**: Detailed error information
- **Download Data**: CSV/JSON export

Key metrics to watch:
- **RPS (Requests Per Second)**: Throughput
- **Median/P95/P99 Response Times**: Latency distribution
- **Failure Rate**: Should be < 1%

## Benchmarking with pytest

### pytest-benchmark

pytest-benchmark provides standardized benchmarking:

```python
import pytest

def test_benchmark_endpoint(benchmark):
    def make_request():
        return requests.get("http://localhost:8000/health")
    
    result = benchmark(make_request)
    assert result.status_code == 200
```

### Running Benchmarks

```bash
# Run all benchmarks
pytest tests/performance/test_load_api.py -v --benchmark-only

# Compare against previous run
pytest tests/performance/ --benchmark-compare

# Save benchmark results
pytest tests/performance/ --benchmark-save=baseline
```

### Benchmark Output

```
----------------------------------- benchmark: 1 tests -----------------------------------
Name (time in ms)          Min       Max      Mean  StdDev    Median     IQR  Outliers
------------------------------------------------------------------------------------------
test_benchmark_ask       245.12   890.34   412.56  123.45   398.12  98.76       2;1
------------------------------------------------------------------------------------------
```

## Key Metrics

### Response Time Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **P50 (Median)** | 50% of requests faster than this | < 500ms |
| **P95** | 95% of requests faster than this | < 2s |
| **P99** | 99% of requests faster than this | < 5s |
| **Max** | Slowest request | < 10s |

### Throughput Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **RPS** | Requests per second | > 100 |
| **QPS** | Queries per second (RAG) | > 50 |
| **Upload Rate** | Documents per minute | > 10 |

### Resource Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **CPU** | Average CPU utilization | < 70% |
| **Memory** | RAM usage | Stable (no growth) |
| **DB Connections** | Active database connections | < 80% of pool |
| **Network I/O** | Bandwidth usage | < 80% capacity |

### Error Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Error Rate** | Percentage of failed requests | < 1% |
| **Timeout Rate** | Requests exceeding timeout | < 0.1% |
| **5xx Rate** | Server errors | < 0.01% |

## Performance Targets

### RAG Engine Specific Targets

#### Health Check Endpoint
- P95 latency: < 100ms
- RPS: > 1000
- Error rate: 0%

#### Search Endpoint
- P95 latency: < 500ms
- RPS: > 200
- Supports pagination up to 100 items

#### Ask Endpoint (Simple Query)
- P95 latency: < 2s
- RPS: > 50
- Success rate: > 99%

#### Ask Endpoint (Hybrid + Rerank)
- P95 latency: < 5s
- RPS: > 20
- Success rate: > 99%

#### Document Upload
- P95 latency: < 3s (per document)
- Throughput: > 10 docs/minute
- Max file size: 50MB

### Scalability Targets

- **Horizontal Scaling**: Support 10+ API replicas
- **Database**: Handle 1M+ documents
- **Concurrent Users**: 1000+ simultaneous users
- **Daily Queries**: 100K+ questions per day

## Best Practices

### 1. Test in Production-Like Environment

```python
# Use production-like data volumes
# Use similar network conditions
# Use production infrastructure (if possible)
```

### 2. Warm Up the System

```python
# Always warm up caches before measuring
def warmup():
    for _ in range(100):
        client.get("/api/v1/ask", ...)
    
    # Wait for caches to stabilize
    time.sleep(5)
```

### 3. Use Realistic Data

```python
# Use varied questions, not the same one
questions = [
    "What is RAG?",
    "Explain vector databases",
    "How does hybrid search work?",
    # ... many more
]

question = random.choice(questions)
```

### 4. Monitor All Layers

- Application metrics
- Database performance
- Vector store latency
- LLM API response times
- Network I/O

### 5. Test Failure Scenarios

```python
# Test with slow LLM responses
# Test with database connection issues
# Test with high error rates
```

### 6. Document and Track

```python
# Save benchmark results
# Track trends over time
# Set up alerts for regressions
```

### 7. Run Regularly

- **CI/CD**: Quick smoke tests on every build
- **Daily**: Automated load tests
- **Weekly**: Full stress tests
- **Monthly**: Capacity planning tests

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Performance Tests

on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM

jobs:
  performance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Start services
        run: docker-compose up -d
      
      - name: Run Locust
        run: |
          locust -f tests/performance/locustfile.py \
            --headless \
            -u 100 \
            -r 10 \
            --run-time 5m \
            --csv=results
      
      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: performance-results
          path: results_*.csv
```

## Common Pitfalls

### 1. Testing from Same Machine

Running load generator and application on same machine skews results.

**Solution**: Use separate machines or containers.

### 2. Not Cleaning Up

Leaving test data affects subsequent tests.

**Solution**: Always clean up in `on_stop()`:

```python
def on_stop(self):
    # Delete test documents
    for doc_id in self.documents:
        self.client.delete(f"/api/v1/documents/{doc_id}")
```

### 3. Ignoring Network Latency

Local testing doesn't account for real network conditions.

**Solution**: Test from different geographic locations.

### 4. Testing Only Happy Path

Only testing successful requests misses real-world issues.

**Solution**: Include error scenarios:

```python
@task
def invalid_request(self):
    self.client.post("/api/v1/ask", json={"invalid": "data"})
```

## Conclusion

Performance testing is not a one-time activity but a continuous practice:

1. **Establish baselines** - Know your normal performance
2. **Set targets** - Define acceptable thresholds
3. **Monitor continuously** - Detect regressions early
4. **Test realistically** - Match production conditions
5. **Optimize iteratively** - Fix biggest bottlenecks first

With proper performance testing, you can ensure your RAG Engine delivers fast, reliable responses even under heavy load.

## Additional Resources

- [Locust Documentation](https://docs.locust.io/)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
- [Google SRE Book - Monitoring](https://sre.google/sre-book/monitoring-distributed-systems/)
- [Performance Testing Best Practices](https://www.guru99.com/performance-testing.html)
