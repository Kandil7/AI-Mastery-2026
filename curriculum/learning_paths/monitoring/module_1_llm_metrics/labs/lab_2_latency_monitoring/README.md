# Lab 2: Latency Monitoring with SLOs

## 🎯 Lab Overview

**Duration:** 2-3 hours  
**Difficulty:** Intermediate  
**Prerequisites:** Lab 1 completion, understanding of percentiles

In this lab, you will implement comprehensive latency monitoring for LLM applications, including time-to-first-token (TTFT) tracking and SLO-based alerting.

---

## 📋 Learning Objectives

After completing this lab, you will be able to:

1. **Implement** latency histogram metrics for LLM requests
2. **Calculate** percentile latencies (p50, p95, p99) using PromQL
3. **Define** and track Service Level Objectives (SLOs)
4. **Create** latency-focused Grafana dashboards
5. **Configure** SLO-based alerting rules

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Latency Monitoring Flow                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  LLM Request                                                             │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐                                                         │
│  │ Start Timer │ ← Record request start time                            │
│  └──────┬──────┘                                                         │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────┐                                                         │
│  │   TTFT      │ ← Record time to first token (streaming)               │
│  │   Marker    │                                                         │
│  └──────┬──────┘                                                         │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────┐                                                         │
│  │   Stream    │ ← Record inter-token latencies                         │
│  │   Tokens    │                                                         │
│  └──────┬──────┘                                                         │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────┐                                                         │
│  │ End Timer   │ ← Record total latency                                  │
│  └──────┬──────┘                                                         │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              Prometheus Histogram Metrics                        │    │
│  │  • llm_request_duration_seconds_bucket                          │    │
│  │  • llm_time_to_first_token_seconds_bucket                       │    │
│  │  • llm_inter_token_latency_seconds_bucket                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📝 Exercises

### Exercise 1: Implement Latency Histogram

Add latency tracking to the metrics module:

```python
# llm_service/metrics.py

# TODO: Add the following histogram metrics
# 1. Request duration histogram (already in Lab 1)
# 2. Time-to-first-token (TTFT) histogram
# 3. Inter-token latency histogram
# 4. Queue time histogram (if applicable)

# Example bucket configuration for sub-second latencies
LATENCY_BUCKETS = [
    0.005, 0.01, 0.025, 0.05, 0.075, 0.1,
    0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf')
]
```

### Exercise 2: Calculate Percentiles in PromQL

Write PromQL queries for latency percentiles:

```promql
# TODO: Write queries for:
# 1. p50 (median) latency
histogram_quantile(0.50, sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))

# 2. p95 latency
histogram_quantile(0.95, sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))

# 3. p99 latency
histogram_quantile(0.99, sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))

# 4. p95 latency by model
histogram_quantile(0.95, sum(rate(llm_request_duration_seconds_bucket[5m])) by (le, model))
```

### Exercise 3: Define SLO Rules

Create SLO recording and alerting rules:

```yaml
# prometheus/rules/llm_slos.yml

groups:
  - name: llm_slo_recording
    rules:
      # TODO: Add recording rules for:
      # - p95 latency
      # - p99 latency
      # - Error rate
      # - Availability

  - name: llm_slo_alerts
    rules:
      # TODO: Add alerting rules for:
      # - SLO burn rate high
      # - Latency SLO violated
      # - Error budget exhausted
```

---

## 🔧 Setup Instructions

### Step 1: Update Lab 1 Configuration

```bash
cd curriculum/learning_paths/monitoring/module_1_llm_metrics/labs/lab_2_latency_monitoring

# Copy base configuration from Lab 1
cp -r ../lab_1_token_tracking/* .

# Add SLO rules directory
mkdir -p prometheus/rules
```

### Step 2: Add SLO Rules

Create the SLO rules file:

```yaml
# prometheus/rules/llm_slos.yml
groups:
  - name: llm_latency_slos
    interval: 30s
    rules:
      # Recording rules for latency percentiles
      - record: llm:request_duration:p50
        expr: |
          histogram_quantile(0.50, 
            sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))
      
      - record: llm:request_duration:p95
        expr: |
          histogram_quantile(0.95, 
            sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))
      
      - record: llm:request_duration:p99
        expr: |
          histogram_quantile(0.99, 
            sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))
```

### Step 3: Update Prometheus Configuration

```yaml
# prometheus/prometheus.yml
rule_files:
  - "rules/llm_slos.yml"
  - "rules/llm_alerts.yml"
```

---

## ✅ Verification

### Check 1: Latency Metrics Are Recorded

```bash
# Send test requests
for i in {1..20}; do
  curl -X POST http://localhost:8000/chat \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Test latency", "model": "gpt-3.5-turbo"}'
done

# Check histogram metrics
curl -s http://localhost:8000/metrics | grep llm_request_duration
```

### Check 2: Percentiles Are Calculated

```bash
# Query p95 latency
curl -G "http://localhost:9090/api/v1/query" \
  --data-urlencode "query=histogram_quantile(0.95, sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))"
```

### Check 3: SLO Rules Are Loaded

```bash
# Check Prometheus rules
curl http://localhost:9090/api/v1/rules | jq '.data.groups[].name'
```

---

## 📊 SLO Targets

| Metric | Target | Window | Error Budget |
|--------|--------|--------|--------------|
| Availability | 99.9% | 30 days | 43.2 minutes |
| p95 Latency | < 2s | 7 days | N/A |
| p99 Latency | < 5s | 7 days | N/A |
| Error Rate | < 0.1% | 1 hour | N/A |

---

## 🚨 Alerting Rules

```yaml
# prometheus/rules/llm_alerts.yml
groups:
  - name: llm_latency_alerts
    rules:
      - alert: LLMHighLatencyP95
        expr: llm:request_duration:p95 > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High p95 latency detected"
          description: "p95 latency is {{ $value }}s (threshold: 2s)"
      
      - alert: LLMHighLatencyP99
        expr: llm:request_duration:p99 > 5
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Critical p99 latency detected"
          description: "p99 latency is {{ $value }}s (threshold: 5s)"
```

---

## 📚 Additional Resources

- [Prometheus Histograms](https://prometheus.io/docs/practices/histograms/)
- [SLO Best Practices](https://sre.google/sre-book/service-level-objectives/)
- [Error Budgets](https://sre.google/sre-book/error-budgets/)

---

*Lab Duration: 2-3 hours*  
*Next: Lab 3 - Hallucination Detection*
