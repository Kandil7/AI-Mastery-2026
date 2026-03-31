# Module 1 Coding Challenges

## Challenge Overview

Complete these coding challenges to demonstrate your mastery of LLM metrics implementation.

---

## Easy Challenge: Token Counter Implementation

**Time Estimate:** 30-45 minutes  
**Difficulty:** Easy

### Problem

Implement a token counter that tracks input and output tokens for LLM requests and exports them as Prometheus metrics.

### Requirements

1. Create a Python class `TokenCounter` that:
   - Counts tokens using tiktoken library
   - Tracks cumulative input and output tokens
   - Calculates cost based on model pricing

2. Implement the following methods:
   ```python
   class TokenCounter:
       def count_tokens(self, text: str, model: str) -> int:
           """Count tokens for given text and model."""
           pass
       
       def calculate_cost(self, model: str, input_tokens: int, 
                         output_tokens: int) -> float:
           """Calculate cost for given token counts."""
           pass
       
       def get_stats(self) -> dict:
           """Return current token statistics."""
           pass
   ```

3. Add Prometheus metrics:
   - Counter for total input tokens
   - Counter for total output tokens
   - Gauge for current cost

### Test Case

```python
counter = TokenCounter()

# Test token counting
input_tokens = counter.count_tokens("Hello, how are you?", "gpt-4")
assert input_tokens > 0

# Test cost calculation
cost = counter.calculate_cost("gpt-4", 100, 50)
assert cost > 0

# Test stats
stats = counter.get_stats()
assert "total_input_tokens" in stats
assert "total_output_tokens" in stats
assert "total_cost" in stats
```

### Deliverables

- `token_counter.py` - Implementation file
- `test_token_counter.py` - Test file
- `requirements.txt` - Dependencies

### Evaluation Criteria

- [ ] Correct token counting
- [ ] Accurate cost calculation
- [ ] Prometheus metrics exported
- [ ] Tests pass

---

## Medium Challenge: Latency Dashboard

**Time Estimate:** 1-2 hours  
**Difficulty:** Medium

### Problem

Create a comprehensive Grafana dashboard for monitoring LLM latency metrics including percentiles, trends, and SLO compliance.

### Requirements

1. Create a Grafana dashboard JSON with the following panels:
   - **Stat Panel:** Current p95 latency
   - **Stat Panel:** Current p99 latency
   - **Time Series:** Latency over time (p50, p90, p95, p99)
   - **Histogram:** Latency distribution
   - **Table:** Latency by endpoint
   - **Gauge:** SLO compliance percentage

2. Include dashboard variables for:
   - Environment (production, staging, development)
   - Model (multi-select)
   - Endpoint (multi-select)
   - Time range

3. Add appropriate thresholds:
   - Green: p95 < 1s
   - Yellow: p95 1-2s
   - Red: p95 > 2s

### PromQL Queries

Your dashboard must include these queries:

```promql
# p95 latency
histogram_quantile(0.95, 
  sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))

# p99 latency
histogram_quantile(0.99, 
  sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))

# Request rate
sum(rate(llm_requests_total[5m]))

# Error rate
sum(rate(llm_errors_total[5m])) / sum(rate(llm_requests_total[5m]))
```

### Deliverables

- `latency_dashboard.json` - Grafana dashboard export
- `dashboard_readme.md` - Documentation explaining each panel
- `promql_queries.md` - List of all PromQL queries used

### Evaluation Criteria

- [ ] All required panels present
- [ ] Variables working correctly
- [ ] Thresholds configured
- [ ] Dashboard is visually clear
- [ ] PromQL queries are correct

---

## Hard Challenge: Cost Anomaly Detection System

**Time Estimate:** 3-4 hours  
**Difficulty:** Hard

### Problem

Build an automated cost anomaly detection system that identifies unusual spending patterns and triggers alerts.

### Requirements

1. **Data Collection:**
   - Collect hourly cost data from Prometheus
   - Store baseline metrics for comparison
   - Track costs by model, endpoint, and user segment

2. **Anomaly Detection:**
   Implement three detection methods:
   ```python
   class CostAnomalyDetector:
       def detect_spike(self, current_rate: float, 
                       baseline_rate: float, 
                       threshold_multiplier: float = 3.0) -> bool:
           """Detect sudden cost spikes."""
           pass
       
       def detect_trend_change(self, historical_data: list[float],
                              window_hours: int = 24) -> dict:
           """Detect significant trend changes."""
           pass
       
       def detect_budget_breach(self, current_spend: float,
                               budget_limit: float,
                               warning_threshold: float = 0.8) -> dict:
           """Detect budget threshold breaches."""
           pass
   ```

3. **Alerting:**
   - Create Prometheus alerting rules
   - Implement alert routing (Slack, email, PagerDuty)
   - Include runbook links in alerts

4. **Dashboard:**
   - Cost trend visualization
   - Anomaly timeline
   - Budget tracking
   - Model cost breakdown

### Implementation Details

```python
# Example anomaly detection logic
def detect_anomaly(self, metrics: dict) -> Alert:
    current_cost = metrics['current_hourly_cost']
    baseline_cost = metrics['baseline_hourly_cost']
    
    # Calculate z-score
    z_score = (current_cost - baseline_cost['mean']) / baseline_cost['std']
    
    if abs(z_score) > 3:  # 3 standard deviations
        return Alert(
            severity="critical" if z_score > 4 else "warning",
            type="cost_anomaly",
            message=f"Cost anomaly detected: z-score={z_score:.2f}",
            details={
                "current": current_cost,
                "baseline_mean": baseline_cost['mean'],
                "baseline_std": baseline_cost['std'],
                "deviation_percent": ((current_cost - baseline_cost['mean']) 
                                     / baseline_cost['mean'] * 100)
            }
        )
```

### Prometheus Alert Rules

```yaml
groups:
  - name: cost_anomaly_alerts
    rules:
      - alert: CostSpikeDetected
        expr: |
          (
            sum(rate(llm_cost_usd_total[1h])) 
            / 
            avg_over_time(sum(rate(llm_cost_usd_total[1h]))[7d:1h])
          ) > 3
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Cost spike detected"
          description: "Current cost is {{ $value | humanize }}x baseline"
      
      - alert: BudgetCritical
        expr: |
          sum(increase(llm_cost_usd_total[1d])) > 100
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Daily budget exceeded"
          description: "Daily cost: ${{ $value }}"
```

### Deliverables

- `anomaly_detector.py` - Main detection logic
- `alert_rules.yml` - Prometheus alerting rules
- `cost_dashboard.json` - Grafana dashboard
- `runbook.md` - Incident response runbook
- `test_anomaly_detector.py` - Test suite

### Evaluation Criteria

- [ ] All three detection methods implemented
- [ ] Accurate anomaly detection (tested with sample data)
- [ ] Alert rules correctly configured
- [ ] Dashboard provides actionable insights
- [ ] Runbook is comprehensive
- [ ] Code is well-documented and tested

---

## Submission Guidelines

1. Create a directory for each challenge:
   ```
   challenges/
   ├── easy/
   │   ├── token_counter.py
   │   ├── test_token_counter.py
   │   └── requirements.txt
   ├── medium/
   │   ├── latency_dashboard.json
   │   ├── dashboard_readme.md
   │   └── promql_queries.md
   └── hard/
       ├── anomaly_detector.py
       ├── alert_rules.yml
       ├── cost_dashboard.json
       ├── runbook.md
       └── test_anomaly_detector.py
   ```

2. Include a `README.md` in each directory with:
   - Setup instructions
   - How to run tests
   - Expected output

3. Submit by pushing to your repository and sharing the link.

---

## Grading Rubric

| Criteria | Easy | Medium | Hard |
|----------|------|--------|------|
| Functionality | 40% | 30% | 25% |
| Code Quality | 30% | 25% | 25% |
| Testing | 20% | 25% | 25% |
| Documentation | 10% | 20% | 25% |

---

*Complete at least 2 out of 3 challenges to pass Module 1*
