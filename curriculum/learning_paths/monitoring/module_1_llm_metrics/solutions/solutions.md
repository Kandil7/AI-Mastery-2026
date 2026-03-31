# Module 1 Solutions

## Lab Solutions

### Lab 1: Token Tracking Solution

#### Complete metrics.py Implementation

```python
"""
Complete Prometheus Metrics Implementation for Token Tracking
"""

from prometheus_client import Counter, Histogram, Gauge
from typing import Optional

# Token Counters
LLM_TOKENS_INPUT_TOTAL = Counter(
    name='llm_tokens_input_total',
    documentation='Total number of input tokens processed',
    labelnames=['model', 'endpoint', 'environment']
)

LLM_TOKENS_OUTPUT_TOTAL = Counter(
    name='llm_tokens_output_total',
    documentation='Total number of output tokens generated',
    labelnames=['model', 'endpoint', 'environment']
)

# Token Distribution
LLM_TOKENS_PER_REQUEST = Histogram(
    name='llm_tokens_per_request',
    documentation='Distribution of tokens per request',
    labelnames=['model', 'type'],
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000, float('inf')]
)

# Cost Metrics
LLM_COST_USD_TOTAL = Counter(
    name='llm_cost_usd_total',
    documentation='Total cost in USD for LLM usage',
    labelnames=['model', 'endpoint', 'environment']
)

LLM_COST_PER_REQUEST = Histogram(
    name='llm_cost_per_request',
    documentation='Cost distribution per request in USD',
    labelnames=['model', 'endpoint'],
    buckets=[0.0001, 0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
)

# Request Metrics
LLM_REQUESTS_TOTAL = Counter(
    name='llm_requests_total',
    documentation='Total number of LLM requests processed',
    labelnames=['model', 'endpoint', 'environment', 'status']
)

LLM_REQUEST_DURATION = Histogram(
    name='llm_request_duration_seconds',
    documentation='LLM request duration in seconds',
    labelnames=['model', 'endpoint'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, float('inf')]
)


def record_token_usage(
    model: str,
    endpoint: str,
    environment: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
    duration_seconds: float,
    status: str = "success"
) -> None:
    """Record complete token usage metrics for a request."""
    
    # Record token counters
    LLM_TOKENS_INPUT_TOTAL.labels(
        model=model, endpoint=endpoint, environment=environment
    ).inc(input_tokens)
    
    LLM_TOKENS_OUTPUT_TOTAL.labels(
        model=model, endpoint=endpoint, environment=environment
    ).inc(output_tokens)
    
    # Record token distributions
    LLM_TOKENS_PER_REQUEST.labels(model=model, type="input").observe(input_tokens)
    LLM_TOKENS_PER_REQUEST.labels(model=model, type="output").observe(output_tokens)
    
    # Record cost
    LLM_COST_USD_TOTAL.labels(
        model=model, endpoint=endpoint, environment=environment
    ).inc(cost_usd)
    LLM_COST_PER_REQUEST.labels(model=model, endpoint=endpoint).observe(cost_usd)
    
    # Record request metrics
    LLM_REQUESTS_TOTAL.labels(
        model=model, endpoint=endpoint, environment=environment, status=status
    ).inc()
    LLM_REQUEST_DURATION.labels(model=model, endpoint=endpoint).observe(duration_seconds)
```

#### Complete Cost Calculator

```python
class CostCalculator:
    """Calculate LLM costs based on token usage."""
    
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    }
    
    @classmethod
    def calculate(cls, model: str, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts."""
        if model not in cls.PRICING:
            # Default pricing for unknown models
            pricing = {"input": 0.001, "output": 0.002}
        else:
            pricing = cls.PRICING[model]
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return round(input_cost + output_cost, 6)
    
    @classmethod
    def get_pricing(cls, model: str) -> dict:
        """Get pricing information for a model."""
        return cls.PRICING.get(model, {"input": 0.001, "output": 0.002})
```

---

### Lab 2: Latency Monitoring Solution

#### SLO Recording Rules

```yaml
# prometheus/rules/llm_slos.yml
groups:
  - name: llm_latency_slos
    interval: 30s
    rules:
      # Latency percentiles
      - record: llm:request_duration:p50
        expr: |
          histogram_quantile(0.50, 
            sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))
      
      - record: llm:request_duration:p90
        expr: |
          histogram_quantile(0.90, 
            sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))
      
      - record: llm:request_duration:p95
        expr: |
          histogram_quantile(0.95, 
            sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))
      
      - record: llm:request_duration:p99
        expr: |
          histogram_quantile(0.99, 
            sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))
      
      # Latency by model
      - record: llm:request_duration:p95:by_model
        expr: |
          histogram_quantile(0.95, 
            sum(rate(llm_request_duration_seconds_bucket[5m])) by (le, model))
      
      # Request and error rates
      - record: llm:requests:rate_per_second
        expr: sum(rate(llm_requests_total[5m]))
      
      - record: llm:errors:rate_per_second
        expr: sum(rate(llm_errors_total[5m]))
      
      - record: llm:error_ratio
        expr: |
          sum(rate(llm_errors_total[5m])) 
          / 
          sum(rate(llm_requests_total[5m]))
      
      # Availability (inverse of error ratio)
      - record: llm:availability
        expr: |
          1 - (
            sum(rate(llm_errors_total[5m])) 
            / 
            sum(rate(llm_requests_total[5m]))
          )
```

#### SLO Alerting Rules

```yaml
# prometheus/rules/llm_alerts.yml
groups:
  - name: llm_slo_alerts
    rules:
      # High latency alerts
      - alert: LLMHighLatencyP95
        expr: llm:request_duration:p95 > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High p95 latency detected"
          description: "p95 latency is {{ $value }}s (threshold: 2s)"
          runbook_url: "https://runbooks.example.com/llm-high-latency"
      
      - alert: LLMHighLatencyP99
        expr: llm:request_duration:p99 > 5
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Critical p99 latency detected"
          description: "p99 latency is {{ $value }}s (threshold: 5s)"
          runbook_url: "https://runbooks.example.com/llm-critical-latency"
      
      # Error rate alerts
      - alert: LLMHighErrorRate
        expr: llm:error_ratio > 0.01
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 1%)"
      
      # SLO burn rate alerts
      - alert: LLMSLOBurnRateHigh
        expr: |
          (
            sum(rate(llm_errors_total[1h])) 
            / 
            sum(rate(llm_requests_total[1h]))
          ) > 0.001
        for: 2h
        labels:
          severity: critical
        annotations:
          summary: "LLM SLO burn rate too high"
          description: "Error rate is {{ $value | humanizePercentage }}"
```

---

### Lab 3: Hallucination Detection Solution

#### Self-Consistency Checker Implementation

```python
"""
Self-Consistency Hallucination Detector
"""

import numpy as np
from typing import List, Dict
from sentence_transformers import SentenceTransformer


class SelfConsistencyChecker:
    """Detect hallucinations via self-consistency checking."""
    
    def __init__(self, llm_client, n_samples: int = 3, 
                 model_name: str = "all-MiniLM-L6-v2"):
        self.llm = llm_client
        self.n_samples = n_samples
        self.embedding_model = SentenceTransformer(model_name)
    
    def check(self, prompt: str) -> Dict:
        """
        Check response consistency by sampling multiple times.
        
        Returns:
            dict with consistency_score and details
        """
        # Generate multiple responses
        responses = []
        for _ in range(self.n_samples):
            response = self.llm.generate(prompt, temperature=0.7)
            responses.append(response)
        
        # Calculate pairwise similarities
        similarities = self._calculate_pairwise_similarities(responses)
        
        # Calculate overall consistency score
        consistency_score = np.mean(similarities) if similarities else 1.0
        
        return {
            "consistency_score": float(consistency_score),
            "is_consistent": consistency_score > 0.7,
            "n_samples": self.n_samples,
            "similarities": similarities,
            "responses": responses,
            "min_similarity": float(np.min(similarities)) if similarities else 1.0,
            "max_similarity": float(np.max(similarities)) if similarities else 1.0,
            "std_similarity": float(np.std(similarities)) if similarities else 0.0
        }
    
    def _calculate_pairwise_similarities(self, responses: List[str]) -> List[float]:
        """Calculate pairwise cosine similarities between responses."""
        if len(responses) < 2:
            return [1.0]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(responses)
        
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        return similarities
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(dot_product / (norm_a * norm_b))
```

#### Fact Checker Implementation

```python
"""
Fact Verification Against Knowledge Base
"""

from typing import List, Dict


class FactChecker:
    """Verify facts in LLM responses against a knowledge base."""
    
    def __init__(self, knowledge_base, llm_client):
        self.kb = knowledge_base
        self.llm = llm_client
    
    def verify(self, response: str) -> Dict:
        """
        Verify facts in response against knowledge base.
        
        Returns:
            dict with factuality_score and flagged claims
        """
        # Extract claims from response
        claims = self._extract_claims(response)
        
        if not claims:
            return {
                "factuality_score": 1.0,
                "total_claims": 0,
                "verified_claims": 0,
                "flagged_claims": [],
                "is_factual": True
            }
        
        # Verify each claim
        verified_claims = []
        flagged_claims = []
        
        for claim in claims:
            verification = self._verify_claim(claim)
            if verification["verified"]:
                verified_claims.append(verification)
            else:
                flagged_claims.append(verification)
        
        # Calculate factuality score
        factuality_score = len(verified_claims) / len(claims) if claims else 1.0
        
        return {
            "factuality_score": float(factuality_score),
            "total_claims": len(claims),
            "verified_claims": len(verified_claims),
            "flagged_claims": flagged_claims,
            "is_factual": factuality_score > 0.9,
            "claim_details": verified_claims + flagged_claims
        }
    
    def _extract_claims(self, response: str) -> List[str]:
        """Extract factual claims from response using LLM."""
        extraction_prompt = f"""
Extract all factual claims from the following text.
Return each claim as a separate bullet point.
Only include verifiable factual statements, not opinions.

Text: {response}

Claims:
"""
        claims_response = self.llm.generate(extraction_prompt)
        
        # Parse claims from response
        claims = []
        for line in claims_response.split('\n'):
            line = line.strip()
            if line and not line.startswith('```'):
                # Remove bullet points and numbering
                line = line.lstrip('•-*').strip()
                line = line.lstrip('1234567890.').strip()
                if len(line) > 10:  # Filter out very short lines
                    claims.append(line)
        
        return claims
    
    def _verify_claim(self, claim: str) -> Dict:
        """Verify a single claim against knowledge base."""
        # Search knowledge base
        results = self.kb.search(claim, top_k=3)
        
        if not results:
            return {
                "claim": claim,
                "verified": False,
                "reason": "No supporting evidence found",
                "evidence": []
            }
        
        # Check if claim is supported
        max_relevance = max(r.get("relevance_score", 0) for r in results)
        
        return {
            "claim": claim,
            "verified": max_relevance > 0.7,
            "support_score": max_relevance,
            "reason": "Supported by knowledge base" if max_relevance > 0.7 else "Insufficient support",
            "evidence": results
        }
```

---

## Challenge Solutions

### Easy Challenge: Token Counter Solution

```python
"""
Token Counter Implementation - Solution
"""

import tiktoken
from prometheus_client import Counter, Gauge
from typing import Dict


class TokenCounter:
    """Count tokens and track usage metrics."""
    
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
    }
    
    # Prometheus metrics
    INPUT_TOKENS = Counter('llm_input_tokens_total', 'Total input tokens')
    OUTPUT_TOKENS = Counter('llm_output_tokens_total', 'Total output tokens')
    TOTAL_COST = Gauge('llm_total_cost', 'Total cost in USD')
    
    def __init__(self):
        self._encoders: Dict[str, tiktoken.Encoding] = {}
        self._total_input = 0
        self._total_output = 0
        self._total_cost = 0.0
    
    def count_tokens(self, text: str, model: str) -> int:
        """Count tokens for given text and model."""
        try:
            if model not in self._encoders:
                self._encoders[model] = tiktoken.encoding_for_model(model)
            return len(self._encoders[model].encode(text))
        except KeyError:
            # Fallback for unknown models
            if "cl100k_base" not in self._encoders:
                self._encoders["cl100k_base"] = tiktoken.get_encoding("cl100k_base")
            return len(self._encoders["cl100k_base"].encode(text))
        except Exception:
            # Rough estimate
            return len(text) // 4
    
    def calculate_cost(self, model: str, input_tokens: int, 
                      output_tokens: int) -> float:
        """Calculate cost for given token counts."""
        pricing = self.PRICING.get(model, {"input": 0.001, "output": 0.002})
        
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return round(input_cost + output_cost, 6)
    
    def record_usage(self, model: str, input_text: str, output_text: str):
        """Record token usage and update metrics."""
        input_tokens = self.count_tokens(input_text, model)
        output_tokens = self.count_tokens(output_text, model)
        cost = self.calculate_cost(model, input_tokens, output_tokens)
        
        # Update internal counters
        self._total_input += input_tokens
        self._total_output += output_tokens
        self._total_cost += cost
        
        # Update Prometheus metrics
        self.INPUT_TOKENS.inc(input_tokens)
        self.OUTPUT_TOKENS.inc(output_tokens)
        self.TOTAL_COST.set(self._total_cost)
    
    def get_stats(self) -> dict:
        """Return current token statistics."""
        return {
            "total_input_tokens": self._total_input,
            "total_output_tokens": self._total_output,
            "total_tokens": self._total_input + self._total_output,
            "total_cost": round(self._total_cost, 6),
            "token_ratio": round(self._total_output / self._total_input, 3) 
                          if self._total_input > 0 else 0
        }
```

### Medium Challenge: Dashboard Solution

The complete Grafana dashboard JSON is provided in the lab files. Key PromQL queries:

```promql
# p95 Latency
histogram_quantile(0.95, 
  sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))

# p99 Latency  
histogram_quantile(0.99, 
  sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))

# Token Rate
sum(rate(llm_tokens_input_total[5m])) * 60

# Cost Rate
sum(rate(llm_cost_usd_total[5m]))

# Error Rate
sum(rate(llm_errors_total[5m])) / sum(rate(llm_requests_total[5m]))

# Availability
1 - (sum(rate(llm_errors_total[5m])) / sum(rate(llm_requests_total[5m])))
```

### Hard Challenge: Anomaly Detector Solution

```python
"""
Cost Anomaly Detection System - Solution
"""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass
class Alert:
    severity: str
    alert_type: str
    message: str
    details: dict
    timestamp: datetime


class CostAnomalyDetector:
    """Detect cost anomalies in LLM usage."""
    
    def __init__(self, prometheus_client):
        self.prometheus = prometheus_client
        self.baseline_cache = {}
    
    def detect_spike(self, current_rate: float, 
                    baseline_rate: float, 
                    threshold_multiplier: float = 3.0) -> bool:
        """Detect sudden cost spikes."""
        if baseline_rate == 0:
            return current_rate > 0
        
        ratio = current_rate / baseline_rate
        return ratio > threshold_multiplier
    
    def detect_trend_change(self, historical_data: List[float],
                           window_hours: int = 24) -> Dict:
        """Detect significant trend changes using statistical analysis."""
        if len(historical_data) < 2:
            return {"anomaly_detected": False, "reason": "Insufficient data"}
        
        # Calculate statistics
        mean = np.mean(historical_data)
        std = np.std(historical_data)
        current = historical_data[-1]
        
        # Calculate z-score
        if std == 0:
            z_score = 0
        else:
            z_score = (current - mean) / std
        
        # Detect trend direction
        if len(historical_data) >= 3:
            recent_avg = np.mean(historical_data[-3:])
            older_avg = np.mean(historical_data[:-3]) if len(historical_data) > 3 else historical_data[0]
            trend = "increasing" if recent_avg > older_avg else "decreasing"
        else:
            trend = "unknown"
        
        anomaly_detected = abs(z_score) > 2
        
        return {
            "anomaly_detected": anomaly_detected,
            "z_score": float(z_score),
            "mean": float(mean),
            "std": float(std),
            "current": float(current),
            "trend": trend,
            "deviation_percent": float((current - mean) / mean * 100) if mean > 0 else 0
        }
    
    def detect_budget_breach(self, current_spend: float,
                            budget_limit: float,
                            warning_threshold: float = 0.8) -> Dict:
        """Detect budget threshold breaches."""
        ratio = current_spend / budget_limit if budget_limit > 0 else 0
        
        if ratio >= 1.0:
            status = "breached"
        elif ratio >= warning_threshold:
            status = "warning"
        else:
            status = "ok"
        
        return {
            "status": status,
            "current_spend": current_spend,
            "budget_limit": budget_limit,
            "ratio": ratio,
            "remaining": budget_limit - current_spend,
            "remaining_percent": (1 - ratio) * 100
        }
    
    def run_detection(self) -> List[Alert]:
        """Run all detection methods and return alerts."""
        alerts = []
        
        # Get current metrics
        current_cost = self._get_current_hourly_cost()
        baseline_cost = self._get_baseline_hourly_cost()
        historical_costs = self._get_historical_costs(hours=168)  # 1 week
        
        # Detect spike
        if self.detect_spike(current_cost, baseline_cost["mean"]):
            alerts.append(Alert(
                severity="warning",
                alert_type="cost_spike",
                message=f"Cost spike detected: {current_cost:.2f}/hr vs baseline {baseline_cost['mean']:.2f}/hr",
                details={
                    "current": current_cost,
                    "baseline": baseline_cost["mean"],
                    "ratio": current_cost / baseline_cost["mean"] if baseline_cost["mean"] > 0 else 0
                },
                timestamp=datetime.now()
            ))
        
        # Detect trend change
        trend_result = self.detect_trend_change(historical_costs)
        if trend_result["anomaly_detected"]:
            alerts.append(Alert(
                severity="warning",
                alert_type="trend_change",
                message=f"Cost trend anomaly: z-score={trend_result['z_score']:.2f}",
                details=trend_result,
                timestamp=datetime.now()
            ))
        
        # Detect budget breach
        budget_result = self.detect_budget_breach(
            current_spend=self._get_daily_spend(),
            budget_limit=100.0  # Daily budget
        )
        if budget_result["status"] != "ok":
            alerts.append(Alert(
                severity="critical" if budget_result["status"] == "breached" else "warning",
                alert_type="budget_breach",
                message=f"Budget {budget_result['status']}: ${budget_result['current_spend']:.2f}/${budget_result['budget_limit']:.2f}",
                details=budget_result,
                timestamp=datetime.now()
            ))
        
        return alerts
    
    def _get_current_hourly_cost(self) -> float:
        """Get current hourly cost from Prometheus."""
        # Implementation would query Prometheus
        return 0.0
    
    def _get_baseline_hourly_cost(self) -> Dict:
        """Get baseline hourly cost statistics."""
        # Implementation would query Prometheus for historical average
        return {"mean": 1.0, "std": 0.5}
    
    def _get_historical_costs(self, hours: int) -> List[float]:
        """Get historical cost data."""
        # Implementation would query Prometheus
        return [1.0] * hours
    
    def _get_daily_spend(self) -> float:
        """Get total spend for current day."""
        # Implementation would query Prometheus
        return 0.0
```

---

*End of Module 1 Solutions*
