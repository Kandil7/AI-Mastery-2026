# Module 1: LLM-Specific Metrics - Comprehensive Theory

## Table of Contents

1. [Introduction to LLM Observability](#1-introduction-to-llm-observability)
2. [Token Economics and Cost Tracking](#2-token-economics-and-cost-tracking)
3. [Latency Metrics and Performance Analysis](#3-latency-metrics-and-performance-analysis)
4. [Quality Metrics and Hallucination Detection](#4-quality-metrics-and-hallucination-detection)
5. [Prometheus Integration for LLMs](#5-prometheus-integration-for-llms)
6. [Grafana Dashboard Design](#6-grafana-dashboard-design)
7. [SLOs and Error Budgets for LLMs](#7-slos-and-error-budgets-for-llms)
8. [Cost Optimization Strategies](#8-cost-optimization-strategies)
9. [Production Best Practices](#9-production-best-practices)
10. [Troubleshooting Guide](#10-troubleshooting-guide)

---

## 1. Introduction to LLM Observability

### 1.1 Why LLM Observability Differs from Traditional Monitoring

Large Language Models introduce unique observability challenges that traditional application monitoring cannot address:

#### Traditional Application Metrics vs. LLM Metrics

| Aspect | Traditional Apps | LLM Systems |
|--------|-----------------|-------------|
| **Response Time** | Deterministic, consistent | Variable, content-dependent |
| **Output Size** | Fixed schemas | Variable token counts |
| **Cost Model** | Fixed infrastructure | Per-token pricing |
| **Quality** | Binary (pass/fail tests) | Continuous (hallucination rate) |
| **Failure Modes** | Exceptions, timeouts | Hallucinations, drift, degradation |

#### The Three Pillars of LLM Observability

```
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Observability Stack                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Metrics   │    │    Logs     │    │   Traces    │         │
│  │             │    │             │    │             │         │
│  │ • Tokens    │    │ • Prompts   │    │ • LLM calls │         │
│  │ • Latency   │    │ • Responses │    │ • Embeddings│         │
│  │ • Cost      │    │ • Errors    │    │ • RAG steps │         │
│  │ • Quality   │    │ • Metadata  │    │ • Cache     │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Unified Observability Platform              │   │
│  │         (Prometheus + Grafana + OpenTelemetry)          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 The LLM Request Lifecycle

Understanding the complete request lifecycle is essential for effective monitoring:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        LLM Request Lifecycle                              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│  User Request                                                             │
│       │                                                                   │
│       ▼                                                                   │
│  ┌─────────────┐                                                         │
│  │ 1. Input    │ ← Measure: Input tokens, validation time               │
│  │    Validation│                                                        │
│  └──────┬──────┘                                                         │
│         │                                                                 │
│         ▼                                                                 │
│  ┌─────────────┐                                                         │
│  │ 2. Prompt   │ ← Measure: Prompt construction time, template vars     │
│  │    Building │                                                        │
│  └──────┬──────┘                                                         │
│         │                                                                 │
│         ▼                                                                 │
│  ┌─────────────┐                                                         │
│  │ 3. Embedding│ ← Measure: Embedding latency, vector DB query time     │
│  │    + RAG    │                                                        │
│  └──────┬──────┘                                                         │
│         │                                                                 │
│         ▼                                                                 │
│  ┌─────────────┐                                                         │
│  │ 4. LLM API  │ ← Measure: TTFT, inter-token latency, total tokens     │
│  │    Call     │                                                        │
│  └──────┬──────┘                                                         │
│         │                                                                 │
│         ▼                                                                 │
│  ┌─────────────┐                                                         │
│  │ 5. Response │ ← Measure: Parsing time, validation, formatting        │
│  │    Processing│                                                       │
│  └──────┬──────┘                                                         │
│         │                                                                 │
│         ▼                                                                 │
│  ┌─────────────┐                                                         │
│  │ 6. Output   │ ← Measure: Delivery time, user acknowledgment          │
│  │    Delivery │                                                        │
│  └─────────────┘                                                         │
│                                                                           │
│  Total Latency = Σ(all stages)                                           │
│                                                                           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Observability Questions for LLMs

Every monitoring system should answer these critical questions:

#### Performance Questions
1. What is the p95 latency for LLM requests?
2. How many tokens are we consuming per minute?
3. What is the time-to-first-token for streaming responses?
4. Are we hitting rate limits from providers?

#### Cost Questions
1. What is our daily/weekly/monthly token cost?
2. Which endpoints consume the most tokens?
3. What is the cost per user session?
4. Are there anomalous cost spikes?

#### Quality Questions
1. What is our hallucination rate?
2. How consistent are responses to similar prompts?
3. What is the user satisfaction score?
4. Are we detecting prompt injection attempts?

#### Reliability Questions
1. What is our error rate by model/provider?
2. How often do we hit rate limits?
3. What is our availability percentage?
4. How quickly do we detect and respond to incidents?

---

## 2. Token Economics and Cost Tracking

### 2.1 Understanding Token Counting

Tokens are the fundamental unit of LLM pricing and performance measurement.

#### Token Definition

A token is approximately:
- **English text:** 4 characters or 0.75 words
- **Code:** 1-2 characters per token (varies by language)
- **Other languages:** Varies significantly (Chinese: ~1.5 chars/token)

#### Token Counting Methods

```python
# Method 1: Using tiktoken (OpenAI)
import tiktoken

def count_tokens_tiktoken(text: str, model: str = "gpt-4") -> int:
    """Count tokens using OpenAI's tiktoken library."""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

# Method 2: Using transformers tokenizer
from transformers import AutoTokenizer

def count_tokens_transformers(text: str, model_name: str) -> int:
    """Count tokens using HuggingFace tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return len(tokenizer.encode(text))

# Method 3: Approximate counting (fast, less accurate)
def count_tokens_approximate(text: str) -> int:
    """Rough estimate: 4 characters ≈ 1 token."""
    return len(text) // 4
```

#### Token Count Variability by Model

| Model | Tokenizer | Avg chars/token | Notes |
|-------|-----------|-----------------|-------|
| GPT-4 | tiktoken (cl100k_base) | ~4.0 | Most accurate |
| GPT-3.5 | tiktoken (cl100k_base) | ~4.0 | Same as GPT-4 |
| Claude | Custom | ~3.5 | Slightly more efficient |
| Llama 2 | SentencePiece | ~4.5 | Varies by fine-tune |
| Mistral | SentencePiece | ~4.2 | Good for code |

### 2.2 Cost Calculation Formulas

#### Basic Cost Formula

```
Total Cost = (Input Tokens × Input Price) + (Output Tokens × Output Price)
```

#### Detailed Cost Breakdown

```python
class LLMPricingCalculator:
    """Calculate LLM costs across multiple providers."""
    
    PRICING = {
        # OpenAI pricing (per 1K tokens)
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        
        # Anthropic pricing (per 1K tokens)
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "claude-3-sonnet": {"input": 0.003, "output": 0.015},
        "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        
        # Google pricing (per 1K tokens)
        "gemini-pro": {"input": 0.00025, "output": 0.0005},
        
        # Self-hosted (estimated compute cost per 1K tokens)
        "llama-2-70b": {"input": 0.0007, "output": 0.0009},
        "mistral-large": {"input": 0.0005, "output": 0.0007},
    }
    
    def calculate_cost(self, model: str, input_tokens: int, 
                       output_tokens: int) -> float:
        """Calculate cost for a single request."""
        if model not in self.PRICING:
            raise ValueError(f"Unknown model: {model}")
        
        pricing = self.PRICING[model]
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        
        return input_cost + output_cost
    
    def calculate_monthly_projection(self, daily_requests: int,
                                     avg_input_tokens: int,
                                     avg_output_tokens: int,
                                     model: str) -> float:
        """Project monthly costs based on average usage."""
        daily_cost = daily_requests * self.calculate_cost(
            model, avg_input_tokens, avg_output_tokens
        )
        return daily_cost * 30
```

#### Cost Per User Calculation

```
Cost Per User = Total Period Cost / Active Users in Period
```

```python
def calculate_cost_per_user(total_cost: float, 
                           user_sessions: dict[str, int]) -> dict[str, float]:
    """
    Calculate cost allocated per user based on their token usage.
    
    Args:
        total_cost: Total LLM cost for the period
        user_sessions: Dict of user_id -> token_count
    
    Returns:
        Dict of user_id -> allocated_cost
    """
    total_tokens = sum(user_sessions.values())
    cost_per_token = total_cost / total_tokens if total_tokens > 0 else 0
    
    return {
        user_id: tokens * cost_per_token 
        for user_id, tokens in user_sessions.items()
    }
```

### 2.3 Prometheus Metrics for Token Tracking

#### Core Token Metrics

```yaml
# Prometheus metric definitions for token tracking

# Counter: Total input tokens (cumulative)
llm_tokens_input_total{model="gpt-4", endpoint="/chat"} 125000

# Counter: Total output tokens (cumulative)
llm_tokens_output_total{model="gpt-4", endpoint="/chat"} 87500

# Gauge: Current tokens per minute rate
llm_tokens_per_minute{model="gpt-4"} 15000

# Histogram: Token distribution per request
llm_tokens_per_request_bucket{model="gpt-4", type="input", le="100"} 450
llm_tokens_per_request_bucket{model="gpt-4", type="input", le="500"} 890
llm_tokens_per_request_bucket{model="gpt-4", type="input", le="1000"} 950
llm_tokens_per_request_bucket{model="gpt-4", type="input", le="+Inf"} 1000

# Counter: Total cost in USD
llm_cost_usd_total{model="gpt-4", environment="production"} 125.50

# Gauge: Current cost per minute
llm_cost_per_minute{environment="production"} 0.85
```

#### Python Implementation with Prometheus Client

```python
from prometheus_client import Counter, Gauge, Histogram, Summary
from prometheus_client import start_http_server
import time

# Define metrics
TOKENS_INPUT = Counter(
    'llm_tokens_input_total',
    'Total input tokens processed',
    ['model', 'endpoint', 'environment']
)

TOKENS_OUTPUT = Counter(
    'llm_tokens_output_total',
    'Total output tokens generated',
    ['model', 'endpoint', 'environment']
)

TOKENS_PER_REQUEST = Histogram(
    'llm_tokens_per_request',
    'Tokens per request distribution',
    ['model', 'type'],  # type: input or output
    buckets=[10, 50, 100, 250, 500, 1000, 2500, 5000, 10000]
)

COST_USD = Counter(
    'llm_cost_usd_total',
    'Total cost in USD',
    ['model', 'environment']
)

COST_PER_MINUTE = Gauge(
    'llm_cost_per_minute',
    'Current cost per minute',
    ['environment']
)

class TokenTracker:
    """Track token usage and costs with Prometheus metrics."""
    
    def __init__(self, model: str, endpoint: str, environment: str = "production"):
        self.model = model
        self.endpoint = endpoint
        self.environment = environment
        self.pricing = self._get_pricing(model)
    
    def _get_pricing(self, model: str) -> dict:
        """Get pricing for the specified model."""
        pricing_map = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
        }
        return pricing_map.get(model, {"input": 0.001, "output": 0.002})
    
    def record_tokens(self, input_tokens: int, output_tokens: int):
        """Record token usage for a request."""
        # Record counters
        TOKENS_INPUT.labels(
            model=self.model,
            endpoint=self.endpoint,
            environment=self.environment
        ).inc(input_tokens)
        
        TOKENS_OUTPUT.labels(
            model=self.model,
            endpoint=self.endpoint,
            environment=self.environment
        ).inc(output_tokens)
        
        # Record histograms
        TOKENS_PER_REQUEST.labels(
            model=self.model,
            type="input"
        ).observe(input_tokens)
        
        TOKENS_PER_REQUEST.labels(
            model=self.model,
            type="output"
        ).observe(output_tokens)
        
        # Calculate and record cost
        cost = self._calculate_cost(input_tokens, output_tokens)
        COST_USD.labels(
            model=self.model,
            environment=self.environment
        ).inc(cost)
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for given token counts."""
        input_cost = (input_tokens / 1000) * self.pricing["input"]
        output_cost = (output_tokens / 1000) * self.pricing["output"]
        return input_cost + output_cost

# Start the metrics server
def start_metrics_server(port: int = 8000):
    """Start Prometheus metrics HTTP server."""
    start_http_server(port)
    print(f"Metrics server started on port {port}")
    print(f"Metrics available at http://localhost:{port}/metrics")
```

### 2.4 Budget Tracking and Alerts

#### Budget Configuration

```yaml
# budget_config.yaml
budgets:
  daily:
    limit: 100.00  # USD
    warning_threshold: 0.8  # 80% of limit
    critical_threshold: 0.95  # 95% of limit
  
  monthly:
    limit: 2500.00  # USD
    warning_threshold: 0.8
    critical_threshold: 0.95
  
  per_endpoint:
    "/chat":
      daily_limit: 50.00
    "/completion":
      daily_limit: 30.00
    "/embedding":
      daily_limit: 20.00

alerts:
  budget_warning:
    condition: "current_spend / budget_limit > 0.8"
    severity: warning
    channels: ["slack", "email"]
  
  budget_critical:
    condition: "current_spend / budget_limit > 0.95"
    severity: critical
    channels: ["slack", "email", "pagerduty"]
  
  anomalous_spike:
    condition: "current_rate > baseline_rate * 3"
    severity: warning
    channels: ["slack"]
```

#### Budget Monitoring Implementation

```python
class BudgetMonitor:
    """Monitor LLM spending against budgets."""
    
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.spending_tracker = defaultdict(float)
        self.baseline_rates = {}
    
    def record_spend(self, amount: float, endpoint: str = "default"):
        """Record spending for budget tracking."""
        today = datetime.now().date()
        key = f"{today}:{endpoint}"
        self.spending_tracker[key] += amount
    
    def check_budgets(self) -> list[Alert]:
        """Check all budgets and return any triggered alerts."""
        alerts = []
        today = datetime.now().date()
        
        for endpoint, budget_config in self.config.get('per_endpoint', {}).items():
            key = f"{today}:{endpoint}"
            current_spend = self.spending_tracker.get(key, 0)
            daily_limit = budget_config.get('daily_limit', float('inf'))
            
            ratio = current_spend / daily_limit if daily_limit > 0 else 0
            
            if ratio > self.config['daily']['critical_threshold']:
                alerts.append(Alert(
                    severity="critical",
                    message=f"Budget critical: {endpoint} at {ratio*100:.1f}%",
                    current_spend=current_spend,
                    limit=daily_limit
                ))
            elif ratio > self.config['daily']['warning_threshold']:
                alerts.append(Alert(
                    severity="warning",
                    message=f"Budget warning: {endpoint} at {ratio*100:.1f}%",
                    current_spend=current_spend,
                    limit=daily_limit
                ))
        
        return alerts
    
    def detect_anomalies(self) -> list[Alert]:
        """Detect anomalous spending patterns."""
        alerts = []
        
        for endpoint in self.spending_tracker:
            current_rate = self._get_current_rate(endpoint)
            baseline = self.baseline_rates.get(endpoint, current_rate)
            
            if current_rate > baseline * 3:  # 3x baseline
                alerts.append(Alert(
                    severity="warning",
                    message=f"Anomalous spike detected for {endpoint}",
                    current_rate=current_rate,
                    baseline=baseline
                ))
        
        return alerts
```

---

## 3. Latency Metrics and Performance Analysis

### 3.1 Understanding LLM Latency Components

LLM latency consists of multiple components that must be measured separately:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      LLM Latency Breakdown                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Total Latency                                                           │
│  │                                                                       │
│  ├── Network Latency (5-50ms)                                           │
│  │   └── TCP connection, TLS handshake, data transfer                   │
│  │                                                                      │
│  ├── Queue Time (0-5000ms)                                              │
│  │   └── Time waiting in provider's queue                               │
│  │                                                                      │
│  ├── Time to First Token - TTFT (100-2000ms) ⭐ CRITICAL                │
│  │   ├── Prompt processing                                              │
│  │   ├── Model inference start                                          │
│  │   └── First token generation                                         │
│  │                                                                      │
│  ├── Inter-Token Latency (10-100ms per token)                           │
│  │   └── Time between consecutive tokens                                │
│  │                                                                      │
│  └── Post-Processing (5-50ms)                                           │
│      └── Response parsing, validation, formatting                       │
│                                                                          │
│  For streaming: User-perceived latency ≈ TTFT                           │
│  For non-streaming: User-perceived latency ≈ Total Latency              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Latency Percentiles Explained

#### Why Percentiles Matter

Average latency is misleading for LLM systems due to high variance:

```
Example Request Latencies (ms):
[150, 180, 200, 220, 250, 280, 350, 500, 800, 2500]

Average (mean): 543ms  ← Misleading!
Median (p50): 265ms    ← Better
p95: 2500ms            ← What most users experience as "slow"
p99: 2500ms            ← Worst cases
```

#### Percentile Definitions

| Percentile | Meaning | Use Case |
|------------|---------|----------|
| p50 (median) | 50% of requests faster | Typical user experience |
| p75 | 75% of requests faster | Good experience threshold |
| p90 | 90% of requests faster | SLO target |
| p95 | 95% of requests faster | Critical SLO target |
| p99 | 99% of requests faster | Worst case analysis |
| p99.9 | 99.9% of requests faster | Extreme outliers |

#### Prometheus Histogram for Latency

```yaml
# Histogram metric for request duration
llm_request_duration_seconds_bucket{le="0.1"} 100
llm_request_duration_seconds_bucket{le="0.25"} 450
llm_request_duration_seconds_bucket{le="0.5"} 890
llm_request_duration_seconds_bucket{le="1.0"} 950
llm_request_duration_seconds_bucket{le="2.5"} 980
llm_request_duration_seconds_bucket{le="5.0"} 990
llm_request_duration_seconds_bucket{le="10.0"} 995
llm_request_duration_seconds_bucket{le="+Inf"} 1000
llm_request_duration_seconds_sum 1250.5
llm_request_duration_seconds_count 1000
```

#### Calculating Percentiles in PromQL

```promql
# p50 (median) latency
histogram_quantile(0.50, 
  rate(llm_request_duration_seconds_bucket[5m]))

# p95 latency
histogram_quantile(0.95, 
  rate(llm_request_duration_seconds_bucket[5m]))

# p99 latency
histogram_quantile(0.99, 
  rate(llm_request_duration_seconds_bucket[5m]))

# p99 latency by model
histogram_quantile(0.99, 
  sum by (le, model) (rate(llm_request_duration_seconds_bucket[5m])))
```

### 3.3 Time-to-First-Token (TTFT) Deep Dive

TTFT is the most critical latency metric for streaming LLM responses.

#### TTFT Components

```python
class TTFTAnalyzer:
    """Analyze Time-to-First-Token metrics."""
    
    def measure_ttft(self, response_stream) -> dict:
        """
        Measure TTFT from a streaming response.
        
        Returns:
            dict with ttft_ms, tokens, and breakdown
        """
        start_time = time.time()
        first_token_time = None
        tokens = []
        
        for chunk in response_stream:
            if chunk.content and first_token_time is None:
                first_token_time = time.time()
            
            if chunk.content:
                tokens.append(chunk.content)
        
        ttft_ms = (first_token_time - start_time) * 1000 if first_token_time else None
        
        return {
            "ttft_ms": ttft_ms,
            "total_tokens": len(tokens),
            "total_time_ms": (time.time() - start_time) * 1000,
            "tokens_per_second": len(tokens) / ((time.time() - start_time) or 1)
        }
```

#### TTFT Benchmarks by Model

| Model | Typical TTFT | p95 TTFT | Notes |
|-------|-------------|----------|-------|
| GPT-4 Turbo | 200-400ms | 800ms | Fastest GPT-4 variant |
| GPT-4 | 400-800ms | 1500ms | Higher quality, slower |
| GPT-3.5 Turbo | 100-300ms | 600ms | Fastest OpenAI |
| Claude 3 Opus | 500-1000ms | 2000ms | Highest quality |
| Claude 3 Haiku | 100-250ms | 500ms | Fastest Claude |
| Llama 2 70B (self-hosted) | 200-500ms | 1000ms | Depends on hardware |

#### TTFT Optimization Strategies

1. **Prompt Optimization**
   - Reduce prompt length
   - Use system messages efficiently
   - Pre-compute embeddings

2. **Connection Management**
   - Keep connections warm
   - Use connection pooling
   - Implement retry with backoff

3. **Model Selection**
   - Use faster models for simple tasks
   - Implement model cascading
   - Consider distillation

### 3.4 Latency SLOs for LLMs

#### Defining Latency SLOs

```yaml
# SLO Configuration for LLM Service
slos:
  latency:
    # Streaming endpoints
    streaming:
      ttft_p95: 500ms      # Time to first token
      total_p95: 5000ms    # Total response time
      availability: 99.9%  # Uptime requirement
    
    # Non-streaming endpoints
    completion:
      total_p95: 3000ms
      total_p99: 8000ms
      availability: 99.9%
    
    # Embedding endpoints
    embedding:
      total_p95: 200ms
      total_p99: 500ms
      availability: 99.99%

  error_budget:
    monthly: 43.2 minutes  # For 99.9% availability
    weekly: 10.08 minutes
    daily: 1.44 minutes
```

#### Error Budget Calculation

```
For 99.9% availability:
- Monthly error budget = 30 days × 24 hours × 60 minutes × (1 - 0.999)
                       = 43,200 minutes × 0.001
                       = 43.2 minutes

For 99.99% availability:
- Monthly error budget = 43,200 minutes × 0.0001
                       = 4.32 minutes
```

#### SLO Monitoring Implementation

```python
class SLOMonitor:
    """Monitor SLO compliance and error budget consumption."""
    
    def __init__(self, slo_config: dict):
        self.slo_config = slo_config
        self.request_log = []
    
    def record_request(self, latency_ms: float, success: bool):
        """Record a request for SLO tracking."""
        self.request_log.append({
            "timestamp": datetime.now(),
            "latency_ms": latency_ms,
            "success": success
        })
    
    def calculate_slo_compliance(self, window_hours: int = 24) -> dict:
        """Calculate SLO compliance over a time window."""
        cutoff = datetime.now() - timedelta(hours=window_hours)
        recent_requests = [
            r for r in self.request_log 
            if r["timestamp"] > cutoff
        ]
        
        if not recent_requests:
            return {"compliance": 100.0, "total_requests": 0}
        
        # Calculate latency SLO compliance
        p95_latency = self._calculate_percentile(
            [r["latency_ms"] for r in recent_requests], 95
        )
        latency_target = self.slo_config["latency"]["completion"]["total_p95"]
        latency_compliant = p95_latency <= latency_target
        
        # Calculate availability
        success_count = sum(1 for r in recent_requests if r["success"])
        availability = success_count / len(recent_requests)
        availability_target = float(self.slo_config["latency"]["completion"]["availability"].rstrip('%')) / 100
        
        return {
            "total_requests": len(recent_requests),
            "p95_latency_ms": p95_latency,
            "latency_target_ms": latency_target,
            "latency_compliant": latency_compliant,
            "availability": availability * 100,
            "availability_target": availability_target * 100,
            "availability_compliant": availability >= availability_target
        }
    
    def calculate_error_budget_remaining(self) -> dict:
        """Calculate remaining error budget."""
        # Implementation for error budget tracking
        pass
    
    def _calculate_percentile(self, values: list[float], percentile: int) -> float:
        """Calculate percentile from a list of values."""
        if not values:
            return 0
        sorted_values = sorted(values)
        index = (percentile / 100) * (len(sorted_values) - 1)
        lower = int(index)
        upper = min(lower + 1, len(sorted_values) - 1)
        weight = index - lower
        return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
```

---

## 4. Quality Metrics and Hallucination Detection

### 4.1 Understanding Hallucination

Hallucination occurs when an LLM generates factually incorrect or nonsensical information.

#### Hallucination Types

| Type | Description | Example | Detection Method |
|------|-------------|---------|------------------|
| **Factual** | Incorrect facts | "Paris is the capital of Germany" | Fact-checking API |
| **Contextual** | Contradicts provided context | Ignores RAG results | Context consistency |
| **Confabulation** | Made-up sources/citations | "According to Smith et al. (2023)..." | Citation verification |
| **Logical** | Internal contradictions | Self-contradicting statements | Self-consistency check |

#### Hallucination Rate Formula

```
Hallucination Rate = (Hallucinated Responses / Total Responses) × 100
```

### 4.2 Hallucination Detection Methods

#### Method 1: Self-Consistency Check

```python
class SelfConsistencyChecker:
    """Detect hallucinations via self-consistency."""
    
    def __init__(self, llm_client, n_samples: int = 3):
        self.llm = llm_client
        self.n_samples = n_samples
    
    def check_consistency(self, prompt: str) -> dict:
        """
        Check response consistency by sampling multiple times.
        
        Returns:
            dict with consistency_score and detected_issues
        """
        responses = []
        for _ in range(self.n_samples):
            response = self.llm.generate(prompt, temperature=0.7)
            responses.append(response)
        
        # Compare responses for consistency
        consistency_score = self._calculate_consistency(responses)
        
        return {
            "consistency_score": consistency_score,
            "responses": responses,
            "is_consistent": consistency_score > 0.8,
            "variance": self._calculate_variance(responses)
        }
    
    def _calculate_consistency(self, responses: list[str]) -> float:
        """Calculate consistency score between responses."""
        if len(responses) < 2:
            return 1.0
        
        # Use embedding similarity to compare responses
        embeddings = [self._embed(r) for r in responses]
        similarities = []
        
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)
        
        return sum(similarities) / len(similarities) if similarities else 1.0
    
    def _embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        # Implementation using embedding model
        pass
    
    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        return dot_product / (norm_a * norm_b) if norm_a and norm_b else 0
```

#### Method 2: Fact-Checking with Knowledge Base

```python
class FactChecker:
    """Verify facts against knowledge base."""
    
    def __init__(self, knowledge_base: KnowledgeBase, llm_client):
        self.kb = knowledge_base
        self.llm = llm_client
    
    def verify_response(self, prompt: str, response: str) -> dict:
        """
        Verify response facts against knowledge base.
        
        Returns:
            dict with factuality_score and flagged_claims
        """
        # Extract claims from response
        claims = self._extract_claims(response)
        
        # Verify each claim
        verified_claims = []
        for claim in claims:
            verification = self._verify_claim(claim)
            verified_claims.append(verification)
        
        # Calculate factuality score
        true_claims = sum(1 for c in verified_claims if c["verified"])
        factuality_score = true_claims / len(verified_claims) if verified_claims else 1.0
        
        return {
            "factuality_score": factuality_score,
            "total_claims": len(verified_claims),
            "verified_claims": true_claims,
            "flagged_claims": [c for c in verified_claims if not c["verified"]],
            "is_factual": factuality_score > 0.9
        }
    
    def _extract_claims(self, response: str) -> list[str]:
        """Extract factual claims from response."""
        # Use LLM to extract claims
        extraction_prompt = f"""
        Extract all factual claims from the following text.
        Return each claim as a separate statement.
        
        Text: {response}
        
        Claims:
        """
        claims_response = self.llm.generate(extraction_prompt)
        return [c.strip() for c in claims_response.split('\n') if c.strip()]
    
    def _verify_claim(self, claim: str) -> dict:
        """Verify a single claim against knowledge base."""
        # Search knowledge base
        results = self.kb.search(claim, top_k=3)
        
        if not results:
            return {"claim": claim, "verified": False, "reason": "No supporting evidence"}
        
        # Check if claim is supported
        support_score = max(r["relevance_score"] for r in results)
        return {
            "claim": claim,
            "verified": support_score > 0.8,
            "support_score": support_score,
            "evidence": results
        }
```

#### Method 3: RAG Context Consistency

```python
class ContextConsistencyChecker:
    """Check if response is consistent with RAG context."""
    
    def __init__(self, llm_client):
        self.llm = llm_client
    
    def check_consistency(self, context: str, response: str) -> dict:
        """
        Check if response is consistent with provided context.
        
        Returns:
            dict with consistency_score and contradictions
        """
        # Use LLM to check for contradictions
        check_prompt = f"""
        Context: {context}
        
        Response: {response}
        
        Does the response contradict or go beyond the provided context?
        Identify any specific contradictions.
        
        Rate consistency from 0 (completely contradictory) to 1 (fully consistent).
        
        JSON Output:
        {{
            "consistency_score": <float 0-1>,
            "contradictions": [<list of contradictions>],
            "unsupported_claims": [<list of claims not in context>]
        }}
        """
        
        result = self.llm.generate_json(check_prompt)
        return result
    
    def calculate_nli_score(self, context: str, response: str) -> float:
        """
        Calculate Natural Language Inference score.
        
        Uses NLI model to determine if response is:
        - Entailed by context (score ~1.0)
        - Neutral to context (score ~0.5)
        - Contradicts context (score ~0.0)
        """
        # Implementation using NLI model (e.g., roberta-large-mnli)
        pass
```

### 4.3 Quality Metrics Dashboard

#### Key Quality Metrics

```yaml
# Quality metrics to track
quality_metrics:
  hallucination_rate:
    description: "Percentage of responses containing hallucinations"
    target: "< 5%"
    critical: "> 10%"
  
  self_consistency_score:
    description: "Average consistency across multiple samples"
    target: "> 0.85"
    critical: "< 0.7"
  
  factuality_score:
    description: "Percentage of claims verified against KB"
    target: "> 0.9"
    critical: "< 0.75"
  
  context_relevance:
    description: "Relevance of retrieved context to query"
    target: "> 0.8"
    critical: "< 0.6"
  
  user_satisfaction:
    description: "Average user rating (1-5)"
    target: "> 4.0"
    critical: "< 3.0"
```

#### Prometheus Quality Metrics

```python
# Quality metrics definitions
HALLUCINATION_RATE = Gauge(
    'llm_hallucination_rate',
    'Current hallucination rate',
    ['model', 'endpoint']
)

SELF_CONSISTENCY_SCORE = Gauge(
    'llm_self_consistency_score',
    'Self-consistency score',
    ['model']
)

FACTUALITY_SCORE = Gauge(
    'llm_factuality_score',
    'Factuality verification score',
    ['model', 'endpoint']
)

USER_SATISFACTION = Histogram(
    'llm_user_satisfaction',
    'User satisfaction ratings',
    ['endpoint'],
    buckets=[1, 2, 3, 4, 5]
)

CONTEXT_RELEVANCE = Gauge(
    'llm_context_relevance',
    'RAG context relevance score',
    ['endpoint']
)
```

---

## 5. Prometheus Integration for LLMs

### 5.1 Prometheus Architecture for LLMs

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Prometheus Monitoring Stack                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                 │
│  │   LLM App   │    │   LLM App   │    │   LLM App   │                 │
│  │  (Service 1)│    │  (Service 2)│    │  (Service N)│                 │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                 │
│         │                  │                  │                         │
│         │ /metrics         │ /metrics         │ /metrics                │
│         │                  │                  │                         │
│         ▼                  ▼                  ▼                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Prometheus Server                             │   │
│  │                                                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │   │
│  │  │   TSDB       │  │   Query      │  │   Alert      │          │   │
│  │  │   Storage    │  │   Engine     │  │   Manager    │          │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│         │                                                               │
│         │ Query                                                         │
│         ▼                                                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                      Grafana                                     │   │
│  │                  (Dashboards & Alerts)                           │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    monitor: 'llm-monitoring'
    environment: 'production'

# Alertmanager configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

# Rule files for recording rules and alerts
rule_files:
  - "rules/llm_metrics.yml"
  - "rules/llm_alerts.yml"

# Scrape configurations
scrape_configs:
  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']
  
  # LLM application metrics
  - job_name: 'llm-app'
    static_configs:
      - targets: ['llm-service:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
  
  # Node exporter for system metrics
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
  
  # Custom LLM exporter
  - job_name: 'llm-exporter'
    static_configs:
      - targets: ['llm-exporter:9090']
    scrape_interval: 5s
```

### 5.3 Recording Rules for LLM Metrics

```yaml
# rules/llm_metrics.yml
groups:
  - name: llm_token_metrics
    interval: 30s
    rules:
      # Token rate per minute
      - record: llm:tokens_input:rate_per_minute
        expr: rate(llm_tokens_input_total[5m]) * 60
      
      - record: llm:tokens_output:rate_per_minute
        expr: rate(llm_tokens_output_total[5m]) * 60
      
      # Cost rate per hour
      - record: llm:cost:rate_per_hour
        expr: rate(llm_cost_usd_total[1h])
      
      # Token ratio (output/input)
      - record: llm:tokens:output_input_ratio
        expr: |
          sum(rate(llm_tokens_output_total[5m])) 
          / 
          sum(rate(llm_tokens_input_total[5m]))

  - name: llm_latency_metrics
    interval: 30s
    rules:
      # p95 latency
      - record: llm:request_duration:p95
        expr: |
          histogram_quantile(0.95, 
            sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))
      
      # p99 latency
      - record: llm:request_duration:p99
        expr: |
          histogram_quantile(0.99, 
            sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))
      
      # Request rate
      - record: llm:requests:rate_per_second
        expr: rate(llm_requests_total[5m])
      
      # Error rate
      - record: llm:errors:rate_per_second
        expr: rate(llm_errors_total[5m])
```

### 5.4 Alert Rules for LLM Systems

```yaml
# rules/llm_alerts.yml
groups:
  - name: llm_cost_alerts
    rules:
      # High cost rate
      - alert: LLMHighCostRate
        expr: llm:cost:rate_per_hour > 50
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High LLM cost rate detected"
          description: "Cost rate is ${{ $value }}/hour (threshold: $50/hour)"
      
      # Budget exceeded
      - alert: LLMBudgetExceeded
        expr: |
          sum(increase(llm_cost_usd_total[1d])) > 100
        for: 0m
        labels:
          severity: critical
        annotations:
          summary: "Daily LLM budget exceeded"
          description: "Daily cost is ${{ $value }} (budget: $100)"

  - name: llm_latency_alerts
    rules:
      # High p95 latency
      - alert: LLMHighLatency
        expr: llm:request_duration:p95 > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High LLM latency detected"
          description: "p95 latency is {{ $value }}s (threshold: 2s)"
      
      # SLO burn rate high
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
          description: "Error rate is {{ $value | humanizePercentage }} (threshold: 0.1%)"

  - name: llm_quality_alerts
    rules:
      # High hallucination rate
      - alert: LLMHighHallucinationRate
        expr: llm_hallucination_rate > 0.1
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High hallucination rate detected"
          description: "Hallucination rate is {{ $value | humanizePercentage }} (threshold: 10%)"
      
      # Low user satisfaction
      - alert: LLMLowUserSatisfaction
        expr: |
          histogram_avg(llm_user_satisfaction) < 3
        for: 30m
        labels:
          severity: warning
        annotations:
          summary: "Low user satisfaction detected"
          description: "Average satisfaction is {{ $value }} (threshold: 3.0)"
```

---

## 6. Grafana Dashboard Design

### 6.1 Dashboard Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    LLM Monitoring Dashboard                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Header / Variables                            │   │
│  │  Environment: [Production ▼]  Model: [All ▼]  Time: [Last 1h ▼] │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐  │
│  │   Requests   │ │   p95 Latency│ │  Error Rate  │ │    Cost      │  │
│  │    15.2K     │ │    1.2s      │ │    0.05%     │ │   $45.20     │  │
│  │   +12%       │ │   -5%        │ │   +0.01%     │ │   +8%        │  │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘  │
│                                                                          │
│  ┌─────────────────────────────────┐ ┌────────────────────────────────┐ │
│  │     Request Rate (5 min)        │ │     Token Usage Over Time      │ │
│  │  [───────────────────────]      │ │  [────────────────────────]    │ │
│  │  [───────────────────────]      │ │  [────────────────────────]    │ │
│  └─────────────────────────────────┘ └────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────┐ ┌────────────────────────────────┐ │
│  │     Latency Distribution        │ │     Cost by Model              │ │
│  │  [Histogram Panel]              │ │  [Pie Chart]                   │ │
│  └─────────────────────────────────┘ └────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────┐ ┌────────────────────────────────┐ │
│  │     Hallucination Rate          │ │     User Satisfaction          │ │
│  │  [Gauge: 3.2%]                  │ │  [Gauge: 4.2/5]                │ │
│  └─────────────────────────────────┘ └────────────────────────────────┘ │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    Recent Alerts                                 │   │
│  │  [Warning] High latency at 14:32  [Info] New deployment at 14:00│   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Complete Dashboard JSON

```json
{
  "dashboard": {
    "title": "LLM Production Monitoring",
    "tags": ["llm", "production", "monitoring"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(rate(llm_requests_total[5m]))",
            "legendFormat": "req/s"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "reqps",
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 100},
                {"color": "red", "value": 500}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 0, "y": 0}
      },
      {
        "id": 2,
        "title": "p95 Latency",
        "type": "stat",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(llm_request_duration_seconds_bucket[5m])) by (le))",
            "legendFormat": "p95"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "s",
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 1},
                {"color": "red", "value": 2}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 6, "y": 0}
      },
      {
        "id": 3,
        "title": "Token Usage",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(rate(llm_tokens_input_total[5m]))",
            "legendFormat": "Input tokens/s"
          },
          {
            "expr": "sum(rate(llm_tokens_output_total[5m]))",
            "legendFormat": "Output tokens/s"
          }
        ],
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
      },
      {
        "id": 4,
        "title": "Cost Over Time",
        "type": "timeseries",
        "targets": [
          {
            "expr": "sum(rate(llm_cost_usd_total[5m]))",
            "legendFormat": "Cost/min"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "currencyUSD"
          }
        },
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
      },
      {
        "id": 5,
        "title": "Hallucination Rate",
        "type": "gauge",
        "targets": [
          {
            "expr": "llm_hallucination_rate",
            "legendFormat": "Rate"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "percent",
            "min": 0,
            "max": 100,
            "thresholds": {
              "steps": [
                {"color": "green", "value": null},
                {"color": "yellow", "value": 5},
                {"color": "red", "value": 10}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 12, "y": 8}
      },
      {
        "id": 6,
        "title": "User Satisfaction",
        "type": "gauge",
        "targets": [
          {
            "expr": "histogram_avg(llm_user_satisfaction)",
            "legendFormat": "Avg Rating"
          }
        ],
        "fieldConfig": {
          "defaults": {
            "unit": "short",
            "min": 1,
            "max": 5,
            "thresholds": {
              "steps": [
                {"color": "red", "value": null},
                {"color": "yellow", "value": 3},
                {"color": "green", "value": 4}
              ]
            }
          }
        },
        "gridPos": {"h": 8, "w": 6, "x": 18, "y": 8}
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
```

### 6.3 Dashboard Variables

```json
{
  "templating": {
    "list": [
      {
        "name": "environment",
        "type": "query",
        "query": "label_values(llm_requests_total, environment)",
        "current": {"text": "production", "value": "production"},
        "label": "Environment"
      },
      {
        "name": "model",
        "type": "query",
        "query": "label_values(llm_requests_total{environment=\"$environment\"}, model)",
        "current": {"text": "All", "value": "$__all"},
        "label": "Model",
        "multi": true,
        "includeAll": true
      },
      {
        "name": "endpoint",
        "type": "query",
        "query": "label_values(llm_requests_total{environment=\"$environment\"}, endpoint)",
        "current": {"text": "All", "value": "$__all"},
        "label": "Endpoint",
        "multi": true,
        "includeAll": true
      }
    ]
  }
}
```

---

## 7. SLOs and Error Budgets for LLMs

### 7.1 Defining Service Level Objectives

#### SLO Framework for LLMs

```yaml
# SLO Definitions
service_level_objectives:
  availability:
    target: 99.9%
    window: 30 days
    measurement: "Successful requests / Total requests"
  
  latency:
    streaming:
      ttft_p95: 500ms
      window: 7 days
    completion:
      total_p95: 3000ms
      window: 7 days
  
  quality:
    hallucination_rate:
      target: "< 5%"
      window: 7 days
    user_satisfaction:
      target: "> 4.0"
      window: 30 days
  
  cost:
    daily_budget:
      target: "$100"
      window: 1 day
    cost_per_request:
      target: "< $0.01"
      window: 7 days
```

### 7.2 Error Budget Policy

```yaml
# Error Budget Policy
error_budget_policy:
  monthly_budget:
    availability_99_9: 43.2 minutes
    availability_99_99: 4.32 minutes
  
  burn_rate_alerts:
    - name: "Burn rate 2x"
      condition: "error_rate > 2x allowed rate"
      duration: "1 hour"
      action: "Page on-call"
    
    - name: "Burn rate 6x"
      condition: "error_rate > 6x allowed rate"
      duration: "5 minutes"
      action: "Page on-call + escalate"
    
    - name: "Burn rate 14.4x"
      condition: "error_rate > 14.4x allowed rate"
      duration: "1 minute"
      action: "Emergency response"
  
  budget_exhausted:
    action: "Freeze deployments"
    review: "Required before resuming"
    notification: "All stakeholders"
```

### 7.3 SLO Implementation

```python
class SLOTracker:
    """Track SLO compliance and error budget consumption."""
    
    def __init__(self, slo_config: dict):
        self.slo_config = slo_config
        self.metrics = {}
    
    def record_request(self, success: bool, latency_ms: float, 
                       hallucination: bool = False):
        """Record a request for SLO tracking."""
        timestamp = datetime.now()
        
        # Track availability
        self._update_metric("availability", {
            "success": success,
            "timestamp": timestamp
        })
        
        # Track latency
        self._update_metric("latency", {
            "latency_ms": latency_ms,
            "timestamp": timestamp
        })
        
        # Track quality
        self._update_metric("quality", {
            "hallucination": hallucination,
            "timestamp": timestamp
        })
    
    def calculate_error_budget_remaining(self, metric_type: str) -> dict:
        """Calculate remaining error budget for a metric."""
        config = self.slo_config[metric_type]
        target = config["target"]
        window = config["window"]
        
        # Calculate allowed errors
        total_requests = self._get_total_requests(window)
        allowed_errors = total_requests * (1 - target)
        actual_errors = self._get_actual_errors(window, metric_type)
        
        remaining = allowed_errors - actual_errors
        remaining_percent = (remaining / allowed_errors * 100) if allowed_errors > 0 else 100
        
        return {
            "metric": metric_type,
            "target": target,
            "window": window,
            "allowed_errors": allowed_errors,
            "actual_errors": actual_errors,
            "remaining_errors": remaining,
            "remaining_percent": remaining_percent,
            "status": "healthy" if remaining > 0 else "exhausted"
        }
    
    def calculate_burn_rate(self, metric_type: str, window_minutes: int) -> float:
        """
        Calculate error budget burn rate.
        
        Burn rate = (actual error rate) / (allowed error rate)
        Burn rate > 1 means consuming budget faster than allowed
        """
        config = self.slo_config[metric_type]
        allowed_rate = 1 - config["target"]
        
        actual_errors = self._get_errors_in_window(metric_type, window_minutes)
        total_requests = self._get_requests_in_window(window_minutes)
        actual_rate = actual_errors / total_requests if total_requests > 0 else 0
        
        return actual_rate / allowed_rate if allowed_rate > 0 else 0
```

---

## 8. Cost Optimization Strategies

### 8.1 Cost Analysis Framework

```python
class CostOptimizer:
    """Analyze and optimize LLM costs."""
    
    def __init__(self, metrics_client: PrometheusClient):
        self.metrics = metrics_client
    
    def analyze_cost_breakdown(self, period: str = "7d") -> dict:
        """Analyze cost breakdown by various dimensions."""
        return {
            "by_model": self._get_cost_by_model(period),
            "by_endpoint": self._get_cost_by_endpoint(period),
            "by_user": self._get_cost_by_user(period),
            "by_time": self._get_cost_by_hour(period),
            "trends": self._analyze_cost_trends(period)
        }
    
    def identify_optimization_opportunities(self) -> list[dict]:
        """Identify cost optimization opportunities."""
        opportunities = []
        
        # Check for expensive model usage
        expensive_requests = self._find_expensive_model_usage()
        if expensive_requests:
            opportunities.append({
                "type": "model_downgrade",
                "description": f"{len(expensive_requests)} requests using expensive models",
                "potential_savings": self._calculate_downgrade_savings(expensive_requests),
                "priority": "high"
            })
        
        # Check for long prompts
        long_prompts = self._find_long_prompts()
        if long_prompts:
            opportunities.append({
                "type": "prompt_optimization",
                "description": f"{len(long_prompts)} requests with very long prompts",
                "potential_savings": self._calculate_prompt_savings(long_prompts),
                "priority": "medium"
            })
        
        # Check for caching opportunities
        cacheable_requests = self._find_cacheable_requests()
        if cacheable_requests:
            opportunities.append({
                "type": "caching",
                "description": f"{len(cacheable_requests)} requests could be cached",
                "potential_savings": self._calculate_cache_savings(cacheable_requests),
                "priority": "high"
            })
        
        return opportunities
    
    def _find_cacheable_requests(self) -> list:
        """Find requests that could benefit from caching."""
        # Implementation to find repeated similar prompts
        pass
```

### 8.2 Caching Strategies

```python
class LLMCache:
    """Implement caching for LLM responses."""
    
    def __init__(self, redis_client, similarity_threshold: float = 0.95):
        self.redis = redis_client
        self.similarity_threshold = similarity_threshold
        self.embedding_model = self._load_embedding_model()
    
    def get_cached_response(self, prompt: str) -> Optional[str]:
        """Get cached response for similar prompt."""
        prompt_embedding = self._embed(prompt)
        
        # Search for similar cached prompts
        similar_keys = self._find_similar_prompts(prompt_embedding)
        
        for key in similar_keys:
            cached_data = self.redis.get(key)
            if cached_data:
                data = json.loads(cached_data)
                if data["similarity"] >= self.similarity_threshold:
                    # Record cache hit metric
                    self._record_cache_hit()
                    return data["response"]
        
        return None
    
    def cache_response(self, prompt: str, response: str, ttl: int = 3600):
        """Cache a response for future use."""
        prompt_embedding = self._embed(prompt)
        cache_key = f"llm:cache:{hashlib.sha256(prompt.encode()).hexdigest()}"
        
        cache_data = {
            "prompt": prompt,
            "response": response,
            "embedding": prompt_embedding,
            "timestamp": datetime.now().isoformat(),
            "similarity": 1.0
        }
        
        self.redis.setex(cache_key, ttl, json.dumps(cache_data))
        self._record_cache_miss()
    
    def get_cache_metrics(self) -> dict:
        """Get cache performance metrics."""
        return {
            "hit_rate": self._calculate_hit_rate(),
            "total_hits": self._get_total_hits(),
            "total_misses": self._get_total_misses(),
            "avg_latency_saved_ms": self._calculate_latency_saved()
        }
```

### 8.3 Model Cascading

```python
class ModelCascader:
    """Implement model cascading for cost optimization."""
    
    def __init__(self, models: list[dict]):
        """
        Initialize with ordered list of models (cheapest to most expensive).
        
        Args:
            models: List of model configs with:
                - name: Model identifier
                - client: LLM client instance
                - confidence_threshold: Min confidence to accept response
        """
        self.models = models
    
    def generate(self, prompt: str, **kwargs) -> dict:
        """
        Generate response using model cascade.
        
        Tries models in order until confidence threshold is met.
        """
        for model_config in self.models:
            response = model_config["client"].generate(prompt, **kwargs)
            confidence = self._calculate_confidence(response)
            
            if confidence >= model_config["confidence_threshold"]:
                return {
                    "response": response,
                    "model_used": model_config["name"],
                    "confidence": confidence,
                    "cascade_level": self.models.index(model_config)
                }
        
        # Return last (most expensive) model response if no confidence threshold met
        last_model = self.models[-1]
        return {
            "response": response,
            "model_used": last_model["name"],
            "confidence": confidence,
            "cascade_level": len(self.models) - 1
        }
    
    def _calculate_confidence(self, response: str) -> float:
        """Calculate confidence score for a response."""
        # Implementation using various heuristics
        pass
```

---

## 9. Production Best Practices

### 9.1 Monitoring Checklist

#### Pre-Deployment Checklist
- [ ] All metrics endpoints configured
- [ ] Prometheus scrape targets verified
- [ ] Grafana dashboards imported
- [ ] Alert rules tested
- [ ] SLO targets defined
- [ ] Runbooks created

#### Post-Deployment Checklist
- [ ] Metrics flowing correctly
- [ ] Dashboards displaying data
- [ ] Alerts not firing falsely
- [ ] Baseline metrics captured
- [ ] On-call team trained

### 9.2 Metric Collection Best Practices

```python
class BestPractices:
    """Best practices for LLM metric collection."""
    
    # DO: Use appropriate metric types
    METRIC_TYPES = {
        "counters": ["total_requests", "total_tokens", "total_errors"],
        "gauges": ["current_connections", "queue_depth", "hallucination_rate"],
        "histograms": ["request_duration", "tokens_per_request", "user_satisfaction"]
    }
    
    # DO: Use consistent labeling
    STANDARD_LABELS = ["model", "endpoint", "environment", "version"]
    
    # DON'T: Use high-cardinality labels
    HIGH_CARDINALITY_LABELS_TO_AVOID = [
        "user_id",      # Use aggregated user segments instead
        "request_id",   # Use traces instead
        "prompt_hash"   # Too many unique values
    ]
    
    # DO: Set appropriate scrape intervals
    SCRAPE_INTERVALS = {
        "latency_metrics": "5s",
        "token_metrics": "15s",
        "cost_metrics": "60s",
        "quality_metrics": "30s"
    }
    
    # DO: Implement metric expiration
    METRIC_RETENTION = {
        "raw_metrics": "15d",
        "downsampled_1h": "90d",
        "downsampled_1d": "1y"
    }
```

### 9.3 Dashboard Design Principles

1. **Start with the most important metrics** - Request rate, latency, errors
2. **Use appropriate visualizations** - Time series for trends, gauges for current state
3. **Include context** - Show targets, thresholds, and comparisons
4. **Enable drill-down** - Allow filtering by model, endpoint, environment
5. **Keep it updated** - Remove unused panels, add new metrics as needed

---

## 10. Troubleshooting Guide

### 10.1 Common Issues and Solutions

#### Issue: Metrics Not Appearing in Prometheus

**Symptoms:**
- Targets show as DOWN in Prometheus UI
- No data in Grafana dashboards

**Diagnosis:**
```bash
# Check if metrics endpoint is accessible
curl http://llm-service:8000/metrics

# Check Prometheus targets
curl http://prometheus:9090/api/v1/targets

# Check Prometheus logs
docker logs prometheus
```

**Solutions:**
1. Verify the metrics endpoint is running
2. Check network connectivity between Prometheus and target
3. Verify scrape configuration in prometheus.yml
4. Check for firewall rules blocking access

#### Issue: High Cardinality Errors

**Symptoms:**
- Prometheus memory usage growing rapidly
- "Error: Out of bounds" in logs

**Diagnosis:**
```promql
# Check label cardinality
count by (__name__) ({__name__=~".+"})

# Find high-cardinality metrics
topk(10, count by (label_name) ({__name__=~".+"}))
```

**Solutions:**
1. Remove high-cardinality labels (user_id, request_id)
2. Use metric relabeling to drop unnecessary labels
3. Implement metric aggregation before export

#### Issue: Alert Fatigue

**Symptoms:**
- Too many alerts firing
- Team ignoring alerts

**Solutions:**
1. Increase alert thresholds
2. Add `for` duration to require sustained issues
3. Implement alert grouping in Alertmanager
4. Create tiered alerting (warning vs critical)
5. Regular alert review and tuning

### 10.2 Performance Troubleshooting

```python
class PerformanceDebugger:
    """Debug LLM performance issues."""
    
    def diagnose_high_latency(self) -> dict:
        """Diagnose causes of high latency."""
        diagnosis = {
            "network_latency": self._check_network_latency(),
            "queue_time": self._check_queue_time(),
            "model_latency": self._check_model_latency(),
            "post_processing": self._check_post_processing()
        }
        
        # Identify bottleneck
        max_component = max(diagnosis, key=diagnosis.get)
        
        return {
            "breakdown": diagnosis,
            "bottleneck": max_component,
            "recommendations": self._get_recommendations(max_component)
        }
    
    def _check_network_latency(self) -> float:
        """Check network latency to LLM provider."""
        # Implementation
        pass
    
    def _check_queue_time(self) -> float:
        """Check time spent in provider queue."""
        # Implementation
        pass
    
    def _get_recommendations(self, bottleneck: str) -> list[str]:
        """Get recommendations based on bottleneck."""
        recommendations = {
            "network_latency": [
                "Use connection pooling",
                "Deploy closer to provider",
                "Check for network congestion"
            ],
            "queue_time": [
                "Implement request queuing",
                "Use priority queues for important requests",
                "Consider dedicated endpoints"
            ],
            "model_latency": [
                "Use faster model variant",
                "Reduce prompt length",
                "Implement caching"
            ],
            "post_processing": [
                "Optimize response parsing",
                "Use async processing",
                "Reduce validation overhead"
            ]
        }
        return recommendations.get(bottleneck, [])
```

---

## Summary

This comprehensive theory module covers all aspects of LLM-specific metrics:

1. **Token Economics** - Understanding and tracking token usage and costs
2. **Latency Analysis** - Measuring and optimizing response times
3. **Quality Metrics** - Detecting hallucinations and measuring output quality
4. **Prometheus Integration** - Setting up metrics collection
5. **Grafana Dashboards** - Creating effective visualizations
6. **SLOs and Error Budgets** - Defining and tracking reliability targets
7. **Cost Optimization** - Strategies for reducing LLM costs
8. **Best Practices** - Production-ready patterns and anti-patterns
9. **Troubleshooting** - Common issues and solutions

Master these concepts to build production-grade observability for your LLM systems.

---

*End of Module 1 Theory Content*  
*Total Lines: 800+*  
*Next: Hands-on Labs*
