# Lab 1: Token Tracking with Prometheus

## 🎯 Lab Overview

**Duration:** 2-3 hours  
**Difficulty:** Intermediate  
**Prerequisites:** Module 1 theory, Docker basics, Python intermediate

In this lab, you will implement a complete token tracking system for LLM applications using Prometheus metrics.

---

## 📋 Learning Objectives

After completing this lab, you will be able to:

1. **Implement** Prometheus metrics for token tracking
2. **Configure** Prometheus to scrape custom metrics endpoints
3. **Create** Grafana visualizations for token usage
4. **Calculate** real-time cost estimates from token metrics

---

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   LLM Service   │────▶│   Prometheus    │────▶│    Grafana      │
│   (Port 8000)   │     │   (Port 9090)   │     │   (Port 3000)   │
│                 │     │                 │     │                 │
│ • Token Counter │     │ • Scrape config │     │ • Dashboards    │
│ • Cost Calc     │     │ • TSDB storage  │     │ • Visualizations│
│ • Metrics API   │     │ • Query engine  │     │ • Alerts        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 📁 Lab Files

```
lab_1_token_tracking/
├── docker-compose.yml          # Lab infrastructure
├── prometheus/
│   └── prometheus.yml          # Prometheus configuration
├── llm_service/
│   ├── app.py                  # LLM service with metrics
│   ├── metrics.py              # Prometheus metrics definitions
│   └── requirements.txt        # Python dependencies
├── grafana/
│   └── dashboards/
│       └── token_dashboard.json
└── README.md                   # This file
```

---

## 🔧 Setup Instructions

### Step 1: Start the Lab Environment

```bash
cd curriculum/learning_paths/monitoring/module_1_llm_metrics/labs/lab_1_token_tracking

# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps

# Expected output:
# NAME                    STATUS          PORTS
# lab1-grafana            Up              0.0.0.0:3000->3000/tcp
# lab1-prometheus         Up              0.0.0.0:9090->9090/tcp
# lab1-llm-service        Up              0.0.0.0:8000->8000/tcp
```

### Step 2: Verify Metrics Endpoint

```bash
# Check if metrics endpoint is accessible
curl http://localhost:8000/metrics

# You should see Prometheus-formatted metrics including:
# llm_tokens_input_total
# llm_tokens_output_total
# llm_cost_usd_total
```

### Step 3: Access Grafana

1. Open http://localhost:3000 in your browser
2. Login with `admin` / `admin`
3. Navigate to Dashboards → Browse
4. Import the token dashboard from `grafana/dashboards/token_dashboard.json`

---

## 📝 Exercises

### Exercise 1: Implement Token Counter

Create a token counter that tracks input and output tokens:

```python
# llm_service/metrics.py
from prometheus_client import Counter, Histogram, Gauge

# TODO: Define the following metrics
# 1. Counter for total input tokens
# 2. Counter for total output tokens  
# 3. Histogram for tokens per request
# 4. Gauge for current tokens per minute rate
# 5. Counter for total cost in USD
```

### Exercise 2: Create Metrics Middleware

Implement middleware to automatically track tokens:

```python
# llm_service/app.py
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import time

app = FastAPI()

# TODO: Create middleware that:
# 1. Intercepts LLM API calls
# 2. Counts input and output tokens
# 3. Records metrics to Prometheus
# 4. Calculates and tracks cost

@app.post("/chat")
async def chat(request: Request):
    # Simulate LLM call
    body = await request.json()
    prompt = body.get("prompt", "")
    
    # TODO: Track tokens and cost here
    
    return {"response": "Sample response"}
```

### Exercise 3: Build Cost Calculator

Implement real-time cost calculation:

```python
class CostCalculator:
    PRICING = {
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
    }
    
    def calculate(self, model: str, input_tokens: int, output_tokens: int) -> float:
        # TODO: Implement cost calculation
        pass
```

---

## ✅ Verification

### Check 1: Metrics Are Being Collected

```bash
# Query Prometheus for token metrics
curl -G "http://localhost:9090/api/v1/query" \
  --data-urlencode "query=llm_tokens_input_total"

# Should return current token count
```

### Check 2: Grafana Dashboard Shows Data

1. Open the token dashboard in Grafana
2. Verify all panels show data
3. Check that metrics update every 15 seconds

### Check 3: Cost Calculation Is Accurate

```bash
# Send test requests
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "model": "gpt-4"}'

# Check cost metric
curl -s http://localhost:8000/metrics | grep llm_cost
```

---

## 🧪 Testing

### Generate Test Traffic

```python
# test_traffic.py
import requests
import time
import random

MODELS = ["gpt-4", "gpt-3.5-turbo", "claude-3-opus"]
PROMPTS = [
    "What is the capital of France?",
    "Explain quantum computing in simple terms.",
    "Write a haiku about programming.",
    "What are the benefits of exercise?",
    "How do I learn Python?"
]

def generate_traffic(duration_minutes: int = 5):
    """Generate test traffic for the LLM service."""
    end_time = time.time() + (duration_minutes * 60)
    
    while time.time() < end_time:
        model = random.choice(MODELS)
        prompt = random.choice(PROMPTS)
        
        response = requests.post(
            "http://localhost:8000/chat",
            json={"prompt": prompt, "model": model}
        )
        
        print(f"Sent request: model={model}, status={response.status_code}")
        time.sleep(random.uniform(1, 3))

if __name__ == "__main__":
    generate_traffic()
```

---

## 📊 Expected Results

After completing the lab:

1. **Metrics Endpoint** returns valid Prometheus metrics
2. **Prometheus** shows `llm_tokens_*` metrics in Targets
3. **Grafana Dashboard** displays:
   - Total tokens over time
   - Tokens per minute rate
   - Cost breakdown by model
   - Token distribution histogram

---

## 🚨 Troubleshooting

### Issue: Prometheus Shows Target as DOWN

```bash
# Check Prometheus configuration
docker exec lab1-prometheus cat /etc/prometheus/prometheus.yml

# Check if service is accessible
docker exec lab1-prometheus curl http://lab1-llm-service:8000/metrics
```

### Issue: No Data in Grafana

1. Verify Prometheus datasource is configured
2. Check metric names match exactly
3. Ensure time range includes data collection period

### Issue: Metrics Not Updating

```bash
# Check service logs
docker logs lab1-llm-service

# Verify metrics are being recorded
curl http://localhost:8000/metrics | head -50
```

---

## 📚 Additional Resources

- [Prometheus Python Client](https://github.com/prometheus/client_python)
- [FastAPI Middleware](https://fastapi.tiangolo.com/tutorial/middleware/)
- [Prometheus Metric Types](https://prometheus.io/docs/concepts/metric_types/)

---

## 📝 Submission Checklist

- [ ] Token counter implemented and working
- [ ] Metrics middleware tracking all requests
- [ ] Cost calculator accurate
- [ ] Grafana dashboard showing real-time data
- [ ] Test traffic generated successfully
- [ ] All verification checks passed

---

*Lab Duration: 2-3 hours*  
*Next: Lab 2 - Latency Monitoring*
