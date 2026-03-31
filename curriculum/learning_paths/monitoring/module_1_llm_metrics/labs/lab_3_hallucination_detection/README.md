# Lab 3: Hallucination Detection Pipeline

## 🎯 Lab Overview

**Duration:** 3-4 hours  
**Difficulty:** Advanced  
**Prerequisites:** Labs 1-2 completion, understanding of embeddings

In this lab, you will implement a hallucination detection pipeline using self-consistency checks and fact verification.

---

## 📋 Learning Objectives

After completing this lab, you will be able to:

1. **Implement** self-consistency checking for hallucination detection
2. **Build** fact verification against a knowledge base
3. **Track** hallucination rate metrics in Prometheus
4. **Create** quality-focused Grafana dashboards
5. **Configure** quality-based alerting

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   Hallucination Detection Pipeline                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Query                                                              │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │                    LLM Response Generation                       │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              Parallel Detection Methods                          │    │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │    │
│  │  │ Self-        │  │ Fact-        │  │ Context      │          │    │
│  │  │ Consistency  │  │ Checking     │  │ Consistency  │          │    │
│  │  │              │  │              │  │              │          │    │
│  │  │ • Sample 3x  │  │ • Extract    │  │ • Compare    │          │    │
│  │  │ • Compare    │  │   claims     │  │   with RAG   │          │    │
│  │  │ • Score      │  │ • Verify KB  │  │   context    │          │    │
│  │  └──────────────┘  └──────────────┘  └──────────────┘          │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              Aggregated Quality Score                            │    │
│  │  • Hallucination Rate                                            │    │
│  │  • Factuality Score                                              │    │
│  │  • Consistency Score                                             │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │              Prometheus Metrics Export                           │    │
│  │  • llm_hallucination_rate                                        │    │
│  │  • llm_factuality_score                                          │    │
│  │  • llm_self_consistency_score                                    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 📝 Exercises

### Exercise 1: Self-Consistency Checker

Implement multi-sample consistency checking:

```python
# hallucination_detector.py

class SelfConsistencyChecker:
    def __init__(self, llm_client, n_samples: int = 3):
        self.llm = llm_client
        self.n_samples = n_samples
    
    def check(self, prompt: str) -> dict:
        """
        Check response consistency by sampling multiple times.
        
        Returns:
            dict with consistency_score (0-1) and details
        """
        # TODO: Implement the following:
        # 1. Generate n_samples responses with temperature > 0
        # 2. Embed each response
        # 3. Calculate pairwise cosine similarity
        # 4. Return average similarity as consistency score
        pass
```

### Exercise 2: Fact Verification

Build fact checking against a knowledge base:

```python
class FactChecker:
    def __init__(self, knowledge_base, llm_client):
        self.kb = knowledge_base
        self.llm = llm_client
    
    def verify(self, response: str) -> dict:
        """
        Verify facts in response against knowledge base.
        
        Returns:
            dict with factuality_score (0-1) and flagged claims
        """
        # TODO: Implement the following:
        # 1. Extract claims from response using LLM
        # 2. Search knowledge base for each claim
        # 3. Score each claim's support
        # 4. Return overall factuality score
        pass
```

### Exercise 3: Quality Metrics Export

Export quality metrics to Prometheus:

```python
# metrics.py

# TODO: Add the following quality metrics
HALLUCINATION_RATE = Gauge(
    'llm_hallucination_rate',
    'Current hallucination rate',
    ['model', 'endpoint']
)

FACTUALITY_SCORE = Gauge(
    'llm_factuality_score',
    'Factuality verification score',
    ['model', 'endpoint']
)

SELF_CONSISTENCY_SCORE = Gauge(
    'llm_self_consistency_score',
    'Self-consistency score',
    ['model']
)
```

---

## 🔧 Setup Instructions

### Step 1: Create Lab Directory

```bash
cd curriculum/learning_paths/monitoring/module_1_llm_metrics/labs/lab_3_hallucination_detection

# Copy base from Lab 1
cp -r ../lab_1_token_tracking/* .

# Create hallucination detector module
mkdir -p hallucination_detector
```

### Step 2: Install Additional Dependencies

```txt
# requirements.txt additions
sentence-transformers==2.2.2
numpy==1.24.3
scikit-learn==1.3.0
```

### Step 3: Create Knowledge Base

```python
# knowledge_base.py

class SimpleKnowledgeBase:
    """Simple in-memory knowledge base for fact checking."""
    
    def __init__(self):
        self.facts = {
            "paris": "Paris is the capital of France",
            "germany": "Berlin is the capital of Germany",
            "quantum": "Quantum computing uses quantum mechanics principles",
            # Add more facts...
        }
    
    def search(self, query: str, top_k: int = 3) -> list:
        """Search for relevant facts."""
        # Simple keyword matching (use embeddings in production)
        results = []
        query_lower = query.lower()
        
        for key, fact in self.facts.items():
            if key in query_lower:
                results.append({
                    "fact": fact,
                    "relevance_score": 0.8  # Simplified
                })
        
        return results[:top_k]
```

---

## ✅ Verification

### Check 1: Self-Consistency Works

```python
# test_consistency.py
from hallucination_detector import SelfConsistencyChecker

checker = SelfConsistencyChecker(llm_client, n_samples=3)
result = checker.check("What is the capital of France?")

print(f"Consistency Score: {result['consistency_score']}")
print(f"Is Consistent: {result['is_consistent']}")
```

### Check 2: Fact Checking Works

```python
# test_fact_check.py
from hallucination_detector import FactChecker

checker = FactChecker(knowledge_base, llm_client)
result = checker.verify("Paris is the capital of France.")

print(f"Factuality Score: {result['factuality_score']}")
print(f"Flagged Claims: {result['flagged_claims']}")
```

### Check 3: Metrics Are Exported

```bash
# Check quality metrics
curl -s http://localhost:8000/metrics | grep llm_hallucination
curl -s http://localhost:8000/metrics | grep llm_factuality
```

---

## 📊 Quality Thresholds

| Metric | Good | Warning | Critical |
|--------|------|---------|----------|
| Hallucination Rate | < 5% | 5-10% | > 10% |
| Factuality Score | > 0.9 | 0.75-0.9 | < 0.75 |
| Consistency Score | > 0.85 | 0.7-0.85 | < 0.7 |

---

## 🚨 Quality Alerts

```yaml
# prometheus/rules/quality_alerts.yml
groups:
  - name: llm_quality_alerts
    rules:
      - alert: LLMHighHallucinationRate
        expr: llm_hallucination_rate > 0.1
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "High hallucination rate detected"
          description: "Hallucination rate is {{ $value | humanizePercentage }}"
      
      - alert: LLMLowFactualityScore
        expr: llm_factuality_score < 0.75
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "Low factuality score detected"
          description: "Factuality score is {{ $value }}"
```

---

## 📚 Additional Resources

- [Self-Consistency Paper](https://arxiv.org/abs/2203.11171)
- [Hallucination Detection Survey](https://arxiv.org/abs/2301.07301)
- [Evidently AI](https://github.com/evidentlyai/evidently)

---

*Lab Duration: 3-4 hours*  
*Module 1 Labs Complete*
