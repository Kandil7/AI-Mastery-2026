# System Design Question: Fraud Detection System

## Problem Statement

Design a real-time fraud detection system for a financial services company that processes 10,000 transactions per second.

---

## Requirements

### Functional Requirements
1. Score each transaction with fraud probability in < 100ms
2. Block high-risk transactions automatically
3. Support rule-based and ML-based detection
4. Enable analysts to review flagged transactions
5. Learn from analyst decisions to improve model

### Non-Functional Requirements
1. 99.99% availability
2. P99 latency < 100ms
3. Handle 10K TPS with burst to 50K
4. False positive rate < 1%
5. Regulatory compliance (audit logs, explainability)

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Load Balancer                            │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                    Transaction Gateway API                       │
│  - Input validation, Rate limiting, Auth                         │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Rule Engine   │    │   ML Scoring    │    │  Feature Store  │
│ (Fast path)     │    │   Service       │    │  (Redis)        │
└────────┬────────┘    └────────┬────────┘    └─────────────────┘
         │                      │
         └──────────┬───────────┘
                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Decision Aggregator                           │
│  Combines rule + ML scores, applies business logic               │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
          ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   APPROVE       │    │   FLAG/REVIEW   │    │    BLOCK        │
│                 │    │                 │    │                 │
└─────────────────┘    └────────┬────────┘    └─────────────────┘
                                │
                    ┌───────────▼────────────┐
                    │   Case Management      │
                    │   (Analyst Review)     │
                    └───────────┬────────────┘
                                │
                    ┌───────────▼────────────┐
                    │  Feedback Loop         │
                    │  (Model Retraining)    │
                    └────────────────────────┘
```

---

## Component Deep Dive

### 1. Feature Store (Redis Cluster)

**Purpose**: Real-time feature computation and caching

**Features stored**:
- User: transaction count (1h, 24h, 7d), avg amount, location history
- Device: fingerprint, historical fraud rate
- Merchant: category risk, chargeback rate

**Design decisions**:
- Redis Cluster for horizontal scaling
- TTL-based expiration for time-windowed features
- Write-behind caching to persist to data warehouse

### 2. ML Scoring Service

**Model Architecture**:
- Ensemble of gradient boosting (XGBoost) + neural network
- Features: ~200 engineered features
- Latency budget: 50ms

**Serving Strategy**:
- TensorFlow Serving / Triton for inference
- Model A/B testing for safe rollouts
- Shadow mode for new models

**Model Features**:
- Transaction amount, time, location
- Velocity features (transactions per time window)
- Device/IP reputation scores
- User behavior patterns

### 3. Rule Engine

**Purpose**: Fast path for obvious fraud patterns

**Examples**:
- Blocklisted cards/devices
- Velocity limits exceeded
- Geographic impossibility (two countries in 10 min)

**Why separate**:
- Lower latency (< 5ms)
- Easier to update without ML retraining
- Regulatory requirements for explicit rules

### 4. Decision Aggregator

**Logic**:
```python
def aggregate_decision(rule_score, ml_score, context):
    # Hard blocks from rules
    if rule_score == BLOCK:
        return BLOCK, "Rule trigger"
    
    # Combine scores
    combined = 0.3 * rule_score + 0.7 * ml_score
    
    # Apply thresholds
    if combined > 0.9:
        return BLOCK, "High fraud probability"
    elif combined > 0.5:
        return REVIEW, "Medium risk - needs review"
    else:
        return APPROVE, "Low risk"
```

---

## Scaling Considerations

### Handling 10K TPS

1. **Horizontal scaling**: Stateless scoring services behind LB
2. **Feature caching**: Redis Cluster with 99.9% hit rate
3. **Async processing**: Write audit logs asynchronously
4. **Connection pooling**: Reuse DB/cache connections

### Handling Burst (50K TPS)

1. **Auto-scaling**: K8s HPA based on CPU/request queue
2. **Load shedding**: Degrade gracefully (skip ML, use rules only)
3. **Queue buffering**: Kafka for async processing
4. **Rate limiting**: Per-merchant limits

---

## Data Pipeline

```
Transactions → Kafka → Flink (streaming) → Feature Store
                  │
                  └──→ Data Warehouse → Batch Training → Model Registry
```

### Streaming Pipeline (Flink)
- Compute real-time aggregates (5min, 1h windows)
- Update feature store
- Generate training labels

### Batch Pipeline
- Daily model retraining
- Feature importance analysis
- Performance monitoring

---

## Monitoring & Observability

### Key Metrics
- **Latency**: P50, P99, P999
- **Fraud rate**: True positive, false positive, false negative
- **Model performance**: AUC, Precision@Recall curve
- **Business**: $ blocked, $ charged back

### Alerting
- Latency spike > 100ms P99
- Model drift detection (PSI score)
- False positive rate increase
- Unusual approval/block ratio

---

## Trade-offs & Alternatives

| Decision | Choice | Alternative | Why |
|----------|--------|-------------|-----|
| Feature Store | Redis | Feast | Lower latency, simpler ops |
| ML Framework | XGBoost | Deep Learning | Interpretability requirement |
| Message Queue | Kafka | RabbitMQ | Higher throughput needed |
| Deployment | K8s | Lambda | Consistent latency, no cold start |

---

## Interview Discussion Points

1. **How would you handle model drift?**
   - Monitor PSI score, automatic alerts, champion/challenger testing

2. **How to ensure explainability for regulators?**
   - SHAP values for each decision, rule audit logs

3. **What if ML service goes down?**
   - Fallback to rules-only mode, alert on SLA breach

4. **How to prevent adversarial attacks?**
   - Input validation, rate limiting, anomaly detection on patterns
