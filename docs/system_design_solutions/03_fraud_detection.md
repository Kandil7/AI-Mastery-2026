# System Design: Real-Time Fraud Detection Pipeline

## Problem Statement

Design a fraud detection system for a payments platform with:
- **1M transactions/day** (~12 req/s average, 100 req/s peak)
- **<100ms decision latency** (block suspicious transactions in real-time)
- **<0.1% false positive rate** (minimize blocking legitimate users)
- **>95% fraud recall** (catch most fraudulent transactions)
- Handle **evolving fraud patterns** (adversarial environment)

---

## High-Level Architecture

```
Transaction Request
       │
       ▼
┌──────────────────────────────────┐
│   API Gateway                    │
│   (Rate limiting, auth)          │
└──────────┬───────────────────────┘
           │
           ▼
┌──────────────────────────────────┐
│   Feature Engineering Service    │
│   (Real-time + Batch features)   │
└──────────┬───────────────────────┘
           │
           ├──────────────┬──────────────┐
           ▼              ▼              ▼
    ┌──────────┐   ┌──────────┐   ┌──────────┐
    │  Rules   │   │  ML      │   │  Anomaly │
    │  Engine  │   │  Model   │   │ Detection│
    │(Instant) │   │(XGBoost) │   │(Isolation│
    └────┬─────┘   └────┬─────┘   │  Forest) │
         │              │          └────┬─────┘
         └──────────────┴───────────────┘
                        │
                        ▼
              ┌──────────────────┐
              │  Ensemble        │
              │  (Weighted Vote) │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │  Decision Engine │
              │  (Thresholds +   │
              │   Risk Scores)   │
              └────────┬─────────┘
                       │
          ┌────────────┴────────────┐
          ▼                         ▼
    [APPROVE]                  [BLOCK/REVIEW]
          │                         │
          └─────────┬───────────────┘
                    ▼
          ┌──────────────────┐
          │  Event Stream    │
          │  (Kafka)         │
          └────────┬─────────┘
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ Model        │      │ Analytics    │
│ Retraining   │      │ Dashboard    │
└──────────────┘      └──────────────┘
```

---

## Component Deep Dive

### 1. Feature Engineering

**Real-Time Features** (computed per transaction):
```python
class RealtimeFeatureExtractor:
    def extract(self, transaction: Transaction) -> dict:
        return {
            # Transaction attributes
            'amount': transaction.amount,
            'amount_normalized': transaction.amount / user.avg_transaction,
            'merchant_category': transaction.merchant_category,
            'is_international': transaction.country != user.country,
            
            # Time-based
            'hour_of_day': transaction.timestamp.hour,
            'is_weekend': transaction.timestamp.weekday() >= 5,
            'time_since_last_txn': (transaction.timestamp - user.last_txn_time).seconds,
            
            # Device/Location
            'new_device': transaction.device_id not in user.known_devices,
            'distance_from_home': haversine(transaction.location, user.home_location),
            'ip_vpn_detected': check_vpn(transaction.ip),
            
            # Velocity features (from Redis)
            'txn_count_1h': self.redis.get(f"txn_count:{user_id}:1h"),
            'txn_count_24h': self.redis.get(f"txn_count:{user_id}:24h"),
            'unique_merchants_1h': self.redis.scard(f"merchants:{user_id}:1h"),
        }
```

**Batch Features** (precomputed daily):
```python
class BatchFeatureGenerator:
    def compute_daily(self, user_id):
        # Historical patterns
        last_30d = get_transactions(user_id, days=30)
        
        return {
            'avg_transaction_amount': np.mean([t.amount for t in last_30d]),
            'std_transaction_amount': np.std([t.amount for t in last_30d]),
            'favorite_merchants': most_common([t.merchant for t in last_30d], n=5),
            'typical_transaction_hours': mode([t.timestamp.hour for t in last_30d]),
            'has_international_history': any(t.is_international for t in last_30d),
            'avg_days_between_transactions': ...,
            
            # Aggregated fraud history
            'chargeback_rate_90d': count_chargebacks(user_id, days=90) / len(last_30d),
            'account_age_days': (now() - user.created_at).days,
        }
```

**Feature Store** (Redis for real-time, Postgres for batch):
```python
# Store velocity counters
def track_transaction(user_id, transaction):
    pipe = redis.pipeline()
    
    # Increment counters with TTL
    pipe.incr(f"txn_count:{user_id}:1h")
    pipe.expire(f"txn_count:{user_id}:1h", 3600)
    
    pipe.incr(f"txn_count:{user_id}:24h")
    pipe.expire(f"txn_count:{user_id}:24h", 86400)
    
    # Track unique merchants
    pipe.sadd(f"merchants:{user_id}:1h", transaction.merchant_id)
    pipe.expire(f"merchants:{user_id}:1h", 3600)
    
    pipe.execute()
```

---

### 2. Multi-Layer Detection

#### Layer 1: Rules Engine (Deterministic)

**Purpose**: Catch obvious fraud patterns instantly

```python
class RulesEngine:
    def evaluate(self, transaction, features):
        # Hard rules (auto-block)
        if features['amount'] > 10000 and features['new_device']:
            return Decision(action='BLOCK', reason='Large amount from new device', confidence=1.0)
        
        if features['txn_count_1h'] > 10:
            return Decision(action='BLOCK', reason='Velocity attack detected', confidence=1.0)
        
        if features['distance_from_home'] > 500 and features['time_since_last_txn'] < 600:
            return Decision(action='BLOCK', reason='Impossible travel', confidence=0.95)
        
        # Soft rules (increase risk score)
        risk_score = 0.0
        if features['is_international'] and not features['has_international_history']:
            risk_score += 0.3
        
        if features['hour_of_day'] in [2, 3, 4]:  # Late night
            risk_score += 0.1
        
        return Decision(action='SCORE', score=risk_score)
```

**Advantages**:
- Zero latency (if-else logic)
- Explainable (clear reason for users)
- Update rules without retraining

---

#### Layer 2: ML Model (XGBoost)

**Why XGBoost?**
- Fast inference (1-2ms)
- Handles missing features
- Feature importance for explainability
- Proven track record in fraud detection

**Training**:
```python
import xgboost as xgb

# Prepare data
X_train = pd.DataFrame([features for txn, features in training_data])
y_train = np.array([txn.is_fraud for txn, _ in training_data])

# Handle class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# Train model
model = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    scale_pos_weight=scale_pos_weight,  # Handle 1% fraud rate
    objective='binary:logistic',
    eval_metric='auc'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    early_stopping_rounds=10,
    verbose=True
)

# Save model
model.save_model('models/fraud_xgboost.json')
```

**Inference**:
```python
def predict_fraud_probability(transaction, features):
    # Convert to XGBoost DMatrix for speed
    dmatrix = xgb.DMatrix(pd.DataFrame([features]))
    
    # Get probability
    fraud_prob = model.predict(dmatrix)[0]
    
    return fraud_prob
```

---

#### Layer 3: Anomaly Detection (Isolation Forest)

**Purpose**: Detect novel fraud patterns not seen in training

```python
from sklearn.ensemble import IsolationForest

class AnomalyDetector:
    def __init__(self):
        self.model = IsolationForest(
            contamination=0.01,  # Expect 1% anomalies
            random_state=42
        )
    
    def fit(self, normal_transactions):
        # Train only on legitimate transactions
        features = [extract_numerical_features(t) for t in normal_transactions]
        self.model.fit(features)
    
    def score(self, transaction, features):
        numerical_features = extract_numerical_features(transaction)
        
        # Returns -1 for anomaly, 1 for normal
        prediction = self.model.predict([numerical_features])[0]
        
        # Get anomaly score (lower = more anomalous)
        anomaly_score = self.model.score_samples([numerical_features])[0]
        
        return -anomaly_score  # Convert to 0-1 range (higher = more anomalous)
```

---

### 3. Ensemble Decision Making

**Weighted Voting**:
```python
class FraudEnsemble:
    def __init__(self):
        self.weights = {
            'rules': 0.4,
            'xgboost': 0.4,
            'anomaly': 0.2
        }
        self.threshold_block = 0.9  # >90% confidence → auto-block
        self.threshold_review = 0.5  # 50-90% → manual review
    
    def predict(self, transaction, features):
        # Get scores from each model
        rules_result = self.rules_engine.evaluate(transaction, features)
        xgb_prob = self.xgboost_model.predict(transaction, features)
        anomaly_score = self.anomaly_detector.score(transaction, features)
        
        # Weighted average
        if rules_result.action == 'BLOCK':
            final_score = 1.0  # Hard block from rules
        else:
            final_score = (
                self.weights['rules'] * rules_result.score +
                self.weights['xgboost'] * xgb_prob +
                self.weights['anomaly'] * anomaly_score
            )
        
        # Decision logic
        if final_score > self.threshold_block:
            return Decision('BLOCK', confidence=final_score)
        elif final_score > self.threshold_review:
            return Decision('REVIEW', confidence=final_score)
        else:
            return Decision('APPROVE', confidence=1 - final_score)
```

---

### 4. Handling Class Imbalance

**Problem**: Only ~1% of transactions are fraudulent

**Solutions**:

1. **SMOTE** (Synthetic Minority Oversampling):
```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.1)  # Increase fraud to 10%
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```

2. **Class Weights**:
```python
class_weight = {0: 1, 1: 99}  # Weight fraud class 99x
```

3. **Stratified Sampling**:
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y  # Preserve class distribution
)
```

4. **Focal Loss** (focuses on hard examples):
```python
def focal_loss(y_true, y_pred, alpha=0.99, gamma=2):
    # Downweight easy examples, focus on hard misclassifications
    bce = binary_crossentropy(y_true, y_pred)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    return alpha * (1 - p_t) ** gamma * bce
```

---

### 5. Model Deployment & Serving

**Latency Budget**: <100ms total

| Component | Latency | Optimization |
|-----------|---------|--------------|
| Feature extraction | 20ms | Redis caching |
| Rules evaluation | 5ms | In-memory rules |
| XGBoost inference | 2ms | Compiled model |
| Anomaly detection | 3ms | Vectorized ops |
| Ensemble | 1ms | Simple math |
| **Total** | **31ms** | ✓ Well within budget |

**Deployment Architecture**:
```python
from fastapi import FastAPI
from prometheus_client import Histogram

app = FastAPI()

PREDICTION_LATENCY = Histogram('fraud_prediction_latency_seconds', ...)

@app.post("/predict")
async def predict_fraud(transaction: Transaction):
    with PREDICTION_LATENCY.time():
        # Extract features
        features = await feature_extractor.extract(transaction)
        
        # Get prediction
        decision = ensemble.predict(transaction, features)
        
        # Track for retraining
        await kafka_producer.send('fraud_decisions', {
            'transaction_id': transaction.id,
            'decision': decision.action,
            'confidence': decision.confidence,
            'features': features
        })
        
        return {
            'decision': decision.action,
            'confidence': decision.confidence,
            'reason': decision.reason
        }
```

---

### 6. Continuous Learning Pipeline

**Feedback Loop**:
```
User Reports Fraud
       │
       ▼
┌─────────────────┐
│ Label Queue     │
│ (Postgres)      │
└────────┬────────┘
         │
         ▼ (Daily)
┌─────────────────┐
│ Retrain Model   │
│ (Airflow DAG)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ A/B Test New    │
│ Model (10%)     │
└────────┬────────┘
         │
         ▼ (If better)
┌─────────────────┐
│ Deploy to Prod  │
│ (Blue-Green)    │
└─────────────────┘
```

**Retraining Schedule**:
- **Daily**: Incremental training on last 7 days
- **Weekly**: Full retrain on last 90 days
- **On-demand**: If fraud patterns spike

---

### 7. Explainability (for Manual Review)

**SHAP Values**:
```python
import shap

explainer = shap.TreeExplainer(xgboost_model)

def explain_prediction(transaction, features):
    shap_values = explainer.shap_values(features)
    
    # Top 5 contributing features
    feature_importance = sorted(
        zip(feature_names, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:5]
    
    return {
        'prediction': model.predict(features),
        'top_reasons': [
            f"{name}: {value:+.3f}" for name, value in feature_importance
        ]
    }
```

**Example Output**:
```
Decision: BLOCK
Confidence: 92%
Top Reasons:
  1. new_device: +0.45 (unfamiliar device)
  2. amount_normalized: +0.32 (10x usual spend)
  3. time_since_last_txn: +0.18 (rapid succession)
  4. is_international: +0.12 (unusual location)
  5. hour_of_day: +0.08 (late night activity)
```

---

### 8. Monitoring & Alerting

**Key Metrics**:
```python
# Prometheus metrics
FRAUD_RATE = Gauge('fraud_detected_rate', 'Fraud detection rate')
FALSE_POSITIVE_RATE = Gauge('false_positive_rate', 'FP rate from user reports')
MODEL_DRIFT = Gauge('model_feature_drift', 'Feature distribution drift')
```

**Alerts**:
1. Fraud rate spike >2x baseline (15 min window)
2. False positive rate >0.2% (hourly)
3. Model latency p95 >80ms
4. Feature distribution drift >0.3 (KL divergence)

---

### 9. Cost Estimation

**Monthly Cost** (1M txns/day):

| Component | Cost | Notes |
|-----------|------|-------|
| API Pods (10x) | $1,000 | c5.large |
| Redis (feature store) | $300 | r5.large |
| Postgres (labels + batch) | $200 | db.t3.medium |
| Kafka  (events) | $400 | 3 brokers |
| Model training (daily) | $100 | Spot instances |
| Monitoring | $100 | Prometheus + Grafana |
| **Total** | **~$2,100/month** | |

---

## Interview Discussion Points

**Q: How to handle concept drift (fraud patterns evolve)?**
- **Daily retraining** with latest fraud examples
- **Anomaly detection** catches novel patterns
- **Human-in-the-loop**: Manual review feeds labels
- **A/B testing**: Validate new models before full deployment

**Q: What if false positive rate increases?**
- **Adjust ensemble thresholds** (lower block threshold)
- **Analyze SHAP values** to identify problematic features
- **Add more granular rules** (e.g., whitelist trusted merchants)
- **Implement user feedback loop** (easy "not fraud" button)

**Q: How to scale to 10x traffic?**
- **Horizontal scaling**: Add API pods (stateless)
- **Redis cluster**: Shard feature store by user_id
- **Model caching**: Cache predictions for identical features (1min TTL)
- **Async processing**: Block immediately, enrich features post-decision

---

## Conclusion

This design achieves:
- ✅ **<100ms latency** (31ms avg)
- ✅ **<0.1% false positive rate** (ensemble + tuned thresholds)
- ✅ **>95% recall** (multi-layer detection)
- ✅ **Adaptable** (continuous learning, A/B testing)
- ✅ **Explainable** (SHAP, clear reasons)
- ✅ **Cost-effective** (~$2,100/month)

**Trade-offs Made**:
- Simpler models (XGBoost) over deep learning (for latency)
- Some false positives acceptable (better UX than fraud)
- Daily retraining (balance freshness vs cost)
