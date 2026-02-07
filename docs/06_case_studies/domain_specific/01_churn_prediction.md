# Case Study 1: Churn Prediction for SaaS Platform

## Executive Summary

**Problem**: A SaaS company faced 15% monthly churn, causing $2M annual revenue loss.

**Solution**: Built an ML churn predictor that flags at-risk customers 30 days ahead.

**Impact**: Reduced churn from 15% -> 9%, saving ~$800K annually and lifting retained MRR by ~6%.

**System design snapshot** (full design in `docs/system_design_solutions/06_churn_prediction.md`):
- SLOs: p99 <150 ms; feature freshness <30 minutes; batch scoring done by 06:00 UTC.
- Scale: ~120M usage events/day across ~12k customers; ~450 at-risk accounts flagged monthly.
- Cost guardrails: < $0.002 per online score; batch pipeline < $65/day all-in.
- Data quality gates: schema contracts via Great Expectations; Airflow fails on drift or
  volume anomalies.
- Reliability: blue/green deploys with 24h shadow traffic; auto rollback if AUC drops >5%
  or alerts spike.

---

## Business Context

### Company Profile
- **Industry**: B2B SaaS (Project Management Software)
- **Customer Base**: 12,000 companies
- **MRR**: $1.8M
- **Problem**: High churn rate driving up customer acquisition costs

### Key Challenges
1. No visibility into which customers are at risk
2. Customer success team reactive, not proactive
3. Lost revenue difficult to recover once churned

---

## Data & Features

### Data Sources
- **Usage Data**:logged events (logins, feature usage, API calls)
- **Billing Data**: Payment history, plan changes, failed payments
- **Support Data**: Ticket volume, response times, NPS scores
- **Firmographic**: Company size, industry, geography

### Feature Engineering

**Behavioral Features** (30-day windows):
```python
# Engagement score
login_frequency = logins_last_30d / 30
feature_usage_breadth = unique_features_used / total_features
api_calls_trend = (api_calls_last_7d - api_calls_prev_7d) / api_calls_prev_7d

# Collaboration signals
active_users_ratio = active_users / total_seats
team_growth = new_users_last_30d - churned_users__last_30d
```

**Health Signals**:
- Failed payment attempts
- Downgrade requests
- Support ticket sentiment (NLP)
- Time since last login

**Total Features**: 47 engineered features

### Data contracts and quality gates
- Schemas for usage, billing, and support tables are versioned and backward-compatible (no drops).
- Great Expectations checks: null ratios, unexpected categories, row-count deltas +/-15%.
- Late-arriving data handled with a T-2 day watermark; partitions reprocessed if completeness <99%.
- Backfills once per month with audit trail; feature store versions features with lineage tags.

---

## Model Development

### Approach Comparison

| Model | Precision | Recall | F1 | AUC-ROC | Notes |
|-------|-----------|--------|-----|---------|-------|
| Logistic Regression | 0.62 | 0.71 | 0.66 | 0.78 | Fast, interpretable |
| Random Forest | 0.68 | 0.74 | 0.71 | 0.83 | Feature importance clear |
| XGBoost | **0.73** | **0.79** | **0.76** | **0.87** | **Selected** |
| Neural Network | 0.70 | 0.76 | 0.73 | 0.85 | Harder to explain |

**Selected Model**: XGBoost
- **Reason**: Best balance of performance and interpretability
- **Threshold**: 0.35 (optimized for recall to catch more at-risk customers)

### Hyperparameter Tuning

```python
best_params = {
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'scale_pos_weight': 2.5  # Handle class imbalance
}
```

### Cross-Validation
- **Strategy**: Time-series split (respects temporal nature)
- **Validation AUC**: 0.86 +/- 0.02
- **Test AUC**: 0.87

---

## Production Deployment

### Architecture

```
Customer DB -> Feature Pipeline (Airflow) -> Redis Feature Store
                                               |
                                               v
                                   Scoring Service (FastAPI)
                                               |
                                               v
                              CRM (Salesforce) + Alert Dashboard
```

### Components

**1. Feature Pipeline** (Airflow DAG, runs daily):
```python
def extract_features(customer_id, end_date):
    # 30-day behavioral window
    usage = get_usage_metrics(customer_id, end_date - 30, end_date)
    billing = get_billing_metrics(customer_id, end_date - 30, end_date)
    support = get_support_metrics(customer_id, end_date - 30, end_date)
    
    features = {
        'login_frequency': usage['logins'] / 30,
        'feature_breadth': len(usage['features_used']) / 50,
        'failed_payments': billing['payment_failures'],
        'support_ticket_sentiment': support['avg_sentiment'],
        # ... 43 more features
    }
    return features
```

**2. Scoring Service** (FastAPI):
```python
@app.post("/predict_churn")
async def predict_churn(customer_id: int):
    features = redis_client.get(f"features:{customer_id}")
    prediction = model.predict_proba([features])[0][1]
    
    if prediction > 0.35:
        # Trigger intervention workflow
        create_salesforce_task(customer_id, risk_score=prediction)
    
    return {"customer_id": customer_id, "churn_risk": prediction}
```

**3. Intervention System**:
- High risk (>0.7): Immediate account manager outreach
- Medium risk (0.35-0.7): Automated personalized email + feature recommendations
- Low risk (<0.35): Monitor

### Operational SLOs and runbook
- p99 online scoring latency <150 ms; FastAPI autoscaled to keep CPU <60%.
- Feature freshness target <30 minutes; requests return 503 if feature timestamp is stale.
- Batch SLA: daily DAG finishes by 06:00 UTC; auto-reruns if failure occurs before 04:00 UTC.
- Runbook highlights:
  - Late partitions: pause scoring for affected tenants, backfill T-1, notify CS via Slack.
  - Redis evictions: fallback to on-demand feature compute and warm caches for top 5% accounts.
  - Salesforce API rate limit: enqueue tasks to SQS and trickle once limits reset.

### Observability and drift control
- Metrics: AUC/PR by cohort, feature freshness lag, batch completeness, intervention send rate.
- Alerts: page if AUC_7d < baseline - 0.05 or feature null ratio jumps >5%; Slack if p95 latency
  >120 ms.
- Deployment safety: shadow for 24h, then blue/green; rollback on KPI regression or alert fatigue
  >2x baseline.

---

## Results & Impact

### Model Performance in Production

**Precision-Recall Trade-off**:
- At threshold 0.35: 73% precision, 79% recall
- **Interpretation**: Of 100 flagged customers, 73 churn; we catch 79% of all churners.

**Monthly Predictions**:
- ~450 customers flagged as at-risk
- Customer success team reaches out to all within 48 hours

### Experiment evidence (A/B)

| Experiment | Variant | Retained MRR uplift | Save rate | Notes |
|------------|---------|---------------------|-----------|-------|
| Outreach channel | Email vs Phone | +3.2% | +4.8 pts | Phone wins for accounts >$500 MRR |
| Incentive | Discount vs Enablement | +0.9% | +1.5 pts | Enablement cheaper; kept as default |
| Threshold | 0.35 vs 0.45 | - | -2.1 pts | 0.35 preserved to protect recall |

Guardrails: NPS delta > -1 allowed; support ticket load capped at +5% during tests.

### Business Impact (6 months post-launch)

| Metric | Before ML | After ML | Improvement |
|--------|-----------|----------|-------------|
| **Monthly Churn Rate** | 15% | 9% | **-40%** |
| **Customer Lifetime Value** | $12,400 | $18,600 | **+50%** |
| **Retention Revenue** | - | $800K/year | **New** |
| **CS Team Efficiency** | Reactive | 30-day proactive | **Qualitative** |

**Revenue Calculation**:
- Prevented churns: ~360 customers/year (30/month)
- Avg MRR per customer: $150
- Annual impact: 360 x $150 x 12 = **$648K** saved
- With LTV multiplier (avg stays 12 months): **$800K**

### Security and compliance
- PII fields tokenized at ingestion; feature store holds hashed customer IDs and non-PII aggregates.
- RBAC: data eng (write), data sci (train/promote), CS dashboards read-only via API gateway.
- Audit logs for predictions and interventions; GDPR/CCPA deletions propagate within 30 days.
- Retention: raw events 180 days; derived aggregates 2 years with storage lifecycle policies.

### Feature Importance (Top 10)

```
1. login_frequency_30d          0.18
2. api_calls_trend_7d           0.14
3. failed_payment_attempts      0.12
4. support_ticket_sentiment     0.09
5. active_users_ratio           0.08
6. feature_usage_breadth        0.07
7. days_since_last_login        0.06
8. team_growth_30d              0.05
9. plan_downgrade_request       0.05
10. nps_score                   0.04
```

**Insights**:
- **Login frequency** is #1 signal (disengagement clear indicator)
- **Billing issues** (failed payments) highly predictive
- **Team dynamics** matter (active users ratio, team growth)

---

## Challenges & Solutions

### Challenge 1: Class Imbalance
- **Problem**: Only 15% of customers churn (85% negative class)
- **Solution**: 
  - SMOTE oversampling during training
  - `scale_pos_weight=2.5` in XGBoost
  - Precision-recall optimization instead of accuracy

### Challenge 2: Concept Drift
- **Problem**: Customer behavior changes over time (new features launched, market shifts)
- **Solution**:
  - Weekly model retraining with latest 12 months of data
  - Monitoring pipeline tracks AUC, precision, recall over time
  - Alert if performance drops >5%

### Challenge 3: Interpretability for CS Team
- **Problem**: Sales team needs to understand *why* a customer is flagged
- **Solution**:
  - SHAP values for each prediction (top 5 contributing features)
  - Dashboard shows "risk drivers" per customer
  - Training session for CS team on model insights

---

## Lessons Learned

### What Worked

1. **Feature Engineering > Model Selection**
   - Moving from 15 raw features -> 47 engineered features improved AUC by 0.12
   - Behavioral trends (7-day vs 30-day) more predictive than point-in-time stats

2. **Time-Series Validation Critical**
   - Random split gave overly optimistic results (AUC 0.92)
   - Time-series split revealed true performance (AUC 0.87)

3. **Business-Aligned Threshold**
   - Default 0.5 threshold missed too many churners (recall 0.58)
   - Lowering to 0.35 caught 79% of churners, acceptable precision

### What Didn't Work

1. **Deep Learning Overkill**
   - Tried LSTM for sequential behavior modeling
   - Marginal improvement (+0.02 AUC) not worth complexity
   - XGBoost + hand-crafted time-based features sufficient

2. **Real-Time Scoring Unnecessary**
   - Initially built real-time scoring API
   - Daily batch scoring sufficient (churn is gradual, not sudden)
   - Saved infrastructure costs

---

## Technical Implementation

### Code Snippet: Feature Engineering

```python
import pandas as pd
import numpy as np

def create_behavioral_features(user_events: pd.DataFrame, window_days: int = 30) -> dict:
    """
    Create behavioral features from user event logs.
    
    Args:
        user_events: DataFrame with columns [user_id, event_type, timestamp]
        window_days: Lookback window
    
    Returns:
        Dictionary of engineered features
    """
    end_date = user_events['timestamp'].max()
    start_date = end_date - pd.Timedelta(days=window_days)
    
    window_events = user_events[user_events['timestamp'] >= start_date]
    
    features = {}
    
    # Engagement metrics
    features['login_frequency'] = (window_events[window_events['event_type'] == 'login'].shape[0] 
                                   / window_days)
    
    # Feature usage breadth
    feature_mask = window_events['event_type'].str.contains('feature_')
    features_used = window_events[feature_mask]['event_type'].nunique()
    features['feature_breadth'] = features_used / 50  # 50 total features in product
    
    # Trend analysis (last 7d vs previous 7d)
    last_7d = window_events[window_events['timestamp'] >= end_date - pd.Timedelta(days=7)]
    prev_7d = window_events[(window_events['timestamp'] >= end_date - pd.Timedelta(days=14)) &
                           (window_events['timestamp'] < end_date - pd.Timedelta(days=7))]
    
    last_7d_count = last_7d.shape[0]
    prev_7d_count = prev_7d.shape[0]
    features['activity_trend'] = (last_7d_count - prev_7d_count) / (prev_7d_count + 1)  # Avoid /0
    
    return features
```

### Model Training Pipeline

```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score, classification_report

# Time-series cross-validation
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred_proba)
    print(f"Validation AUC: {auc:.3f}")

# Final model on all data
final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X, y)
final_model.save_model("churn_model_v1.json")
```

---

## Next Steps & Future Improvements

### Short-Term (Q1 2026)
- [ ] A/B test intervention strategies (email vs phone outreach)
- [ ] Add customer health score to product UI (self-service)
- [ ] Expand to predict *when* churn will happen (survival analysis)

### Medium-Term (Q2-Q3 2026)
- [ ] Causal inference to understand *why* (not just predict)
- [ ] Personalized recommendations to improve health score
- [ ] Real-time alerts for sudden drops in engagement

### Long-Term (2027)
- [ ] Prescriptive analytics: "What actions will reduce churn by X%?"
- [ ] Multi-model stacking (XGBoost + neural net ensemble)

---

## Conclusion

This churn prediction system demonstrates end-to-end ML engineering:
- **Data Engineering**: 47 behavioral features from multiple sources
- **Modeling**: XGBoost optimized for business metric (recall > precision)
- **Production**: Daily batch scoring with Airflow + FastAPI
- **Impact**: $800K annual revenue saved, 40% churn reduction

**Key takeaway**: Feature engineering and business alignment mattered more than model complexity.

Architecture and ops blueprint: `docs/system_design_solutions/06_churn_prediction.md`.

---

**Contact**: Implementation details in `src/ml/churn_prediction.py`.
Notebooks: `notebooks/case_studies/churn_prediction.ipynb`
