# Data Quality Framework for AI/ML Production Systems

*Comprehensive guide for senior AI/ML engineers building robust data governance systems*

**Last Updated**: February 17, 2026  
**Version**: 2.1  
**Target Audience**: Senior AI/ML Engineers, Data Governance Specialists, MLOps Engineers

---

## 1. Introduction to Data Quality in AI/ML Systems

Data quality is the cornerstone of reliable AI/ML systems. Poor data quality directly translates to:
- Degraded model performance (up to 30% accuracy loss in production)
- Increased model drift and concept shift
- Higher operational costs due to retraining and debugging
- Regulatory compliance failures (GDPR, CCPA, HIPAA)

In production AI systems, data quality issues are often more costly than model architecture flaws because:
- Data issues propagate silently through pipelines
- They're harder to detect than model failures
- Remediation requires data engineering effort, not just ML expertise

> **Production Insight**: At scale, 60-80% of ML engineering time is spent on data quality assurance and remediation, not model development.

### Why Data Quality Matters for AI/ML

| Impact Area | Consequence of Poor Data Quality |
|-------------|----------------------------------|
| Model Performance | Training instability, poor generalization, biased predictions |
| Operational Reliability | Unexpected service outages, inconsistent API responses |
| Business Trust | Erosion of stakeholder confidence in AI decisions |
| Compliance Risk | Violations of data governance regulations |
| Cost Efficiency | Increased cloud compute costs from reprocessing |

---

## 2. Comprehensive Data Quality Dimensions Framework

The industry-standard framework for data quality assessment includes seven core dimensions:

### 2.1 Accuracy
*Degree to which data correctly represents the real-world entity*

**Metrics**:
- `accuracy_score = (correct_records / total_records) * 100`
- Threshold: ≥99.5% for critical systems, ≥98% for non-critical

**AI/ML Specific**: 
- Label accuracy for training data
- Ground truth alignment for validation sets
- Measurement error in sensor data

### 2.2 Completeness
*Degree to which all required data is present*

**Metrics**:
- `completeness_rate = (non_null_values / total_possible_values) * 100`
- Null rate per column: `null_count / row_count`

**Thresholds**:
- Primary keys: 100% complete
- Critical features: ≥99.9% complete
- Non-critical features: ≥95% complete

### 2.3 Consistency
*Degree to which data is coherent across systems and over time*

**Types**:
- Cross-system consistency (e.g., customer ID matches across CRM and billing)
- Temporal consistency (e.g., order date ≤ shipment date)
- Referential consistency (foreign key constraints)

**Metric**: `consistency_ratio = (consistent_records / total_records) * 100`

### 2.4 Timeliness
*Degree to which data is available when needed*

**Metrics**:
- Data latency: `current_time - data_timestamp`
- SLA compliance rate: `% of records within SLA window`

**AI/ML Thresholds**:
- Real-time models: < 5 minutes latency
- Batch models: < 24 hours for daily data
- Historical analysis: < 7 days for weekly aggregates

### 2.5 Validity
*Degree to which data conforms to defined formats, types, and business rules*

**Validation Types**:
- Format validation (email regex, phone number patterns)
- Type validation (integer vs float, enum values)
- Range validation (age: 0-120, probability: 0-1)
- Business rule validation (order amount > 0)

### 2.6 Uniqueness
*Degree to which data contains no duplicates*

**Metrics**:
- Duplicate rate: `duplicate_records / total_records`
- Primary key uniqueness violation count

**Thresholds**:
- Primary keys: 0% duplicates allowed
- Business entities: < 0.1% duplicate rate

### 2.7 Integrity
*Degree to which relationships between data elements are maintained*

**Types**:
- Referential integrity (foreign key constraints)
- Semantic integrity (logical relationships)
- Transactional integrity (ACID properties)

**Metric**: `integrity_score = (valid_relationships / total_relationships) * 100`

---

## 3. Data Quality Assessment Methodology

### 3.1 Data Profiling Techniques

#### Statistical Profiling
```python
# Example: PySpark data profiling
from pyspark.sql.functions import *
from pyspark.sql.types import *

def profile_dataset(df):
    stats = {
        'row_count': df.count(),
        'column_count': len(df.columns),
        'null_counts': {col: df.filter(col(df[col]).isNull()).count() 
                       for col in df.columns},
        'distinct_counts': {col: df.select(col).distinct().count() 
                           for col in df.columns},
        'value_ranges': {col: (df.select(min(col), max(col)).first() 
                             if isinstance(df.schema[col].dataType, (IntegerType, FloatType)) 
                             else None) for col in df.columns}
    }
    return stats
```

#### Pattern Profiling
- Regular expression matching for format validation
- N-gram analysis for text data quality
- Entropy calculation for data randomness assessment

#### Distribution Profiling
- Histogram analysis for numerical features
- Frequency tables for categorical features
- Skewness and kurtosis metrics

### 3.2 Anomaly Detection Algorithms

#### Statistical Methods
- **Z-score**: `z = (x - μ) / σ` (threshold: |z| > 3.0)
- **IQR**: Outliers = {x < Q1 - 1.5×IQR or x > Q3 + 1.5×IQR}
- **Mahalanobis distance**: For multivariate outlier detection

#### Machine Learning Methods
```python
# Isolation Forest for anomaly detection
from sklearn.ensemble import IsolationForest

def detect_anomalies(data, contamination=0.01):
    # Remove non-numeric columns for simplicity
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    clf = IsolationForest(contamination=contamination, random_state=42)
    anomalies = clf.fit_predict(data[numeric_cols])
    return anomalies == -1  # -1 indicates anomaly
```

#### Time Series Anomaly Detection
- Seasonal decomposition + residual analysis
- Prophet-based anomaly detection
- LSTM autoencoders for complex patterns

### 3.3 Schema Validation and Constraint Checking

#### Schema Definition (JSON Schema example)
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "type": "object",
  "properties": {
    "user_id": {
      "type": "string",
      "pattern": "^[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}$"
    },
    "timestamp": {
      "type": "string",
      "format": "date-time"
    },
    "feature_vector": {
      "type": "array",
      "items": {"type": "number"},
      "minItems": 100,
      "maxItems": 100
    },
    "label": {
      "type": "integer",
      "minimum": 0,
      "maximum": 1
    }
  },
  "required": ["user_id", "timestamp", "feature_vector", "label"],
  "additionalProperties": false
}
```

#### Constraint Validation Patterns
- **Domain constraints**: Value ranges, enums
- **Referential constraints**: Foreign key relationships
- **Temporal constraints**: Date ordering, validity periods
- **Cardinality constraints**: One-to-one, one-to-many relationships

---

## 4. Data Quality Monitoring and Alerting

### 4.1 Real-time Monitoring Patterns

#### Streaming Monitoring Architecture
```
Data Ingestion → Feature Store → Quality Monitor → Alerting System
       ↑                ↑               ↑
   Raw Data         Processed Data    Quality Metrics
```

**Implementation Patterns**:
- **Window-based monitoring**: Sliding windows (5m, 15m, 1h)
- **Micro-batch monitoring**: Per-batch quality assessment
- **Event-driven monitoring**: Trigger on specific data events

#### Metrics Collection Pipeline
```python
# Example: Real-time quality metrics collector
class DataQualityMonitor:
    def __init__(self, window_size=300):  # 5 minutes in seconds
        self.window_size = window_size
        self.metrics_buffer = deque(maxlen=window_size)
    
    def collect_metrics(self, batch_data):
        metrics = {
            'timestamp': datetime.now(),
            'row_count': batch_data.count(),
            'null_rate': self.calculate_null_rate(batch_data),
            'anomaly_rate': self.detect_anomalies(batch_data),
            'distribution_drift': self.calculate_distribution_drift(batch_data),
            'constraint_violations': self.check_constraints(batch_data)
        }
        self.metrics_buffer.append(metrics)
        return metrics
```

### 4.2 Alerting Thresholds and Escalation Policies

#### Tiered Alerting System
| Severity | Threshold | Response Time | Escalation Path |
|----------|-----------|---------------|-----------------|
| INFO     | Warning level | 24h | Slack channel |
| WARNING  | Moderate deviation | 4h | Email + Slack |
| CRITICAL | Severe degradation | 15m | PagerDuty + Phone call |
| EMERGENCY | System failure | 5m | Executive alert + War room |

**Example Thresholds**:
- Null rate > 5% → WARNING
- Anomaly rate > 10% → CRITICAL
- Distribution drift (KS test p-value < 0.01) → WARNING
- Constraint violations > 0 → CRITICAL (for primary keys)

### 4.3 Integration with Observability Systems

#### Prometheus/Grafana Integration
```yaml
# prometheus_rules.yml
groups:
- name: data_quality_rules
  rules:
  - alert: HighNullRate
    expr: data_null_rate{job="feature_pipeline"} > 0.05
    for: 15m
    labels:
      severity: warning
    annotations:
      summary: "High null rate detected in {{ $labels.job }}"
      description: "Null rate is {{ $value }}% (threshold: 5%)"

  - alert: DistributionDrift
    expr: data_kl_divergence{job="model_input"} > 0.5
    for: 30m
    labels:
      severity: critical
    annotations:
      summary: "Significant distribution drift detected"
```

#### OpenTelemetry Integration
- Custom metrics for data quality dimensions
- Trace context propagation through data pipelines
- Log correlation with data quality events

---

## 5. Data Quality Remediation Strategies

### 5.1 Automated Correction Patterns

#### Rule-based Correction
```python
def auto_correct_data(df):
    # Fix common formatting issues
    df = df.withColumn("email", trim(lower(col("email"))))
    df = df.withColumn("phone", regexp_replace(col("phone"), r"[^0-9]", ""))
    
    # Impute missing values based on business rules
    df = df.fillna({
        "age": 35,  # Default age
        "category": "unknown",  # Default category
        "score": df.agg(avg("score")).first()[0]  # Mean imputation
    })
    
    # Validate and correct outliers
    quantiles = df.approxQuantile("amount", [0.01, 0.99], 0.01)
    df = df.filter((col("amount") >= quantiles[0]) & (col("amount") <= quantiles[1]))
    
    return df
```

#### ML-based Correction
- Autoencoders for reconstruction-based correction
- Generative models for missing value imputation
- Transfer learning for domain-specific correction

### 5.2 Human-in-the-Loop Workflows

#### Active Learning for Data Correction
```
Automated Detection → Confidence Scoring → Human Review Queue → 
Feedback Loop → Model Retraining
```

**Workflow Implementation**:
1. **Confidence threshold**: Records with correction confidence < 0.85 go to human review
2. **Prioritization**: High-impact records (model inputs, customer data) prioritized
3. **Review interface**: Web-based annotation tool with context
4. **Feedback integration**: Human corrections used to retrain correction models

#### Case Management System
```python
class DataCorrectionCase:
    def __init__(self, record_id, issue_type, confidence, priority):
        self.record_id = record_id
        self.issue_type = issue_type  # 'null', 'outlier', 'inconsistent'
        self.confidence = confidence  # 0.0-1.0
        self.priority = priority  # 'low', 'medium', 'high', 'critical'
        self.status = 'pending_review'
        self.assignee = None
        self.resolution = None
        self.timestamp = datetime.now()
```

### 5.3 Data Lineage-Based Root Cause Analysis

#### Lineage Tracking Architecture
```
Raw Data → ETL Jobs → Feature Store → Model Training → Predictions
    ↑           ↑              ↑               ↑             ↑
Lineage Graph ←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←←
```

**Root Cause Analysis Steps**:
1. **Trace backward**: From quality issue to source system
2. **Identify transformation points**: Where quality degraded
3. **Analyze dependencies**: Which upstream systems contributed
4. **Quantify impact**: How much each source contributed to the issue

#### Lineage Query Example
```sql
-- Find all transformations that affected a specific field
SELECT 
    job_name,
    transformation_type,
    input_sources,
    output_fields,
    quality_metrics_before,
    quality_metrics_after
FROM data_lineage
WHERE output_field = 'customer_risk_score'
  AND timestamp > '2026-02-16 00:00:00'
ORDER BY timestamp DESC;
```

---

## 6. AI/ML Specific Data Quality Considerations

### 6.1 Embedding Quality Metrics

#### Embedding Space Quality
- **Intra-class cohesion**: Distance between similar embeddings
- **Inter-class separation**: Distance between different class embeddings
- **Dimensionality utilization**: Effective rank of embedding matrix

**Metrics**:
- Silhouette score: > 0.5 for good clustering
- Davies-Bouldin index: < 1.0 for good separation
- Embedding entropy: Measures information content

#### Embedding Drift Detection
```python
def detect_embedding_drift(current_embeddings, baseline_embeddings, threshold=0.1):
    # Calculate cosine similarity between distributions
    current_mean = np.mean(current_embeddings, axis=0)
    baseline_mean = np.mean(baseline_embeddings, axis=0)
    
    cosine_sim = np.dot(current_mean, baseline_mean) / (
        np.linalg.norm(current_mean) * np.linalg.norm(baseline_mean)
    )
    
    # Calculate KL divergence for distribution shape
    kl_div = scipy.stats.entropy(
        current_embeddings.flatten(), 
        baseline_embeddings.flatten()
    )
    
    return {
        'cosine_similarity': cosine_sim,
        'kl_divergence': kl_div,
        'drift_detected': cosine_sim < (1 - threshold) or kl_div > threshold
    }
```

### 6.2 Feature Drift Detection

#### Statistical Drift Detection
- **KS test**: Kolmogorov-Smirnov test for distribution differences
- **Chi-square test**: For categorical feature drift
- **Population Stability Index (PSI)**: Standard industry metric

**PSI Calculation**:
```
PSI = Σ[(Actual_% - Expected_%) * ln(Actual_% / Expected_%)]
Thresholds: < 0.1 (no drift), 0.1-0.2 (minor drift), > 0.2 (significant drift)
```

#### ML-based Drift Detection
- Two-sample classifier approach
- Autoencoder reconstruction error
- Deep learning drift detectors (e.g., DeepLog)

### 6.3 Model Input Validation

#### Runtime Input Validation
```python
# Pydantic model for input validation
from pydantic import BaseModel, validator, conlist
from typing import List

class ModelInput(BaseModel):
    user_features: conlist(float, min_items=100, max_items=100)
    context_features: conlist(float, min_items=50, max_items=50)
    metadata: dict
    
    @validator('user_features')
    def validate_user_features(cls, v):
        if any(x < -10 or x > 10 for x in v):
            raise ValueError("Feature values must be in range [-10, 10]")
        if len([x for x in v if abs(x) < 1e-6]) > 10:
            raise ValueError("Too many near-zero features (>10)")
        return v
    
    @validator('metadata')
    def validate_metadata(cls, v):
        required_keys = ['user_id', 'timestamp', 'session_id']
        if not all(key in v for key in required_keys):
            raise ValueError(f"Missing required metadata keys: {required_keys}")
        return v
```

#### Pre-processing Validation
- Check for NaN/Inf values before model inference
- Validate feature scaling ranges
- Ensure categorical encoding consistency

---

## 7. Production Implementation Patterns

### 7.1 CI/CD Integration for Data Quality

#### GitOps for Data Quality
```
Code Changes → CI Pipeline → Data Quality Tests → 
Deployment Approval → Production Deployment
```

**CI Pipeline Stages**:
1. **Static analysis**: Schema validation, constraint checking
2. **Unit tests**: Individual data quality checks
3. **Integration tests**: End-to-end pipeline validation
4. **Performance tests**: Quality check execution time
5. **Approval gates**: Quality thresholds must be met

#### Example GitHub Actions Workflow
```yaml
name: Data Quality Pipeline
on: [push, pull_request]

jobs:
  data_quality:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Run data quality checks
      run: |
        python -m pytest tests/data_quality/ --junitxml=junit.xml
        
    - name: Check quality thresholds
      run: |
        python scripts/check_quality_thresholds.py
        
    - name: Fail if quality below threshold
      if: ${{ failure() }}
      run: |
        echo "Data quality checks failed. See junit.xml for details."
        exit 1
```

### 7.2 Database-Level Constraints and Validations

#### SQL Constraints
```sql
-- Example: Comprehensive table constraints
CREATE TABLE customer_events (
    event_id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    event_type VARCHAR(50) NOT NULL CHECK (event_type IN ('login', 'purchase', 'view')),
    timestamp TIMESTAMP NOT NULL CHECK (timestamp <= NOW()),
    amount DECIMAL(10,2) CHECK (amount >= 0),
    session_id VARCHAR(100) NOT NULL,
    
    -- Referential integrity
    CONSTRAINT fk_user FOREIGN KEY (user_id) REFERENCES users(user_id),
    
    -- Unique constraints
    UNIQUE (event_id, timestamp),
    
    -- Check constraints for business rules
    CHECK (
        CASE 
            WHEN event_type = 'purchase' THEN amount > 0
            ELSE amount = 0
        END
    )
);
```

#### Materialized Views for Quality Monitoring
```sql
-- Daily quality summary view
CREATE MATERIALIZED VIEW daily_quality_summary AS
SELECT 
    DATE(timestamp) as day,
    COUNT(*) as total_records,
    COUNT(CASE WHEN user_id IS NULL THEN 1 END) as null_user_id,
    COUNT(CASE WHEN amount < 0 THEN 1 END) as negative_amounts,
    COUNT(CASE WHEN event_type NOT IN ('login', 'purchase', 'view') THEN 1 END) as invalid_events
FROM customer_events
GROUP BY DATE(timestamp);
```

### 7.3 Data Quality SLAs and SLOs

#### Service Level Objectives (SLOs)
| Metric | Target | Measurement Period | Alert Threshold |
|--------|--------|-------------------|-----------------|
| Data completeness | 99.99% | Daily | < 99.9% |
| Data accuracy | 99.95% | Per batch | < 99.5% |
| Processing latency | < 5min | Per record | > 15min |
| Constraint violations | 0 | Continuous | > 0 |
| Drift detection | < 1hr | Continuous | > 4hrs |

#### SLA Documentation Template
```markdown
## Data Quality SLA: Customer Event Pipeline

**Service**: Real-time customer event processing
**Owner**: Data Engineering Team
**Measurement**: Daily at 00:05 UTC

### SLOs
- **Completeness**: 99.99% of expected events processed
- **Accuracy**: 99.95% of events have correct user_id mapping
- **Timeliness**: 99.9% of events processed within 5 minutes
- **Consistency**: 100% referential integrity with user master data

### Error Budget
- Monthly error budget: 0.01% (≈ 4.3 minutes downtime per month)
- When budget exhausted: Automatic deployment freeze

### Remediation Process
1. Immediate notification to on-call engineer
2. Root cause analysis within 30 minutes
3. Fix deployed within 2 hours for critical issues
```

---

## 8. Case Studies and Real-World Examples

### 8.1 Financial Services: Fraud Detection System

**Challenge**: High false positive rate due to data quality issues in transaction data.

**Solution**:
- Implemented comprehensive data quality framework with 7 dimensions
- Added real-time monitoring for null rates and outlier detection
- Created automated correction for common formatting issues
- Established human-in-the-loop for high-value transaction review

**Results**:
- False positive rate reduced by 65%
- Data completeness improved from 92% to 99.98%
- Model retraining frequency reduced by 70%

### 8.2 Healthcare: Patient Outcome Prediction

**Challenge**: Model performance degradation due to inconsistent patient demographics data.

**Solution**:
- Implemented schema validation with strict constraints
- Added temporal consistency checks (admission date ≤ discharge date)
- Built lineage tracking to identify source system issues
- Created automated remediation for common data entry errors

**Results**:
- Model AUC improved from 0.72 to 0.89
- Data validation failures reduced by 95%
- Regulatory audit pass rate: 100%

### 8.3 E-commerce: Recommendation Engine

**Challenge**: Recommendation quality degradation due to feature drift and stale embeddings.

**Solution**:
- Implemented embedding quality metrics and drift detection
- Added feature drift monitoring with PSI thresholds
- Created automated re-embedding pipeline triggered by drift
- Integrated quality metrics into CI/CD pipeline

**Results**:
- Click-through rate increased by 22%
- Embedding quality (silhouette score) improved from 0.3 to 0.65
- Drift detection time reduced from 24h to 15min

---

## 9. Tools and Frameworks Comparison

### 9.1 Overview of Major Data Quality Tools

| Tool | Language | Key Features | Best For | Limitations |
|------|----------|--------------|----------|-------------|
| **Great Expectations** | Python | Declarative expectations, rich validation, data docs, integration with MLflow | Complex validation, team collaboration, documentation | Steeper learning curve, Python-only |
| **Deequ** | Scala/Java | AWS integration, scalable, statistical tests, built for Spark | Large-scale data processing, AWS environments | JVM ecosystem, less Python integration |
| **Soda Core** | Python | YAML-based configuration, CLI-first, open source | Simple setups, quick adoption, DevOps integration | Less mature, smaller community |
| **Monte Carlo** | SaaS | Automated lineage, anomaly detection, alerting | Enterprise teams, no-code requirements | Costly, vendor lock-in |
| **Acceldata** | SaaS | Real-time monitoring, AI-powered insights | Real-time analytics, streaming data | Expensive, limited customization |

### 9.2 Implementation Comparison

#### Great Expectations Example
```python
import great_expectations as ge
from great_expectations.core import ExpectationSuite

# Define expectation suite
suite = ExpectationSuite(expectation_suite_name="customer_data_suite")

suite.add_expectation(
    ge.expectation.ExpectColumnValuesToNotBeNull(column="user_id")
)
suite.add_expectation(
    ge.expectation.ExpectColumnValuesToBeBetween(
        column="age", min_value=0, max_value=120
    )
)
suite.add_expectation(
    ge.expectation.ExpectColumnDistinctValuesToEqualSet(
        column="event_type", value_set=["login", "purchase", "view"]
    )
)

# Validate data
df = ge.read_csv("customer_data.csv")
results = df.validate(suite)
```

#### Deequ Example
```scala
import com.amazon.deequ.checks._
import com.amazon.deequ.constraints._

val verificationResult = VerificationSuite()
  .onData(dataset)
  .addCheck(
    Check(CheckLevel.Error, "Data quality checks")
      .hasCompleteness("user_id", _ >= 0.999)
      .hasUniqueness("user_id", _ == 1.0)
      .hasEntropy("category", _ > 0.5)
  )
  .run()
```

#### Soda Core Example
```yaml
# soda.yaml
data_source: my_postgres_db

tables:
  customer_events:
    checks:
      - row_count: ["> 1000"]
      - missing_count:
          column: user_id
          warn: when > 10
          fail: when > 100
      - valid_format:
          column: email
          format: email
```

### 9.3 Tool Selection Guidelines

**Choose Great Expectations when**:
- You need comprehensive validation capabilities
- Team prefers Python ecosystem
- Documentation and collaboration are important
- You want to integrate with existing ML workflows

**Choose Deequ when**:
- You're processing large datasets on Spark
- You're in AWS environment
- You need high-performance validation
- Your team is comfortable with Scala/Java

**Choose Soda Core when**:
- You want simple, YAML-based configuration
- Quick setup is prioritized
- You're using modern DevOps practices
- Budget constraints exist

**Choose SaaS solutions when**:
- You need enterprise-grade support
- Real-time monitoring is critical
- Your team lacks data engineering expertise
- Time-to-value is the primary concern

---

## 10. Best Practices from Production Experience

### 10.1 Foundational Principles

1. **Shift Left**: Integrate data quality checks as early as possible in the pipeline
2. **Measure Everything**: If you can't measure it, you can't improve it
3. **Automate First**: Manual processes don't scale in production
4. **Context Matters**: Quality thresholds should be business-context aware
5. **Iterate Continuously**: Data quality is a journey, not a destination

### 10.2 Implementation Checklist

✅ **Before Production**:
- [ ] Define quality dimensions and thresholds for each dataset
- [ ] Implement schema validation and constraint checking
- [ ] Set up monitoring and alerting infrastructure
- [ ] Create remediation workflows and escalation paths
- [ ] Document data quality SLAs and SLOs

✅ **During Production**:
- [ ] Monitor quality metrics continuously
- [ ] Review alert patterns weekly
- [ ] Update thresholds based on business requirements
- [ ] Conduct quarterly data quality audits
- [ ] Train new team members on quality processes

✅ **Post-Incident**:
- [ ] Perform root cause analysis with lineage tracing
- [ ] Update validation rules to prevent recurrence
- [ ] Review and adjust alerting thresholds
- [ ] Document lessons learned and share across teams

### 10.3 Common Pitfalls and How to Avoid Them

| Pitfall | Impact | Prevention Strategy |
|---------|--------|---------------------|
| **Over-engineering validation** | Slow pipelines, maintenance overhead | Start simple, add complexity only when needed |
| **Ignoring business context** | False positives/negatives | Involve domain experts in threshold setting |
| **Silent failures** | Undetected data corruption | Implement fail-fast validation in critical paths |
| **Lack of ownership** | Quality degradation over time | Assign data quality owners per dataset |
| **No feedback loop** | Quality issues recur | Close the loop: alerts → remediation → prevention |

---

## Appendix A: Reference Implementation Templates

### A.1 Data Quality Dashboard Template
```python
# Streamlit dashboard for data quality monitoring
import streamlit as st
import pandas as pd
import plotly.express as px

def create_quality_dashboard():
    st.title("Data Quality Dashboard")
    
    # Load quality metrics
    metrics_df = load_quality_metrics()
    
    # Summary cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Completeness", f"{metrics_df['completeness'].iloc[-1]:.2%}", 
                f"{metrics_df['completeness'].iloc[-1] - metrics_df['completeness'].iloc[-2]:.2%}")
    col2.metric("Accuracy", f"{metrics_df['accuracy'].iloc[-1]:.2%}", 
                f"{metrics_df['accuracy'].iloc[-1] - metrics_df['accuracy'].iloc[-2]:.2%}")
    col3.metric("Anomaly Rate", f"{metrics_df['anomaly_rate'].iloc[-1]:.2%}", 
                f"{metrics_df['anomaly_rate'].iloc[-1] - metrics_df['anomaly_rate'].iloc[-2]:.2%}")
    col4.metric("Constraint Violations", metrics_df['violations'].iloc[-1], 
                metrics_df['violations'].iloc[-1] - metrics_df['violations'].iloc[-2])
    
    # Trend charts
    fig = px.line(metrics_df, x='timestamp', y=['completeness', 'accuracy', 'anomaly_rate'])
    st.plotly_chart(fig)
```

### A.2 Data Quality Test Suite Template
```python
# pytest test suite for data quality
import pytest
import pandas as pd
from typing import List

def test_data_completeness(df):
    """Test that critical columns have acceptable null rates"""
    critical_columns = ['user_id', 'timestamp', 'event_type']
    for col in critical_columns:
        null_rate = df[col].isnull().mean()
        assert null_rate <= 0.001, f"Null rate for {col} too high: {null_rate:.2%}"

def test_data_accuracy(df):
    """Test that data values conform to business rules"""
    # Age should be reasonable
    assert df['age'].between(0, 120).all(), "Invalid age values detected"
    
    # Amount should be non-negative
    assert (df['amount'] >= 0).all(), "Negative amounts detected"

def test_data_consistency(df):
    """Test cross-field consistency"""
    # Order date should be before shipment date
    if 'order_date' in df.columns and 'shipment_date' in df.columns:
        assert (df['order_date'] <= df['shipment_date']).all(), \
            "Order date after shipment date detected"

def test_data_uniqueness(df):
    """Test for duplicate records"""
    assert df.duplicated().sum() == 0, "Duplicate records detected"
```

### A.3 Data Quality Incident Response Template

**Incident Report Template**:
```
Incident ID: DQ-2026-02-17-001
Timestamp: 2026-02-17 14:30 UTC
Severity: CRITICAL
Affected Systems: Customer Event Pipeline, Recommendation Engine

Symptoms:
- Null rate for user_id increased to 12.3% (threshold: 0.1%)
- Model prediction accuracy dropped from 92% to 78%
- Alert triggered: "High null rate in customer events"

Root Cause:
- ETL job failure in legacy system caused incomplete data export
- Data validation step was bypassed during emergency deployment

Impact:
- 15% of recommendations served with incorrect user context
- Estimated business impact: $24K in lost revenue

Resolution:
- Rolled back problematic ETL job
- Reprocessed last 24 hours of data
- Added additional validation step in pipeline

Preventive Actions:
- Enhanced monitoring for ETL job success rates
- Added circuit breaker for data quality failures
- Updated deployment checklist to include data quality verification
```

---

*This documentation represents industry best practices as of 2026. Regular updates are recommended to incorporate emerging techniques and tools.*