# System Design: A/B Testing Framework for ML Models

## Problem Statement

Design an A/B testing framework for machine learning models to:
- **Compare multiple model variants** simultaneously
- **Ensure statistical significance** before making decisions
- **Minimize user impact** from experimental models
- Support **multi-armed bandit** for adaptive testing
- Provide **real-time dashboards** for monitoring
- Handle **10M+ daily active users**

---

## High-Level Architecture

```
User Request
     │
     ▼
┌──────────────────────────────┐
│  Experiment Assignment       │
│  Service                     │
│  (Consistent Hashing)        │
└──────────┬───────────────────┘
           │
  ┌────────┴────────┬──────────┐
  ▼                 ▼          ▼
┌─────┐         ┌─────┐    ┌─────┐
│Model│         │Model│    │Model│
│  A  │ (70%)   │  B  │(20%)  C  │(10%)
│(Ctrl)│         │(Var1)   │(Var2)│
└──┬──┘         └──┬──┘    └──┬──┘
   │               │           │
   └───────────────┴───────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │  Event Logging       │
        │  (Kafka)             │
        └──────────┬───────────┘
                   │
        ┌──────────┴───────────┐
        ▼                      ▼
┌──────────────┐      ┌──────────────┐
│ Metrics      │      │ Statistical  │
│ Aggregation  │      │ Analysis     │
│ (Spark)      │      │ (Python)     │
└──────┬───────┘      └──────┬───────┘
       │                     │
       └──────────┬──────────┘
                  ▼
        ┌──────────────────┐
        │  Dashboard       │
        │  (Grafana/Custom)│
        └──────────────────┘
```

---

## Component Deep Dive

### 1. Experiment Configuration

**Experiment Definition**:
```python
from dataclasses import dataclass
from typing import List, Dict
from enum import Enum

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"

@dataclass
class ModelVariant:
    name: str
    model_id: str
    traffic_allocation: float  # 0.0 to 1.0
    
@dataclass
class Experiment:
    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus
    
    # Models being tested
    control: ModelVariant  # Baseline model
    variants: List[ModelVariant]  # Experimental models
    
    # Experiment parameters
    start_date: str
    end_date: str
    
    # Success metrics
    primary_metric: str  # e.g., "click_through_rate"
    secondary_metrics: List[str]  # e.g., ["revenue", "engagement"]
    
    # Statistical parameters
    minimum_sample_size: int  # Per variant
    confidence_level: float  # e.g., 0.95
    minimum_detectable_effect: float  # e.g., 0.05 (5% improvement)
    
    # Safety guardrails
    max_users_impacted: int
    alert_thresholds: Dict[str, float]  # e.g., {"error_rate": 0.05}

# Example
experiment = Experiment(
    experiment_id="exp_2026_001",
    name="New Recommendation Algorithm",
    description="Testing collaborative filtering vs matrix factorization",
    status=ExperimentStatus.RUNNING,
    control=ModelVariant("baseline", "model_v1", 0.7),
    variants=[
        ModelVariant("cf", "model_v2", 0.2),
        ModelVariant("mf", "model_v3", 0.1)
    ],
    start_date="2026-01-04",
    end_date="2026-01-18",
    primary_metric="click_through_rate",
    secondary_metrics=["revenue_per_user", "session_duration"],
    minimum_sample_size=10000,
    confidence_level=0.95,
    minimum_detectable_effect=0.05
)
```

---

### 2. User Assignment Strategy

**Consistent Hashing** (ensures users always see same variant):

```python
import hashlib

class ExperimentAssigner:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.variants = [experiment.control] + experiment.variants
        
        # Precompute cumulative weights
        self.cumulative_weights = []
        total = 0
        for variant in self.variants:
            total += variant.traffic_allocation
            self.cumulative_weights.append(total)
    
    def assign_variant(self, user_id: str) -> ModelVariant:
        """Assign user to a model variant consistently."""
        # Hash user_id + experiment_id for consistency
        hash_input = f"{user_id}:{self.experiment.experiment_id}"
        hash_value = int(hashlib.md5(hash_input.encode()).hexdigest(), 16)
        
        # Normalize to [0, 1]
        normalized = (hash_value % 10000) / 10000.0
        
        # Find assigned variant
        for i, cumulative_weight in enumerate(self.cumulative_weights):
            if normalized < cumulative_weight:
                return self.variants[i]
        
        return self.variants[-1]  # Fallback
    
    def get_variant_name(self, user_id: str) -> str:
        """Get variant name for logging."""
        return self.assign_variant(user_id).name

# Test assignment
assigner = ExperimentAssigner(experiment)

# Simulate 10K users
assignments = {}
for user_id in range(10000):
    variant = assigner.get_variant_name(str(user_id))
    assignments[variant] = assignments.get(variant, 0) + 1

print("Traffic Distribution:")
for variant, count in sorted(assignments.items()):
    print(f"  {variant}: {count} ({count/10000*100:.1f}%)")

# Expected:
# baseline: ~7000 (70%)
# cf: ~2000 (20%)
# mf: ~1000 (10%)
```

---

### 3. Event Tracking

**Logging Framework**:

```python
from dataclasses import asdict
import json
import time

@dataclass
class ExperimentEvent:
    event_id: str
    timestamp: float
    user_id: str
    experiment_id: str
    variant_name: str
    
    # Context
    session_id: str
    device_type: str
    
    # Event data
    event_type: str  # "impression", "click", "purchase"
    event_value: float  # e.g., revenue for purchase
    
    # Model output
    model_prediction: float
    model_latency_ms: float

class EventLogger:
    def __init__(self, kafka_producer):
        self.kafka = kafka_producer
        self.topic = "ml_experiments"
    
    async def log_event(self, event: ExperimentEvent):
        """Log experiment event to Kafka."""
        message = json.dumps(asdict(event))
        
        await self.kafka.send(
            self.topic,
            key=event.user_id.encode(),  # Partition by user
            value=message.encode()
        )
    
    async def log_model_serving(self, user_id, experiment_id, variant, 
                                  prediction, latency):
        """Log model serving event."""
        event = ExperimentEvent(
            event_id=generate_id(),
            timestamp=time.time(),
            user_id=user_id,
            experiment_id=experiment_id,
            variant_name=variant.name,
            session_id=get_session_id(),
            device_type=get_device_type(),
            event_type="impression",
            event_value=0.0,
            model_prediction=prediction,
            model_latency_ms=latency
        )
        
        await self.log_event(event)
    
    async def log_user_action(self, user_id, experiment_id, action_type, value=0.0):
        """Log user action (click, purchase, etc.)."""
        variant = get_user_variant(user_id, experiment_id)
        
        event = ExperimentEvent(
            event_id=generate_id(),
            timestamp=time.time(),
            user_id=user_id,
            experiment_id=experiment_id,
            variant_name=variant.name,
            session_id=get_session_id(),
            device_type=get_device_type(),
            event_type=action_type,
            event_value=value,
            model_prediction=0.0,
            model_latency_ms=0.0
        )
        
        await self.log_event(event)
```

---

### 4. Metrics Computation

**Real-Time Aggregation** (Spark Streaming):

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, avg, sum as spark_sum, window

spark = SparkSession.builder.appName("ExperimentMetrics").getOrCreate()

# Read from Kafka
events = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", "ml_experiments") \
    .load()

# Parse JSON
from pyspark.sql.types import StructType, StructField, StringType, DoubleType

schema = StructType([
    StructField("user_id", StringType()),
    StructField("experiment_id", StringType()),
    StructField("variant_name", StringType()),
    StructField("event_type", StringType()),
    StructField("event_value", DoubleType()),
    StructField("timestamp", DoubleType())
])

parsed_events = events.selectExpr("CAST(value AS STRING)") \
    .select(from_json(col("value"), schema).alias("data")) \
    .select("data.*")

# Compute metrics (5-minute windows)
metrics = parsed_events \
    .withColumn("timestamp", to_timestamp(col("timestamp"))) \
    .groupBy(
        window("timestamp", "5 minutes"),
        "experiment_id",
        "variant_name"
    ) \
    .agg(
        count("user_id").alias("total_impressions"),
        spark_sum(when(col("event_type") == "click", 1).otherwise(0)).alias("clicks"),
        spark_sum(when(col("event_type") == "purchase", 1).otherwise(0)).alias("purchases"),
        spark_sum("event_value").alias("total_revenue"),
        avg(when(col("event_type") == "impression", col("model_latency_ms"))).alias("avg_latency_ms")
    ) \
    .withColumn("ctr", col("clicks") / col("total_impressions")) \
    .withColumn("conversion_rate", col("purchases") / col("total_impressions")) \
    .withColumn("revenue_per_user", col("total_revenue") / col("total_impressions"))

# Write to database
metrics.writeStream \
    .format("jdbc") \
    .option("url", "jdbc:postgresql://localhost:5432/experiments") \
    .option("dbtable", "experiment_metrics") \
    .option("checkpointLocation", "/tmp/checkpoints") \
    .start()
```

---

### 5. Statistical Significance Testing

**Two-sample t-test** for comparing variants:

```python
import numpy as np
from scipy import stats

class StatisticalAnalyzer:
    def __init__(self, confidence_level=0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
    
    def ttest_variants(self, control_data, variant_data):
        """
        Perform two-sample t-test.
        
        Returns:
            - p_value: Probability of observing difference by chance
            - ci_lower, ci_upper: Confidence interval for difference
            - is_significant: Whether difference is statistically significant
        \"\"\"\n        # Two-sided t-test
        t_stat, p_value = stats.ttest_ind(variant_data, control_data)
        
        # Compute effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(control_data) + np.var(variant_data)) / 2)
        cohens_d = (np.mean(variant_data) - np.mean(control_data)) / pooled_std
        
        # Confidence interval for difference
        diff_mean = np.mean(variant_data) - np.mean(control_data)
        diff_std = np.sqrt(np.var(variant_data)/len(variant_data) + 
                           np.var(control_data)/len(control_data))
        
        z_critical = stats.norm.ppf(1 - self.alpha/2)
        ci_lower = diff_mean - z_critical * diff_std
        ci_upper = diff_mean + z_critical * diff_std
        
        return {
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'effect_size': cohens_d,
            'mean_difference': diff_mean,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'control_mean': np.mean(control_data),
            'variant_mean': np.mean(variant_data)
        }
    
    def required_sample_size(self, baseline_conversion, mde, power=0.8):
        """
        Calculate required sample size per variant.
        
        Args:
            baseline_conversion: Current conversion rate (e.g., 0.10 for 10%)
            mde: Minimum detectable effect (e.g., 0.05 for 5% relative improvement)
            power: Statistical power (1 - beta)
        \"\"\"\n        from statsmodels.stats.power import zt_ind_solve_power
        
        delta = baseline_conversion * mde
        
        n = zt_ind_solve_power(
            effect_size=delta / np.sqrt(baseline_conversion * (1 - baseline_conversion)),
            alpha=self.alpha,
            power=power,
            alternative='two-sided'
        )
        
        return int(np.ceil(n))

# Example usage
analyzer = StatisticalAnalyzer()

# Simulate data
control = np.random.binomial(1, 0.10, 50000)  # 10% CTR
variant = np.random.binomial(1, 0.11, 50000)  # 11% CTR (10% relative improvement)

result = analyzer.ttest_variants(control, variant)

print(f"Control CTR: {result['control_mean']:.4f}")
print(f"Variant CTR: {result['variant_mean']:.4f}")
print(f"Difference: {result['mean_difference']:.4f}")
print(f"95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")
print(f"P-value: {result['p_value']:.6f}")
print(f"Significant: {result['is_significant']}")
print(f"Effect size (Cohen's d): {result['effect_size']:.4f}")

# Sample size calculation
required_n = analyzer.required_sample_size(baseline_conversion=0.10, mde=0.05)
print(f"\nRequired sample size per variant: {required_n:,}")
```

---

### 6. Multi-Armed Bandit (Adaptive Testing)

**Thompson Sampling** for adaptive allocation:

```python
from scipy.stats import beta as beta_dist

class ThompsonSamplingBandit:
    \"""
    Multi-armed bandit for adaptive A/B testing.
    Automatically allocates more traffic to better-performing variants.
    \"""
    
    def __init__(self, variants: List[str]):
        self.variants = variants
        
        # Beta distribution parameters (successes + 1, failures + 1)
        self.alpha = {v: 1 for v in variants}  # Prior: Beta(1, 1) = Uniform
        self.beta = {v: 1 for v in variants}
    
    def select_variant(self) -> str:
        \"\"\"Select variant using Thompson sampling.\"\"\"
        # Sample from each variant's Beta distribution
        samples = {
            v: beta_dist.rvs(self.alpha[v], self.beta[v])
            for v in self.variants
        }
        
        # Return variant with highest sample
        return max(samples, key=samples.get)
    
    def update(self, variant: str, reward: float):
        \"\"\"Update distribution based on observed reward.\"\"\"
        if reward > 0:
            self.alpha[variant] += 1  # Success
        else:
            self.beta[variant] += 1   # Failure
    
    def get_win_probability(self, variant: str) -> float:
        \"\"\"Estimate probability this variant is best.\"\"\"
        n_samples = 10000
        counts = {v: 0 for v in self.variants}
        
        for _ in range(n_samples):
            samples = {v: beta_dist.rvs(self.alpha[v], self.beta[v]) 
                      for v in self.variants}
            winner = max(samples, key=samples.get)
            counts[winner] += 1
        
        return counts.get(variant, 0) / n_samples

# Example
bandit = ThompsonSamplingBandit(["control", "variant_a", "variant_b"])

# Simulate 10K users
for _ in range(10000):
    # Select variant
    variant = bandit.select_variant()
    
    # Simulate user interaction (variant_b is actually better)
    true_ctr = {"control": 0.10, "variant_a": 0.09, "variant_b": 0.12}
    reward = 1 if np.random.rand() < true_ctr[variant] else 0
    
    # Update bandit
    bandit.update(variant, reward)

# Print results
print("\\nBandit Learning Results:")
for variant in bandit.variants:
    mean_ctr = bandit.alpha[variant] / (bandit.alpha[variant] + bandit.beta[variant])
    win_prob = bandit.get_win_probability(variant)
    print(f"{variant}:")
    print(f"  Estimated CTR: {mean_ctr:.4f}")
    print(f"  Win Probability: {win_prob:.2%}")
    print(f"  Traffic Received: {bandit.alpha[variant] + bandit.beta[variant] - 2}")
```

---

### 7. Safety Guardrails

**Automated Experiment Monitoring**:

```python
class ExperimentMonitor:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.alert_triggered = False
    
    def check_guardrails(self, metrics: Dict[str, Dict[str, float]]):
        \"\"\"
        Check if any variant violates safety guardrails.
        
        Args:
            metrics: {variant_name: {metric_name: value}}
        \"\"\"\n        alerts = []
        
        for variant_name, variant_metrics in metrics.items():
            # Skip control
            if variant_name == self.experiment.control.name:
                continue
            
            # Check each guardrail
            for metric, threshold in self.experiment.alert_thresholds.items():
                if metric in variant_metrics:
                    value = variant_metrics[metric]
                    
                    # Bad metrics (higher is worse)
                    if metric in ["error_rate", "latency_p95"]:
                        if value > threshold:
                            alerts.append({
                                'variant': variant_name,
                                'metric': metric,
                                'value': value,
                                'threshold': threshold,
                                'severity': 'critical'
                            })
        
        return alerts
    
    async def auto_pause_if_needed(self, alerts):
        \"\"\"Automatically pause experiment if critical issues detected.\"\"\"
        critical_alerts = [a for a in alerts if a['severity'] == 'critical']
        
        if critical_alerts and not self.alert_triggered:
            # Pause experiment
            await pause_experiment(self.experiment.experiment_id)
            
            # Send notifications
            await send_alert(
                title=f"Experiment {self.experiment.name} Auto-Paused",
                message=f"Critical issues detected: {critical_alerts}",
                severity="critical"
            )
            
            self.alert_triggered = True
```

---

### 8. Dashboard & Visualization

**Metrics Dashboard** (Streamlit example):

```python
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.title("A/B Testing Dashboard")

# Select experiment
experiments = load_experiments()
selected_exp = st.selectbox("Select Experiment", experiments)

# Load metrics
metrics_df = load_experiment_metrics(selected_exp)

# Traffic Distribution
st.subheader("Traffic Distribution")
traffic = metrics_df.groupby('variant_name')['impressions'].sum()
fig = go.Figure(data=[go.Pie(labels=traffic.index, values=traffic.values)])
st.plotly_chart(fig)

# CTR Comparison
st.subheader("Click-Through Rate")
ctr_data = metrics_df.groupby('variant_name').agg({
    'clicks': 'sum',
    'impressions': 'sum'
})
ctr_data['ctr'] = ctr_data['clicks'] / ctr_data['impressions']

fig = go.Figure(data=[
    go.Bar(x=ctr_data.index, y=ctr_data['ctr'], 
           text=[f"{v:.2%}" for v in ctr_data['ctr']])
])
fig.update_layout(yaxis_title="CTR")
st.plotly_chart(fig)

# Statistical Significance
st.subheader("Statistical Test")
control_clicks = get_clicks(selected_exp, "control")
for variant in get_variants(selected_exp):
    variant_clicks = get_clicks(selected_exp, variant)
    
    result = analyzer.ttest_variants(control_clicks, variant_clicks)
    
    st.write(f"**{variant} vs Control**")
    st.write(f"- Difference: {result['mean_difference']*100:.2f}%")
    st.write(f"- P-value: {result['p_value']:.4f}")
    st.write(f"- Significant: {'✅ Yes' if result['is_significant'] else '❌ No'}")
    st.write(f"- 95% CI: [{result['ci_lower']*100:.2f}%, {result['ci_upper']*100:.2f}%]")
```

---

### 9. Cost Estimation

**Monthly Cost** (10M DAU, 5 concurrent experiments):

| Component | Cost | Notes |
|-----------|------|-------|
| Kafka Cluster (3 brokers) | $600 | m5.large |
| Spark Streaming (5 workers) | $2,000 | c5.2xlarge |
| PostgreSQL (metrics DB) | $500 | db.r5.xlarge |
| Redis (assignment cache) | $300 | r5.large |
| Monitoring (Grafana Cloud) | $200 | Pro tier |
| S3 Storage (event logs) | $100 | 1TB/month |
| **Total** | **~$3,700/month** | |

---

## Interview Discussion Points

**Q: How to prevent peeking (premature analysis)?**
- **Sequential testing**: Adjust p-values for multiple looks (Bonferroni)
- **Fixed horizon**: Only analyze after reaching sample size
- **Bayesian approach**: Continuously monitor posterior probabilities

**Q: What if variants have different conversion windows?**
- **Use survival analysis**: Kaplan-Meier curves
- **Time-to-event metrics**: Median time to purchase
- **Cohort analysis**: Track users over 7/14/30 days

**Q: How to handle novelty effect?**
- **Run longer experiments**: 2-4 weeks minimum
- **Monitor day-over-day trends**: Check if effect decays
- **Holdout group**: Keep 5% always on control

**Q: Multiple comparison problem (10+ variants)?**
- **Bonferroni correction**: Adjust alpha = 0.05 / n_variants
- **False Discovery Rate** (FDR): Control expected proportion of false positives
- **Hierarchical testing**: Test groups, then individual variants

---

## Conclusion

This framework provides:
- ✅ **Statistically rigorous** A/B testing
- ✅ **Adaptive allocation** with multi-armed bandits
- ✅ **Real-time monitoring** and auto-pause
- ✅ **Scalable** to 10M+ DAU
- ✅ **Cost-effective** (~$3,700/month)

**Key Decisions**:
- Consistent hashing for assignment
- Kafka + Spark for event processing
- Thompson Sampling for adaptive testing
- Automated guardrails for safety
