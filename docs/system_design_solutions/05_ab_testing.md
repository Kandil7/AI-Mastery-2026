# System Design: A/B Testing Platform for ML Models

## Problem Statement

Design an A/B testing platform for ML models that can:
- Support experiments on 10M daily active users
- Enable safe rollout of new model versions
- Provide statistical significance testing
- Support multiple concurrent experiments
- Minimize performance impact (<5ms overhead)
- Handle traffic splitting with consistent user assignment
- Generate automated experiment reports

---

## High-Level Architecture

```
┌────────────────┐
│   User Request │
└───────┬────────┘
        │
        ▼
┌──────────────────────────────────────┐
│      Experiment Assignment Service    │
│  - Consistent hashing by user_id     │
│  - Experiment config cache            │
│  - Feature flags integration          │
└────────┬─────────────────────────────┘
         │
         ├────────────┬────────────┐
         ▼            ▼            ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ Control  │ │Treatment │ │Treatment │
  │ (Model   │ │ (Model   │ │ (Model   │
  │  v1.0)   │ │  v2.0)   │ │  v3.0)   │
  │  90%     │ │  5%      │ │  5%      │
  └────┬─────┘ └────┬─────┘ └────┬─────┘
       │            │            │
       └────────────┴────────────┘
                    │
                    ▼
       ┌────────────────────────┐
       │   Event Logging        │
       │   (Kafka / Kinesis)    │
       └────────┬───────────────┘
                │
       ┌────────┴────────┐
       ▼                 ▼
┌──────────────┐  ┌──────────────┐
│  Analytics   │  │  Real-time   │
│  Pipeline    │  │  Dashboard   │
│  (Spark)     │  │  (Grafana)   │
└──────────────┘  └──────────────┘
```

---

## Component Deep Dive

### 1. Experiment Configuration Management

**Experiment Config Schema**:

```python
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum

class ExperimentStatus(Enum):
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"

@dataclass
class ExperimentConfig:
    experiment_id: str
    name: str
    description: str
    status: ExperimentStatus
    
    # Traffic allocation
    variants: Dict[str, float]  # {"control": 0.9, "treatment_a": 0.05, "treatment_b": 0.05}
    
    # Targeting
    user_filter: Dict  # {"country": ["US", "CA"], "platform": ["ios", "android"]}
    start_date: datetime
    end_date: datetime
    
    # Success metrics
    primary_metric: str  # e.g., "click_through_rate"
    secondary_metrics: List[str]  # e.g., ["revenue_per_user", "session_duration"]
    
    # Statistical parameters
    minimum_sample_size: int  # e.g., 10000 per variant
    alpha: float  # Significance level (e.g., 0.05)
    power: float  # Statistical power (e.g., 0.8)
    mde: float  # Minimum detectable effect (e.g., 0.02 = 2% uplift)
```

**Config Storage**:
```python
class ExperimentStore:
    def __init__(self):
        self.redis = redis.Redis()
        self.postgres = psycopg2.connect(...)
        
    def get_active_experiments(self, user_context):
        """Get all experiments applicable to this user"""
        # Check cache first
        cache_key = f"exp:active:{hash(user_context)}"
        if cached := self.redis.get(cache_key):
            return json.loads(cached)
        
        # Query database
        experiments = self.postgres.execute("""
            SELECT * FROM experiments 
            WHERE status = 'running'
              AND start_date <= NOW()
              AND end_date >= NOW()
              AND user_matches_filter(%s, user_filter)
        """, (user_context,))
        
        # Cache for 5 minutes
        self.redis.setex(cache_key, 300, json.dumps(experiments))
        return experiments
```

---

### 2. User Assignment Algorithm

**Consistent Hashing**:

```python
import hashlib
from typing import Optional

class ExperimentAssigner:
    def __init__(self):
        self.config_store = ExperimentStore()
        
    def assign_variant(
        self, 
        user_id: str, 
        experiment_id: str,
        experiment_config: ExperimentConfig
    ) -> str:
        """
        Assign user to variant using consistent hashing.
        Same user always gets same variant for same experiment.
        """
        # Create hash from user_id + experiment_id
        hash_input = f"{user_id}:{experiment_id}".encode('utf-8')
        hash_digest = hashlib.md5(hash_input).hexdigest()
        hash_value = int(hash_digest, 16)
        
        # Map to 0-100 range
        bucket = hash_value % 100
        
        # Assign based on cumulative percentages
        cumulative = 0
        for variant_name, percentage in experiment_config.variants.items():
            cumulative += int(percentage * 100)
            if bucket < cumulative:
                return variant_name
        
        return "control"  # Fallback
    
    def get_user_assignments(self, user_id: str, user_context: dict) -> Dict[str, str]:
        """Get all experiment assignments for a user"""
        experiments = self.config_store.get_active_experiments(user_context)
        
        assignments = {}
        for exp in experiments:
            variant = self.assign_variant(user_id, exp.experiment_id, exp)
            assignments[exp.experiment_id] = variant
        
        return assignments
```

**Properties**:
- ✅ **Deterministic**: Same user always gets same variant
- ✅ **Distributed**: No need for centralized assignment storage
- ✅ **Fast**: O(1) hash computation (~0.1ms)
- ✅ **Balanced**: Uniform distribution across variants

---

### 3. Feature Flag Integration

**Combining Experiments with Feature Flags**:

```python
class FeatureFlagService:
    def __init__(self):
        self.assigner = ExperimentAssigner()
        self.config = RemoteConfig()  # LaunchDarkly, Split.io, etc.
        
    def is_enabled(self, feature_name: str, user_id: str, user_context: dict) -> bool:
        """Check if feature is enabled for user"""
        # Get feature config
        feature = self.config.get_feature(feature_name)
        
        if feature.rollout_type == "experiment":
            # Part of A/B test
            variant = self.assigner.assign_variant(
                user_id, 
                feature.experiment_id,
                feature.experiment_config
            )
            return variant in feature.enabled_variants
        
        elif feature.rollout_type == "percentage":
            # Gradual rollout
            hash_val = int(hashlib.md5(f"{user_id}:{feature_name}".encode()).hexdigest(), 16)
            return (hash_val % 100) < feature.rollout_percentage
        
        else:
            # Static on/off
            return feature.enabled
```

---

### 4. Event Logging Pipeline

**Real-time Event Streaming**:

```python
from kafka import KafkaProducer
import json

class ExperimentLogger:
    def __init__(self):
        self.producer = KafkaProducer(
            bootstrap_servers=['kafka:9092'],
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        
    def log_assignment(self, user_id, experiment_id, variant, timestamp):
        """Log experiment assignment"""
        event = {
            "event_type": "assignment",
            "user_id": user_id,
            "experiment_id": experiment_id,
            "variant": variant,
            "timestamp": timestamp.isoformat()
        }
        self.producer.send('experiment_assignments', value=event)
    
    def log_metric(self, user_id, experiment_id, variant, metric_name, value, timestamp):
        """Log metric event (e.g., click, purchase)"""
        event = {
            "event_type": "metric",
            "user_id": user_id,
            "experiment_id": experiment_id,
            "variant": variant,
            "metric_name": metric_name,
            "value": value,
            "timestamp": timestamp.isoformat()
        }
        self.producer.send('experiment_metrics', value=event)

# Usage in API
@app.post("/predict")
async def predict(request: PredictRequest):
    # Assign experiment
    variant = assigner.assign_variant(request.user_id, "model_v2_test", config)
    
    # Log assignment
    logger.log_assignment(
        request.user_id,
        "model_v2_test",
        variant,
        datetime.utcnow()
    )
    
    # Get prediction from appropriate model
    model = "model_v1" if variant == "control" else "model_v2"
    result = await models[model].predict(request.data)
    
    # Log prediction made
    logger.log_metric(
        request.user_id,
        "model_v2_test",
        variant,
        "prediction_made",
        1,
        datetime.utcnow()
    )
    
    return result
```

---

### 5. Statistical Analysis Engine

**Two-Sample t-Test for Continuous Metrics**:

```python
from scipy import stats
import numpy as np

class StatisticalAnalyzer:
    def __init__(self, alpha=0.05):
        self.alpha = alpha  # Significance level
        
    def analyze_continuous_metric(
        self, 
        control_samples: List[float], 
        treatment_samples: List[float]
    ) -> dict:
        """
        Compare continuous metric (e.g., revenue, latency) between variants.
        Returns test results with confidence intervals.
        """
        # Calculate statistics
        control_mean = np.mean(control_samples)
        treatment_mean = np.mean(treatment_samples)
        control_std = np.std(control_samples, ddof=1)
        treatment_std = np.std(treatment_samples, ddof=1)
        
        # Two-sample t-test (Welch's t-test, unequal variances)
        t_stat, p_value = stats.ttest_ind(
            treatment_samples, 
            control_samples,
            equal_var=False
        )
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt((control_std**2 + treatment_std**2) / 2)
        cohens_d = (treatment_mean - control_mean) / pooled_std
        
        # Confidence interval for difference
        n_control = len(control_samples)
        n_treatment = len(treatment_samples)
        se_diff = np.sqrt(control_std**2/n_control + treatment_std**2/n_treatment)
        
        # 95% CI
        critical_value = stats.t.ppf(1 - self.alpha/2, n_control + n_treatment - 2)
        ci_lower = (treatment_mean - control_mean) - critical_value * se_diff
        ci_upper = (treatment_mean - control_mean) + critical_value * se_diff
        
        return {
            "control_mean": control_mean,
            "treatment_mean": treatment_mean,
            "absolute_difference": treatment_mean - control_mean,
            "relative_uplift_pct": (treatment_mean - control_mean) / control_mean * 100,
            "p_value": p_value,
            "is_significant": p_value < self.alpha,
            "cohens_d": cohens_d,
            "confidence_interval_95": [ci_lower, ci_upper],
            "sample_size_control": n_control,
            "sample_size_treatment": n_treatment
        }
```

**Chi-Square Test for Binary Metrics** (e.g., click-through rate):

```python
def analyze_binary_metric(
    self,
    control_successes: int,
    control_total: int,
    treatment_successes: int,
    treatment_total: int
) -> dict:
    """
    Compare binary metric (e.g., CTR, conversion rate) between variants.
    """
    # Calculate rates
    control_rate = control_successes / control_total
    treatment_rate = treatment_successes / treatment_total
    
    # Contingency table
    observed = np.array([
        [control_successes, control_total - control_successes],
        [treatment_successes, treatment_total - treatment_successes]
    ])
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(observed)
    
    # Confidence interval for difference in proportions
    se_diff = np.sqrt(
        control_rate * (1 - control_rate) / control_total +
        treatment_rate * (1 - treatment_rate) / treatment_total
    )
    z_critical = stats.norm.ppf(1 - self.alpha/2)
    ci_lower = (treatment_rate - control_rate) - z_critical * se_diff
    ci_upper = (treatment_rate - control_rate) + z_critical * se_diff
    
    return {
        "control_rate": control_rate,
        "treatment_rate": treatment_rate,
        "absolute_difference": treatment_rate - control_rate,
        "relative_uplift_pct": (treatment_rate - control_rate) / control_rate * 100,
        "p_value": p_value,
        "is_significant": p_value < self.alpha,
        "confidence_interval_95": [ci_lower, ci_upper],
        "sample_size_control": control_total,
        "sample_size_treatment": treatment_total
    }
```

---

### 6. Sample Size Calculator

**Pre-Experiment Power Analysis**:

```python
from statsmodels.stats.power import tt_ind_solve_power

class SampleSizeCalculator:
    def calculate_required_sample_size(
        self,
        baseline_rate: float,
        mde: float,  # Minimum detectable effect (e.g., 0.02 = 2%)
        alpha: float = 0.05,
        power: float = 0.80,
        ratio: float = 1.0  # Treatment / Control ratio
    ) -> int:
        """
        Calculate required sample size per variant.
        
        Example:
        - Baseline CTR: 10%
        - Want to detect: 2% absolute lift (10% -> 12%)
        - Significance: 5%
        - Power: 80%
        """
        # Effect size (Cohen's h for proportions)
        p1 = baseline_rate
        p2 = baseline_rate + mde
        
        # Arcsine transformation
        effect_size = 2 * (np.arcsin(np.sqrt(p2)) - np.arcsin(np.sqrt(p1)))
        
        # Calculate sample size using power analysis
        n_treatment = tt_ind_solve_power(
            effect_size=effect_size,
            alpha=alpha,
            power=power,
            ratio=ratio,
            alternative='two-sided'
        )
        
        n_control = n_treatment * ratio
        
        return {
            "sample_size_per_variant": int(np.ceil(n_treatment)),
            "total_sample_size": int(np.ceil(n_treatment * (1 + ratio))),
            "expected_duration_days": self._estimate_duration(n_treatment, daily_traffic=10_000_000)
        }
    
    def _estimate_duration(self, required_sample, daily_traffic):
        """Estimate how long experiment needs to run"""
        # Assume 10% of traffic in experiment (90% control, 10% treatment)
        daily_sample = daily_traffic * 0.10
        return int(np.ceil(required_sample / daily_sample))
```

**Example**:
```python
calc = SampleSizeCalculator()
result = calc.calculate_required_sample_size(
    baseline_rate=0.10,  # 10% baseline CTR
    mde=0.02,  # Want to detect 2% absolute lift
    alpha=0.05,
    power=0.80
)
# Output: {"sample_size_per_variant": 3842, "total_sample_size": 7684, "expected_duration_days": 1}
```

---

### 7. Sequential Testing (Early Stopping)

**Always-Valid p-Values**:

```python
class SequentialTester:
    """
    Allows peeking at results without inflating Type I error.
    Based on: https://www.optimizely.com/insights/blog/can-i-stop-my-test-yet/
    """
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.alpha_spending = self._get_alpha_spending_function()
        
    def should_stop_experiment(
        self,
        current_sample_size: int,
        planned_sample_size: int,
        current_p_value: float
    ) -> dict:
        """
        Check if experiment can be stopped early.
        
        Returns:
        - decision: "stop_success" | "stop_futility" | "continue"
        - adjusted_alpha: Threshold for significance at this sample size
        """
        # Calculate information fraction (how far along we are)
        info_fraction = current_sample_size / planned_sample_size
        
        # Get adjusted alpha for this fraction
        adjusted_alpha = self.alpha_spending(info_fraction)
        
        if current_p_value < adjusted_alpha:
            return {
                "decision": "stop_success",
                "reason": f"Significant at adjusted α={adjusted_alpha:.4f}",
                "can_ship_treatment": True
            }
        
        # Futility check: Even with remaining samples, unlikely to succeed
        if info_fraction > 0.5 and current_p_value > 0.5:
            return {
                "decision": "stop_futility",
                "reason": "Unlikely to achieve significance",
                "can_ship_treatment": False
            }
        
        return {
            "decision": "continue",
            "progress_pct": info_fraction * 100,
            "adjusted_alpha": adjusted_alpha
        }
    
    def _get_alpha_spending_function(self):
        """O'Brien-Fleming alpha spending (conservative early, liberal late)"""
        def spending(t):
            if t <= 0:
                return 0
            return 2 * (1 - stats.norm.cdf(stats.norm.ppf(1 - self.alpha/2) / np.sqrt(t)))
        return spending
```

---

### 8. Multi-Armed Bandit (Adaptive Allocation)

**Thompson Sampling**:

```python
class ThompsonSampler:
    """
    Dynamically allocate traffic to best-performing variant.
    Balances exploration vs exploitation.
    """
    def __init__(self):
        # Beta distribution parameters for each variant
        self.successes = defaultdict(lambda: 1)  # Prior: 1 success
        self.failures = defaultdict(lambda: 1)   # Prior: 1 failure
        
    def select_variant(self, experiment_id: str, variants: List[str]) -> str:
        """Sample from posterior distributions, pick best"""
        samples = {}
        for variant in variants:
            # Sample from Beta distribution
            alpha = self.successes[f"{experiment_id}:{variant}"]
            beta = self.failures[f"{experiment_id}:{variant}"]
            samples[variant] = np.random.beta(alpha, beta)
        
        # Return variant with highest sampled value
        return max(samples, key=samples.get)
    
    def update(self, experiment_id: str, variant: str, success: bool):
        """Update posterior based on observed outcome"""
        key = f"{experiment_id}:{variant}"
        if success:
            self.successes[key] += 1
        else:
            self.failures[key] += 1
    
    def get_win_probability(self, experiment_id: str, variants: List[str], n_samples=10000):
        """Estimate probability each variant is best"""
        wins = defaultdict(int)
        
        for _ in range(n_samples):
            samples = {}
            for variant in variants:
                alpha = self.successes[f"{experiment_id}:{variant}"]
                beta = self.failures[f"{experiment_id}:{variant}"]
                samples[variant] = np.random.beta(alpha, beta)
            
            winner = max(samples, key=samples.get)
            wins[winner] += 1
        
        return {v: wins[v] / n_samples for v in variants}
```

**Usage**:
```python
bandit = ThompsonSampler()

# User request comes in
variant = bandit.select_variant("experiment_123", ["control", "treatment_a", "treatment_b"])

# Serve variant, observe outcome
user_clicked = serve_variant_and_observe(variant)

# Update posterior
bandit.update("experiment_123", variant, success=user_clicked)

# Check after 1000 samples
probs = bandit.get_win_probability("experiment_123", ["control", "treatment_a", "treatment_b"])
# Output: {"control": 0.23, "treatment_a": 0.71, "treatment_b": 0.06}
# -> treatment_a is winning with 71% probability
```

---

### 9. Automated Reporting Dashboard

**Real-Time Experiment Dashboard**:

```python
class ExperimentDashboard:
    def __init__(self):
        self.analyzer = StatisticalAnalyzer()
        self.db = PostgreSQL()
        
    def generate_report(self, experiment_id: str) -> dict:
        """Generate comprehensive experiment report"""
        # Fetch data
        data = self.db.query(f"""
            SELECT 
                variant,
                COUNT(DISTINCT user_id) as users,
                SUM(CASE WHEN metric_name = 'click' THEN 1 ELSE 0 END) as clicks,
                AVG(CASE WHEN metric_name = 'revenue' THEN value ELSE NULL END) as avg_revenue
            FROM experiment_events
            WHERE experiment_id = '{experiment_id}'
            GROUP BY variant
        """)
        
        control = data[data['variant'] == 'control'].iloc[0]
        treatment = data[data['variant'] == 'treatment'].iloc[0]
        
        # Analyze primary metric (CTR)
        ctr_results = self.analyzer.analyze_binary_metric(
            control_successes=control['clicks'],
            control_total=control['users'],
            treatment_successes=treatment['clicks'],
            treatment_total=treatment['users']
        )
        
        # Analyze secondary metric (revenue)
        revenue_results = self.analyzer.analyze_continuous_metric(
            control_samples=self._get_revenue_samples(experiment_id, 'control'),
            treatment_samples=self._get_revenue_samples(experiment_id, 'treatment')
        )
        
        return {
            "experiment_id": experiment_id,
            "status": "running",
            "duration_days": self._get_duration(experiment_id),
            "sample_sizes": {
                "control": control['users'],
                "treatment": treatment['users']
            },
            "primary_metric": {
                "name": "click_through_rate",
                **ctr_results
            },
            "secondary_metrics": {
                "revenue_per_user": revenue_results
            },
            "recommendation": self._get_recommendation(ctr_results, revenue_results)
        }
    
    def _get_recommendation(self, ctr, revenue):
        """Automated decision recommendation"""
        if ctr['is_significant'] and ctr['relative_uplift_pct'] > 0:
            if revenue['is_significant'] and revenue['relative_uplift_pct'] > 0:
                return "SHIP: Both primary and secondary metrics improved"
            else:
                return "CONSIDER: Primary metric improved, but revenue flat"
        elif ctr['is_significant'] and ctr['relative_uplift_pct'] < 0:
            return "REJECT: Primary metric degraded"
        else:
            return "CONTINUE: Not enough evidence yet"
```

**Output Example**:
```json
{
  "experiment_id": "model_v2_test",
  "status": "running",
  "duration_days": 7,
  "sample_sizes": {
    "control": 45000,
    "treatment": 5000
  },
  "primary_metric": {
    "name": "click_through_rate",
    "control_rate": 0.103,
    "treatment_rate": 0.118,
    "relative_uplift_pct": 14.56,
    "p_value": 0.002,
    "is_significant": true,
    "confidence_interval_95": [0.005, 0.025]
  },
  "secondary_metrics": {
    "revenue_per_user": {
      "control_mean": 1.42,
      "treatment_mean": 1.51,
      "relative_uplift_pct": 6.34,
      "p_value": 0.08,
      "is_significant": false
    }
  },
  "recommendation": "CONSIDER: Primary metric improved, but revenue flat"
}
```

---

## Performance & Scalability

**Latency Overhead** (Target: <5ms):

| Operation | Latency | Notes |
|-----------|---------|-------|
| Config lookup | 1ms | Redis cache hit |
| Hash computation | 0.1ms | MD5 hash |
| Variant assignment | 0.1ms | Array lookup |
| Event logging (async) | 0.5ms | Fire-and-forget Kafka |
| **Total** | **~2ms** | Within target ✓ |

**Throughput**:
- 10M daily users = 115 req/s average, 1000 req/s peak
- Experiment service can handle 50K assignments/sec (single instance)
- Kafka throughput: 1M events/sec (6-node cluster)

---

## Cost Estimation

**Monthly Cost** (10M daily users, 5 concurrent experiments):

| Component | Cost | Notes |
|-----------|------|-------|
| Experiment Service | $200 | 2x c5.large EC2 |
| Redis (config cache) | $50 | ElastiCache r6g.large |
| Kafka Cluster | $600 | 6x m5.large |
| Spark Analytics | $800 | EMR cluster (batch processing) |
| PostgreSQL | $300 | RDS db.r5.large |
| Data Storage (S3) | $50 | Event logs (30-day retention) |
| **Total** | **~$2,000/month** | |

---

## Trade-offs & Decisions

| Decision | Option A | Option B | Choice |
|----------|----------|----------|--------|
| Assignment | Consistent hashing | Centralized DB | Hashing (faster, scalable) |
| Analysis | Frequentist (t-test) | Bayesian (Thompson) | Both (depends on use case) |
| Logging | Synchronous | Asynchronous | Async (lower latency) |
| Early Stopping | Fixed sample size | Sequential testing | Sequential (faster decisions) |

---

## Interview Discussion Points

1. **How to prevent sample ratio mismatch (SRM)?**
   - Monitor: Actual traffic split should match config (90/10 → verify)
   - Causes: Bot traffic, client-side bucketing bugs, caching issues
   - Detection: Chi-square test on assignment counts

2. **How to handle multiple testing problem?**
   - Bonferroni correction: Divide α by number of tests
   - Primary metric only for decisions, secondary for insights
   - Consider false discovery rate (FDR) control

3. **What if experiment traffic overlaps?**
   - Namespace experiments (search, ranking, UI)
   - Orthogonality check: Ensure no interaction effects
   - Holdout group: 5% never in experiments (control for long-term effects)

4. **How to A/B test ranking/recommendation models?**
   - Interleaving: Mix results from both models
   - Metrics: Click-through rate, session success rate
   - Watch for position bias

---

## Conclusion

This A/B testing platform supports 10M daily users with:
- **<2ms latency** overhead for variant assignment
- **Statistical rigor** (power analysis, sequential testing)
- **Safe rollouts** (gradual traffic increase, automated stopping)
- **Cost-effective** (~$2K/month)

**Production-ready checklist**:
- ✅ Handles concurrent experiments
- ✅ Consistent user assignment
- ✅ Real-time dashboards
- ✅ Automated statistical analysis
- ✅ Early stopping capability

**Typical Interview Flow**:
1. Requirements (users, experiments, metrics)
2. Assignment algorithm (consistent hashing)
3. Statistical testing (t-test, sample size)
4. Architecture (event logging, analytics pipeline)
5. Trade-offs (fixed vs sequential, frequentist vs Bayesian)
