# Module 2: Drift Detection for Embeddings - Comprehensive Theory

## Table of Contents

1. [Introduction to Drift in ML Systems](#1-introduction-to-drift-in-ml-systems)
2. [Types of Drift](#2-types-of-drift)
3. [Statistical Methods for Drift Detection](#3-statistical-methods-for-drift-detection)
4. [Embedding-Specific Drift](#4-embedding-specific-drift)
5. [Evidently AI for Drift Detection](#5-evidently-ai-for-drift-detection)
6. [Production Implementation](#6-production-implementation)
7. [Drift Response Strategies](#7-drift-response-strategies)
8. [Case Studies](#8-case-studies)

---

## 1. Introduction to Drift in ML Systems

### 1.1 What is Drift?

Drift (also called data drift or concept drift) refers to the phenomenon where the statistical properties of the target variable or input features change over time, causing model performance to degrade.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Drift in Production                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Training Time                          Production Time                  │
│       │                                    │                             │
│       ▼                                    ▼                             │
│  ┌─────────────┐                     ┌─────────────┐                    │
│  │  Training   │                     │  Production │                    │
│  │    Data     │                     │    Data     │                    │
│  │             │                     │             │                    │
│  │  P_train(X) │                     │  P_prod(X)  │                    │
│  │  P_train(Y) │                     │  P_prod(Y)  │                    │
│  └─────────────┘                     └─────────────┘                    │
│       │                                    │                             │
│       │                                    │  ⚠ DRIFT OCCURS            │
│       │                                    │  WHEN:                     │
│       │                                    │  P_train ≠ P_prod          │
│       ▼                                    ▼                             │
│  ┌─────────────┐                     ┌─────────────┐                    │
│  │   Model     │────────────────────▶│  Degraded   │                    │
│  │  Training   │                     │ Performance │                    │
│  └─────────────┘                     └─────────────┘                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Why Drift Matters for LLMs

LLM systems are particularly susceptible to drift due to:

1. **Evolving Language**: New terms, slang, and usage patterns emerge constantly
2. **Domain Changes**: User queries shift based on trends, news, seasons
3. **Embedding Model Updates**: New versions produce different vector spaces
4. **Knowledge Base Changes**: RAG systems depend on up-to-date documents
5. **User Behavior**: Interaction patterns change over time

### 1.3 Drift Detection Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Drift Detection Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Production Data                                                         │
│       │                                                                  │
│       ▼                                                                  │
│  ┌─────────────┐                                                         │
│  │ 1. Data     │ ← Collect production embeddings/features               │
│  │    Collection│                                                        │
│  └──────┬──────┘                                                         │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────┐                                                         │
│  │ 2. Reference│ ← Load baseline/reference distribution                 │
│  │    Loading  │                                                        │
│  └──────┬──────┘                                                         │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────┐                                                         │
│  │ 3. Statistical│ ← Apply statistical tests (KS, PSI, etc.)            │
│  │    Tests    │                                                        │
│  └──────┬──────┘                                                         │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────┐                                                         │
│  │ 4. Threshold│ ← Compare against drift thresholds                     │
│  │    Check    │                                                        │
│  └──────┬──────┘                                                         │
│         │                                                                  │
│         ▼                                                                  │
│  ┌─────────────┐                                                         │
│  │ 5. Alert &  │ ← Trigger alerts and response workflows                │
│  │    Response │                                                        │
│  └─────────────┘                                                         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Types of Drift

### 2.1 Covariate Drift (Feature Drift)

Covariate drift occurs when the distribution of input features changes while the relationship between features and target remains the same.

**Mathematical Definition:**
```
P_train(X) ≠ P_prod(X)
P_train(Y|X) = P_prod(Y|X)
```

**Example in LLMs:**
- User queries shift from technical topics to casual conversation
- New terminology emerges (e.g., new AI model names)
- Seasonal changes in query topics

**Detection Methods:**
- Kolmogorov-Smirnov test
- Population Stability Index (PSI)
- Wasserstein distance

### 2.2 Concept Drift

Concept drift occurs when the relationship between input features and target variable changes.

**Mathematical Definition:**
```
P_train(X) = P_prod(X) (may or may not change)
P_train(Y|X) ≠ P_prod(Y|X)
```

**Example in LLMs:**
- Same query gets different expected answers over time
- Sentiment associated with terms changes
- Fact updates (e.g., "current president" changes)

**Detection Methods:**
- Performance monitoring
- Error rate tracking
- User feedback analysis

### 2.3 Prior Probability Drift (Label Drift)

Prior probability drift occurs when the distribution of target labels changes.

**Mathematical Definition:**
```
P_train(Y) ≠ P_prod(Y)
```

**Example in LLMs:**
- Classification task: class distribution shifts
- Sentiment analysis: more negative queries during crisis
- Intent detection: new intent categories emerge

### 2.4 Embedding Drift

Embedding drift is specific to systems using vector embeddings.

**Causes:**
1. Embedding model version changes
2. Vocabulary changes
3. Dimension distribution shifts
4. Clustering pattern changes

**Detection Methods:**
- Cosine similarity distribution comparison
- PCA projection comparison
- Clustering metric comparison

---

## 3. Statistical Methods for Drift Detection

### 3.1 Kolmogorov-Smirnov (KS) Test

The KS test compares two distributions by measuring the maximum distance between their cumulative distribution functions (CDFs).

**Formula:**
```
D = max|F_train(x) - F_prod(x)|
```

Where:
- D is the KS statistic (0 to 1)
- F_train is the CDF of training data
- F_prod is the CDF of production data

**Implementation:**
```python
from scipy import stats
import numpy as np

def ks_test(train_data: np.ndarray, prod_data: np.ndarray) -> dict:
    """
    Perform Kolmogorov-Smirnov test for drift detection.
    
    Returns:
        dict with statistic, p_value, and drift_detected
    """
    statistic, p_value = stats.ks_2samp(train_data, prod_data)
    
    return {
        "test": "Kolmogorov-Smirnov",
        "statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < 0.05,
        "interpretation": interpret_ks_result(statistic)
    }

def interpret_ks_result(statistic: float) -> str:
    """Interpret KS statistic value."""
    if statistic < 0.1:
        return "No significant drift"
    elif statistic < 0.2:
        return "Minor drift detected"
    elif statistic < 0.3:
        return "Moderate drift detected"
    else:
        return "Significant drift detected"
```

**Thresholds:**
| KS Statistic | Interpretation | Action |
|--------------|----------------|--------|
| < 0.1 | No drift | Continue monitoring |
| 0.1 - 0.2 | Minor drift | Investigate |
| 0.2 - 0.3 | Moderate drift | Plan remediation |
| > 0.3 | Significant drift | Immediate action |

### 3.2 Population Stability Index (PSI)

PSI measures how much a population has shifted over time by comparing binned distributions.

**Formula:**
```
PSI = Σ((Actual% - Expected%) × ln(Actual% / Expected%))
```

**Implementation:**
```python
import numpy as np

def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> dict:
    """
    Calculate Population Stability Index.
    
    Args:
        expected: Reference distribution (training data)
        actual: Current distribution (production data)
        bins: Number of bins for discretization
    
    Returns:
        dict with PSI value and interpretation
    """
    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)
    
    # Calculate percentages in each bin
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    # Convert to percentages (add small epsilon to avoid log(0))
    epsilon = 0.0001
    expected_pct = (expected_counts + epsilon) / (len(expected) + epsilon * bins)
    actual_pct = (actual_counts + epsilon) / (len(actual) + epsilon * bins)
    
    # Calculate PSI
    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return {
        "test": "Population Stability Index",
        "psi": psi,
        "drift_detected": psi > 0.1,
        "interpretation": interpret_psi(psi),
        "bin_details": {
            "expected_pct": expected_pct.tolist(),
            "actual_pct": actual_pct.tolist()
        }
    }

def interpret_psi(psi: float) -> str:
    """Interpret PSI value."""
    if psi < 0.1:
        return "No significant drift"
    elif psi < 0.2:
        return "Minor drift detected"
    else:
        return "Significant drift detected"
```

**Thresholds:**
| PSI Value | Interpretation | Action |
|-----------|----------------|--------|
| < 0.1 | No drift | Continue monitoring |
| 0.1 - 0.2 | Minor drift | Investigate |
| > 0.2 | Significant drift | Immediate action |

### 3.3 Wasserstein Distance (Earth Mover's Distance)

Wasserstein distance measures the minimum "work" needed to transform one distribution into another.

**Formula (1D):**
```
W(P, Q) = ∫|F_P(x) - F_Q(x)| dx
```

**Implementation:**
```python
from scipy import stats

def wasserstein_distance(train_data: np.ndarray, prod_data: np.ndarray) -> dict:
    """
    Calculate Wasserstein distance between distributions.
    
    Returns:
        dict with distance and normalized score
    """
    distance = stats.wasserstein_distance(train_data, prod_data)
    
    # Normalize by combined standard deviation
    combined_std = np.std(np.concatenate([train_data, prod_data]))
    normalized_distance = distance / combined_std if combined_std > 0 else distance
    
    return {
        "test": "Wasserstein Distance",
        "distance": distance,
        "normalized_distance": normalized_distance,
        "drift_detected": normalized_distance > 0.5,
        "interpretation": interpret_wasserstein(normalized_distance)
    }

def interpret_wasserstein(normalized_distance: float) -> str:
    """Interpret normalized Wasserstein distance."""
    if normalized_distance < 0.25:
        return "No significant drift"
    elif normalized_distance < 0.5:
        return "Minor drift detected"
    elif normalized_distance < 1.0:
        return "Moderate drift detected"
    else:
        return "Significant drift detected"
```

### 3.4 Chi-Square Test

For categorical data, the chi-square test compares observed vs. expected frequencies.

**Implementation:**
```python
from scipy import stats

def chi_square_test(expected_counts: np.ndarray, observed_counts: np.ndarray) -> dict:
    """
    Perform Chi-Square test for categorical drift.
    
    Returns:
        dict with statistic, p_value, and drift_detected
    """
    statistic, p_value = stats.chisquare(f_obs=observed_counts, f_exp=expected_counts)
    
    return {
        "test": "Chi-Square",
        "statistic": statistic,
        "p_value": p_value,
        "drift_detected": p_value < 0.05,
        "degrees_of_freedom": len(expected_counts) - 1
    }
```

---

## 4. Embedding-Specific Drift

### 4.1 Embedding Distribution Analysis

Embeddings are high-dimensional vectors, requiring specialized drift detection approaches.

**Key Metrics:**
1. **Mean Vector Shift**: Change in average embedding position
2. **Covariance Change**: Change in embedding spread/orientation
3. **Nearest Neighbor Distance**: Change in local structure

**Implementation:**
```python
import numpy as np
from scipy.spatial.distance import cosine

class EmbeddingDriftDetector:
    """Detect drift in embedding distributions."""
    
    def __init__(self, reference_embeddings: np.ndarray):
        """
        Initialize with reference embeddings.
        
        Args:
            reference_embeddings: Baseline embeddings (N x D matrix)
        """
        self.reference = reference_embeddings
        self.reference_mean = np.mean(reference_embeddings, axis=0)
        self.reference_cov = np.cov(reference_embeddings.T)
        self.reference_norms = np.linalg.norm(reference_embeddings, axis=1)
    
    def detect_mean_shift(self, current_embeddings: np.ndarray) -> dict:
        """Detect shift in mean embedding position."""
        current_mean = np.mean(current_embeddings, axis=0)
        
        # Calculate Euclidean distance between means
        mean_shift = np.linalg.norm(current_mean - self.reference_mean)
        
        # Normalize by reference standard deviation
        ref_std = np.std(self.reference_norms)
        normalized_shift = mean_shift / ref_std if ref_std > 0 else mean_shift
        
        return {
            "metric": "mean_shift",
            "value": float(mean_shift),
            "normalized": float(normalized_shift),
            "drift_detected": normalized_shift > 0.5
        }
    
    def detect_covariance_change(self, current_embeddings: np.ndarray) -> dict:
        """Detect change in embedding covariance structure."""
        current_cov = np.cov(current_embeddings.T)
        
        # Frobenius norm of covariance difference
        cov_diff = np.linalg.norm(current_cov - self.reference_cov, 'fro')
        
        return {
            "metric": "covariance_change",
            "value": float(cov_diff),
            "drift_detected": cov_diff > np.linalg.norm(self.reference_cov, 'fro') * 0.3
        }
    
    def detect_nearest_neighbor_change(self, current_embeddings: np.ndarray, 
                                       k: int = 5) -> dict:
        """Detect change in nearest neighbor distances."""
        from sklearn.neighbors import NearestNeighbors
        
        # Fit NN on reference
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(self.reference)
        
        # Get distances for current embeddings
        distances, _ = nn.kneighbors(current_embeddings)
        avg_nn_distance = np.mean(distances[:, -1])  # k-th neighbor
        
        # Compare to reference
        ref_distances, _ = nn.kneighbors(self.reference)
        ref_avg_nn = np.mean(ref_distances[:, -1])
        
        ratio = avg_nn_distance / ref_avg_nn if ref_avg_nn > 0 else 1.0
        
        return {
            "metric": "nearest_neighbor_change",
            "current_avg": float(avg_nn_distance),
            "reference_avg": float(ref_avg_nn),
            "ratio": float(ratio),
            "drift_detected": ratio > 1.5 or ratio < 0.67
        }
```

### 4.2 Dimension-Wise Analysis

Analyze drift in individual embedding dimensions to identify specific areas of change.

```python
def analyze_dimension_drift(reference: np.ndarray, 
                           current: np.ndarray,
                           threshold: float = 0.1) -> dict:
    """
    Analyze drift per embedding dimension.
    
    Returns:
        dict with per-dimension drift scores and flagged dimensions
    """
    n_dimensions = reference.shape[1]
    drift_scores = []
    flagged_dimensions = []
    
    for dim in range(n_dimensions):
        ref_dim = reference[:, dim]
        curr_dim = current[:, dim]
        
        # KS test for this dimension
        statistic, p_value = stats.ks_2samp(ref_dim, curr_dim)
        drift_scores.append(statistic)
        
        if statistic > threshold:
            flagged_dimensions.append({
                "dimension": dim,
                "ks_statistic": statistic,
                "p_value": p_value
            })
    
    return {
        "dimension_drift_scores": drift_scores,
        "mean_drift": np.mean(drift_scores),
        "max_drift": np.max(drift_scores),
        "flagged_dimensions": flagged_dimensions,
        "n_flagged": len(flagged_dimensions),
        "drift_detected": len(flagged_dimensions) > n_dimensions * 0.1
    }
```

### 4.3 PCA-Based Drift Detection

Use PCA to project high-dimensional embeddings and detect drift in principal components.

```python
from sklearn.decomposition import PCA

def pca_drift_detection(reference: np.ndarray, 
                       current: np.ndarray,
                       n_components: int = 10) -> dict:
    """
    Detect drift using PCA projection.
    
    Returns:
        dict with PCA-based drift metrics
    """
    # Fit PCA on reference
    pca = PCA(n_components=n_components)
    ref_projected = pca.fit_transform(reference)
    curr_projected = pca.transform(current)
    
    # Compare projections
    drift_results = {}
    
    for i in range(n_components):
        ref_pc = ref_projected[:, i]
        curr_pc = curr_projected[:, i]
        
        statistic, p_value = stats.ks_2samp(ref_pc, curr_pc)
        drift_results[f"PC{i+1}"] = {
            "ks_statistic": statistic,
            "p_value": p_value,
            "variance_explained": pca.explained_variance_ratio_[i],
            "drift_detected": p_value < 0.05
        }
    
    # Overall drift (any PC shows drift)
    any_drift = any(r["drift_detected"] for r in drift_results.values())
    
    return {
        "pca_drift_results": drift_results,
        "variance_explained": pca.explained_variance_ratio_.tolist(),
        "drift_detected": any_drift,
        "n_components_with_drift": sum(1 for r in drift_results.values() if r["drift_detected"])
    }
```

---

## 5. Evidently AI for Drift Detection

### 5.1 Introduction to Evidently

Evidently is an open-source Python library for ML model monitoring and drift detection.

**Key Features:**
- Data drift detection
- Target drift detection
- Model performance monitoring
- Interactive reports
- Integration with MLflow, Grafana

### 5.2 Basic Drift Report

```python
from evidently.report import Report
from evidently.metrics import DataDriftTable, DataDriftPlot
from evidently.column_mapping import ColumnMapping
import pandas as pd

def generate_drift_report(reference_data: pd.DataFrame,
                         current_data: pd.DataFrame,
                         column_names: list = None) -> Report:
    """
    Generate Evidently drift report.
    
    Args:
        reference_data: Baseline data
        current_data: Current production data
        column_names: Columns to analyze (None = all numeric)
    
    Returns:
        Evidently Report object
    """
    # Create column mapping
    column_mapping = ColumnMapping()
    if column_names:
        column_mapping.numerical_features = column_names
    
    # Create report
    report = Report(metrics=[
        DataDriftTable(),
        DataDriftPlot(),
    ])
    
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    
    return report

# Save report
report.save_html("drift_report.html")
```

### 5.3 Custom Metrics

```python
from evidently.metrics import Metric
from evidently.core import Callsite
from evidently.renderers.html_widgets import WidgetSize
from evidently.base_metric import MetricResult

class EmbeddingDriftMetric(Metric):
    """Custom metric for embedding drift detection."""
    
    def __init__(self, reference_embeddings: np.ndarray):
        self.reference_embeddings = reference_embeddings
        super().__init__()
    
    def calculate_value(self, current_data: pd.DataFrame) -> EmbeddingDriftResult:
        """Calculate embedding drift metric."""
        current_embeddings = np.array(current_data['embedding'].tolist())
        
        detector = EmbeddingDriftDetector(self.reference_embeddings)
        
        mean_shift = detector.detect_mean_shift(current_embeddings)
        cov_change = detector.detect_covariance_change(current_embeddings)
        
        return EmbeddingDriftResult(
            mean_shift=mean_shift,
            covariance_change=cov_change,
            drift_detected=mean_shift["drift_detected"] or cov_change["drift_detected"]
        )
```

### 5.4 Dashboard Integration

```python
from evidently.dashboard import Dashboard
from evidently.tabs import DataDriftTab

def create_drift_dashboard(reference_data: pd.DataFrame,
                          current_data: pd.DataFrame) -> Dashboard:
    """Create interactive drift dashboard."""
    dashboard = Dashboard(tabs=[
        DataDriftTab(),
    ])
    
    dashboard.calculate(
        reference_data=reference_data,
        current_data=current_data
    )
    
    return dashboard

# Save dashboard
dashboard.save_html("drift_dashboard.html")
```

---

## 6. Production Implementation

### 6.1 Batch vs. Streaming Detection

**Batch Detection:**
- Run periodically (hourly, daily)
- Process accumulated data
- Lower computational overhead
- Higher detection latency

**Streaming Detection:**
- Process data in real-time
- Immediate drift detection
- Higher computational cost
- More complex implementation

### 6.2 Reference Data Management

```python
class ReferenceDataManager:
    """Manage reference data for drift detection."""
    
    def __init__(self, storage_path: str, max_references: int = 5):
        self.storage_path = storage_path
        self.max_references = max_references
        self.references = {}
    
    def save_reference(self, name: str, data: np.ndarray, 
                      metadata: dict = None) -> str:
        """Save reference data."""
        import pickle
        from datetime import datetime
        
        reference_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        reference_data = {
            "id": reference_id,
            "name": name,
            "data": data,
            "metadata": metadata or {},
            "created_at": datetime.now().isoformat()
        }
        
        filepath = f"{self.storage_path}/{reference_id}.pkl"
        with open(filepath, 'wb') as f:
            pickle.dump(reference_data, f)
        
        self.references[reference_id] = reference_data
        self._cleanup_old_references()
        
        return reference_id
    
    def load_reference(self, reference_id: str) -> dict:
        """Load reference data by ID."""
        import pickle
        
        if reference_id in self.references:
            return self.references[reference_id]
        
        filepath = f"{self.storage_path}/{reference_id}.pkl"
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def get_latest_reference(self, name: str) -> dict:
        """Get latest reference for a given name."""
        matching = [
            (rid, ref) for rid, ref in self.references.items()
            if ref["name"] == name
        ]
        
        if not matching:
            return None
        
        return max(matching, key=lambda x: x[1]["created_at"])[1]
    
    def _cleanup_old_references(self):
        """Remove old references beyond max_references."""
        if len(self.references) <= self.max_references:
            return
        
        # Sort by creation time
        sorted_refs = sorted(
            self.references.items(),
            key=lambda x: x[1]["created_at"]
        )
        
        # Remove oldest
        for ref_id, _ in sorted_refs[:-self.max_references]:
            del self.references[ref_id]
```

### 6.3 Threshold Tuning

```python
class ThresholdTuner:
    """Tune drift detection thresholds based on historical data."""
    
    def __init__(self, historical_drift_events: list):
        self.events = historical_drift_events
    
    def optimize_threshold(self, metric: str, 
                          target_false_positive_rate: float = 0.05) -> float:
        """
        Find optimal threshold for a metric.
        
        Args:
            metric: Name of the metric to tune
            target_false_positive_rate: Desired FPR
        
        Returns:
            Optimal threshold value
        """
        # Extract metric values from confirmed drift and non-drift events
        drift_values = [
            e[metric] for e in self.events 
            if e["confirmed_drift"]
        ]
        non_drift_values = [
            e[metric] for e in self.events 
            if not e["confirmed_drift"]
        ]
        
        # Find threshold that achieves target FPR
        thresholds = np.linspace(
            min(non_drift_values + drift_values),
            max(non_drift_values + drift_values),
            100
        )
        
        best_threshold = thresholds[0]
        best_fpr_diff = float('inf')
        
        for threshold in thresholds:
            fp_count = sum(1 for v in non_drift_values if v > threshold)
            fpr = fp_count / len(non_drift_values) if non_drift_values else 0
            
            fpr_diff = abs(fpr - target_false_positive_rate)
            
            if fpr_diff < best_fpr_diff:
                best_fpr_diff = fpr_diff
                best_threshold = threshold
        
        return best_threshold
```

---

## 7. Drift Response Strategies

### 7.1 Response Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Drift Response Hierarchy                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Drift Severity           Response Action                                │
│       │                        │                                         │
│       ▼                        ▼                                         │
│  ┌─────────┐           ┌──────────────────┐                            │
│  │  Minor  │──────────▶│ • Log event      │                            │
│  │ (PSI<0.2)│          │ • Increase monitoring│                         │
│  └─────────┘           │ • Review in weekly│                            │
│                        └──────────────────┘                            │
│                                                                          │
│  ┌─────────┐           ┌──────────────────┐                            │
│  │Moderate │──────────▶│ • Alert team     │                            │
│  │(0.2<PSI<0.3)│       │ • Investigate root│                            │
│  └─────────┘           │ • Prepare remediation│                         │
│                        └──────────────────┘                            │
│                                                                          │
│  ┌─────────┐           ┌──────────────────┐                            │
│  │ Significant│─────────▶│ • Immediate alert│                            │
│  │ (PSI>0.3)│          │ • Trigger response│                            │
│  └─────────┘           │ • Consider rollback│                            │
│                        │ • Plan retraining │                            │
│                        └──────────────────┘                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Automated Response Workflow

```python
class DriftResponseWorkflow:
    """Automated response to detected drift."""
    
    def __init__(self, alert_manager, model_registry, config: dict):
        self.alert_manager = alert_manager
        self.model_registry = model_registry
        self.config = config
    
    def handle_drift(self, drift_result: dict) -> dict:
        """
        Handle detected drift based on severity.
        
        Returns:
            dict with actions taken
        """
        severity = self._assess_severity(drift_result)
        actions = []
        
        if severity == "minor":
            actions = self._handle_minor_drift(drift_result)
        elif severity == "moderate":
            actions = self._handle_moderate_drift(drift_result)
        elif severity == "significant":
            actions = self._handle_significant_drift(drift_result)
        
        return {
            "severity": severity,
            "actions_taken": actions,
            "timestamp": datetime.now().isoformat()
        }
    
    def _assess_severity(self, drift_result: dict) -> str:
        """Assess drift severity."""
        psi = drift_result.get("psi", 0)
        
        if psi < 0.2:
            return "minor"
        elif psi < 0.3:
            return "moderate"
        else:
            return "significant"
    
    def _handle_minor_drift(self, drift_result: dict) -> list:
        """Handle minor drift."""
        actions = []
        
        # Log event
        self._log_event("minor_drift", drift_result)
        actions.append("logged_event")
        
        # Increase monitoring frequency
        self._increase_monitoring_frequency()
        actions.append("increased_monitoring")
        
        return actions
    
    def _handle_moderate_drift(self, drift_result: dict) -> list:
        """Handle moderate drift."""
        actions = self._handle_minor_drift(drift_result)
        
        # Alert team
        self.alert_manager.send_alert(
            severity="warning",
            title="Moderate Drift Detected",
            details=drift_result
        )
        actions.append("alerted_team")
        
        # Start investigation
        self._start_investigation(drift_result)
        actions.append("started_investigation")
        
        return actions
    
    def _handle_significant_drift(self, drift_result: dict) -> list:
        """Handle significant drift."""
        actions = self._handle_moderate_drift(drift_result)
        
        # Critical alert
        self.alert_manager.send_alert(
            severity="critical",
            title="Significant Drift Detected - Immediate Action Required",
            details=drift_result
        )
        actions.append("critical_alert")
        
        # Consider model rollback
        if self.config.get("auto_rollback", False):
            self._trigger_rollback()
            actions.append("triggered_rollback")
        
        # Schedule retraining
        self._schedule_retraining()
        actions.append("scheduled_retraining")
        
        return actions
```

---

## 8. Case Studies

### 8.1 Case Study: RAG System Drift

**Scenario:** A RAG-based customer support chatbot experienced declining answer quality.

**Detection:**
- Embedding drift detected (PSI = 0.35)
- Root cause: New product documentation changed terminology
- KS test showed significant shift in query embeddings

**Resolution:**
1. Updated reference embeddings with new documentation
2. Retrained embedding model on updated corpus
3. Implemented weekly drift monitoring

**Outcome:**
- Answer quality restored to baseline
- Drift detection now catches terminology changes within 24 hours

### 8.2 Case Study: Seasonal Drift

**Scenario:** E-commerce recommendation system showed periodic performance drops.

**Detection:**
- Monthly drift patterns detected
- Correlated with seasonal product changes
- PSI peaked during holiday seasons

**Resolution:**
1. Implemented seasonal reference data
2. Created separate drift thresholds per season
3. Pre-emptive model updates before peak seasons

**Outcome:**
- Reduced false positive alerts by 60%
- Improved recommendation relevance during peak periods

---

## Summary

This module covered:

1. **Types of Drift** - Covariate, concept, prior probability, and embedding drift
2. **Statistical Methods** - KS test, PSI, Wasserstein distance, Chi-square
3. **Embedding Analysis** - Mean shift, covariance change, PCA-based detection
4. **Evidently AI** - Implementation and customization
5. **Production Patterns** - Reference management, threshold tuning, response workflows

Master these techniques to maintain reliable LLM systems in production.

---

*End of Module 2 Theory Content*  
*Total Lines: 800+*
