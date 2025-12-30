"""
Model Monitoring Module
=======================
Drift detection, performance monitoring, and alerting.

Concepts:
- Data Drift: Input distribution changes
- Concept Drift: P(y|X) changes
- Model Performance Degradation

Methods:
- KS Test (Kolmogorov-Smirnov)
- PSI (Population Stability Index)
- Chi-square test for categorical
- Performance metrics tracking

Author: AI-Mastery-2026
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import json
import logging

logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class DriftResult:
    """Result of drift detection test."""
    feature_name: str
    drift_detected: bool
    statistic: float
    p_value: float
    threshold: float
    method: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class PerformanceMetrics:
    """Model performance metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ============================================================
# STATISTICAL TESTS
# ============================================================

def ks_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """
    Kolmogorov-Smirnov Test for continuous variables.
    
    Tests if two samples come from the same distribution.
    
    Statistic: D = max|F_ref(x) - F_cur(x)|
    
    Args:
        reference: Reference (training) distribution
        current: Current (production) distribution
    
    Returns:
        Tuple of (KS statistic, p-value)
    
    Interpretation:
        - D close to 0: Distributions are similar
        - D close to 1: Distributions are very different
        - p-value < 0.05: Reject null hypothesis (different distributions)
    """
    from scipy import stats
    statistic, p_value = stats.ks_2samp(reference, current)
    return float(statistic), float(p_value)


def ks_test_manual(reference: np.ndarray, current: np.ndarray) -> float:
    """
    Manual implementation of KS test statistic.
    
    For educational purposes.
    """
    ref_sorted = np.sort(reference)
    cur_sorted = np.sort(current)
    
    # Combine and sort all values
    all_values = np.concatenate([ref_sorted, cur_sorted])
    all_values = np.sort(np.unique(all_values))
    
    # Compute empirical CDFs
    n_ref, n_cur = len(reference), len(current)
    
    max_diff = 0
    for v in all_values:
        cdf_ref = np.sum(reference <= v) / n_ref
        cdf_cur = np.sum(current <= v) / n_cur
        max_diff = max(max_diff, abs(cdf_ref - cdf_cur))
    
    return max_diff


def psi(reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index (PSI).
    
    Measures shift between two distributions using binning.
    
    PSI = Σ (Actual% - Expected%) × ln(Actual% / Expected%)
    
    Interpretation:
        - PSI < 0.1: No significant change
        - 0.1 ≤ PSI < 0.2: Slight change
        - PSI ≥ 0.2: Significant change (drift detected)
    
    Args:
        reference: Reference distribution
        current: Current distribution
        buckets: Number of bins
    
    Returns:
        PSI value
    """
    # Create bins from reference distribution
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    bins = np.linspace(min_val, max_val, buckets + 1)
    
    # Bin counts
    ref_counts, _ = np.histogram(reference, bins)
    cur_counts, _ = np.histogram(current, bins)
    
    # Convert to percentages (add epsilon to avoid division by zero)
    epsilon = 1e-10
    ref_pct = ref_counts / len(reference) + epsilon
    cur_pct = cur_counts / len(current) + epsilon
    
    # PSI calculation
    psi_value = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    
    return float(psi_value)


def chi_square_test(reference: np.ndarray, current: np.ndarray) -> Tuple[float, float]:
    """
    Chi-square test for categorical variables.
    
    Tests if observed frequencies differ from expected.
    
    Args:
        reference: Reference category counts
        current: Current category counts
    
    Returns:
        Tuple of (chi-square statistic, p-value)
    """
    from scipy import stats
    
    # Get unique categories
    categories = np.unique(np.concatenate([reference, current]))
    
    # Count occurrences
    ref_counts = np.array([np.sum(reference == c) for c in categories])
    cur_counts = np.array([np.sum(current == c) for c in categories])
    
    # Expected counts under null hypothesis
    total_ref = len(reference)
    total_cur = len(current)
    expected = ref_counts * (total_cur / total_ref)
    
    # Chi-square test
    statistic, p_value = stats.chisquare(cur_counts, expected)
    
    return float(statistic), float(p_value)


# ============================================================
# DRIFT DETECTOR
# ============================================================

class DriftDetector:
    """
    Unified drift detection for multiple features.
    
    Monitors data drift and concept drift in production.
    
    Example:
        >>> detector = DriftDetector()
        >>> detector.set_reference(X_train)
        >>> 
        >>> # In production
        >>> current_batch = get_production_data()
        >>> results = detector.detect_drift(current_batch)
        >>> 
        >>> for result in results:
        ...     if result.drift_detected:
        ...         alert(f"Drift in {result.feature_name}")
    """
    
    def __init__(self, 
                 method: str = 'ks',
                 threshold: float = 0.05,
                 feature_names: Optional[List[str]] = None):
        """
        Args:
            method: 'ks' (KS test), 'psi' (Population Stability Index)
            threshold: p-value threshold for KS, or PSI threshold
            feature_names: Optional names for features
        """
        self.method = method
        self.threshold = threshold
        self.feature_names = feature_names
        self.reference_data: Optional[np.ndarray] = None
        self.reference_stats: Dict[str, Dict] = {}
    
    def set_reference(self, data: np.ndarray):
        """
        Set reference (training) data distribution.
        
        Args:
            data: Training data (n_samples, n_features)
        """
        self.reference_data = np.asarray(data)
        
        # Compute and store statistics
        n_features = data.shape[1] if data.ndim > 1 else 1
        
        if self.feature_names is None:
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
        
        for i, name in enumerate(self.feature_names):
            feature_data = data[:, i] if data.ndim > 1 else data
            self.reference_stats[name] = {
                'mean': float(np.mean(feature_data)),
                'std': float(np.std(feature_data)),
                'min': float(np.min(feature_data)),
                'max': float(np.max(feature_data)),
                'median': float(np.median(feature_data))
            }
        
        logger.info(f"Reference data set: {len(data)} samples, {n_features} features")
    
    def detect_drift(self, current_data: np.ndarray) -> List[DriftResult]:
        """
        Detect drift in current data compared to reference.
        
        Args:
            current_data: Current production data
        
        Returns:
            List of DriftResult for each feature
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")
        
        current_data = np.asarray(current_data)
        results = []
        
        n_features = current_data.shape[1] if current_data.ndim > 1 else 1
        
        for i in range(n_features):
            ref_feature = self.reference_data[:, i] if self.reference_data.ndim > 1 else self.reference_data
            cur_feature = current_data[:, i] if current_data.ndim > 1 else current_data
            
            if self.method == 'ks':
                statistic, p_value = ks_test(ref_feature, cur_feature)
                drift_detected = p_value < self.threshold
            elif self.method == 'psi':
                statistic = psi(ref_feature, cur_feature)
                p_value = 1.0 - min(statistic / 0.2, 1.0)  # Approximate
                drift_detected = statistic >= self.threshold
            else:
                raise ValueError(f"Unknown method: {self.method}")
            
            result = DriftResult(
                feature_name=self.feature_names[i],
                drift_detected=drift_detected,
                statistic=statistic,
                p_value=p_value,
                threshold=self.threshold,
                method=self.method
            )
            results.append(result)
        
        # Log alerts
        drifted_features = [r.feature_name for r in results if r.drift_detected]
        if drifted_features:
            logger.warning(f"Drift detected in features: {drifted_features}")
        
        return results
    
    def get_reference_stats(self) -> Dict[str, Dict]:
        """Get reference data statistics."""
        return self.reference_stats


# ============================================================
# PERFORMANCE MONITOR
# ============================================================

class PerformanceMonitor:
    """
    Monitor model performance over time.
    
    Tracks:
    - Prediction latencies (p50, p95, p99)
    - Classification metrics (accuracy, precision, recall, F1)
    - Regression metrics (MSE, MAE, R²)
    - Error rates
    
    Example:
        >>> monitor = PerformanceMonitor(window_size=1000)
        >>> 
        >>> # Record predictions
        >>> monitor.record_prediction(y_true=1, y_pred=1, latency_ms=50)
        >>> 
        >>> # Get metrics
        >>> metrics = monitor.get_metrics()
        >>> print(f"Accuracy: {metrics.accuracy}")
    """
    
    def __init__(self, window_size: int = 1000):
        """
        Args:
            window_size: Number of recent predictions to track
        """
        self.window_size = window_size
        
        # Use deque for efficient sliding window
        self.y_true_history = deque(maxlen=window_size)
        self.y_pred_history = deque(maxlen=window_size)
        self.latency_history = deque(maxlen=window_size)
        
        self.total_predictions = 0
        self.total_errors = 0
    
    def record_prediction(self, y_true: Optional[float] = None,
                          y_pred: Optional[float] = None,
                          latency_ms: Optional[float] = None,
                          error: bool = False):
        """
        Record a prediction for monitoring.
        
        Args:
            y_true: Ground truth (if available)
            y_pred: Model prediction
            latency_ms: Inference latency in milliseconds
            error: Whether an error occurred
        """
        self.total_predictions += 1
        
        if y_true is not None:
            self.y_true_history.append(y_true)
        if y_pred is not None:
            self.y_pred_history.append(y_pred)
        if latency_ms is not None:
            self.latency_history.append(latency_ms)
        if error:
            self.total_errors += 1
    
    def get_metrics(self, task: str = 'classification') -> PerformanceMetrics:
        """
        Compute current performance metrics.
        
        Args:
            task: 'classification' or 'regression'
        
        Returns:
            PerformanceMetrics dataclass
        """
        metrics = PerformanceMetrics()
        
        # Latency metrics
        if self.latency_history:
            latencies = list(self.latency_history)
            metrics.latency_p50_ms = float(np.percentile(latencies, 50))
            metrics.latency_p95_ms = float(np.percentile(latencies, 95))
            metrics.latency_p99_ms = float(np.percentile(latencies, 99))
        
        # Label-based metrics
        if self.y_true_history and self.y_pred_history:
            y_true = np.array(list(self.y_true_history))
            y_pred = np.array(list(self.y_pred_history))
            
            if task == 'classification':
                metrics.accuracy = float(np.mean(y_true == y_pred))
                
                # Binary classification metrics
                if len(np.unique(y_true)) == 2:
                    tp = np.sum((y_true == 1) & (y_pred == 1))
                    fp = np.sum((y_true == 0) & (y_pred == 1))
                    fn = np.sum((y_true == 1) & (y_pred == 0))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                    
                    metrics.precision = float(precision)
                    metrics.recall = float(recall)
                    metrics.f1_score = float(f1)
            
            elif task == 'regression':
                metrics.mse = float(np.mean((y_true - y_pred) ** 2))
                metrics.mae = float(np.mean(np.abs(y_true - y_pred)))
        
        return metrics
    
    def get_error_rate(self) -> float:
        """Get overall error rate."""
        if self.total_predictions == 0:
            return 0.0
        return self.total_errors / self.total_predictions
    
    def check_degradation(self, 
                          metric_name: str,
                          threshold: float,
                          comparison: str = 'lt') -> bool:
        """
        Check if a metric has degraded past threshold.
        
        Args:
            metric_name: Name of metric to check
            threshold: Threshold value
            comparison: 'lt' (less than) or 'gt' (greater than)
        
        Returns:
            True if degradation detected
        """
        metrics = self.get_metrics()
        value = getattr(metrics, metric_name, None)
        
        if value is None:
            return False
        
        if comparison == 'lt':
            return value < threshold
        else:
            return value > threshold
    
    def reset(self):
        """Reset all tracked metrics."""
        self.y_true_history.clear()
        self.y_pred_history.clear()
        self.latency_history.clear()
        self.total_predictions = 0
        self.total_errors = 0


# ============================================================
# ALERTING
# ============================================================

class AlertManager:
    """
    Simple alerting system for monitoring.
    
    In production, integrate with:
    - PagerDuty
    - Slack
    - Email
    - AWS SNS
    
    Example:
        >>> alert_manager = AlertManager()
        >>> alert_manager.add_handler(slack_handler)
        >>> alert_manager.add_handler(email_handler)
        >>> 
        >>> alert_manager.alert(
        ...     severity="critical",
        ...     message="Model accuracy dropped below 80%",
        ...     metadata={"model": "fraud_detection", "accuracy": 0.75}
        ... )
    """
    
    def __init__(self):
        self.handlers: List[Callable] = []
        self.alert_history: List[Dict] = []
    
    def add_handler(self, handler: Callable):
        """Add an alert handler function."""
        self.handlers.append(handler)
    
    def alert(self, severity: str, message: str, 
              metadata: Optional[Dict[str, Any]] = None):
        """
        Send an alert.
        
        Args:
            severity: 'info', 'warning', 'critical'
            message: Alert message
            metadata: Additional context
        """
        alert_data = {
            'severity': severity,
            'message': message,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.alert_history.append(alert_data)
        
        # Log
        log_method = {
            'info': logger.info,
            'warning': logger.warning,
            'critical': logger.critical
        }.get(severity, logger.info)
        
        log_method(f"[{severity.upper()}] {message}")
        
        # Send to handlers
        for handler in self.handlers:
            try:
                handler(alert_data)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")
    
    def get_alerts(self, severity: Optional[str] = None,
                   limit: int = 100) -> List[Dict]:
        """Get recent alerts, optionally filtered by severity."""
        alerts = self.alert_history[-limit:]
        
        if severity:
            alerts = [a for a in alerts if a['severity'] == severity]
        
        return alerts


# ============================================================
# PROMETHEUS INTEGRATION
# ============================================================

def generate_prometheus_metrics(monitor: PerformanceMonitor,
                                 detector: Optional[DriftDetector] = None) -> str:
    """
    Generate Prometheus-compatible metrics string.
    
    Example output:
        # HELP model_latency_seconds Model inference latency
        # TYPE model_latency_seconds histogram
        model_latency_p50_ms 45.2
        model_latency_p95_ms 120.5
    
    Args:
        monitor: PerformanceMonitor instance
        detector: Optional DriftDetector instance
    
    Returns:
        Prometheus metrics string
    """
    lines = []
    metrics = monitor.get_metrics()
    
    # Latency metrics
    if metrics.latency_p50_ms is not None:
        lines.append("# HELP model_latency_ms Model inference latency in milliseconds")
        lines.append("# TYPE model_latency_ms gauge")
        lines.append(f'model_latency_p50_ms {metrics.latency_p50_ms}')
        lines.append(f'model_latency_p95_ms {metrics.latency_p95_ms}')
        lines.append(f'model_latency_p99_ms {metrics.latency_p99_ms}')
    
    # Accuracy
    if metrics.accuracy is not None:
        lines.append("# HELP model_accuracy Classification accuracy")
        lines.append("# TYPE model_accuracy gauge")
        lines.append(f'model_accuracy {metrics.accuracy}')
    
    # Error rate
    error_rate = monitor.get_error_rate()
    lines.append("# HELP model_error_rate Prediction error rate")
    lines.append("# TYPE model_error_rate gauge")
    lines.append(f'model_error_rate {error_rate}')
    
    # Total predictions
    lines.append("# HELP model_predictions_total Total predictions made")
    lines.append("# TYPE model_predictions_total counter")
    lines.append(f'model_predictions_total {monitor.total_predictions}')
    
    return '\n'.join(lines)


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Data structures
    'DriftResult', 'PerformanceMetrics',
    # Statistical tests
    'ks_test', 'ks_test_manual', 'psi', 'chi_square_test',
    # Monitors
    'DriftDetector', 'PerformanceMonitor', 'AlertManager',
    # Utils
    'generate_prometheus_metrics',
]
