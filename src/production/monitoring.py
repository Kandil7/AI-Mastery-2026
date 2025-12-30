"""
Monitoring Module

This module implements monitoring and observability for ML models in production,
including metrics collection, logging, and alerting.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
import json
from datetime import datetime
import threading
import queue
import psutil
import GPUtil
from collections import defaultdict, deque
import statistics


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Enumeration of metric types."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class Metric:
    """Data class for metrics."""
    name: str
    value: float
    type: MetricType
    labels: Dict[str, str]
    timestamp: float


class MetricsCollector:
    """Collects and stores metrics."""
    
    def __init__(self):
        self.metrics: Dict[str, List[Metric]] = defaultdict(list)
        self.lock = threading.Lock()
    
    def record_metric(self, name: str, value: float, metric_type: MetricType, labels: Dict[str, str] = None):
        """Record a metric."""
        if labels is None:
            labels = {}
        
        with self.lock:
            metric = Metric(
                name=name,
                value=value,
                type=metric_type,
                labels=labels,
                timestamp=time.time()
            )
            self.metrics[name].append(metric)
    
    def get_metrics(self, name: str) -> List[Metric]:
        """Get metrics by name."""
        return self.metrics.get(name, [])
    
    def get_latest_value(self, name: str) -> Optional[float]:
        """Get the latest value for a metric."""
        metrics = self.get_metrics(name)
        if metrics:
            return metrics[-1].value
        return None
    
    def get_recent_values(self, name: str, seconds: int = 60) -> List[float]:
        """Get recent values for a metric."""
        cutoff_time = time.time() - seconds
        metrics = self.get_metrics(name)
        return [m.value for m in metrics if m.timestamp >= cutoff_time]


class ModelMonitor:
    """Monitors model performance and system resources."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics_collector = MetricsCollector()
        self.prediction_times = deque(maxlen=1000)  # Keep last 1000 prediction times
        self.prediction_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Initialize system monitoring
        self.system_monitor = SystemMonitor()
    
    def record_prediction(self, input_data: np.ndarray, prediction: Any, execution_time: float):
        """Record a prediction event."""
        self.prediction_count += 1
        self.prediction_times.append(execution_time)
        
        # Record metrics
        self.metrics_collector.record_metric(
            name=f"{self.model_name}_prediction_count",
            value=self.prediction_count,
            metric_type=MetricType.COUNTER,
            labels={"model": self.model_name}
        )
        
        self.metrics_collector.record_metric(
            name=f"{self.model_name}_prediction_time",
            value=execution_time,
            metric_type=MetricType.HISTOGRAM,
            labels={"model": self.model_name}
        )
        
        # Record input statistics
        if isinstance(input_data, np.ndarray):
            self.metrics_collector.record_metric(
                name=f"{self.model_name}_input_mean",
                value=float(np.mean(input_data)),
                metric_type=MetricType.GAUGE,
                labels={"model": self.model_name}
            )
            self.metrics_collector.record_metric(
                name=f"{self.model_name}_input_std",
                value=float(np.std(input_data)),
                metric_type=MetricType.GAUGE,
                labels={"model": self.model_name}
            )
    
    def record_error(self):
        """Record an error event."""
        self.error_count += 1
        
        self.metrics_collector.record_metric(
            name=f"{self.model_name}_error_count",
            value=self.error_count,
            metric_type=MetricType.COUNTER,
            labels={"model": self.model_name}
        )
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        if not self.prediction_times:
            return {}
        
        return {
            "model_name": self.model_name,
            "prediction_count": self.prediction_count,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.prediction_count, 1),
            "avg_prediction_time": statistics.mean(self.prediction_times),
            "p50_prediction_time": statistics.median(self.prediction_times),
            "p95_prediction_time": float(np.percentile(self.prediction_times, 95)),
            "p99_prediction_time": float(np.percentile(self.prediction_times, 99)),
            "uptime_seconds": time.time() - self.start_time
        }
    
    def check_drift(self, new_data: np.ndarray, reference_data: np.ndarray, threshold: float = 0.1) -> bool:
        """Check for data drift."""
        # Simple statistical drift detection
        new_mean = np.mean(new_data)
        ref_mean = np.mean(reference_data)
        
        drift_detected = abs(new_mean - ref_mean) > threshold * max(abs(ref_mean), 1e-8)
        
        self.metrics_collector.record_metric(
            name=f"{self.model_name}_drift_detected",
            value=int(drift_detected),
            metric_type=MetricType.GAUGE,
            labels={"model": self.model_name}
        )
        
        return drift_detected


class SystemMonitor:
    """Monitors system resources."""
    
    def __init__(self):
        self.metrics_collector = MetricsCollector()
        self.resource_history = deque(maxlen=100)  # Keep last 100 readings
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect system resource metrics."""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent
        
        # GPU metrics if available
        gpu_percent = 0.0
        gpu_memory_percent = 0.0
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_percent = gpu.load * 100
                gpu_memory_percent = gpu.memoryUtil * 100
        except:
            pass  # GPU not available
        
        # Record metrics
        self.metrics_collector.record_metric(
            name="system_cpu_percent",
            value=cpu_percent,
            metric_type=MetricType.GAUGE
        )
        
        self.metrics_collector.record_metric(
            name="system_memory_percent",
            value=memory_percent,
            metric_type=MetricType.GAUGE
        )
        
        self.metrics_collector.record_metric(
            name="system_disk_percent",
            value=disk_percent,
            metric_type=MetricType.GAUGE
        )
        
        if gpu_percent > 0:
            self.metrics_collector.record_metric(
                name="system_gpu_percent",
                value=gpu_percent,
                metric_type=MetricType.GAUGE
            )
            
            self.metrics_collector.record_metric(
                name="system_gpu_memory_percent",
                value=gpu_memory_percent,
                metric_type=MetricType.GAUGE
            )
        
        # Store in history
        self.resource_history.append({
            "timestamp": time.time(),
            "cpu": cpu_percent,
            "memory": memory_percent,
            "disk": disk_percent,
            "gpu": gpu_percent,
            "gpu_memory": gpu_memory_percent
        })
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "disk_percent": disk_percent,
            "gpu_percent": gpu_percent,
            "gpu_memory_percent": gpu_memory_percent
        }
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get system health summary."""
        if not self.resource_history:
            return {"status": "no_data", "timestamp": time.time()}
        
        recent_data = list(self.resource_history)[-10:]  # Last 10 readings
        
        avg_cpu = statistics.mean([d["cpu"] for d in recent_data])
        avg_memory = statistics.mean([d["memory"] for d in recent_data])
        avg_disk = statistics.mean([d["disk"] for d in recent_data])
        
        # Determine health status
        if avg_cpu > 90 or avg_memory > 90 or avg_disk > 90:
            status = "critical"
        elif avg_cpu > 75 or avg_memory > 75 or avg_disk > 75:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "avg_cpu_percent": avg_cpu,
            "avg_memory_percent": avg_memory,
            "avg_disk_percent": avg_disk,
            "timestamp": time.time()
        }


class AlertManager:
    """Manages alerts based on metrics."""
    
    def __init__(self):
        self.alerts = []
        self.rules = []
    
    def add_alert_rule(self, metric_name: str, condition: Callable[[float], bool], message: str):
        """Add an alert rule."""
        self.rules.append({
            "metric_name": metric_name,
            "condition": condition,
            "message": message
        })
    
    def check_alerts(self, metrics_collector: MetricsCollector):
        """Check all alert rules against current metrics."""
        for rule in self.rules:
            latest_value = metrics_collector.get_latest_value(rule["metric_name"])
            if latest_value is not None and rule["condition"](latest_value):
                alert = {
                    "timestamp": time.time(),
                    "metric_name": rule["metric_name"],
                    "value": latest_value,
                    "message": rule["message"]
                }
                self.alerts.append(alert)
                logger.warning(f"Alert: {rule['message']} (value: {latest_value})")
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts."""
        return self.alerts


class ModelDriftDetector:
    """Detects model drift in production."""
    
    def __init__(self, model_name: str, reference_data: np.ndarray, threshold: float = 0.1):
        self.model_name = model_name
        self.reference_data = reference_data
        self.threshold = threshold
        self.metrics_collector = MetricsCollector()
    
    def detect_drift(self, current_data: np.ndarray) -> Dict[str, Any]:
        """Detect drift between reference and current data."""
        # Calculate statistical differences
        ref_mean = np.mean(self.reference_data)
        ref_std = np.std(self.reference_data)
        curr_mean = np.mean(current_data)
        curr_std = np.std(current_data)
        
        mean_diff = abs(ref_mean - curr_mean)
        std_diff = abs(ref_std - curr_std)
        
        # Use Kolmogorov-Smirnov test for distribution comparison
        from scipy import stats
        ks_stat, p_value = stats.ks_2samp(self.reference_data.flatten(), current_data.flatten())
        
        drift_detected = ks_stat > self.threshold
        
        # Record metrics
        self.metrics_collector.record_metric(
            name=f"{self.model_name}_drift_mean_diff",
            value=mean_diff,
            metric_type=MetricType.GAUGE
        )
        
        self.metrics_collector.record_metric(
            name=f"{self.model_name}_drift_std_diff",
            value=std_diff,
            metric_type=MetricType.GAUGE
        )
        
        self.metrics_collector.record_metric(
            name=f"{self.model_name}_drift_ks_stat",
            value=ks_stat,
            metric_type=MetricType.GAUGE
        )
        
        return {
            "drift_detected": drift_detected,
            "mean_difference": mean_diff,
            "std_difference": std_diff,
            "ks_statistic": ks_stat,
            "p_value": p_value,
            "threshold": self.threshold
        }


class PerformanceTracker:
    """Tracks model performance over time."""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.metrics_collector = MetricsCollector()
        self.predictions = []
        self.targets = []
    
    def add_prediction(self, prediction: float, target: float):
        """Add a prediction-target pair."""
        self.predictions.append(prediction)
        self.targets.append(target)
        
        # Calculate and record metrics
        if len(self.predictions) >= 2:
            mse = np.mean((np.array(self.predictions) - np.array(self.targets)) ** 2)
            mae = np.mean(np.abs(np.array(self.predictions) - np.array(self.targets)))
            
            self.metrics_collector.record_metric(
                name=f"{self.model_name}_mse",
                value=mse,
                metric_type=MetricType.GAUGE
            )
            
            self.metrics_collector.record_metric(
                name=f"{self.model_name}_mae",
                value=mae,
                metric_type=MetricType.GAUGE
            )
    
    def get_accuracy_metrics(self) -> Dict[str, float]:
        """Get accuracy metrics."""
        if len(self.predictions) < 2:
            return {}
        
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))
        rmse = np.sqrt(mse)
        
        # R-squared
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            "mse": float(mse),
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2)
        }


# Global monitoring instances
model_monitors: Dict[str, ModelMonitor] = {}
system_monitor = SystemMonitor()
alert_manager = AlertManager()


def initialize_monitoring():
    """Initialize monitoring components."""
    # Add some default alert rules
    alert_manager.add_alert_rule(
        "system_cpu_percent",
        lambda x: x > 90,
        "High CPU usage detected"
    )
    
    alert_manager.add_alert_rule(
        "system_memory_percent",
        lambda x: x > 90,
        "High memory usage detected"
    )
    
    alert_manager.add_alert_rule(
        f"model_prediction_time",
        lambda x: x > 1.0,  # More than 1 second
        "Slow prediction time detected"
    )


def get_monitor(model_name: str) -> ModelMonitor:
    """Get or create a monitor for a model."""
    if model_name not in model_monitors:
        model_monitors[model_name] = ModelMonitor(model_name)
    return model_monitors[model_name]


def log_prediction(model_name: str, input_data: np.ndarray, prediction: Any, execution_time: float):
    """Log a prediction event."""
    monitor = get_monitor(model_name)
    monitor.record_prediction(input_data, prediction, execution_time)


def log_error(model_name: str):
    """Log an error event."""
    monitor = get_monitor(model_name)
    monitor.record_error()


def get_model_performance(model_name: str) -> Dict[str, Any]:
    """Get performance metrics for a model."""
    monitor = get_monitor(model_name)
    return monitor.get_performance_metrics()


def get_system_metrics() -> Dict[str, Any]:
    """Get system metrics."""
    return system_monitor.collect_system_metrics()


def get_system_health() -> Dict[str, Any]:
    """Get system health."""
    return system_monitor.get_system_health()


# Initialize monitoring when module is loaded
initialize_monitoring()