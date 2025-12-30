"""
Benchmarking and Performance Evaluation Tools
============================================

This module provides tools for benchmarking and evaluating the performance of 
machine learning models, algorithms, and systems in the AI-Mastery-2026 toolkit.

Features:
- Model performance evaluation
- Algorithm efficiency benchmarks
- System performance monitoring
- Scalability testing
- Cost-performance tradeoff analysis
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Callable, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import warnings
from functools import wraps
import psutil
import GPUtil
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import gc


@dataclass
class BenchmarkResult:
    """Data class to store benchmark results."""
    name: str
    metric: str
    value: float
    unit: str
    timestamp: str = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class PerformanceMetrics:
    """Data class to store comprehensive performance metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    throughput_per_second: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    gpu_memory_usage_mb: Optional[float] = None
    gpu_utilization_percent: Optional[float] = None
    training_time_seconds: Optional[float] = None
    inference_time_seconds: Optional[float] = None
    model_size_mb: Optional[float] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class BenchmarkSuite:
    """Comprehensive benchmarking suite for ML models and algorithms."""
    
    def __init__(self, output_dir: str = "benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.results: List[BenchmarkResult] = []
        
    def add_result(self, result: BenchmarkResult):
        """Add a benchmark result."""
        self.results.append(result)
        
    def save_results(self, filename: str = None):
        """Save benchmark results to file."""
        if filename is None:
            filename = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        results_path = self.output_dir / filename
        results_dict = [
            {
                "name": r.name,
                "metric": r.metric,
                "value": r.value,
                "unit": r.unit,
                "timestamp": r.timestamp,
                "metadata": r.metadata
            }
            for r in self.results
        ]
        
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"Benchmark results saved to {results_path}")
        
    def compare_results(self, results1: List[BenchmarkResult], 
                       results2: List[BenchmarkResult]) -> pd.DataFrame:
        """Compare two sets of benchmark results."""
        df1 = pd.DataFrame([
            {"name": r.name, "metric": r.metric, "value": r.value, "group": "Group1"}
            for r in results1
        ])
        
        df2 = pd.DataFrame([
            {"name": r.name, "metric": r.metric, "value": r.value, "group": "Group2"}
            for r in results2
        ])
        
        return pd.concat([df1, df2], ignore_index=True)


class ModelBenchmark:
    """Benchmark for machine learning models."""
    
    def __init__(self, model, X_test, y_test, model_name: str = "model"):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.model_name = model_name
        self.metrics = PerformanceMetrics()
        
    def benchmark_inference(self, num_samples: int = None) -> PerformanceMetrics:
        """Benchmark model inference performance."""
        if num_samples is None:
            num_samples = len(self.X_test)
        
        # Select subset of data
        X_subset = self.X_test[:num_samples]
        y_subset = self.y_test[:num_samples]
        
        # Measure latency
        start_time = time.time()
        latencies = []
        
        for i in range(len(X_subset)):
            sample = X_subset[i:i+1] if X_subset.ndim > 1 else np.array([X_subset[i]])
            
            inference_start = time.perf_counter()
            prediction = self.model.predict(sample)
            inference_end = time.perf_counter()
            
            latencies.append((inference_end - inference_start) * 1000)  # Convert to ms
        
        inference_time = time.time() - start_time
        
        # Calculate throughput
        throughput = len(X_subset) / inference_time if inference_time > 0 else 0
        
        # Calculate percentiles
        latencies = np.array(latencies)
        p50 = np.percentile(latencies, 50)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        # Calculate memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        
        # Calculate CPU usage
        cpu_percent = psutil.cpu_percent()
        
        # Calculate GPU usage if available
        gpu_memory_mb = 0
        gpu_util_percent = 0
        if torch.cuda.is_available():
            gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
            gpu_util_percent = torch.cuda.utilization()
        
        # Calculate model size
        model_size_mb = 0
        try:
            import pickle
            model_bytes = len(pickle.dumps(self.model))
            model_size_mb = model_bytes / (1024 * 1024)
        except:
            pass  # If model can't be pickled, skip size calculation
        
        # Update metrics
        self.metrics.latency_p50_ms = float(p50)
        self.metrics.latency_p95_ms = float(p95)
        self.metrics.latency_p99_ms = float(p99)
        self.metrics.throughput_per_second = float(throughput)
        self.metrics.memory_usage_mb = float(memory_mb)
        self.metrics.cpu_usage_percent = float(cpu_percent)
        self.metrics.gpu_memory_usage_mb = float(gpu_memory_mb)
        self.metrics.gpu_utilization_percent = float(gpu_util_percent)
        self.metrics.model_size_mb = float(model_size_mb)
        self.metrics.inference_time_seconds = float(inference_time)
        
        return self.metrics
    
    def benchmark_accuracy(self) -> PerformanceMetrics:
        """Benchmark model accuracy and related metrics."""
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Determine if classification or regression
        if len(np.unique(self.y_test)) <= 20:  # Assume classification if <= 20 unique values
            # Classification metrics
            self.metrics.accuracy = float(accuracy_score(self.y_test, y_pred))
            
            # Only calculate if we have more than 2 classes or if it's binary
            if len(np.unique(self.y_test)) > 1:
                try:
                    self.metrics.precision = float(precision_score(self.y_test, y_pred, average='weighted', zero_division=0))
                    self.metrics.recall = float(recall_score(self.y_test, y_pred, average='weighted', zero_division=0))
                    self.metrics.f1_score = float(f1_score(self.y_test, y_pred, average='weighted', zero_division=0))
                except:
                    # If precision/recall/f1 can't be calculated, skip them
                    pass
        else:
            # Regression metrics
            self.metrics.mse = float(mean_squared_error(self.y_test, y_pred))
            self.metrics.mae = float(mean_absolute_error(self.y_test, y_pred))
            self.metrics.r2_score = float(r2_score(self.y_test, y_pred))
        
        return self.metrics
    
    def run_complete_benchmark(self, num_samples: int = None) -> PerformanceMetrics:
        """Run complete benchmark including accuracy and performance."""
        self.benchmark_accuracy()
        self.benchmark_inference(num_samples)
        return self.metrics


class AlgorithmBenchmark:
    """Benchmark for algorithms and functions."""
    
    def __init__(self, algorithm_fn: Callable, name: str = "algorithm"):
        self.algorithm_fn = algorithm_fn
        self.name = name
        self.results = []
    
    def benchmark_time_complexity(self, input_sizes: List[int], 
                                 setup_fn: Callable = None) -> List[Tuple[int, float]]:
        """Benchmark time complexity with different input sizes."""
        times = []
        
        for size in input_sizes:
            if setup_fn:
                test_input = setup_fn(size)
            else:
                # Default: create random array of given size
                test_input = np.random.random(size)
            
            start_time = time.time()
            self.algorithm_fn(test_input)
            end_time = time.time()
            
            execution_time = end_time - start_time
            times.append((size, execution_time))
            self.results.append(BenchmarkResult(
                name=f"{self.name}_time",
                metric="execution_time",
                value=execution_time,
                unit="seconds",
                metadata={"input_size": size}
            ))
        
        return times
    
    def benchmark_space_complexity(self, input_sizes: List[int], 
                                  setup_fn: Callable = None) -> List[Tuple[int, float]]:
        """Benchmark space complexity with different input sizes."""
        memory_usages = []
        
        for size in input_sizes:
            if setup_fn:
                test_input = setup_fn(size)
            else:
                # Default: create random array of given size
                test_input = np.random.random(size)
            
            # Measure memory before
            gc.collect()
            process = psutil.Process()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Run algorithm
            self.algorithm_fn(test_input)
            
            # Measure memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            memory_used = mem_after - mem_before
            
            memory_usages.append((size, memory_used))
            self.results.append(BenchmarkResult(
                name=f"{self.name}_space",
                metric="memory_usage",
                value=memory_used,
                unit="MB",
                metadata={"input_size": size}
            ))
        
        return memory_usages


def benchmark_decorator(func: Callable) -> Callable:
    """Decorator to benchmark function execution time and memory usage."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Measure memory before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Measure time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        # Measure memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        execution_time = end_time - start_time
        memory_used = mem_after - mem_before
        
        print(f"Function {func.__name__}:")
        print(f"  Execution time: {execution_time:.4f} seconds")
        print(f"  Memory used: {memory_used:.2f} MB")
        
        return result
    return wrapper


class ScalabilityBenchmark:
    """Benchmark for testing scalability with increasing data sizes."""
    
    def __init__(self, model_factory: Callable, data_generator: Callable):
        self.model_factory = model_factory
        self.data_generator = data_generator
    
    def benchmark_scaling(self, sizes: List[int]) -> pd.DataFrame:
        """Benchmark how performance scales with data size."""
        results = []
        
        for size in sizes:
            print(f"Benchmarking with data size: {size}")
            
            # Generate data
            X, y = self.data_generator(size)
            
            # Create and train model
            model = self.model_factory()
            
            # Measure training time
            start_time = time.time()
            model.fit(X, y)
            training_time = time.time() - start_time
            
            # Measure inference time
            start_time = time.time()
            predictions = model.predict(X[:100])  # Test on 100 samples
            inference_time = time.time() - start_time
            
            # Measure memory usage
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            
            results.append({
                "data_size": size,
                "training_time_seconds": training_time,
                "inference_time_seconds": inference_time,
                "memory_usage_mb": memory_mb,
                "throughput": 100 / inference_time if inference_time > 0 else 0
            })
        
        return pd.DataFrame(results)


class CostPerformanceAnalyzer:
    """Analyze cost-performance tradeoffs for different models/configurations."""
    
    def __init__(self):
        self.costs = {}
        self.performance = {}
    
    def add_model_config(self, name: str, 
                        training_cost: float, 
                        inference_cost: float,
                        performance_metrics: PerformanceMetrics):
        """Add a model configuration with its costs and performance."""
        self.costs[name] = {
            "training_cost": training_cost,
            "inference_cost": inference_cost,
            "total_cost": training_cost + inference_cost
        }
        self.performance[name] = performance_metrics
    
    def analyze_tradeoffs(self) -> pd.DataFrame:
        """Analyze cost-performance tradeoffs."""
        results = []
        
        for name in self.costs:
            cost = self.costs[name]["total_cost"]
            perf = self.performance[name]
            
            # Calculate cost-performance ratios
            if perf.accuracy is not None:
                accuracy_cost_ratio = perf.accuracy / cost if cost > 0 else 0
            else:
                accuracy_cost_ratio = 0
                
            if perf.latency_p95_ms is not None:
                latency_cost_ratio = perf.latency_p95_ms / cost if cost > 0 else float('inf')
            else:
                latency_cost_ratio = float('inf')
            
            results.append({
                "model": name,
                "total_cost": cost,
                "accuracy": perf.accuracy,
                "latency_p95_ms": perf.latency_p95_ms,
                "accuracy_cost_ratio": accuracy_cost_ratio,
                "latency_cost_ratio": latency_cost_ratio
            })
        
        return pd.DataFrame(results).sort_values("accuracy_cost_ratio", ascending=False)


def plot_benchmark_results(results_df: pd.DataFrame, x_col: str, y_col: str, 
                         title: str = "Benchmark Results"):
    """Plot benchmark results."""
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=results_df, x=x_col, y=y_col, marker='o')
    plt.title(title)
    plt.xlabel(x_col.replace('_', ' ').title())
    plt.ylabel(y_col.replace('_', ' ').title())
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def compare_models_benchmark(models: Dict[str, Any], X_test, y_test) -> pd.DataFrame:
    """Compare multiple models on the same test data."""
    results = []
    
    for name, model in models.items():
        print(f"Benchmarking {name}...")
        
        # Create model benchmark
        model_bench = ModelBenchmark(model, X_test, y_test, name)
        metrics = model_bench.run_complete_benchmark(num_samples=100)  # Use subset for speed
        
        result = {
            "model": name,
            "accuracy": metrics.accuracy,
            "latency_p95_ms": metrics.latency_p95_ms,
            "throughput_per_second": metrics.throughput_per_second,
            "memory_usage_mb": metrics.memory_usage_mb
        }
        results.append(result)
    
    return pd.DataFrame(results)


# Example usage and testing functions
def run_example_benchmarks():
    """Run example benchmarks to demonstrate the tools."""
    from sklearn.datasets import make_classification, make_regression
    from sklearn.model_selection import train_test_split
    from src.ml.classical import RandomForestScratch, LinearRegressionScratch
    
    print("Running Example Benchmarks...")
    
    # Generate sample data
    X_cls, y_cls = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_reg, y_reg = make_regression(n_samples=1000, n_features=20, random_state=42)
    X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    # Example 1: Model benchmarking
    print("\n1. Model Benchmarking Example")
    rf_model = RandomForestScratch(n_estimators=50, max_depth=10)
    rf_model.fit(X_train_cls, y_train_cls)
    
    rf_bench = ModelBenchmark(rf_model, X_test_cls, y_test_cls, "RandomForest")
    rf_metrics = rf_bench.run_complete_benchmark(num_samples=100)
    
    print(f"Random Forest Metrics:")
    print(f"  Accuracy: {rf_metrics.accuracy:.3f}")
    print(f"  P95 Latency: {rf_metrics.latency_p95_ms:.2f}ms")
    print(f"  Throughput: {rf_metrics.throughput_per_second:.2f} inferences/sec")
    
    # Example 2: Algorithm benchmarking
    print("\n2. Algorithm Benchmarking Example")
    def sort_algorithm(arr):
        return np.sort(arr)
    
    alg_bench = AlgorithmBenchmark(sort_algorithm, "numpy_sort")
    time_results = alg_bench.benchmark_time_complexity([100, 500, 1000, 2000])
    
    print("Time complexity results:")
    for size, time_taken in time_results:
        print(f"  Size {size}: {time_taken:.4f}s")
    
    # Example 3: Model comparison
    print("\n3. Model Comparison Example")
    models = {
        "RandomForest": RandomForestScratch(n_estimators=50, max_depth=10),
        "LinearModel": LinearRegressionScratch()  # Assuming this works for classification too
    }
    
    # Train models
    models["RandomForest"].fit(X_train_cls, y_train_cls)
    # Note: LinearRegressionScratch is for regression, so we'll skip for classification comparison
    
    # For demonstration, just benchmark the random forest
    comparison_df = pd.DataFrame([{
        "model": "RandomForest",
        "accuracy": rf_metrics.accuracy,
        "latency_p95_ms": rf_metrics.latency_p95_ms,
        "throughput_per_second": rf_metrics.throughput_per_second,
        "memory_usage_mb": rf_metrics.memory_usage_mb
    }])
    
    print("Model comparison:")
    print(comparison_df)
    
    # Example 4: Scalability benchmark
    print("\n4. Scalability Benchmark Example")
    def model_factory():
        return RandomForestScratch(n_estimators=10, max_depth=5)
    
    def data_generator(size):
        X, y = make_classification(n_samples=size, n_features=10, n_classes=2, random_state=42)
        return X, y
    
    scale_bench = ScalabilityBenchmark(model_factory, data_generator)
    scale_results = scale_bench.benchmark_scaling([100, 500, 1000])
    
    print("Scalability results:")
    print(scale_results)
    
    # Example 5: Cost-performance analysis
    print("\n5. Cost-Performance Analysis Example")
    analyzer = CostPerformanceAnalyzer()
    
    # Add dummy configurations
    perf1 = PerformanceMetrics()
    perf1.accuracy = 0.92
    perf1.latency_p95_ms = 45.2
    
    perf2 = PerformanceMetrics()
    perf2.accuracy = 0.89
    perf2.latency_p95_ms = 23.1
    
    analyzer.add_model_config("HighAccuracy", 100.0, 10.0, perf1)
    analyzer.add_model_config("LowLatency", 50.0, 15.0, perf2)
    
    tradeoff_df = analyzer.analyze_tradeoffs()
    print("Cost-performance tradeoffs:")
    print(tradeoff_df)
    
    print("\nExample benchmarks completed!")


if __name__ == "__main__":
    run_example_benchmarks()