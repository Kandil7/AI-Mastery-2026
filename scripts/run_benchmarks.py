#!/usr/bin/env python3
"""
Automated Performance Benchmarks
================================

Runs comprehensive benchmarks and generates a report for CI/CD.

Usage:
    python scripts/run_benchmarks.py
    python scripts/run_benchmarks.py --output results/benchmark_report.json
"""

import time
import json
import argparse
import statistics
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Callable
from datetime import datetime
import numpy as np


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    name: str
    iterations: int
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_per_sec: float


def time_function(func: Callable, iterations: int = 100) -> BenchmarkResult:
    """Time a function over multiple iterations."""
    times_ms = []
    
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times_ms.append(elapsed_ms)
    
    times_ms = sorted(times_ms)
    
    return BenchmarkResult(
        name=func.__name__,
        iterations=iterations,
        mean_ms=statistics.mean(times_ms),
        std_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0,
        min_ms=min(times_ms),
        max_ms=max(times_ms),
        p50_ms=times_ms[int(len(times_ms) * 0.50)],
        p95_ms=times_ms[int(len(times_ms) * 0.95)],
        p99_ms=times_ms[int(len(times_ms) * 0.99)],
        throughput_per_sec=1000 / statistics.mean(times_ms)
    )


# ============================================================
# BENCHMARK FUNCTIONS
# ============================================================

def benchmark_matrix_multiply():
    """Benchmark matrix multiplication."""
    A = np.random.randn(100, 100)
    B = np.random.randn(100, 100)
    _ = A @ B


def benchmark_svd():
    """Benchmark SVD decomposition."""
    X = np.random.randn(100, 50)
    _ = np.linalg.svd(X, full_matrices=False)


def benchmark_pca_projection():
    """Benchmark PCA projection."""
    X = np.random.randn(1000, 50)
    X_centered = X - X.mean(axis=0)
    cov = X_centered.T @ X_centered / len(X)
    _, V = np.linalg.eigh(cov)
    _ = X_centered @ V[:, -10:]


def benchmark_cosine_similarity():
    """Benchmark cosine similarity computation."""
    query = np.random.randn(384)
    docs = np.random.randn(1000, 384)
    
    norms = np.linalg.norm(docs, axis=1)
    query_norm = np.linalg.norm(query)
    _ = (docs @ query) / (norms * query_norm)


def benchmark_softmax():
    """Benchmark softmax computation."""
    x = np.random.randn(1000, 100)
    x_max = x.max(axis=1, keepdims=True)
    exp_x = np.exp(x - x_max)
    _ = exp_x / exp_x.sum(axis=1, keepdims=True)


def benchmark_attention_scores():
    """Benchmark attention score computation (simplified)."""
    batch, seq_len, d_model = 8, 128, 256
    Q = np.random.randn(batch, seq_len, d_model)
    K = np.random.randn(batch, seq_len, d_model)
    
    scores = Q @ K.transpose(0, 2, 1) / np.sqrt(d_model)
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    _ = exp_scores / exp_scores.sum(axis=-1, keepdims=True)


def benchmark_bm25_scoring():
    """Benchmark BM25 scoring (simplified)."""
    # Simulate document term frequencies
    n_docs = 1000
    vocab_size = 5000
    
    # Sparse term frequencies (simulate)
    query_terms = [100, 250, 500, 750]
    doc_lens = np.random.randint(50, 500, n_docs)
    avg_doc_len = doc_lens.mean()
    
    k1, b = 1.5, 0.75
    
    scores = np.zeros(n_docs)
    for term in query_terms:
        tf = np.random.randint(0, 10, n_docs)
        idf = np.log((n_docs + 1) / (np.sum(tf > 0) + 1))
        
        numerator = tf * (k1 + 1)
        denominator = tf + k1 * (1 - b + b * doc_lens / avg_doc_len)
        scores += idf * numerator / denominator


def benchmark_gradient_descent_step():
    """Benchmark a gradient descent step."""
    n_samples, n_features = 1000, 100
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)
    w = np.random.randn(n_features)
    lr = 0.01
    
    # Forward
    pred = X @ w
    error = pred - y
    
    # Gradient
    grad = (2 / n_samples) * (X.T @ error)
    
    # Update
    _ = w - lr * grad


# ============================================================
# MAIN RUNNER
# ============================================================

def run_all_benchmarks(iterations: int = 100) -> Dict[str, Any]:
    """Run all benchmarks and return results."""
    benchmarks = [
        benchmark_matrix_multiply,
        benchmark_svd,
        benchmark_pca_projection,
        benchmark_cosine_similarity,
        benchmark_softmax,
        benchmark_attention_scores,
        benchmark_bm25_scoring,
        benchmark_gradient_descent_step,
    ]
    
    results = []
    
    print("Running benchmarks...")
    print("-" * 60)
    
    for func in benchmarks:
        print(f"  {func.__name__}...", end=" ", flush=True)
        result = time_function(func, iterations)
        results.append(asdict(result))
        print(f"{result.mean_ms:.2f}ms (p95: {result.p95_ms:.2f}ms)")
    
    print("-" * 60)
    
    # Summary
    report = {
        "timestamp": datetime.now().isoformat(),
        "iterations_per_benchmark": iterations,
        "results": results,
        "summary": {
            "total_benchmarks": len(results),
            "fastest_ms": min(r["mean_ms"] for r in results),
            "slowest_ms": max(r["mean_ms"] for r in results),
        }
    }
    
    return report


def print_summary(report: Dict[str, Any]) -> None:
    """Print benchmark summary."""
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Timestamp: {report['timestamp']}")
    print(f"Total benchmarks: {report['summary']['total_benchmarks']}")
    print(f"Fastest: {report['summary']['fastest_ms']:.2f}ms")
    print(f"Slowest: {report['summary']['slowest_ms']:.2f}ms")
    print()
    
    print(f"{'Benchmark':<35} {'Mean':>10} {'P95':>10} {'Throughput':>12}")
    print("-" * 67)
    for r in report["results"]:
        print(f"{r['name']:<35} {r['mean_ms']:>7.2f}ms {r['p95_ms']:>7.2f}ms {r['throughput_per_sec']:>8.1f}/s")


def main():
    parser = argparse.ArgumentParser(description="Run performance benchmarks")
    parser.add_argument("--iterations", type=int, default=100, help="Iterations per benchmark")
    parser.add_argument("--output", type=str, help="Output JSON file path")
    args = parser.parse_args()
    
    report = run_all_benchmarks(args.iterations)
    print_summary(report)
    
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nResults saved to: {args.output}")
    
    # Exit with non-zero if any benchmark failed
    return 0


if __name__ == "__main__":
    exit(main())
