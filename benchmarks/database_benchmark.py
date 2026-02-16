"""
Database Benchmark Utilities

Simple benchmarking tools for comparing database performance.
"""

import time
import statistics
from contextlib import contextmanager
from typing import Callable, Any, List, Dict
import random
import string


class BenchmarkResult:
    def __init__(self, name: str):
        self.name = name
        self.times: List[float] = []

    def add(self, duration: float):
        self.times.append(duration)

    def report(self) -> Dict[str, float]:
        if not self.times:
            return {}

        return {
            "name": self.name,
            "count": len(self.times),
            "min": min(self.times),
            "max": max(self.times),
            "mean": statistics.mean(self.times),
            "median": statistics.median(self.times),
            "stdev": statistics.stdev(self.times) if len(self.times) > 1 else 0,
            "p95": self.percentile(95),
            "p99": self.percentile(99),
            "total": sum(self.times),
        }

    def percentile(self, p: int) -> float:
        if not self.times:
            return 0
        sorted_times = sorted(self.times)
        idx = int(len(sorted_times) * p / 100)
        return sorted_times[min(idx, len(sorted_times) - 1)]


@contextmanager
def benchmark(name: str, result_store: Dict[str, BenchmarkResult] = None):
    """Context manager for simple benchmarking."""
    result = BenchmarkResult(name)
    start = time.perf_counter()
    try:
        yield result
    finally:
        duration = time.perf_counter() - start
        result.add(duration)
        if result_store is not None:
            if name not in result_store:
                result_store[name] = result
            else:
                result_store[name].times.extend(result.times)


def generate_test_data(num_rows: int = 1000) -> List[Dict]:
    """Generate random test data for benchmarking."""
    return [
        {
            "id": i,
            "name": f"User_{i}_{''.join(random.choices(string.ascii_lowercase, k=5))}",
            "email": f"user{i}@example.com",
            "age": random.randint(18, 80),
            "balance": random.uniform(0, 10000),
            "status": random.choice(["active", "inactive", "pending"]),
            "created_at": f"2024-{random.randint(1, 12):02d}-{random.randint(1, 28):02d}",
        }
        for i in range(num_rows)
    ]


def benchmark_query(
    query_func: Callable,
    iterations: int = 100,
    warmup: int = 10,
) -> Dict[str, float]:
    """
    Benchmark a query function.

    Args:
        query_func: Function to benchmark (should execute query and return)
        iterations: Number of times to run
        warmup: Number of warmup runs

    Returns:
        Dictionary with benchmark statistics
    """
    # Warmup
    for _ in range(warmup):
        query_func()

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        query_func()
        times.append(time.perf_counter() - start)

    return {
        "iterations": iterations,
        "min": min(times),
        "max": max(times),
        "mean": statistics.mean(times),
        "median": statistics.median(times),
        "p95": sorted(times)[int(len(times) * 0.95)],
    }


def compare_queries(
    queries: Dict[str, Callable], iterations: int = 100
) -> Dict[str, Dict]:
    """
    Compare multiple query implementations.

    Args:
        queries: Dictionary of name -> query function
        iterations: Number of iterations per query

    Returns:
        Dictionary of results per query
    """
    results = {}
    for name, query_func in queries.items():
        print(f"Benchmarking: {name}")
        results[name] = benchmark_query(query_func, iterations)

    return results


# Example usage for PostgreSQL
def postgres_benchmark_example():
    """Example PostgreSQL benchmarking."""
    import psycopg2

    conn = psycopg2.connect("postgresql://localhost/testdb")

    # Create test table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS benchmark_test (
            id SERIAL PRIMARY KEY,
            data JSONB,
            value INTEGER
        )
    """)
    conn.commit()

    # Benchmark insert
    def insert_query():
        import json

        data = {"key": "value", "number": random.randint(1, 1000)}
        conn.execute(
            "INSERT INTO benchmark_test (data, value) VALUES (%s, %s)",
            (json.dumps(data), random.randint(1, 1000)),
        )
        conn.commit()

    result = benchmark_query(insert_query, iterations=100)
    print(f"Insert: {result['mean'] * 1000:.2f}ms")


# Example usage for Redis
def redis_benchmark_example():
    """Example Redis benchmarking."""
    import redis

    r = redis.Redis()

    # Benchmark string operations
    def set_get():
        key = f"bench:{random.randint(1, 1000)}"
        r.set(key, "value")
        r.get(key)

    result = benchmark_query(set_get, iterations=1000)
    print(f"Redis SET/GET: {result['mean'] * 1000:.2f}ms")


if __name__ == "__main__":
    # Run example benchmarks
    print("Database Benchmark Tools")
    print("=" * 40)

    # Generate test data
    data = generate_test_data(100)
    print(f"Generated {len(data)} test records")
    print(f"Sample: {data[0]}")
