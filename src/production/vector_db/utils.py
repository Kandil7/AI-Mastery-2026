"""
Vector Operations Utilities
============================

Utility functions for vector operations.

Functions:
    cosine_similarity: Calculate cosine similarity between two vectors
    euclidean_distance: Calculate Euclidean distance between two vectors
    dot_product: Calculate dot product of two vectors
    normalize_vector: Normalize a vector to unit length
    batch_similarity: Calculate similarity between batches of vectors
    benchmark_index: Benchmark index performance

Author: AI-Mastery-2026
"""

import time
from typing import Any, Dict, List

import numpy as np

from .core import VectorIndex


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-12)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-12)
    return np.dot(v1_norm, v2_norm)


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors."""
    return np.linalg.norm(v1 - v2)


def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate dot product of two vectors."""
    return np.dot(v1, v2)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def batch_similarity(
    vectors1: np.ndarray, vectors2: np.ndarray, metric: str = "cosine"
) -> np.ndarray:
    """
    Calculate similarity between batches of vectors.

    Args:
        vectors1: Array of shape (n, dim)
        vectors2: Array of shape (m, dim)
        metric: Similarity metric ('cosine', 'euclidean', 'dot')

    Returns:
        Similarity matrix of shape (n, m)
    """
    if metric == "cosine":
        # Normalize vectors
        v1_norm = vectors1 / (np.linalg.norm(vectors1, axis=1, keepdims=True) + 1e-12)
        v2_norm = vectors2 / (np.linalg.norm(vectors2, axis=1, keepdims=True) + 1e-12)
        return np.dot(v1_norm, v2_norm.T)
    elif metric == "dot":
        return np.dot(vectors1, vectors2.T)
    elif metric == "euclidean":
        # Calculate pairwise distances
        diff = vectors1[:, np.newaxis, :] - vectors2[np.newaxis, :, :]
        return -np.linalg.norm(diff, axis=2)  # Negative for similarity
    else:
        raise ValueError(f"Unknown metric: {metric}")


def benchmark_index(
    index: VectorIndex, queries: List[np.ndarray], k: int = 10, num_iterations: int = 10
) -> Dict[str, float]:
    """Benchmark index performance."""
    times = []

    for _ in range(num_iterations):
        start_time = time.time()
        for query in queries:
            index.search(query, k)
        end_time = time.time()
        times.append(end_time - start_time)

    return {
        "avg_time_per_query": np.mean(times) / len(queries),
        "std_time_per_query": np.std(times) / len(queries),
        "total_time": np.mean(times),
        "qps": len(queries) / np.mean(times),
    }
