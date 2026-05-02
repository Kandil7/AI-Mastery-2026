"""
ML Learning Module Utilities
===========================

Common utility functions for ML experiments.

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Tuple, List, Optional


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, ...]:
    """
    Split data into training and test sets

    Args:
        X: Features
        y: Labels
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(y)
    n_test = int(n_samples * test_size)

    indices = np.random.permutation(n_samples)
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return (X[train_indices], X[test_indices], y[train_indices], y[test_indices])


def one_hot_encode(y: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    """
    Convert labels to one-hot encoding

    Args:
        y: Labels
        n_classes: Number of classes

    Returns:
        One-hot encoded labels
    """
    if n_classes is None:
        n_classes = len(np.unique(y))

    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1

    return one_hot


def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize features (z-score normalization)

    X_scaled = (X - mean) / std

    Args:
        X: Input features

    Returns:
        X_scaled, mean, std
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1  # Avoid division by zero

    X_scaled = (X - mean) / std

    return X_scaled, mean, std


def min_max_scale(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Min-max scaling to [0, 1]

    Args:
        X: Input features

    Returns:
        X_scaled, min, max
    """
    min_val = np.min(X, axis=0)
    max_val = np.max(X, axis=0)

    range_val = max_val - min_val
    range_val[range_val == 0] = 1

    X_scaled = (X - min_val) / range_val

    return X_scaled, min_val, max_val


def generate_moons(
    n_samples: int = 100, noise: float = 0.1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate moon-shaped (crescent) data

    Args:
        n_samples: Number of samples
        noise: Gaussian noise standard deviation

    Returns:
        X, y
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Outer moon
    theta_out = np.linspace(0, np.pi, n_samples_out)
    X_out = np.column_stack(
        [
            np.cos(theta_out) + np.random.randn(n_samples_out) * noise,
            np.sin(theta_out) + np.random.randn(n_samples_out) * noise,
        ]
    )

    # Inner moon
    theta_in = np.linspace(0, np.pi, n_samples_in)
    X_in = np.column_stack(
        [
            1 - np.cos(theta_in) + np.random.randn(n_samples_in) * noise,
            0.5 - np.sin(theta_in) + np.random.randn(n_samples_in) * noise,
        ]
    )

    X = np.vstack([X_out, X_in])
    y = np.array([0] * n_samples_out + [1] * n_samples_in)

    return X, y


def generate_circles(
    n_samples: int = 100, noise: float = 0.05, factor: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate concentric circles data

    Args:
        n_samples: Number of samples
        noise: Gaussian noise
        factor: Ratio of inner to outer radius

    Returns:
        X, y
    """
    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Outer circle
    t = np.linspace(0, 2 * np.pi, n_samples_out)
    X_out = np.column_stack(
        [
            np.cos(t) + np.random.randn(n_samples_out) * noise,
            np.sin(t) + np.random.randn(n_samples_out) * noise,
        ]
    )

    # Inner circle
    t = np.linspace(0, 2 * np.pi, n_samples_in)
    X_in = np.column_stack(
        [
            factor * np.cos(t) + np.random.randn(n_samples_in) * noise,
            factor * np.sin(t) + np.random.randn(n_samples_in) * noise,
        ]
    )

    X = np.vstack([X_out, X_in])
    y = np.array([0] * n_samples_out + [1] * n_samples_in)

    return X, y


def generate_blobs(
    n_samples: int = 100,
    n_features: int = 2,
    n_clusters: int = 3,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate blob (cluster) data

    Args:
        n_samples: Total samples
        n_features: Number of features
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        X, y
    """
    np.random.seed(random_state)

    samples_per_cluster = n_samples // n_clusters

    # Random cluster centers
    centers = np.random.randn(n_clusters, n_features) * 3

    X_list = []
    y_list = []

    for i in range(n_clusters):
        # Generate around center with std=1
        cluster_data = np.random.randn(samples_per_cluster, n_features) + centers[i]
        X_list.append(cluster_data)
        y_list.append(np.full(samples_per_cluster, i))

    # Handle remainder
    remainder = n_samples - len(y_list) * samples_per_cluster
    if remainder > 0:
        center_idx = np.random.randint(0, n_clusters)
        cluster_data = np.random.randn(remainder, n_features) + centers[center_idx]
        X_list.append(cluster_data)
        y_list.append(np.full(remainder, center_idx))

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    return X, y


def compute_class_weights(y: np.ndarray) -> dict:
    """Compute class weights for imbalanced data"""
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)

    weights = {}
    for c, count in zip(classes, counts):
        weights[c] = n_samples / (len(classes) * count)

    return weights


def shuffle_data(
    X: np.ndarray, y: np.ndarray, random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle X and y together"""
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.random.permutation(len(y))
    return X[indices], y[indices]


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute accuracy"""
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MSE"""
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² score"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot


print("Utilities loaded successfully!")
