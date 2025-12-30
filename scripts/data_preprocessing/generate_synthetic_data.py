"""
Synthetic Data Generator
========================
Generate synthetic datasets for ML training and testing.

Supports:
- Classification data
- Regression data
- Time series data
- Text/NLP data
- Recommendation system data

Author: AI-Mastery-2026
"""

import numpy as np
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# ============================================================
# DATA GENERATION FUNCTIONS
# ============================================================

def generate_classification_data(
    n_samples: int = 1000,
    n_features: int = 20,
    n_classes: int = 2,
    n_informative: int = 10,
    n_redundant: int = 5,
    class_sep: float = 1.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic classification dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Total number of features
        n_classes: Number of classes
        n_informative: Number of informative features
        n_redundant: Number of redundant features
        class_sep: Class separation factor
        random_state: Random seed
    
    Returns:
        Tuple of (X, y) arrays
    """
    np.random.seed(random_state)
    
    # Generate informative features from class centroids
    centroids = np.random.randn(n_classes, n_informative) * class_sep
    
    # Assign samples to classes
    y = np.random.randint(0, n_classes, n_samples)
    
    # Generate features
    X_informative = centroids[y] + np.random.randn(n_samples, n_informative) * 0.5
    
    # Generate redundant features as linear combinations
    if n_redundant > 0:
        weights = np.random.randn(n_informative, n_redundant)
        X_redundant = X_informative @ weights + np.random.randn(n_samples, n_redundant) * 0.1
    else:
        X_redundant = np.zeros((n_samples, 0))
    
    # Generate noise features
    n_noise = n_features - n_informative - n_redundant
    X_noise = np.random.randn(n_samples, max(0, n_noise))
    
    # Combine
    X = np.hstack([X_informative, X_redundant, X_noise])[:, :n_features]
    
    return X, y


def generate_regression_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int = 5,
    noise: float = 0.1,
    bias: float = 0.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic regression dataset.
    
    Args:
        n_samples: Number of samples
        n_features: Total number of features
        n_informative: Number of features used in target
        noise: Standard deviation of noise
        bias: Bias term
        random_state: Random seed
    
    Returns:
        Tuple of (X, y) arrays
    """
    np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    
    # True coefficients
    coef = np.zeros(n_features)
    coef[:n_informative] = np.random.randn(n_informative)
    
    # Generate target
    y = X @ coef + bias + np.random.randn(n_samples) * noise
    
    return X, y


def generate_time_series(
    n_samples: int = 1000,
    n_features: int = 1,
    trend: str = "linear",  # linear, quadratic, none
    seasonality: int = 0,  # Period of seasonality (0 = none)
    noise: float = 0.1,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate synthetic time series data.
    
    Args:
        n_samples: Length of time series
        n_features: Number of parallel series
        trend: Type of trend
        seasonality: Seasonal period (0 for none)
        noise: Noise level
        random_state: Random seed
    
    Returns:
        Time series array of shape (n_samples, n_features)
    """
    np.random.seed(random_state)
    
    t = np.arange(n_samples)
    series = np.zeros((n_samples, n_features))
    
    for i in range(n_features):
        # Trend component
        if trend == "linear":
            trend_component = 0.01 * t
        elif trend == "quadratic":
            trend_component = 0.0001 * t ** 2
        else:
            trend_component = np.zeros(n_samples)
        
        # Seasonality component
        if seasonality > 0:
            season_component = np.sin(2 * np.pi * t / seasonality)
        else:
            season_component = np.zeros(n_samples)
        
        # Noise
        noise_component = np.random.randn(n_samples) * noise
        
        series[:, i] = trend_component + season_component + noise_component
    
    return series


def generate_text_data(
    n_samples: int = 100,
    n_classes: int = 3,
    vocab_size: int = 1000,
    seq_length: Tuple[int, int] = (10, 50),
    random_state: int = 42
) -> Tuple[List[str], List[int]]:
    """
    Generate synthetic text classification data.
    
    Creates pseudo-text with class-specific vocabulary distributions.
    
    Args:
        n_samples: Number of samples
        n_classes: Number of classes
        vocab_size: Size of vocabulary
        seq_length: (min, max) sequence length
        random_state: Random seed
    
    Returns:
        Tuple of (texts, labels)
    """
    np.random.seed(random_state)
    
    # Create vocabulary
    vocab = [f"word_{i}" for i in range(vocab_size)]
    
    # Class-specific word distributions
    class_distributions = []
    for _ in range(n_classes):
        probs = np.random.dirichlet(np.ones(vocab_size) * 0.5)
        class_distributions.append(probs)
    
    texts = []
    labels = []
    
    for _ in range(n_samples):
        label = np.random.randint(0, n_classes)
        length = np.random.randint(seq_length[0], seq_length[1])
        
        # Sample words from class distribution
        word_indices = np.random.choice(
            vocab_size, size=length, p=class_distributions[label]
        )
        text = " ".join([vocab[i] for i in word_indices])
        
        texts.append(text)
        labels.append(label)
    
    return texts, labels


def generate_recommendation_data(
    n_users: int = 100,
    n_items: int = 500,
    n_interactions: int = 5000,
    rating_range: Tuple[int, int] = (1, 5),
    sparsity: float = 0.95,
    random_state: int = 42
) -> Dict[str, np.ndarray]:
    """
    Generate synthetic recommendation system data.
    
    Args:
        n_users: Number of users
        n_items: Number of items
        n_interactions: Number of user-item interactions
        rating_range: (min, max) rating values
        sparsity: Matrix sparsity (fraction of missing entries)
        random_state: Random seed
    
    Returns:
        Dict with user_ids, item_ids, ratings arrays
    """
    np.random.seed(random_state)
    
    # Generate user and item latent factors
    n_factors = 10
    user_factors = np.random.randn(n_users, n_factors)
    item_factors = np.random.randn(n_items, n_factors)
    
    # Generate interactions
    user_ids = np.random.randint(0, n_users, n_interactions)
    item_ids = np.random.randint(0, n_items, n_interactions)
    
    # Calculate ratings based on latent factors
    true_ratings = np.sum(
        user_factors[user_ids] * item_factors[item_ids], axis=1
    )
    
    # Scale to rating range
    min_r, max_r = rating_range
    ratings = (true_ratings - true_ratings.min()) / (true_ratings.max() - true_ratings.min())
    ratings = ratings * (max_r - min_r) + min_r
    ratings = np.round(ratings).astype(int)
    ratings = np.clip(ratings, min_r, max_r)
    
    return {
        "user_ids": user_ids,
        "item_ids": item_ids,
        "ratings": ratings
    }


def generate_embedding_data(
    n_samples: int = 1000,
    n_clusters: int = 5,
    embedding_dim: int = 768,
    cluster_std: float = 0.3,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic embedding vectors with cluster structure.
    
    Useful for testing vector search and clustering algorithms.
    
    Args:
        n_samples: Number of embeddings
        n_clusters: Number of clusters
        embedding_dim: Embedding dimension
        cluster_std: Within-cluster standard deviation
        random_state: Random seed
    
    Returns:
        Tuple of (embeddings, cluster_labels)
    """
    np.random.seed(random_state)
    
    # Generate cluster centers on unit sphere
    centers = np.random.randn(n_clusters, embedding_dim)
    centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
    
    # Assign samples to clusters
    labels = np.random.randint(0, n_clusters, n_samples)
    
    # Generate embeddings around centers
    embeddings = centers[labels] + np.random.randn(n_samples, embedding_dim) * cluster_std
    
    # Normalize to unit sphere
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    return embeddings, labels


# ============================================================
# DATA SAVING UTILITIES
# ============================================================

def save_dataset(
    data: Dict[str, Any],
    output_path: str,
    format: str = "npz"
):
    """
    Save generated dataset to file.
    
    Args:
        data: Dictionary of arrays/lists
        output_path: Output file path
        format: Output format (npz, json, csv)
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "npz":
        np.savez(path, **{k: np.array(v) for k, v in data.items()})
    elif format == "json":
        with open(path, 'w') as f:
            json.dump({k: v.tolist() if hasattr(v, 'tolist') else v for k, v in data.items()}, f)
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"Saved dataset to {path}")


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    """Generate all synthetic datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic ML datasets")
    parser.add_argument("--output-dir", default="data/synthetic", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating synthetic datasets...")
    
    # Classification
    X, y = generate_classification_data(n_samples=5000, random_state=args.seed)
    save_dataset({"X": X, "y": y}, output_dir / "classification.npz")
    print(f"✓ Classification: {X.shape}")
    
    # Regression
    X, y = generate_regression_data(n_samples=5000, random_state=args.seed)
    save_dataset({"X": X, "y": y}, output_dir / "regression.npz")
    print(f"✓ Regression: {X.shape}")
    
    # Time series
    ts = generate_time_series(n_samples=1000, seasonality=50, random_state=args.seed)
    save_dataset({"series": ts}, output_dir / "time_series.npz")
    print(f"✓ Time series: {ts.shape}")
    
    # Text
    texts, labels = generate_text_data(n_samples=1000, random_state=args.seed)
    save_dataset({"texts": texts, "labels": labels}, output_dir / "text_classification.json", format="json")
    print(f"✓ Text: {len(texts)} samples")
    
    # Recommendations
    rec_data = generate_recommendation_data(random_state=args.seed)
    save_dataset(rec_data, output_dir / "recommendations.npz")
    print(f"✓ Recommendations: {len(rec_data['ratings'])} interactions")
    
    # Embeddings
    embeddings, clusters = generate_embedding_data(random_state=args.seed)
    save_dataset({"embeddings": embeddings, "clusters": clusters}, output_dir / "embeddings.npz")
    print(f"✓ Embeddings: {embeddings.shape}")
    
    print(f"\nAll datasets saved to {output_dir}")


if __name__ == "__main__":
    main()
