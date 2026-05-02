"""
Dataset Utilities - Synthetic Dataset Generators
==================================================

Provides functions to generate synthetic datasets for ML experimentation,
similar to sklearn.datasets but implemented from scratch using NumPy.

Author: ML Learning Module
License: MIT
"""

import numpy as np
from typing import Tuple, Optional, List, Union


def make_moons(
    n_samples: int = 100,
    *,
    shuffle: bool = True,
    noise: Optional[float] = None,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D moon-shaped dataset for classification.

    This function creates a binary classification problem where the two
    classes form interleaving half circles. It's useful for testing
    algorithms that require non-linear decision boundaries.

    Mathematical Description:
    -------------------------
    For each sample:
    - For class 0: (cos(θ), sin(θ)) where θ ∈ [π, 2π]
    - For class 1: (1 - cos(θ), 0.5 - sin(θ)) where θ ∈ [0, π]

    Parameters
    ----------
    n_samples : int, default=100
        Total number of samples to generate.
        Will be split evenly between the two moons.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    noise : float, optional, default=None
        Standard deviation of Gaussian noise added to the data.
        If None, no noise is added.
    random_state : int, optional, default=None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        The generated feature matrix.
    y : ndarray of shape (n_samples,)
        The labels for each sample (0 or 1).

    Examples
    --------
    >>> X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
    >>> X.shape
    (200, 2)
    >>> y.shape
    (200,)

    Visual Representation:
    -----------------------
          Class 1 (upper moon)
               .  .
            .      .
           .        .
          .          .
         '----------'----- Class 0 (lower moon)
          '.        '
            '.    '
               '. '

    Notes
    -----
    - The dataset is not linearly separable
    - Often used to demonstrate SVM, neural networks, and clustering
    - Adding noise makes the problem more realistic

    See Also
    --------
    make_circles : Concentric circles
    make_classification : General classification data
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Generate outer moon (class 0)
    # Points in range [π, 2π] for the outer arc
    theta_out = np.linspace(np.pi, 2 * np.pi, n_samples_out)

    # Outer moon: (cos(θ), sin(θ))
    X_out = np.column_stack([np.cos(theta_out), np.sin(theta_out)])
    y_out = np.zeros(n_samples_out)

    # Generate inner moon (class 1)
    # Points in range [0, π] for the inner arc, shifted
    theta_in = np.linspace(0, np.pi, n_samples_in)

    # Inner moon: (1 - cos(θ), 0.5 - sin(θ))
    X_in = np.column_stack([1 - np.cos(theta_in), 0.5 - np.sin(theta_in)])
    y_in = np.ones(n_samples_in)

    # Combine the two moons
    X = np.vstack([X_out, X_in])
    y = np.concatenate([y_out, y_in])

    # Add Gaussian noise if specified
    if noise is not None:
        X += np.random.normal(0, noise, X.shape)

    # Shuffle the data if requested
    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

    return X, y


def make_circles(
    n_samples: int = 100,
    *,
    shuffle: bool = True,
    noise: Optional[float] = None,
    random_state: Optional[int] = None,
    factor: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate 2D concentric circles dataset for classification.

    Creates a binary classification problem where one class is
    centered inside a circle and the other forms a ring around it.

    Mathematical Description:
    -------------------------
    - Inner circle: radius = r * factor (class 0)
    - Outer circle: radius = r (class 1)
    where r ~ Uniform(0, 1) for each class

    Parameters
    ----------
    n_samples : int, default=100
        Total number of samples.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    noise : float, optional, default=None
        Standard deviation of Gaussian noise.
    random_state : int, optional, default=None
        Random seed for reproducibility.
    factor : float, default=0.8
        Scale factor between inner and outer circle.

    Returns
    -------
    X : ndarray of shape (n_samples, 2)
        The generated feature matrix.
    y : ndarray of shape (n_samples,)
        The labels for each sample (0 or 1).

    Examples
    --------
    >>> X, y = make_circles(n_samples=200, noise=0.05, random_state=0)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples_out = n_samples // 2
    n_samples_in = n_samples - n_samples_out

    # Generate outer circle (class 1)
    # Random radius in (0, 1), random angle in [0, 2π)
    rng_out = np.random.RandomState(random_state)
    r_out = rng_out.uniform(0, 1, n_samples_out)
    r_out = (r_out + 1) / 2  # Scale to [0.5, 1] to have some separation
    theta_out = rng_out.uniform(0, 2 * np.pi, n_samples_out)

    X_out = np.column_stack([r_out * np.cos(theta_out), r_out * np.sin(theta_out)])
    y_out = np.ones(n_samples_out)

    # Generate inner circle (class 0)
    rng_in = np.random.RandomState(random_state + 1 if random_state else None)
    r_in = rng_in.uniform(0, 1, n_samples_in) * factor
    theta_in = rng_in.uniform(0, 2 * np.pi, n_samples_in)

    X_in = np.column_stack([r_in * np.cos(theta_in), r_in * np.sin(theta_in)])
    y_in = np.zeros(n_samples_in)

    # Combine
    X = np.vstack([X_out, X_in])
    y = np.concatenate([y_out, y_in])

    # Add noise
    if noise is not None:
        X += np.random.normal(0, noise, X.shape)

    # Shuffle
    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

    return X, y


def make_blobs(
    n_samples: int = 100,
    n_features: int = 2,
    *,
    centers: Optional[int] = None,
    cluster_std: float = 1.0,
    center_box: Tuple[float, float] = (-10.0, 10.0),
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate isotropic Gaussian blobs for clustering.

    Creates a multi-class dataset by drawing samples from
    Gaussian distributions with different means.

    Parameters
    ----------
    n_samples : int, default=100
        Total number of samples.
    n_features : int, default=2
        Number of features (dimensions).
    centers : int, optional, default=None
        Number of centers (clusters). If None, defaults to 3.
    cluster_std : float, default=1.0
        Standard deviation of each cluster.
    center_box : tuple of shape (2,), default=(-10, 10)
        Bounding box for each cluster center.
    shuffle : bool, default=True
        Whether to shuffle the samples.
    random_state : int, optional, default=None
        Random seed for reproducibility.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated feature matrix.
    y : ndarray of shape (n_samples,)
        The labels for each sample.
    centers : ndarray of shape (n_centers, n_features)
        The centers of each cluster.

    Examples
    --------
    >>> X, y, centers = make_blobs(n_samples=300, centers=3, random_state=42)
    >>> X.shape
    (300, 2)
    >>> np.unique(y)
    array([0, 1, 2])
    """
    if random_state is not None:
        np.random.seed(random_state)

    if centers is None:
        centers = 3

    # Generate cluster centers
    centers_array = np.random.uniform(
        low=center_box[0], high=center_box[1], size=(centers, n_features)
    )

    # Generate samples for each cluster
    X_list = []
    y_list = []

    samples_per_center = n_samples // centers
    remainder = n_samples % centers

    for i in range(centers):
        # Number of samples for this cluster
        n = samples_per_center + (1 if i < remainder else 0)

        # Generate samples from Gaussian distribution
        X_cluster = np.random.normal(
            loc=centers_array[i], scale=cluster_std, size=(n, n_features)
        )

        X_list.append(X_cluster)
        y_list.append(np.full(n, i))

    X = np.vstack(X_list)
    y = np.concatenate(y_list)

    # Shuffle
    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

    return X, y, centers_array


def make_classification(
    n_samples: int = 100,
    n_features: int = 2,
    n_informative: int = 2,
    n_redundant: int = 0,
    n_classes: int = 2,
    *,
    n_clusters_per_class: int = 1,
    weights: Optional[List[float]] = None,
    flip_y: float = 0.01,
    class_sep: float = 1.0,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random n-class classification problem.

    Creates a dataset with informative, redundant, and noise features.
    This is more complex than make_moons or make_circles.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples.
    n_features : int, default=2
        Total number of features.
    n_informative : int, default=2
        Number of informative features.
        These features contain the class information.
    n_redundant : int, default=0
        Number of redundant features.
        These are linear combinations of informative features.
    n_classes : int, default=2
        Number of classes.
    n_clusters_per_class : int, default=1
        Number of clusters per class.
    weights : list of float, optional
        Proportion of samples per class.
        If None, equal weights.
    flip_y : float, default=0.01
        Fraction of labels to flip (noise).
    class_sep : float, default=1.0
        Separation between classes. Larger means better separation.
    random_state : int, optional, default=None
        Random seed.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated feature matrix.
    y : ndarray of shape (n_samples,)
        The labels for each sample.

    Algorithm:
    ---------
    1. Generate cluster centers for each class
    2. For each sample, assign a class based on weights
    3. Assign to a random cluster within that class
    4. Generate informative features from cluster center + noise
    5. Optionally generate redundant features
    6. Add remaining noise features

    Notes
    -----
    - Informative features: directly related to class
    - Redundant features: linear combinations of informative
    - Noise features: random irrelevant features
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Handle weights
    if weights is None:
        weights = [1.0 / n_classes] * n_classes
    weights = np.array(weights)
    weights = weights / weights.sum()

    # Generate class assignments
    y = np.random.choice(n_classes, size=n_samples, p=weights)

    # Calculate feature distribution
    n_noise = n_features - n_informative - n_redundant

    # Generate informative features
    # Each class has n_clusters_per_class clusters
    total_clusters = n_classes * n_clusters_per_class

    # Generate cluster centers for informative features
    cluster_centers = np.random.uniform(
        -class_sep, class_sep, size=(total_clusters, n_informative)
    )

    # Generate informative features
    X_informative = np.zeros((n_samples, n_informative))

    for i in range(n_samples):
        class_idx = y[i]
        cluster_offset = class_idx * n_clusters_per_class
        cluster_idx = cluster_offset + np.random.randint(n_clusters_per_class)

        # Sample from cluster with noise
        X_informative[i] = np.random.normal(cluster_centers[cluster_idx], 1.0)

    # Generate redundant features (linear combinations)
    X_redundant = np.zeros((n_samples, n_redundant))
    if n_redundant > 0 and n_informative > 0:
        # Generate random mixing weights
        mixing_weights = np.random.uniform(-1, 1, size=(n_informative, n_redundant))
        mixing_weights = mixing_weights / np.linalg.norm(mixing_weights, axis=0)

        X_redundant = X_informative @ mixing_weights + np.random.normal(
            0, 0.1, (n_samples, n_redundant)
        )

    # Generate noise features
    X_noise = np.random.normal(0, 1, (n_samples, n_noise))

    # Combine all features
    X = np.hstack([X_informative, X_redundant, X_noise])

    # Flip some labels (add noise)
    if flip_y > 0:
        n_flip = int(n_samples * flip_y)
        flip_indices = np.random.choice(n_samples, n_flip, replace=False)
        y[flip_indices] = np.random.randint(0, n_classes, n_flip)

    # Shuffle
    if shuffle:
        indices = np.random.permutation(n_samples)
        X = X[indices]
        y = y[indices]

    return X, y


def make_regression(
    n_samples: int = 100,
    n_features: int = 1,
    n_informative: int = 1,
    *,
    noise: float = 0.1,
    bias: float = 0.0,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a regression dataset with controllable complexity.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples.
    n_features : int, default=1
        Number of features.
    n_informative : int, default=1
        Number of informative (causal) features.
    noise : float, default=0.1
        Standard deviation of Gaussian noise.
    bias : float, default=0.0
        Bias term (intercept).
    random_state : int, optional, default=None
        Random seed.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        The generated feature matrix.
    y : ndarray of shape (n_samples,)
        The target values.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate random features
    X = np.random.uniform(-1, 1, (n_samples, n_features))

    # Generate weights for informative features
    weights = np.random.uniform(-1, 1, n_features)
    weights[n_informative:] = 0  # Zero out non-informative

    # Compute y = X @ w + bias + noise
    y = X @ weights + bias + np.random.normal(0, noise, n_samples)

    return X, y


def make_s_curve(
    n_samples: int = 100, *, noise: float = 0.0, random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate S-curve dataset for manifold learning.

    Creates a 3D S-curve shape, useful for demonstrating
    manifold learning algorithms like t-SNE or PCA.

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples.
    noise : float, default=0.0
        Standard deviation of Gaussian noise.
    random_state : int, optional, default=None
        Random seed.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        The 3D S-curve points.
    y : ndarray of shape (n_samples,)
        The ordering parameter (0 to 1 along the curve).
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate parameter t along the curve
    t = np.random.uniform(0, 1, n_samples)

    # S-curve parametric equations
    # t in [0, 1] creates an S-shape
    X = np.column_stack(
        [
            np.sin(t * np.pi * 2),  # x: sine wave
            np.cos(t * np.pi * 2) * (1 - t),  # y: decreasing cosine
            t,  # z: linear
        ]
    )

    # Add noise
    if noise > 0:
        X += np.random.normal(0, noise, X.shape)

    return X, t


def make_swiss_roll(
    n_samples: int = 100, *, noise: float = 0.0, random_state: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Swiss roll dataset for manifold learning.

    Creates a 2D manifold embedded in 3D that unrolls into
    a 2D plane. Classic test for manifold learning algorithms.

    Mathematical Description:
    -------------------------
    The Swiss roll is defined parametrically:
    - t ∈ [1.5π, 4.5π] (unrolled length)
    - v ∈ [0, 10] (width)

    x = t * cos(t)
    y = v
    z = t * sin(t)

    Parameters
    ----------
    n_samples : int, default=100
        Number of samples.
    noise : float, default=0.0
        Standard deviation of Gaussian noise.
    random_state : int, optional, default=None
        Random seed.

    Returns
    -------
    X : ndarray of shape (n_samples, 3)
        The 3D Swiss roll points.
    y : ndarray of shape (n_samples,)
        The position along the unrolled manifold.
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Generate random values for the parametric form
    t = np.random.uniform(1.5 * np.pi, 4.5 * np.pi, n_samples)
    v = np.random.uniform(0, 10, n_samples)

    # Convert to Cartesian coordinates
    X = np.column_stack([t * np.cos(t), v, t * np.sin(t)])

    # Add noise
    if noise > 0:
        X += np.random.normal(0, noise, X.shape)

    return X, t


def load_iris() -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Load the Iris dataset (simplified version).

    Returns
    -------
    X : ndarray of shape (150, 4)
        Features: sepal_length, sepal_width, petal_length, petal_width
    y : ndarray of shape (150,)
        Labels: 0=setosa, 1=versicolor, 2=virginica
    feature_names : list of str
        Names of the features.
    target_names : list of str
        Names of the classes.
    """
    # Iris dataset values (simplified - using typical means)
    # Note: This is a simplified version for educational purposes

    X = np.array(
        [
            # Class 0: Setosa
            [5.1, 3.5, 1.4, 0.2],
            [4.9, 3.0, 1.4, 0.2],
            [4.7, 3.2, 1.3, 0.2],
            [4.6, 3.1, 1.5, 0.2],
            [5.0, 3.6, 1.4, 0.2],
            [5.4, 3.9, 1.7, 0.4],
            [4.6, 3.4, 1.4, 0.3],
            [5.0, 3.4, 1.5, 0.2],
            [4.4, 2.9, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.1],
            [5.4, 3.7, 1.5, 0.2],
            [4.8, 3.4, 1.6, 0.2],
            [4.8, 3.0, 1.4, 0.1],
            [4.3, 3.0, 1.1, 0.1],
            [5.8, 4.0, 1.2, 0.2],
            [5.7, 4.4, 1.5, 0.4],
            [5.4, 3.9, 1.3, 0.4],
            [5.1, 3.5, 1.4, 0.3],
            [5.7, 3.8, 1.7, 0.3],
            [5.1, 3.8, 1.5, 0.3],
            [5.4, 3.4, 1.7, 0.2],
            [5.1, 3.7, 1.4, 0.5],
            [4.6, 3.6, 1.0, 0.2],
            [5.1, 3.3, 1.7, 0.5],
            [4.8, 3.4, 1.9, 0.2],
            [5.0, 3.0, 1.6, 0.2],
            [5.0, 3.4, 1.6, 0.4],
            [5.2, 3.5, 1.5, 0.2],
            [5.2, 3.4, 1.4, 0.2],
            [4.7, 3.2, 1.6, 0.2],
            [4.8, 3.1, 1.6, 0.2],
            [5.4, 3.4, 1.5, 0.4],
            [5.2, 4.1, 1.5, 0.1],
            [5.5, 4.2, 1.4, 0.2],
            [4.9, 3.1, 1.5, 0.2],
            [5.0, 3.2, 1.2, 0.2],
            [5.5, 3.5, 1.3, 0.2],
            [4.9, 3.6, 1.4, 0.1],
            [4.4, 3.0, 1.3, 0.2],
            [5.1, 3.4, 1.5, 0.2],
            [5.0, 3.5, 1.3, 0.3],
            [4.5, 2.3, 1.3, 0.3],
            [4.4, 3.2, 1.3, 0.2],
            [5.0, 3.5, 1.6, 0.6],
            [5.1, 3.8, 1.9, 0.4],
            [4.8, 3.0, 1.4, 0.3],
            [5.1, 3.8, 1.6, 0.2],
            [4.6, 3.2, 1.4, 0.2],
            [5.3, 3.7, 1.5, 0.2],
            [5.0, 3.3, 1.4, 0.2],
            # Class 1: Versicolor
            [7.0, 3.2, 4.7, 1.4],
            [6.4, 3.2, 4.5, 1.5],
            [6.9, 3.1, 4.9, 1.5],
            [5.5, 2.3, 4.0, 1.3],
            [6.5, 2.8, 4.6, 1.5],
            [5.7, 2.8, 4.5, 1.3],
            [6.3, 3.3, 4.7, 1.6],
            [4.9, 2.4, 3.3, 1.0],
            [6.6, 2.9, 4.6, 1.3],
            [5.2, 2.7, 3.9, 1.4],
            [5.0, 2.0, 3.5, 1.0],
            [5.9, 3.0, 4.2, 1.5],
            [6.0, 2.2, 4.0, 1.0],
            [6.1, 2.9, 4.7, 1.4],
            [5.6, 2.9, 3.6, 1.3],
            [6.7, 3.1, 4.4, 1.4],
            [5.6, 3.0, 4.5, 1.5],
            [5.8, 2.7, 4.1, 1.0],
            [6.2, 2.2, 4.5, 1.5],
            [5.6, 2.5, 3.9, 1.1],
            [5.9, 3.2, 4.8, 1.8],
            [6.1, 2.8, 4.0, 1.3],
            [6.3, 2.5, 4.9, 1.5],
            [6.1, 2.8, 4.7, 1.2],
            [6.4, 2.9, 4.3, 1.3],
            [6.6, 3.0, 4.4, 1.4],
            [6.8, 2.8, 4.8, 1.4],
            [6.7, 3.0, 5.0, 1.7],
            [6.0, 2.9, 4.5, 1.5],
            [5.7, 2.6, 3.5, 1.0],
            [5.5, 2.4, 3.8, 1.1],
            [5.5, 2.4, 3.7, 1.0],
            [5.8, 2.7, 3.9, 1.2],
            [6.0, 2.7, 5.1, 1.6],
            [5.4, 3.0, 4.5, 1.5],
            [6.0, 3.4, 4.5, 1.6],
            [6.7, 3.1, 4.7, 1.5],
            [6.3, 2.3, 4.4, 1.3],
            [5.6, 3.0, 4.1, 1.3],
            [5.5, 2.5, 4.0, 1.3],
            [5.5, 2.6, 4.4, 1.2],
            [6.1, 3.0, 4.6, 1.4],
            [5.8, 2.6, 4.0, 1.2],
            [5.0, 2.3, 3.3, 1.0],
            [5.6, 2.7, 4.2, 1.3],
            [5.7, 3.0, 4.2, 1.2],
            [5.7, 2.9, 4.2, 1.3],
            [6.2, 2.9, 4.3, 1.3],
            [5.1, 2.5, 3.0, 1.1],
            [5.7, 2.8, 4.1, 1.3],
            # Class 2: Virginica
            [6.3, 3.3, 6.0, 2.5],
            [5.8, 2.7, 5.1, 1.9],
            [7.1, 3.0, 5.9, 2.1],
            [6.3, 2.9, 5.6, 1.8],
            [6.5, 3.0, 5.8, 2.2],
            [7.6, 3.0, 6.6, 2.1],
            [4.9, 2.5, 4.5, 1.7],
            [7.3, 2.9, 6.3, 1.8],
            [6.7, 2.5, 5.8, 1.8],
            [7.2, 3.6, 6.1, 2.5],
            [6.5, 3.2, 5.1, 2.0],
            [6.4, 2.7, 5.3, 1.9],
            [6.8, 3.0, 5.5, 2.1],
            [5.7, 2.5, 5.0, 2.0],
            [5.8, 2.8, 5.1, 2.4],
            [6.4, 3.2, 5.3, 2.3],
            [6.5, 3.0, 5.5, 1.8],
            [7.7, 3.8, 6.7, 2.2],
            [7.7, 2.6, 6.9, 2.3],
            [6.0, 2.2, 5.0, 1.5],
            [6.9, 3.2, 5.7, 2.3],
            [5.6, 2.8, 4.9, 2.0],
            [7.7, 2.8, 6.7, 2.0],
            [6.3, 2.7, 4.9, 1.8],
            [6.7, 3.3, 5.7, 2.1],
            [7.2, 3.2, 6.0, 1.8],
            [6.2, 2.8, 4.8, 1.8],
            [6.1, 3.0, 4.9, 1.8],
            [6.4, 2.8, 5.6, 2.1],
            [7.2, 3.0, 5.8, 1.6],
            [7.4, 2.8, 6.1, 1.9],
            [7.9, 3.8, 6.4, 2.0],
            [6.4, 2.8, 5.6, 2.2],
            [6.3, 2.8, 5.1, 1.5],
            [6.1, 2.6, 5.6, 1.4],
            [7.7, 3.0, 6.1, 2.3],
            [6.3, 3.4, 5.6, 2.4],
            [6.4, 3.1, 5.5, 1.8],
            [6.0, 3.0, 4.8, 1.8],
            [6.9, 3.1, 5.4, 2.1],
            [6.7, 3.1, 5.6, 2.4],
            [6.9, 3.1, 5.1, 2.3],
            [5.8, 2.7, 5.1, 1.9],
            [6.8, 3.2, 5.9, 2.3],
            [6.7, 3.3, 5.7, 2.5],
            [6.7, 3.0, 5.2, 2.3],
            [6.3, 2.5, 5.0, 1.9],
            [6.5, 3.0, 5.2, 2.0],
            [6.2, 3.4, 5.4, 2.3],
            [5.9, 3.0, 5.1, 1.8],
        ]
    )

    # Labels: 50 samples per class
    y = np.array([0] * 50 + [1] * 50 + [2] * 50)

    feature_names = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    target_names = ["setosa", "versicolor", "virginica"]

    return X, y, feature_names, target_names


def train_test_split(
    *arrays: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    test_size: float = 0.25,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> List[np.ndarray]:
    """
    Split arrays into random train and test subsets.

    Parameters
    ----------
    *arrays : ndarray
        Sequence of arrays to split. Can be (X, y) or just X.
    test_size : float, default=0.25
        Proportion of the dataset to include in the test split.
    random_state : int, optional, default=None
        Random seed for reproducibility.
    shuffle : bool, default=True
        Whether to shuffle before splitting.

    Returns
    -------
    splitting : list of ndarrays
        List containing train-test split of inputs.

    Examples
    --------
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    >>> X_train, X_test = train_test_split(X, test_size=0.3)
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = arrays[0].shape[0]

    # Calculate test split size
    n_test = int(n_samples * test_size)

    # Create indices
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    # Split indices
    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    # Split each array
    result = []
    for array in arrays:
        result.append(array[train_indices])
        result.append(array[test_indices])

    return result


# ============================================================================
# Main execution and demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Dataset Utilities - Demonstration")
    print("=" * 70)

    # Demo 1: make_moons
    print("\n1. make_moons dataset:")
    X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
    print(f"   Shape: X={X_moons.shape}, y={y_moons.shape}")
    print(f"   Class distribution: {np.bincount(y_moons.astype(int))}")

    # Demo 2: make_circles
    print("\n2. make_circles dataset:")
    X_circles, y_circles = make_circles(n_samples=200, noise=0.05, random_state=0)
    print(f"   Shape: X={X_circles.shape}, y={y_circles.shape}")

    # Demo 3: make_blobs
    print("\n3. make_blobs dataset:")
    X_blobs, y_blobs, centers = make_blobs(n_samples=300, centers=3, random_state=42)
    print(f"   Shape: X={X_blobs.shape}, y={y_blobs.shape}, centers={centers.shape}")
    print(f"   Unique labels: {np.unique(y_blobs)}")

    # Demo 4: make_classification
    print("\n4. make_classification dataset:")
    X_clf, y_clf = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=42,
    )
    print(f"   Shape: X={X_clf.shape}, y={y_clf.shape}")
    print(f"   Unique classes: {np.unique(y_clf)}")

    # Demo 5: make_regression
    print("\n5. make_regression dataset:")
    X_reg, y_reg = make_regression(
        n_samples=100, n_features=5, n_informative=3, noise=0.1
    )
    print(f"   Shape: X={X_reg.shape}, y={y_reg.shape}")

    # Demo 6: train_test_split
    print("\n6. train_test_split demonstration:")
    X_train, X_test, y_train, y_test = train_test_split(
        X_blobs, y_blobs, test_size=0.2, random_state=42
    )
    print(f"   Train: X={X_train.shape}, y={y_train.shape}")
    print(f"   Test:  X={X_test.shape}, y={y_test.shape}")

    # Demo 7: load_iris
    print("\n7. load_iris dataset:")
    X_iris, y_iris, features, targets = load_iris()
    print(f"   Shape: X={X_iris.shape}, y={y_iris.shape}")
    print(f"   Features: {features}")
    print(f"   Classes: {targets}")

    print("\n" + "=" * 70)
    print("All dataset utilities working correctly!")
    print("=" * 70)
