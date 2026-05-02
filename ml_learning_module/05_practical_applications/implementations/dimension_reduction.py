"""
Dimensionality Reduction Implementation
========================================

Principal Component Analysis (PCA) and t-SNE implementations.

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PCAResult:
    """Store PCA results"""

    transformed: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    mean: np.ndarray


class PCA:
    """
    Principal Component Analysis (PCA)

    Finds orthogonal directions (principal components) that maximize
    variance in the data.

    Mathematical Foundation:
    =======================

    1. Center the data:
       X_centered = X - mean(X)

    2. Compute covariance matrix:
       S = (1/n) * X_centered^T @ X_centered

    3. Find eigenvectors and eigenvalues:
       S @ v = λ @ v

    4. Sort by eigenvalue (largest first) - these are principal components

    5. Project data onto top k components:
       X_projected = X_centered @ V_k

    Why PCA Works:
    - First PC: Direction of maximum variance
    - Second PC: Orthogonal to first, next maximum variance
    - And so on...

    Eigenvalue = variance in that direction
    Explained variance ratio = eigenvalue / sum(eigenvalues)
    """

    def __init__(self, n_components: Optional[int] = None):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X: np.ndarray) -> "PCA":
        """
        Fit PCA to data

        Args:
            X: Input data, shape (n_samples, n_features)
        """
        n_samples, n_features = X.shape

        # Determine number of components
        if self.n_components is None:
            self.n_components = n_features

        # 1. Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 2. Compute covariance matrix
        # S = (1/(n-1)) * X^T @ X (using n-1 for unbiased estimator)
        covariance = (X_centered.T @ X_centered) / (n_samples - 1)

        # 3. Find eigenvectors and eigenvalues
        # Use numpy's eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # 4. Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 5. Store components
        self.components = eigenvectors[:, : self.n_components]
        self.explained_variance = eigenvalues[: self.n_components]

        # 6. Compute explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = self.explained_variance / total_variance

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data to principal component space"""
        X_centered = X - self.mean
        return X_centered @ self.components

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step"""
        return self.fit(X).transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Transform back to original space"""
        return X_transformed @ self.components.T + self.mean

    def get_cumulative_variance(self, n_components: int) -> float:
        """Get cumulative explained variance for n components"""
        return np.sum(self.explained_variance_ratio[:n_components])


class KernelPCA:
    """
    Kernel PCA

    Extends PCA using kernel trick to capture non-linear patterns.

    Kernels:
    - Linear: K(x, y) = x^T y
    - Polynomial: K(x, y) = (γ * x^T y + r)^d
    - RBF (Gaussian): K(x, y) = exp(-γ ||x - y||²)
    """

    def __init__(
        self,
        n_components: int = 2,
        kernel: str = "rbf",
        gamma: float = 1.0,
        degree: int = 3,
        coef0: float = 1.0,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0

        self.alphas = None
        self.lambdas = None
        self.X_train = None

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix"""
        if self.kernel == "linear":
            return X1 @ X2.T

        elif self.kernel == "poly":
            return (self.gamma * (X1 @ X2.T) + self.coef0) ** self.degree

        elif self.kernel == "rbf":
            # RBF kernel: exp(-gamma * ||x - y||^2)
            # Compute pairwise squared distances
            X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
            sq_dists = X1_sq + X2_sq - 2 * (X1 @ X2.T)
            return np.exp(-self.gamma * sq_dists)

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def fit(self, X: np.ndarray) -> "KernelPCA":
        """Fit Kernel PCA"""
        n_samples = X.shape[0]
        self.X_train = X

        # Compute kernel matrix
        K = self._compute_kernel(X, X)

        # Center kernel matrix (approximate)
        one_n = np.ones((n_samples, n_samples)) / n_samples
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(K_centered)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Keep top components
        self.lambdas = eigenvalues[: self.n_components]
        self.alphas = eigenvectors[:, : self.n_components]

        # Normalize alphas
        for i in range(self.n_components):
            if self.lambdas[i] > 0:
                self.alphas[:, i] /= np.sqrt(self.lambdas[i])

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using kernel PCA"""
        # Compute kernel between new and training data
        K_test = self._compute_kernel(X, self.X_train)

        # Project onto principal components
        return K_test @ self.alphas


def demo_pca():
    """Demonstrate PCA"""
    print("=" * 60)
    print("PCA Demo")
    print("=" * 60)

    # Create data with strong correlation
    np.random.seed(42)
    n_samples = 500

    # Create 4D data with strong correlations
    # x1, x2 are strongly correlated
    # x3, x4 are also correlated but separate
    x1 = np.random.randn(n_samples)
    x2 = 2 * x1 + 0.5 * np.random.randn(n_samples)  # correlated with x1
    x3 = np.random.randn(n_samples)
    x4 = -0.5 * x3 + 0.3 * np.random.randn(n_samples)  # correlated with x3

    X = np.column_stack([x1, x2, x3, x4])

    print(f"Original data shape: {X.shape}")
    print(f"Original variance per feature: {np.var(X, axis=0)}")

    # Apply PCA
    pca = PCA(n_components=2)
    X_transformed = pca.fit_transform(X)

    print(f"\nTransformed shape: {X_transformed.shape}")
    print(f"\nExplained variance ratio:")
    for i, ratio in enumerate(pca.explained_variance_ratio):
        print(f"  PC{i + 1}: {ratio:.4f} ({ratio * 100:.2f}%)")

    print(f"\nCumulative variance (2 components): {pca.get_cumulative_variance(2):.4f}")

    # Show components
    print(f"\nPrincipal Components (eigenvectors):")
    for i, comp in enumerate(pca.components.T):
        print(f"  PC{i + 1}: {comp}")

    # Visualize
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Original data
        axes[0].scatter(X[:, 0], X[:, 1], alpha=0.5)
        axes[0].set_xlabel("Feature 1")
        axes[0].set_ylabel("Feature 2")
        axes[0].set_title("Original Data (2D projection)")

        # Transformed data
        axes[1].scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.5)
        axes[1].set_xlabel("PC1")
        axes[1].set_ylabel("PC2")
        axes[1].set_title("PCA Transformed")

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available")


def demo_kernel_pca():
    """Demonstrate Kernel PCA"""
    print("\n" + "=" * 60)
    print("Kernel PCA Demo")
    print("=" * 60)

    # Create non-linear data (circles)
    np.random.seed(42)
    n_samples = 200

    # Inner circle
    angle = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    r_inner = np.random.uniform(0, 1, n_samples // 2)
    inner = np.column_stack([r_inner * np.cos(angle), r_inner * np.sin(angle)])

    # Outer circle
    angle = np.random.uniform(0, 2 * np.pi, n_samples // 2)
    r_outer = np.random.uniform(2, 3, n_samples // 2)
    outer = np.column_stack([r_outer * np.cos(angle), r_outer * np.sin(angle)])

    X = np.vstack([inner, outer])
    labels = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Standard PCA won't separate well
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    # Kernel PCA with RBF
    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.5)
    X_kpca = kpca.fit_transform(X)

    print("Linear PCA might not separate concentric circles")
    print("Kernel PCA with RBF can do non-linear separation")


if __name__ == "__main__":
    demo_pca()
    demo_kernel_pca()

    print("\n" + "=" * 60)
    print("Dimensionality Reduction complete!")
    print("=" * 60)
