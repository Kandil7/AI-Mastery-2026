"""
Practical Applications - Examples
==================================

Real-world examples for classification, regression, clustering.

Author: AI-Mastery-2026
"""

import numpy as np
import matplotlib.pyplot as plt


def example_classification_comparison():
    """Example 1: Compare Classification Algorithms"""
    print("=" * 60)
    print("Example 1: Classification Algorithm Comparison")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n_samples = 200

    # Generate 3-class data with different patterns
    # Class 0: Linear boundary
    X0 = np.random.randn(n_samples // 3, 2) + np.array([-2, -2])
    y0 = np.zeros(n_samples // 3)

    # Class 1: Another linear region
    X1 = np.random.randn(n_samples // 3, 2) + np.array([2, 2])
    y1 = np.ones(n_samples // 3)

    # Class 2: Circular region
    angle = np.random.uniform(0, 2 * np.pi, n_samples // 3)
    r = np.random.uniform(0, 2, n_samples // 3)
    X2 = np.column_stack([r * np.cos(angle), r * np.sin(angle)])
    y2 = np.full(n_samples // 3, 2)

    X = np.vstack([X0, X1, X2])
    y = np.hstack([y0, y1, y2])

    print(f"Dataset: {len(y)} samples, 3 classes")
    print(f"  Class 0: {sum(y == 0)} samples (linear region)")
    print(f"  Class 1: {sum(y == 1)} samples (linear region)")
    print(f"  Class 2: {sum(y == 2)} samples (circular region)")

    # Note: In practice, would train actual classifiers here
    print("\nAlgorithm Suitability:")
    print("-" * 40)
    print("Logistic Regression: Good for classes 0 & 1 (linear)")
    print("Decision Tree: Can handle mixed boundaries")
    print("Random Forest: Best for circular regions + linear")

    # Visualize
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["blue", "red", "green"]
    markers = ["o", "s", "^"]

    for cls in range(3):
        mask = y == cls
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            c=colors[cls],
            marker=markers[cls],
            label=f"Class {cls}",
            alpha=0.6,
        )

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("3-Class Classification Dataset")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def example_regression_comparison():
    """Example 2: Regression Algorithm Comparison"""
    print("\n" + "=" * 60)
    print("Example 2: Regression Algorithm Comparison")
    print("=" * 60)

    # Generate data with different patterns
    np.random.seed(42)
    X = np.linspace(0, 10, 100)

    # Pattern 1: Linear with noise
    y_linear = 2 * X + 1 + np.random.randn(100) * 0.5

    # Pattern 2: Quadratic
    y_quad = 0.3 * X**2 - 2 * X + 5 + np.random.randn(100) * 0.5

    # Pattern 3: Sinusoidal
    y_sine = 3 * np.sin(X) + 5 + np.random.randn(100) * 0.5

    print("Pattern Analysis:")
    print("-" * 40)
    print(f"Linear: y = 2x + 1 (use LinearRegression)")
    print(f"Quadratic: y = 0.3x² - 2x + 5 (use Polynomial)")
    print(f"Sinusoidal: y = 3sin(x) + 5 (use Gradient Boosting)")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(X, y_linear, alpha=0.5)
    axes[0].plot(X, 2 * X + 1, "r-", linewidth=2, label="True")
    axes[0].set_title("Linear Pattern")
    axes[0].legend()

    axes[1].scatter(X, y_quad, alpha=0.5)
    axes[1].plot(X, 0.3 * X**2 - 2 * X + 5, "r-", linewidth=2, label="True")
    axes[1].set_title("Quadratic Pattern")
    axes[1].legend()

    axes[2].scatter(X, y_sine, alpha=0.5)
    axes[2].plot(X, 3 * np.sin(X) + 5, "r-", linewidth=2, label="True")
    axes[2].set_title("Sinusoidal Pattern")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def example_clustering_comparison():
    """Example 3: Clustering Algorithm Comparison"""
    print("\n" + "=" * 60)
    print("Example 3: Clustering Algorithm Comparison")
    print("=" * 60)

    # Different cluster shapes
    np.random.seed(42)

    # Data 1: Well-separated spherical clusters
    X1 = np.vstack([np.random.randn(50, 2) + [-3, -3], np.random.randn(50, 2) + [3, 3]])

    # Data 2: Concentric circles
    angle = np.linspace(0, 2 * np.pi, 100)
    inner = np.column_stack([np.cos(angle) * 0.5, np.sin(angle) * 0.5])
    outer = np.column_stack([np.cos(angle) * 2, np.sin(angle) * 2])
    X2 = np.vstack([inner, outer])

    # Data 3: Irregular shapes
    X3 = np.vstack(
        [
            np.random.randn(30, 2) + [-2, -2],
            np.random.randn(30, 2) + [-2, 2],
            np.random.randn(40, 2) + [3, 0],
        ]
    )

    print("Cluster Types and Best Algorithms:")
    print("-" * 40)
    print("Data 1 (Spherical): K-Means ✓, DBSCAN ✓")
    print("Data 2 (Concentric): K-Means ✗, DBSCAN ✓")
    print("Data 3 (Irregular): K-Means ~, DBSCAN ✓")

    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].scatter(X1[:, 0], X1[:, 1], alpha=0.6)
    axes[0].set_title("Spherical Clusters\n(K-Means works)")

    axes[1].scatter(X2[:, 0], X2[:, 1], alpha=0.6)
    axes[1].set_title("Concentric Circles\n(K-Means fails)")

    axes[2].scatter(X3[:, 0], X3[:, 1], alpha=0.6)
    axes[2].set_title("Irregular Shapes\n(DBSCAN works)")

    plt.tight_layout()
    plt.show()


def example_pca_visualization():
    """Example 4: PCA for Visualization"""
    print("\n" + "=" * 60)
    print("Example 4: PCA for Dimensionality Reduction")
    print("=" * 60)

    # High-dimensional data
    np.random.seed(42)
    n_samples = 100

    # Create 4D data with some correlation
    # x1, x2 highly correlated
    # x3, x4 independent
    x1 = np.random.randn(n_samples)
    x2 = 2 * x1 + 0.5 * np.random.randn(n_samples)  # correlated with x1
    x3 = np.random.randn(n_samples)
    x4 = np.random.randn(n_samples)

    X = np.column_stack([x1, x2, x3, x4])

    print(f"Original dimensions: {X.shape[1]}")
    print("Variance in each feature:")
    for i in range(4):
        print(f"  Feature {i + 1}: {np.var(X[:, i]):.2f}")

    # Compute PCA
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    cov = np.cov(X_centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    eigenvalues = eigenvalues[::-1]

    print(f"\nEigenvalues: {eigenvalues[::-1]}")
    print(f"Explained variance ratio:")
    total = sum(eigenvalues)
    for i, ev in enumerate(eigenvalues[::-1]):
        print(f"  PC{i + 1}: {ev / total * 100:.1f}%")

    # Project to 2D
    top_2_eigenvectors = eigenvectors[:, -2:]
    X_pca = X_centered @ top_2_eigenvectors

    print(f"\nReduced to 2D: {X_pca.shape}")
    print("This captures 2 most important directions!")

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].scatter(X[:, 0], X[:, 1], alpha=0.6)
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].set_title("Original Data (2D projection)")

    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].set_title("After PCA (2D)")

    plt.tight_layout()
    plt.show()


def example_metrics_selection():
    """Example 5: Choosing the Right Metric"""
    print("\n" + "=" * 60)
    print("Example 5: Metric Selection Guide")
    print("=" * 60)

    print("""
    When to Use Each Metric:
    =========================
    
    BALANCED CLASSES:
    - Accuracy: Simple baseline
    - F1-Score: Good general purpose
    
    IMBALANCED CLASSES:
    - Precision: When false positives are costly
      Example: Spam detection (don't want to miss real email)
    - Recall: When false negatives are costly
      Example: Disease detection (don't want to miss sick patients)
    - ROC-AUC: When you need threshold-independent measure
    
    REGRESSION:
    - R²: General purpose, interpretable
    - RMSE: When large errors are particularly bad
    - MAE: When outliers are just noise
    
    Time Series:
    - MAE: More robust to anomalies
    - RMSE: Penalizes late predictions more
    """)


def run_all_examples():
    """Run all examples"""
    example_classification_comparison()
    example_regression_comparison()
    example_clustering_comparison()
    example_pca_visualization()
    example_metrics_selection()


if __name__ == "__main__":
    run_all_examples()

    print("\n" + "=" * 60)
    print("All practical examples completed!")
    print("=" * 60)
