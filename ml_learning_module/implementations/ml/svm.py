"""
Support Vector Machine (SVM) Implementation
============================================

Complete SVM implementation with linear, polynomial, and RBF kernels.
Based on "Hands-On Machine Learning" concepts.

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Optional, Tuple


class SVM:
    """
    Support Vector Machine Classifier

    SVM finds the optimal hyperplane that maximizes the margin
    between classes.

    Mathematical Foundation:
    =======================

    For linearly separable data:
        minimize: (1/2)||w||²
        subject to: y_i(w·x_i + b) ≥ 1

    For soft margin (with regularization C):
        minimize: (1/2)||w||² + C * Σ ξ_i
        subject to: y_i(w·x_i + b) ≥ 1 - ξ_i

    Key concepts:
    - Support vectors: Points on or beyond the margin
    - Margin: Distance between parallel hyperplanes
    - Kernel trick: Transform to higher dimension for non-linear boundaries

    Kernel Functions:
    - Linear: K(a, b) = a·b
    - Polynomial: K(a, b) = (γ·a·b + r)^d
    - RBF (Gaussian): K(a, b) = exp(-γ||a-b||²)
    """

    def __init__(
        self,
        C: float = 1.0,
        kernel: str = "rbf",
        gamma: float = "scale",
        degree: int = 3,
        coef0: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ):
        """
        Initialize SVM.

        Args:
            C: Regularization parameter (inverse of margin width)
            kernel: 'linear', 'poly', or 'rbf'
            gamma: Kernel coefficient for 'rbf' and 'poly'
            degree: Degree for polynomial kernel
            coef0: Independent term for polynomial kernel
            max_iter: Maximum iterations for SMO
            tol: Tolerance for stopping criteria
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.max_iter = max_iter
        self.tol = tol

        # Model parameters (set during fit)
        self.support_vectors_ = None
        self.support_labels_ = None
        self.alphas_ = None
        self.bias_ = None
        self.n_support_ = None

    def _compute_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute kernel matrix between X1 and X2"""

        if self.kernel == "linear":
            return X1 @ X2.T

        elif self.kernel == "poly":
            gamma = self._get_gamma(X1)
            return (gamma * (X1 @ X2.T) + self.coef0) ** self.degree

        elif self.kernel == "rbf":
            gamma = self._get_gamma(X1)
            # ||a-b||² = ||a||² + ||b||² - 2a·b
            X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
            sq_dists = X1_sq + X2_sq - 2 * (X1 @ X2.T)
            return np.exp(-gamma * sq_dists)

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _get_gamma(self, X: np.ndarray) -> float:
        """Get gamma value"""
        if self.gamma == "scale":
            return 1.0 / (X.shape[1] * X.var())
        elif self.gamma == "auto":
            return 1.0 / X.shape[1]
        return float(self.gamma)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        """
        Fit SVM using Simplified SMO (Sequential Minimal Optimization)

        SMO breaks the large QP problem into smaller 2-variable problems.

        Args:
            X: Training features (n_samples, n_features)
            y: Training labels (-1 or 1)
        """
        n_samples, n_features = X.shape

        # Convert labels to -1, 1 if needed
        if len(np.unique(y)) == 2:
            y = np.where(y == 0, -1, 1)

        # Initialize
        self.alphas = np.zeros(n_samples)
        self.bias = 0.0

        # Precompute kernel matrix
        self._kernel_matrix = self._compute_kernel(X, X)
        self._X = X
        self._y = y

        # SMO optimization
        for iteration in range(self.max_iter):
            num_changed = 0

            for i in range(n_samples):
                # Select i randomly for stochastic SMO
                j = np.random.randint(0, n_samples)
                if j == i:
                    continue

                # Compute errors
                E_i = self._compute_error(i)
                E_j = self._compute_error(j)

                # Check if alpha_i can be optimized
                if self._check_KKT(i, E_i):
                    continue

                # Compute bounds
                L, H = self._compute_bounds(i, j, y[i], y[j])

                if L == H:
                    continue

                # Compute optimal alpha_j
                eta = (
                    2 * self._kernel_matrix[i, j]
                    - self._kernel_matrix[i, i]
                    - self._kernel_matrix[j, j]
                )

                if eta >= 0:
                    continue

                alpha_j_old = self.alphas[j]
                self.alphas[j] -= (y[j] * (E_i - E_j)) / eta

                # Clip alpha_j to bounds
                self.alphas[j] = np.clip(self.alphas[j], L, H)

                # Check for significant change
                if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                    continue

                # Update alpha_i
                self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])

                # Update bias
                b1 = (
                    self.bias
                    - E_i
                    - y[i] * (self.alphas[i] - alpha_j_old) * self._kernel_matrix[i, i]
                    - y[j] * (self.alphas[j] - alpha_j_old) * self._kernel_matrix[i, j]
                )

                b2 = (
                    self.bias
                    - E_j
                    - y[i] * (self.alphas[i] - alpha_j_old) * self._kernel_matrix[i, j]
                    - y[j] * (self.alphas[j] - alpha_j_old) * self._kernel_matrix[j, j]
                )

                if 0 < self.alphas[i] < self.C:
                    self.bias = b1
                elif 0 < self.alphas[j] < self.C:
                    self.bias = b2
                else:
                    self.bias = (b1 + b2) / 2

                num_changed += 1

            # Check convergence
            if num_changed == 0:
                break

        # Store support vectors
        support_idx = self.alphas > 1e-7
        self.support_vectors_ = X[support_idx]
        self.support_labels_ = y[support_idx]
        self.alphas_ = self.alphas[support_idx]
        self.n_support_ = np.sum(support_idx)

        return self

    def _compute_error(self, i: int) -> float:
        """Compute prediction error for sample i"""
        return (
            np.sum(self.alphas * self._y * self._kernel_matrix[:, i])
            + self.bias
            - self._y[i]
        )

    def _check_KKT(self, i: int, E_i: float) -> bool:
        """Check KKT conditions for sample i"""
        r = E_i * self._y[i]

        if self.alphas[i] == 0:
            return r < -self.tol  # Should be at lower bound
        elif self.alphas[i] == self.C:
            return r > self.tol  # Should be at upper bound
        else:
            return abs(r) > self.tol  # Should be on margin

    def _compute_bounds(
        self, i: int, j: int, y_i: int, y_j: int
    ) -> Tuple[float, float]:
        """Compute bounds for alpha_j"""
        if y_i != y_j:
            L = max(0, self.alphas[j] - self.alphas[i])
            H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
        else:
            L = max(0, self.alphas[i] + self.alphas[j] - self.C)
            H = min(self.C, self.alphas[i] + self.alphas[j])
        return L, H

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for samples in X.

        Args:
            X: Samples (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,)
        """
        if self.support_vectors_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        # Compute kernel between test and support vectors
        K = self._compute_kernel(X, self.support_vectors_)

        # Compute decision function
        decision = np.sum(K * self.alphas_ * self.support_labels_, axis=1) + self.bias

        # Return predictions (-1 or 1)
        return np.sign(decision)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function for X"""
        if self.support_vectors_ is None:
            raise ValueError("Model not fitted. Call fit() first.")

        K = self._compute_kernel(X, self.support_vectors_)
        return np.sum(K * self.alphas_ * self.support_labels_, axis=1) + self.bias

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy"""
        predictions = self.predict(X)
        y_converted = np.where(y == 0, -1, y)
        return np.mean(predictions == y_converted)


class LinearSVC:
    """
    Linear Support Vector Classifier (SVM with linear kernel)

    More efficient than kernel SVM for large datasets.

    Uses coordinate descent optimization.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000, tol: float = 1e-4):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol
        self.weights = None
        self.bias = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearSVC":
        """Fit LinearSVC"""
        n_samples, n_features = X.shape

        # Convert labels to -1, 1
        y_converted = np.where(y == 0, -1, 1)

        # Initialize
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Coordinate descent
        for _ in range(self.max_iter):
            for i in range(n_samples):
                # Compute margin
                margin = y_converted[i] * (self.weights @ X[i] + self.bias)

                if margin < 1:
                    # Misclassified or within margin
                    self.weights -= self.tol * (
                        self.weights - self.C * y_converted[i] * X[i]
                    )
                    self.bias -= self.tol * (-self.C * y_converted[i])
                else:
                    # Correctly classified
                    self.weights -= self.tol * self.weights

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels"""
        scores = X @ self.weights + self.bias
        return np.where(scores >= 0, 1, 0)


def demo_svm():
    """Demonstrate SVM with different kernels"""
    print("=" * 60)
    print("SVM Demo - Based on Hands-On ML")
    print("=" * 60)

    # Generate sample data
    np.random.seed(42)

    # Linear separable
    X_linear = np.vstack(
        [np.random.randn(50, 2) + [-2, -2], np.random.randn(50, 2) + [2, 2]]
    )
    y_linear = np.array([0] * 50 + [1] * 50)

    # Non-linear (moons-like)
    n = 50
    X_moon1 = np.column_stack(
        [
            np.cos(np.linspace(0, np.pi, n)) + np.random.randn(n) * 0.1,
            np.sin(np.linspace(0, np.pi, n)) + np.random.randn(n) * 0.1 - 0.5,
        ]
    )
    X_moon2 = np.column_stack(
        [
            1 - np.cos(np.linspace(0, np.pi, n)) + np.random.randn(n) * 0.1,
            0.5 - np.sin(np.linspace(0, np.pi, n)) + np.random.randn(n) * 0.1,
        ]
    )
    X_moon = np.vstack([X_moon1, X_moon2])
    y_moon = np.array([0] * n + [1] * n)

    print("\n--- Linear Data (Linear Kernel) ---")
    svm_linear = SVM(kernel="linear", C=1.0)
    svm_linear.fit(X_linear, y_linear)
    acc = svm_linear.score(X_linear, y_linear)
    print(f"Linear SVM accuracy: {acc:.2%}")
    print(f"Support vectors: {svm_linear.n_support_}")

    print("\n--- Non-linear Data (RBF Kernel) ---")
    svm_rbf = SVM(kernel="rbf", C=1.0, gamma=0.5)
    svm_rbf.fit(X_moon, y_moon)
    acc = svm_rbf.score(X_moon, y_moon)
    print(f"RBF SVM accuracy: {acc:.2%}")
    print(f"Support vectors: {svm_rbf.n_support_}")

    print("\n--- Same data with Polynomial Kernel ---")
    svm_poly = SVM(kernel="poly", degree=3, C=1.0)
    svm_poly.fit(X_moon, y_moon)
    acc = svm_poly.score(X_moon, y_moon)
    print(f"Polynomial SVM accuracy: {acc:.2%}")


if __name__ == "__main__":
    demo_svm()
    print("\nSVM implementation complete!")
