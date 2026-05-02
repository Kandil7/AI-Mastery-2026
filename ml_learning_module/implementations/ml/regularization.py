"""
Regularization Utilities
========================

L1, L2, and Elastic Net regularization implementations
for various ML models.

Author: ML Learning Module
"""

import numpy as np
from typing import Optional, Callable


class L1Regularizer:
    """
    L1 Regularization (Lasso)

    Adds sum of absolute values of coefficients to the loss:
    R(θ) = λ * Σ|θⱼ|

    Effect: Creates sparse solutions (some coefficients become exactly zero)
    """

    def __init__(self, alpha: float = 1.0):
        """
        Parameters
        ----------
        alpha : float, default=1.0
            Regularization strength (λ)
        """
        self.alpha = alpha

    def __call__(self, weights: np.ndarray) -> float:
        """Compute L1 penalty."""
        return self.alpha * np.sum(np.abs(weights))

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Compute subgradient of L1 penalty."""
        # Subgradient: sign(weights) except at 0
        return self.alpha * np.sign(weights)

    def prox(self, weights: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Proximal operator (soft-thresholding) for L1.

        This is used in proximal gradient descent (like ISTA).
        """
        return np.sign(weights) * np.maximum(
            np.abs(weights) - self.alpha * learning_rate, 0
        )


class L2Regularizer:
    """
    L2 Regularization (Ridge)

    Adds sum of squared coefficients to the loss:
    R(θ) = (λ/2) * Σθⱼ²

    Effect: Shrinks coefficients toward zero but rarely exactly zero
    """

    def __init__(self, alpha: float = 1.0):
        """
        Parameters
        ----------
        alpha : float, default=1.0
            Regularization strength (λ)
        """
        self.alpha = alpha

    def __call__(self, weights: np.ndarray) -> float:
        """Compute L2 penalty."""
        return 0.5 * self.alpha * np.sum(weights**2)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Compute gradient of L2 penalty."""
        return self.alpha * weights


class ElasticNetRegularizer:
    """
    Elastic Net Regularization

    Combines L1 and L2 regularization:
    R(θ) = λ * (ρ * Σ|θⱼ| + (1-ρ)/2 * Σθⱼ²)

    Effect: Balance between feature selection (L1) and stability (L2)
    """

    def __init__(self, alpha: float = 1.0, l1_ratio: float = 0.5):
        """
        Parameters
        ----------
        alpha : float, default=1.0
            Overall regularization strength (λ)
        l1_ratio : float, default=0.5
            Mix ratio between L1 and L2 (ρ)
            - 0 = pure L2
            - 1 = pure L1
        """
        self.alpha = alpha
        self.l1_ratio = l1_ratio

    def __call__(self, weights: np.ndarray) -> float:
        """Compute Elastic Net penalty."""
        l1_part = self.l1_ratio * np.sum(np.abs(weights))
        l2_part = 0.5 * (1 - self.l1_ratio) * np.sum(weights**2)
        return self.alpha * (l1_part + l2_part)

    def gradient(self, weights: np.ndarray) -> np.ndarray:
        """Compute gradient of Elastic Net penalty."""
        l1_grad = self.l1_ratio * np.sign(weights)
        l2_grad = (1 - self.l1_ratio) * weights
        return self.alpha * (l1_grad + l2_grad)


class RegularizedLinearRegression:
    """
    Linear Regression with Regularization

    Supports L1 (Lasso), L2 (Ridge), and Elastic Net regularization.
    """

    def __init__(
        self,
        regularization: str = "l2",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        fit_intercept: bool = True,
        max_iter: int = 1000,
        tol: float = 1e-4,
        learning_rate: float = 0.01,
    ):
        """
        Parameters
        ----------
        regularization : str, default='l2'
            Type of regularization: 'l1', 'l2', 'elastic', or 'none'
        alpha : float, default=1.0
            Regularization strength
        l1_ratio : float, default=0.5
            Mix ratio for Elastic Net (only used if regularization='elastic')
        fit_intercept : bool, default=True
            Whether to fit intercept
        max_iter : int, default=1000
            Maximum iterations for coordinate descent
        tol : float, default=1e-4
            Convergence tolerance
        learning_rate : float, default=0.01
            Learning rate for gradient descent (used for L2)
        """
        self.regularization = regularization
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate

        self.coef_ = None
        self.intercept_ = None

        # Initialize regularizer
        if regularization == "l1":
            self._regularizer = L1Regularizer(alpha)
        elif regularization == "l2":
            self._regularizer = L2Regularizer(alpha)
        elif regularization == "elastic":
            self._regularizer = ElasticNetRegularizer(alpha, l1_ratio)
        else:
            self._regularizer = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegularizedLinearRegression":
        """Fit the model."""
        n_samples, n_features = X.shape

        if self.regularization == "l2":
            return self._fit_gradient_descent(X, y)
        else:
            return self._fit_coordinate_descent(X, y)

    def _fit_gradient_descent(
        self, X: np.ndarray, y: np.ndarray
    ) -> "RegularizedLinearRegression":
        """Fit using gradient descent (for L2)."""
        n_samples, n_features = X.shape

        # Standardize features
        self._X_mean = np.mean(X, axis=0)
        self._X_std = np.std(X, axis=0)
        self._X_std[self._X_std == 0] = 1
        X_scaled = (X - self._X_mean) / self._X_std

        y_mean = np.mean(y)
        y_scaled = y - y_mean

        # Initialize
        weights = np.zeros(n_features)

        for _ in range(self.max_iter):
            weights_old = weights.copy()

            # Compute gradient
            predictions = X_scaled @ weights
            residual = predictions - y_scaled

            # Gradient of MSE
            grad_mse = (1 / n_samples) * X_scaled.T @ residual

            # Gradient of regularizer
            if self._regularizer:
                grad_reg = self._regularizer.gradient(weights)
            else:
                grad_reg = 0

            # Update
            weights -= self.learning_rate * (grad_mse + grad_reg)

            # Check convergence
            if np.max(np.abs(weights - weights_old)) < self.tol:
                break

        # Transform back to original scale
        self.coef_ = weights / self._X_std
        self.intercept_ = y_mean - self._X_mean @ self.coef_

        return self

    def _fit_coordinate_descent(
        self, X: np.ndarray, y: np.ndarray
    ) -> "RegularizedLinearRegression":
        """Fit using coordinate descent (for L1 and Elastic Net)."""
        n_samples, n_features = X.shape

        # Standardize
        self._X_mean = np.mean(X, axis=0)
        self._X_std = np.std(X, axis=0)
        self._X_std[self._X_std == 0] = 1
        X_scaled = (X - self._X_mean) / self._X_std

        y_mean = np.mean(y)
        y_scaled = y - y_mean

        weights = np.zeros(n_features)

        for _ in range(self.max_iter):
            weights_old = weights.copy()

            for j in range(n_features):
                # Compute residual excluding feature j
                residual = y_scaled - X_scaled @ weights + X_scaled[:, j] * weights[j]

                # Correlation with feature j
                rho_j = X_scaled[:, j] @ residual / n_samples

                # Soft thresholding for L1/Elastic Net
                if self._regularizer:
                    # Coordinate descent update with proximal operator
                    # This is a simplified version
                    weights[j] = self._regularizer.prox(
                        np.array([rho_j]),
                        1.0
                        / (
                            X_scaled[:, j] @ X_scaled[:, j] / n_samples
                            + self.alpha * (1 - self.l1_ratio)
                        ),
                    )[0]
                else:
                    weights[j] = rho_j

            if np.max(np.abs(weights - weights_old)) < self.tol:
                break

        self.coef_ = weights / self._X_std
        self.intercept_ = y_mean - self._X_mean @ self.coef_

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        return X @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


class EarlyStopping:
    """
    Early Stopping Regularization

    Stops training when validation loss stops improving.
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Parameters
        ----------
        patience : int, default=10
            Number of epochs with no improvement to wait
        min_delta : float, default=0.0
            Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.wait = 0
        self.stopped_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        """Return True if should stop training."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                return True
        return False

    def reset(self):
        """Reset early stopping state."""
        self.best_loss = float("inf")
        self.wait = 0
        self.stopped_epoch = 0


class L1RegularizedLogisticRegression:
    """
    Logistic Regression with L1 Regularization (for sparse solutions)
    """

    def __init__(self, alpha: float = 1.0, max_iter: int = 1000, tol: float = 1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def _sigmoid(self, z):
        """Numerically stable sigmoid."""
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "L1RegularizedLogisticRegression":
        """Fit the model using coordinate descent."""
        n_samples, n_features = X.shape

        # Standardize
        self._X_mean = np.mean(X, axis=0)
        self._X_std = np.std(X, axis=0)
        self._X_std[self._X_std == 0] = 1
        X_scaled = (X - self._X_mean) / self._X_std

        # Initialize weights
        weights = np.zeros(n_features)

        for _ in range(self.max_iter):
            weights_old = weights.copy()

            for j in range(n_features):
                # Compute predictions
                z = X_scaled @ weights
                p = self._sigmoid(z)

                # Residual for feature j
                residual = y - p

                # Correlation
                rho_j = X_scaled[:, j] @ residual / n_samples

                # Soft thresholding
                if abs(rho_j) > self.alpha:
                    weights[j] = np.sign(rho_j) * (abs(rho_j) - self.alpha)
                else:
                    weights[j] = 0

            if np.max(np.abs(weights - weights_old)) < self.tol:
                break

        self.coef_ = weights / self._X_std

        # Compute intercept
        z = X_scaled @ weights
        p = self._sigmoid(z)
        self.intercept_ = np.log(np.mean(p) / (1 - np.mean(p)) + 1e-10)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        X_scaled = (X - self._X_mean) / self._X_std
        z = X_scaled @ self.coef_ + self.intercept_
        p = self._sigmoid(z)
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probas = self.predict_proba(X)
        return np.where(probas[:, 1] >= 0.5, 1, 0)


# ============================================================================
# Demonstration
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Regularization Utilities - Demonstration")
    print("=" * 70)

    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    # Create data with some irrelevant features
    X = np.random.randn(n_samples, n_features)
    # Only first 3 features are relevant
    y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.5

    # Split
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    print("\n1. L2 Regularization (Ridge):")
    ridge = RegularizedLinearRegression(
        regularization="l2", alpha=1.0, learning_rate=0.1, max_iter=1000
    )
    ridge.fit(X_train, y_train)
    print(f"   Train R²: {ridge.score(X_train, y_train):.4f}")
    print(f"   Test R²:  {ridge.score(X_test, y_test):.4f}")
    print(
        f"   Non-zero coefficients: {np.sum(np.abs(ridge.coef_) > 0.01)}/{n_features}"
    )

    print("\n2. L1 Regularization (Lasso):")
    lasso = RegularizedLinearRegression(regularization="l1", alpha=0.1, max_iter=1000)
    lasso.fit(X_train, y_train)
    print(f"   Train R²: {lasso.score(X_train, y_train):.4f}")
    print(f"   Test R²:  {lasso.score(X_test, y_test):.4f}")
    print(
        f"   Non-zero coefficients: {np.sum(np.abs(lasso.coef_) > 0.01)}/{n_features}"
    )

    print("\n3. Elastic Net Regularization:")
    elastic = RegularizedLinearRegression(
        regularization="elastic", alpha=0.1, l1_ratio=0.5, max_iter=1000
    )
    elastic.fit(X_train, y_train)
    print(f"   Train R²: {elastic.score(X_train, y_train):.4f}")
    print(f"   Test R²:  {elastic.score(X_test, y_test):.4f}")
    print(
        f"   Non-zero coefficients: {np.sum(np.abs(elastic.coef_) > 0.01)}/{n_features}"
    )

    print("\n4. L1 Regularized Logistic Regression:")
    # Binary classification
    y_binary = (y > 0).astype(int)
    X_train_bin, X_test_bin = X[:80], X[80:]
    y_train_bin, y_test_bin = y_binary[:80], y_binary[80:]

    lr_l1 = L1RegularizedLogisticRegression(alpha=0.5, max_iter=1000)
    lr_l1.fit(X_train_bin, y_train_bin)
    accuracy = np.mean(lr_l1.predict(X_test_bin) == y_test_bin)
    print(f"   Test Accuracy: {accuracy:.4f}")
    print(
        f"   Non-zero coefficients: {np.sum(np.abs(lr_l1.coef_) > 0.01)}/{n_features}"
    )

    print("\n" + "=" * 70)
    print("Regularization demonstration complete!")
    print("=" * 70)
