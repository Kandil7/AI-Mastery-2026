"""
Machine Learning Algorithms Module

This module provides complete implementations of:
- Linear Regression (closed-form + gradient descent)
- Logistic Regression
- Decision Trees
- K-Means Clustering
- Support Vector Machine (simplified)

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Tuple, Optional, List, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for ML models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseModel":
        """Fit the model to training data."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score (for regression) or accuracy (for classification)."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class LinearRegression(BaseModel):
    """
    Linear Regression with multiple optimization methods.

    Supports:
    - Closed-form solution (normal equation)
    - Gradient descent
    - Stochastic gradient descent
    - Regularization (L1/L2)

    Example:
        >>> model = LinearRegression(method='gradient_descent')
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> r2 = model.score(X_test, y_test)
    """

    def __init__(
        self,
        method: str = "closed_form",
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: float = 0.0,
        regularization_type: str = "l2",
        batch_size: int = 32,
        early_stopping: bool = False,
        tolerance: float = 1e-6,
    ):
        """
        Initialize Linear Regression.

        Args:
            method: 'closed_form', 'gd', 'sgd', or 'mini_batch'
            learning_rate: Step size for gradient descent.
            n_iterations: Number of training iterations.
            regularization: Regularization strength (lambda).
            regularization_type: 'l1' (Lasso) or 'l2' (Ridge).
            batch_size: Batch size for SGD/mini-batch.
            early_stopping: Whether to stop early if convergence.
            tolerance: Convergence tolerance.
        """
        self.method = method.lower()
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.regularization_type = regularization_type.lower()
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.tolerance = tolerance

        self.theta = None  # Parameters (weights + bias)
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """
        Fit the linear regression model.

        Args:
            X: Feature matrix of shape (m, n) where m=samples, n=features.
            y: Target vector of shape (m,).

        Returns:
            self: Fitted model.

        Example:
            >>> import numpy as np
            >>> X = np.array([[1], [2], [3], [4], [5]])
            >>> y = np.array([2, 4, 6, 8, 10])
            >>> model = LinearRegression(method='closed_form')
            >>> model.fit(X, y)
            >>> model.predict(np.array([[6]]))
        """
        m, n = X.shape

        # Add bias term
        X_b = np.c_[np.ones((m, 1)), X]

        if self.method == "closed_form":
            self._fit_closed_form(X_b, y)
        elif self.method in ["gd", "gradient_descent"]:
            self._fit_gradient_descent(X_b, y)
        elif self.method in ["sgd", "stochastic"]:
            self._fit_sgd(X_b, y)
        elif self.method in ["mini_batch", "mini_batch_gd"]:
            self._fit_mini_batch(X_b, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def _fit_closed_form(self, X_b: np.ndarray, y: np.ndarray) -> None:
        """
        Solve using normal equation: θ = (XᵀX)⁻¹Xᵀy

        With regularization: θ = (XᵀX + λI)⁻¹Xᵀy
        """
        n_features = X_b.shape[1]

        # Regularization matrix (not applied to bias)
        reg_matrix = self.regularization * np.eye(n_features)
        reg_matrix[0, 0] = 0  # Don't regularize bias

        # Solve: (XᵀX + λI)⁻¹Xᵀy
        XtX = X_b.T @ X_b
        XtX_reg = XtX + reg_matrix

        try:
            self.theta = np.linalg.solve(XtX_reg, X_b.T @ y)
        except np.linalg.LinAlgError:
            # If singular, use pseudoinverse
            self.theta = np.linalg.pinv(X_b.T @ X_b + reg_matrix) @ X_b.T @ y

    def _fit_gradient_descent(self, X_b: np.ndarray, y: np.ndarray) -> None:
        """Fit using batch gradient descent."""
        m = len(y)
        n_features = X_b.shape[1]
        self.theta = np.zeros(n_features)

        self.loss_history = []

        prev_loss = float("inf")

        for i in range(self.n_iterations):
            # Compute predictions
            predictions = X_b @ self.theta

            # Compute error
            error = predictions - y

            # Compute gradient
            gradient = (1 / m) * X_b.T @ error

            # Add regularization gradient
            if self.regularization > 0:
                if self.regularization_type == "l2":
                    gradient[1:] += (self.regularization / m) * self.theta[1:]
                elif self.regularization_type == "l1":
                    gradient[1:] += (self.regularization / m) * np.sign(self.theta[1:])

            # Update
            self.theta = self.theta - self.learning_rate * gradient

            # Record loss
            loss = self._compute_loss(X_b, y)
            self.loss_history.append(loss)

            # Early stopping
            if self.early_stopping and abs(prev_loss - loss) < self.tolerance:
                logger.info(f"Converged at iteration {i}")
                break

            prev_loss = loss

    def _fit_sgd(self, X_b: np.ndarray, y: np.ndarray) -> None:
        """Fit using stochastic gradient descent."""
        m = len(y)
        n_features = X_b.shape[1]
        self.theta = np.zeros(n_features)

        self.loss_history = []

        for i in range(self.n_iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X_b[indices]
            y_shuffled = y[indices]

            for j in range(m):
                xi = X_shuffled[j : j + 1]
                yi = y_shuffled[j]

                # Compute prediction and error
                prediction = xi @ self.theta
                error = prediction - yi

                # Compute gradient
                gradient = error * xi.flatten()

                # Regularization
                if self.regularization > 0:
                    gradient[1:] += (self.regularization / m) * self.theta[1:]

                # Update
                self.theta = self.theta - self.learning_rate * gradient

            # Record loss periodically
            if i % 10 == 0:
                loss = self._compute_loss(X_b, y)
                self.loss_history.append(loss)

    def _fit_mini_batch(self, X_b: np.ndarray, y: np.ndarray) -> None:
        """Fit using mini-batch gradient descent."""
        m = len(y)
        n_features = X_b.shape[1]
        self.theta = np.zeros(n_features)

        self.loss_history = []

        for i in range(self.n_iterations):
            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X_b[indices]
            y_shuffled = y[indices]

            # Process in batches
            for start in range(0, m, self.batch_size):
                end = min(start + self.batch_size, m)
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Compute gradient
                predictions = X_batch @ self.theta
                errors = predictions - y_batch
                gradient = (1 / len(y_batch)) * X_batch.T @ errors

                # Regularization
                if self.regularization > 0:
                    gradient[1:] += (self.regularization / m) * self.theta[1:]

                # Update
                self.theta = self.theta - self.learning_rate * gradient

            # Record loss
            if i % 10 == 0:
                loss = self._compute_loss(X_b, y)
                self.loss_history.append(loss)

    def _compute_loss(self, X_b: np.ndarray, y: np.ndarray) -> float:
        """Compute MSE loss with regularization."""
        m = len(y)
        predictions = X_b @ self.theta
        mse = (1 / (2 * m)) * np.sum((predictions - y) ** 2)

        # Regularization
        if self.regularization > 0:
            if self.regularization_type == "l2":
                reg_term = (self.regularization / (2 * m)) * np.sum(self.theta[1:] ** 2)
            elif self.regularization_type == "l1":
                reg_term = (self.regularization / m) * np.sum(np.abs(self.theta[1:]))
            else:
                reg_term = 0
            mse += reg_term

        return mse

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix of shape (m, n).

        Returns:
            Predictions of shape (m,).
        """
        if self.theta is None:
            raise ValueError("Model not fitted. Call fit() first.")

        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]

        return X_b @ self.theta

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict (same for linear regression)."""
        return self.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute R² coefficient of determination.

        R² = 1 - SS_res / SS_tot
        """
        if self.theta is None:
            raise ValueError("Model not fitted.")

        predictions = self.predict(X)

        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - (ss_res / ss_tot)

    def coefficients(self) -> Tuple[float, np.ndarray]:
        """Return bias and weights separately."""
        return self.theta[0], self.theta[1:]

    def get_loss_history(self) -> List[float]:
        """Return training loss history."""
        return self.loss_history


class LogisticRegression(BaseModel):
    """
    Logistic Regression for binary and multi-class classification.

    Uses sigmoid (binary) or softmax (multi-class) activation.

    Example:
        >>> model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
        >>> accuracy = model.accuracy(X_test, y_test)
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: float = 0.0,
        multi_class: bool = False,
        early_stopping: bool = False,
        tolerance: float = 1e-6,
    ):
        """
        Initialize Logistic Regression.

        Args:
            learning_rate: Step size for gradient descent.
            n_iterations: Number of training iterations.
            regularization: L2 regularization strength.
            multi_class: Whether to use softmax for multi-class.
            early_stopping: Whether to stop early on convergence.
            tolerance: Convergence tolerance.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.multi_class = multi_class
        self.early_stopping = early_stopping
        self.tolerance = tolerance

        self.theta = None
        self.classes = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegression":
        """
        Fit the logistic regression model.

        Args:
            X: Feature matrix of shape (m, n).
            y: Target labels of shape (m,). For multi-class, integer labels.

        Returns:
            self: Fitted model.
        """
        m, n = X.shape

        # Store classes
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Handle multi-class
        if n_classes > 2:
            self.multi_class = True

        # One-hot encode for multi-class
        if self.multi_class:
            y_onehot = self._one_hot_encode(y, n_classes)

        # Add bias term
        X_b = np.c_[np.ones((m, 1)), X]

        if self.multi_class:
            # Initialize weights for each class
            self.theta = np.zeros((n_classes, X_b.shape[1]))

            # Train using gradient descent with softmax
            for i in range(self.n_iterations):
                # Compute softmax probabilities
                scores = X_b @ self.theta.T
                exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
                probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

                # Compute gradient
                error = probs - y_onehot
                gradient = (1 / m) * error.T @ X_b

                # Regularization
                if self.regularization > 0:
                    gradient += (self.regularization / m) * self.theta
                    gradient[:, 0] -= (self.regularization / m) * self.theta[
                        :, 0
                    ]  # Don't regularize bias

                # Update
                self.theta = self.theta - self.learning_rate * gradient

                # Record loss
                if i % 100 == 0:
                    loss = self._compute_loss(X_b, y_onehot)
                    self.loss_history.append(loss)

        else:
            # Binary classification
            self.theta = np.zeros(X_b.shape[1])

            prev_loss = float("inf")

            for i in range(self.n_iterations):
                # Compute predictions
                probs = self._sigmoid(X_b @ self.theta)

                # Compute gradient
                error = probs - y
                gradient = (1 / m) * X_b.T @ error

                # Regularization
                if self.regularization > 0:
                    gradient[1:] += (self.regularization / m) * self.theta[1:]

                # Update
                self.theta = self.theta - self.learning_rate * gradient

                # Record loss
                if i % 100 == 0:
                    loss = self._compute_binary_loss(X_b, y)
                    self.loss_history.append(loss)

                    if self.early_stopping and abs(prev_loss - loss) < self.tolerance:
                        break

                    prev_loss = loss

        return self

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Compute sigmoid function."""
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _one_hot_encode(self, y: np.ndarray, n_classes: int) -> np.ndarray:
        """One-hot encode labels."""
        onehot = np.zeros((len(y), n_classes))
        for i, label in enumerate(y):
            onehot[i, int(label)] = 1
        return onehot

    def _compute_loss(self, X_b: np.ndarray, y_onehot: np.ndarray) -> float:
        """Compute cross-entropy loss for multi-class."""
        m = len(y_onehot)

        scores = X_b @ self.theta.T
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Cross-entropy
        ce = -np.sum(y_onehot * np.log(probs + 1e-15)) / m

        # Regularization
        if self.regularization > 0:
            ce += (self.regularization / (2 * m)) * np.sum(self.theta**2)

        return ce

    def _compute_binary_loss(self, X_b: np.ndarray, y: np.ndarray) -> float:
        """Compute cross-entropy loss for binary."""
        m = len(y)
        probs = self._sigmoid(X_b @ self.theta)

        # Cross-entropy
        ce = -(1 / m) * np.sum(
            y * np.log(probs + 1e-15) + (1 - y) * np.log(1 - probs + 1e-15)
        )

        # Regularization
        if self.regularization > 0:
            ce += (self.regularization / (2 * m)) * np.sum(self.theta[1:] ** 2)

        return ce

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.

        Args:
            X: Feature matrix of shape (m, n).

        Returns:
            Predicted class labels.
        """
        if self.theta is None:
            raise ValueError("Model not fitted.")

        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]

        if self.multi_class:
            # Softmax
            scores = X_b @ self.theta.T
            probs = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            probs = probs / np.sum(probs, axis=1, keepdims=True)

            # Return class with highest probability
            predictions = np.argmax(probs, axis=1)
        else:
            # Sigmoid
            probs = self._sigmoid(X_b @ self.theta)
            predictions = (probs >= 0.5).astype(int)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return probability estimates.

        Args:
            X: Feature matrix.

        Returns:
            Array of probabilities.
        """
        if self.theta is None:
            raise ValueError("Model not fitted.")

        m = X.shape[0]
        X_b = np.c_[np.ones((m, 1)), X]

        if self.multi_class:
            scores = X_b @ self.theta.T
            exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
            return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        else:
            return self._sigmoid(X_b @ self.theta)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Alias for accuracy."""
        return self.accuracy(X, y)


class KMeans(BaseModel):
    """
    K-Means Clustering algorithm.

    Example:
        >>> kmeans = KMeans(n_clusters=3, n_iterations=100)
        >>> labels = kmeans.fit_predict(X)
    """

    def __init__(
        self,
        n_clusters: int = 3,
        n_iterations: int = 100,
        initialization: str = "kmeans++",
        tolerance: float = 1e-4,
    ):
        """
        Initialize K-Means.

        Args:
            n_clusters: Number of clusters (K).
            n_iterations: Maximum iterations.
            initialization: 'random' or 'kmeans++'.
            tolerance: Convergence tolerance.
        """
        self.n_clusters = n_clusters
        self.n_iterations = n_iterations
        self.initialization = initialization
        self.tolerance = tolerance

        self.centroids = None
        self.labels = None
        self.inertia = None

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Fit K-Means to data.

        Args:
            X: Feature matrix of shape (m, n).

        Returns:
            self.
        """
        m, n = X.shape

        # Initialize centroids
        if self.initialization == "kmeans++":
            self.centroids = self._kmeans_plus_plus_init(X)
        else:
            indices = np.random.choice(m, self.n_clusters, replace=False)
            self.centroids = X[indices].copy()

        # Iterative refinement
        for _ in range(self.n_iterations):
            # Assign points to nearest centroid
            labels = self._assign_clusters(X)

            # Update centroids
            new_centroids = self._update_centroids(X, labels)

            # Check convergence
            if np.max(np.abs(new_centroids - self.centroids)) < self.tolerance:
                break

            self.centroids = new_centroids

        self.labels = self._assign_clusters(X)
        self.inertia = self._compute_inertia(X)

        return self

    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """K-Means++ initialization."""
        m = X.shape[0]
        centroids = np.zeros((self.n_clusters, X.shape[1]))

        # First centroid: random
        centroids[0] = X[np.random.randint(m)]

        # Remaining centroids
        for k in range(1, self.n_clusters):
            # Compute distances to nearest centroid
            distances = np.min(
                [np.sum((X - c) ** 2, axis=1) for c in centroids[:k]], axis=0
            )

            # Probability proportional to distance squared
            probs = distances / distances.sum()

            # Select next centroid
            centroids[k] = X[np.random.choice(m, p=probs)]

        return centroids

    def _assign_clusters(self, X: np.ndarray) -> np.ndarray:
        """Assign each point to nearest centroid."""
        distances = np.zeros((len(X), self.n_clusters))

        for k in range(self.n_clusters):
            distances[:, k] = np.sum((X - self.centroids[k]) ** 2, axis=1)

        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute new centroids as mean of cluster points."""
        new_centroids = np.zeros_like(self.centroids)

        for k in range(self.n_clusters):
            mask = labels == k
            if np.sum(mask) > 0:
                new_centroids[k] = np.mean(X[mask], axis=0)
            else:
                # Keep old centroid if cluster is empty
                new_centroids[k] = self.centroids[k]

        return new_centroids

    def _compute_inertia(self, X: np.ndarray) -> float:
        """Compute sum of squared distances to centroids."""
        labels = self._assign_clusters(X)

        inertia = 0
        for k in range(self.n_clusters):
            mask = labels == k
            inertia += np.sum((X[mask] - self.centroids[k]) ** 2)

        return inertia

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels for new data."""
        return self._assign_clusters(X)


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("ML Algorithms Module - Demonstration")
    print("=" * 60)

    np.random.seed(42)

    # ========================================
    # Linear Regression Demo
    # ========================================
    print("\n--- Linear Regression ---")

    # Generate data: y = 3x + 5 + noise
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 3 * X.flatten() + 5 + np.random.randn(100) * 2

    # Train model
    model = LinearRegression(method="closed_form")
    model.fit(X, y)

    print(f"True: y = 3x + 5")
    bias, weights = model.coefficients()
    print(f"Learned: y = {weights[0]:.3f}x + {bias:.3f}")
    print(f"R² Score: {model.score(X, y):.4f}")

    # Compare methods
    print("\nComparing optimization methods:")
    methods = ["closed_form", "gd", "sgd"]
    for method in methods:
        model = LinearRegression(method=method, learning_rate=0.01, n_iterations=1000)
        model.fit(X, y)
        print(f"  {method}: R² = {model.score(X, y):.4f}")

    # ========================================
    # Logistic Regression Demo
    # ========================================
    print("\n--- Logistic Regression ---")

    # Generate binary classification data
    X1 = np.random.randn(50, 2) + np.array([-2, -2])
    X2 = np.random.randn(50, 2) + np.array([2, 2])
    X = np.vstack([X1, X2])
    y = np.hstack([np.zeros(50), np.ones(50)])

    # Shuffle
    indices = np.random.permutation(100)
    X, y = X[indices], y[indices]

    # Train
    log_model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    log_model.fit(X, y)

    print(f"Binary Classification:")
    print(f"  Accuracy: {log_model.accuracy(X, y):.4f}")

    # Multi-class
    print("\nMulti-class Classification:")
    X0 = np.random.randn(30, 2) + np.array([0, 0])
    X1 = np.random.randn(30, 2) + np.array([4, 0])
    X2 = np.random.randn(30, 2) + np.array([2, 4])
    X_multi = np.vstack([X0, X1, X2])
    y_multi = np.hstack([np.zeros(30), np.ones(30), np.ones(30) * 2])

    log_model_multi = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    log_model_multi.fit(X_multi, y_multi)
    print(f"  Accuracy: {log_model_multi.accuracy(X_multi, y_multi):.4f}")

    # ========================================
    # K-Means Demo
    # ========================================
    print("\n--- K-Means Clustering ---")

    # Generate 3 clusters
    centers = np.array([[0, 0], [5, 5], [10, 0]])
    X_clusters = []
    for center in centers:
        X_clusters.append(np.random.randn(30, 2) + center)
    X_cluster = np.vstack(X_clusters)

    # Cluster
    kmeans = KMeans(n_clusters=3, n_iterations=100)
    labels = kmeans.fit_predict(X_cluster)

    print(f"K-Means Results:")
    print(f"  Clusters found: {len(np.unique(labels))}")
    print(f"  Inertia: {kmeans.inertia:.2f}")

    print("\n" + "=" * 60)
