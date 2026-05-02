"""
Advanced Machine Learning Algorithms

This module provides:
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest
- Gradient Boosting
- Naive Bayes

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Tuple, List, Optional, Callable
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class DecisionTree:
    """
    Decision Tree Classifier and Regressor.

    Uses Information Gain (Entropy) or Gini Impurity for splitting.

    Example:
        >>> tree = DecisionTree(max_depth=5)
        >>> tree.fit(X, y)
        >>> predictions = tree.predict(X_test)
    """

    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = "gini",
    ):
        """
        Initialize Decision Tree.

        Args:
            max_depth: Maximum tree depth.
            min_samples_split: Minimum samples to split a node.
            min_samples_leaf: Minimum samples in a leaf.
            criterion: 'gini' or 'entropy'.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.tree = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        """
        Build decision tree from training data.

        Args:
            X: Feature matrix (m, n).
            y: Target labels (m,).

        Returns:
            self.
        """
        self.n_classes = len(np.unique(y))
        self.tree = self._build_tree(X, y, depth=0)
        return self

    def _gini(self, y: np.ndarray) -> float:
        """Compute Gini impurity."""
        if len(y) == 0:
            return 0
        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / len(y)
        return 1 - np.sum(probs**2)

    def _entropy(self, y: np.ndarray) -> float:
        """Compute entropy."""
        if len(y) == 0:
            return 0
        counts = np.bincount(y, minlength=self.n_classes)
        probs = counts / len(y)
        # Avoid log(0)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _information_gain(
        self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray
    ) -> float:
        """Compute information gain from a split."""
        n = len(y)
        if n == 0:
            return 0

        if self.criterion == "gini":
            parent_impurity = self._gini(y)
            left_impurity = self._gini(y_left)
            right_impurity = self._gini(y_right)
        else:
            parent_impurity = self._entropy(y)
            left_impurity = self._entropy(y_left)
            right_impurity = self._entropy(y_right)

        n_left, n_right = len(y_left), len(y_right)

        if n_left == 0 or n_right == 0:
            return 0

        gain = (
            parent_impurity
            - (n_left / n) * left_impurity
            - (n_right / n) * right_impurity
        )
        return gain

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """Find the best feature and threshold for splitting."""
        best_gain = -1
        best_feature = 0
        best_threshold = 0

        n_features = X.shape[1]

        for feature in range(n_features):
            # Get unique values and sort
            values = np.unique(X[:, feature])
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                # Split data
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                # Check minimum leaf size
                if (
                    np.sum(left_mask) < self.min_samples_leaf
                    or np.sum(right_mask) < self.min_samples_leaf
                ):
                    continue

                # Compute information gain
                gain = self._information_gain(y, y[left_mask], y[right_mask])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> dict:
        """Recursively build the decision tree."""
        n_samples = len(y)
        n_labels = len(np.unique(y))

        # Stopping conditions
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or n_labels == 1
        ):
            # Return leaf node
            return self._create_leaf(y)

        # Find best split
        feature, threshold = self._best_split(X, y)

        if feature is None:
            return self._create_leaf(y)

        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Check minimum leaf size
        if (
            np.sum(left_mask) < self.min_samples_leaf
            or np.sum(right_mask) < self.min_samples_leaf
        ):
            return self._create_leaf(y)

        # Recursively build children
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree,
        }

    def _create_leaf(self, y: np.ndarray) -> dict:
        """Create a leaf node."""
        if self.n_classes == 2:
            # Binary classification
            return {"value": int(np.mean(y) > 0.5)}
        else:
            # Multi-class
            counts = np.bincount(y, minlength=self.n_classes)
            return {"value": int(np.argmax(counts))}

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples."""
        return np.array([self._predict_single(x, self.tree) for x in X])

    def _predict_single(self, x: np.ndarray, node: dict) -> int:
        """Predict for a single sample."""
        if "value" in node:
            return node["value"]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_single(x, node["left"])
        else:
            return self._predict_single(x, node["right"])

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        return np.mean(self.predict(X) == y)


class RandomForest:
    """
    Random Forest Classifier.

    Ensemble of decision trees with bootstrap sampling and feature randomness.

    Example:
        >>> rf = RandomForest(n_trees=100, max_depth=10)
        >>> rf.fit(X, y)
        >>> predictions = rf.predict(X_test)
    """

    def __init__(
        self,
        n_trees: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: str = "sqrt",
        random_state: Optional[int] = None,
    ):
        """
        Initialize Random Forest.

        Args:
            n_trees: Number of decision trees.
            max_depth: Maximum depth of trees.
            min_samples_split: Minimum samples to split.
            min_samples_leaf: Minimum samples in leaf.
            max_features: 'sqrt', 'log2', or None (all features).
            random_state: Random seed.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        """
        Build random forest from training data.

        Args:
            X: Feature matrix (m, n).
            y: Target labels (m,).

        Returns:
            self.
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n_features = X.shape[1]

        # Determine max features for each tree
        if self.max_features == "sqrt":
            self.n_max_features = int(np.sqrt(self.n_features))
        elif self.max_features == "log2":
            self.n_max_features = int(np.log2(self.n_features))
        else:
            self.n_max_features = self.n_features

        # Build trees
        self.trees = []
        for _ in range(self.n_trees):
            # Bootstrap sample
            indices = np.random.choice(len(X), len(X), replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Random feature selection
            if self.n_max_features < self.n_features:
                feature_indices = np.random.choice(
                    self.n_features, self.n_max_features, replace=False
                )
                X_boot = X_boot[:, feature_indices]

            # Build tree
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels using majority voting.

        Args:
            X: Feature matrix.

        Returns:
            Predicted labels.
        """
        # Collect predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])

        # Majority vote
        from scipy import stats

        result = stats.mode(predictions, axis=0, keepdims=False)[0]

        return result

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        all_probs = []

        for tree in self.trees:
            probs = np.zeros((len(X), 2))
            preds = tree.predict(X)
            probs[preds == 0, 0] = 1
            probs[preds == 1, 1] = 1
            all_probs.append(probs)

        return np.mean(all_probs, axis=0)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        return np.mean(self.predict(X) == y)


class SVM:
    """
    Support Vector Machine (SVM) Classifier.

    Uses Sequential Minimal Optimization (SMO) algorithm for training.

    Example:
        >>> svm = SVM(kernel='rbf', C=1.0)
        >>> svm.fit(X, y)
        >>> predictions = svm.predict(X_test)
    """

    def __init__(
        self,
        kernel: str = "linear",
        C: float = 1.0,
        gamma: float = "scale",
        degree: int = 3,
        max_iter: int = 1000,
        tol: float = 1e-3,
    ):
        """
        Initialize SVM.

        Args:
            kernel: 'linear', 'poly', 'rbf', or 'sigmoid'.
            C: Regularization parameter.
            gamma: Kernel coefficient for 'rbf', 'poly', 'sigmoid'.
            degree: Degree for polynomial kernel.
            max_iter: Maximum iterations for SMO.
            tol: Tolerance for stopping criteria.
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol

        self.alpha = None
        self.b = 0
        self.X_train = None
        self.y_train = None
        self.support_vectors = None

    def _kernel_function(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute kernel function."""
        if self.kernel == "linear":
            return np.dot(x1, x2)

        elif self.kernel == "poly":
            return (np.dot(x1, x2) + 1) ** self.degree

        elif self.kernel == "rbf":
            if self.gamma == "scale":
                gamma = 1 / (x1.shape[0] * x1.var())
            else:
                gamma = self.gamma
            return np.exp(-gamma * np.sum((x1 - x2) ** 2))

        elif self.kernel == "sigmoid":
            return np.tanh(np.dot(x1, x2) + 1)

        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")

    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """Compute kernel matrix for all pairs of samples."""
        n = X.shape[0]
        K = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                K[i, j] = self._kernel_function(X[i], X[j])
                K[j, i] = K[i, j]

        return K

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVM":
        """
        Train SVM using SMO algorithm.

        Args:
            X: Feature matrix (m, n).
            y: Target labels (m,), must be -1 or 1.

        Returns:
            self.
        """
        n_samples, n_features = X.shape
        self.X_train = X
        self.y_train = y

        # Initialize alpha parameters
        self.alpha = np.zeros(n_samples)

        # Compute kernel matrix
        self.K = self._compute_kernel_matrix(X)

        # SMO algorithm
        passes = 0
        while passes < self.max_iter:
            num_changed = 0

            for i in range(n_samples):
                # Compute error for sample i
                E_i = self._compute_error(i)

                # Check KKT conditions
                if self._check_KKT(i, E_i):
                    continue

                # Select j randomly
                j = np.random.randint(0, n_samples)
                while j == i:
                    j = np.random.randint(0, n_samples)

                # Compute error for sample j
                E_j = self._compute_error(j)

                # Save old values
                alpha_i_old = self.alpha[i]
                alpha_j_old = self.alpha[j]

                # Compute bounds L and H
                if y[i] != y[j]:
                    L = max(0, alpha_j_old - alpha_i_old)
                    H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                else:
                    L = max(0, alpha_i_old + alpha_j_old - self.C)
                    H = min(self.C, alpha_i_old + alpha_j_old)

                if L >= H:
                    continue

                # Compute eta
                eta = 2 * self.K[i, j] - self.K[i, i] - self.K[j, j]

                if eta >= 0:
                    continue

                # Update alpha_j
                self.alpha[j] = alpha_j_old - (y[j] * (E_i - E_j)) / eta

                # Clip alpha_j
                self.alpha[j] = np.clip(self.alpha[j], L, H)

                # Check if change is significant
                if abs(self.alpha[j] - alpha_j_old) < 1e-5:
                    continue

                # Update alpha_i
                self.alpha[i] = alpha_i_old + y[i] * y[j] * (
                    alpha_j_old - self.alpha[j]
                )

                # Update bias
                b1 = (
                    self.b
                    - E_i
                    - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, i]
                    - y[j] * (self.alpha[j] - alpha_j_old) * self.K[i, j]
                )

                b2 = (
                    self.b
                    - E_j
                    - y[i] * (self.alpha[i] - alpha_i_old) * self.K[i, j]
                    - y[j] * (self.alpha[j] - alpha_j_old) * self.K[j, j]
                )

                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2

                num_changed += 1

            if num_changed == 0:
                passes += 1
            else:
                passes = 0

        # Store support vectors
        self.support_vectors = self.alpha > 1e-7

        return self

    def _compute_error(self, i: int) -> float:
        """Compute error for sample i."""
        return (
            np.sum(self.alpha * self.y_train * self.K[:, i]) + self.b - self.y_train[i]
        )

    def _check_KKT(self, i: int, E_i: float) -> bool:
        """Check KKT conditions."""
        r = E_i * self.y_train[i]

        if (
            (self.alpha[i] == 0 and r < -self.tol)
            or (self.alpha[i] == self.C and r > self.tol)
            or (0 < self.alpha[i] < self.C and abs(r) > self.tol)
        ):
            return False
        return True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix.

        Returns:
            Predicted labels (-1 or 1).
        """
        predictions = []

        for x in X:
            result = 0
            for i in range(len(self.X_train)):
                if self.alpha[i] > 1e-7:
                    result += (
                        self.alpha[i]
                        * self.y_train[i]
                        * self._kernel_function(x, self.X_train[i])
                    )
            result += self.b
            predictions.append(np.sign(result))

        return np.array(predictions)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        return np.mean(self.predict(X) == y)


class GradientBoosting:
    """
    Gradient Boosting Classifier.

    Sequential ensemble method that builds trees to correct previous errors.

    Example:
        >>> gb = GradientBoosting(n_estimators=100, learning_rate=0.1)
        >>> gb.fit(X, y)
        >>> predictions = gb.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
    ):
        """
        Initialize Gradient Boosting.

        Args:
            n_estimators: Number of boosting stages.
            learning_rate: Shrinkage rate.
            max_depth: Maximum depth of trees.
            min_samples_split: Minimum samples to split.
            min_samples_leaf: Minimum samples in leaf.
            subsample: Fraction of samples for each tree.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample

        self.trees = []
        self.initial_prediction = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoosting":
        """
        Train gradient boosting model.

        Args:
            X: Feature matrix.
            y: Target labels (0 or 1).

        Returns:
            self.
        """
        n_samples = len(y)

        # Initialize with log-odds
        self.initial_prediction = np.log(np.mean(y) / (1 - np.mean(y)))

        # Initial predictions
        F = np.full(n_samples, self.initial_prediction)

        # Sequential boosting
        for i in range(self.n_estimators):
            # Compute pseudo-residuals (negative gradient)
            p = 1 / (1 + np.exp(-F))
            residuals = y - p

            # Subsample if needed
            if self.subsample < 1.0:
                indices = np.random.choice(
                    n_samples, int(n_samples * self.subsample), replace=False
                )
                X_sub = X[indices]
                residuals_sub = residuals[indices]
            else:
                X_sub = X
                residuals_sub = residuals

            # Fit regression tree to residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            )
            tree.fit(X_sub, residuals_sub)

            # Update predictions
            tree_output = tree.predict(X)
            F += self.learning_rate * tree_output

            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        # Compute final predictions
        F = np.full(len(X), self.initial_prediction)

        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)

        # Convert to class labels
        probabilities = 1 / (1 + np.exp(-F))
        return (probabilities > 0.5).astype(int)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        return np.mean(self.predict(X) == y)


class DecisionTreeRegressor:
    """Decision Tree Regressor (for Gradient Boosting)."""

    def __init__(self, max_depth=3, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, 0)
        return self

    def _build_tree(self, X, y, depth):
        n_samples = len(y)
        mean = np.mean(y)

        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or n_samples < 2 * self.min_samples_leaf
        ):
            return {"value": mean}

        # Find best split
        best_gain = -np.inf
        best_feature = 0
        best_threshold = 0

        for feature in range(X.shape[1]):
            values = np.unique(X[:, feature])
            thresholds = (values[:-1] + values[1:]) / 2

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if (
                    np.sum(left_mask) < self.min_samples_leaf
                    or np.sum(right_mask) < self.min_samples_leaf
                ):
                    continue

                # MSE reduction
                current_mse = np.var(y) * n_samples
                left_mse = (
                    np.var(y[left_mask]) * np.sum(left_mask)
                    if np.sum(left_mask) > 0
                    else 0
                )
                right_mse = (
                    np.var(y[right_mask]) * np.sum(right_mask)
                    if np.sum(right_mask) > 0
                    else 0
                )
                gain = current_mse - left_mse - right_mse

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_gain <= 0:
            return {"value": mean}

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def predict(self, X):
        return np.array([self._predict(x, self.tree) for x in X])

    def _predict(self, x, node):
        if "value" in node:
            return node["value"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict(x, node["left"])
        return self._predict(x, node["right"])


class NaiveBayes:
    """
    Naive Bayes Classifier.

    Uses Bayes theorem with independence assumption.

    Example:
        >>> nb = NaiveBayes()
        >>> nb.fit(X, y)
        >>> predictions = nb.predict(X_test)
    """

    def __init__(self, laplace: float = 1.0):
        """
        Initialize Naive Bayes.

        Args:
            laplace: Laplace smoothing parameter.
        """
        self.laplace = laplace
        self.class_priors = {}
        self.feature_probs = {}
        self.classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NaiveBayes":
        """
        Train Naive Bayes classifier.

        Args:
            X: Feature matrix.
            y: Target labels.

        Returns:
            self.
        """
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        n_features = X.shape[1]
        n_samples = len(y)

        # Compute class priors
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples

        # Compute conditional probabilities
        self.feature_probs = {}

        for c in self.classes:
            X_c = X[y == c]
            n_c = len(X_c)

            self.feature_probs[c] = []

            for feature in range(n_features):
                # Calculate mean and std for Gaussian
                mean = np.mean(X_c[:, feature])
                std = np.std(X_c[:, feature])

                if std == 0:
                    std = 1e-6  # Prevent division by zero

                self.feature_probs[c].append((mean, std))

        return self

    def _gaussian_pdf(self, x, mean, std):
        """Compute Gaussian probability density."""
        return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        n_samples = len(X)
        n_classes = len(self.classes)

        probs = np.zeros((n_samples, n_classes))

        for i, x in enumerate(X):
            for j, c in enumerate(self.classes):
                # Prior
                log_prob = np.log(self.class_priors[c])

                # Likelihood
                for feature in range(len(x)):
                    mean, std = self.feature_probs[c][feature]
                    log_prob += np.log(self._gaussian_pdf(x[feature], mean, std))

                probs[i, j] = log_prob

        # Normalize
        probs = np.exp(probs - np.max(probs, axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy."""
        return np.mean(self.predict(X) == y)


# ========================================
# MAIN DEMONSTRATION
# ========================================

if __name__ == "__main__":
    print("=" * 60)
    print("ADVANCED ML ALGORITHMS DEMONSTRATION")
    print("=" * 60)

    np.random.seed(42)

    # ========================================
    # Decision Tree
    # ========================================
    print("\n--- Decision Tree ---")

    # Generate data
    X = np.random.randn(200, 2)
    y = ((X[:, 0] > 0) & (X[:, 1] > 0)).astype(int)

    tree = DecisionTree(max_depth=5)
    tree.fit(X, y)

    print(f"Decision Tree Accuracy: {tree.accuracy(X, y):.4f}")

    # ========================================
    # Random Forest
    # ========================================
    print("\n--- Random Forest ---")

    rf = RandomForest(n_trees=50, max_depth=10, random_state=42)
    rf.fit(X, y)

    print(f"Random Forest Accuracy: {rf.accuracy(X, y):.4f}")

    # ========================================
    # SVM
    # ========================================
    print("\n--- SVM ---")

    # Linear separable data
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    X2 = np.random.randn(50, 2) + np.array([-2, -2])
    X_svm = np.vstack([X1, X2])
    y_svm = np.hstack([np.ones(50), -np.ones(50)])

    svm = SVM(kernel="linear", C=1.0)
    svm.fit(X_svm, y_svm)

    print(f"SVM Accuracy: {svm.accuracy(X_svm, y_svm):.4f}")
    print(f"Number of support vectors: {np.sum(svm.alpha > 1e-7)}")

    # ========================================
    # Gradient Boosting
    # ========================================
    print("\n--- Gradient Boosting ---")

    gb = GradientBoosting(n_estimators=50, learning_rate=0.1)
    gb.fit(X, y)

    print(f"Gradient Boosting Accuracy: {gb.accuracy(X, y):.4f}")

    # ========================================
    # Naive Bayes
    # ========================================
    print("\n--- Naive Bayes ---")

    # Generate continuous data
    X1 = np.random.randn(50, 2) + np.array([0, 0])
    X2 = np.random.randn(50, 2) + np.array([3, 3])
    X_nb = np.vstack([X1, X2])
    y_nb = np.hstack([np.zeros(50), np.ones(50)])

    nb = NaiveBayes()
    nb.fit(X_nb, y_nb)

    print(f"Naive Bayes Accuracy: {nb.accuracy(X_nb, y_nb):.4f}")

    print("\n" + "=" * 60)
