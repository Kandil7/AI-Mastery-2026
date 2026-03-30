"""
Machine Learning Algorithms Module.

This module provides from-scratch implementations of fundamental ML algorithms,
including linear regression, logistic regression, decision trees, random forests,
K-means clustering, and PCA.

Example Usage:
    >>> import numpy as np
    >>> from ml_algorithms import LinearRegression, LogisticRegression
    >>> from ml_algorithms import DecisionTreeClassifier, RandomForestClassifier
    >>> from ml_algorithms import KMeans, PCA
    >>> 
    >>> # Linear regression
    >>> X = np.random.randn(100, 2)
    >>> y = 2 * X[:, 0] + 3 * X[:, 1] + np.random.randn(100) * 0.1
    >>> model = LinearRegression()
    >>> model.fit(X, y)
    >>> predictions = model.predict(X)
    >>> 
    >>> # K-means clustering
    >>> X = np.random.randn(300, 2)
    >>> kmeans = KMeans(n_clusters=3)
    >>> labels = kmeans.fit_predict(X)
"""

from typing import Union, List, Dict, Any, Optional, Tuple, Callable
import numpy as np
from numpy.typing import ArrayLike
import logging
from collections import Counter
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

ArrayLike2D = Union[np.ndarray, List[List[float]]]
ArrayLike1D = Union[np.ndarray, List[float]]


@dataclass
class ModelMetrics:
    """Container for model evaluation metrics."""
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    mse: Optional[float] = None
    rmse: Optional[float] = None
    mae: Optional[float] = None
    r2_score: Optional[float] = None


class LinearRegression:
    """
    Linear Regression implementation from scratch.
    
    Supports both closed-form solution (normal equation) and gradient descent.
    
    Model: y = X @ w + b
    
    Attributes:
        weights: Learned weights (coefficients).
        bias: Learned bias (intercept).
        history: Training history (loss per iteration).
    
    Example:
        >>> model = LinearRegression()
        >>> X = np.array([[1], [2], [3], [4]])
        >>> y = np.array([2, 4, 6, 8])
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
        >>> np.allclose(predictions, y, atol=0.1)
        True
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
        regularization: Optional[str] = None,
        reg_lambda: float = 0.01
    ):
        """
        Initialize Linear Regression.
        
        Args:
            learning_rate: Step size for gradient descent. Default: 0.01.
            n_iterations: Number of iterations for gradient descent. Default: 1000.
            fit_intercept: Whether to fit bias term. Default: True.
            regularization: Regularization type ('l1', 'l2', 'elastic'). Default: None.
            reg_lambda: Regularization strength. Default: 0.01.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.history: List[float] = []
        
        logger.debug(f"LinearRegression initialized: lr={learning_rate}, "
                    f"iterations={n_iterations}, regularization={regularization}")
    
    def fit(
        self,
        X: ArrayLike2D,
        y: ArrayLike1D,
        method: str = 'gradient_descent'
    ) -> 'LinearRegression':
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target vector (n_samples,).
            method: Fitting method ('gradient_descent' or 'normal').
        
        Returns:
            self: Fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).flatten()
        
        n_samples, n_features = X.shape
        
        if method == 'normal':
            self._fit_normal(X, y)
        else:
            self._fit_gradient_descent(X, y)
        
        logger.info(f"LinearRegression fitted: weights shape = {self.weights.shape}")
        return self
    
    def _fit_normal(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using normal equation (closed-form solution)."""
        if self.fit_intercept:
            X_b = np.c_[np.ones(X.shape[0]), X]
        else:
            X_b = X
        
        # Normal equation: w = (X^T X)^-1 X^T y
        try:
            if self.regularization == 'l2':
                # Ridge regression
                n_features = X_b.shape[1]
                reg_matrix = self.reg_lambda * np.eye(n_features)
                reg_matrix[0, 0] = 0  # Don't regularize bias
                self.weights = np.linalg.solve(
                    X_b.T @ X_b + reg_matrix,
                    X_b.T @ y
                )
            else:
                self.weights = np.linalg.lstsq(X_b, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Use pseudo-inverse for singular matrices
            self.weights = np.linalg.pinv(X_b) @ y
        
        if self.fit_intercept:
            self.bias = self.weights[0]
            self.weights = self.weights[1:]
        else:
            self.bias = 0.0
    
    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit using gradient descent."""
        n_samples, n_features = X.shape
        
        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.history = []
        
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = self._predict_internal(X)
            
            # Compute loss
            loss = self._compute_loss(y, y_pred)
            self.history.append(loss)
            
            # Compute gradients
            error = y_pred - y
            dw = (1 / n_samples) * X.T @ error
            db = (1 / n_samples) * np.sum(error)
            
            # Add regularization gradient
            if self.regularization == 'l2':
                dw += self.reg_lambda * self.weights
            elif self.regularization == 'l1':
                dw += self.reg_lambda * np.sign(self.weights)
            
            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
        
        logger.debug(f"Gradient descent completed, final loss = {self.history[-1]:.6f}")
    
    def _compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss with optional regularization."""
        mse = np.mean((y_true - y_pred) ** 2)
        
        if self.regularization == 'l2':
            reg_term = 0.5 * self.reg_lambda * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            reg_term = self.reg_lambda * np.sum(np.abs(self.weights))
        else:
            reg_term = 0.0
        
        return 0.5 * mse + reg_term
    
    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        """Internal prediction without validation."""
        return X @ self.weights + self.bias
    
    def predict(self, X: ArrayLike2D) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X: Feature matrix (n_samples, n_features).
        
        Returns:
            np.ndarray: Predictions (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        return self._predict_internal(X)
    
    def score(self, X: ArrayLike2D, y: ArrayLike1D) -> float:
        """
        Compute R² score.
        
        Args:
            X: Feature matrix.
            y: True target values.
        
        Returns:
            float: R² score.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot)
    
    def get_metrics(self, X: ArrayLike2D, y: ArrayLike1D) -> ModelMetrics:
        """
        Compute regression metrics.
        
        Args:
            X: Feature matrix.
            y: True target values.
        
        Returns:
            ModelMetrics: Metrics container.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        y_pred = self.predict(X)
        
        mse = np.mean((y - y_pred) ** 2)
        
        return ModelMetrics(
            mse=float(mse),
            rmse=float(np.sqrt(mse)),
            mae=float(np.mean(np.abs(y - y_pred))),
            r2_score=float(self.score(X, y))
        )


class LogisticRegression:
    """
    Logistic Regression implementation from scratch.
    
    Binary and multiclass classification using sigmoid or softmax.
    
    Model: P(y=1|x) = σ(X @ w + b)
    
    Example:
        >>> model = LogisticRegression()
        >>> X = np.array([[1, 2], [2, 3], [3, 4], [6, 7]])
        >>> y = np.array([0, 0, 1, 1])
        >>> model.fit(X, y)
        >>> predictions = model.predict(X)
    """
    
    def __init__(
        self,
        learning_rate: float = 0.1,
        n_iterations: int = 1000,
        fit_intercept: bool = True,
        regularization: Optional[str] = None,
        reg_lambda: float = 0.01
    ):
        """
        Initialize Logistic Regression.
        
        Args:
            learning_rate: Step size for gradient descent.
            n_iterations: Number of iterations.
            fit_intercept: Whether to fit bias term.
            regularization: Regularization type ('l1', 'l2').
            reg_lambda: Regularization strength.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.fit_intercept = fit_intercept
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.classes_: Optional[np.ndarray] = None
        self.history: List[float] = []
        
        logger.debug(f"LogisticRegression initialized: lr={learning_rate}")
    
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function."""
        # Clip to avoid overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        """Softmax activation function for multiclass."""
        z = z - np.max(z, axis=1, keepdims=True)  # Numerical stability
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(
        self,
        X: ArrayLike2D,
        y: ArrayLike1D
    ) -> 'LogisticRegression':
        """
        Fit the model to training data.
        
        Args:
            X: Feature matrix (n_samples, n_features).
            y: Target labels (n_samples,).
        
        Returns:
            self: Fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32).flatten()
        
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize weights
        if n_classes == 2:
            self.weights = np.zeros(n_features)
            self.bias = 0.0
        else:
            self.weights = np.zeros((n_features, n_classes))
            self.bias = np.zeros(n_classes)
        
        self.history = []
        
        # Create label mapping for multiclass
        if n_classes > 2:
            label_to_idx = {c: i for i, c in enumerate(self.classes_)}
            y_encoded = np.array([label_to_idx[yi] for yi in y])
        else:
            y_encoded = y
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            logits = X @ self.weights + self.bias
            
            if n_classes == 2:
                probs = self._sigmoid(logits)
            else:
                probs = self._softmax(logits)
                probs = probs.flatten()
            
            # Compute loss (cross-entropy)
            loss = self._compute_loss(y_encoded, probs, n_classes)
            self.history.append(loss)
            
            # Compute gradients
            if n_classes == 2:
                error = probs - y_encoded
                dw = (1 / n_samples) * X.T @ error
                db = (1 / n_samples) * np.sum(error)
            else:
                # One-hot encode y
                y_onehot = np.zeros((n_samples, n_classes))
                y_onehot[np.arange(n_samples), y_encoded] = 1
                
                probs_2d = self._softmax(logits)
                error = probs_2d - y_onehot
                dw = (1 / n_samples) * X.T @ error
                db = (1 / n_samples) * np.sum(error, axis=0)
            
            # Add regularization
            if self.regularization == 'l2':
                dw += self.reg_lambda * self.weights
            
            # Update weights
            self.weights -= self.learning_rate * dw
            if n_classes == 2:
                self.bias -= self.learning_rate * db
            else:
                self.bias -= self.learning_rate * db
        
        logger.info(f"LogisticRegression fitted: {n_classes} classes")
        return self
    
    def _compute_loss(
        self,
        y: np.ndarray,
        probs: np.ndarray,
        n_classes: int
    ) -> float:
        """Compute cross-entropy loss."""
        epsilon = 1e-15
        probs = np.clip(probs, epsilon, 1 - epsilon)
        
        if n_classes == 2:
            loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
        else:
            loss = -np.mean(np.log(probs[np.arange(len(y)), y]))
        
        # Add regularization
        if self.regularization == 'l2':
            loss += 0.5 * self.reg_lambda * np.sum(self.weights ** 2)
        
        return float(loss)
    
    def predict_proba(self, X: ArrayLike2D) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature matrix.
        
        Returns:
            np.ndarray: Probabilities (n_samples, n_classes).
        """
        X = np.asarray(X, dtype=np.float64)
        
        logits = X @ self.weights + self.bias
        
        if len(self.classes_) == 2:
            prob_pos = self._sigmoid(logits)
            return np.column_stack([1 - prob_pos, prob_pos])
        else:
            return self._softmax(logits)
    
    def predict(self, X: ArrayLike2D) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix.
        
        Returns:
            np.ndarray: Predicted labels.
        """
        X = np.asarray(X, dtype=np.float64)
        probs = self.predict_proba(X)
        return self.classes_[np.argmax(probs, axis=1)]
    
    def score(self, X: ArrayLike2D, y: ArrayLike1D) -> float:
        """
        Compute accuracy score.
        
        Args:
            X: Feature matrix.
            y: True labels.
        
        Returns:
            float: Accuracy.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        
        predictions = self.predict(X)
        return float(np.mean(predictions == y))
    
    def get_metrics(self, X: ArrayLike2D, y: ArrayLike1D) -> ModelMetrics:
        """
        Compute classification metrics.
        
        Args:
            X: Feature matrix.
            y: True labels.
        
        Returns:
            ModelMetrics: Metrics container.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        
        y_pred = self.predict(X)
        
        # Binary classification metrics
        if len(self.classes_) == 2:
            tp = np.sum((y_pred == 1) & (y == 1))
            tn = np.sum((y_pred == 0) & (y == 0))
            fp = np.sum((y_pred == 1) & (y == 0))
            fn = np.sum((y_pred == 0) & (y == 1))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        else:
            precision = recall = f1 = None
        
        return ModelMetrics(
            accuracy=float(np.mean(y_pred == y)),
            precision=precision,
            recall=recall,
            f1_score=f1
        )


class DecisionTreeNode:
    """Node in a decision tree."""
    
    def __init__(
        self,
        feature_idx: Optional[int] = None,
        threshold: Optional[float] = None,
        left: Optional['DecisionTreeNode'] = None,
        right: Optional['DecisionTreeNode'] = None,
        value: Optional[Any] = None
    ):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf value
    
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    """
    Decision Tree Classifier implementation from scratch.
    
    Uses Gini impurity or entropy for splitting.
    
    Example:
        >>> tree = DecisionTreeClassifier(max_depth=3)
        >>> X = np.array([[1, 2], [2, 3], [3, 4], [6, 7]])
        >>> y = np.array([0, 0, 1, 1])
        >>> tree.fit(X, y)
        >>> predictions = tree.predict(X)
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = 'gini',
        max_features: Optional[int] = None,
        random_state: Optional[int] = None
    ):
        """
        Initialize Decision Tree.
        
        Args:
            max_depth: Maximum tree depth. None for unlimited.
            min_samples_split: Minimum samples to split a node.
            min_samples_leaf: Minimum samples in a leaf.
            criterion: Split criterion ('gini' or 'entropy').
            max_features: Max features to consider for split.
            random_state: Random seed.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.random_state = random_state
        
        self.root: Optional[DecisionTreeNode] = None
        self.classes_: Optional[np.ndarray] = None
        self.n_features_: int = 0
        
        if random_state is not None:
            np.random.seed(random_state)
        
        logger.debug(f"DecisionTreeClassifier initialized: max_depth={max_depth}")
    
    def fit(
        self,
        X: ArrayLike2D,
        y: ArrayLike1D
    ) -> 'DecisionTreeClassifier':
        """
        Fit the decision tree.
        
        Args:
            X: Feature matrix.
            y: Target labels.
        
        Returns:
            self: Fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32).flatten()
        
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        
        if self.max_features is None:
            self.max_features = self.n_features_
        
        self.root = self._build_tree(X, y, depth=0)
        
        logger.info(f"DecisionTreeClassifier fitted: root node created")
        return self
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """Compute Gini impurity."""
        if len(y) == 0:
            return 0
        
        probs = np.bincount(y, minlength=len(self.classes_)) / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _entropy(self, y: np.ndarray) -> float:
        """Compute entropy."""
        if len(y) == 0:
            return 0
        
        probs = np.bincount(y, minlength=len(self.classes_)) / len(y)
        probs = probs[probs > 0]  # Avoid log(0)
        return -np.sum(probs * np.log2(probs))
    
    def _impurity(self, y: np.ndarray) -> float:
        """Compute impurity based on criterion."""
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _information_gain(
        self,
        y: np.ndarray,
        y_left: np.ndarray,
        y_right: np.ndarray
    ) -> float:
        """Compute information gain from a split."""
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        parent_impurity = self._impurity(y)
        child_impurity = (n_left / n) * self._impurity(y_left) + \
                        (n_right / n) * self._impurity(y_right)
        
        return parent_impurity - child_impurity
    
    def _best_split(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """Find the best split for a node."""
        n_samples, n_features = X.shape
        
        if n_samples < self.min_samples_split:
            return None, None, 0
        
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        # Random feature subset
        feature_indices = np.random.choice(
            n_features,
            min(self.max_features, n_features),
            replace=False
        )
        
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or \
                   np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                gain = self._information_gain(
                    y,
                    y[left_mask],
                    y[right_mask]
                )
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        depth: int
    ) -> DecisionTreeNode:
        """Recursively build the tree."""
        n_samples = len(y)
        
        # Check stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n_samples < self.min_samples_split or \
           len(np.unique(y)) == 1:
            # Create leaf node
            most_common_class = np.bincount(y).argmax()
            return DecisionTreeNode(value=most_common_class)
        
        # Find best split
        feature_idx, threshold, gain = self._best_split(X, y)
        
        if feature_idx is None or gain == 0:
            most_common_class = np.bincount(y).argmax()
            return DecisionTreeNode(value=most_common_class)
        
        # Split data
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        
        # Recursively build subtrees
        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(
            feature_idx=feature_idx,
            threshold=threshold,
            left=left_child,
            right=right_child
        )
    
    def _predict_sample(self, x: np.ndarray, node: DecisionTreeNode) -> int:
        """Predict class for a single sample."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature_idx] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: ArrayLike2D) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature matrix.
        
        Returns:
            np.ndarray: Predicted labels.
        """
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_sample(x, self.root) for x in X])
    
    def predict_proba(self, X: ArrayLike2D) -> np.ndarray:
        """
        Predict class probabilities (based on leaf class distribution).
        
        Args:
            X: Feature matrix.
        
        Returns:
            np.ndarray: Probabilities (n_samples, n_classes).
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        probs = np.zeros((n_samples, n_classes))
        
        for i, x in enumerate(X):
            leaf_class = self._predict_sample(x, self.root)
            probs[i, leaf_class] = 1.0
        
        return probs
    
    def score(self, X: ArrayLike2D, y: ArrayLike1D) -> float:
        """Compute accuracy."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        return float(np.mean(self.predict(X) == y))


class RandomForestClassifier:
    """
    Random Forest Classifier implementation from scratch.
    
    Ensemble of decision trees with bagging and feature randomness.
    
    Example:
        >>> rf = RandomForestClassifier(n_estimators=10, max_depth=5)
        >>> X = np.random.randn(100, 5)
        >>> y = np.random.randint(0, 2, 100)
        >>> rf.fit(X, y)
        >>> predictions = rf.predict(X)
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        criterion: str = 'gini',
        max_features: str = 'sqrt',
        bootstrap: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize Random Forest.
        
        Args:
            n_estimators: Number of trees.
            max_depth: Maximum tree depth.
            min_samples_split: Minimum samples to split.
            min_samples_leaf: Minimum samples in leaf.
            criterion: Split criterion.
            max_features: Features per split ('sqrt', 'log2', or int).
            bootstrap: Use bootstrap samples.
            random_state: Random seed.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        self.trees: List[DecisionTreeClassifier] = []
        self.classes_: Optional[np.ndarray] = None
        
        if random_state is not None:
            np.random.seed(random_state)
        
        logger.debug(f"RandomForestClassifier initialized: {n_estimators} trees")
    
    def fit(
        self,
        X: ArrayLike2D,
        y: ArrayLike1D
    ) -> 'RandomForestClassifier':
        """
        Fit the random forest.
        
        Args:
            X: Feature matrix.
            y: Target labels.
        
        Returns:
            self: Fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32).flatten()
        
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        
        # Determine max_features
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            max_features = self.max_features
        else:
            max_features = n_features
        
        self.trees = []
        
        for i in range(self.n_estimators):
            # Bootstrap sample
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_boot, y_boot = X[indices], y[indices]
            else:
                X_boot, y_boot = X, y
            
            # Create and fit tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                criterion=self.criterion,
                max_features=max_features,
                random_state=self.random_state + i if self.random_state else None
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        
        logger.info(f"RandomForestClassifier fitted: {len(self.trees)} trees")
        return self
    
    def predict(self, X: ArrayLike2D) -> np.ndarray:
        """
        Predict class labels by majority voting.
        
        Args:
            X: Feature matrix.
        
        Returns:
            np.ndarray: Predicted labels.
        """
        X = np.asarray(X, dtype=np.float64)
        
        # Collect predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Majority vote
        result = np.array([
            np.bincount(predictions[:, i], minlength=len(self.classes_)).argmax()
            for i in range(X.shape[0])
        ])
        
        return self.classes_[result]
    
    def predict_proba(self, X: ArrayLike2D) -> np.ndarray:
        """
        Predict class probabilities by averaging tree probabilities.
        
        Args:
            X: Feature matrix.
        
        Returns:
            np.ndarray: Probabilities.
        """
        X = np.asarray(X, dtype=np.float64)
        
        # Average probabilities from all trees
        probas = np.array([tree.predict_proba(X) for tree in self.trees])
        return np.mean(probas, axis=0)
    
    def score(self, X: ArrayLike2D, y: ArrayLike1D) -> float:
        """Compute accuracy."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int32)
        return float(np.mean(self.predict(X) == y))


class KMeans:
    """
    K-Means Clustering implementation from scratch.
    
    Iterative algorithm to partition data into k clusters.
    
    Example:
        >>> kmeans = KMeans(n_clusters=3)
        >>> X = np.random.randn(300, 2)
        >>> labels = kmeans.fit_predict(X)
        >>> centers = kmeans.cluster_centers_
    """
    
    def __init__(
        self,
        n_clusters: int = 3,
        max_iterations: int = 300,
        init: str = 'kmeans++',
        random_state: Optional[int] = None,
        n_init: int = 10
    ):
        """
        Initialize K-Means.
        
        Args:
            n_clusters: Number of clusters.
            max_iterations: Maximum iterations per run.
            init: Initialization method ('random' or 'kmeans++').
            random_state: Random seed.
            n_init: Number of runs with different seeds.
        """
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.init = init
        self.random_state = random_state
        self.n_init = n_init
        
        self.cluster_centers_: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: float = 0.0
        self.n_iterations_: int = 0
        
        if random_state is not None:
            np.random.seed(random_state)
        
        logger.debug(f"KMeans initialized: k={n_clusters}")
    
    def fit(
        self,
        X: ArrayLike2D
    ) -> 'KMeans':
        """
        Fit K-Means to data.
        
        Args:
            X: Feature matrix.
        
        Returns:
            self: Fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        
        best_inertia = float('inf')
        best_centers = None
        best_labels = None
        best_n_iter = 0
        
        seeds = range(self.n_init) if self.random_state is None else \
                [self.random_state + i for i in range(self.n_init)]
        
        for seed in seeds:
            np.random.seed(seed)
            centers, labels, inertia, n_iter = self._fit_single(X)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
                best_n_iter = n_iter
        
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iterations_ = best_n_iter
        
        logger.info(f"KMeans fitted: inertia={best_inertia:.4f}, iterations={best_n_iter}")
        return self
    
    def _fit_single(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, float, int]:
        """Single run of K-Means."""
        n_samples, n_features = X.shape
        
        # Initialize centers
        if self.init == 'kmeans++':
            centers = self._kmeans_plus_plus_init(X)
        else:
            indices = np.random.choice(n_samples, self.n_clusters, replace=False)
            centers = X[indices]
        
        labels = np.zeros(n_samples, dtype=np.int32)
        
        for iteration in range(self.max_iterations):
            # Assign clusters
            new_labels = self._assign_clusters(X, centers)
            
            # Check convergence
            if np.all(labels == new_labels):
                break
            
            labels = new_labels
            
            # Update centers
            for k in range(self.n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    centers[k] = np.mean(X[mask], axis=0)
        
        # Compute inertia
        inertia = self._compute_inertia(X, centers, labels)
        
        return centers, labels, inertia, iteration + 1
    
    def _kmeans_plus_plus_init(self, X: np.ndarray) -> np.ndarray:
        """K-means++ initialization."""
        n_samples, n_features = X.shape
        centers = np.zeros((self.n_clusters, n_features))
        
        # Choose first center randomly
        idx = np.random.randint(n_samples)
        centers[0] = X[idx]
        
        for k in range(1, self.n_clusters):
            # Compute distances to nearest center
            distances = np.zeros(n_samples)
            for i, x in enumerate(X):
                min_dist = min(np.sum((x - c) ** 2) for c in centers[:k])
                distances[i] = min_dist
            
            # Choose next center with probability proportional to distance squared
            probs = distances / distances.sum()
            idx = np.random.choice(n_samples, p=probs)
            centers[k] = X[idx]
        
        return centers
    
    def _assign_clusters(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Assign each sample to nearest cluster."""
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for k, center in enumerate(centers):
            distances[:, k] = np.sum((X - center) ** 2, axis=1)
        
        return np.argmin(distances, axis=1)
    
    def _compute_inertia(
        self,
        X: np.ndarray,
        centers: np.ndarray,
        labels: np.ndarray
    ) -> float:
        """Compute sum of squared distances to cluster centers."""
        inertia = 0.0
        for k, center in enumerate(centers):
            mask = labels == k
            if np.sum(mask) > 0:
                inertia += np.sum((X[mask] - center) ** 2)
        return float(inertia)
    
    def fit_predict(self, X: ArrayLike2D) -> np.ndarray:
        """
        Fit and return labels.
        
        Args:
            X: Feature matrix.
        
        Returns:
            np.ndarray: Cluster labels.
        """
        self.fit(X)
        return self.labels_
    
    def predict(self, X: ArrayLike2D) -> np.ndarray:
        """
        Predict cluster labels for new data.
        
        Args:
            X: Feature matrix.
        
        Returns:
            np.ndarray: Cluster labels.
        """
        X = np.asarray(X, dtype=np.float64)
        return self._assign_clusters(X, self.cluster_centers_)
    
    def transform(self, X: ArrayLike2D) -> np.ndarray:
        """
        Transform data to cluster distance space.
        
        Args:
            X: Feature matrix.
        
        Returns:
            np.ndarray: Distances to each cluster center.
        """
        X = np.asarray(X, dtype=np.float64)
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for k, center in enumerate(self.cluster_centers_):
            distances[:, k] = np.sqrt(np.sum((X - center) ** 2, axis=1))
        
        return distances


class PCA:
    """
    Principal Component Analysis implementation from scratch.
    
    Dimensionality reduction using eigendecomposition or SVD.
    
    Example:
        >>> pca = PCA(n_components=2)
        >>> X = np.random.randn(100, 5)
        >>> X_reduced = pca.fit_transform(X)
        >>> explained = pca.explained_variance_ratio_
    """
    
    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        svd_solver: str = 'full'
    ):
        """
        Initialize PCA.
        
        Args:
            n_components: Number of components or variance to retain (0-1).
            svd_solver: SVD solver ('full', 'randomized').
        """
        self.n_components = n_components
        self.svd_solver = svd_solver
        
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None
        self.n_components_: int = 0
        
        logger.debug(f"PCA initialized: n_components={n_components}")
    
    def fit(
        self,
        X: ArrayLike2D
    ) -> 'PCA':
        """
        Fit PCA to data.
        
        Args:
            X: Feature matrix.
        
        Returns:
            self: Fitted model.
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # Center data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        cov_matrix = np.cov(X_centered, rowvar=False)
        
        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]
        
        # Determine number of components
        if self.n_components is None:
            self.n_components_ = min(n_samples, n_features)
        elif isinstance(self.n_components, float) and 0 < self.n_components < 1:
            cumsum = np.cumsum(eigenvalues) / np.sum(eigenvalues)
            self.n_components_ = np.searchsorted(cumsum, self.n_components) + 1
        else:
            self.n_components_ = int(self.n_components)
        
        self.n_components_ = min(self.n_components_, n_features)
        
        # Store results
        self.components_ = eigenvectors[:, :self.n_components_].T
        self.explained_variance_ = eigenvalues[:self.n_components_]
        self.explained_variance_ratio_ = eigenvalues[:self.n_components_] / np.sum(eigenvalues)
        
        logger.info(f"PCA fitted: {self.n_components_} components, "
                   f"explained variance ratio = {np.sum(self.explained_variance_ratio_):.4f}")
        return self
    
    def transform(self, X: ArrayLike2D) -> np.ndarray:
        """
        Transform data to principal component space.
        
        Args:
            X: Feature matrix.
        
        Returns:
            np.ndarray: Transformed data.
        """
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: ArrayLike2D) -> np.ndarray:
        """
        Fit and transform data.
        
        Args:
            X: Feature matrix.
        
        Returns:
            np.ndarray: Transformed data.
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X: ArrayLike2D) -> np.ndarray:
        """
        Reconstruct data from principal components.
        
        Args:
            X: Transformed data.
        
        Returns:
            np.ndarray: Reconstructed data.
        """
        X = np.asarray(X, dtype=np.float64)
        return X @ self.components_ + self.mean_
    
    def get_reconstruction_error(self, X: ArrayLike2D) -> float:
        """
        Compute reconstruction error.
        
        Args:
            X: Original data.
        
        Returns:
            float: Mean squared reconstruction error.
        """
        X = np.asarray(X, dtype=np.float64)
        X_transformed = self.transform(X)
        X_reconstructed = self.inverse_transform(X_transformed)
        return float(np.mean((X - X_reconstructed) ** 2))


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("ML Algorithms Module - Demonstration")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Linear Regression
    print("\n1. Linear Regression:")
    X_reg = np.random.randn(100, 3)
    y_reg = 2 * X_reg[:, 0] + 3 * X_reg[:, 1] - 1.5 * X_reg[:, 2] + np.random.randn(100) * 0.5
    lr = LinearRegression(learning_rate=0.1, n_iterations=1000)
    lr.fit(X_reg, y_reg)
    print(f"   Weights: {lr.weights}")
    print(f"   Bias: {lr.bias:.4f}")
    print(f"   R² Score: {lr.score(X_reg, y_reg):.4f}")
    
    # Logistic Regression
    print("\n2. Logistic Regression:")
    X_clf = np.random.randn(200, 2)
    y_clf = (X_clf[:, 0] + X_clf[:, 1] > 0).astype(int)
    logreg = LogisticRegression(learning_rate=0.1, n_iterations=1000)
    logreg.fit(X_clf, y_clf)
    print(f"   Accuracy: {logreg.score(X_clf, y_clf):.4f}")
    
    # Decision Tree
    print("\n3. Decision Tree Classifier:")
    tree = DecisionTreeClassifier(max_depth=5, random_state=42)
    tree.fit(X_clf, y_clf)
    print(f"   Accuracy: {tree.score(X_clf, y_clf):.4f}")
    
    # Random Forest
    print("\n4. Random Forest Classifier:")
    rf = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
    rf.fit(X_clf, y_clf)
    print(f"   Accuracy: {rf.score(X_clf, y_clf):.4f}")
    
    # K-Means
    print("\n5. K-Means Clustering:")
    X_cluster = np.vstack([
        np.random.randn(100, 2) + [2, 2],
        np.random.randn(100, 2) + [-2, -2],
        np.random.randn(100, 2) + [2, -2]
    ])
    kmeans = KMeans(n_clusters=3, random_state=42)
    labels = kmeans.fit_predict(X_cluster)
    print(f"   Inertia: {kmeans.inertia_:.4f}")
    print(f"   Iterations: {kmeans.n_iterations_}")
    print(f"   Cluster centers:\n{kmeans.cluster_centers_}")
    
    # PCA
    print("\n6. Principal Component Analysis:")
    X_pca = np.random.randn(100, 10)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X_pca)
    print(f"   Original shape: {X_pca.shape}")
    print(f"   Reduced shape: {X_reduced.shape}")
    print(f"   Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"   Total explained variance: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    print("\n" + "=" * 60)
