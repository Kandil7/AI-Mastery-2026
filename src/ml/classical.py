"""
Classical Machine Learning Algorithms
=====================================
From-scratch implementations following the White-Box Approach.

Algorithms:
- Linear Regression (closed-form & gradient descent)
- Logistic Regression (binary & multiclass)
- K-Nearest Neighbors
- Decision Trees (ID3/C4.5/CART)
- Support Vector Machines (with kernel trick)
- Random Forest & Gradient Boosting

Author: AI-Mastery-2026
"""

import numpy as np
from typing import List, Optional, Tuple, Union, Callable
from collections import Counter
from abc import ABC, abstractmethod
try:
    from src.core.math_operations import sigmoid, softmax
    from src.core.optimization import SGD, Adam, l2_regularization
except ImportError:
    # Try relative import if src is not in path but running as package
    from ..core.math_operations import sigmoid, softmax
    from ..core.optimization import SGD, Adam, l2_regularization


# ============================================================
# BASE CLASSES
# ============================================================

class BaseEstimator(ABC):
    """Abstract base class for all estimators."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseEstimator':
        """Fit model to data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass


class BaseClassifier(BaseEstimator):
    """Base class for classifiers."""
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute accuracy score."""
        predictions = self.predict(X)
        return np.mean(predictions == y)


class BaseRegressor(BaseEstimator):
    """Base class for regressors."""
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute R² score."""
        predictions = self.predict(X)
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot)


# ============================================================
# LINEAR REGRESSION
# ============================================================

class LinearRegressionScratch(BaseRegressor):
    """
    Linear Regression from scratch.
    
    Model: ŷ = Xw + b
    Loss: MSE = (1/n) Σ(yᵢ - ŷᵢ)²
    
    Methods:
        - Closed-form (Normal Equation): w = (XᵀX)⁻¹Xᵀy
        - Gradient Descent: w = w - α∇L
    
    Mathematical Derivation:
        ∂L/∂w = (2/n) Xᵀ(Xw - y)
    
    Args:
        method: 'closed_form' or 'gradient_descent'
        learning_rate: Step size for gradient descent
        n_iterations: Number of gradient descent iterations
        regularization: 'l1', 'l2', or None
        reg_lambda: Regularization strength
    
    Example:
        >>> model = LinearRegressionScratch(method='gradient_descent')
        >>> model.fit(X_train, y_train)
        >>> predictions = model.predict(X_test)
    """
    
    def __init__(self, method: str = 'closed_form',
                 learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 regularization: Optional[str] = None,
                 reg_lambda: float = 0.01):
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.weights = None
        self.bias = None
        self.loss_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegressionScratch':
        """
        Fit linear regression model.
        
        Args:
            X: Features (n_samples, n_features)
            y: Target (n_samples,)
        
        Returns:
            Self for chaining
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        
        if self.method == 'closed_form':
            self._fit_closed_form(X, y)
        else:
            self._fit_gradient_descent(X, y)
        
        return self
    
    def _fit_closed_form(self, X: np.ndarray, y: np.ndarray):
        """
        Closed-form solution using Normal Equation.
        
        w = (XᵀX)⁻¹Xᵀy
        
        Note: Computationally expensive for large n_features (O(n³))
        """
        # Add bias column
        X_b = np.column_stack([np.ones(X.shape[0]), X])
        
        if self.regularization == 'l2':
            # Ridge: (XᵀX + λI)⁻¹Xᵀy
            n_features = X_b.shape[1]
            I = np.eye(n_features)
            I[0, 0] = 0  # Don't regularize bias
            theta = np.linalg.inv(X_b.T @ X_b + self.reg_lambda * I) @ X_b.T @ y
        else:
            # Standard OLS
            theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        self.bias = theta[0]
        self.weights = theta[1:]
    
    def _fit_gradient_descent(self, X: np.ndarray, y: np.ndarray):
        """
        Gradient descent optimization.
        
        Gradient: ∂L/∂w = (2/n) Xᵀ(Xw - y)
        """
        n_samples, n_features = X.shape
        
        # Initialize
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        
        for i in range(self.n_iterations):
            # Forward pass
            y_pred = X @ self.weights + self.bias
            
            # Compute loss
            loss = np.mean((y_pred - y) ** 2)
            
            # Add regularization to loss
            if self.regularization == 'l2':
                loss += self.reg_lambda * np.sum(self.weights ** 2)
            elif self.regularization == 'l1':
                loss += self.reg_lambda * np.sum(np.abs(self.weights))
            
            self.loss_history.append(loss)
            
            # Compute gradients
            dw = (2 / n_samples) * X.T @ (y_pred - y)
            db = (2 / n_samples) * np.sum(y_pred - y)
            
            # Add regularization gradient
            if self.regularization == 'l2':
                dw += 2 * self.reg_lambda * self.weights
            elif self.regularization == 'l1':
                dw += self.reg_lambda * np.sign(self.weights)
            
            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values."""
        X = np.asarray(X)
        return X @ self.weights + self.bias


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

class LogisticRegressionScratch(BaseClassifier):
    """
    Logistic Regression from scratch.
    
    Model: P(y=1|x) = σ(xᵀw + b) = 1 / (1 + e^{-(xᵀw + b)})
    
    Loss (Binary Cross-Entropy):
        L = -(1/n) Σ[yᵢ log(p̂ᵢ) + (1-yᵢ) log(1-p̂ᵢ)]
    
    Gradient:
        ∂L/∂w = (1/n) Xᵀ(σ(Xw) - y)
    
    Args:
        learning_rate: Step size
        n_iterations: Training iterations
        regularization: 'l1', 'l2', or None
        reg_lambda: Regularization strength
        multiclass: 'ovr' (one-vs-rest) or 'softmax'
    
    Example:
        >>> model = LogisticRegressionScratch(regularization='l2')
        >>> model.fit(X_train, y_train)
        >>> probs = model.predict_proba(X_test)
    """
    
    def __init__(self, learning_rate: float = 0.01,
                 n_iterations: int = 1000,
                 regularization: Optional[str] = 'l2',
                 reg_lambda: float = 0.01,
                 multiclass: str = 'ovr'):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.reg_lambda = reg_lambda
        self.multiclass = multiclass
        self.weights = None
        self.bias = None
        self.classes_ = None
        self.loss_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionScratch':
        """Fit logistic regression model."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        if n_classes == 2:
            self._fit_binary(X, y)
        elif self.multiclass == 'softmax':
            self._fit_softmax(X, y)
        else:
            self._fit_ovr(X, y)
        
        return self
    
    def _fit_binary(self, X: np.ndarray, y: np.ndarray):
        """Binary logistic regression."""
        n_samples, n_features = X.shape
        
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []
        
        for i in range(self.n_iterations):
            # Forward
            z = X @ self.weights + self.bias
            y_pred = sigmoid(z)
            
            # Clip for numerical stability
            y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
            
            # Binary cross-entropy loss
            loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
            
            # Add regularization
            if self.regularization == 'l2':
                loss += self.reg_lambda * np.sum(self.weights ** 2)
            
            self.loss_history.append(loss)
            
            # Gradients
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Regularization gradient
            if self.regularization == 'l2':
                dw += 2 * self.reg_lambda * self.weights
            
            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def _fit_softmax(self, X: np.ndarray, y: np.ndarray):
        """Softmax (multinomial) logistic regression."""
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        
        # One-hot encode y
        y_onehot = np.zeros((n_samples, n_classes))
        for i, c in enumerate(self.classes_):
            y_onehot[y == c, i] = 1
        
        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        self.loss_history = []
        
        for i in range(self.n_iterations):
            # Forward
            z = X @ self.weights + self.bias
            y_pred = softmax(z)
            
            # Cross-entropy loss
            loss = -np.mean(np.sum(y_onehot * np.log(y_pred + 1e-15), axis=1))
            
            if self.regularization == 'l2':
                loss += self.reg_lambda * np.sum(self.weights ** 2)
            
            self.loss_history.append(loss)
            
            # Gradients
            dw = (1 / n_samples) * X.T @ (y_pred - y_onehot)
            db = (1 / n_samples) * np.sum(y_pred - y_onehot, axis=0)
            
            if self.regularization == 'l2':
                dw += 2 * self.reg_lambda * self.weights
            
            # Update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def _fit_ovr(self, X: np.ndarray, y: np.ndarray):
        """One-vs-Rest multiclass strategy."""
        n_classes = len(self.classes_)
        self.classifiers_ = []
        
        for c in self.classes_:
            # Binary classification for class c vs rest
            y_binary = (y == c).astype(int)
            clf = LogisticRegressionScratch(
                learning_rate=self.learning_rate,
                n_iterations=self.n_iterations,
                regularization=self.regularization,
                reg_lambda=self.reg_lambda
            )
            clf._fit_binary(X, y_binary)
            self.classifiers_.append(clf)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        X = np.asarray(X)
        
        if len(self.classes_) == 2:
            prob_positive = sigmoid(X @ self.weights + self.bias)
            return np.column_stack([1 - prob_positive, prob_positive])
        elif self.multiclass == 'softmax':
            z = X @ self.weights + self.bias
            return softmax(z)
        else:
            # OvR: combine probabilities
            probs = np.column_stack([
                sigmoid(X @ clf.weights + clf.bias) 
                for clf in self.classifiers_
            ])
            return probs / probs.sum(axis=1, keepdims=True)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        probs = self.predict_proba(X)
        indices = np.argmax(probs, axis=1)
        return self.classes_[indices]


# ============================================================
# K-NEAREST NEIGHBORS
# ============================================================

class KNNScratch(BaseClassifier):
    """
    K-Nearest Neighbors from scratch.
    
    Algorithm:
        1. Store all training data
        2. For new point, find k closest neighbors
        3. Predict by majority vote (classification) or mean (regression)
    
    Time Complexity: O(n) per prediction (naive implementation)
    
    Args:
        k: Number of neighbors
        metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        weights: 'uniform' or 'distance' (inverse distance weighting)
    
    Note:
        - Lazy learner: no training, all computation at prediction
        - Suffers from "curse of dimensionality" in high dimensions
    
    Example:
        >>> knn = KNNScratch(k=5, metric='euclidean')
        >>> knn.fit(X_train, y_train)
        >>> predictions = knn.predict(X_test)
    """
    
    def __init__(self, k: int = 5, metric: str = 'euclidean',
                 weights: str = 'uniform'):
        self.k = k
        self.metric = metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KNNScratch':
        """Store training data (lazy learning)."""
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y)
        self.classes_ = np.unique(y)
        return self
    
    def _distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute distance between two points."""
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'cosine':
            dot = np.dot(x1, x2)
            norm = np.linalg.norm(x1) * np.linalg.norm(x2)
            return 1 - (dot / norm) if norm != 0 else 1
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def _predict_single(self, x: np.ndarray) -> int:
        """Predict for a single sample."""
        # Compute distances to all training points
        distances = np.array([self._distance(x, x_train) 
                              for x_train in self.X_train])
        
        # Get k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_labels = self.y_train[k_indices]
        k_distances = distances[k_indices]
        
        if self.weights == 'uniform':
            # Simple majority vote
            counts = Counter(k_labels)
            return counts.most_common(1)[0][0]
        else:
            # Distance-weighted voting
            weights = 1 / (k_distances + 1e-8)  # Avoid division by zero
            class_weights = {c: 0 for c in self.classes_}
            
            for label, weight in zip(k_labels, weights):
                class_weights[label] += weight
            
            return max(class_weights.keys(), key=lambda c: class_weights[c])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        X = np.asarray(X)
        return np.array([self._predict_single(x) for x in X])


# ============================================================
# DECISION TREE
# ============================================================

class DecisionTreeNode:
    """Node in a decision tree."""
    
    def __init__(self, feature_index: Optional[int] = None,
                 threshold: Optional[float] = None,
                 left: Optional['DecisionTreeNode'] = None,
                 right: Optional['DecisionTreeNode'] = None,
                 value: Optional[int] = None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # Leaf node class
    
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeScratch(BaseClassifier):
    """
    Decision Tree Classifier from scratch.
    
    Splitting Criteria:
        - Gini Impurity: G = 1 - Σpᵢ²
        - Entropy: H = -Σpᵢ log(pᵢ)
        - Information Gain: IG = H(parent) - Σ(nⱼ/n)H(childⱼ)
    
    Args:
        max_depth: Maximum tree depth
        min_samples_split: Minimum samples to split a node
        criterion: 'gini' or 'entropy'
    
    Example:
        >>> tree = DecisionTreeScratch(max_depth=5, criterion='gini')
        >>> tree.fit(X_train, y_train)
        >>> predictions = tree.predict(X_test)
    """
    
    def __init__(self, max_depth: int = 10,
                 min_samples_split: int = 2,
                 criterion: str = 'gini'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        self.classes_ = None
        self.n_features_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTreeScratch':
        """Build decision tree."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.classes_ = np.unique(y)
        self.n_features_ = X.shape[1]
        self.root = self._grow_tree(X, y, depth=0)
        return self
    
    def _gini(self, y: np.ndarray) -> float:
        """
        Gini Impurity: G = 1 - Σpᵢ²
        
        Range: [0, 0.5] for binary, [0, 1-1/n] for n classes
        0 = pure node
        """
        if len(y) == 0:
            return 0
        
        counts = np.bincount(y.astype(int))
        probs = counts / len(y)
        return 1 - np.sum(probs ** 2)
    
    def _entropy(self, y: np.ndarray) -> float:
        """
        Entropy: H = -Σpᵢ log₂(pᵢ)
        
        Range: [0, log₂(n)] for n classes
        0 = pure node
        """
        if len(y) == 0:
            return 0
        
        counts = np.bincount(y.astype(int))
        probs = counts[counts > 0] / len(y)
        return -np.sum(probs * np.log2(probs))
    
    def _impurity(self, y: np.ndarray) -> float:
        """Compute impurity based on criterion."""
        if self.criterion == 'gini':
            return self._gini(y)
        else:
            return self._entropy(y)
    
    def _information_gain(self, y: np.ndarray, y_left: np.ndarray, 
                          y_right: np.ndarray) -> float:
        """
        Information Gain = H(parent) - weighted_avg(H(children))
        """
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n_left == 0 or n_right == 0:
            return 0
        
        parent_impurity = self._impurity(y)
        child_impurity = (n_left / n) * self._impurity(y_left) + \
                         (n_right / n) * self._impurity(y_right)
        
        return parent_impurity - child_impurity
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float]]:
        """Find best feature and threshold for splitting."""
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        for feature in range(self.n_features_):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _grow_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> DecisionTreeNode:
        """Recursively grow tree."""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            # Create leaf node with majority class
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        
        # Find best split
        feature, threshold = self._best_split(X, y)
        
        if feature is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return DecisionTreeNode(value=leaf_value)
        
        # Split data
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        
        # Grow children
        left = self._grow_tree(X[left_mask], y[left_mask], depth + 1)
        right = self._grow_tree(X[right_mask], y[right_mask], depth + 1)
        
        return DecisionTreeNode(
            feature_index=feature,
            threshold=threshold,
            left=left,
            right=right
        )
    
    def _predict_single(self, x: np.ndarray, node: DecisionTreeNode) -> int:
        """Traverse tree for single prediction."""
        if node.is_leaf():
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        X = np.asarray(X)
        return np.array([self._predict_single(x, self.root) for x in X])


# ============================================================
# RANDOM FOREST
# ============================================================

class RandomForestScratch(BaseClassifier):
    """
    Random Forest Classifier from scratch.
    
    Ensemble method combining:
        1. Bootstrap aggregating (Bagging)
        2. Random feature selection at each split
    
    Args:
        n_estimators: Number of trees
        max_depth: Maximum depth per tree
        max_features: Features to consider at each split ('sqrt', 'log2', int)
        bootstrap: Whether to use bootstrap sampling
    
    Example:
        >>> rf = RandomForestScratch(n_estimators=100, max_depth=10)
        >>> rf.fit(X_train, y_train)
        >>> predictions = rf.predict(X_test)
    """
    
    def __init__(self, n_estimators: int = 100,
                 max_depth: int = 10,
                 min_samples_split: int = 2,
                 max_features: Union[str, int] = 'sqrt',
                 bootstrap: bool = True):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.trees = []
        self.feature_indices = []
        self.classes_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestScratch':
        """Train random forest."""
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        
        # Determine number of features per split
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        else:
            max_features = min(self.max_features, n_features)
        
        self.trees = []
        self.feature_indices = []
        
        for _ in range(self.n_estimators):
            # Bootstrap sample
            if self.bootstrap:
                indices = np.random.choice(n_samples, size=n_samples, replace=True)
                X_sample = X[indices]
                y_sample = y[indices]
            else:
                X_sample = X
                y_sample = y
            
            # Random feature subset
            feature_idx = np.random.choice(n_features, size=max_features, replace=False)
            X_subset = X_sample[:, feature_idx]
            
            # Train tree
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_subset, y_sample)
            
            self.trees.append(tree)
            self.feature_indices.append(feature_idx)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict by majority vote."""
        X = np.asarray(X)
        
        # Collect predictions from all trees
        all_preds = np.array([
            tree.predict(X[:, feature_idx])
            for tree, feature_idx in zip(self.trees, self.feature_indices)
        ])  # Shape: (n_estimators, n_samples)
        
        # Majority vote
        predictions = np.array([
            Counter(all_preds[:, i]).most_common(1)[0][0]
            for i in range(X.shape[0])
        ])
        
        return predictions


# ============================================================
# NAIVE BAYES
# ============================================================

class GaussianNBScratch(BaseClassifier):
    """
    Gaussian Naive Bayes from scratch.
    
    Assumption: Features are independent and normally distributed.
    
    Bayes' Theorem:
        P(y|X) ∝ P(X|y) × P(y)
        P(X|y) = ∏ P(xᵢ|y) = ∏ N(xᵢ; μᵢ, σᵢ²)
    
    Example:
        >>> nb = GaussianNBScratch()
        >>> nb.fit(X_train, y_train)
        >>> predictions = nb.predict(X_test)
    """
    
    def __init__(self):
        self.classes_ = None
        self.class_priors_ = None
        self.means_ = None
        self.variances_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianNBScratch':
        """Compute class priors and Gaussian parameters."""
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        # Initialize
        self.class_priors_ = np.zeros(n_classes)
        self.means_ = np.zeros((n_classes, n_features))
        self.variances_ = np.zeros((n_classes, n_features))
        
        for i, c in enumerate(self.classes_):
            X_c = X[y == c]
            self.class_priors_[i] = len(X_c) / n_samples
            self.means_[i] = np.mean(X_c, axis=0)
            self.variances_[i] = np.var(X_c, axis=0) + 1e-9  # Add epsilon for stability
        
        return self
    
    def _gaussian_pdf(self, x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
        """Gaussian probability density function."""
        return np.exp(-0.5 * ((x - mean) ** 2) / var) / np.sqrt(2 * np.pi * var)
    
    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict log probabilities."""
        X = np.asarray(X)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        log_probs = np.zeros((n_samples, n_classes))
        
        for i, c in enumerate(self.classes_):
            # Log prior
            log_prior = np.log(self.class_priors_[i])
            
            # Log likelihood (sum of log Gaussians)
            log_likelihood = np.sum(
                np.log(self._gaussian_pdf(X, self.means_[i], self.variances_[i])),
                axis=1
            )
            
            log_probs[:, i] = log_prior + log_likelihood
        
        return log_probs
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        log_probs = self.predict_log_proba(X)
        indices = np.argmax(log_probs, axis=1)
        return self.classes_[indices]


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Base
    'BaseEstimator', 'BaseClassifier', 'BaseRegressor',
    # Regression
    'LinearRegressionScratch',
    # Classification
    'LogisticRegressionScratch', 'KNNScratch', 'DecisionTreeScratch',
    'RandomForestScratch', 'GaussianNBScratch',
]
