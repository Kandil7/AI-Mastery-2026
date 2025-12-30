# Week 03: Classical Machine Learning

Implementing classic ML algorithms from scratch.

## Learning Objectives
1. Implement linear/logistic regression
2. Build decision trees
3. Understand ensemble methods

---

## 1. Linear Regression

```python
import numpy as np
from typing import Tuple

class LinearRegression:
    """
    Linear Regression using Normal Equation and Gradient Descent.
    
    Model: y = X @ w + b
    Loss: MSE = (1/n) * Σ(y - ŷ)²
    """
    def __init__(self, method: str = 'normal'):
        self.method = method
        self.weights = None
        self.bias = None
    
    def fit_normal_equation(self, X: np.ndarray, y: np.ndarray):
        """
        Closed-form solution: w = (X^T X)^(-1) X^T y
        """
        # Add bias column
        X_b = np.c_[np.ones(X.shape[0]), X]
        
        # Normal equation
        theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        
        self.bias = theta[0]
        self.weights = theta[1:]
    
    def fit_gradient_descent(
        self, X: np.ndarray, y: np.ndarray,
        lr: float = 0.01, epochs: int = 1000
    ):
        """
        Gradient descent optimization.
        
        Gradients:
        ∂L/∂w = -2/n * X^T(y - ŷ)
        ∂L/∂b = -2/n * Σ(y - ŷ)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for _ in range(epochs):
            y_pred = X @ self.weights + self.bias
            error = y - y_pred
            
            self.weights += lr * (2/n_samples) * X.T @ error
            self.bias += lr * (2/n_samples) * np.sum(error)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs):
        if self.method == 'normal':
            self.fit_normal_equation(X, y)
        else:
            self.fit_gradient_descent(X, y, **kwargs)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R² score."""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - ss_res / ss_tot


# Test
np.random.seed(42)
X = np.random.randn(100, 3)
true_weights = np.array([2, -1, 0.5])
y = X @ true_weights + 3 + np.random.randn(100) * 0.1

model = LinearRegression(method='normal')
model.fit(X, y)
print(f"True weights: {true_weights}")
print(f"Learned weights: {model.weights}")
print(f"True bias: 3.0")
print(f"Learned bias: {model.bias:.4f}")
print(f"R² score: {model.score(X, y):.4f}")
```

---

## 2. Logistic Regression

```python
class LogisticRegression:
    """
    Logistic Regression for binary classification.
    
    Model: p(y=1|x) = σ(x @ w + b)
    Loss: Binary Cross-Entropy
    """
    def __init__(self, lr: float = 0.01, epochs: int = 1000):
        self.lr = lr
        self.epochs = epochs
    
    def sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.losses = []
        
        for epoch in range(self.epochs):
            # Forward
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)
            
            # Loss (BCE)
            eps = 1e-10
            loss = -np.mean(y * np.log(y_pred + eps) + (1-y) * np.log(1-y_pred + eps))
            self.losses.append(loss)
            
            # Gradients
            error = y_pred - y
            dw = (1/n_samples) * X.T @ error
            db = (1/n_samples) * np.sum(error)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.sigmoid(X @ self.weights + self.bias)
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)
    
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return np.mean(self.predict(X) == y)


# Test
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=5, random_state=42)

model = LogisticRegression(lr=0.1, epochs=500)
model.fit(X, y)
print(f"Accuracy: {model.accuracy(X, y):.2%}")
```

---

## 3. Decision Trees

```python
class Node:
    """Decision tree node."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # For leaf nodes

class DecisionTree:
    """
    Decision Tree Classifier using Gini impurity.
    
    Gini = 1 - Σ(p_i)²
    """
    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def gini(self, y: np.ndarray) -> float:
        """Calculate Gini impurity."""
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)
    
    def information_gain(self, y: np.ndarray, left_idx: np.ndarray, right_idx: np.ndarray) -> float:
        """Calculate information gain from split."""
        n = len(y)
        n_left = len(left_idx)
        n_right = len(right_idx)
        
        parent_gini = self.gini(y)
        left_gini = self.gini(y[left_idx])
        right_gini = self.gini(y[right_idx])
        
        weighted_child_gini = (n_left/n) * left_gini + (n_right/n) * right_gini
        return parent_gini - weighted_child_gini
    
    def best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
        """Find best feature and threshold for split."""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                left_idx = np.where(X[:, feature] <= threshold)[0]
                right_idx = np.where(X[:, feature] > threshold)[0]
                
                if len(left_idx) == 0 or len(right_idx) == 0:
                    continue
                
                gain = self.information_gain(y, left_idx, right_idx)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Recursively build decision tree."""
        n_samples = len(y)
        n_classes = len(np.unique(y))
        
        # Stopping criteria
        if depth >= self.max_depth or n_classes == 1 or n_samples < self.min_samples_split:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)
        
        # Find best split
        feature, threshold = self.best_split(X, y)
        
        if feature is None:
            leaf_value = np.argmax(np.bincount(y))
            return Node(value=leaf_value)
        
        # Split data
        left_idx = np.where(X[:, feature] <= threshold)[0]
        right_idx = np.where(X[:, feature] > threshold)[0]
        
        # Recurse
        left_child = self.build_tree(X[left_idx], y[left_idx], depth + 1)
        right_child = self.build_tree(X[right_idx], y[right_idx], depth + 1)
        
        return Node(feature=feature, threshold=threshold, left=left_child, right=right_child)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.root = self.build_tree(X, y)
    
    def predict_sample(self, x: np.ndarray, node: Node) -> int:
        if node.value is not None:
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        return self.predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array([self.predict_sample(x, self.root) for x in X])


# Test
X, y = make_classification(n_samples=200, n_features=5, random_state=42)
tree = DecisionTree(max_depth=5)
tree.fit(X, y)
accuracy = np.mean(tree.predict(X) == y)
print(f"Decision Tree Accuracy: {accuracy:.2%}")
```

---

## 4. Random Forest

```python
class RandomForest:
    """
    Random Forest using bagging and feature subsampling.
    """
    def __init__(self, n_estimators: int = 10, max_depth: int = 10, max_features: str = 'sqrt'):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = []
        self.feature_indices = []
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        
        # Determine number of features per tree
        if self.max_features == 'sqrt':
            n_selected = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            n_selected = int(np.log2(n_features))
        else:
            n_selected = n_features
        
        for _ in range(self.n_estimators):
            # Bootstrap sampling
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Feature subsampling
            feat_idx = np.random.choice(n_features, n_selected, replace=False)
            X_sub = X_boot[:, feat_idx]
            
            # Train tree
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sub, y_boot)
            
            self.trees.append(tree)
            self.feature_indices.append(feat_idx)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.zeros((len(X), self.n_estimators))
        
        for i, (tree, feat_idx) in enumerate(zip(self.trees, self.feature_indices)):
            predictions[:, i] = tree.predict(X[:, feat_idx])
        
        # Majority vote
        return np.array([np.argmax(np.bincount(row.astype(int))) for row in predictions])


# Test
rf = RandomForest(n_estimators=10, max_depth=5)
rf.fit(X, y)
accuracy = np.mean(rf.predict(X) == y)
print(f"Random Forest Accuracy: {accuracy:.2%}")
```

---

## 5. K-Nearest Neighbors

```python
class KNN:
    """K-Nearest Neighbors Classifier."""
    
    def __init__(self, k: int = 5):
        self.k = k
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
    
    def euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        
        for x in X:
            distances = [self.euclidean_distance(x, x_train) for x_train in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_labels = self.y_train[k_indices]
            prediction = np.argmax(np.bincount(k_labels))
            predictions.append(prediction)
        
        return np.array(predictions)


knn = KNN(k=5)
knn.fit(X, y)
print(f"KNN Accuracy: {np.mean(knn.predict(X) == y):.2%}")
```

---

## 6. Support Vector Machine (Linear)

```python
class LinearSVM:
    """
    Linear SVM using gradient descent.
    
    Hinge Loss: max(0, 1 - y * (w·x + b))
    """
    def __init__(self, C: float = 1.0, lr: float = 0.001, epochs: int = 1000):
        self.C = C
        self.lr = lr
        self.epochs = epochs
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        
        for _ in range(self.epochs):
            for i, x in enumerate(X):
                margin = y_[i] * (x @ self.weights + self.bias)
                
                if margin >= 1:
                    self.weights -= self.lr * 2 * self.weights / n_samples
                else:
                    self.weights -= self.lr * (2 * self.weights / n_samples - self.C * y_[i] * x)
                    self.bias -= self.lr * (-self.C * y_[i])
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.sign(X @ self.weights + self.bias)


svm = LinearSVM(C=1.0, epochs=500)
svm.fit(X, y)
y_svm = np.where(y <= 0, -1, 1)
print(f"SVM Accuracy: {np.mean(svm.predict(X) == y_svm):.2%}")
```

---

## Summary

| Algorithm | Type | Key Idea |
|-----------|------|----------|
| Linear Regression | Regression | Minimize MSE |
| Logistic Regression | Classification | Sigmoid + BCE |
| Decision Tree | Both | Recursive splitting |
| Random Forest | Both | Ensemble of trees |
| KNN | Both | Distance-based voting |
| SVM | Classification | Maximum margin |
