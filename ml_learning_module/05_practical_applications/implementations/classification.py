"""
Classification Pipeline Implementation
======================================

Complete classification pipeline with multiple algorithms
and evaluation metrics.

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """Store classification results"""

    predictions: np.ndarray
    probabilities: Optional[np.ndarray]
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: np.ndarray
    classification_report: Dict


class LogisticRegressionClassifier:
    """
    Logistic Regression for Binary Classification

    Uses sigmoid function to output probability:
        P(y=1|x) = 1 / (1 + e^(-z))
        where z = w·x + b

    Trained using gradient descent on cross-entropy loss.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 1000,
        regularization: float = 0.0,
    ):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.regularization = regularization
        self.weights = None
        self.bias = None
        self.loss_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid"""
        return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionClassifier":
        """Train the classifier"""
        n_samples, n_features = X.shape

        # Initialize weights
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for epoch in range(self.n_epochs):
            # Forward pass
            linear = np.dot(X, self.weights) + self.bias
            predictions = self._sigmoid(linear)

            # Compute loss (cross-entropy with L2 regularization)
            eps = 1e-15
            predictions = np.clip(predictions, eps, 1 - eps)
            loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            loss += self.regularization * np.sum(self.weights**2) / 2
            self.loss_history.append(loss)

            # Compute gradients
            dw = np.dot(X.T, (predictions - y)) / n_samples
            db = np.mean(predictions - y)

            # Add regularization
            dw += self.regularization * self.weights

            # Update weights
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities"""
        linear = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict class labels"""
        return (self.predict_proba(X) >= threshold).astype(int)


class DecisionTreeClassifier:
    """
    Decision Tree Classifier

    Recursively splits data based on feature thresholds
    to maximize information gain (or minimize Gini impurity).
    """

    def __init__(
        self, max_depth: int = 10, min_samples_split: int = 2, min_samples_leaf: int = 1
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None

    def _gini_impurity(self, y: np.ndarray) -> float:
        """Compute Gini impurity"""
        if len(y) == 0:
            return 0

        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities**2)

    def _information_gain(
        self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray
    ) -> float:
        """Compute information gain"""
        n = len(y)
        if n == 0:
            return 0

        parent_impurity = self._gini_impurity(y)
        n_left, n_right = len(y_left), len(y_right)

        if n_left == 0 or n_right == 0:
            return 0

        child_impurity = (n_left / n) * self._gini_impurity(y_left) + (
            n_right / n
        ) * self._gini_impurity(y_right)

        return parent_impurity - child_impurity

    def _find_best_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[int, float, float]:
        """Find best feature and threshold for split"""
        best_gain = 0
        best_feature = 0
        best_threshold = 0

        n_features = X.shape[1]

        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])

            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = ~left_mask

                if (
                    np.sum(left_mask) < self.min_samples_leaf
                    or np.sum(right_mask) < self.min_samples_leaf
                ):
                    continue

                gain = self._information_gain(y, y[left_mask], y[right_mask])

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0):
        """Recursively build the tree"""
        n_samples = len(y)

        # Stopping conditions
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            # Return leaf with most common class
            return {"class": np.bincount(y).argmax()}

        # Find best split
        feature, threshold, gain = self._find_best_split(X, y)

        if gain == 0:
            return {"class": np.bincount(y).argmax()}

        # Split
        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Build children
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_tree,
            "right": right_tree,
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Build the decision tree"""
        self.tree = self._build_tree(X, y)
        return self

    def _predict_sample(self, x: np.ndarray, node: dict) -> int:
        """Predict for a single sample"""
        if "class" in node:
            return node["class"]

        if x[node["feature"]] <= node["threshold"]:
            return self._predict_sample(x, node["left"])
        else:
            return self._predict_sample(x, node["right"])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels"""
        return np.array([self._predict_sample(x, self.tree) for x in X])


class RandomForestClassifier:
    """
    Random Forest Classifier

    Ensemble of decision trees using bootstrap aggregation (bagging)
    and random feature selection at each split.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        min_samples_split: int = 2,
        n_features: Optional[int] = None,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestClassifier":
        """Build the forest"""
        np.random.seed(42)
        n_samples = X.shape[0]

        if self.n_features is None:
            self.n_features = int(np.sqrt(X.shape[1]))

        # Train multiple trees
        for i in range(self.n_estimators):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]

            # Train tree
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, min_samples_split=self.min_samples_split
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using majority vote"""
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])

        # Majority vote
        most_common = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
        )

        return most_common


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray
) -> ClassificationResult:
    """Evaluate classification results"""

    # Basic metrics
    accuracy = np.mean(y_true == y_pred)

    # Confusion matrix
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1

    # Per-class metrics
    precision = []
    recall = []
    f1_scores = []

    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        precision.append(prec)
        recall.append(rec)
        f1_scores.append(f1)

    # Averages
    avg_precision = np.mean(precision)
    avg_recall = np.mean(recall)
    avg_f1 = np.mean(f1_scores)

    return ClassificationResult(
        predictions=y_pred,
        probabilities=None,
        accuracy=accuracy,
        precision=avg_precision,
        recall=avg_recall,
        f1_score=avg_f1,
        confusion_matrix=cm,
        classification_report={
            "precision": precision,
            "recall": recall,
            "f1": f1_scores,
            "support": np.bincount(y_true, minlength=n_classes).tolist(),
        },
    )


def demo_classification():
    """Demonstrate classification algorithms"""
    print("=" * 60)
    print("Classification Demo")
    print("=" * 60)

    # Create sample data
    np.random.seed(42)
    n_samples = 200

    # Class 0
    X0 = np.random.randn(n_samples, 2) + np.array([-2, -2])
    y0 = np.zeros(n_samples)

    # Class 1
    X1 = np.random.randn(n_samples, 2) + np.array([2, 2])
    y1 = np.ones(n_samples)

    # Combine
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    # Shuffle
    indices = np.random.permutation(2 * n_samples)
    X = X[indices]
    y = y[indices]

    # Train/test split
    split = int(0.8 * len(y))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"Training samples: {len(y_train)}")
    print(f"Test samples: {len(y_test)}")

    # ========================================
    # Demo 1: Logistic Regression
    # ========================================
    print("\n--- Logistic Regression ---")

    lr = LogisticRegressionClassifier(learning_rate=0.1, n_epochs=1000)
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    result_lr = evaluate_classification(y_test, y_pred_lr)

    print(f"Accuracy: {result_lr.accuracy:.4f}")
    print(f"Precision: {result_lr.precision:.4f}")
    print(f"Recall: {result_lr.recall:.4f}")
    print(f"F1 Score: {result_lr.f1_score:.4f}")
    print(f"Confusion Matrix:\n{result_lr.confusion_matrix}")

    # ========================================
    # Demo 2: Decision Tree
    # ========================================
    print("\n--- Decision Tree ---")

    dt = DecisionTreeClassifier(max_depth=5)
    dt.fit(X_train, y_train)

    y_pred_dt = dt.predict(X_test)
    result_dt = evaluate_classification(y_test, y_pred_dt)

    print(f"Accuracy: {result_dt.accuracy:.4f}")
    print(f"F1 Score: {result_dt.f1_score:.4f}")

    # ========================================
    # Demo 3: Random Forest
    # ========================================
    print("\n--- Random Forest ---")

    rf = RandomForestClassifier(n_estimators=50, max_depth=5)
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    result_rf = evaluate_classification(y_test, y_pred_rf)

    print(f"Accuracy: {result_rf.accuracy:.4f}")
    print(f"F1 Score: {result_rf.f1_score:.4f}")

    # ========================================
    # Summary
    # ========================================
    print("\n--- Summary ---")
    print(f"Logistic Regression: {result_lr.accuracy:.4f}")
    print(f"Decision Tree:       {result_dt.accuracy:.4f}")
    print(f"Random Forest:       {result_rf.accuracy:.4f}")


if __name__ == "__main__":
    demo_classification()

    print("\n" + "=" * 60)
    print("Classification complete!")
    print("=" * 60)
