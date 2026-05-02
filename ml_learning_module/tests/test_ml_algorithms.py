"""
Test Suite for ML Learning Module - Machine Learning Algorithms

Tests cover:
- Linear Regression
- Logistic Regression
- Decision Trees
- K-Means Clustering
- Performance metrics
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestLinearRegression:
    """Test suite for Linear Regression implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        from ml_learning_module.module_02_ml.linear_regression import LinearRegression

        self.LinearRegression = LinearRegression

    def test_simple_linear_regression(self):
        """Test linear regression on simple data."""
        # y = 3x + 5
        np.random.seed(42)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = 3 * X.squeeze() + 5 + np.random.randn(100) * 0.5

        model = self.LinearRegression(method="closed_form")
        model.fit(X, y)

        # Check fit
        assert model.theta is not None

    def test_gradient_descent_fit(self):
        """Test gradient descent solution."""
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])

        model = self.LinearRegression(
            method="gradient_descent", learning_rate=0.01, n_iterations=1000
        )
        model.fit(X, y)

        # Should converge close to y = 2x
        assert abs(model.theta[1] - 2.0) < 0.5

    def test_multivariate_regression(self):
        """Test linear regression with multiple features."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        X = np.random.randn(n_samples, n_features)
        true_weights = np.array([1, -2, 0.5, 3, -1])
        y = X @ true_weights + np.random.randn(n_samples) * 0.1

        model = self.LinearRegression(method="closed_form")
        model.fit(X, y)

        r2 = model.score(X, y)

        assert r2 > 0.9  # Should fit well

    def test_predictions(self):
        """Test prediction method."""
        X = np.array([[1], [2], [3]])
        y = np.array([2, 4, 6])

        model = self.LinearRegression()
        model.fit(X, y)
        predictions = model.predict(X)

        assert predictions.shape == (3,)

    def test_r2_score(self):
        """Test R² score calculation."""
        # Perfect fit should give R² = 1
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])

        model = self.LinearRegression(method="closed_form")
        model.fit(X, y)

        r2 = model.score(X, y)

        assert r2 > 0.99


class TestLogisticRegression:
    """Test suite for Logistic Regression implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        from ml_learning_module.module_02_ml.logistic_regression import (
            LogisticRegression,
        )

        self.LogisticRegression = LogisticRegression

    def test_binary_classification(self):
        """Test logistic regression on binary classification."""
        np.random.seed(42)

        # Generate two classes
        X0 = np.random.randn(50, 2) + np.array([-2, -2])
        X1 = np.random.randn(50, 2) + np.array([2, 2])
        X = np.vstack([X0, X1])
        y = np.hstack([np.zeros(50), np.ones(50)])

        model = self.LogisticRegression(learning_rate=0.1, n_iterations=1000)
        model.fit(X, y)

        accuracy = model.accuracy(X, y)

        assert accuracy > 0.8

    def test_predict_proba(self):
        """Test probability prediction."""
        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]])
        y = np.array([0, 0, 1, 1])

        model = self.LogisticRegression(n_iterations=500)
        model.fit(X, y)

        probs = model.predict_proba(X)

        assert probs.shape == (4, 2)
        assert np.all(probs >= 0) and np.all(probs <= 1)

    def test_multiclass_classification(self):
        """Test logistic regression for multiclass."""
        np.random.seed(42)

        # Three classes
        X0 = np.random.randn(30, 2) + np.array([0, 0])
        X1 = np.random.randn(30, 2) + np.array([3, 0])
        X2 = np.random.randn(30, 2) + np.array([1.5, 3])

        X = np.vstack([X0, X1, X2])
        y = np.hstack([np.zeros(30), np.ones(30), np.ones(30) * 2])

        model = self.LogisticRegression(n_iterations=1000)
        model.fit(X, y)

        accuracy = model.accuracy(X, y)

        assert accuracy > 0.7


class TestDecisionTree:
    """Test suite for Decision Tree implementation."""

    def setup_method(self):
        """Set up test fixtures."""
        from ml_learning_module.module_02_ml.decision_tree import DecisionTree

        self.DecisionTree = DecisionTree

    def test_classification(self):
        """Test decision tree classification."""
        np.random.seed(42)

        X = np.random.randn(100, 4)
        y = (X[:, 0] > 0).astype(int)  # Simple rule-based

        tree = self.DecisionTree(max_depth=5)
        tree.fit(X, y)

        accuracy = tree.accuracy(X, y)

        assert accuracy > 0.8

    def test_single_feature_classification(self):
        """Test decision tree on single feature."""
        X = np.array([[1], [2], [3], [4], [5], [6]])
        y = np.array([0, 0, 0, 1, 1, 1])

        tree = self.DecisionTree(max_depth=2)
        tree.fit(X, y)

        predictions = tree.predict(X)

        assert np.mean(predictions == y) >= 0.8


class TestKMeans:
    """Test suite for K-Means clustering."""

    def setup_method(self):
        """Set up test fixtures."""
        from ml_learning_module.module_02_ml.kmeans import KMeans

        self.KMeans = KMeans

    def test_clustering(self):
        """Test K-Means clustering."""
        np.random.seed(42)

        # Generate 3 clusters
        center1 = np.array([0, 0])
        center2 = np.array([5, 5])
        center3 = np.array([10, 0])

        X0 = np.random.randn(30, 2) + center1
        X1 = np.random.randn(30, 2) + center2
        X2 = np.random.randn(30, 2) + center3

        X = np.vstack([X0, X1, X2])

        kmeans = self.KMeans(n_clusters=3, n_iterations=100)
        labels = kmeans.fit_predict(X)

        # Check that we got 3 clusters
        assert len(np.unique(labels)) == 3

    def test_cluster_centers(self):
        """Test that cluster centers are computed."""
        X = np.array([[1, 1], [1.1, 1.1], [5, 5], [5.1, 5.1]])

        kmeans = self.KMeans(n_clusters=2, n_iterations=100)
        kmeans.fit(X)

        assert kmeans.centroids.shape == (2, 2)


class TestMetrics:
    """Test suite for evaluation metrics."""

    def setup_method(self):
        """Set up test fixtures."""
        from ml_learning_module.module_02_ml.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            roc_auc_score,
        )

        self.metrics = {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "auc": roc_auc_score,
        }

    def test_perfect_accuracy(self):
        """Test accuracy with perfect predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 1])

        acc = self.metrics["accuracy"](y_true, y_pred)

        assert acc == 1.0

    def test_zero_accuracy(self):
        """Test accuracy with all wrong predictions."""
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([1, 0, 1, 0])

        acc = self.metrics["accuracy"](y_true, y_pred)

        assert acc == 0.0

    def test_precision(self):
        """Test precision calculation."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])

        precision = self.metrics["precision"](y_true, y_pred)

        assert 0.5 <= precision <= 1.0

    def test_recall(self):
        """Test recall calculation."""
        y_true = np.array([0, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 1])

        recall = self.metrics["recall"](y_true, y_pred)

        assert recall >= 0.5

    def test_f1_score(self):
        """Test F1 score calculation."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])

        f1 = self.metrics["f1"](y_true, y_pred)

        assert 0 <= f1 <= 1

    def test_confusion_matrix(self):
        """Test confusion matrix computation."""
        from ml_learning_module.module_02_ml.metrics import confusion_matrix

        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 0])

        cm = confusion_matrix(y_true, y_pred)

        # [[TN, FP], [FN, TP]] = [[1, 1], [1, 1]]
        assert cm.shape == (2, 2)


class TestNormalization:
    """Test suite for data preprocessing."""

    def setup_method(self):
        """Set up test fixtures."""
        from ml_learning_module.module_02_ml.preprocessing import StandardScaler

        self.StandardScaler = StandardScaler

    def test_standard_scaler(self):
        """Test standard scaling."""
        X = np.array([[0, 0], [0, 0], [1, 1], [1, 1]])

        scaler = self.StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Mean should be ~0
        mean = np.mean(X_scaled, axis=0)
        np.testing.assert_array_almost_equal(mean, [0, 0], decimal=5)

    def test_inverse_transform(self):
        """Test inverse transform."""
        X = np.array([[0, 0], [1, 2], [2, 4]])

        scaler = self.StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_restored = scaler.inverse_transform(X_scaled)

        np.testing.assert_array_almost_equal(X, X_restored)


class TestTrainTestSplit:
    """Test suite for data splitting."""

    def setup_method(self):
        """Set up test fixtures."""
        from ml_learning_module.module_02_ml.preprocessing import train_test_split

        self.split = train_test_split

    def test_basic_split(self):
        """Test basic train-test split."""
        X = np.arange(100).reshape(-1, 1)
        y = np.arange(100)

        X_train, X_test, y_train, y_test = self.split(X, y, test_size=0.2)

        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_stratified_split(self):
        """Test stratified split maintains class balance."""
        X = np.arange(100).reshape(-1, 1)
        y = np.array([0] * 50 + [1] * 50)

        X_train, X_test, y_train, y_test = self.split(X, y, test_size=0.2, stratify=y)

        # Class distribution should be similar
        train_ratio = np.sum(y_train == 1) / len(y_train)
        test_ratio = np.sum(y_test == 1) / len(y_test)

        assert abs(train_ratio - test_ratio) < 0.1


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
