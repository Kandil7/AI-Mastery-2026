"""
Comprehensive Test Suite for ML Learning Module
================================================

This test file verifies all implementations work correctly,
imports are valid, and modules are properly integrated.

Tests cover:
- Mathematical foundations (vectors, matrices, calculus, probability)
- ML algorithms (linear regression, SVM, ensembles)
- Dataset utilities
- Pipeline utilities
- NLP implementations

Author: ML Learning Module
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestImports:
    """Verify all module imports work correctly."""

    def test_math_imports(self):
        """Test mathematical foundation imports."""
        from implementations.math.vectors import Vector
        from implementations.math.matrices import Matrix
        from implementations.math.calculus import gradient, hessian
        from implementations.math.probability import NormalDistribution

        assert True

    def test_ml_imports(self):
        """Test ML algorithm imports."""
        from implementations.ml.linear_regression import LinearRegression
        from implementations.ml.evaluation import accuracy_score, mean_squared_error
        from implementations.ml.advanced_algorithms import (
            LogisticRegression,
            DecisionTree,
            RandomForest,
            GradientBoosting,
        )

        assert True

    def test_new_ml_imports(self):
        """Test new HOML-style implementations."""
        from implementations.ml.svm import SVM
        from implementations.ml.pipeline import StandardScaler, Pipeline, GridSearchCV
        from implementations.ml.ensembles import (
            VotingClassifier,
            StackingClassifier,
            AdaBoost,
        )

        assert True

    def test_dataset_imports(self):
        """Test dataset utilities imports."""
        from implementations.ml.datasets import (
            make_moons,
            make_circles,
            make_blobs,
            make_classification,
            make_regression,
            train_test_split,
            load_iris,
        )

        assert True

    def test_nlp_imports(self):
        """Test NLP module imports."""
        from implementations.nlp.word_embeddings import Word2Vec
        from implementations.nlp.positional_encoding import PositionalEncoding

        assert True

    def test_deep_learning_imports(self):
        """Test deep learning module imports."""
        from implementations.deep_learning.cnn import CNN
        from implementations.deep_learning.rnn import RNN
        from implementations.deep_learning.lstm import LSTM

        assert True

    def test_section_implementations(self):
        """Test section-specific implementations."""
        # Neural networks
        from implementations.perceptron import Perceptron
        from implementations.mlp import MLP

        # NLP
        from implementations.text_preprocessing import Tokenizer, Vocabulary, BagOfWords

        # Practical applications
        from implementations.classification import KNNClassifier, NaiveBayesClassifier
        from implementations.regression import RidgeRegression
        from implementations.clustering import KMeans
        from implementations.dimension_reduction import PCA

        assert True


class TestDatasets:
    """Test dataset generation utilities."""

    def test_make_moons(self):
        """Test make_moons dataset generation."""
        from implementations.ml.datasets import make_moons

        X, y = make_moons(n_samples=200, noise=0.1, random_state=42)

        assert X.shape == (200, 2)
        assert y.shape == (200,)
        assert len(np.unique(y)) == 2  # Binary classification
        assert np.all(np.isin(y, [0, 1]))

    def test_make_moons_shuffle(self):
        """Test shuffling behavior."""
        from implementations.ml.datasets import make_moons

        X1, y1 = make_moons(n_samples=100, shuffle=False)
        X2, y2 = make_moons(n_samples=100, shuffle=True)

        # Without shuffle, classes should be ordered
        assert np.all(y1[:50] == 0)
        assert np.all(y1[50:] == 1)

    def test_make_circles(self):
        """Test make_circles dataset generation."""
        from implementations.ml.datasets import make_circles

        X, y = make_circles(n_samples=200, noise=0.05, random_state=0)

        assert X.shape == (200, 2)
        assert y.shape == (200,)
        assert len(np.unique(y)) == 2

    def test_make_blobs(self):
        """Test make_blobs dataset generation."""
        from implementations.ml.datasets import make_blobs

        X, y, centers = make_blobs(n_samples=300, centers=3, random_state=42)

        assert X.shape == (300, 2)
        assert y.shape == (300,)
        assert centers.shape == (3, 2)
        assert len(np.unique(y)) == 3

    def test_make_classification(self):
        """Test make_classification dataset generation."""
        from implementations.ml.datasets import make_classification

        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5, n_classes=3, random_state=42
        )

        assert X.shape == (200, 10)
        assert y.shape == (200,)
        assert len(np.unique(y)) == 3

    def test_make_regression(self):
        """Test make_regression dataset generation."""
        from implementations.ml.datasets import make_regression

        X, y = make_regression(n_samples=100, n_features=5, n_informative=3)

        assert X.shape == (100, 5)
        assert y.shape == (100,)

    def test_train_test_split(self):
        """Test train_test_split function."""
        from implementations.ml.datasets import train_test_split

        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        assert X_train.shape == (80, 5)
        assert X_test.shape == (20, 5)
        assert y_train.shape == (80,)
        assert y_test.shape == (20,)

    def test_load_iris(self):
        """Test load_iris dataset."""
        from implementations.ml.datasets import load_iris

        X, y, feature_names, target_names = load_iris()

        assert X.shape == (150, 4)
        assert y.shape == (150,)
        assert len(np.unique(y)) == 3
        assert len(feature_names) == 4
        assert len(target_names) == 3


class TestPipeline:
    """Test pipeline utilities."""

    def test_standard_scaler(self):
        """Test StandardScaler."""
        from implementations.ml.pipeline import StandardScaler

        X = np.array([[1, 2], [3, 4], [5, 6]])

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # After scaling, mean should be ~0, std should be ~1
        assert np.allclose(np.mean(X_scaled, axis=0), [0, 0], atol=1e-10)
        assert np.allclose(np.std(X_scaled, axis=0), [1, 1], atol=1e-10)

    def test_min_max_scaler(self):
        """Test MinMaxScaler."""
        from implementations.ml.pipeline import MinMaxScaler

        X = np.array([[1, 0], [2, 1], [3, 2]])

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        # After scaling, min should be 0, max should be 1
        assert np.allclose(np.min(X_scaled, axis=0), [0, 0])
        assert np.allclose(np.max(X_scaled, axis=0), [1, 1])

    def test_pipeline_integration(self):
        """Test Pipeline integration."""
        from implementations.ml.pipeline import Pipeline, StandardScaler
        from implementations.ml.linear_regression import LinearRegression

        # Create sample data
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = X @ np.array([1, 2, 3]) + 0.1 * np.random.randn(100)

        # Create pipeline
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "regressor",
                    LinearRegression(
                        method="gradient_descent", learning_rate=0.1, n_iterations=1000
                    ),
                ),
            ]
        )

        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        # Check prediction shape
        assert y_pred.shape == y.shape


class TestSVM:
    """Test SVM implementation."""

    def test_svm_linear(self):
        """Test linear SVM."""
        from implementations.ml.svm import SVM
        from implementations.ml.datasets import make_blobs

        # Create linearly separable data
        X, y = make_blobs(n_samples=100, centers=2, random_state=42)

        svm = SVM(kernel="linear", C=1.0)
        svm.fit(X, y)

        # Check training accuracy
        y_pred = svm.predict(X)
        accuracy = np.mean(y_pred == y)

        assert accuracy >= 0.8  # Should be reasonably accurate

    def test_svm_rbf(self):
        """Test RBF kernel SVM."""
        from implementations.ml.svm import SVM
        from implementations.ml.datasets import make_moons

        X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

        svm = SVM(kernel="rbf", C=1.0, gamma="scale")
        svm.fit(X, y, n_iterations=200)

        y_pred = svm.predict(X)
        accuracy = np.mean(y_pred == y)

        assert accuracy >= 0.7


class TestEnsembles:
    """Test ensemble methods."""

    def test_voting_classifier(self):
        """Test VotingClassifier."""
        from implementations.ml.ensembles import VotingClassifier
        from implementations.ml.advanced_algorithms import LogisticRegression
        from implementations.ml.datasets import make_blobs

        X, y = make_blobs(n_samples=100, centers=2, random_state=42)

        clf1 = LogisticRegression()
        clf2 = LogisticRegression()  # Simple second classifier

        voting = VotingClassifier(
            estimators=[("lr1", clf1), ("lr2", clf2)], voting="hard"
        )

        voting.fit(X, y)

        assert voting.predict(X).shape == y.shape

    def test_adaboost(self):
        """Test AdaBoost."""
        from implementations.ml.ensembles import AdaBoost
        from implementations.ml.datasets import make_blobs

        X, y = make_blobs(n_samples=100, centers=2, random_state=42)

        ada = AdaBoost(n_estimators=10, learning_rate=1.0)
        ada.fit(X, y)

        y_pred = ada.predict(X)
        assert y_pred.shape == y.shape


class TestMathFoundations:
    """Test mathematical foundations."""

    def test_vector_operations(self):
        """Test vector operations."""
        from implementations.math.vectors import Vector

        v1 = Vector([1, 2, 3])
        v2 = Vector([4, 5, 6])

        # Test addition
        v_sum = v1 + v2
        assert np.allclose(v_sum.data, [5, 7, 9])

        # Test dot product
        dot = v1.dot(v2)
        assert dot == 32  # 1*4 + 2*5 + 3*6

    def test_matrix_operations(self):
        """Test matrix operations."""
        from implementations.math.matrices import Matrix

        m1 = Matrix([[1, 2], [3, 4]])
        m2 = Matrix([[5, 6], [7, 8]])

        # Test multiplication
        result = m1 @ m2
        expected = np.array([[19, 22], [43, 50]])
        assert np.allclose(result.data, expected)

    def test_calculus_derivatives(self):
        """Test calculus derivatives."""
        from implementations.math.calculus import gradient

        # f(x, y) = x^2 + y^2
        def func(x):
            return np.sum(x**2)

        grad = gradient(func, np.array([3.0, 4.0]))
        assert np.allclose(grad, [6, 8])  # 2x, 2y

    def test_probability_distribution(self):
        """Test probability distributions."""
        from implementations.math.probability import NormalDistribution

        dist = NormalDistribution(mean=0, std=1)

        # Check probability density
        pdf = dist.pdf(0)
        assert np.isclose(pdf, 0.3989, atol=0.01)  # ~1/sqrt(2pi)


class TestMLAlgorithms:
    """Test ML algorithms."""

    def test_linear_regression(self):
        """Test linear regression."""
        from implementations.ml.linear_regression import LinearRegression

        # y = 3x + 5 + noise
        np.random.seed(42)
        X = np.linspace(0, 10, 100).reshape(-1, 1)
        y = 3 * X.squeeze() + 5 + np.random.randn(100) * 0.5

        model = LinearRegression(
            method="gradient_descent", learning_rate=0.01, n_iterations=1000
        )
        model.fit(X, y)

        assert model.theta is not None

    def test_logistic_regression(self):
        """Test logistic regression."""
        from implementations.ml.advanced_algorithms import LogisticRegression
        from implementations.ml.datasets import make_blobs

        X, y = make_blobs(n_samples=100, centers=2, random_state=42)

        clf = LogisticRegression()
        clf.fit(X, y, learning_rate=0.1, n_iterations=1000)

        assert clf.weights is not None

    def test_decision_tree(self):
        """Test decision tree."""
        from implementations.ml.advanced_algorithms import DecisionTree
        from implementations.ml.datasets import make_blobs

        X, y = make_blobs(n_samples=100, centers=2, random_state=42)

        dt = DecisionTree(max_depth=5)
        dt.fit(X, y)

        y_pred = dt.predict(X)
        assert y_pred.shape == y.shape


class TestEvaluation:
    """Test evaluation metrics."""

    def test_accuracy_score(self):
        """Test accuracy metric."""
        from implementations.ml.evaluation import accuracy_score

        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])

        acc = accuracy_score(y_true, y_pred)
        assert acc == 0.8

    def test_mean_squared_error(self):
        """Test MSE metric."""
        from implementations.ml.evaluation import mean_squared_error

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.2, 2.9, 3.8, 5.2])

        mse = mean_squared_error(y_true, y_pred)
        assert mse > 0

    def test_r2_score(self):
        """Test R² score."""
        from implementations.ml.evaluation import r2_score

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = y_true  # Perfect prediction

        r2 = r2_score(y_true, y_pred)
        assert r2 == 1.0


class TestNLP:
    """Test NLP implementations."""

    def test_text_preprocessing(self):
        """Test text preprocessing."""
        from implementations.text_preprocessing import Tokenizer

        tokenizer = Tokenizer()
        text = "Hello, world! This is a test."

        tokens = tokenizer.tokenize(text)

        assert len(tokens) > 0

    def test_bag_of_words(self):
        """Test bag of words."""
        from implementations.text_preprocessing import BagOfWords

        corpus = ["hello world", "world hello", "hello hello"]

        bow = BagOfWords()
        bow.fit(corpus)

        features = bow.transform(["hello world"])
        assert features.shape[1] > 0


class TestNeuralNetworks:
    """Test neural network implementations."""

    def test_perceptron(self):
        """Test perceptron."""
        from implementations.perceptron import Perceptron

        # AND gate
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 0, 1])

        p = Perceptron(n_inputs=2, learning_rate=0.1)
        p.fit(X, y, n_iterations=100)

        # Should learn AND gate
        pred = p.predict(np.array([1, 1]))
        assert pred[0] == 1

    def test_mlp(self):
        """Test multi-layer perceptron."""
        from implementations.mlp import MLP

        # XOR problem
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])

        mlp = MLP(n_inputs=2, hidden_sizes=[4], n_outputs=1, learning_rate=0.1)
        mlp.fit(X, y, n_iterations=1000)

        # XOR should be learnable with hidden layer
        predictions = mlp.predict(X)
        assert predictions.shape == y.shape


class TestPracticalApplications:
    """Test practical application implementations."""

    def test_knn_classifier(self):
        """Test KNN classifier."""
        from implementations.classification import KNNClassifier
        from implementations.ml.datasets import make_blobs

        X, y = make_blobs(n_samples=50, centers=2, random_state=42)

        knn = KNNClassifier(k=3)
        knn.fit(X, y)

        pred = knn.predict(X[:5])
        assert len(pred) == 5

    def test_kmeans_clustering(self):
        """Test K-means clustering."""
        from implementations.clustering import KMeans
        from implementations.ml.datasets import make_blobs

        X, y = make_blobs(n_samples=60, centers=3, random_state=42)

        kmeans = KMeans(n_clusters=3)
        kmeans.fit(X)

        assert len(kmeans.centroids) == 3

    def test_pca(self):
        """Test PCA dimensionality reduction."""
        from implementations.dimension_reduction import PCA
        from implementations.ml.datasets import make_blobs

        X, _ = make_blobs(n_samples=50, centers=2, random_state=42)

        pca = PCA(n_components=1)
        X_transformed = pca.fit_transform(X)

        assert X_transformed.shape[1] == 1


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
    print("\n" + "=" * 70)
    print("All comprehensive tests completed!")
    print("=" * 70)
