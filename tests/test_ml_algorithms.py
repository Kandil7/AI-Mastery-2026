"""
Unit tests for ML algorithms.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_regression
import sys
sys.path.insert(0, '..')


class TestLinearRegression:
    """Tests for LinearRegressionScratch."""
    
    def test_linear_regression_fit(self):
        from src.ml.classical import LinearRegressionScratch
        
        # Simple linear data
        X = np.array([[1], [2], [3], [4], [5]])
        y = np.array([2, 4, 6, 8, 10])
        
        model = LinearRegressionScratch(method='closed_form')
        model.fit(X, y)
        
        # Should find y = 2x
        assert model.weights[0] == pytest.approx(2.0, rel=0.1)
    
    def test_gradient_descent_convergence(self):
        from src.ml.classical import LinearRegressionScratch
        
        np.random.seed(42)
        X = np.random.randn(100, 3)
        true_weights = np.array([1.5, -2.0, 0.5])
        y = X @ true_weights + np.random.randn(100) * 0.1
        
        model = LinearRegressionScratch(
            method='gradient_descent',
            learning_rate=0.1,
            n_iterations=1000
        )
        model.fit(X, y)
        
        # Loss should decrease
        assert model.loss_history[-1] < model.loss_history[0]
    
    def test_prediction_shape(self):
        from src.ml.classical import LinearRegressionScratch
        
        X = np.random.randn(50, 5)
        y = np.random.randn(50)
        
        model = LinearRegressionScratch()
        model.fit(X, y)
        predictions = model.predict(X)
        
        assert predictions.shape == (50,)


class TestLogisticRegression:
    """Tests for LogisticRegressionScratch."""
    
    def test_binary_classification(self):
        from src.ml.classical import LogisticRegressionScratch
        
        X, y = make_classification(
            n_samples=200, n_features=4, n_classes=2, 
            n_informative=3, random_state=42
        )
        
        model = LogisticRegressionScratch(n_iterations=500, learning_rate=0.1)
        model.fit(X, y)
        
        accuracy = model.score(X, y)
        assert accuracy > 0.7  # Should achieve reasonable accuracy
    
    def test_predict_proba_shape(self):
        from src.ml.classical import LogisticRegressionScratch
        
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        model = LogisticRegressionScratch(n_iterations=100)
        model.fit(X, y)
        proba = model.predict_proba(X)
        
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestKNN:
    """Tests for KNNScratch."""
    
    def test_knn_classification(self):
        from src.ml.classical import KNNScratch
        
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        model = KNNScratch(k=5)
        model.fit(X, y)
        
        # Should at least get training data right with k=1
        model_k1 = KNNScratch(k=1)
        model_k1.fit(X, y)
        accuracy = model_k1.score(X, y)
        assert accuracy == 1.0  # Perfect on training with k=1
    
    def test_different_metrics(self):
        from src.ml.classical import KNNScratch
        
        X = np.array([[0, 0], [1, 1], [2, 2]])
        y = np.array([0, 1, 1])
        
        for metric in ['euclidean', 'manhattan', 'cosine']:
            model = KNNScratch(k=1, metric=metric)
            model.fit(X, y)
            predictions = model.predict(X)
            assert len(predictions) == 3


class TestDecisionTree:
    """Tests for DecisionTreeScratch."""
    
    def test_decision_tree_fit(self):
        from src.ml.classical import DecisionTreeScratch
        
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        model = DecisionTreeScratch(max_depth=5)
        model.fit(X, y)
        
        assert model.root is not None
    
    def test_perfect_split(self):
        from src.ml.classical import DecisionTreeScratch
        
        # Perfectly separable data
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 0, 1, 1])
        
        model = DecisionTreeScratch(max_depth=3)
        model.fit(X, y)
        
        accuracy = model.score(X, y)
        assert accuracy == 1.0


class TestRandomForest:
    """Tests for RandomForestScratch."""
    
    def test_random_forest_ensemble(self):
        from src.ml.classical import RandomForestScratch
        
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        model = RandomForestScratch(n_estimators=10, max_depth=5)
        model.fit(X, y)
        
        assert len(model.trees) == 10
    
    def test_random_forest_accuracy(self):
        from src.ml.classical import RandomForestScratch
        
        X, y = make_classification(n_samples=200, n_features=6, random_state=42)
        
        model = RandomForestScratch(n_estimators=20, max_depth=5)
        model.fit(X, y)
        
        accuracy = model.score(X, y)
        assert accuracy > 0.8


class TestNaiveBayes:
    """Tests for GaussianNBScratch."""
    
    def test_gaussian_nb(self):
        from src.ml.classical import GaussianNBScratch
        
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        model = GaussianNBScratch()
        model.fit(X, y)
        
        predictions = model.predict(X)
        assert predictions.shape == (100,)
    
    def test_class_priors(self):
        from src.ml.classical import GaussianNBScratch
        
        X = np.random.randn(100, 3)
        y = np.array([0]*60 + [1]*40)
        
        model = GaussianNBScratch()
        model.fit(X, y)
        
        assert model.class_priors_[0] == pytest.approx(0.6)
        assert model.class_priors_[1] == pytest.approx(0.4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
