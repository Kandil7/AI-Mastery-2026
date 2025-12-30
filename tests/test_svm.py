"""
Tests for SVM (Support Vector Machine) implementation.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.ml.classical import SVMScratch


class TestSVMScratch:
    """Test cases for SVMScratch."""
    
    @pytest.fixture
    def binary_classification_data(self):
        """Generate binary classification dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_classes=2,
            random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        return X_train, X_test, y_train, y_test
    
    @pytest.fixture
    def linearly_separable_data(self):
        """Generate perfectly linearly separable data."""
        np.random.seed(42)
        
        # Class 0: centered around (-2, -2)
        X0 = np.random.randn(50, 2) + np.array([-2, -2])
        # Class 1: centered around (2, 2)
        X1 = np.random.randn(50, 2) + np.array([2, 2])
        
        X = np.vstack([X0, X1])
        y = np.array([0] * 50 + [1] * 50)
        
        return X, y
    
    def test_initialization(self):
        """Test SVM initialization."""
        svm = SVMScratch(C=1.0, learning_rate=0.001, n_iterations=1000)
        
        assert svm.C == 1.0
        assert svm.learning_rate == 0.001
        assert svm.n_iterations == 1000
        assert svm.kernel == 'linear'
        assert svm.weights is None
        assert svm.bias is None
    
    def test_fit_basic(self, linearly_separable_data):
        """Test that SVM can fit on linearly separable data."""
        X, y = linearly_separable_data
        
        svm = SVMScratch(C=1.0, learning_rate=0.01, n_iterations=500)
        svm.fit(X, y)
        
        assert svm.weights is not None
        assert svm.bias is not None
        assert len(svm.weights) == X.shape[1]
        assert len(svm.loss_history) == 500
    
    def test_predict_shape(self, linearly_separable_data):
        """Test prediction output shape."""
        X, y = linearly_separable_data
        
        svm = SVMScratch(C=1.0, learning_rate=0.01, n_iterations=500)
        svm.fit(X, y)
        
        predictions = svm.predict(X)
        
        assert predictions.shape == y.shape
        assert set(predictions).issubset(set(y))
    
    def test_accuracy_linearly_separable(self, linearly_separable_data):
        """Test that SVM achieves high accuracy on linearly separable data."""
        X, y = linearly_separable_data
        
        svm = SVMScratch(C=1.0, learning_rate=0.01, n_iterations=1000)
        svm.fit(X, y)
        
        predictions = svm.predict(X)
        accuracy = np.mean(predictions == y)
        
        # Should achieve at least 95% accuracy on linearly separable data
        assert accuracy >= 0.95, f"Expected accuracy >= 0.95, got {accuracy}"
    
    def test_decision_function(self, linearly_separable_data):
        """Test decision function output."""
        X, y = linearly_separable_data
        
        svm = SVMScratch(C=1.0, learning_rate=0.01, n_iterations=500)
        svm.fit(X, y)
        
        decision = svm.decision_function(X)
        
        assert decision.shape == (len(X),)
        # Class 1 samples should have positive decision values (mostly)
        assert np.mean(decision[y == 1] > 0) > 0.9
        # Class 0 samples should have negative decision values (mostly)
        assert np.mean(decision[y == 0] < 0) > 0.9
    
    def test_predict_proba(self, linearly_separable_data):
        """Test probability estimation."""
        X, y = linearly_separable_data
        
        svm = SVMScratch(C=1.0, learning_rate=0.01, n_iterations=500)
        svm.fit(X, y)
        
        proba = svm.predict_proba(X)
        
        assert proba.shape == (len(X), 2)
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), np.ones(len(X)))
        # All probabilities should be between 0 and 1
        assert np.all(proba >= 0) and np.all(proba <= 1)
    
    def test_binary_classification(self, binary_classification_data):
        """Test on realistic binary classification data."""
        X_train, X_test, y_train, y_test = binary_classification_data
        
        svm = SVMScratch(C=1.0, learning_rate=0.001, n_iterations=1000)
        svm.fit(X_train, y_train)
        
        # Training accuracy
        train_pred = svm.predict(X_train)
        train_acc = np.mean(train_pred == y_train)
        
        # Test accuracy
        test_pred = svm.predict(X_test)
        test_acc = np.mean(test_pred == y_test)
        
        # Should achieve reasonable accuracy
        assert train_acc > 0.6, f"Training accuracy {train_acc} too low"
        assert test_acc > 0.5, f"Test accuracy {test_acc} too low"
    
    def test_support_vectors(self, linearly_separable_data):
        """Test that support vectors are identified."""
        X, y = linearly_separable_data
        
        svm = SVMScratch(C=1.0, learning_rate=0.01, n_iterations=500)
        svm.fit(X, y)
        
        # Support vectors should be a subset of training data
        assert svm.support_vectors_ is not None
        assert len(svm.support_vectors_) <= len(X)
    
    def test_loss_decreases(self, linearly_separable_data):
        """Test that loss generally decreases during training."""
        X, y = linearly_separable_data
        
        svm = SVMScratch(C=1.0, learning_rate=0.01, n_iterations=500)
        svm.fit(X, y)
        
        # Compare first and last loss values
        initial_loss = np.mean(svm.loss_history[:50])
        final_loss = np.mean(svm.loss_history[-50:])
        
        assert final_loss < initial_loss, "Loss should decrease during training"
    
    def test_regularization_effect(self, binary_classification_data):
        """Test that C parameter affects regularization."""
        X_train, X_test, y_train, y_test = binary_classification_data
        
        # Low C = more regularization
        svm_low_c = SVMScratch(C=0.01, learning_rate=0.001, n_iterations=500)
        svm_low_c.fit(X_train, y_train)
        
        # High C = less regularization
        svm_high_c = SVMScratch(C=100.0, learning_rate=0.001, n_iterations=500)
        svm_high_c.fit(X_train, y_train)
        
        # Low C should result in smaller weights (more regularization)
        low_c_norm = np.linalg.norm(svm_low_c.weights)
        high_c_norm = np.linalg.norm(svm_high_c.weights)
        
        assert low_c_norm < high_c_norm, "Low C should produce smaller weight norm"
    
    def test_score_method(self, binary_classification_data):
        """Test the score method from BaseClassifier."""
        X_train, X_test, y_train, y_test = binary_classification_data
        
        svm = SVMScratch(C=1.0, learning_rate=0.001, n_iterations=500)
        svm.fit(X_train, y_train)
        
        score = svm.score(X_test, y_test)
        
        assert 0.0 <= score <= 1.0
    
    def test_nonbinary_raises_error(self):
        """Test that non-binary classification raises an error."""
        X = np.random.randn(30, 5)
        y = np.array([0] * 10 + [1] * 10 + [2] * 10)  # 3 classes
        
        svm = SVMScratch()
        
        with pytest.raises(ValueError, match="binary classification"):
            svm.fit(X, y)
    
    def test_different_label_types(self):
        """Test with different label types (not 0/1)."""
        np.random.seed(42)
        X = np.random.randn(100, 5)
        
        # Test with -1/1 labels
        y_minus_one = np.array([-1] * 50 + [1] * 50)
        svm1 = SVMScratch(n_iterations=100)
        svm1.fit(X, y_minus_one)
        pred1 = svm1.predict(X)
        assert set(pred1).issubset({-1, 1})
        
        # Test with string labels
        y_string = np.array(['cat'] * 50 + ['dog'] * 50)
        svm2 = SVMScratch(n_iterations=100)
        svm2.fit(X, y_string)
        pred2 = svm2.predict(X)
        assert set(pred2).issubset({'cat', 'dog'})


class TestLSTMConv2D:
    """Test cases for LSTM and Conv2D layers."""
    
    def test_lstm_forward(self):
        """Test LSTM forward pass."""
        from src.ml.deep_learning import LSTM
        
        batch_size = 8
        timesteps = 10
        input_size = 20
        hidden_size = 32
        
        lstm = LSTM(input_size=input_size, hidden_size=hidden_size)
        X = np.random.randn(batch_size, timesteps, input_size)
        
        output = lstm.forward(X)
        
        assert output.shape == (batch_size, hidden_size)
    
    def test_lstm_return_sequences(self):
        """Test LSTM with return_sequences=True."""
        from src.ml.deep_learning import LSTM
        
        batch_size = 8
        timesteps = 10
        input_size = 20
        hidden_size = 32
        
        lstm = LSTM(input_size=input_size, hidden_size=hidden_size, return_sequences=True)
        X = np.random.randn(batch_size, timesteps, input_size)
        
        output = lstm.forward(X)
        
        assert output.shape == (batch_size, timesteps, hidden_size)
    
    def test_conv2d_forward(self):
        """Test Conv2D forward pass."""
        from src.ml.deep_learning import Conv2D
        
        batch_size = 4
        in_channels = 3
        height = 28
        width = 28
        out_channels = 32
        kernel_size = 3
        
        conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size
        )
        X = np.random.randn(batch_size, in_channels, height, width)
        
        output = conv.forward(X)
        
        expected_h = height - kernel_size + 1
        expected_w = width - kernel_size + 1
        assert output.shape == (batch_size, out_channels, expected_h, expected_w)
    
    def test_conv2d_with_padding(self):
        """Test Conv2D with padding to maintain spatial size."""
        from src.ml.deep_learning import Conv2D
        
        batch_size = 4
        in_channels = 3
        height = 28
        width = 28
        out_channels = 32
        kernel_size = 3
        padding = 1
        
        conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding
        )
        X = np.random.randn(batch_size, in_channels, height, width)
        
        output = conv.forward(X)
        
        # With padding=1 and kernel=3, output size should match input
        assert output.shape == (batch_size, out_channels, height, width)
    
    def test_maxpool2d(self):
        """Test MaxPool2D layer."""
        from src.ml.deep_learning import MaxPool2D
        
        batch_size = 4
        channels = 32
        height = 28
        width = 28
        pool_size = 2
        
        pool = MaxPool2D(pool_size=pool_size)
        X = np.random.randn(batch_size, channels, height, width)
        
        output = pool.forward(X)
        
        expected_h = height // pool_size
        expected_w = width // pool_size
        assert output.shape == (batch_size, channels, expected_h, expected_w)
    
    def test_flatten(self):
        """Test Flatten layer."""
        from src.ml.deep_learning import Flatten
        
        batch_size = 4
        channels = 32
        height = 7
        width = 7
        
        flatten = Flatten()
        X = np.random.randn(batch_size, channels, height, width)
        
        output = flatten.forward(X)
        
        expected_size = channels * height * width
        assert output.shape == (batch_size, expected_size)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
