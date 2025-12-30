"""
Unit tests for deep learning components.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '..')

from src.ml.deep_learning import (
    Dense, Activation, Dropout, BatchNormalization,
    MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss,
    NeuralNetwork
)


class TestDenseLayer:
    """Tests for Dense layer."""

    def test_dense_forward_shape(self):
        """Test forward pass shape."""
        layer = Dense(5, 3)
        X = np.random.randn(10, 5)
        output = layer.forward(X)
        
        assert output.shape == (10, 3)

    def test_dense_backward_shape(self):
        """Test backward pass shape."""
        layer = Dense(5, 3)
        X = np.random.randn(10, 5)
        _ = layer.forward(X)
        
        output_gradient = np.random.randn(10, 3)
        input_gradient = layer.backward(output_gradient, 0.01)
        
        assert input_gradient.shape == (10, 5)
        assert layer.weights.shape == (5, 3)
        assert layer.bias.shape == (1, 3)

    def test_dense_weight_update(self):
        """Test that weights are updated during backward pass."""
        layer = Dense(4, 2)
        X = np.random.randn(8, 4)
        y = np.random.randn(8, 2)
        
        # Forward pass
        output = layer.forward(X)
        
        # Compute gradient
        loss = MSELoss()
        output_gradient = loss.backward(output, y)
        
        # Store initial weights
        initial_weights = layer.weights.copy()
        initial_bias = layer.bias.copy()
        
        # Backward pass
        layer.backward(output_gradient, 0.01)
        
        # Check weights changed
        assert not np.allclose(layer.weights, initial_weights)
        assert not np.allclose(layer.bias, initial_bias)

    def test_dense_weight_initialization(self):
        """Test different weight initialization methods."""
        input_size = 100
        output_size = 50
        
        # Test Xavier initialization
        layer_xavier = Dense(input_size, output_size, weight_init='xavier')
        init_weights_xavier = layer_xavier.weights.copy()
        
        # Test He initialization
        layer_he = Dense(input_size, output_size, weight_init='he')
        init_weights_he = layer_he.weights.copy()
        
        # He initialization should have higher variance than Xavier
        assert np.var(init_weights_he) > np.var(init_weights_xavier)


class TestActivationLayer:
    """Tests for Activation layer."""

    def test_relu_activation(self):
        """Test ReLU activation."""
        activation = Activation('relu')
        X = np.array([[-2, -1], [0, 1], [2, 3]])
        
        output = activation.forward(X)
        expected = np.array([[0, 0], [0, 1], [2, 3]])
        
        np.testing.assert_array_equal(output, expected)

    def test_sigmoid_activation(self):
        """Test Sigmoid activation."""
        activation = Activation('sigmoid')
        X = np.array([0.0, np.log(3)])  # log(3) because sigmoid(log(3)) = 0.75
        
        output = activation.forward(X)
        expected = np.array([0.5, 0.75])
        
        np.testing.assert_array_almost_equal(output, expected, decimal=5)

    def test_softmax_activation(self):
        """Test Softmax activation."""
        activation = Activation('softmax')
        X = np.array([[1, 2, 3]])
        
        output = activation.forward(X)
        expected_sum = 1.0  # Softmax should sum to 1
        
        assert abs(np.sum(output) - expected_sum) < 1e-6

    def test_activation_backward_shape(self):
        """Test activation backward pass shapes."""
        X = np.random.randn(5, 10)
        output_gradient = np.random.randn(5, 10)
        
        for activation_name in ['relu', 'sigmoid', 'tanh', 'softmax']:
            activation = Activation(activation_name)
            _ = activation.forward(X)
            
            input_gradient = activation.backward(output_gradient, 0.01)
            assert input_gradient.shape == X.shape


class TestDropoutLayer:
    """Tests for Dropout layer."""

    def test_dropout_forward_training(self):
        """Test dropout forward pass during training."""
        dropout = Dropout(rate=0.5)
        X = np.ones((10, 20))
        
        output = dropout.forward(X, training=True)
        
        # Should have approximately 50% zeros
        zero_fraction = np.sum(output == 0) / output.size
        assert 0.4 < zero_fraction < 0.6
        
        # Non-zero elements should be scaled by 1/(1-rate) = 2
        non_zero_elements = output[output != 0]
        assert np.allclose(non_zero_elements, 2.0)

    def test_dropout_forward_inference(self):
        """Test dropout forward pass during inference."""
        dropout = Dropout(rate=0.5)
        X = np.ones((10, 20))
        
        output = dropout.forward(X, training=False)
        
        # Should not change anything during inference
        np.testing.assert_array_equal(output, X)

    def test_dropout_backward(self):
        """Test dropout backward pass."""
        dropout = Dropout(rate=0.5)
        X = np.random.randn(5, 10)
        output_gradient = np.random.randn(5, 10)
        
        _ = dropout.forward(X, training=True)
        input_gradient = dropout.backward(output_gradient, 0.01)
        
        assert input_gradient.shape == X.shape


class TestBatchNormalization:
    """Tests for BatchNormalization layer."""

    def test_batch_normalization_forward(self):
        """Test batch normalization forward pass."""
        bn = BatchNormalization(n_features=5)
        X = np.random.randn(20, 5)
        
        # Training mode
        output_train = bn.forward(X, training=True)
        
        # Should have mean ~0 and std ~1
        mean = np.mean(output_train, axis=0)
        std = np.std(output_train, axis=0)
        
        assert np.allclose(mean, 0, atol=0.1)
        assert np.allclose(std, 1, atol=0.1)

    def test_batch_normalization_shape(self):
        """Test batch normalization preserves shape."""
        bn = BatchNormalization(n_features=10)
        X = np.random.randn(15, 10)
        
        output = bn.forward(X, training=True)
        assert output.shape == X.shape

    def test_batch_normalization_learning(self):
        """Test that batch normalization updates learnable parameters."""
        bn = BatchNormalization(n_features=5)
        X = np.random.randn(10, 5)
        y_true = np.random.randn(10, 5)
        
        # Store initial parameters
        initial_gamma = bn.gamma.copy()
        initial_beta = bn.beta.copy()
        
        # Forward and backward pass
        output = bn.forward(X, training=True)
        loss = MSELoss()
        grad = loss.backward(output, y_true)
        _ = bn.backward(grad, 0.01)
        
        # Parameters should be updated
        assert not np.allclose(bn.gamma, initial_gamma)
        assert not np.allclose(bn.beta, initial_beta)


class TestLossFunctions:
    """Tests for loss functions."""

    def test_mse_loss(self):
        """Test MSE loss."""
        loss_fn = MSELoss()
        
        y_pred = np.array([[1, 2], [3, 4]])
        y_true = np.array([[1, 2], [3, 4]])
        
        loss = loss_fn.forward(y_pred, y_true)
        assert loss == pytest.approx(0.0)  # Perfect prediction
        
        # Test with different values
        y_true = np.array([[0, 0], [0, 0]])
        expected_loss = np.mean((y_pred - y_true) ** 2)
        actual_loss = loss_fn.forward(y_pred, y_true)
        assert actual_loss == pytest.approx(expected_loss)

    def test_cross_entropy_loss(self):
        """Test Cross-Entropy loss."""
        loss_fn = CrossEntropyLoss()
        
        # Perfect prediction (1.0 for correct class)
        y_pred = np.array([[0, 0, 1], [1, 0, 0]])
        y_true = np.array([2, 0])
        
        loss = loss_fn.forward(y_pred, y_true)
        # Should be very close to 0 (perfect prediction)
        assert loss < 0.1

    def test_cross_entropy_backward_shape(self):
        """Test Cross-Entropy backward pass shape."""
        loss_fn = CrossEntropyLoss()
        
        y_pred = np.random.randn(10, 5)
        y_true = np.random.randint(0, 5, 10)
        
        grad = loss_fn.backward(y_pred, y_true)
        assert grad.shape == y_pred.shape

    def test_binary_cross_entropy_loss(self):
        """Test Binary Cross-Entropy loss."""
        loss_fn = BinaryCrossEntropyLoss()
        
        # Perfect binary predictions
        y_pred = np.array([[0.9], [0.1]])
        y_true = np.array([[1], [0]])
        
        loss = loss_fn.forward(y_pred, y_true)
        assert loss < 0.5  # Should be low for good predictions


class TestNeuralNetwork:
    """Tests for NeuralNetwork."""

    def test_neural_network_basic(self):
        """Test basic neural network functionality."""
        model = NeuralNetwork()
        model.add(Dense(4, 8))
        model.add(Activation('relu'))
        model.add(Dense(8, 1))
        model.add(Activation('sigmoid'))
        
        model.compile(loss=BinaryCrossEntropyLoss(), learning_rate=0.1)
        
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 2, (100, 1))
        
        history = model.fit(X, y, epochs=5, batch_size=32, verbose=False)
        
        # Should have training history
        assert len(history['loss']) == 5
        assert len(history['accuracy']) == 5

    def test_neural_network_multiclass(self):
        """Test neural network for multiclass classification."""
        model = NeuralNetwork()
        model.add(Dense(4, 16))
        model.add(Activation('relu'))
        model.add(Dense(16, 3))
        model.add(Activation('softmax'))
        
        model.compile(loss=CrossEntropyLoss(), learning_rate=0.1)
        
        X = np.random.randn(100, 4)
        y = np.random.randint(0, 3, 100)  # 3 classes
        
        history = model.fit(X, y, epochs=3, batch_size=32, verbose=False)
        
        assert len(history['loss']) == 3

    def test_neural_network_prediction(self):
        """Test neural network prediction."""
        model = NeuralNetwork()
        model.add(Dense(5, 10))
        model.add(Activation('relu'))
        model.add(Dense(10, 3))
        model.add(Activation('softmax'))
        
        model.compile(loss=CrossEntropyLoss(), learning_rate=0.01)
        
        X = np.random.randn(20, 5)
        
        # Before training, predictions should be valid probability distributions
        probs = model.predict_proba(X)
        assert probs.shape == (20, 3)
        assert np.allclose(np.sum(probs, axis=1), 1.0)  # Sum to 1
        
        predictions = model.predict(X)
        assert predictions.shape == (20,)
        assert np.all(predictions >= 0) and np.all(predictions < 3)  # Valid class indices

    def test_neural_network_with_validation(self):
        """Test neural network with validation data."""
        model = NeuralNetwork()
        model.add(Dense(3, 6))
        model.add(Activation('relu'))
        model.add(Dense(6, 2))
        model.add(Activation('softmax'))
        
        model.compile(loss=CrossEntropyLoss(), learning_rate=0.1)
        
        X_train = np.random.randn(80, 3)
        y_train = np.random.randint(0, 2, 80)
        X_val = np.random.randn(20, 3)
        y_val = np.random.randint(0, 2, 20)
        
        history = model.fit(
            X_train, y_train,
            epochs=5,
            batch_size=16,
            validation_data=(X_val, y_val),
            verbose=False
        )
        
        # Should have both training and validation metrics
        assert len(history['loss']) == 5
        assert len(history['val_loss']) == 5
        assert len(history['accuracy']) == 5
        assert len(history['val_accuracy']) == 5

    def test_neural_network_summary(self):
        """Test neural network summary method."""
        model = NeuralNetwork()
        model.add(Dense(10, 20))
        model.add(Activation('relu'))
        model.add(Dense(20, 5))
        
        # This should run without error and print summary
        model.summary()


def test_deep_learning_integration():
    """Integration test for deep learning components."""
    # Create a simple neural network
    model = NeuralNetwork()
    model.add(Dense(4, 16))
    model.add(BatchNormalization(16))
    model.add(Activation('relu'))
    model.add(Dropout(0.3))
    model.add(Dense(16, 8))
    model.add(Activation('relu'))
    model.add(Dense(8, 1))
    model.add(Activation('sigmoid'))
    
    model.compile(loss=BinaryCrossEntropyLoss(), learning_rate=0.01)
    
    # Generate synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (np.sum(X, axis=1) > 0).astype(int).reshape(-1, 1)
    
    # Train model
    history = model.fit(X, y, epochs=10, batch_size=32, verbose=False)
    
    # Evaluate performance
    loss, accuracy = model.evaluate(X, y)
    
    # Should achieve reasonable performance on this simple task
    assert accuracy >= 0.6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])