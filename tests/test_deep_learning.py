"""
Unit tests for deep learning components.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, '..')


class TestDenseLayers:
    """Tests for Dense layer."""
    
    def test_dense_output_shape(self):
        from src.ml.deep_learning import Dense
        
        layer = Dense(10, 5)
        x = np.random.randn(32, 10)
        output = layer.forward(x)
        
        assert output.shape == (32, 5)
    
    def test_dense_backward(self):
        from src.ml.deep_learning import Dense
        
        layer = Dense(10, 5)
        x = np.random.randn(32, 10)
        output = layer.forward(x)
        
        grad = np.random.randn(32, 5)
        input_grad = layer.backward(grad, 0.01)
        
        assert input_grad.shape == (32, 10)


class TestActivations:
    """Tests for Activation layers."""
    
    def test_relu_forward(self):
        from src.ml.deep_learning import Activation
        
        layer = Activation('relu')
        x = np.array([[-1, 0, 1], [2, -2, 3]])
        output = layer.forward(x)
        
        expected = np.array([[0, 0, 1], [2, 0, 3]])
        np.testing.assert_array_equal(output, expected)
    
    def test_sigmoid_range(self):
        from src.ml.deep_learning import Activation
        
        layer = Activation('sigmoid')
        x = np.random.randn(10, 5)
        output = layer.forward(x)
        
        assert np.all((output >= 0) & (output <= 1))
    
    def test_softmax_sum(self):
        from src.ml.deep_learning import Activation
        
        layer = Activation('softmax')
        x = np.random.randn(5, 3)
        output = layer.forward(x)
        
        sums = np.sum(output, axis=1)
        np.testing.assert_array_almost_equal(sums, np.ones(5))


class TestDropout:
    """Tests for Dropout layer."""
    
    def test_dropout_training(self):
        from src.ml.deep_learning import Dropout
        
        layer = Dropout(rate=0.5)
        x = np.ones((100, 100))
        
        output = layer.forward(x, training=True)
        
        # About half should be zero (with scaling)
        zeros = np.sum(output == 0)
        assert zeros > 0
    
    def test_dropout_inference(self):
        from src.ml.deep_learning import Dropout
        
        layer = Dropout(rate=0.5)
        x = np.ones((10, 10))
        
        output = layer.forward(x, training=False)
        
        # No dropout during inference
        np.testing.assert_array_equal(output, x)


class TestBatchNorm:
    """Tests for BatchNormalization layer."""
    
    def test_batchnorm_output_shape(self):
        from src.ml.deep_learning import BatchNormalization
        
        layer = BatchNormalization(64)
        x = np.random.randn(32, 64)
        output = layer.forward(x, training=True)
        
        assert output.shape == (32, 64)
    
    def test_batchnorm_mean_std(self):
        from src.ml.deep_learning import BatchNormalization
        
        layer = BatchNormalization(64)
        x = np.random.randn(100, 64) * 5 + 10
        output = layer.forward(x, training=True)
        
        # Output should be normalized (mean~0, std~1)
        assert np.abs(np.mean(output)) < 0.5
        assert 0.5 < np.std(output) < 1.5


class TestLossFunctions:
    """Tests for loss functions."""
    
    def test_mse_loss(self):
        from src.ml.deep_learning import MSELoss
        
        loss_fn = MSELoss()
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        
        assert loss_fn.forward(y_pred, y_true) == 0.0
    
    def test_cross_entropy_loss(self):
        from src.ml.deep_learning import CrossEntropyLoss
        
        loss_fn = CrossEntropyLoss()
        y_true = np.array([[1, 0], [0, 1]])
        y_pred = np.array([[0.9, 0.1], [0.1, 0.9]])
        
        loss = loss_fn.forward(y_pred, y_true)
        assert loss > 0


class TestNeuralNetwork:
    """Tests for NeuralNetwork class."""
    
    def test_forward_pass(self):
        from src.ml.deep_learning import NeuralNetwork, Dense, Activation
        
        model = NeuralNetwork()
        model.add(Dense(4, 8))
        model.add(Activation('relu'))
        model.add(Dense(8, 2))
        model.add(Activation('softmax'))
        
        x = np.random.randn(10, 4)
        output = model.forward(x)
        
        assert output.shape == (10, 2)
        np.testing.assert_array_almost_equal(output.sum(axis=1), np.ones(10))
    
    def test_xor_problem(self):
        from src.ml.deep_learning import (
            NeuralNetwork, Dense, Activation, CrossEntropyLoss
        )
        
        # XOR dataset
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([0, 1, 1, 0])
        
        model = NeuralNetwork()
        model.add(Dense(2, 8, weight_init='he'))
        model.add(Activation('relu'))
        model.add(Dense(8, 2))
        model.add(Activation('softmax'))
        
        model.compile(loss=CrossEntropyLoss(), learning_rate=0.5)
        
        # Train
        history = model.fit(X, y, epochs=500, batch_size=4, verbose=False)
        
        # Should learn XOR
        predictions = model.predict(X)
        accuracy = np.mean(predictions == y)
        assert accuracy >= 0.75  # XOR is learnable


class TestConvolutions:
    """Tests for convolution operations."""
    
    def test_conv2d_shape(self):
        from src.ml.deep_learning import conv2d_single
        
        image = np.random.randn(10, 10)
        kernel = np.ones((3, 3)) / 9  # Average filter
        
        output = conv2d_single(image, kernel)
        
        assert output.shape == (8, 8)  # (10-3+1, 10-3+1)
    
    def test_maxpool_shape(self):
        from src.ml.deep_learning import max_pool2d
        
        image = np.random.randn(8, 8)
        
        output = max_pool2d(image, pool_size=2, stride=2)
        
        assert output.shape == (4, 4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
