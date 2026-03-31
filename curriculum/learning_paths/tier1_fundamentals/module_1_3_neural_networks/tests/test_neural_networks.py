"""
Tests for Module 1.3: Neural Networks.
"""

import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from part1_fundamentals.module_1_3_neural_networks.activations import (
    Sigmoid, Tanh, ReLU, LeakyReLU, ELU, Swish, GELU, Softmax,
    get_activation,
)
from part1_fundamentals.module_1_3_neural_networks.losses import (
    MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss, HuberLoss,
    get_loss,
)
from part1_fundamentals.module_1_3_neural_networks.layers import (
    Dense, Conv2D, MaxPool2D, Dropout, BatchNormalization, Flatten, Sequential,
)
from part1_fundamentals.module_1_3_neural_networks.optimizers import (
    SGD, Adam, RMSprop, Adagrad, AdamW,
    get_optimizer, LearningRateScheduler,
)
from part1_fundamentals.module_1_3_neural_networks.mlp import (
    MLP, Trainer, create_mlp_classifier,
)


class TestActivations(unittest.TestCase):
    """Tests for activation functions."""
    
    def test_sigmoid_range(self):
        """Test sigmoid output range."""
        sigmoid = Sigmoid()
        x = np.array([-10, -1, 0, 1, 10])
        output = sigmoid.forward(x)
        
        self.assertTrue(np.all(output > 0) and np.all(output < 1))
        self.assertAlmostEqual(output[2], 0.5)  # sigmoid(0) = 0.5
    
    def test_tanh_range(self):
        """Test tanh output range."""
        tanh = Tanh()
        x = np.array([-10, -1, 0, 1, 10])
        output = tanh.forward(x)
        
        self.assertTrue(np.all(output >= -1) and np.all(output <= 1))
        self.assertAlmostEqual(output[2], 0)  # tanh(0) = 0
    
    def test_relu(self):
        """Test ReLU activation."""
        relu = ReLU()
        x = np.array([-2, -1, 0, 1, 2])
        output = relu.forward(x)
        
        expected = np.array([0, 0, 0, 1, 2])
        np.testing.assert_array_equal(output, expected)
    
    def test_leaky_relu(self):
        """Test LeakyReLU activation."""
        leaky_relu = LeakyReLU(alpha=0.1)
        x = np.array([-2, -1, 0, 1, 2])
        output = leaky_relu.forward(x)
        
        self.assertAlmostEqual(output[0], -0.2)
        self.assertAlmostEqual(output[3], 1)
    
    def test_softmax(self):
        """Test softmax sums to 1."""
        softmax = Softmax()
        x = np.array([[1, 2, 3], [4, 5, 6]])
        output = softmax.forward(x)
        
        np.testing.assert_array_almost_equal(output.sum(axis=1), 1.0)
    
    def test_activation_backward(self):
        """Test activation backward pass."""
        relu = ReLU()
        x = np.array([-1, 0, 1, 2])
        output = relu.forward(x)
        grad_output = np.ones_like(x)
        grad_input = relu.backward(grad_output)
        
        expected = np.array([0, 0, 1, 1])
        np.testing.assert_array_equal(grad_input, expected)
    
    def test_get_activation(self):
        """Test activation factory."""
        relu = get_activation('relu')
        self.assertIsInstance(relu, ReLU)
        
        sigmoid = get_activation('sigmoid')
        self.assertIsInstance(sigmoid, Sigmoid)


class TestLosses(unittest.TestCase):
    """Tests for loss functions."""
    
    def test_mse_loss(self):
        """Test MSE loss."""
        mse = MSELoss()
        y_pred = np.array([[1.0], [2.0], [3.0]])
        y_true = np.array([[1.0], [2.0], [3.0]])
        
        loss = mse.forward(y_pred, y_true)
        self.assertAlmostEqual(loss, 0.0)
    
    def test_mse_gradient(self):
        """Test MSE gradient."""
        mse = MSELoss()
        y_pred = np.array([[1.0], [2.0]])
        y_true = np.array([[2.0], [3.0]])
        
        mse.forward(y_pred, y_true)
        grad = mse.backward()
        
        # Gradient should point towards reducing error
        self.assertTrue(np.all(grad > 0))
    
    def test_cross_entropy_loss(self):
        """Test cross-entropy loss."""
        ce = CrossEntropyLoss(from_logits=True)
        logits = np.array([[2.0, 1.0, 0.1]])
        labels = np.array([0])
        
        loss = ce.forward(logits, labels)
        self.assertGreater(loss, 0)
    
    def test_bce_loss(self):
        """Test binary cross-entropy loss."""
        bce = BinaryCrossEntropyLoss(from_logits=False)
        y_pred = np.array([[0.9], [0.1]])
        y_true = np.array([[1], [0]])
        
        loss = bce.forward(y_pred, y_true)
        self.assertLess(loss, 0.2)  # Good predictions should have low loss
    
    def test_huber_loss(self):
        """Test Huber loss."""
        huber = HuberLoss(delta=1.0)
        y_pred = np.array([[1.0], [5.0]])  # Second is outlier
        y_true = np.array([[1.0], [2.0]])
        
        loss = huber.forward(y_pred, y_true)
        self.assertGreater(loss, 0)


class TestLayers(unittest.TestCase):
    """Tests for neural network layers."""
    
    def test_dense_forward(self):
        """Test dense layer forward pass."""
        dense = Dense(input_size=10, output_size=5)
        x = np.random.randn(32, 10)
        output = dense.forward(x)
        
        self.assertEqual(output.shape, (32, 5))
    
    def test_dense_backward(self):
        """Test dense layer backward pass."""
        dense = Dense(input_size=10, output_size=5)
        x = np.random.randn(32, 10)
        output = dense.forward(x)
        grad_output = np.random.randn(32, 5)
        grad_input = dense.backward(grad_output)
        
        self.assertEqual(grad_input.shape, (32, 10))
        self.assertEqual(dense.grad_weight.shape, (5, 10))
    
    def test_conv2d_forward(self):
        """Test Conv2D forward pass."""
        conv = Conv2D(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        x = np.random.randn(32, 3, 28, 28)
        output = conv.forward(x)
        
        self.assertEqual(output.shape, (32, 16, 28, 28))
    
    def test_maxpool_forward(self):
        """Test MaxPool2D forward pass."""
        pool = MaxPool2D(kernel_size=2, stride=2)
        x = np.random.randn(32, 16, 28, 28)
        output = pool.forward(x)
        
        self.assertEqual(output.shape, (32, 16, 14, 14))
    
    def test_dropout_training(self):
        """Test dropout in training mode."""
        dropout = Dropout(p=0.5)
        x = np.ones((32, 100))
        output = dropout.forward(x, training=True)
        
        # Mean should be approximately 1.0 (inverted dropout)
        self.assertAlmostEqual(output.mean(), 1.0, places=1)
    
    def test_dropout_eval(self):
        """Test dropout in eval mode."""
        dropout = Dropout(p=0.5)
        x = np.ones((32, 100))
        output = dropout.forward(x, training=False)
        
        np.testing.assert_array_equal(output, x)
    
    def test_batchnorm(self):
        """Test batch normalization."""
        bn = BatchNormalization(num_features=100)
        x = np.random.randn(32, 100) * 5 + 10  # Non-standardized
        
        output = bn.forward(x, training=True)
        
        # Output should be approximately standardized
        self.assertAlmostEqual(output.mean(), 0, places=1)
        self.assertAlmostEqual(output.std(), 1, places=1)
    
    def test_sequential(self):
        """Test sequential model."""
        model = Sequential([
            Dense(10, 5),
            ReLU(),
            Dense(5, 2),
        ])
        
        x = np.random.randn(32, 10)
        output = model.forward(x)
        
        self.assertEqual(output.shape, (32, 2))


class TestOptimizers(unittest.TestCase):
    """Tests for optimizers."""
    
    def test_sgd(self):
        """Test SGD optimizer."""
        optimizer = SGD(learning_rate=0.1)
        params = {'w': np.array([5.0])}
        grads = {'w': np.array([1.0])}
        
        params = optimizer.step(params, grads)
        
        self.assertAlmostEqual(params['w'][0], 4.9)
    
    def test_adam(self):
        """Test Adam optimizer."""
        optimizer = Adam(learning_rate=0.1)
        params = {'w': np.array([5.0])}
        grads = {'w': np.array([1.0])}
        
        params = optimizer.step(params, grads)
        
        # Adam should update the parameter
        self.assertNotAlmostEqual(params['w'][0], 5.0)
    
    def test_learning_rate_scheduler(self):
        """Test learning rate scheduler."""
        scheduler = LearningRateScheduler('step', initial_lr=0.1, milestones=[30], gamma=0.1)
        
        self.assertAlmostEqual(scheduler.get_lr(0), 0.1)
        self.assertAlmostEqual(scheduler.get_lr(29), 0.1)
        self.assertAlmostEqual(scheduler.get_lr(30), 0.01)
    
    def test_get_optimizer(self):
        """Test optimizer factory."""
        sgd = get_optimizer('sgd', learning_rate=0.01)
        self.assertIsInstance(sgd, SGD)
        
        adam = get_optimizer('adam', learning_rate=0.001)
        self.assertIsInstance(adam, Adam)


class TestMLP(unittest.TestCase):
    """Tests for MLP model."""
    
    def test_mlp_forward(self):
        """Test MLP forward pass."""
        model = MLP(
            input_size=10,
            hidden_sizes=[32, 16],
            output_size=2,
        )
        
        x = np.random.randn(32, 10)
        output = model.forward(x)
        
        self.assertEqual(output.shape, (32, 2))
    
    def test_mlp_predict(self):
        """Test MLP prediction."""
        model = MLP(
            input_size=10,
            hidden_sizes=[32],
            output_size=3,
        )
        
        x = np.random.randn(5, 10)
        predictions = model.predict(x)
        
        self.assertEqual(len(predictions), 5)
        self.assertTrue(all(0 <= p < 3 for p in predictions))
    
    def test_mlp_count_parameters(self):
        """Test parameter counting."""
        model = MLP(
            input_size=10,
            hidden_sizes=[32],
            output_size=2,
        )
        
        # Input->hidden: 10*32 + 32 = 352
        # Hidden->output: 32*2 + 2 = 66
        # Total: 418
        self.assertEqual(model.count_parameters(), 418)
    
    def test_mlp_save_load(self):
        """Test model save and load."""
        model = MLP(
            input_size=10,
            hidden_sizes=[32],
            output_size=2,
        )
        
        x = np.random.randn(5, 10)
        original_output = model.forward(x)
        
        # Save and load
        model.save('test_mlp.json')
        loaded_model = MLP.load('test_mlp.json')
        
        loaded_output = loaded_model.forward(x)
        np.testing.assert_array_almost_equal(original_output, loaded_output)
        
        # Cleanup
        os.remove('test_mlp.json')
    
    def test_trainer(self):
        """Test MLP trainer."""
        np.random.seed(42)
        
        # Create simple classification data
        X = np.random.randn(200, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = MLP(
            input_size=10,
            hidden_sizes=[32],
            output_size=2,
        )
        
        trainer = Trainer(model, loss='cross_entropy', optimizer='adam')
        history = trainer.fit(X, y, epochs=10, verbose=False)
        
        self.assertEqual(len(history.train_loss), 10)
        self.assertLess(history.train_loss[-1], history.train_loss[0])


if __name__ == '__main__':
    unittest.main()
