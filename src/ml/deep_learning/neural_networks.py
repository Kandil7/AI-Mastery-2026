"""
Neural Networks Implementation

This module implements neural networks from scratch using NumPy,
including feedforward networks, backpropagation, and various activation functions.
"""

import numpy as np
from typing import List, Tuple, Callable, Optional
from src.core.optimization import Adam
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    Neural Network implementation from scratch.
    
    This class implements a multi-layer perceptron with configurable architecture,
    activation functions, and optimization methods.
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        activation: str = 'relu',
        output_activation: str = 'linear',
        learning_rate: float = 0.001,
        optimizer: str = 'adam'
    ):
        """
        Initialize Neural Network.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activation: Activation function for hidden layers ('relu', 'sigmoid', 'tanh', 'linear')
            output_activation: Activation function for output layer
            learning_rate: Learning rate for optimization
            optimizer: Optimization algorithm ('sgd', 'adam')
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1  # Number of weight matrices
        
        # Store activation functions
        self.activation_fn = self._get_activation(activation)
        self.activation_deriv = self._get_activation_derivative(activation)
        self.output_activation_fn = self._get_activation(output_activation)
        self.output_activation_deriv = self._get_activation_derivative(output_activation)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers):
            # Xavier/He initialization based on activation function
            if activation == 'relu' or activation == 'linear':
                # He initialization for ReLU
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0 / layer_sizes[i])
            else:
                # Xavier initialization for others
                w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(1.0 / layer_sizes[i])
            
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
        
        # Set learning rate and optimizer
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer
        
        # Initialize optimizer-specific parameters (for Adam)
        if optimizer == 'adam':
            self.optimizer = Adam(learning_rate)
            # For Adam, we need to maintain momentum and velocity for each parameter
            self.m_weights = [np.zeros_like(w) for w in self.weights]
            self.v_weights = [np.zeros_like(w) for w in self.weights]
            self.m_biases = [np.zeros_like(b) for b in self.biases]
            self.v_biases = [np.zeros_like(b) for b in self.biases]
            self.t = 0  # Time step for bias correction
    
    def _get_activation(self, name: str) -> Callable:
        """Get activation function by name."""
        if name == 'relu':
            return lambda x: np.maximum(0, x)
        elif name == 'sigmoid':
            return lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow
        elif name == 'tanh':
            return lambda x: np.tanh(x)
        elif name == 'linear':
            return lambda x: x
        else:
            raise ValueError(f"Unknown activation function: {name}")
    
    def _get_activation_derivative(self, name: str) -> Callable:
        """Get derivative of activation function by name."""
        if name == 'relu':
            return lambda x: (x > 0).astype(float)
        elif name == 'sigmoid':
            # Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))
            sigmoid_fn = lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return lambda x: sigmoid_fn(x) * (1 - sigmoid_fn(x))
        elif name == 'tanh':
            # Derivative of tanh: 1 - tanh(x)^2
            return lambda x: 1 - np.tanh(x) ** 2
        elif name == 'linear':
            return lambda x: np.ones_like(x)
        else:
            raise ValueError(f"Unknown activation function: {name}")
    
    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Forward pass through the network.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Tuple of (output, layer_inputs, layer_outputs)
        """
        X = np.asarray(X)
        
        # Store inputs and outputs for each layer for backpropagation
        layer_inputs = [X]  # Inputs to each layer (before activation)
        layer_outputs = [X]  # Outputs from each layer (after activation)
        
        current_input = X
        
        # Forward through hidden layers
        for i in range(self.n_layers - 1):
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            layer_inputs.append(z)
            
            a = self.activation_fn(z)
            layer_outputs.append(a)
            
            current_input = a
        
        # Output layer
        z = np.dot(current_input, self.weights[-1]) + self.biases[-1]
        layer_inputs.append(z)
        
        # Apply output activation function
        output = self.output_activation_fn(z)
        layer_outputs.append(output)
        
        return output, layer_inputs, layer_outputs
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray, 
                 layer_inputs: List[np.ndarray], layer_outputs: List[np.ndarray]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Backward pass through the network (backpropagation).
        
        Args:
            X: Input data
            y: True labels
            output: Network output
            layer_inputs: Inputs to each layer (from forward pass)
            layer_outputs: Outputs from each layer (from forward pass)
            
        Returns:
            Tuple of (weight_gradients, bias_gradients)
        """
        m = X.shape[0]  # Number of samples
        
        # Initialize gradients
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Compute output layer error (derivative of loss w.r.t. output)
        # For now, assume mean squared error: L = 0.5 * (y_pred - y_true)^2
        # So dL/dA = (A - y)
        delta = output - y
        
        # Backpropagate through layers (starting from output layer)
        for i in range(self.n_layers - 1, -1, -1):
            # Compute gradients for weights and biases
            weight_gradients[i] = np.dot(layer_outputs[i].T, delta) / m
            bias_gradients[i] = np.mean(delta, axis=0, keepdims=True)
            
            # Compute error for previous layer (if not the first layer)
            if i > 0:
                # Derivative of activation function
                activation_deriv = self.activation_deriv if i < self.n_layers - 1 else self.output_activation_deriv
                
                # Propagate error backward
                delta = np.dot(delta, self.weights[i].T) * activation_deriv(layer_inputs[i])
        
        return weight_gradients, bias_gradients
    
    def update_parameters(self, weight_gradients: List[np.ndarray], bias_gradients: List[np.ndarray]):
        """
        Update network parameters using computed gradients.
        
        Args:
            weight_gradients: Gradients for weights
            bias_gradients: Gradients for biases
        """
        if self.optimizer_name == 'adam':
            # Adam optimizer implementation
            self.t += 1
            
            # Update weights
            for i in range(len(self.weights)):
                # Update momentum and velocity for weights
                self.m_weights[i] = 0.9 * self.m_weights[i] + 0.1 * weight_gradients[i]
                self.v_weights[i] = 0.999 * self.v_weights[i] + 0.001 * (weight_gradients[i] ** 2)
                
                # Bias correction
                m_corrected = self.m_weights[i] / (1 - 0.9 ** self.t)
                v_corrected = self.v_weights[i] / (1 - 0.999 ** self.t)
                
                # Update weights
                self.weights[i] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)
            
            # Update biases
            for i in range(len(self.biases)):
                # Update momentum and velocity for biases
                self.m_biases[i] = 0.9 * self.m_biases[i] + 0.1 * bias_gradients[i]
                self.v_biases[i] = 0.999 * self.v_biases[i] + 0.001 * (bias_gradients[i] ** 2)
                
                # Bias correction
                m_corrected = self.m_biases[i] / (1 - 0.9 ** self.t)
                v_corrected = self.v_biases[i] / (1 - 0.999 ** self.t)
                
                # Update biases
                self.biases[i] -= self.learning_rate * m_corrected / (np.sqrt(v_corrected) + 1e-8)
        else:
            # Standard gradient descent
            for i in range(len(self.weights)):
                self.weights[i] -= self.learning_rate * weight_gradients[i]
                self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        epochs: int = 1000, 
        verbose: bool = True,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> List[float]:
        """
        Train the neural network.
        
        Args:
            X: Training features
            y: Training targets
            epochs: Number of training epochs
            verbose: Whether to print progress
            validation_data: Optional validation data (X_val, y_val)
            
        Returns:
            List of training losses over epochs
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Forward pass
            output, layer_inputs, layer_outputs = self.forward(X)
            
            # Compute loss (mean squared error)
            loss = np.mean((output - y) ** 2)
            losses.append(loss)
            
            # Backward pass
            weight_gradients, bias_gradients = self.backward(X, y, output, layer_inputs, layer_outputs)
            
            # Update parameters
            self.update_parameters(weight_gradients, bias_gradients)
            
            # Compute validation loss if provided
            if validation_data is not None:
                X_val, y_val = validation_data
                val_output, _, _ = self.forward(X_val)
                val_loss = np.mean((val_output - y_val) ** 2)
                val_losses.append(val_loss)
            
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")
                if validation_data is not None:
                    print(f"Validation Loss: {val_loss:.6f}")
        
        return losses
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        output, _, _ = self.forward(X)
        return output


class MultiClassClassifier(NeuralNetwork):
    """
    Multi-class classifier using neural network with softmax output.
    """
    
    def __init__(
        self,
        layer_sizes: List[int],
        learning_rate: float = 0.001,
        optimizer: str = 'adam'
    ):
        """
        Initialize Multi-class Classifier.
        
        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., n_classes]
            learning_rate: Learning rate for optimization
            optimizer: Optimization algorithm
        """
        super().__init__(
            layer_sizes=layer_sizes,
            activation='relu',
            output_activation='softmax',  # Use softmax for multi-class
            learning_rate=learning_rate,
            optimizer=optimizer
        )
    
    def _get_activation(self, name: str) -> Callable:
        """Get activation function by name, including softmax."""
        if name == 'softmax':
            def softmax(x):
                # Subtract max for numerical stability
                exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return exp_x / np.sum(exp_x, axis=1, keepdims=True)
            return softmax
        else:
            return super()._get_activation(name)
    
    def _get_activation_derivative(self, name: str) -> Callable:
        """Get derivative of activation function by name."""
        if name == 'softmax':
            # For simplicity, we'll return a placeholder
            # The actual derivative of softmax is more complex and depends on the loss function
            return lambda x: np.ones_like(x)
        else:
            return super()._get_activation_derivative(name)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make class predictions on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class labels
        """
        probabilities = super().predict(X)
        return np.argmax(probabilities, axis=1)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities on new data.
        
        Args:
            X: Input features
            
        Returns:
            Predicted class probabilities
        """
        return super().predict(X)


def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error loss."""
    return np.mean((y_true - y_pred) ** 2)


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate cross-entropy loss.
    
    Args:
        y_true: True labels (one-hot encoded)
        y_pred: Predicted probabilities
        
    Returns:
        Cross-entropy loss
    """
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy score."""
    return np.mean(y_true == y_pred)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)