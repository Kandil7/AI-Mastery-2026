"""
Multi-Layer Perceptron (MLP) Implementation
=============================================

This module provides a complete MLP implementation from scratch
with forward propagation, backpropagation, and various optimizers.

Mathematical Foundation:
-------------------------
1. Forward Propagation:
   For each layer l:
       Z[l] = W[l] @ A[l-1] + b[l]
       A[l] = σ(Z[l])

   Where:
       W[l] - Weight matrix for layer l
       b[l] - Bias vector for layer l
       A[l-1] - Activation from previous layer
       σ - Activation function

2. Backpropagation:
   Compute gradients using chain rule:

   For output layer:
       dZ[L] = A[L] - Y  (for softmax + cross-entropy)
       dW[L] = (1/m) @ dZ[L] @ A[L-1].T
       db[L] = (1/m) @ sum(dZ[L])

   For hidden layer l:
       dZ[l] = (W[l+1].T @ dZ[l+1]) * σ'(Z[l])
       dW[l] = (1/m) @ dZ[l] @ A[l-1].T
       db[l] = (1/m) @ sum(dZ[l])

3. Gradient Descent Update:
   W = W - learning_rate * dW
   b = b - learning_rate * db

Author: AI-Mastery-2026
"""

import numpy as np
from typing import List, Tuple, Callable, Optional


class ActivationFunctions:
    """Collection of activation functions and their derivatives"""

    @staticmethod
    def sigmoid(z: np.ndarray) -> np.ndarray:
        """Sigmoid activation: σ(z) = 1 / (1 + e^(-z))"""
        # Clip to prevent overflow
        z_clipped = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z_clipped))

    @staticmethod
    def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
        """Derivative: σ'(z) = σ(z) * (1 - σ(z))"""
        sig = ActivationFunctions.sigmoid(z)
        return sig * (1 - sig)

    @staticmethod
    def relu(z: np.ndarray) -> np.ndarray:
        """ReLU: max(0, z)"""
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z: np.ndarray) -> np.ndarray:
        """ReLU derivative: 1 if z > 0, else 0"""
        return (z > 0).astype(float)

    @staticmethod
    def tanh(z: np.ndarray) -> np.ndarray:
        """Hyperbolic tangent"""
        return np.tanh(z)

    @staticmethod
    def tanh_derivative(z: np.ndarray) -> np.ndarray:
        """Tanh derivative: 1 - tanh²(z)"""
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def softmax(z: np.ndarray) -> np.ndarray:
        """Softmax for multi-class classification"""
        # Numerical stability: subtract max
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class Layer:
    """
    Single Layer of MLP

    Contains:
    - Weights (W)
    - Biases (b)
    - Pre-activation values (Z)
    - Activations (A)
    - Gradients (dW, db)
    """

    def __init__(self, n_inputs: int, n_neurons: int, activation: str = "relu"):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.activation = activation

        # Initialize weights using He initialization for ReLU
        if activation == "relu":
            scale = np.sqrt(2.0 / n_inputs)
        else:
            scale = np.sqrt(1.0 / n_inputs)

        self.weights = np.random.randn(n_inputs, n_neurons) * scale
        self.bias = np.zeros((1, n_neurons))

        # Cache for backward pass
        self.Z = None  # Pre-activation
        self.A = None  # Post-activation
        self.A_prev = None  # Previous activation

    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        """
        Forward pass through layer

        Args:
            A_prev: Activation from previous layer

        Returns:
            Activation output
        """
        self.A_prev = A_prev

        # Linear transformation
        self.Z = np.dot(A_prev, self.weights) + self.bias

        # Apply activation
        if self.activation == "sigmoid":
            self.A = ActivationFunctions.sigmoid(self.Z)
        elif self.activation == "relu":
            self.A = ActivationFunctions.relu(self.Z)
        elif self.activation == "tanh":
            self.A = ActivationFunctions.tanh(self.Z)
        elif self.activation == "softmax":
            self.A = ActivationFunctions.softmax(self.Z)
        elif self.activation == "none":
            self.A = self.Z
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        return self.A

    def backward(self, dA: np.ndarray) -> np.ndarray:
        """
        Backward pass through layer

        Args:
            dA: Gradient from next layer

        Returns:
            Gradient with respect to previous layer activation
        """
        m = dA.shape[0]  # Batch size

        # Compute activation derivative
        if self.activation == "sigmoid":
            dZ = dA * ActivationFunctions.sigmoid_derivative(self.Z)
        elif self.activation == "relu":
            dZ = dA * ActivationFunctions.relu_derivative(self.Z)
        elif self.activation == "tanh":
            dZ = dA * ActivationFunctions.tanh_derivative(self.Z)
        elif self.activation == "softmax" or self.activation == "none":
            dZ = dA
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

        # Compute gradients
        self.dW = np.dot(self.A_prev.T, dZ) / m
        self.db = np.sum(dZ, axis=0, keepdims=True) / m

        # Gradient with respect to previous layer
        dA_prev = np.dot(dZ, self.weights.T)

        return dA_prev


class MLP:
    """
    Multi-Layer Perceptron

    A fully-connected neural network with multiple layers.

    Architecture:
        Input -> Hidden Layers -> Output Layer

    Example:
        MLP with 2 hidden layers:
        Input(4) -> Dense(8, relu) -> Dense(8, relu) -> Dense(3, softmax)
    """

    def __init__(
        self,
        layer_sizes: List[int],
        activations: List[str],
        learning_rate: float = 0.01,
        loss: str = "crossentropy",
    ):
        """
        Initialize MLP

        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            activations: List of activation functions for each layer
            learning_rate: Learning rate for gradient descent
            loss: Loss function ('crossentropy' or 'mse')
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.loss = loss

        # Create layers
        self.layers = []
        for i in range(len(layer_sizes) - 1):
            layer = Layer(
                n_inputs=layer_sizes[i],
                n_neurons=layer_sizes[i + 1],
                activation=activations[i],
            )
            self.layers.append(layer)

        # Training history
        self.loss_history = []
        self.accuracy_history = []

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass through all layers

        Args:
            X: Input features

        Returns:
            Model predictions
        """
        A = X
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def compute_loss(self, Y_true: np.ndarray, Y_pred: np.ndarray) -> float:
        """
        Compute loss

        Args:
            Y_true: True labels (one-hot encoded for crossentropy)
            Y_pred: Predicted probabilities

        Returns:
            Loss value
        """
        if self.loss == "crossentropy":
            # Cross entropy loss with numerical stability
            epsilon = 1e-15
            Y_pred = np.clip(Y_pred, epsilon, 1 - epsilon)
            return -np.mean(np.sum(Y_true * np.log(Y_pred), axis=1))
        elif self.loss == "mse":
            return np.mean((Y_true - Y_pred) ** 2)
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

    def backward(self, Y_true: np.ndarray):
        """
        Backward pass through all layers

        Args:
            Y_true: True labels
        """
        m = Y_true.shape[0]

        # Compute initial gradient based on loss
        if self.loss == "crossentropy" and self.layers[-1].activation == "softmax":
            # Special case: softmax + cross-entropy gradient
            dA = self.layers[-1].A - Y_true
        else:
            dA = 2 * (self.layers[-1].A - Y_true) / m

        # Backward pass through layers (in reverse)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def update_weights(self):
        """Update weights using gradient descent"""
        for layer in self.layers:
            layer.weights -= self.learning_rate * layer.dW
            layer.bias -= self.learning_rate * layer.db

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        epochs: int = 1000,
        batch_size: Optional[int] = None,
        verbose: bool = True,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> "MLP":
        """
        Train the MLP

        Args:
            X: Training features
            Y: Training labels (one-hot encoded for multi-class)
            epochs: Number of training epochs
            batch_size: Batch size for mini-batch gradient descent
            verbose: Print progress
            validation_data: Tuple of (X_val, Y_val) for validation

        Returns:
            Self (for method chaining)
        """
        n_samples = X.shape[0]

        if batch_size is None:
            batch_size = n_samples

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            # Mini-batch training
            epoch_loss = 0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]

                # Forward pass
                Y_pred = self.forward(X_batch)

                # Compute loss
                loss = self.compute_loss(Y_batch, Y_pred)
                epoch_loss += loss
                n_batches += 1

                # Backward pass
                self.backward(Y_batch)

                # Update weights
                self.update_weights()

            # Average epoch loss
            avg_loss = epoch_loss / n_batches
            self.loss_history.append(avg_loss)

            # Compute training accuracy
            train_pred = self.predict(X)
            train_acc = np.mean(np.argmax(Y_pred, axis=1) == np.argmax(Y_batch, axis=1))
            self.accuracy_history.append(train_acc)

            # Validation
            if validation_data is not None and (epoch + 1) % 100 == 0:
                X_val, Y_val = validation_data
                val_pred = self.predict(X_val)
                val_acc = np.mean(
                    np.argmax(val_pred, axis=1) == np.argmax(Y_val, axis=1)
                )
                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}"
                )
            elif verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predicted probabilities
        """
        return self.forward(X)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels

        Args:
            X: Input features

        Returns:
            Predicted class indices
        """
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)

    def score(self, X: np.ndarray, Y: np.ndarray) -> float:
        """
        Calculate accuracy

        Args:
            X: Features
            Y: True labels

        Returns:
            Accuracy
        """
        predictions = self.predict_classes(X)
        true_labels = np.argmax(Y, axis=1)
        return np.mean(predictions == true_labels)


def demo_mlp():
    """Demonstrate MLP on various datasets"""
    print("=" * 60)
    print("MLP Demo")
    print("=" * 60)

    # ========================================
    # Demo 1: XOR Problem (classic non-linear)
    # ========================================
    print("\n--- XOR Problem ---")

    # XOR data
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    Y_xor = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])  # One-hot encoded

    print("XOR Training Data:")
    for i in range(len(X_xor)):
        print(f"  {X_xor[i]} -> {np.argmax(Y_xor[i])}")

    # Create MLP with 2 hidden layers
    mlp = MLP(
        layer_sizes=[2, 4, 4, 2],  # 2 inputs, 2 hidden layers, 2 outputs
        activations=["relu", "relu", "softmax"],
        learning_rate=0.5,
        loss="crossentropy",
    )

    # Train
    mlp.fit(X_xor, Y_xor, epochs=1000, verbose=False)

    # Test
    predictions = mlp.predict_classes(X_xor)
    print(f"\nPredictions: {predictions}")
    print(f"Accuracy: {mlp.score(X_xor, Y_xor) * 100:.0f}%")

    # ========================================
    # Demo 2: Iris-like classification
    # ========================================
    print("\n--- Multi-class Classification ---")

    np.random.seed(42)

    # Generate 3-class data
    n_samples = 50

    # Class 0
    X0 = np.random.randn(n_samples, 2) + np.array([-2, -2])
    # Class 1
    X1 = np.random.randn(n_samples, 2) + np.array([0, 2])
    # Class 2
    X2 = np.random.randn(n_samples, 2) + np.array([2, -2])

    X = np.vstack([X0, X1, X2])

    # One-hot encode labels
    y = np.array([0] * n_samples + [1] * n_samples + [2] * n_samples)
    Y = np.zeros((len(y), 3))
    Y[np.arange(len(y)), y] = 1

    print(f"Dataset: {len(y)} samples, 3 classes")

    # Train MLP
    mlp_iris = MLP(
        layer_sizes=[2, 8, 8, 3],
        activations=["relu", "relu", "softmax"],
        learning_rate=0.1,
        loss="crossentropy",
    )

    mlp_iris.fit(X, Y, epochs=500, verbose=False)

    accuracy = mlp_iris.score(X, Y)
    print(f"Accuracy: {accuracy * 100:.1f}%")

    return mlp


def plot_training_history(mlp: MLP):
    """Plot training history"""
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Loss
        ax1.plot(mlp.loss_history)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title("Training Loss")
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(mlp.accuracy_history)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Training Accuracy")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available")


if __name__ == "__main__":
    mlp = demo_mlp()
    plot_training_history(mlp)

    print("\n" + "=" * 60)
    print("MLP implementation complete!")
    print("=" * 60)
