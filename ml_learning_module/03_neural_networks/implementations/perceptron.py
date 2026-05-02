"""
Perceptron Implementation
=========================

This module provides a complete Perceptron implementation from scratch
with training and prediction capabilities.

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Optional, Callable


class Perceptron:
    """
    Single-layer Perceptron for Binary Classification

    The perceptron is the simplest form of a neural network - a single
    neuron that makes decisions by weighing input evidence.

    Mathematical Model:
        ŷ = σ(w₁x₁ + w₂x₂ + ... + wₙxₙ + b)

    Where:
        σ is the activation function (step function for basic perceptron)
        w are the weights
        b is the bias
        x are the inputs

    Learning Algorithm:
        For each misclassified sample:
            w = w + learning_rate * (y - ŷ) * x
            b = b + learning_rate * (y - ŷ)

    Limitations:
        - Only handles linearly separable data
        - Cannot solve XOR problem
        - Converges only if data is linearly separable
    """

    def __init__(
        self,
        n_inputs: int,
        learning_rate: float = 0.1,
        activation: str = "step",
        max_epochs: int = 1000,
    ):
        """
        Initialize the Perceptron

        Args:
            n_inputs: Number of input features
            learning_rate: Step size for weight updates
            activation: 'step', 'sigmoid', or 'ReLU'
            max_epochs: Maximum training iterations
        """
        self.n_inputs = n_inputs
        self.learning_rate = learning_rate
        self.activation = activation
        self.max_epochs = max_epochs

        # Initialize weights using Xavier initialization
        # For step function, small random values work well
        self.weights = np.random.randn(n_inputs) * 0.01
        self.bias = 0.0

        # Training history
        self.loss_history = []

    def _activate(self, z: np.ndarray) -> np.ndarray:
        """
        Apply activation function to weighted sum

        Args:
            z: Weighted sum (before activation)

        Returns:
            Activated output
        """
        if self.activation == "step":
            return (z >= 0).astype(float)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-z))
        elif self.activation == "ReLU":
            return np.maximum(0, z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _activation_derivative(self, z: np.ndarray) -> np.ndarray:
        """
        Compute derivative of activation function

        Used for backpropagation (though perceptron typically uses
        the update rule directly)

        Args:
            z: Pre-activation values

        Returns:
            Derivative of activation
        """
        if self.activation == "step":
            # Step function derivative is 0 almost everywhere
            # For gradient purposes, use 1 where z=0
            return np.ones_like(z)
        elif self.activation == "sigmoid":
            sig = self._activate(z)
            return sig * (1 - sig)
        elif self.activation == "ReLU":
            return (z > 0).astype(float)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass - compute predictions

        Args:
            X: Input features, shape (n_samples, n_features)

        Returns:
            Predicted class labels
        """
        # Compute weighted sum + bias
        z = np.dot(X, self.weights) + self.bias

        # Apply activation
        return self._activate(z)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions

        Args:
            X: Input features

        Returns:
            Predicted class labels (0 or 1)
        """
        return (self.forward(X) >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of class 1

        For sigmoid activation, returns probability
        For step/ReLU, returns binary prediction

        Args:
            X: Input features

        Returns:
            Probabilities or predictions
        """
        if self.activation == "sigmoid":
            return self.forward(X)
        else:
            return self.predict(X)

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> "Perceptron":
        """
        Train the perceptron using the perceptron learning rule

        For each sample, if misclassified:
            w = w + η * (y - ŷ) * x
            b = b + η * (y - ŷ)

        Args:
            X: Training features, shape (n_samples, n_features)
            y: Training labels (0 or 1)
            verbose: Print progress

        Returns:
            Self (for method chaining)
        """
        y = np.asarray(y)

        for epoch in range(self.max_epochs):
            # Forward pass
            predictions = self.forward(X)

            # Calculate error
            error = y - predictions

            # Update weights and bias
            # w = w + η * Σ (y - ŷ) * x
            self.weights += self.learning_rate * np.dot(error, X)
            self.bias += self.learning_rate * np.sum(error)

            # Track loss
            loss = np.mean(np.abs(error))
            self.loss_history.append(loss)

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{self.max_epochs}, Loss: {loss:.4f}")

            # Early stopping if perfect
            if loss == 0:
                if verbose:
                    print(f"Converged at epoch {epoch + 1}")
                break

        return self

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score

        Args:
            X: Test features
            y: True labels

        Returns:
            Accuracy (0 to 1)
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def get_params(self) -> dict:
        """Get model parameters"""
        return {
            "weights": self.weights,
            "bias": self.bias,
            "n_inputs": self.n_inputs,
            "learning_rate": self.learning_rate,
            "activation": self.activation,
        }

    def set_params(self, weights: np.ndarray, bias: float):
        """Set model parameters"""
        self.weights = weights
        self.bias = bias


def create_and_test_perceptron():
    """Create and test a perceptron on linearly separable data"""
    print("=" * 60)
    print("Perceptron Demo")
    print("=" * 60)

    # Create linearly separable data
    np.random.seed(42)

    # Class 0: centered at (-2, -2)
    X0 = np.random.randn(50, 2) + np.array([-2, -2])
    y0 = np.zeros(50)

    # Class 1: centered at (2, 2)
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    y1 = np.ones(50)

    # Combine
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    print(f"Dataset: {len(y)} samples, {X.shape[1]} features")
    print(f"Class 0: {np.sum(y == 0)}, Class 1: {np.sum(y == 1)}")

    # Create and train perceptron
    perceptron = Perceptron(n_inputs=2, learning_rate=0.1, max_epochs=500)
    perceptron.fit(X, y, verbose=True)

    # Evaluate
    train_accuracy = perceptron.score(X, y)
    print(f"\nTraining Accuracy: {train_accuracy * 100:.2f}%")

    # Visualize decision boundary
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot data points
        ax.scatter(X0[:, 0], X0[:, 1], c="blue", marker="o", label="Class 0", alpha=0.6)
        ax.scatter(X1[:, 0], X1[:, 1], c="red", marker="s", label="Class 1", alpha=0.6)

        # Plot decision boundary
        # Decision boundary: w1*x1 + w2*x2 + b = 0
        # Solve for x2: x2 = -(w1*x1 + b) / w2
        x1_range = np.linspace(-6, 6, 100)
        w = perceptron.weights
        b = perceptron.bias

        if abs(w[1]) > 1e-10:
            x2_boundary = -(w[0] * x1_range + b) / w[1]
            ax.plot(x1_range, x2_boundary, "g-", linewidth=2, label="Decision Boundary")

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_title(
            f"Perceptron Decision Boundary (Accuracy: {train_accuracy * 100:.1f}%)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    except ImportError:
        print("Matplotlib not available for visualization")

    return perceptron


def demonstrate_limitations():
    """Demonstrate perceptron limitations with XOR data"""
    print("\n" + "=" * 60)
    print("Perceptron Limitation: XOR Problem")
    print("=" * 60)

    # XOR data - NOT linearly separable
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([0, 1, 1, 0])  # XOR logic

    print("XOR Data:")
    for i in range(len(X_xor)):
        print(f"  {X_xor[i]} -> {y_xor[i]}")

    # Try to train perceptron
    perceptron = Perceptron(n_inputs=2, learning_rate=0.1, max_epochs=100)
    perceptron.fit(X_xor, y_xor, verbose=False)

    predictions = perceptron.predict(X_xor)
    accuracy = perceptron.score(X_xor, y_xor)

    print(f"\nPerceptron predictions: {predictions}")
    print(f"Accuracy: {accuracy * 100:.0f}%")
    print("\nNote: Perceptron cannot solve XOR (not linearly separable)")
    print("This is why we need multi-layer perceptrons (MLPs)!")


if __name__ == "__main__":
    # Run demo
    create_and_test_perceptron()
    demonstrate_limitations()

    print("\n" + "=" * 60)
    print("Perceptron implementation complete!")
    print("=" * 60)
