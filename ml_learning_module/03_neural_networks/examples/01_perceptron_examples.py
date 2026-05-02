"""
Neural Networks - Worked Examples
==================================

Step-by-step examples for perceptron and MLP concepts.

Author: AI-Mastery-2026
"""

import numpy as np
import matplotlib.pyplot as plt


def example_perceptron_learning():
    """Example 1: Perceptron Learning Rule"""
    print("=" * 60)
    print("Example 1: Perceptron Learning Rule")
    print("=" * 60)

    # Simple AND gate data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 0, 0, 1])  # AND gate

    print("AND Gate Training Data:")
    for i in range(len(X)):
        print(f"  {X[i]} -> {y[i]}")

    # Initialize perceptron
    np.random.seed(42)
    weights = np.random.randn(2) * 0.5
    bias = 0.0
    learning_rate = 0.1

    print(f"\nInitial weights: {weights}, bias: {bias}")

    # Training (single pass for demonstration)
    print("\nTraining steps:")
    for epoch in range(10):
        errors = 0
        for i in range(len(X)):
            # Forward pass
            z = np.dot(X[i], weights) + bias
            prediction = 1 if z >= 0 else 0

            # Compute error
            error = y[i] - prediction

            if error != 0:
                errors += 1
                # Update weights (learning rule)
                weights += learning_rate * error * X[i]
                bias += learning_rate * error
                print(
                    f"  Sample {X[i]}: predicted={prediction}, actual={y[i]}, "
                    f"updated weights={weights}, bias={bias:.2f}"
                )

        if errors == 0:
            print(f"  Converged after {epoch + 1} epochs")
            break

    # Test
    print("\nFinal predictions:")
    for i in range(len(X)):
        z = np.dot(X[i], weights) + bias
        pred = 1 if z >= 0 else 0
        print(f"  {X[i]} -> {pred} (expected {y[i]})")


def example_activation_functions():
    """Example 2: Activation Functions Comparison"""
    print("\n" + "=" * 60)
    print("Example 2: Activation Functions")
    print("=" * 60)

    z = np.linspace(-5, 5, 100)

    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-z))
    sigmoid_deriv = sigmoid * (1 - sigmoid)

    # ReLU
    relu = np.maximum(0, z)
    relu_deriv = (z > 0).astype(float)

    # Tanh
    tanh = np.tanh(z)
    tanh_deriv = 1 - tanh**2

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Sigmoid
    axes[0, 0].plot(z, sigmoid, "b-", linewidth=2)
    axes[0, 0].set_title("Sigmoid")
    axes[0, 0].grid(True, alpha=0.3)

    axes[1, 0].plot(z, sigmoid_deriv, "r-", linewidth=2)
    axes[1, 0].set_title("Sigmoid Derivative")
    axes[1, 0].grid(True, alpha=0.3)

    # ReLU
    axes[0, 1].plot(z, relu, "b-", linewidth=2)
    axes[0, 1].set_title("ReLU")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 1].plot(z, relu_deriv, "r-", linewidth=2)
    axes[1, 1].set_title("ReLU Derivative")
    axes[1, 1].grid(True, alpha=0.3)

    # Tanh
    axes[0, 2].plot(z, tanh, "b-", linewidth=2)
    axes[0, 2].set_title("Tanh")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 2].plot(z, tanh_deriv, "r-", linewidth=2)
    axes[1, 2].set_title("Tanh Derivative")
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print properties
    print("\nActivation Function Properties:")
    print("-" * 40)
    print(f"Sigmoid: range [{sigmoid.min():.2f}, {sigmoid.max():.2f}]")
    print(f"ReLU: range [{relu.min():.2f}, {relu.max():.2f}]")
    print(f"Tanh: range [{tanh.min():.2f}, {tanh.max():.2f}]")


def example_backpropagation():
    """Example 3: Backpropagation Step by Step"""
    print("\n" + "=" * 60)
    print("Example 3: Backpropagation")
    print("=" * 60)

    # Simple network: 2 inputs -> 2 hidden -> 1 output
    # Forward: x -> h -> y

    # Simple data: OR gate
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [1]])  # OR gate

    print("OR Gate: Training to learn y = x1 OR x2")

    # Initialize network manually
    np.random.seed(42)

    # Network architecture: 2 -> 2 -> 1
    W1 = np.random.randn(2, 2) * 0.5  # Input to hidden (2x2)
    b1 = np.zeros((1, 2))  # Hidden bias
    W2 = np.random.randn(2, 1) * 0.5  # Hidden to output
    b2 = np.zeros((1, 1))  # Output bias

    print(f"\nInitial weights:")
    print(f"  W1 (input->hidden):\n{W1}")
    print(f"  W2 (hidden->output):\n{W2}")

    # Forward pass for first sample [0, 0]
    x = X[0:1]  # Keep 2D
    target = y[0:1]

    print(f"\nForward pass for x={x[0]}, target={target[0]}")

    # Hidden layer
    z1 = x @ W1 + b1
    a1 = np.tanh(z1)  # Activation
    print(f"  z1 (pre-activation): {z1}")
    print(f"  a1 (tanh): {a1}")

    # Output layer
    z2 = a1 @ W2 + b2
    a2 = 1 / (1 + np.exp(-z2))  # Sigmoid
    print(f"  z2 (pre-activation): {z2}")
    print(f"  a2 (sigmoid): {a2}")

    # Compute loss (cross-entropy or MSE)
    loss = np.mean((a2 - target) ** 2)
    print(f"  Loss: {loss:.4f}")

    # Backward pass
    print("\nBackward pass:")

    # Output layer gradient
    d_loss = 2 * (a2 - target) / target.shape[1]
    d_sigmoid = a2 * (1 - a2)
    delta2 = d_loss * d_sigmoid  # Output delta
    print(f"  delta2 (output): {delta2}")

    dW2 = a1.T @ delta2
    db2 = np.sum(delta2, axis=0, keepdims=True)
    print(f"  dW2: {dW2}")
    print(f"  db2: {db2}")

    # Hidden layer gradient
    delta1 = delta2 @ W2.T * (1 - a1**2)  # tanh derivative
    print(f"  delta1 (hidden): {delta1}")

    dW1 = x.T @ delta1
    db1 = np.sum(delta1, axis=0, keepdims=True)
    print(f"  dW1: {dW1}")
    print(f"  db1: {db1}")

    # Update weights (learning rate = 0.5)
    learning_rate = 0.5
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    print(f"\nUpdated weights:")
    print(f"  W1:\n{W1}")
    print(f"  W2:\n{W2}")


def example_xor_problem():
    """Example 4: Why We Need Hidden Layers"""
    print("\n" + "=" * 60)
    print("Example 4: XOR Problem - Why Hidden Layers?")
    print("=" * 60)

    # XOR data
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0, 1, 1, 0])

    print("XOR Data:")
    for i in range(len(X)):
        print(f"  {X[i]} -> {y[i]}")

    print("\nWhy single perceptron can't solve XOR:")
    print("-" * 40)
    print("Linear separability requirement:")
    print("  - Single perceptron creates a linear decision boundary")
    print("  - XOR data cannot be separated by a single line")
    print("")
    print("Visual representation:")
    print("  y")
    print("  1 |  X")
    print("    |     X")
    print("  0 +-------- X")
    print("    0     1")
    print("         x")
    print("")
    print("  No single line can separate X from X!")
    print("")
    print("Solution: Add hidden layer!")
    print("  - First layer: transforms space")
    print("  - Second layer: linearly separates")


def visualize_decision_boundary():
    """Example 5: Visualize Neural Network Decision Boundaries"""
    print("\n" + "=" * 60)
    print("Example 5: Decision Boundary Visualization")
    print("=" * 60)

    # Generate data
    np.random.seed(42)
    n_samples = 200

    # Create two interleaving half-circles
    theta = np.linspace(0, np.pi, n_samples // 2)
    X1 = np.column_stack(
        [
            np.cos(theta) + np.random.randn(n_samples // 2) * 0.1,
            np.sin(theta) + np.random.randn(n_samples // 2) * 0.1,
        ]
    )
    X2 = np.column_stack(
        [
            1 - np.cos(theta) + np.random.randn(n_samples // 2) * 0.1,
            1 - np.sin(theta) + np.random.randn(n_samples // 2) * 0.1,
        ]
    )

    X = np.vstack([X1, X2])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    # Note: Full MLP training would go here
    # For visualization, we'll create a mock decision boundary

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot points
    ax.scatter(
        X[y == 0, 0], X[y == 0, 1], c="blue", marker="o", label="Class 0", alpha=0.6
    )
    ax.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        c="red",
        marker="s",
        label="Class 1",
        ax=ax,
        alpha=0.6,
    )

    # Create mesh for decision boundary
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Mock decision boundary (non-linear)
    Z = np.sin(xx * 3) * np.cos(yy * 3) > 0
    ax.contourf(xx, yy, Z, alpha=0.2, colors=["blue", "red"])
    ax.contour(xx, yy, Z, colors="green", linewidths=2, linestyles="--")

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Non-linear Decision Boundary (MLP)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def example_gradient_descent_comparison():
    """Example 6: Compare different optimizers"""
    print("\n" + "=" * 60)
    print("Example 6: Optimizer Comparison")
    print("=" * 60)

    # Simple convex function: f(x) = x^2
    def f(x):
        return x**2

    def df(x):
        return 2 * x

    # Starting point
    x = -5

    # Learning rate
    lr = 0.1

    print("Optimizing f(x) = x² starting from x = -5")
    print("-" * 40)

    # Different optimizer simulations
    x_gd = x
    x_momentum = x
    v = 0  # velocity

    history_gd = [x_gd]
    history_momentum = [x_momentum]

    print(f"{'Epoch':<6} {'GD':<10} {'Momentum':<10}")
    print("-" * 26)

    for epoch in range(10):
        # Gradient Descent
        x_gd = x_gd - lr * df(x_gd)
        history_gd.append(x_gd)

        # Momentum
        beta = 0.9
        v = beta * v + lr * df(x_momentum)
        x_momentum = x_momentum - v
        history_momentum.append(x_momentum)

        if epoch < 5 or epoch == 9:
            print(f"{epoch + 1:<6} {x_gd:<10.4f} {x_momentum:<10.4f}")

    print(f"\nFinal: GD = {x_gd:.4f}, Momentum = {x_momentum:.4f}")
    print(f"True minimum at x = 0")


def run_all_examples():
    """Run all examples"""
    example_perceptron_learning()
    example_activation_functions()
    example_backpropagation()
    example_xor_problem()
    visualize_decision_boundary()
    example_gradient_descent_comparison()


if __name__ == "__main__":
    run_all_examples()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
