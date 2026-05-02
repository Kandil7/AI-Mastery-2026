"""
Neural Networks - Exercises
===========================

Practice problems for perceptrons, MLPs, and backpropagation.

Author: AI-Mastery-2026
"""

import numpy as np


# ============================================================================
# EXERCISE 1: Perceptron Implementation
# ============================================================================


def exercise_perceptron_basic():
    """
    Exercise 1: Basic Perceptron

    Create a simple perceptron for AND gate:
    - Input: two binary values (0 or 1)
    - Output: 1 only if both inputs are 1

    Data:
        (0,0) -> 0
        (0,1) -> 0
        (1,0) -> 0
        (1,1) -> 1
    """
    # Your code here
    # 1. Define training data
    # 2. Initialize weights randomly
    # 3. Train for a few epochs
    # 4. Test on all 4 inputs

    pass


# ============================================================================
# EXERCISE 2: Sigmoid Activation
# ============================================================================


def exercise_sigmoid():
    """
    Exercise 2: Implement Sigmoid Function

    Implement sigmoid and its derivative:
    - sigmoid(x) = 1 / (1 + e^(-x))
    - sigmoid_prime(x) = sigmoid(x) * (1 - sigmoid(x))

    Test with x = 0, x = 1, x = -1
    """

    def sigmoid(x):
        # Your code here
        pass

    def sigmoid_derivative(x):
        # Your code here
        pass

    # Test cases
    test_values = [0, 1, -1]
    print("Sigmoid tests:")
    for x in test_values:
        print(f"  sigmoid({x}) = {sigmoid(x):.4f}")
        print(f"  sigmoid'({x}) = {sigmoid_derivative(x):.4f}")


# ============================================================================
# EXERCISE 3: Forward Pass
# ============================================================================


def exercise_forward_pass():
    """
    Exercise 3: Compute Forward Pass

    Given a simple 2-layer network:
    - Input: [1, 2]
    - Weights1: [[0.1, 0.2], [0.3, 0.4]] (2x2)
    - Bias1: [0.1, 0.1]
    - Weights2: [0.5, 0.6] (2x1)
    - Bias2: 0.2

    Compute:
    1. z1 = x @ W1 + b1
    2. a1 = tanh(z1)
    3. z2 = a1 @ W2 + b2
    4. a2 = sigmoid(z2)

    Expected output: probability between 0 and 1
    """
    # Given values
    x = np.array([1, 2])
    W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    b1 = np.array([0.1, 0.1])
    W2 = np.array([0.5, 0.6])
    b2 = 0.2

    # Your code here
    pass


# ============================================================================
# EXERCISE 4: Backpropagation Gradient
# ============================================================================


def exercise_backprop_gradient():
    """
    Exercise 4: Compute Gradients

    Given:
    - Loss: MSE = (y - ŷ)²
    - Output activation: sigmoid
    - Single training sample

    Compute gradient dLoss/dW for output layer weights.

    Given:
    - Input to output layer (a1): [0.5, 0.8]
    - Target (y): 1
    - Prediction (a2): 0.7

    Steps:
    1. dLoss/da2 = 2(a2 - y)
    2. da2/dz2 = a2(1 - a2) [sigmoid derivative]
    3. dz2/dW2 = a1
    4. dLoss/dW2 = dLoss/da2 * da2/dz2 * dz2/dW2
    """
    # Given
    a1 = np.array([0.5, 0.8])
    y = 1
    a2 = 0.7

    # Your code here
    pass


# ============================================================================
# EXERCISE 5: XOR with MLP
# ============================================================================


def exercise_xor_mlp():
    """
    Exercise 5: Design MLP for XOR

    Show why we need hidden layers for XOR:

    Sketch the architecture:
    - Input: 2 neurons
    - Hidden: ? neurons (try 2)
    - Output: 1 neuron

    Explain:
    1. Without hidden layer: can we separate XOR?
    2. With hidden layer: what does the hidden represent?
    """
    print("XOR MLP Architecture Design")
    print("-" * 40)

    print("""
    Solution:
    
    Input (2) -> Hidden (2) -> Output (1)
    
    Hidden neurons can learn:
    - H1: activates for (0,1) OR (1,0)
    - H2: activates for (1,1)
    
    Then output can combine these properly!
    
    This creates a non-linear transformation
    that makes the data linearly separable.
    """)


# ============================================================================
# EXERCISE 6: Learning Rate Effects
# ============================================================================


def exercise_learning_rate():
    """
    Exercise 6: Compare Learning Rates

    For function f(x) = x², compare:
    - Learning rate = 0.1 (should converge)
    - Learning rate = 1.5 (should oscillate/diverge)

    Start from x = 4, run 10 iterations
    """

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    print("Learning Rate Comparison")
    print("-" * 40)

    for lr in [0.1, 1.5]:
        x = 4
        print(f"\nLearning rate = {lr}")
        print(f"Start: x = {x}, f(x) = {f(x)}")

        for i in range(10):
            x = x - lr * df(x)
            print(f"  Step {i + 1}: x = {x:.4f}, f(x) = {f(x):.4f}")


# ============================================================================
# EXERCISE 7: Regularization in Neural Networks
# ============================================================================


def exercise_regularization():
    """
    Exercise 7: Dropout vs L2

    Explain the difference between:
    - L2 Regularization: adds λ||W||² to loss
    - Dropout: randomly sets neurons to 0 during training

    When would you use each?
    """
    print("Regularization Comparison")
    print("-" * 40)

    print("""
    L2 Regularization:
    - Penalizes large weights
    - Always active, affects all parameters
    - Good for: preventing overfitting from large weights
    
    Dropout:
    - Temporarily disables neurons
    - Forces redundancy
    - Good for: preventing co-adaptation
                 (neurons relying too much on each other)
    
    Both can be used together!
    """)


# ============================================================================
# SOLUTIONS
# ============================================================================


def solutions():
    """Print solutions to exercises"""

    print("=" * 60)
    print("SOLUTIONS")
    print("=" * 60)

    # Solution 2
    print("\n--- Exercise 2: Sigmoid ---")

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_deriv(x):
        s = sigmoid(x)
        return s * (1 - s)

    for x in [0, 1, -1]:
        print(f"  sigmoid({x}) = {sigmoid(x):.4f}, derivative = {sigmoid_deriv(x):.4f}")

    # Solution 3
    print("\n--- Exercise 3: Forward Pass ---")
    x = np.array([1, 2])
    W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    b1 = np.array([0.1, 0.1])
    W2 = np.array([0.5, 0.6])
    b2 = 0.2

    z1 = x @ W1 + b1
    a1 = np.tanh(z1)
    z2 = a1 @ W2 + b2
    a2 = 1 / (1 + np.exp(-z2))

    print(f"  z1 = {z1}")
    print(f"  a1 (tanh) = {a1}")
    print(f"  z2 = {z2}")
    print(f"  a2 (sigmoid) = {a2:.4f}")


if __name__ == "__main__":
    print("Running exercises...")

    # Uncomment to see solutions
    # solutions()

    print("\nExercises ready! Uncomment solutions() call to see answers.")
