"""
Python for ML - Exercises
==========================

Practice problems for linear regression and classification.

Author: AI-Mastery-2026
"""

import numpy as np


# ============================================================================
# EXERCISE 1: Linear Regression from Scratch
# ============================================================================


def exercise_linear_regression_gd():
    """
    Exercise 1: Implement Linear Regression with Gradient Descent

    Given data points:
    X = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]

    Implement gradient descent to find:
    y = mx + b

    Steps:
    1. Initialize m=0, b=0, learning_rate=0.01
    2. For 1000 iterations:
       - Compute predictions: y_pred = m*X + b
       - Compute gradients:
         dm = -2 * sum(X * (y - y_pred)) / n
         db = -2 * sum(y - y_pred) / n
       - Update: m = m - lr*dm, b = b - lr*db
    3. Return final m, b
    """
    X = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])

    # Your code here
    pass


# ============================================================================
# EXERCISE 2: Normal Equation
# ============================================================================


def exercise_normal_equation():
    """
    Exercise 2: Solve Linear Regression with Normal Equation

    Given data:
    X = [[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]]  (with bias column)
    y = [2, 4, 5, 4, 5]

    Use normal equation:
    theta = (X^T * X)^(-1) * X^T * y

    Return coefficients: [b, m]
    """
    # Add bias column
    X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
    y = np.array([2, 4, 5, 4, 5])

    # Your code here
    pass


# ============================================================================
# EXERCISE 3: Mean Squared Error
# ============================================================================


def exercise_mse():
    """
    Exercise 3: Calculate MSE

    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]

    MSE = (1/n) * sum((y_true - y_pred)^2)
    """
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])

    # Your code here
    mse = None

    print(f"MSE: {mse}")
    # Expected: 0.375


# ============================================================================
# EXERCISE 4: R² Score
# ============================================================================


def exercise_r2_score():
    """
    Exercise 4: Calculate R² Score

    y_true = [1, 2, 3, 4, 5]
    y_pred = [1.1, 2.1, 2.9, 4.2, 4.8]

    R² = 1 - SS_res/SS_tot
    SS_res = sum((y_true - y_pred)^2)
    SS_tot = sum((y_true - mean(y_true))^2)
    """
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

    # Your code here
    pass


# ============================================================================
# EXERCISE 5: Train-Test Split
# ============================================================================


def exercise_train_test_split():
    """
    Exercise 5: Implement Train-Test Split

    Given:
    X = range(100)
    y = [x*2 for x in range(100)]
    test_size = 0.2

    Split into:
    - First 80% for training
    - Last 20% for testing
    """
    X = np.array(range(100))
    y = np.array([x * 2 for x in range(100)])
    test_size = 0.2

    # Your code here
    pass


# ============================================================================
# EXERCISE 6: Feature Scaling
# ============================================================================


def exercise_feature_scaling():
    """
    Exercise 6: Implement Z-Score Normalization

    Given features:
    X = [[1, 200], [2, 210], [3, 220], [4, 230]]

    Scale each column:
    X_scaled = (X - mean) / std

    After scaling, first column should be ~[-1.26, -0.63, 0, 0.63]
    """
    X = np.array([[1, 200], [2, 210], [3, 220], [4, 230]])

    # Your code here
    pass


# ============================================================================
# EXERCISE 7: Gradient Descent with Learning Rate
# ============================================================================


def exercise_lr_comparison():
    """
    Exercise 7: Compare Different Learning Rates

    Minimize f(x) = x^2 starting from x=4

    Compare:
    - lr=0.1 (good)
    - lr=1.1 (might oscillate)
    - lr=2.1 (will diverge)

    Run 10 iterations each and compare final x values
    """

    def f(x):
        return x**2

    def df(x):
        return 2 * x

    for lr in [0.1, 1.1, 2.1]:
        x = 4
        print(f"\nLearning rate: {lr}")
        for i in range(10):
            x = x - lr * df(x)
        print(f"  Final x: {x:.4f} (target: 0)")


# ============================================================================
# EXERCISE 8: Polynomial Features
# ============================================================================


def exercise_polynomial_features():
    """
    Exercise 8: Create Polynomial Features

    Given X = [1, 2, 3] and degree=2

    Create: [X^0, X^1, X^2]
    Result: [[1,1,1], [1,2,4], [1,3,9]]
    """
    X = np.array([1, 2, 3])
    degree = 2

    # Your code here
    pass


# ============================================================================
# EXERCISE 9: Regularization
# ============================================================================


def exercise_regularization_impact():
    """
    Exercise 9: Understand L1 vs L2 Regularization

    For weights = [3, -2, 5, -1], lambda = 0.5:

    L1 (Lasso): lambda * sum(|w|) = 0.5 * (3+2+5+1) = 5.5
    L2 (Ridge): lambda * sum(w^2) = 0.5 * (9+4+25+1) = 19.5

    Which penalizes more?
    """
    weights = np.array([3, -2, 5, -1])
    lam = 0.5

    # Your code here
    pass


# ============================================================================
# SOLUTIONS
# ============================================================================


def solutions():
    """Print solutions"""

    print("=" * 60)
    print("SOLUTIONS")
    print("=" * 60)

    # Solution 3
    print("\n--- Exercise 3: MSE ---")
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    mse = np.mean((y_true - y_pred) ** 2)
    print(f"MSE = {mse}")

    # Solution 4
    print("\n--- Exercise 4: R² Score ---")
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot
    print(f"R² = {r2:.4f}")

    # Solution 6
    print("\n--- Exercise 6: Feature Scaling ---")
    X = np.array([[1, 200], [2, 210], [3, 220], [4, 230]])
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X_scaled = (X - mean) / std
    print(f"Scaled X:\n{X_scaled}")


if __name__ == "__main__":
    print("Running ML exercises...")
    exercise_lr_comparison()
    # solutions()  # Uncomment to see more solutions
