"""
Calculus and Optimization Module for Machine Learning

This module provides:
- Derivative and gradient computation
- Numerical differentiation
- Optimization algorithms (GD, SGD, Momentum, Adam)

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class OptimizationMethod(Enum):
    """Enumeration of optimization methods."""

    GD = "gd"  # Vanilla Gradient Descent
    SGD = "sgd"  # Stochastic Gradient Descent
    MOMENTUM = "momentum"  # Gradient Descent with Momentum
    RMSPROP = "rmsprop"  # RMSprop
    ADAM = "adam"  # Adam optimizer


class OptimizationResult:
    """Container for optimization results."""

    def __init__(
        self,
        x: np.ndarray,
        fun: float,
        nit: int,
        nfev: int,
        success: bool,
        message: str,
    ):
        self.x = x  # Optimal parameters
        self.fun = fun  # Function value at optimum
        self.nit = nit  # Number of iterations
        self.nfev = nfev  # Number of function evaluations
        self.success = success  # Whether optimization succeeded
        self.message = message  # Status message

    def __repr__(self):
        return f"OptimizationResult(x={self.x}, fun={self.fun:.6f}, nit={self.nit})"


class CalculusOperations:
    """
    Calculus operations for machine learning.

    Provides numerical differentiation and gradient computation.
    """

    def __init__(self, epsilon: float = 1e-7):
        """
        Initialize CalculusOperations.

        Args:
            epsilon: Step size for numerical differentiation.
        """
        self.epsilon = epsilon

    def derivative(self, f: Callable[[float], float], x: float) -> float:
        """
        Compute derivative of single-variable function using central difference.

        f'(x) ≈ (f(x+h) - f(x-h)) / 2h

        Args:
            f: Function to differentiate.
            x: Point at which to evaluate derivative.

        Returns:
            Derivative at point x.

        Example:
            >>> calc = CalculusOperations()
            >>> f = lambda x: x**2
            >>> calc.derivative(f, 3.0)  # Should be ~6.0
            6.0
        """
        h = self.epsilon
        return (f(x + h) - f(x - h)) / (2 * h)

    def partial_derivative(
        self, f: Callable[[np.ndarray], float], x: np.ndarray, idx: int
    ) -> float:
        """
        Compute partial derivative with respect to x[idx].

        Args:
            f: Multi-variable function.
            x: Point at which to evaluate.
            idx: Index of variable to differentiate.

        Returns:
            Partial derivative.
        """
        x = x.astype(float).copy()
        original = x[idx]

        x[idx] = original + self.epsilon
        f_plus = f(x)

        x[idx] = original - self.epsilon
        f_minus = f(x)

        return (f_plus - f_minus) / (2 * self.epsilon)

    def gradient(self, f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        """
        Compute gradient of multi-variable function.

        ∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]

        Args:
            f: Function to compute gradient for.
            x: Point at which to evaluate gradient.

        Returns:
            Gradient vector.

        Example:
            >>> calc = CalculusOperations()
            >>> f = lambda x: x[0]**2 + x[1]**2
            >>> calc.gradient(f, np.array([3.0, 4.0]))
            array([6., 8.])
        """
        x = x.astype(float)
        grad = np.zeros_like(x)

        for i in range(len(x)):
            grad[i] = self.partial_derivative(f, x, i)

        return grad

    def hessian(self, f: Callable[[np.ndarray], float], x: np.ndarray) -> np.ndarray:
        """
        Compute Hessian matrix of second derivatives.

        H[i,j] = ∂²f / (∂xᵢ ∂xⱼ)

        Args:
            f: Function to compute Hessian for.
            x: Point at which to evaluate.

        Returns:
            Hessian matrix.
        """
        n = len(x)
        hessian = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                if i == j:
                    # Second derivative w.r.t. same variable
                    hessian[i, j] = self._second_derivative(f, x, i)
                else:
                    # Mixed partial derivative
                    hessian[i, j] = self._mixed_derivative(f, x, i, j)

        return hessian

    def _second_derivative(
        self, f: Callable[[np.ndarray], float], x: np.ndarray, idx: int
    ) -> float:
        """Compute second derivative w.r.t. single variable."""
        x = x.astype(float).copy()
        original = x[idx]

        x[idx] = original + self.epsilon
        f_plus = f(x)

        x[idx] = original
        f_mid = f(x)

        x[idx] = original - self.epsilon
        f_minus = f(x)

        return (f_plus - 2 * f_mid + f_minus) / (self.epsilon**2)

    def _mixed_derivative(
        self, f: Callable[[np.ndarray], float], x: np.ndarray, idx1: int, idx2: int
    ) -> float:
        """Compute mixed partial derivative ∂²f/(∂xᵢ∂xⱼ)."""
        h = self.epsilon
        x = x.astype(float).copy()

        # f(x + h*e_i + h*e_j)
        x[idx1] += h
        x[idx2] += h
        f_pp = f(x.copy())

        # f(x + h*e_i - h*e_j)
        x[idx2] -= 2 * h
        f_pm = f(x.copy())

        # f(x - h*e_i + h*e_j)
        x[idx1] -= 2 * h
        x[idx2] += 2 * h
        f_mp = f(x.copy())

        # f(x - h*e_i - h*e_j)
        x[idx2] -= 2 * h
        f_mm = f(x.copy())

        return (f_pp - f_pm - f_mp + f_mm) / (4 * h**2)


class Optimizer:
    """
    Optimization algorithms for machine learning.

    Supports:
    - Gradient Descent (GD)
    - Stochastic Gradient Descent (SGD)
    - Gradient Descent with Momentum
    - RMSprop
    - Adam

    Example Usage:
        >>> opt = Optimizer(learning_rate=0.1, method='adam')
        >>> result = opt.minimize(lambda x: x[0]**2 + x[1]**2, x0=np.array([5.0, 5.0]))
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        method: str = "gd",
        momentum: float = 0.9,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        regularization: float = 0.0,
    ):
        """
        Initialize optimizer.

        Args:
            learning_rate: Step size for parameter updates.
            method: Optimization method ('gd', 'sgd', 'momentum', 'rmsprop', 'adam').
            momentum: Momentum factor for 'momentum' method.
            beta1: Exponential decay for first moment (Adam).
            beta2: Exponential decay for second moment (Adam).
            epsilon: Small constant for numerical stability.
            regularization: L2 regularization strength.
        """
        self.learning_rate = learning_rate
        self.method = method.lower()
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.regularization = regularization

        # State for momentum-based methods
        self._velocity = None
        self._m = None  # First moment (Adam)
        self._v = None  # Second moment (Adam)
        self._t = 0  # Time step (Adam)

        # History
        self.loss_history = []

    def _compute_gradient(self, f: Callable, x: np.ndarray) -> np.ndarray:
        """Compute gradient numerically."""
        calc = CalculusOperations()
        return calc.gradient(f, x)

    def minimize(
        self,
        f: Callable[[np.ndarray], float],
        x0: np.ndarray,
        n_iterations: int = 1000,
        tolerance: float = 1e-6,
        callback: Optional[Callable] = None,
    ) -> OptimizationResult:
        """
        Minimize function f starting from x0.

        Args:
            f: Objective function to minimize.
            x0: Initial parameter values.
            n_iterations: Maximum number of iterations.
            tolerance: Convergence tolerance for gradient magnitude.
            callback: Optional function called after each iteration.

        Returns:
            OptimizationResult containing optimal parameters and info.

        Example:
            >>> opt = Optimizer(learning_rate=0.1)
            >>> result = opt.minimize(lambda x: (x[0]-3)**2 + (x[1]-4)**2,
            ...                       x0=np.array([0.0, 0.0]))
            >>> print(result.x)  # Should be close to [3, 4]
        """
        x = x0.astype(float).copy()
        nfev = 0

        # Initialize optimizer state
        if self.method in ["momentum", "rmsprop", "adam"]:
            self._velocity = np.zeros_like(x)

        if self.method == "adam":
            self._m = np.zeros_like(x)
            self._v = np.zeros_like(x)
            self._t = 0

        self.loss_history = []

        for iteration in range(n_iterations):
            # Compute gradient
            grad = self._compute_gradient(f, x)
            nfev += 1

            # Check convergence
            grad_norm = np.linalg.norm(grad)
            if grad_norm < tolerance:
                return OptimizationResult(
                    x=x,
                    fun=float(f(x)),
                    nit=iteration,
                    nfev=nfev,
                    success=True,
                    message=f"Converged at iteration {iteration}",
                )

            # Store loss history
            loss = f(x) + (self.regularization / 2) * np.sum(x**2)
            self.loss_history.append(loss)

            # Apply regularization gradient (L2)
            if self.regularization > 0:
                grad = grad + self.regularization * x

            # Update based on method
            if self.method == "gd":
                x = x - self.learning_rate * grad

            elif self.method == "momentum":
                self._velocity = self.momentum * self._velocity + grad
                x = x - self.learning_rate * self._velocity

            elif self.method == "rmsprop":
                # Decay running average of squared gradients
                self._velocity = 0.9 * self._velocity + 0.1 * grad**2
                # Update with adaptive learning rate
                x = x - self.learning_rate * grad / (
                    np.sqrt(self._velocity) + self.epsilon
                )

            elif self.method == "adam":
                self._t += 1

                # Update biased first moment estimate
                self._m = self.beta1 * self._m + (1 - self.beta1) * grad

                # Update biased second moment estimate
                self._v = self.beta2 * self._v + (1 - self.beta2) * grad**2

                # Bias correction
                m_hat = self._m / (1 - self.beta1**self._t)
                v_hat = self._v / (1 - self.beta2**self._t)

                # Update
                x = x - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

            else:
                raise ValueError(f"Unknown method: {self.method}")

            # Call callback if provided
            if callback is not None:
                callback(x, iteration)

        return OptimizationResult(
            x=x,
            fun=float(f(x)),
            nit=n_iterations,
            nfev=nfev,
            success=True,
            message=f"Completed {n_iterations} iterations",
        )

    def fit(
        self, X: np.ndarray, y: np.ndarray, loss_fn: Callable, n_iterations: int = 1000
    ) -> np.ndarray:
        """
        Fit a model by minimizing loss function.

        This is a convenience method for common ML use cases.

        Args:
            X: Feature matrix (m, n).
            y: Target vector (m,).
            loss_fn: Loss function that takes (X, y, params) and returns loss.
            n_iterations: Number of training iterations.

        Returns:
            Optimal parameters.
        """

        # For linear regression demo
        def objective(params):
            m, n = X.shape
            theta = params[:-1].reshape(-1, 1)
            bias = params[-1]
            y_pred = X @ theta + bias
            mse = np.mean((y - y_pred.flatten()) ** 2)
            return mse + (self.regularization / 2) * np.sum(theta**2)

        n_features = X.shape[1]
        x0 = np.zeros(n_features + 1)

        result = self.minimize(objective, x0, n_iterations)

        return result.x


def numerical_integration(
    f: Callable, a: float, b: float, n: int = 1000, method: str = "simpson"
) -> float:
    """
    Numerically integrate function f from a to b.

    Args:
        f: Function to integrate.
        a: Lower bound.
        b: Upper bound.
        n: Number of intervals.
        method: 'trapezoid' or 'simpson'.

    Returns:
        Approximate integral value.
    """
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])

    if method == "trapezoid":
        return np.trapz(y, x)
    elif method == "simpson":
        return np.simpson(y, x)
    else:
        raise ValueError(f"Unknown method: {method}")


def find_root(
    f: Callable,
    x0: float,
    method: str = "newton",
    max_iter: int = 100,
    tolerance: float = 1e-7,
) -> float:
    """
    Find root (zero) of function using numerical methods.

    Args:
        f: Function for which to find root.
        x0: Initial guess.
        method: 'newton' or 'bisection'.
        max_iter: Maximum iterations.
        tolerance: Convergence tolerance.

    Returns:
        Root value.
    """
    calc = CalculusOperations()

    if method == "newton":
        x = x0
        for _ in range(max_iter):
            fx = f(x)
            if abs(fx) < tolerance:
                return x
            dfx = calc.derivative(f, x)
            if abs(dfx) < tolerance:
                raise ValueError("Derivative too small")
            x = x - fx / dfx
        return x

    elif method == "bisection":
        # Need bounds where f changes sign
        raise NotImplementedError("Bisection requires bracket")

    else:
        raise ValueError(f"Unknown method: {method}")


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Calculus and Optimization Module - Demonstration")
    print("=" * 60)

    # Test CalculusOperations
    calc = CalculusOperations()

    # Derivative of x^2 at x=3
    f = lambda x: x**2
    deriv = calc.derivative(f, 3.0)
    print(f"\nDerivative of x² at x=3: {deriv:.6f} (expected: 6.0)")

    # Gradient of x² + y² at (3, 4)
    f2 = lambda x: x[0] ** 2 + x[1] ** 2
    grad = calc.gradient(f2, np.array([3.0, 4.0]))
    print(f"Gradient of x² + y² at (3,4): {grad} (expected: [6, 8])")

    # Test Optimizer
    print("\n--- Optimization ---")

    # f(x,y) = (x-3)² + (y-4)², minimum at (3,4)
    target = np.array([3.0, 4.0])
    obj = lambda x: np.sum((x - target) ** 2)

    methods = ["gd", "momentum", "rmsprop", "adam"]

    for method in methods:
        opt = Optimizer(learning_rate=0.1, method=method)
        result = opt.minimize(obj, x0=np.array([0.0, 0.0]), n_iterations=200)

        print(f"\n{method.upper()}:")
        print(f"  Initial: [0.0, 0.0]")
        print(f"  Final: {result.x}")
        print(f"  Loss: {result.fun:.6f}")

    # Linear regression example
    print("\n--- Linear Regression with SGD ---")

    # Generate data: y = 3x + 5 + noise
    np.random.seed(42)
    X = np.linspace(0, 10, 50).reshape(-1, 1)
    y = 3 * X.flatten() + 5 + np.random.randn(50) * 2

    # Add bias term
    X_b = np.c_[np.ones(len(X)), X]

    opt = Optimizer(learning_rate=0.01, method="adam")
    result = opt.minimize(
        lambda params: np.mean((X_b @ params - y) ** 2),
        x0=np.array([0.0, 0.0]),
        n_iterations=1000,
    )

    print(f"True: y = 3x + 5")
    print(f"Learned: y = {result.x[1]:.2f}x + {result.x[0]:.2f}")

    print("\n" + "=" * 60)
