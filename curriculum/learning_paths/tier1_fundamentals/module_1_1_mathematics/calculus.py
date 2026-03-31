"""
Calculus Module for Machine Learning.

This module provides calculus operations essential for ML optimization,
including derivatives, gradients, partial derivatives, chain rule, and optimization algorithms.

Example Usage:
    >>> import numpy as np
    >>> from calculus import CalculusOperations, Optimizer
    >>> 
    >>> # Define a function
    >>> def f(x):
    ...     return x**2 + 2*x + 1
    >>> 
    >>> # Compute numerical derivative
    >>> calc = CalculusOperations()
    >>> derivative = calc.numerical_derivative(f, 3.0)
    >>> print(f"f'(3) = {derivative}")  # Should be 8
    >>> 
    >>> # Gradient descent optimization
    >>> opt = Optimizer(learning_rate=0.1)
    >>> x_min = opt.gradient_descent(f, calc.numerical_gradient_1d, x0=10.0)
    >>> print(f"Minimum at x = {x_min}")  # Should be close to -1
"""

from typing import Callable, Union, List, Tuple, Optional, Dict, Any
import numpy as np
from numpy.typing import ArrayLike
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

ScalarFunction = Callable[[float], float]
VectorFunction = Callable[[np.ndarray], float]
GradientFunction = Callable[[np.ndarray], np.ndarray]
Vector = Union[np.ndarray, List[float]]


class OptimizationMethod(Enum):
    """Optimization methods for gradient-based optimization."""
    GRADIENT_DESCENT = "gradient_descent"
    MOMENTUM = "momentum"
    ADAM = "adam"
    RMSprop = "rmsprop"
    ADAGRAD = "adagrad"


@dataclass
class OptimizationResult:
    """Result of an optimization run."""
    x: np.ndarray
    fun: float
    n_iterations: int
    converged: bool
    history: Dict[str, List[float]]
    gradient_norm: float


class CalculusOperations:
    """
    Calculus operations for machine learning mathematics.
    
    This class provides methods for:
    - Numerical differentiation (finite differences)
    - Gradient computation
    - Partial derivatives
    - Jacobian and Hessian matrices
    - Chain rule implementation
    - Taylor series approximation
    
    Attributes:
        epsilon (float): Step size for numerical differentiation.
    
    Example:
        >>> calc = CalculusOperations(epsilon=1e-5)
        >>> def f(x): return x**2
        >>> calc.numerical_derivative(f, 3.0)
        6.0000000000...
    """
    
    def __init__(self, epsilon: float = 1e-5):
        """
        Initialize CalculusOperations.
        
        Args:
            epsilon: Step size for numerical differentiation. Default: 1e-5.
        """
        self.epsilon = epsilon
        logger.debug(f"CalculusOperations initialized with epsilon={epsilon}")
    
    def numerical_derivative(
        self, 
        f: ScalarFunction, 
        x: float,
        method: str = 'central'
    ) -> float:
        """
        Compute numerical derivative using finite differences.
        
        Methods:
        - 'forward': f'(x) ≈ (f(x+h) - f(x)) / h
        - 'backward': f'(x) ≈ (f(x) - f(x-h)) / h
        - 'central': f'(x) ≈ (f(x+h) - f(x-h)) / (2h) [most accurate]
        
        Args:
            f: Function to differentiate.
            x: Point at which to evaluate derivative.
            method: Finite difference method. Default: 'central'.
        
        Returns:
            float: Approximate derivative value.
        
        Raises:
            ValueError: If method is not supported.
        
        Example:
            >>> calc = CalculusOperations()
            >>> def f(x): return x**3
            >>> calc.numerical_derivative(f, 2.0)
            12.0  # Exact: 3*2² = 12
        """
        h = self.epsilon
        
        if method == 'forward':
            result = (f(x + h) - f(x)) / h
        elif method == 'backward':
            result = (f(x) - f(x - h)) / h
        elif method == 'central':
            result = (f(x + h) - f(x - h)) / (2 * h)
        else:
            error_msg = f"Method must be 'forward', 'backward', or 'central', got '{method}'"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"Numerical derivative at x={x}: {result}")
        return result
    
    def numerical_gradient(
        self, 
        f: VectorFunction, 
        x: np.ndarray,
        method: str = 'central'
    ) -> np.ndarray:
        """
        Compute numerical gradient of a scalar function.
        
        The gradient is the vector of partial derivatives:
        ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
        
        Args:
            f: Scalar function f: ℝⁿ → ℝ.
            x: Point at which to evaluate gradient.
            method: Finite difference method. Default: 'central'.
        
        Returns:
            np.ndarray: Gradient vector.
        
        Example:
            >>> calc = CalculusOperations()
            >>> def f(x): return x[0]**2 + x[1]**2
            >>> x = np.array([3.0, 4.0])
            >>> grad = calc.numerical_gradient(f, x)
            >>> np.allclose(grad, [6.0, 8.0])
            True
        """
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = len(x_arr)
        gradient = np.zeros(n)
        
        for i in range(n):
            # Create perturbation in direction i
            e_i = np.zeros(n)
            e_i[i] = self.epsilon
            
            if method == 'central':
                gradient[i] = (f(x_arr + e_i) - f(x_arr - e_i)) / (2 * self.epsilon)
            elif method == 'forward':
                gradient[i] = (f(x_arr + e_i) - f(x_arr)) / self.epsilon
            else:  # backward
                gradient[i] = (f(x_arr) - f(x_arr - e_i)) / self.epsilon
        
        logger.debug(f"Gradient computed at x={x_arr}, ||∇f||={np.linalg.norm(gradient):.6f}")
        return gradient
    
    def numerical_gradient_1d(self, f: ScalarFunction, x: float) -> float:
        """
        Compute numerical gradient for 1D functions (convenience wrapper).
        
        Args:
            f: Scalar function f: ℝ → ℝ.
            x: Point at which to evaluate.
        
        Returns:
            float: Derivative value.
        
        Example:
            >>> calc = CalculusOperations()
            >>> def f(x): return x**2
            >>> calc.numerical_gradient_1d(f, 3.0)
            6.0
        """
        return self.numerical_derivative(f, x)
    
    def partial_derivative(
        self,
        f: Callable,
        x: np.ndarray,
        var_index: int,
        method: str = 'central'
    ) -> float:
        """
        Compute partial derivative with respect to a specific variable.
        
        Args:
            f: Function f: ℝⁿ → ℝ.
            x: Point at which to evaluate.
            var_index: Index of variable to differentiate with respect to.
            method: Finite difference method. Default: 'central'.
        
        Returns:
            float: Partial derivative value.
        
        Example:
            >>> calc = CalculusOperations()
            >>> def f(x): return x[0]**2 * x[1] + x[1]**2
            >>> x = np.array([2.0, 3.0])
            >>> # ∂f/∂x₀ = 2*x₀*x₁ = 2*2*3 = 12
            >>> calc.partial_derivative(f, x, var_index=0)
            12.0
        """
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = len(x_arr)
        
        if var_index < 0 or var_index >= n:
            error_msg = f"var_index {var_index} out of range [0, {n})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        e_i = np.zeros(n)
        e_i[var_index] = self.epsilon
        
        if method == 'central':
            result = (f(x_arr + e_i) - f(x_arr - e_i)) / (2 * self.epsilon)
        elif method == 'forward':
            result = (f(x_arr + e_i) - f(x_arr)) / self.epsilon
        else:
            result = (f(x_arr) - f(x_arr - e_i)) / self.epsilon
        
        logger.debug(f"Partial derivative ∂f/∂x_{var_index} at x={x_arr}: {result}")
        return result
    
    def jacobian(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray
    ) -> np.ndarray:
        """
        Compute the Jacobian matrix of a vector-valued function.
        
        For f: ℝⁿ → ℝᵐ, the Jacobian J is an m×n matrix:
        J[i,j] = ∂fᵢ/∂xⱼ
        
        Args:
            f: Vector-valued function.
            x: Point at which to evaluate.
        
        Returns:
            np.ndarray: Jacobian matrix (m×n).
        
        Example:
            >>> calc = CalculusOperations()
            >>> def f(x):
            ...     return np.array([x[0]**2, x[0]*x[1], x[1]**2])
            >>> x = np.array([2.0, 3.0])
            >>> J = calc.jacobian(f, x)
            >>> # J = [[2*x₀, 0], [x₁, x₀], [0, 2*x₁]]
            >>> # J = [[4, 0], [3, 2], [0, 6]]
            >>> np.allclose(J, [[4, 0], [3, 2], [0, 6]])
            True
        """
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = len(x_arr)
        
        # Evaluate function at x
        f_x = np.asarray(f(x_arr)).flatten()
        m = len(f_x)
        
        jacobian = np.zeros((m, n))
        
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = self.epsilon
            
            f_plus = np.asarray(f(x_arr + e_j)).flatten()
            f_minus = np.asarray(f(x_arr - e_j)).flatten()
            
            jacobian[:, j] = (f_plus - f_minus) / (2 * self.epsilon)
        
        logger.debug(f"Jacobian computed: shape {jacobian.shape}")
        return jacobian
    
    def hessian(
        self,
        f: VectorFunction,
        x: np.ndarray
    ) -> np.ndarray:
        """
        Compute the Hessian matrix of a scalar function.
        
        The Hessian H is an n×n matrix of second partial derivatives:
        H[i,j] = ∂²f/∂xᵢ∂xⱼ
        
        For smooth functions, H is symmetric (Schwarz's theorem).
        
        Args:
            f: Scalar function f: ℝⁿ → ℝ.
            x: Point at which to evaluate.
        
        Returns:
            np.ndarray: Hessian matrix (n×n).
        
        Example:
            >>> calc = CalculusOperations()
            >>> def f(x): return x[0]**2 + 2*x[0]*x[1] + x[1]**2
            >>> x = np.array([1.0, 2.0])
            >>> H = calc.hessian(f, x)
            >>> # H = [[2, 2], [2, 2]] (constant for quadratic)
            >>> np.allclose(H, [[2, 2], [2, 2]])
            True
        """
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        n = len(x_arr)
        
        hessian = np.zeros((n, n))
        f_x = f(x_arr)
        
        for i in range(n):
            for j in range(i, n):
                e_i = np.zeros(n)
                e_i[i] = self.epsilon
                e_j = np.zeros(n)
                e_j[j] = self.epsilon
                
                # Second-order central difference
                f_pp = f(x_arr + e_i + e_j)
                f_pm = f(x_arr + e_i - e_j)
                f_mp = f(x_arr - e_i + e_j)
                f_mm = f(x_arr - e_i - e_j)
                
                hessian[i, j] = (f_pp - f_pm - f_mp + f_mm) / (4 * self.epsilon ** 2)
                
                # Hessian is symmetric
                if i != j:
                    hessian[j, i] = hessian[i, j]
        
        logger.debug(f"Hessian computed: shape {hessian.shape}")
        return hessian
    
    def taylor_approximation(
        self,
        f: VectorFunction,
        x0: np.ndarray,
        x: np.ndarray,
        order: int = 2
    ) -> float:
        """
        Compute Taylor series approximation of a function.
        
        Taylor expansion around x0:
        f(x) ≈ f(x0) + ∇f(x0)·(x-x0) + ½(x-x0)ᵀ·H(x0)·(x-x0) + ...
        
        Args:
            f: Scalar function.
            x0: Expansion point.
            x: Point at which to evaluate approximation.
            order: Order of approximation (1 or 2). Default: 2.
        
        Returns:
            float: Approximated function value.
        
        Example:
            >>> calc = CalculusOperations()
            >>> import numpy as np
            >>> def f(x): return np.exp(x[0]) + x[1]**2
            >>> x0 = np.array([0.0, 0.0])
            >>> x = np.array([0.1, 0.1])
            >>> approx = calc.taylor_approximation(f, x0, x, order=2)
            >>> exact = f(x)
            >>> abs(approx - exact) < 0.01
            True
        """
        x0_arr = np.asarray(x0, dtype=np.float64).flatten()
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        dx = x_arr - x0_arr
        
        # Zeroth order: f(x0)
        result = f(x0_arr)
        
        if order >= 1:
            # First order: ∇f(x0)·dx
            grad = self.numerical_gradient(f, x0_arr)
            result += np.dot(grad, dx)
        
        if order >= 2:
            # Second order: ½·dxᵀ·H·dx
            hess = self.hessian(f, x0_arr)
            result += 0.5 * dx @ hess @ dx
        
        logger.debug(f"Taylor approximation (order={order}) at x={x_arr}")
        return result
    
    def chain_rule(
        self,
        outer_f: Callable[[float], float],
        inner_f: Callable[[float], float],
        x: float
    ) -> float:
        """
        Compute derivative using the chain rule.
        
        For composite function h(x) = f(g(x)):
        h'(x) = f'(g(x)) · g'(x)
        
        Args:
            outer_f: Outer function f.
            inner_f: Inner function g.
            x: Point at which to evaluate.
        
        Returns:
            float: Derivative of composite function.
        
        Example:
            >>> calc = CalculusOperations()
            >>> import numpy as np
            >>> # h(x) = sin(x²)
            >>> outer = np.sin
            >>> inner = lambda x: x**2
            >>> # h'(x) = cos(x²) · 2x
            >>> # h'(1) = cos(1) · 2 ≈ 1.08
            >>> calc.chain_rule(outer, inner, 1.0)  # doctest: +ELLIPSIS
            1.08...
        """
        # Compute inner function value
        g_x = inner_f(x)
        
        # Compute derivatives
        f_prime_gx = self.numerical_derivative(outer_f, g_x)
        g_prime_x = self.numerical_derivative(inner_f, x)
        
        result = f_prime_gx * g_prime_x
        logger.debug(f"Chain rule: f'(g({x}))·g'({x}) = {result}")
        return result
    
    def directional_derivative(
        self,
        f: VectorFunction,
        x: np.ndarray,
        direction: np.ndarray
    ) -> float:
        """
        Compute directional derivative in a given direction.
        
        The directional derivative is: D_v f(x) = ∇f(x) · v̂
        where v̂ is the unit vector in the direction.
        
        Args:
            f: Scalar function.
            x: Point at which to evaluate.
            direction: Direction vector (will be normalized).
        
        Returns:
            float: Directional derivative value.
        
        Example:
            >>> calc = CalculusOperations()
            >>> def f(x): return x[0]**2 + x[1]**2
            >>> x = np.array([1.0, 1.0])
            >>> direction = np.array([1.0, 0.0])  # x-direction
            >>> calc.directional_derivative(f, x, direction)
            2.0  # ∂f/∂x = 2x = 2 at x=1
        """
        x_arr = np.asarray(x, dtype=np.float64).flatten()
        dir_arr = np.asarray(direction, dtype=np.float64).flatten()
        
        # Normalize direction
        norm = np.linalg.norm(dir_arr)
        if norm < self.epsilon:
            error_msg = "Direction vector cannot be zero"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        v_hat = dir_arr / norm
        
        # Directional derivative = gradient · direction
        grad = self.numerical_gradient(f, x_arr)
        result = np.dot(grad, v_hat)
        
        logger.debug(f"Directional derivative in direction {v_hat}: {result}")
        return result
    
    def gradient_descent_step(
        self,
        f: VectorFunction,
        x: np.ndarray,
        learning_rate: float = 0.01
    ) -> np.ndarray:
        """
        Compute one step of gradient descent.
        
        Update rule: x_new = x - lr · ∇f(x)
        
        Args:
            f: Function to minimize.
            x: Current point.
            learning_rate: Step size. Default: 0.01.
        
        Returns:
            np.ndarray: Updated point.
        
        Example:
            >>> calc = CalculusOperations()
            >>> def f(x): return x[0]**2 + x[1]**2
            >>> x = np.array([3.0, 4.0])
            >>> x_new = calc.gradient_descent_step(f, x, learning_rate=0.1)
            >>> np.all(x_new < x)  # Moved toward minimum
            True
        """
        grad = self.numerical_gradient(f, x)
        x_new = x - learning_rate * grad
        logger.debug(f"Gradient descent step: ||∇f||={np.linalg.norm(grad):.6f}")
        return x_new


class Optimizer:
    """
    Optimization algorithms for machine learning.
    
    This class implements various gradient-based optimization methods:
    - Gradient Descent
    - Momentum
    - Adam
    - RMSprop
    - Adagrad
    
    Example:
        >>> opt = Optimizer(learning_rate=0.01, method='adam')
        >>> def f(x): return x[0]**2 + x[1]**2
        >>> result = opt.minimize(f, x0=np.array([5.0, 5.0]))
        >>> np.allclose(result.x, [0, 0], atol=0.1)
        True
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        method: OptimizationMethod = OptimizationMethod.GRADIENT_DESCENT,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0
    ):
        """
        Initialize optimizer.
        
        Args:
            learning_rate: Step size for updates. Default: 0.001.
            method: Optimization method. Default: GRADIENT_DESCENT.
            beta1: Momentum decay (for Adam/Momentum). Default: 0.9.
            beta2: Squared gradient decay (for Adam/RMSprop). Default: 0.999.
            epsilon: Numerical stability constant. Default: 1e-8.
            weight_decay: L2 regularization strength. Default: 0.0.
        """
        self.learning_rate = learning_rate
        self.method = method
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        
        # State variables
        self._momentum = None
        self._velocity = None
        self._m = None  # First moment (Adam)
        self._v = None  # Second moment (Adam)
        self._t = 0     # Time step (Adam)
        
        logger.debug(f"Optimizer initialized: {method.value}, lr={learning_rate}")
    
    def reset_state(self) -> None:
        """Reset optimizer state for new optimization run."""
        self._momentum = None
        self._velocity = None
        self._m = None
        self._v = None
        self._t = 0
        logger.debug("Optimizer state reset")
    
    def _compute_gradient(
        self, 
        f: VectorFunction, 
        x: np.ndarray,
        grad_fn: Optional[GradientFunction] = None
    ) -> np.ndarray:
        """Compute gradient using provided function or numerical approximation."""
        if grad_fn is not None:
            return grad_fn(x)
        else:
            calc = CalculusOperations()
            return calc.numerical_gradient(f, x)
    
    def _update_step(self, x: np.ndarray, grad: np.ndarray) -> np.ndarray:
        """Compute parameter update based on optimization method."""
        n = len(x)
        
        # Add weight decay to gradient
        if self.weight_decay > 0:
            grad = grad + self.weight_decay * x
        
        if self.method == OptimizationMethod.GRADIENT_DESCENT:
            return x - self.learning_rate * grad
        
        elif self.method == OptimizationMethod.MOMENTUM:
            if self._momentum is None:
                self._momentum = np.zeros(n)
            
            self._momentum = self.beta1 * self._momentum + grad
            return x - self.learning_rate * self._momentum
        
        elif self.method == OptimizationMethod.RMSprop:
            if self._velocity is None:
                self._velocity = np.zeros(n)
            
            self._velocity = self.beta2 * self._velocity + (1 - self.beta2) * grad ** 2
            return x - self.learning_rate * grad / (np.sqrt(self._velocity) + self.epsilon)
        
        elif self.method == OptimizationMethod.ADAGRAD:
            if self._velocity is None:
                self._velocity = np.zeros(n)
            
            self._velocity += grad ** 2
            return x - self.learning_rate * grad / (np.sqrt(self._velocity) + self.epsilon)
        
        elif self.method == OptimizationMethod.ADAM:
            if self._m is None:
                self._m = np.zeros(n)
                self._v = np.zeros(n)
                self._t = 0
            
            self._t += 1
            self._m = self.beta1 * self._m + (1 - self.beta1) * grad
            self._v = self.beta2 * self._v + (1 - self.beta2) * grad ** 2
            
            # Bias correction
            m_hat = self._m / (1 - self.beta1 ** self._t)
            v_hat = self._v / (1 - self.beta2 ** self._t)
            
            return x - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        else:
            error_msg = f"Unknown optimization method: {self.method}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def minimize(
        self,
        f: VectorFunction,
        x0: np.ndarray,
        grad_fn: Optional[GradientFunction] = None,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        callback: Optional[Callable[[int, np.ndarray, float], None]] = None
    ) -> OptimizationResult:
        """
        Minimize a function using gradient-based optimization.
        
        Args:
            f: Function to minimize.
            x0: Initial point.
            grad_fn: Optional gradient function. If None, uses numerical gradient.
            max_iterations: Maximum iterations. Default: 1000.
            tolerance: Convergence tolerance. Default: 1e-6.
            callback: Optional callback(iteration, x, f(x)).
        
        Returns:
            OptimizationResult: Optimization result with history.
        
        Example:
            >>> opt = Optimizer(learning_rate=0.1)
            >>> def f(x): return x[0]**2 + x[1]**2
            >>> result = opt.minimize(f, x0=np.array([5.0, 5.0]))
            >>> result.converged
            True
            >>> np.allclose(result.x, [0, 0], atol=0.1)
            True
        """
        self.reset_state()
        x = np.asarray(x0, dtype=np.float64).flatten()
        
        history = {
            'f': [],
            'grad_norm': [],
            'x': []
        }
        
        converged = False
        prev_f = f(x)
        history['f'].append(float(prev_f))
        history['x'].append(x.copy().tolist())
        
        for iteration in range(max_iterations):
            grad = self._compute_gradient(f, x, grad_fn)
            grad_norm = np.linalg.norm(grad)
            
            history['grad_norm'].append(float(grad_norm))
            
            # Check convergence
            if grad_norm < tolerance:
                converged = True
                logger.debug(f"Converged at iteration {iteration}, ||∇f||={grad_norm:.2e}")
                break
            
            # Update parameters
            x = self._update_step(x, grad)
            
            # Evaluate new function value
            current_f = f(x)
            history['f'].append(float(current_f))
            history['x'].append(x.copy().tolist())
            
            # Check for function value convergence
            if abs(prev_f - current_f) < tolerance:
                converged = True
                logger.debug(f"Converged at iteration {iteration}, Δf={abs(prev_f - current_f):.2e}")
                break
            
            prev_f = current_f
            
            # Callback
            if callback is not None:
                callback(iteration, x, current_f)
        
        result = OptimizationResult(
            x=x,
            fun=float(f(x)),
            n_iterations=iteration + 1,
            converged=converged,
            history=history,
            gradient_norm=float(grad_norm)
        )
        
        logger.info(f"Optimization completed: {iteration + 1} iterations, f(x)={result.fun:.6f}")
        return result
    
    def gradient_descent(
        self,
        f: Union[ScalarFunction, VectorFunction],
        grad_fn: Optional[Callable] = None,
        x0: Union[float, np.ndarray] = 0.0,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ) -> Union[float, np.ndarray]:
        """
        Simple gradient descent optimization (convenience method).
        
        Args:
            f: Function to minimize.
            grad_fn: Optional gradient function.
            x0: Initial point.
            max_iterations: Maximum iterations. Default: 1000.
            tolerance: Convergence tolerance. Default: 1e-6.
        
        Returns:
            Union[float, np.ndarray]: Optimal point.
        
        Example:
            >>> opt = Optimizer(learning_rate=0.1)
            >>> def f(x): return (x - 3)**2
            >>> x_min = opt.gradient_descent(f, x0=10.0)
            >>> np.isclose(x_min, 3.0, atol=0.1)
            True
        """
        # Handle 1D case
        if isinstance(x0, (int, float)):
            x0_arr = np.array([float(x0)])
            
            def f_wrapper(x):
                return f(x[0])
            
            def grad_wrapper(x):
                if grad_fn is not None:
                    return np.array([grad_fn(x[0])])
                else:
                    calc = CalculusOperations()
                    return calc.numerical_gradient(f_wrapper, x)
            
            result = self.minimize(f_wrapper, x0_arr, grad_wrapper, max_iterations, tolerance)
            return float(result.x[0])
        
        else:
            result = self.minimize(f, x0, grad_fn, max_iterations, tolerance)
            return result.x


def numerical_integration(
    f: ScalarFunction,
    a: float,
    b: float,
    n: int = 1000,
    method: str = 'simpson'
) -> float:
    """
    Compute numerical integral of a function.
    
    Methods:
    - 'rectangle': Rectangle rule (Riemann sum)
    - 'trapezoid': Trapezoidal rule
    - 'simpson': Simpson's rule (most accurate for smooth functions)
    
    Args:
        f: Function to integrate.
        a: Lower bound.
        b: Upper bound.
        n: Number of intervals. Default: 1000.
        method: Integration method. Default: 'simpson'.
    
    Returns:
        float: Approximate integral value.
    
    Raises:
        ValueError: If method is not supported.
    
    Example:
        >>> # ∫₀¹ x² dx = 1/3
        >>> def f(x): return x**2
        >>> numerical_integration(f, 0, 1, n=100)  # doctest: +ELLIPSIS
        0.333...
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    
    if method == 'rectangle':
        # Left Riemann sum
        result = h * np.sum(y[:-1])
    elif method == 'trapezoid':
        # Trapezoidal rule
        result = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    elif method == 'simpson':
        # Simpson's rule (requires even number of intervals)
        if n % 2 == 1:
            n -= 1
            x = np.linspace(a, b, n + 1)
            y = np.array([f(xi) for xi in x])
        
        result = (h / 3) * (y[0] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]) + y[-1])
    else:
        error_msg = f"Method must be 'rectangle', 'trapezoid', or 'simpson', got '{method}'"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.debug(f"Numerical integration ({method}): [{a}, {b}] = {result}")
    return result


def find_root(
    f: ScalarFunction,
    x0: float,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
    method: str = 'newton'
) -> float:
    """
    Find root of a function using numerical methods.
    
    Methods:
    - 'newton': Newton-Raphson method (requires derivative)
    - 'bisection': Bisection method (requires bracket)
    - 'secant': Secant method (no derivative needed)
    
    Args:
        f: Function to find root of.
        x0: Initial guess (or lower bound for bisection).
        max_iterations: Maximum iterations. Default: 100.
        tolerance: Convergence tolerance. Default: 1e-10.
        method: Root-finding method. Default: 'newton'.
    
    Returns:
        float: Approximate root.
    
    Example:
        >>> # Find root of x² - 2 = 0 (√2)
        >>> def f(x): return x**2 - 2
        >>> find_root(f, x0=1.5)  # doctest: +ELLIPSIS
        1.41421356...
    """
    calc = CalculusOperations()
    
    if method == 'newton':
        x = x0
        for i in range(max_iterations):
            f_x = f(x)
            if abs(f_x) < tolerance:
                logger.debug(f"Newton's method converged in {i+1} iterations")
                return x
            
            f_prime = calc.numerical_derivative(f, x)
            if abs(f_prime) < 1e-15:
                error_msg = "Derivative too small, Newton's method failed"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            x = x - f_x / f_prime
        
        logger.warning(f"Newton's method did not converge in {max_iterations} iterations")
        return x
    
    elif method == 'bisection':
        # Need to find bracket [a, b] where f(a) * f(b) < 0
        a = x0
        b = x0 + 1
        
        # Expand bracket until we find sign change
        for _ in range(100):
            if f(a) * f(b) < 0:
                break
            b = b + 1
        
        if f(a) * f(b) >= 0:
            error_msg = "Could not find bracket with sign change"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        for i in range(max_iterations):
            c = (a + b) / 2
            f_c = f(c)
            
            if abs(f_c) < tolerance or (b - a) / 2 < tolerance:
                logger.debug(f"Bisection converged in {i+1} iterations")
                return c
            
            if f(a) * f_c < 0:
                b = c
            else:
                a = c
        
        logger.warning(f"Bisection did not converge in {max_iterations} iterations")
        return (a + b) / 2
    
    elif method == 'secant':
        x0_val = x0
        x1_val = x0 + 0.1
        
        for i in range(max_iterations):
            f0 = f(x0_val)
            f1 = f(x1_val)
            
            if abs(f1) < tolerance:
                logger.debug(f"Secant method converged in {i+1} iterations")
                return x1_val
            
            if abs(f1 - f0) < 1e-15:
                error_msg = "Function values too close, secant method failed"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            x2_val = x1_val - f1 * (x1_val - x0_val) / (f1 - f0)
            
            if abs(x2_val - x1_val) < tolerance:
                logger.debug(f"Secant method converged in {i+1} iterations")
                return x2_val
            
            x0_val = x1_val
            x1_val = x2_val
        
        logger.warning(f"Secant method did not converge in {max_iterations} iterations")
        return x1_val
    
    else:
        error_msg = f"Method must be 'newton', 'bisection', or 'secant', got '{method}'"
        logger.error(error_msg)
        raise ValueError(error_msg)


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("Calculus Module - Demonstration")
    print("=" * 60)
    
    calc = CalculusOperations()
    
    # Numerical differentiation
    print("\n1. Numerical Differentiation:")
    def f(x): return x**3 - 2*x + 1
    x = 2.0
    derivative = calc.numerical_derivative(f, x)
    print(f"   f(x) = x³ - 2x + 1")
    print(f"   f'({x}) = {derivative:.6f} (exact: {3*x**2 - 2})")
    
    # Gradient
    print("\n2. Gradient Computation:")
    def g(x): return x[0]**2 + 2*x[1]**2 + x[0]*x[1]
    x = np.array([1.0, 2.0])
    grad = calc.numerical_gradient(g, x)
    print(f"   g(x) = x₀² + 2x₁² + x₀x₁")
    print(f"   ∇g({x}) = {grad}")
    print(f"   Exact: [2x₀ + x₁, 4x₁ + x₀] = [4.0, 9.0]")
    
    # Hessian
    print("\n3. Hessian Matrix:")
    H = calc.hessian(g, x)
    print(f"   H = \n{H}")
    print(f"   Exact: [[2, 1], [1, 4]]")
    
    # Taylor approximation
    print("\n4. Taylor Approximation:")
    x0 = np.array([0.0, 0.0])
    x_eval = np.array([0.1, 0.1])
    exact = g(x_eval)
    approx = calc.taylor_approximation(g, x0, x_eval, order=2)
    print(f"   g({x_eval}) = {exact:.6f}")
    print(f"   Taylor approx = {approx:.6f}")
    print(f"   Error = {abs(exact - approx):.2e}")
    
    # Optimization
    print("\n5. Optimization:")
    def h(x): return (x[0] - 1)**2 + (x[1] - 2)**2
    
    methods = [
        OptimizationMethod.GRADIENT_DESCENT,
        OptimizationMethod.MOMENTUM,
        OptimizationMethod.ADAM
    ]
    
    for method in methods:
        opt = Optimizer(learning_rate=0.1, method=method)
        result = opt.minimize(h, x0=np.array([5.0, 5.0]), max_iterations=100)
        print(f"   {method.value}: x* = {result.x}, f(x*) = {result.fun:.6f}, iterations = {result.n_iterations}")
    
    # Numerical integration
    print("\n6. Numerical Integration:")
    def integrand(x): return x**2
    result = numerical_integration(integrand, 0, 1, n=100, method='simpson')
    print(f"   ∫₀¹ x² dx = {result:.6f} (exact: 1/3 ≈ 0.333333)")
    
    # Root finding
    print("\n7. Root Finding:")
    def poly(x): return x**2 - 2
    root = find_root(poly, x0=1.5, method='newton')
    print(f"   Root of x² - 2 = 0: {root:.10f} (exact: √2 ≈ 1.4142135624)")
    
    print("\n" + "=" * 60)
