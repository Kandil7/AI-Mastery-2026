"""
Module 1.1: Mathematics for Machine Learning.

This module provides comprehensive mathematical foundations for ML including:
- Vector operations (dot product, cross product, norms, projections)
- Matrix operations (multiplication, decomposition, eigenvalues, SVD)
- Calculus (derivatives, gradients, optimization)
- Probability and statistics (distributions, hypothesis testing)

Example Usage:
    >>> from module_1_1_mathematics import VectorOperations, MatrixOperations
    >>> from module_1_1_mathematics import CalculusOperations, Optimizer
    >>> from module_1_1_mathematics import ProbabilityOperations, Distribution
    >>> 
    >>> # Vector operations
    >>> import numpy as np
    >>> v_ops = VectorOperations()
    >>> v1 = np.array([1, 2, 3])
    >>> v2 = np.array([4, 5, 6])
    >>> dot = v_ops.dot_product(v1, v2)
    >>> 
    >>> # Matrix operations
    >>> m_ops = MatrixOperations()
    >>> A = np.array([[1, 2], [3, 4]])
    >>> det = m_ops.determinant(A)
    >>> 
    >>> # Optimization
    >>> calc = CalculusOperations()
    >>> opt = Optimizer(learning_rate=0.1)
    >>> def f(x): return x[0]**2 + x[1]**2
    >>> result = opt.minimize(f, x0=np.array([5.0, 5.0]))
    >>> 
    >>> # Probability
    >>> prob_ops = ProbabilityOperations()
    >>> dist = Distribution.normal(mean=0, std=1)
    >>> samples = dist.sample(1000)
"""

from .vectors import (
    VectorOperations,
    create_basis_vector,
    random_vector,
    random_unit_vector,
    Vector,
)

from .matrices import (
    MatrixOperations,
    create_hadamard_matrix,
    create_vandermonde_matrix,
    Matrix,
)

from .calculus import (
    CalculusOperations,
    Optimizer,
    OptimizationMethod,
    OptimizationResult,
    numerical_integration,
    find_root,
    ScalarFunction,
    VectorFunction,
    GradientFunction,
)

from .probability import (
    ProbabilityOperations,
    HypothesisTesting,
    Distribution,
    DistributionType,
)

__all__ = [
    # Vector operations
    'VectorOperations',
    'create_basis_vector',
    'random_vector',
    'random_unit_vector',
    'Vector',
    
    # Matrix operations
    'MatrixOperations',
    'create_hadamard_matrix',
    'create_vandermonde_matrix',
    'Matrix',
    
    # Calculus
    'CalculusOperations',
    'Optimizer',
    'OptimizationMethod',
    'OptimizationResult',
    'numerical_integration',
    'find_root',
    'ScalarFunction',
    'VectorFunction',
    'GradientFunction',
    
    # Probability
    'ProbabilityOperations',
    'HypothesisTesting',
    'Distribution',
    'DistributionType',
]

__version__ = '1.0.0'
__author__ = 'AI Mastery 2026'
