"""
Matrix Operations Module for Machine Learning.

This module provides comprehensive matrix operations essential for ML mathematics,
including multiplication, determinant, inverse, eigenvalues, SVD, and transformations.

Example Usage:
    >>> import numpy as np
    >>> from matrices import MatrixOperations
    >>> A = np.array([[1, 2], [3, 4]])
    >>> B = np.array([[5, 6], [7, 8]])
    >>> ops = MatrixOperations()
    >>> C = ops.multiply(A, B)
    >>> det = ops.determinant(A)
    >>> print(f"Determinant: {det}")
"""

from typing import Union, List, Tuple, Optional
import numpy as np
from numpy.typing import ArrayLike
import logging

logger = logging.getLogger(__name__)

Matrix = Union[np.ndarray, List[List[float]]]
Vector = Union[np.ndarray, List[float], Tuple[float, ...]]
