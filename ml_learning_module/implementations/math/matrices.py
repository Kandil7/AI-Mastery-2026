"""
Matrix Operations Module for Machine Learning

This module provides comprehensive matrix operations for ML including:
- Basic operations (addition, multiplication, transpose)
- Matrix properties (determinant, inverse, rank)
- Decompositions (LU, QR, eigenvalues, SVD)

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Tuple, Optional, Union
import logging

logger = logging.getLogger(__name__)

Matrix = Union[np.ndarray, list]


class MatrixOperations:
    """
    Comprehensive matrix operations for machine learning.

    Provides methods for:
    - Basic arithmetic operations
    - Matrix properties (determinant, rank, trace)
    - Matrix decompositions (LU, QR, Eig, SVD)
    - Linear transformations

    Example Usage:
        >>> ops = MatrixOperations()
        >>> A = np.array([[1, 2], [3, 4]])
        >>> det = ops.determinant(A)
    """

    def __init__(self, epsilon: float = 1e-10):
        """
        Initialize MatrixOperations.

        Args:
            epsilon: Small value for numerical stability.
        """
        self.epsilon = epsilon

    def _validate_matrix(self, M: Matrix, name: str = "matrix") -> np.ndarray:
        """Validate and convert input to numpy array."""
        try:
            arr = np.asarray(M, dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise TypeError(f"{name} must be array-like")

        if arr.ndim == 0:
            raise ValueError(f"{name} cannot be scalar")

        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        return arr

    def add(self, A: Matrix, B: Matrix) -> np.ndarray:
        """
        Add two matrices element-wise.

        Args:
            A: First matrix.
            B: Second matrix.

        Returns:
            Sum matrix A + B.

        Example:
            >>> ops = MatrixOperations()
            >>> ops.add([[1, 2], [3, 4]], [[5, 6], [7, 8]])
            array([[ 6.,  8.],
                   [10., 12.]])
        """
        A_arr = self._validate_matrix(A, "A")
        B_arr = self._validate_matrix(B, "B")

        if A_arr.shape != B_arr.shape:
            raise ValueError(
                f"Matrices must have same shape: {A_arr.shape} vs {B_arr.shape}"
            )

        return A_arr + B_arr

    def subtract(self, A: Matrix, B: Matrix) -> np.ndarray:
        """Subtract matrices element-wise (A - B)."""
        A_arr = self._validate_matrix(A, "A")
        B_arr = self._validate_matrix(B, "B")

        if A_arr.shape != B_arr.shape:
            raise ValueError(f"Matrices must have same shape")

        return A_arr - B_arr

    def scalar_multiply(self, M: Matrix, scalar: float) -> np.ndarray:
        """Multiply matrix by scalar."""
        M_arr = self._validate_matrix(M, "M")
        return scalar * M_arr

    def matmul(self, A: Matrix, B: Matrix) -> np.ndarray:
        """
        Matrix multiplication (A @ B).

        Note: Inner dimensions must match.

        Args:
            A: Matrix of shape (m, n).
            B: Matrix of shape (n, p).

        Returns:
            Result of shape (m, p).

        Example:
            >>> ops = MatrixOperations()
            >>> ops.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])
            array([[19., 22.],
                   [43., 50.]])
        """
        A_arr = self._validate_matrix(A, "A")
        B_arr = self._validate_matrix(B, "B")

        if A_arr.shape[1] != B_arr.shape[0]:
            raise ValueError(
                f"Inner dimensions mismatch: {A_arr.shape} @ {B_arr.shape}"
            )

        return A_arr @ B_arr

    def transpose(self, M: Matrix) -> np.ndarray:
        """
        Transpose of a matrix.

        Args:
            M: Input matrix.

        Returns:
            Transposed matrix.

        Example:
            >>> ops = MatrixOperations()
            >>> ops.transpose([[1, 2, 3], [4, 5, 6]])
            array([[1., 4.],
                   [2., 5.],
                   [3., 6.]])
        """
        M_arr = self._validate_matrix(M, "M")
        return M_arr.T

    def determinant(self, M: Matrix) -> float:
        """
        Compute the determinant of a square matrix.

        Args:
            M: Square matrix.

        Returns:
            Determinant value.

        Raises:
            ValueError: If matrix is not square.

        Example:
            >>> ops = MatrixOperations()
            >>> ops.determinant([[1, 2], [3, 4]])
            -2.0
        """
        M_arr = self._validate_matrix(M, "M")

        if M_arr.shape[0] != M_arr.shape[1]:
            raise ValueError("Determinant requires square matrix")

        return float(np.linalg.det(M_arr))

    def inverse(self, M: Matrix) -> np.ndarray:
        """
        Compute the matrix inverse.

        Args:
            M: Square, invertible matrix.

        Returns:
            Inverse matrix.

        Raises:
            ValueError: If matrix is not square or is singular.

        Example:
            >>> ops = MatrixOperations()
            >>> ops.inverse([[1, 2], [3, 4]])
            array([[-2. ,  1. ],
                   [ 1.5, -0.5]])
        """
        M_arr = self._validate_matrix(M, "M")

        if M_arr.shape[0] != M_arr.shape[1]:
            raise ValueError("Inverse requires square matrix")

        det = self.determinant(M_arr)
        if abs(det) < self.epsilon:
            raise ValueError("Matrix is singular (determinant = 0)")

        return np.linalg.inv(M_arr)

    def identity(self, n: int) -> np.ndarray:
        """
        Create identity matrix of size n×n.

        Args:
            n: Dimension of identity matrix.

        Returns:
            Identity matrix.

        Example:
            >>> ops = MatrixOperations()
            >>> ops.identity(3)
            array([[1., 0., 0.],
                   [0., 1., 0.],
                   [0., 0., 1.]])
        """
        return np.eye(n)

    def zeros(self, rows: int, cols: int) -> np.ndarray:
        """Create zero matrix of shape (rows, cols)."""
        return np.zeros((rows, cols))

    def trace(self, M: Matrix) -> float:
        """
        Compute the trace (sum of diagonal elements).

        Args:
            M: Square matrix.

        Returns:
            Trace value.

        Example:
            >>> ops = MatrixOperations()
            >>> ops.trace([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
            15.0
        """
        M_arr = self._validate_matrix(M, "M")

        if M_arr.shape[0] != M_arr.shape[1]:
            raise ValueError("Trace requires square matrix")

        return float(np.trace(M_arr))

    def rank(self, M: Matrix) -> int:
        """
        Compute the rank of a matrix.

        The rank is the number of linearly independent rows/columns.

        Args:
            M: Input matrix.

        Returns:
            Rank of matrix.

        Example:
            >>> ops = MatrixOperations()
            >>> ops.rank([[1, 0], [0, 1]])
            2
            >>> ops.rank([[1, 2], [2, 4]])  # Singular
            1
        """
        M_arr = self._validate_matrix(M, "M")
        return int(np.linalg.matrix_rank(M_arr))

    def lu_decomposition(self, M: Matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        LU Decomposition: A = P @ L @ U

        Where:
        - P: Permutation matrix
        - L: Lower triangular
        - U: Upper triangular

        Args:
            M: Square matrix to decompose.

        Returns:
            Tuple of (P, L, U) matrices.

        Example:
            >>> ops = MatrixOperations()
            >>> P, L, U = ops.lu_decomposition([[1, 2], [3, 4]])
        """
        M_arr = self._validate_matrix(M, "M")

        if M_arr.shape[0] != M_arr.shape[1]:
            raise ValueError("LU decomposition requires square matrix")

        P, L, U = np.linalg.lu(M_arr)
        return P, L, U

    def eigenvalues(self, M: Matrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors.

        Returns:
            Tuple of (eigenvalues, eigenvectors) where eigenvectors are columns.

        Example:
            >>> ops = MatrixOperations()
            >>> vals, vecs = ops.eigenvalues([[4, 2], [1, 3]])
        """
        M_arr = self._validate_matrix(M, "M")

        if M_arr.shape[0] != M_arr.shape[1]:
            raise ValueError("Eigenvalues require square matrix")

        eigenvalues, eigenvectors = np.linalg.eig(M_arr)
        return eigenvalues, eigenvectors

    def svd(
        self, M: Matrix, full_matrices: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Singular Value Decomposition: M = U @ Σ @ V^T

        Args:
            M: Input matrix.
            full_matrices: If True, return full-size U and V.

        Returns:
            Tuple of (U, singular_values, V^T).

        Example:
            >>> ops = MatrixOperations()
            >>> U, S, Vt = ops.svd([[1, 2], [3, 4], [5, 6]])
        """
        M_arr = self._validate_matrix(M, "M")
        return np.linalg.svd(M_arr, full_matrices=full_matrices)

    def condition_number(self, M: Matrix) -> float:
        """
        Compute condition number of matrix.

        A high condition number indicates an ill-conditioned matrix.

        Args:
            M: Input matrix.

        Returns:
            Condition number.
        """
        M_arr = self._validate_matrix(M, "M")
        return float(np.linalg.cond(M_arr))

    def frobenius_norm(self, M: Matrix) -> float:
        """
        Compute Frobenius norm of matrix.

        ||M||_F = sqrt(Σ M_ij²)

        Args:
            M: Input matrix.

        Returns:
            Frobenius norm.
        """
        M_arr = self._validate_matrix(M, "M")
        return float(np.linalg.norm(M_arr, "fro"))

    def is_positive_definite(self, M: Matrix) -> bool:
        """
        Check if matrix is positive definite.

        A symmetric matrix is positive definite if all eigenvalues > 0.

        Args:
            M: Square matrix.

        Returns:
            True if positive definite.
        """
        M_arr = self._validate_matrix(M, "M")

        if not np.allclose(M_arr, M_arr.T):
            return False

        try:
            eigenvalues = np.linalg.eigvalsh(M_arr)
            return np.all(eigenvalues > 0)
        except:
            return False

    def is_symmetric(self, M: Matrix, tolerance: float = 1e-9) -> bool:
        """Check if matrix is symmetric."""
        M_arr = self._validate_matrix(M, "M")
        return np.allclose(M_arr, M_arr.T, atol=tolerance)

    def is_orthogonal(self, M: Matrix, tolerance: float = 1e-9) -> bool:
        """Check if matrix is orthogonal (Q^T Q = I)."""
        M_arr = self._validate_matrix(M, "M")

        if M_arr.shape[0] != M_arr.shape[1]:
            return False

        return np.allclose(M_arr.T @ M_arr, np.eye(M_arr.shape[0]), atol=tolerance)


# Helper functions
def create_hadamard_matrix(n: int) -> np.ndarray:
    """
    Create Hadamard matrix of order n.

    A Hadamard matrix H satisfies H^T H = nI.

    Args:
        n: Order of matrix (must be 1, 2, or divisible by 4).

    Returns:
        Hadamard matrix.
    """
    if n == 1:
        return np.array([[1]])

    if n % 4 != 0:
        raise ValueError(
            "Hadamard matrix only exists for n = 1, 2, or n divisible by 4"
        )

    # Sylvester's construction
    H = np.array([[1]])
    for _ in range(int(np.log2(n))):
        H = np.block([[H, H], [H, -H]])

    return H


def create_vandermonde_matrix(values: list, n: int) -> np.ndarray:
    """
    Create Vandermonde matrix from values.

    Args:
        values: List of values for first column.
        n: Number of columns.

    Returns:
        Vandermonde matrix.

    Example:
        >>> create_vandermonde_matrix([1, 2, 3], 3)
        array([[1., 1., 1.],
               [1., 2., 4.],
               [1., 3., 9.]])
    """
    m = len(values)
    V = np.zeros((m, n))

    for i, v in enumerate(values):
        for j in range(n):
            V[i, j] = v**j

    return V


# Main demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Matrix Operations Module - Demonstration")
    print("=" * 60)

    ops = MatrixOperations()

    # Basic operations
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])

    print(f"\nA = \n{A}")
    print(f"B = \n{B}")

    print(f"\nA + B = \n{ops.add(A, B)}")
    print(f"A @ B = \n{ops.matmul(A, B)}")
    print(f"A^T = \n{ops.transpose(A)}")

    print(f"\nDeterminant of A: {ops.determinant(A)}")
    print(f"Trace of A: {ops.trace(A)}")
    print(f"Rank of A: {ops.rank(A)}")
    print(f"Frobenius norm: {ops.frobenius_norm(A)}")

    print("\n--- Matrix Decompositions ---")

    # LU Decomposition
    P, L, U = ops.lu_decomposition(A)
    print(f"\nLU Decomposition of A:")
    print(f"P = \n{P}")
    print(f"L = \n{L}")
    print(f"U = \n{U}")
    print(f"P @ L @ U = \n{P @ L @ U}")

    # SVD
    M = np.array([[1, 2, 3], [4, 5, 6]])
    U, S, Vt = ops.svd(M)
    print(f"\nSVD of M:")
    print(f"U = \n{U}")
    print(f"S = {S}")
    print(f"Vt = \n{Vt}")

    # Eigenvalues
    vals, vecs = ops.eigenvalues(A)
    print(f"\nEigenvalues of A: {vals}")
    print(f"Eigenvectors of A: \n{vecs}")

    print("\n" + "=" * 60)
