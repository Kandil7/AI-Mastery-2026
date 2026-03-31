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


class MatrixOperations:
    """
    Comprehensive matrix operations for machine learning mathematics.
    
    This class provides methods for common matrix operations including:
    - Matrix multiplication and transpose
    - Determinant and inverse
    - Eigenvalues and eigenvectors
    - Singular Value Decomposition (SVD)
    - Matrix factorizations (LU, QR, Cholesky)
    - Rank, trace, and condition number
    - Linear transformations
    
    Attributes:
        epsilon (float): Small value for numerical stability.
    
    Raises:
        ValueError: If input matrices have incompatible shapes.
        TypeError: If inputs are not array-like.
    """
    
    def __init__(self, epsilon: float = 1e-10):
        """
        Initialize MatrixOperations with numerical stability parameter.
        
        Args:
            epsilon: Small value to prevent division by zero. Default: 1e-10.
        
        Example:
            >>> ops = MatrixOperations(epsilon=1e-8)
            >>> ops.epsilon
            1e-08
        """
        self.epsilon = epsilon
        logger.debug(f"MatrixOperations initialized with epsilon={epsilon}")
    
    def _validate_matrix(self, m: Matrix, name: str = "matrix") -> np.ndarray:
        """
        Validate and convert input to numpy array.
        
        Args:
            m: Input matrix (list of lists or numpy array).
            name: Name of the matrix for error messages.
        
        Returns:
            np.ndarray: Validated numpy array.
        
        Raises:
            TypeError: If input is not array-like.
            ValueError: If matrix is empty or contains non-numeric values.
        """
        try:
            arr = np.asarray(m, dtype=np.float64)
        except (TypeError, ValueError) as e:
            logger.error(f"{name} must be array-like: {e}")
            raise TypeError(f"{name} must be array-like (list of lists or numpy array)") from e
        
        if arr.size == 0:
            logger.error(f"{name} cannot be empty")
            raise ValueError(f"{name} cannot be empty")
        
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        
        if arr.ndim != 2:
            logger.error(f"{name} must be 2D, got {arr.ndim}D")
            raise ValueError(f"{name} must be 2D, got {arr.ndim}D")
        
        if not np.issubdtype(arr.dtype, np.number):
            logger.error(f"{name} must contain numeric values")
            raise ValueError(f"{name} must contain numeric values")
        
        return arr
    
    def _validate_square(self, m: np.ndarray, name: str = "matrix") -> None:
        """Validate that a matrix is square."""
        if m.shape[0] != m.shape[1]:
            error_msg = f"{name} must be square, got shape {m.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def _validate_multiplication_dims(
        self, A: np.ndarray, B: np.ndarray
    ) -> None:
        """Validate that matrices can be multiplied."""
        if A.shape[1] != B.shape[0]:
            error_msg = f"Cannot multiply matrices with shapes {A.shape} and {B.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def multiply(self, A: Matrix, B: Matrix) -> np.ndarray:
        """
        Compute matrix multiplication (dot product).
        
        For matrices A (m×n) and B (n×p), returns C (m×p) where:
        C[i,j] = Σ_k A[i,k] * B[k,j]
        
        Args:
            A: Left matrix (m×n).
            B: Right matrix (n×p).
        
        Returns:
            np.ndarray: Product matrix (m×p).
        
        Raises:
            ValueError: If inner dimensions don't match.
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[1, 2], [3, 4]]
            >>> B = [[5, 6], [7, 8]]
            >>> C = ops.multiply(A, B)
            >>> np.allclose(C, [[19, 22], [43, 50]])
            True
        """
        A_arr = self._validate_matrix(A, "A")
        B_arr = self._validate_matrix(B, "B")
        self._validate_multiplication_dims(A_arr, B_arr)
        
        result = np.dot(A_arr, B_arr)
        logger.debug(f"Matrix multiplication: {A_arr.shape} × {B_arr.shape} = {result.shape}")
        return result
    
    def transpose(self, m: Matrix) -> np.ndarray:
        """
        Compute the transpose of a matrix.
        
        The transpose flips rows and columns: (A^T)[i,j] = A[j,i]
        
        Args:
            m: Input matrix.
        
        Returns:
            np.ndarray: Transposed matrix.
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[1, 2, 3], [4, 5, 6]]
            >>> AT = ops.transpose(A)
            >>> np.allclose(AT, [[1, 4], [2, 5], [3, 6]])
            True
        """
        m_arr = self._validate_matrix(m, "m")
        result = m_arr.T
        logger.debug(f"Matrix transposed: {m_arr.shape} -> {result.shape}")
        return result
    
    def trace(self, m: Matrix) -> float:
        """
        Compute the trace of a square matrix.
        
        The trace is the sum of diagonal elements: tr(A) = Σ_i A[i,i]
        
        Properties:
        - tr(A + B) = tr(A) + tr(B)
        - tr(AB) = tr(BA)
        - tr(A) = tr(A^T)
        - tr(A) = Σ λ_i (sum of eigenvalues)
        
        Args:
            m: Square matrix.
        
        Returns:
            float: Trace value.
        
        Raises:
            ValueError: If matrix is not square.
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[1, 2], [3, 4]]
            >>> ops.trace(A)
            5.0
        """
        m_arr = self._validate_matrix(m, "m")
        self._validate_square(m_arr, "m")
        
        result = float(np.trace(m_arr))
        logger.debug(f"Trace of {m_arr.shape} matrix: {result}")
        return result
    
    def determinant(self, m: Matrix) -> float:
        """
        Compute the determinant of a square matrix.
        
        The determinant is a scalar value that encodes properties of the matrix:
        - det(A) = 0 iff A is singular (non-invertible)
        - det(AB) = det(A) * det(B)
        - det(A^T) = det(A)
        - det(A^-1) = 1/det(A)
        
        Args:
            m: Square matrix.
        
        Returns:
            float: Determinant value.
        
        Raises:
            ValueError: If matrix is not square.
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[1, 2], [3, 4]]
            >>> ops.determinant(A)
            -2.0
            >>> # Verification: 1*4 - 2*3 = 4 - 6 = -2
        """
        m_arr = self._validate_matrix(m, "m")
        self._validate_square(m_arr, "m")
        
        result = float(np.linalg.det(m_arr))
        logger.debug(f"Determinant of {m_arr.shape} matrix: {result}")
        return result
    
    def inverse(self, m: Matrix) -> np.ndarray:
        """
        Compute the inverse of a square matrix.
        
        The inverse A^-1 satisfies: A * A^-1 = A^-1 * A = I
        
        Args:
            m: Square, invertible matrix.
        
        Returns:
            np.ndarray: Inverse matrix.
        
        Raises:
            ValueError: If matrix is not square or is singular.
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[4, 7], [2, 6]]
            >>> A_inv = ops.inverse(A)
            >>> I = ops.multiply(A, A_inv)
            >>> np.allclose(I, np.eye(2))
            True
        """
        m_arr = self._validate_matrix(m, "m")
        self._validate_square(m_arr, "m")
        
        det = self.determinant(m_arr)
        if abs(det) < self.epsilon:
            error_msg = f"Matrix is singular (det={det}), cannot invert"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        result = np.linalg.inv(m_arr)
        logger.debug(f"Matrix inverse computed, det={det}")
        return result
    
    def rank(self, m: Matrix) -> int:
        """
        Compute the rank of a matrix.
        
        The rank is the dimension of the vector space spanned by the columns.
        Equivalently, it's the number of linearly independent columns/rows.
        
        Args:
            m: Input matrix.
        
        Returns:
            int: Rank of the matrix.
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            >>> ops.rank(A)
            2
            >>> # Third row is linear combination of first two
        """
        m_arr = self._validate_matrix(m, "m")
        result = int(np.linalg.matrix_rank(m_arr))
        logger.debug(f"Rank of {m_arr.shape} matrix: {result}")
        return result
    
    def null_space(self, m: Matrix) -> np.ndarray:
        """
        Compute an orthonormal basis for the null space of a matrix.
        
        The null space contains all vectors x such that Ax = 0.
        
        Args:
            m: Input matrix.
        
        Returns:
            np.ndarray: Orthonormal basis vectors as columns.
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            >>> ns = ops.null_space(A)
            >>> # Verify: A @ ns should be approximately zero
            >>> np.allclose(ops.multiply(A, ns), 0)
            True
        """
        m_arr = self._validate_matrix(m, "m")
        
        # Use SVD to find null space
        U, S, Vh = np.linalg.svd(m_arr)
        
        # Null space corresponds to singular values near zero
        null_mask = S < self.epsilon
        null_space_dim = m_arr.shape[1] - np.sum(~null_mask)
        
        if null_space_dim == 0:
            # Return empty array with correct shape
            result = np.zeros((m_arr.shape[1], 0))
        else:
            result = Vh[null_mask].T
        
        logger.debug(f"Null space dimension: {result.shape[1]}")
        return result
    
    def eigenvalues(self, m: Matrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues and eigenvectors of a square matrix.
        
        For matrix A, finds λ and v such that: Av = λv
        
        Args:
            m: Square matrix.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (eigenvalues, eigenvectors)
                - eigenvalues: 1D array of eigenvalues
                - eigenvectors: 2D array with eigenvectors as columns
        
        Raises:
            ValueError: If matrix is not square.
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[4, -2], [1, 1]]
            >>> eigenvalues, eigenvectors = ops.eigenvalues(A)
            >>> # Verify: A @ v = λ * v for each eigenpair
            >>> for i in range(len(eigenvalues)):
            ...     Av = ops.multiply(A, eigenvectors[:, i])
            ...     lv = eigenvalues[i] * eigenvectors[:, i]
            ...     print(np.allclose(Av, lv))
            True
            True
        """
        m_arr = self._validate_matrix(m, "m")
        self._validate_square(m_arr, "m")
        
        eigenvalues, eigenvectors = np.linalg.eig(m_arr)
        logger.debug(f"Computed {len(eigenvalues)} eigenvalues for {m_arr.shape} matrix")
        return eigenvalues, eigenvectors
    
    def svd(self, m: Matrix, full_matrices: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Singular Value Decomposition (SVD).
        
        For matrix A (m×n), SVD produces:
        A = U @ Σ @ V^T
        
        Where:
        - U (m×m): Left singular vectors (orthonormal)
        - Σ (m×n): Diagonal matrix of singular values
        - V^T (n×n): Transpose of right singular vectors
        
        Applications:
        - Low-rank approximation
        - Principal Component Analysis (PCA)
        - Matrix completion
        - Noise reduction
        
        Args:
            m: Input matrix.
            full_matrices: If True, return full U and V^T. If False, return reduced.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (U, singular_values, Vt)
                - U: Left singular vectors
                - singular_values: 1D array of singular values (sorted descending)
                - Vt: Transpose of right singular vectors
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            >>> U, S, Vt = ops.svd(A)
            >>> # Reconstruct A
            >>> Sigma = np.diag(S)
            >>> A_reconstructed = ops.multiply(ops.multiply(U, Sigma), Vt)
            >>> np.allclose(A, A_reconstructed)
            True
        """
        m_arr = self._validate_matrix(m, "m")
        
        U, S, Vt = np.linalg.svd(m_arr, full_matrices=full_matrices)
        logger.debug(f"SVD computed: U{U.shape}, S{S.shape}, Vt{Vt.shape}")
        return U, S, Vt
    
    def low_rank_approximation(self, m: Matrix, k: int) -> np.ndarray:
        """
        Compute best rank-k approximation using SVD.
        
        By Eckart-Young theorem, the best rank-k approximation is obtained
        by keeping only the k largest singular values.
        
        Args:
            m: Input matrix.
            k: Target rank.
        
        Returns:
            np.ndarray: Rank-k approximation matrix.
        
        Raises:
            ValueError: If k is invalid.
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
            >>> A_approx = ops.low_rank_approximation(A, k=2)
            >>> ops.rank(A_approx)
            2
        """
        m_arr = self._validate_matrix(m, "m")
        
        if k < 1 or k > min(m_arr.shape):
            error_msg = f"k must be between 1 and min({m_arr.shape})={min(m_arr.shape)}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        U, S, Vt = self.svd(m_arr, full_matrices=False)
        
        # Keep only top k singular values
        S_k = np.diag(S[:k])
        U_k = U[:, :k]
        Vt_k = Vt[:k, :]
        
        result = self.multiply(self.multiply(U_k, S_k), Vt_k)
        logger.debug(f"Rank-{k} approximation computed for {m_arr.shape} matrix")
        return result
    
    def lu_decomposition(self, m: Matrix) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute LU decomposition with partial pivoting.
        
        Decomposes A = P @ L @ U where:
        - P: Permutation matrix
        - L: Lower triangular matrix (unit diagonal)
        - U: Upper triangular matrix
        
        Args:
            m: Square matrix.
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (P, L, U)
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[4, 3], [6, 3]]
            >>> P, L, U = ops.lu_decomposition(A)
            >>> # Verify: P^T @ L @ U = A
            >>> PT = ops.transpose(P)
            >>> reconstructed = ops.multiply(ops.multiply(PT, L), U)
            >>> np.allclose(A, reconstructed)
            True
        """
        m_arr = self._validate_matrix(m, "m")
        self._validate_square(m_arr, "m")
        
        # Use scipy for LU decomposition with pivoting
        try:
            from scipy.linalg import lu
            P, L, U = lu(m_arr)
        except ImportError:
            # Fallback: simple LU without pivoting (may fail for some matrices)
            n = m_arr.shape[0]
            L = np.eye(n)
            U = m_arr.copy()
            P = np.eye(n)
            
            for i in range(n - 1):
                if abs(U[i, i]) < self.epsilon:
                    error_msg = "Zero pivot encountered, LU decomposition failed"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                for j in range(i + 1, n):
                    L[j, i] = U[j, i] / U[i, i]
                    U[j, i:] -= L[j, i] * U[i, i:]
                    U[j, i] = 0
        
        logger.debug(f"LU decomposition computed for {m_arr.shape} matrix")
        return P, L, U
    
    def qr_decomposition(self, m: Matrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute QR decomposition.
        
        Decomposes A = Q @ R where:
        - Q: Orthogonal matrix (Q^T @ Q = I)
        - R: Upper triangular matrix
        
        Args:
            m: Input matrix.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (Q, R)
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[12, -51, 4], [6, 167, -68], [-4, 24, -41]]
            >>> Q, R = ops.qr_decomposition(A)
            >>> # Verify: Q @ R = A and Q^T @ Q = I
            >>> np.allclose(ops.multiply(Q, R), A)
            True
            >>> QT = ops.transpose(Q)
            >>> np.allclose(ops.multiply(QT, Q), np.eye(Q.shape[1]))
            True
        """
        m_arr = self._validate_matrix(m, "m")
        
        Q, R = np.linalg.qr(m_arr)
        logger.debug(f"QR decomposition computed: Q{Q.shape}, R{R.shape}")
        return Q, R
    
    def cholesky(self, m: Matrix) -> np.ndarray:
        """
        Compute Cholesky decomposition for positive definite matrices.
        
        Decomposes A = L @ L^T where L is lower triangular.
        
        Args:
            m: Symmetric, positive definite matrix.
        
        Returns:
            np.ndarray: Lower triangular matrix L.
        
        Raises:
            ValueError: If matrix is not positive definite.
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[4, 12, -16], [12, 37, -43], [-16, -43, 98]]
            >>> L = ops.cholesky(A)
            >>> # Verify: L @ L^T = A
            >>> LT = ops.transpose(L)
            >>> np.allclose(ops.multiply(L, LT), A)
            True
        """
        m_arr = self._validate_matrix(m, "m")
        self._validate_square(m_arr, "m")
        
        try:
            result = np.linalg.cholesky(m_arr)
        except np.linalg.LinAlgError as e:
            error_msg = "Matrix is not positive definite"
            logger.error(error_msg)
            raise ValueError(error_msg) from e
        
        logger.debug(f"Cholesky decomposition computed for {m_arr.shape} matrix")
        return result
    
    def condition_number(self, m: Matrix, p: int = 2) -> float:
        """
        Compute the condition number of a matrix.
        
        The condition number measures sensitivity to numerical errors.
        cond(A) = ||A|| * ||A^-1||
        
        A high condition number indicates an ill-conditioned matrix.
        
        Args:
            m: Input matrix.
            p: Order of norm. Default: 2.
        
        Returns:
            float: Condition number.
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[1, 0], [0, 1]]
            >>> ops.condition_number(A)
            1.0
            >>> # Identity matrix has condition number 1 (perfectly conditioned)
        """
        m_arr = self._validate_matrix(m, "m")
        result = float(np.linalg.cond(m_arr, p=p))
        logger.debug(f"Condition number (p={p}) of {m_arr.shape} matrix: {result}")
        return result
    
    def is_positive_definite(self, m: Matrix) -> bool:
        """
        Check if a matrix is positive definite.
        
        A symmetric matrix A is positive definite if:
        - x^T @ A @ x > 0 for all non-zero vectors x
        - All eigenvalues are positive
        
        Args:
            m: Square matrix.
        
        Returns:
            bool: True if positive definite.
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[2, -1], [-1, 2]]
            >>> ops.is_positive_definite(A)
            True
        """
        m_arr = self._validate_matrix(m, "m")
        self._validate_square(m_arr, "m")
        
        # Check symmetry
        if not np.allclose(m_arr, m_arr.T):
            return False
        
        # Check eigenvalues
        eigenvalues, _ = self.eigenvalues(m_arr)
        result = bool(np.all(eigenvalues > self.epsilon))
        logger.debug(f"Positive definite check: {result}")
        return result
    
    def is_orthogonal(self, m: Matrix, tolerance: float = 1e-9) -> bool:
        """
        Check if a matrix is orthogonal.
        
        A matrix Q is orthogonal if Q^T @ Q = I.
        
        Args:
            m: Square matrix.
            tolerance: Numerical tolerance. Default: 1e-9.
        
        Returns:
            bool: True if orthogonal.
        
        Example:
            >>> ops = MatrixOperations()
            >>> import numpy as np
            >>> theta = np.pi / 4  # 45 degrees
            >>> R = [[np.cos(theta), -np.sin(theta)], 
            ...      [np.sin(theta), np.cos(theta)]]
            >>> ops.is_orthogonal(R)
            True
        """
        m_arr = self._validate_matrix(m, "m")
        self._validate_square(m_arr, "m")
        
        mT = self.transpose(m_arr)
        product = self.multiply(mT, m_arr)
        identity = np.eye(m_arr.shape[0])
        
        result = bool(np.allclose(product, identity, atol=tolerance))
        logger.debug(f"Orthogonality check: {result}")
        return result
    
    def is_symmetric(self, m: Matrix, tolerance: float = 1e-9) -> bool:
        """
        Check if a matrix is symmetric.
        
        A matrix A is symmetric if A = A^T.
        
        Args:
            m: Square matrix.
            tolerance: Numerical tolerance. Default: 1e-9.
        
        Returns:
            bool: True if symmetric.
        """
        m_arr = self._validate_matrix(m, "m")
        self._validate_square(m_arr, "m")
        
        mT = self.transpose(m_arr)
        result = bool(np.allclose(m_arr, mT, atol=tolerance))
        logger.debug(f"Symmetry check: {result}")
        return result
    
    def power_iteration(
        self, m: Matrix, 
        max_iterations: int = 1000,
        tolerance: float = 1e-9
    ) -> Tuple[float, np.ndarray]:
        """
        Find the dominant eigenvalue and eigenvector using power iteration.
        
        Power iteration finds the eigenvalue with largest magnitude.
        
        Algorithm:
        1. Start with random vector b0
        2. Iterate: b_{k+1} = A @ b_k / ||A @ b_k||
        3. Eigenvalue: λ = b^T @ A @ b
        
        Args:
            m: Square matrix.
            max_iterations: Maximum iterations. Default: 1000.
            tolerance: Convergence tolerance. Default: 1e-9.
        
        Returns:
            Tuple[float, np.ndarray]: (dominant_eigenvalue, eigenvector)
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[4, 1], [2, 3]]
            >>> eigenvalue, eigenvector = ops.power_iteration(A)
            >>> # Verify: A @ v ≈ λ * v
            >>> Av = ops.multiply(A, eigenvector)
            >>> lv = eigenvalue * eigenvector
            >>> np.allclose(Av, lv)
            True
        """
        m_arr = self._validate_matrix(m, "m")
        self._validate_square(m_arr, "m")
        
        n = m_arr.shape[0]
        
        # Initialize with random vector
        np.random.seed(42)  # For reproducibility
        b = np.random.randn(n)
        b = b / np.linalg.norm(b)
        
        eigenvalue = 0.0
        for i in range(max_iterations):
            # Power iteration step
            Ab = np.dot(m_arr, b)
            new_eigenvalue = np.dot(b, Ab)
            b_new = Ab / (np.linalg.norm(Ab) + self.epsilon)
            
            # Check convergence
            if abs(new_eigenvalue - eigenvalue) < tolerance:
                logger.debug(f"Power iteration converged in {i+1} iterations")
                return float(new_eigenvalue), b_new
            
            eigenvalue = new_eigenvalue
            b = b_new
        
        logger.warning(f"Power iteration did not converge in {max_iterations} iterations")
        return float(eigenvalue), b
    
    def rotation_matrix_2d(self, angle: float, degrees: bool = False) -> np.ndarray:
        """
        Create a 2D rotation matrix.
        
        Args:
            angle: Rotation angle.
            degrees: If True, angle is in degrees. Default: False (radians).
        
        Returns:
            np.ndarray: 2×2 rotation matrix.
        
        Example:
            >>> ops = MatrixOperations()
            >>> import numpy as np
            >>> R = ops.rotation_matrix_2d(np.pi/2)  # 90 degrees
            >>> # Rotate point (1, 0) by 90 degrees
            >>> v = [1, 0]
            >>> rotated = ops.multiply(R, np.array(v).reshape(2, 1)).flatten()
            >>> np.allclose(rotated, [0, 1])
            True
        """
        if degrees:
            angle = np.radians(angle)
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        result = np.array([
            [cos_a, -sin_a],
            [sin_a, cos_a]
        ])
        logger.debug(f"Created 2D rotation matrix for angle {angle:.4f} rad")
        return result
    
    def rotation_matrix_3d(
        self, 
        angle: float, 
        axis: str = 'z',
        degrees: bool = False
    ) -> np.ndarray:
        """
        Create a 3D rotation matrix around specified axis.
        
        Args:
            angle: Rotation angle.
            axis: Axis of rotation ('x', 'y', or 'z'). Default: 'z'.
            degrees: If True, angle is in degrees. Default: False (radians).
        
        Returns:
            np.ndarray: 3×3 rotation matrix.
        
        Raises:
            ValueError: If axis is not 'x', 'y', or 'z'.
        
        Example:
            >>> ops = MatrixOperations()
            >>> import numpy as np
            >>> R = ops.rotation_matrix_3d(np.pi/2, axis='z')
            >>> # Rotate point (1, 0, 0) by 90 degrees around z-axis
            >>> v = [1, 0, 0]
            >>> rotated = ops.multiply(R, np.array(v).reshape(3, 1)).flatten()
            >>> np.allclose(rotated, [0, 1, 0])
            True
        """
        if degrees:
            angle = np.radians(angle)
        
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        if axis == 'x':
            result = np.array([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ])
        elif axis == 'y':
            result = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
        elif axis == 'z':
            result = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
        else:
            error_msg = f"Axis must be 'x', 'y', or 'z', got '{axis}'"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"Created 3D rotation matrix around {axis}-axis")
        return result
    
    def scaling_matrix(self, scales: List[float]) -> np.ndarray:
        """
        Create a scaling transformation matrix.
        
        Args:
            scales: Scale factors for each dimension.
        
        Returns:
            np.ndarray: Diagonal scaling matrix.
        
        Example:
            >>> ops = MatrixOperations()
            >>> S = ops.scaling_matrix([2, 3])
            >>> np.allclose(S, [[2, 0], [0, 3]])
            True
        """
        result = np.diag(scales)
        logger.debug(f"Created scaling matrix with scales {scales}")
        return result
    
    def identity(self, n: int) -> np.ndarray:
        """
        Create an n×n identity matrix.
        
        Args:
            n: Size of the matrix.
        
        Returns:
            np.ndarray: Identity matrix.
        
        Example:
            >>> ops = MatrixOperations()
            >>> I = ops.identity(3)
            >>> np.allclose(I, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            True
        """
        if n < 1:
            raise ValueError(f"n must be positive, got {n}")
        result = np.eye(n)
        logger.debug(f"Created {n}×{n} identity matrix")
        return result
    
    def zeros(self, rows: int, cols: Optional[int] = None) -> np.ndarray:
        """
        Create a zero matrix.
        
        Args:
            rows: Number of rows.
            cols: Number of columns. If None, creates square matrix.
        
        Returns:
            np.ndarray: Zero matrix.
        
        Example:
            >>> ops = MatrixOperations()
            >>> Z = ops.zeros(2, 3)
            >>> Z.shape
            (2, 3)
            >>> np.all(Z == 0)
            True
        """
        if cols is None:
            cols = rows
        result = np.zeros((rows, cols))
        logger.debug(f"Created {rows}×{cols} zero matrix")
        return result
    
    def ones(self, rows: int, cols: Optional[int] = None) -> np.ndarray:
        """
        Create a matrix of ones.
        
        Args:
            rows: Number of rows.
            cols: Number of columns. If None, creates square matrix.
        
        Returns:
            np.ndarray: Matrix of ones.
        """
        if cols is None:
            cols = rows
        result = np.ones((rows, cols))
        logger.debug(f"Created {rows}×{cols} ones matrix")
        return result
    
    def diagonal(self, diag_values: List[float]) -> np.ndarray:
        """
        Create a diagonal matrix from values.
        
        Args:
            diag_values: Values for the diagonal.
        
        Returns:
            np.ndarray: Diagonal matrix.
        
        Example:
            >>> ops = MatrixOperations()
            >>> D = ops.diagonal([1, 2, 3])
            >>> np.allclose(D, [[1, 0, 0], [0, 2, 0], [0, 0, 3]])
            True
        """
        result = np.diag(diag_values)
        logger.debug(f"Created diagonal matrix with {len(diag_values)} elements")
        return result
    
    def solve_linear_system(self, A: Matrix, b: Vector) -> np.ndarray:
        """
        Solve the linear system Ax = b.
        
        Args:
            A: Coefficient matrix (n×n).
            b: Right-hand side vector (n,).
        
        Returns:
            np.ndarray: Solution vector x.
        
        Raises:
            ValueError: If system is singular or dimensions don't match.
        
        Example:
            >>> ops = MatrixOperations()
            >>> A = [[3, 1], [1, 2]]
            >>> b = [9, 8]
            >>> x = ops.solve_linear_system(A, b)
            >>> # Verify: A @ x = b
            >>> Ax = ops.multiply(A, x.reshape(-1, 1)).flatten()
            >>> np.allclose(Ax, b)
            True
        """
        A_arr = self._validate_matrix(A, "A")
        b_arr = self._validate_matrix(b, "b")
        
        self._validate_square(A_arr, "A")
        
        if A_arr.shape[0] != b_arr.shape[0]:
            error_msg = f"A rows ({A_arr.shape[0]}) must match b length ({b_arr.shape[0]})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Flatten b if needed
        b_flat = b_arr.flatten()
        
        result = np.linalg.solve(A_arr, b_flat)
        logger.debug(f"Solved linear system with {A_arr.shape[0]} equations")
        return result
    
    def least_squares(self, A: Matrix, b: Vector) -> np.ndarray:
        """
        Solve the least squares problem min ||Ax - b||².
        
        For overdetermined systems (more equations than unknowns),
        finds the best approximate solution.
        
        Args:
            A: Coefficient matrix (m×n).
            b: Right-hand side vector (m,).
        
        Returns:
            np.ndarray: Least squares solution x.
        
        Example:
            >>> ops = MatrixOperations()
            >>> # Fit a line y = mx + c to points
            >>> x_data = [1, 2, 3, 4]
            >>> y_data = [1.1, 1.9, 3.2, 3.8]
            >>> A = [[xi, 1] for xi in x_data]  # Design matrix
            >>> x = ops.least_squares(A, y_data)
            >>> m, c = x[0], x[1]
            >>> print(f"Line: y = {m:.2f}x + {c:.2f}")
        """
        A_arr = self._validate_matrix(A, "A")
        b_arr = self._validate_matrix(b, "b")
        
        b_flat = b_arr.flatten()
        
        result, residuals, rank, s = np.linalg.lstsq(A_arr, b_flat, rcond=None)
        logger.debug(f"Least squares solved: rank={rank}, residuals={residuals}")
        return result


def create_hadamard_matrix(n: int) -> np.ndarray:
    """
    Create a Hadamard matrix of order n (if n is a power of 2).
    
    Hadamard matrices have orthogonal rows/columns with entries ±1.
    
    Args:
        n: Order of the matrix (must be power of 2).
    
    Returns:
        np.ndarray: Hadamard matrix.
    
    Raises:
        ValueError: If n is not a power of 2.
    
    Example:
        >>> H = create_hadamard_matrix(4)
        >>> # Check orthogonality: H @ H^T = n * I
        >>> ops = MatrixOperations()
        >>> HHT = ops.multiply(H, ops.transpose(H))
        >>> np.allclose(HHT, 4 * np.eye(4))
        True
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"n must be a power of 2, got {n}")
    
    if n == 1:
        return np.array([[1]])
    
    # Recursive construction
    H_half = create_hadamard_matrix(n // 2)
    top = np.hstack([H_half, H_half])
    bottom = np.hstack([H_half, -H_half])
    result = np.vstack([top, bottom])
    
    logger.debug(f"Created Hadamard matrix of order {n}")
    return result


def create_vandermonde_matrix(x: Vector, n: Optional[int] = None) -> np.ndarray:
    """
    Create a Vandermonde matrix from a vector.
    
    For vector x = [x1, x2, ..., xm], the Vandermonde matrix is:
    [[1, x1, x1², ..., x1^(n-1)],
     [1, x2, x2², ..., x2^(n-1)],
     ...
     [1, xm, xm², ..., xm^(n-1)]]
    
    Args:
        x: Input vector.
        n: Number of columns (degree + 1). If None, uses len(x).
    
    Returns:
        np.ndarray: Vandermonde matrix.
    
    Example:
        >>> x = [1, 2, 3]
        >>> V = create_vandermonde_matrix(x, n=3)
        >>> np.allclose(V, [[1, 1, 1], [1, 2, 4], [1, 3, 9]])
        True
    """
    x_arr = np.asarray(x, dtype=np.float64)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    
    if n is None:
        n = len(x_arr)
    
    result = np.vander(x_arr.flatten(), N=n, increasing=True)
    logger.debug(f"Created Vandermonde matrix {result.shape}")
    return result


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("Matrix Operations Module - Demonstration")
    print("=" * 60)
    
    ops = MatrixOperations()
    
    # Basic operations
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    B = np.array([[5.0, 6.0], [7.0, 8.0]])
    
    print(f"\nA = \n{A}")
    print(f"\nB = \n{B}")
    
    print(f"\nA × B = \n{ops.multiply(A, B)}")
    print(f"\nA^T = \n{ops.transpose(A)}")
    print(f"\ntr(A) = {ops.trace(A)}")
    print(f"\ndet(A) = {ops.determinant(A)}")
    
    print(f"\nA^-1 = \n{ops.inverse(A)}")
    print(f"\nrank(A) = {ops.rank(A)}")
    
    # Eigenvalues
    print("\nEigenvalue decomposition:")
    eigenvalues, eigenvectors = ops.eigenvalues(A)
    print(f"Eigenvalues: {eigenvalues}")
    print(f"Eigenvectors:\n{eigenvectors}")
    
    # SVD
    print("\nSingular Value Decomposition:")
    C = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    U, S, Vt = ops.svd(C)
    print(f"U shape: {U.shape}, S: {S}, Vt shape: {Vt.shape}")
    
    # Low-rank approximation
    print("\nLow-rank approximation:")
    D = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    D_approx = ops.low_rank_approximation(D, k=2)
    print(f"Original rank: {ops.rank(D)}")
    print(f"Approximation rank: {ops.rank(D_approx)}")
    print(f"Approximation error: {np.linalg.norm(D - D_approx):.6f}")
    
    # QR decomposition
    print("\nQR decomposition:")
    P, L, U = ops.lu_decomposition(A)
    print(f"L = \n{L}")
    print(f"U = \n{U}")
    
    # Rotation matrices
    print("\nRotation matrices:")
    R_2d = ops.rotation_matrix_2d(np.pi / 4, degrees=False)
    print(f"2D rotation (45°):\n{R_2d}")
    
    R_3d = ops.rotation_matrix_3d(np.pi / 4, axis='z')
    print(f"3D rotation around z (45°):\n{R_3d}")
    
    # Linear system
    print("\nSolving linear system:")
    E = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([9.0, 8.0])
    x = ops.solve_linear_system(E, b)
    print(f"Solution: x = {x}")
    
    # Power iteration
    print("\nPower iteration:")
    eigenvalue, eigenvector = ops.power_iteration(A)
    print(f"Dominant eigenvalue: {eigenvalue:.6f}")
    print(f"Eigenvector: {eigenvector}")
    
    print("\n" + "=" * 60)
