"""
Vector Operations Module for Machine Learning.

This module provides comprehensive vector operations essential for ML mathematics,
including dot product, cross product, norms, projections, and linear combinations.

Example Usage:
    >>> import numpy as np
    >>> from vectors import VectorOperations
    >>> v1 = np.array([1, 2, 3])
    >>> v2 = np.array([4, 5, 6])
    >>> ops = VectorOperations()
    >>> dot = ops.dot_product(v1, v2)
    >>> print(f"Dot product: {dot}")
    >>> norm = ops.norm(v1)
    >>> print(f"Norm of v1: {norm}")
"""

from typing import Union, List, Tuple, Optional
import numpy as np
from numpy.typing import ArrayLike
import logging

logger = logging.getLogger(__name__)

Vector = Union[np.ndarray, List[float], Tuple[float, ...]]


class VectorOperations:
    """
    Comprehensive vector operations for machine learning mathematics.
    
    This class provides methods for common vector operations including:
    - Dot product and cross product
    - Vector norms (L1, L2, L-infinity)
    - Vector projection and rejection
    - Linear combinations and independence checks
    - Angle calculations between vectors
    - Vector normalization
    
    Attributes:
        epsilon (float): Small value for numerical stability.
    
    Raises:
        ValueError: If input vectors have incompatible shapes.
        TypeError: If inputs are not array-like.
    """
    
    def __init__(self, epsilon: float = 1e-10):
        """
        Initialize VectorOperations with numerical stability parameter.
        
        Args:
            epsilon: Small value to prevent division by zero. Default: 1e-10.
        
        Example:
            >>> ops = VectorOperations(epsilon=1e-8)
            >>> ops.epsilon
            1e-08
        """
        self.epsilon = epsilon
        logger.debug(f"VectorOperations initialized with epsilon={epsilon}")
    
    def _validate_vector(self, v: Vector, name: str = "vector") -> np.ndarray:
        """
        Validate and convert input to numpy array.
        
        Args:
            v: Input vector (list, tuple, or numpy array).
            name: Name of the vector for error messages.
        
        Returns:
            np.ndarray: Validated numpy array.
        
        Raises:
            TypeError: If input is not array-like.
            ValueError: If vector is empty or contains non-numeric values.
        
        Example:
            >>> ops = VectorOperations()
            >>> v = ops._validate_vector([1, 2, 3])
            >>> isinstance(v, np.ndarray)
            True
        """
        try:
            arr = np.asarray(v, dtype=np.float64)
        except (TypeError, ValueError) as e:
            logger.error(f"{name} must be array-like: {e}")
            raise TypeError(f"{name} must be array-like (list, tuple, or numpy array)") from e
        
        if arr.size == 0:
            logger.error(f"{name} cannot be empty")
            raise ValueError(f"{name} cannot be empty")
        
        if not np.issubdtype(arr.dtype, np.number):
            logger.error(f"{name} must contain numeric values")
            raise ValueError(f"{name} must contain numeric values")
        
        return arr
    
    def _validate_same_dimension(
        self, v1: np.ndarray, v2: np.ndarray, 
        op_name: str = "operation"
    ) -> None:
        """
        Validate that two vectors have the same dimension.
        
        Args:
            v1: First vector.
            v2: Second vector.
            op_name: Name of the operation for error messages.
        
        Raises:
            ValueError: If vectors have different dimensions.
        
        Example:
            >>> ops = VectorOperations()
            >>> v1 = np.array([1, 2, 3])
            >>> v2 = np.array([4, 5, 6])
            >>> ops._validate_same_dimension(v1, v2, "addition")
            # No exception raised
        """
        if v1.shape != v2.shape:
            error_msg = f"Vectors must have same shape for {op_name}: {v1.shape} vs {v2.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
    
    def dot_product(self, v1: Vector, v2: Vector) -> float:
        """
        Compute the dot product (inner product) of two vectors.
        
        The dot product is defined as: v1 · v2 = Σ(v1_i * v2_i)
        
        Properties:
        - Commutative: v1 · v2 = v2 · v1
        - Distributive: v1 · (v2 + v3) = v1 · v2 + v1 · v3
        - v · v = ||v||²
        
        Args:
            v1: First input vector.
            v2: Second input vector.
        
        Returns:
            float: The dot product value.
        
        Raises:
            ValueError: If vectors have different dimensions.
        
        Example:
            >>> ops = VectorOperations()
            >>> v1 = [1, 2, 3]
            >>> v2 = [4, 5, 6]
            >>> ops.dot_product(v1, v2)
            32.0
            >>> # Verification: 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
        """
        v1_arr = self._validate_vector(v1, "v1")
        v2_arr = self._validate_vector(v2, "v2")
        self._validate_same_dimension(v1_arr, v2_arr, "dot product")
        
        result = float(np.dot(v1_arr, v2_arr))
        logger.debug(f"Dot product of vectors shape {v1_arr.shape}: {result}")
        return result
    
    def cross_product(self, v1: Vector, v2: Vector) -> np.ndarray:
        """
        Compute the cross product of two 3D vectors.
        
        The cross product is defined only for 3D vectors and produces a vector
        perpendicular to both input vectors.
        
        Properties:
        - Anti-commutative: v1 × v2 = -(v2 × v1)
        - ||v1 × v2|| = ||v1|| * ||v2|| * sin(θ)
        - Result is perpendicular to both v1 and v2
        
        Args:
            v1: First 3D vector.
            v2: Second 3D vector.
        
        Returns:
            np.ndarray: The cross product vector (3D).
        
        Raises:
            ValueError: If vectors are not 3D.
        
        Example:
            >>> ops = VectorOperations()
            >>> v1 = [1, 0, 0]  # x-axis
            >>> v2 = [0, 1, 0]  # y-axis
            >>> result = ops.cross_product(v1, v2)
            >>> np.allclose(result, [0, 0, 1])
            True
            >>> # Result points in z-direction (right-hand rule)
        """
        v1_arr = self._validate_vector(v1, "v1")
        v2_arr = self._validate_vector(v2, "v2")
        
        if v1_arr.shape != (3,) or v2_arr.shape != (3,):
            error_msg = f"Cross product requires 3D vectors, got {v1_arr.shape} and {v2_arr.shape}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        result = np.cross(v1_arr, v2_arr)
        logger.debug(f"Cross product computed for 3D vectors")
        return result
    
    def norm(self, v: Vector, ord: Union[int, str] = 2) -> float:
        """
        Compute the norm (magnitude) of a vector.
        
        Supported norms:
        - ord=1: L1 norm (Manhattan distance) = Σ|v_i|
        - ord=2: L2 norm (Euclidean distance) = √(Σv_i²)
        - ord='inf': L-infinity norm = max|v_i|
        - ord='fro': Frobenius norm (for matrices)
        
        Args:
            v: Input vector.
            ord: Order of the norm. Options: 1, 2, 'inf', 'fro'. Default: 2.
        
        Returns:
            float: The norm value.
        
        Raises:
            ValueError: If ord is not supported.
        
        Example:
            >>> ops = VectorOperations()
            >>> v = [3, 4]
            >>> ops.norm(v, ord=2)
            5.0
            >>> ops.norm(v, ord=1)
            7.0
            >>> ops.norm(v, ord='inf')
            4.0
        """
        v_arr = self._validate_vector(v, "v")
        
        valid_ords = [1, 2, 'inf', 'fro', np.inf]
        if ord not in valid_ords:
            error_msg = f"Norm order must be one of {valid_ords}, got {ord}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        result = float(np.linalg.norm(v_arr, ord=ord))
        logger.debug(f"Norm (ord={ord}) of vector shape {v_arr.shape}: {result}")
        return result
    
    def normalize(self, v: Vector, ord: int = 2) -> np.ndarray:
        """
        Normalize a vector to unit length.
        
        Normalization: v_normalized = v / ||v||
        
        Args:
            v: Input vector.
            ord: Order of norm for normalization. Default: 2.
        
        Returns:
            np.ndarray: Unit vector in the same direction as v.
        
        Raises:
            ValueError: If vector has zero norm.
        
        Example:
            >>> ops = VectorOperations()
            >>> v = [3, 4]
            >>> normalized = ops.normalize(v)
            >>> np.allclose(ops.norm(normalized), 1.0)
            True
            >>> np.allclose(normalized, [0.6, 0.8])
            True
        """
        v_arr = self._validate_vector(v, "v")
        norm_value = self.norm(v_arr, ord=ord)
        
        if norm_value < self.epsilon:
            error_msg = "Cannot normalize zero vector"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        result = v_arr / norm_value
        logger.debug(f"Vector normalized with norm={norm_value}")
        return result
    
    def projection(self, v: Vector, onto: Vector) -> np.ndarray:
        """
        Compute the projection of vector v onto vector 'onto'.
        
        Formula: proj_onto(v) = (v · onto / ||onto||²) * onto
        
        The projection is the component of v in the direction of 'onto'.
        
        Args:
            v: Vector to project.
            onto: Vector to project onto.
        
        Returns:
            np.ndarray: The projection vector.
        
        Raises:
            ValueError: If 'onto' is a zero vector.
        
        Example:
            >>> ops = VectorOperations()
            >>> v = [3, 4]
            >>> onto = [1, 0]  # x-axis
            >>> proj = ops.projection(v, onto)
            >>> np.allclose(proj, [3, 0])
            True
            >>> # The projection of [3,4] onto x-axis is [3,0]
        """
        v_arr = self._validate_vector(v, "v")
        onto_arr = self._validate_vector(onto, "onto")
        self._validate_same_dimension(v_arr, onto_arr, "projection")
        
        onto_norm_sq = self.dot_product(onto_arr, onto_arr)
        
        if onto_norm_sq < self.epsilon:
            error_msg = "Cannot project onto zero vector"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        scalar = self.dot_product(v_arr, onto_arr) / onto_norm_sq
        result = scalar * onto_arr
        logger.debug(f"Projection computed with scalar={scalar}")
        return result
    
    def rejection(self, v: Vector, from_vec: Vector) -> np.ndarray:
        """
        Compute the rejection of vector v from vector 'from_vec'.
        
        The rejection is the component of v perpendicular to 'from_vec'.
        
        Formula: rej_from(v) = v - proj_from(v)
        
        Args:
            v: Vector to compute rejection for.
            from_vec: Vector to reject from.
        
        Returns:
            np.ndarray: The rejection vector (perpendicular component).
        
        Example:
            >>> ops = VectorOperations()
            >>> v = [3, 4]
            >>> from_vec = [1, 0]  # x-axis
            >>> rej = ops.rejection(v, from_vec)
            >>> np.allclose(rej, [0, 4])
            True
            >>> # The rejection is the y-component
        """
        v_arr = self._validate_vector(v, "v")
        from_arr = self._validate_vector(from_vec, "from_vec")
        
        proj = self.projection(v_arr, from_arr)
        result = v_arr - proj
        logger.debug("Rejection computed as v - projection")
        return result
    
    def linear_combination(
        self, vectors: List[Vector], coefficients: List[float]
    ) -> np.ndarray:
        """
        Compute a linear combination of vectors.
        
        Formula: result = c1*v1 + c2*v2 + ... + cn*vn
        
        Args:
            vectors: List of vectors to combine.
            coefficients: List of scalar coefficients.
        
        Returns:
            np.ndarray: The resulting linear combination.
        
        Raises:
            ValueError: If number of vectors doesn't match coefficients,
                       or if vectors have different dimensions.
        
        Example:
            >>> ops = VectorOperations()
            >>> v1 = [1, 0]
            >>> v2 = [0, 1]
            >>> result = ops.linear_combination([v1, v2], [3, 4])
            >>> np.allclose(result, [3, 4])
            True
        """
        if len(vectors) != len(coefficients):
            error_msg = f"Number of vectors ({len(vectors)}) must match coefficients ({len(coefficients)})"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if len(vectors) == 0:
            error_msg = "At least one vector is required"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validated_vectors = [self._validate_vector(v, f"v{i}") 
                           for i, v in enumerate(vectors)]
        
        # Check all vectors have same dimension
        first_shape = validated_vectors[0].shape
        for i, v in enumerate(validated_vectors[1:], 1):
            if v.shape != first_shape:
                error_msg = f"All vectors must have same shape, v0={first_shape}, v{i}={v.shape}"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        result = sum(c * v for c, v in zip(coefficients, validated_vectors))
        logger.debug(f"Linear combination of {len(vectors)} vectors computed")
        return result
    
    def angle_between(self, v1: Vector, v2: Vector, degrees: bool = False) -> float:
        """
        Compute the angle between two vectors.
        
        Formula: θ = arccos((v1 · v2) / (||v1|| * ||v2||))
        
        Args:
            v1: First vector.
            v2: Second vector.
            degrees: If True, return angle in degrees. Default: False (radians).
        
        Returns:
            float: Angle between vectors (in radians or degrees).
        
        Raises:
            ValueError: If vectors have different dimensions or zero norm.
        
        Example:
            >>> ops = VectorOperations()
            >>> v1 = [1, 0]
            >>> v2 = [0, 1]
            >>> angle = ops.angle_between(v1, v2, degrees=True)
            >>> np.isclose(angle, 90.0)
            True
        """
        v1_arr = self._validate_vector(v1, "v1")
        v2_arr = self._validate_vector(v2, "v2")
        self._validate_same_dimension(v1_arr, v2_arr, "angle calculation")
        
        norm1 = self.norm(v1_arr)
        norm2 = self.norm(v2_arr)
        
        if norm1 < self.epsilon or norm2 < self.epsilon:
            error_msg = "Cannot compute angle with zero vector"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        dot = self.dot_product(v1_arr, v2_arr)
        cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)
        
        if degrees:
            result = np.degrees(angle_rad)
        else:
            result = float(angle_rad)
        
        logger.debug(f"Angle between vectors: {result} {'degrees' if degrees else 'radians'}")
        return result
    
    def are_orthogonal(self, v1: Vector, v2: Vector, tolerance: float = 1e-9) -> bool:
        """
        Check if two vectors are orthogonal (perpendicular).
        
        Vectors are orthogonal if their dot product is zero.
        
        Args:
            v1: First vector.
            v2: Second vector.
            tolerance: Numerical tolerance for zero check. Default: 1e-9.
        
        Returns:
            bool: True if vectors are orthogonal.
        
        Example:
            >>> ops = VectorOperations()
            >>> v1 = [1, 0]
            >>> v2 = [0, 1]
            >>> ops.are_orthogonal(v1, v2)
            True
        """
        dot = self.dot_product(v1, v2)
        result = abs(dot) < tolerance
        logger.debug(f"Orthogonality check: dot={dot}, orthogonal={result}")
        return result
    
    def are_parallel(self, v1: Vector, v2: Vector, tolerance: float = 1e-9) -> bool:
        """
        Check if two vectors are parallel.
        
        Vectors are parallel if their cross product has zero norm (3D)
        or if one is a scalar multiple of the other.
        
        Args:
            v1: First vector.
            v2: Second vector.
            tolerance: Numerical tolerance. Default: 1e-9.
        
        Returns:
            bool: True if vectors are parallel.
        
        Example:
            >>> ops = VectorOperations()
            >>> v1 = [1, 2, 3]
            >>> v2 = [2, 4, 6]  # 2 * v1
            >>> ops.are_parallel(v1, v2)
            True
        """
        v1_arr = self._validate_vector(v1, "v1")
        v2_arr = self._validate_vector(v2, "v2")
        self._validate_same_dimension(v1_arr, v2_arr, "parallel check")
        
        # Check using cross product for 3D, or angle for other dimensions
        if v1_arr.shape == (3,):
            cross = np.cross(v1_arr, v2_arr)
            result = self.norm(cross) < tolerance
        else:
            # Use angle: parallel if angle is 0 or π
            try:
                angle = self.angle_between(v1_arr, v2_arr)
                result = angle < tolerance or abs(angle - np.pi) < tolerance
            except ValueError:
                result = False
        
        logger.debug(f"Parallel check: result={result}")
        return result
    
    def gram_schmidt(self, vectors: List[Vector]) -> List[np.ndarray]:
        """
        Apply Gram-Schmidt process to orthogonalize a set of vectors.
        
        The Gram-Schmidt process converts a set of linearly independent vectors
        into an orthonormal set spanning the same subspace.
        
        Algorithm:
        1. u1 = v1
        2. u2 = v2 - proj_u1(v2)
        3. u3 = v3 - proj_u1(v3) - proj_u2(v3)
        4. Normalize all ui to get orthonormal basis
        
        Args:
            vectors: List of linearly independent vectors.
        
        Returns:
            List[np.ndarray]: Orthonormal basis vectors.
        
        Raises:
            ValueError: If vectors are linearly dependent.
        
        Example:
            >>> ops = VectorOperations()
            >>> v1 = [1, 1, 0]
            >>> v2 = [1, 0, 1]
            >>> v3 = [0, 1, 1]
            >>> orthonormal = ops.gram_schmidt([v1, v2, v3])
            >>> # Check orthonormality
            >>> all(np.isclose(ops.norm(v), 1.0) for v in orthonormal)
            True
            >>> all(ops.are_orthogonal(orthonormal[i], orthonormal[j]) 
            ...     for i in range(3) for j in range(i+1, 3))
            True
        """
        if len(vectors) == 0:
            error_msg = "At least one vector is required"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        validated = [self._validate_vector(v, f"v{i}") 
                    for i, v in enumerate(vectors)]
        
        # Check all vectors have same dimension
        first_shape = validated[0].shape
        for i, v in enumerate(validated[1:], 1):
            if v.shape != first_shape:
                error_msg = f"All vectors must have same shape"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        orthogonal = []
        for i, v in enumerate(validated):
            u = v.copy()
            # Subtract projections onto all previous orthogonal vectors
            for u_prev in orthogonal:
                proj = self.projection(u, u_prev)
                u = u - proj
            
            norm_u = self.norm(u)
            if norm_u < self.epsilon:
                error_msg = f"Vectors are linearly dependent at index {i}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            orthogonal.append(u / norm_u)
        
        logger.debug(f"Gram-Schmidt completed: {len(orthogonal)} orthonormal vectors")
        return orthogonal
    
    def distance(self, v1: Vector, v2: Vector, metric: str = 'euclidean') -> float:
        """
        Compute distance between two vectors using various metrics.
        
        Supported metrics:
        - 'euclidean': L2 distance = ||v1 - v2||
        - 'manhattan': L1 distance = Σ|v1_i - v2_i|
        - 'chebyshev': L-infinity distance = max|v1_i - v2_i|
        - 'cosine': Cosine distance = 1 - cosine_similarity
        
        Args:
            v1: First vector.
            v2: Second vector.
            metric: Distance metric. Default: 'euclidean'.
        
        Returns:
            float: Distance between vectors.
        
        Raises:
            ValueError: If metric is not supported.
        
        Example:
            >>> ops = VectorOperations()
            >>> v1 = [0, 0]
            >>> v2 = [3, 4]
            >>> ops.distance(v1, v2, 'euclidean')
            5.0
            >>> ops.distance(v1, v2, 'manhattan')
            7.0
        """
        v1_arr = self._validate_vector(v1, "v1")
        v2_arr = self._validate_vector(v2, "v2")
        self._validate_same_dimension(v1_arr, v2_arr, "distance calculation")
        
        diff = v1_arr - v2_arr
        
        if metric == 'euclidean':
            result = self.norm(diff, ord=2)
        elif metric == 'manhattan':
            result = self.norm(diff, ord=1)
        elif metric == 'chebyshev':
            result = self.norm(diff, ord=np.inf)
        elif metric == 'cosine':
            norm1 = self.norm(v1_arr)
            norm2 = self.norm(v2_arr)
            if norm1 < self.epsilon or norm2 < self.epsilon:
                result = 1.0  # Maximum distance for zero vectors
            else:
                cos_sim = self.dot_product(v1_arr, v2_arr) / (norm1 * norm2)
                result = 1.0 - cos_sim
        else:
            error_msg = f"Unsupported metric: {metric}. Options: euclidean, manhattan, chebyshev, cosine"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.debug(f"Distance ({metric}) between vectors: {result}")
        return result
    
    def outer_product(self, v1: Vector, v2: Vector) -> np.ndarray:
        """
        Compute the outer product of two vectors.
        
        The outer product produces a matrix: (v1 ⊗ v2)[i,j] = v1[i] * v2[j]
        
        Args:
            v1: First vector (will form rows).
            v2: Second vector (will form columns).
        
        Returns:
            np.ndarray: Outer product matrix of shape (len(v1), len(v2)).
        
        Example:
            >>> ops = VectorOperations()
            >>> v1 = [1, 2]
            >>> v2 = [3, 4, 5]
            >>> result = ops.outer_product(v1, v2)
            >>> result.shape
            (2, 3)
            >>> np.allclose(result[0], [3, 4, 5])
            True
            >>> np.allclose(result[1], [6, 8, 10])
            True
        """
        v1_arr = self._validate_vector(v1, "v1")
        v2_arr = self._validate_vector(v2, "v2")
        
        result = np.outer(v1_arr, v2_arr)
        logger.debug(f"Outer product computed: shape {result.shape}")
        return result


def create_basis_vector(dimension: int, index: int) -> np.ndarray:
    """
    Create a standard basis vector (one-hot encoding).
    
    A standard basis vector has 1 at the specified index and 0 elsewhere.
    
    Args:
        dimension: Dimension of the vector space.
        index: Index where the value is 1 (0-indexed).
    
    Returns:
        np.ndarray: Standard basis vector.
    
    Raises:
        ValueError: If index is out of range.
    
    Example:
        >>> e1 = create_basis_vector(3, 0)
        >>> np.allclose(e1, [1, 0, 0])
        True
        >>> e2 = create_basis_vector(3, 1)
        >>> np.allclose(e2, [0, 1, 0])
        True
    """
    if index < 0 or index >= dimension:
        raise ValueError(f"Index {index} out of range for dimension {dimension}")
    
    result = np.zeros(dimension)
    result[index] = 1.0
    logger.debug(f"Created basis vector e_{index} in R^{dimension}")
    return result


def random_vector(
    dimension: int, 
    low: float = 0.0, 
    high: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a random vector with uniform distribution.
    
    Args:
        dimension: Dimension of the vector.
        low: Lower bound (inclusive). Default: 0.0.
        high: Upper bound (exclusive). Default: 1.0.
        seed: Random seed for reproducibility. Default: None.
    
    Returns:
        np.ndarray: Random vector.
    
    Example:
        >>> v = random_vector(3, seed=42)
        >>> v.shape
        (3,)
        >>> all(0 <= x < 1 for x in v)
        True
    """
    if seed is not None:
        np.random.seed(seed)
    
    result = np.random.uniform(low, high, dimension)
    logger.debug(f"Generated random vector in R^{dimension} with range [{low}, {high})")
    return result


def random_unit_vector(dimension: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random unit vector (uniformly distributed on the unit sphere).
    
    Uses the method of normalizing a vector of Gaussian random variables.
    
    Args:
        dimension: Dimension of the vector space.
        seed: Random seed for reproducibility. Default: None.
    
    Returns:
        np.ndarray: Random unit vector.
    
    Example:
        >>> v = random_unit_vector(3, seed=42)
        >>> np.isclose(np.linalg.norm(v), 1.0)
        True
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate Gaussian random vector and normalize
    v = np.random.randn(dimension)
    ops = VectorOperations()
    result = ops.normalize(v)
    logger.debug(f"Generated random unit vector in R^{dimension}")
    return result


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("Vector Operations Module - Demonstration")
    print("=" * 60)
    
    ops = VectorOperations()
    
    # Basic operations
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([4.0, 5.0, 6.0])
    
    print(f"\nv1 = {v1}")
    print(f"v2 = {v2}")
    
    print(f"\nDot product: {ops.dot_product(v1, v2)}")
    print(f"Cross product: {ops.cross_product(v1, v2)}")
    print(f"L2 norm of v1: {ops.norm(v1)}")
    print(f"L1 norm of v1: {ops.norm(v1, ord=1)}")
    
    print(f"\nNormalized v1: {ops.normalize(v1)}")
    print(f"Projection of v1 onto v2: {ops.projection(v1, v2)}")
    
    print(f"\nAngle between v1 and v2: {ops.angle_between(v1, v2, degrees=True):.2f}°")
    print(f"Are orthogonal: {ops.are_orthogonal(v1, v2)}")
    print(f"Are parallel: {ops.are_parallel(v1, v2)}")
    
    print(f"\nDistance (Euclidean): {ops.distance(v1, v2, 'euclidean'):.4f}")
    print(f"Distance (Manhattan): {ops.distance(v1, v2, 'manhattan'):.4f}")
    print(f"Distance (Cosine): {ops.distance(v1, v2, 'cosine'):.4f}")
    
    # Linear combination
    print(f"\nLinear combination 2*v1 + 3*v2: {ops.linear_combination([v1, v2], [2, 3])}")
    
    # Gram-Schmidt
    print("\nGram-Schmidt orthogonalization:")
    vectors = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    orthonormal = ops.gram_schmidt(vectors)
    for i, v in enumerate(orthonormal):
        print(f"  u{i+1} = {v}")
    
    # Basis vectors
    print("\nStandard basis vectors in R³:")
    for i in range(3):
        print(f"  e{i} = {create_basis_vector(3, i)}")
    
    print("\n" + "=" * 60)
