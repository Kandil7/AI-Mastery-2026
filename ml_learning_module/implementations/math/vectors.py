"""
Vector Operations Module for Machine Learning

This module provides comprehensive vector operations essential for ML mathematics,
including dot product, cross product, norms, projections, and linear combinations.

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Union, List, Tuple, Optional
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

    Example Usage:
        >>> import numpy as np
        >>> ops = VectorOperations()
        >>> v1 = np.array([1, 2, 3])
        >>> v2 = np.array([4, 5, 6])
        >>> dot = ops.dot_product(v1, v2)
        >>> print(f"Dot product: {dot}")
    """

    def __init__(self, epsilon: float = 1e-10):
        """
        Initialize VectorOperations with numerical stability parameter.

        Args:
            epsilon: Small value to prevent division by zero. Default: 1e-10.
        """
        self.epsilon = epsilon

    def _validate_vector(self, v: Vector, name: str = "vector") -> np.ndarray:
        """
        Validate and convert input to numpy array.

        Args:
            v: Input vector (list, tuple, or numpy array).
            name: Name of the vector for error messages.

        Returns:
            Validated numpy array.

        Raises:
            TypeError: If input is not array-like.
            ValueError: If vector is empty or contains non-numeric values.
        """
        try:
            arr = np.asarray(v, dtype=np.float64)
        except (TypeError, ValueError) as e:
            logger.error(f"{name} must be array-like: {e}")
            raise TypeError(f"{name} must be array-like (list, tuple, or numpy array)")

        if arr.size == 0:
            raise ValueError(f"{name} cannot be empty")

        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"{name} must contain numeric values")

        return arr

    def _validate_same_dimension(
        self, v1: np.ndarray, v2: np.ndarray, op_name: str = "operation"
    ) -> None:
        """Validate that two vectors have the same dimension."""
        if v1.shape != v2.shape:
            raise ValueError(
                f"Vectors must have same shape for {op_name}: {v1.shape} vs {v2.shape}"
            )

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
            The dot product value (scalar).

        Example:
            >>> ops = VectorOperations()
            >>> ops.dot_product([1, 2, 3], [4, 5, 6])
            32.0
        """
        v1_arr = self._validate_vector(v1, "v1")
        v2_arr = self._validate_vector(v2, "v2")
        self._validate_same_dimension(v1_arr, v2_arr, "dot product")

        return float(np.dot(v1_arr, v2_arr))

    def cross_product(self, v1: Vector, v2: Vector) -> np.ndarray:
        """
        Compute the cross product of two 3D vectors.

        The cross product produces a vector perpendicular to both input vectors.

        Properties:
        - Anti-commutative: v1 × v2 = -(v2 × v1)
        - ||v1 × v2|| = ||v1|| * ||v2|| * sin(θ)

        Args:
            v1: First 3D vector.
            v2: Second 3D vector.

        Returns:
            The cross product vector (3D).

        Raises:
            ValueError: If vectors are not 3D.

        Example:
            >>> ops = VectorOperations()
            >>> ops.cross_product([1, 0, 0], [0, 1, 0])
            array([0., 0., 1.])
        """
        v1_arr = self._validate_vector(v1, "v1")
        v2_arr = self._validate_vector(v2, "v2")

        if v1_arr.shape != (3,) or v2_arr.shape != (3,):
            raise ValueError(
                f"Cross product requires 3D vectors, got {v1_arr.shape} and {v2_arr.shape}"
            )

        return np.cross(v1_arr, v2_arr)

    def norm(self, v: Vector, ord: Union[int, str] = 2) -> float:
        """
        Compute the norm (magnitude) of a vector.

        Supported norms:
        - ord=1: L1 norm (Manhattan distance) = Σ|v_i|
        - ord=2: L2 norm (Euclidean distance) = √(Σv_i²)
        - ord='inf': L-infinity norm = max|v_i|

        Args:
            v: Input vector.
            ord: Order of the norm. Default: 2.

        Returns:
            The norm value (scalar).

        Example:
            >>> ops = VectorOperations()
            >>> ops.norm([3, 4], ord=2)
            5.0
        """
        v_arr = self._validate_vector(v, "v")

        valid_ords = [1, 2, "inf", "fro", np.inf]
        if ord not in valid_ords:
            raise ValueError(f"Norm order must be one of {valid_ords}, got {ord}")

        return float(np.linalg.norm(v_arr, ord=ord))

    def normalize(self, v: Vector, ord: int = 2) -> np.ndarray:
        """
        Normalize a vector to unit length.

        Normalization: v_normalized = v / ||v||

        Args:
            v: Input vector.
            ord: Order of norm for normalization. Default: 2.

        Returns:
            Unit vector in the same direction as v.

        Raises:
            ValueError: If vector has zero norm.

        Example:
            >>> ops = VectorOperations()
            >>> ops.normalize([3, 4])
            array([0.6, 0.8])
        """
        v_arr = self._validate_vector(v, "v")
        norm_value = self.norm(v_arr, ord=ord)

        if norm_value < self.epsilon:
            raise ValueError("Cannot normalize zero vector")

        return v_arr / norm_value

    def projection(self, v: Vector, onto: Vector) -> np.ndarray:
        """
        Compute the projection of vector v onto vector 'onto'.

        Formula: proj_onto(v) = (v · onto / ||onto||²) * onto

        Args:
            v: Vector to project.
            onto: Vector to project onto.

        Returns:
            The projection vector.

        Example:
            >>> ops = VectorOperations()
            >>> ops.projection([3, 4], [1, 0])
            array([3., 0.])
        """
        v_arr = self._validate_vector(v, "v")
        onto_arr = self._validate_vector(onto, "onto")
        self._validate_same_dimension(v_arr, onto_arr, "projection")

        onto_norm_sq = self.dot_product(onto_arr, onto_arr)

        if onto_norm_sq < self.epsilon:
            raise ValueError("Cannot project onto zero vector")

        scalar = self.dot_product(v_arr, onto_arr) / onto_norm_sq
        return scalar * onto_arr

    def rejection(self, v: Vector, from_vec: Vector) -> np.ndarray:
        """
        Compute the rejection of vector v from vector 'from_vec'.

        The rejection is the component of v perpendicular to 'from_vec'.

        Formula: rej_from(v) = v - proj_from(v)

        Args:
            v: Vector to compute rejection for.
            from_vec: Vector to reject from.

        Returns:
            The rejection vector (perpendicular component).

        Example:
            >>> ops = VectorOperations()
            >>> ops.rejection([3, 4], [1, 0])
            array([0., 4.])
        """
        v_arr = self._validate_vector(v, "v")
        from_arr = self._validate_vector(from_vec, "from_vec")

        proj = self.projection(v_arr, from_arr)
        return v_arr - proj

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
            The resulting linear combination vector.

        Raises:
            ValueError: If number of vectors doesn't match coefficients.

        Example:
            >>> ops = VectorOperations()
            >>> ops.linear_combination([[1, 0], [0, 1]], [3, 4])
            array([3., 4.])
        """
        if len(vectors) != len(coefficients):
            raise ValueError(
                f"Number of vectors ({len(vectors)}) must match coefficients ({len(coefficients)})"
            )

        validated_vectors = [
            self._validate_vector(v, f"v{i}") for i, v in enumerate(vectors)
        ]

        return sum(c * v for c, v in zip(coefficients, validated_vectors))

    def angle_between(self, v1: Vector, v2: Vector, degrees: bool = False) -> float:
        """
        Compute the angle between two vectors.

        Formula: θ = arccos((v1 · v2) / (||v1|| * ||v2||))

        Args:
            v1: First vector.
            v2: Second vector.
            degrees: If True, return angle in degrees. Default: False (radians).

        Returns:
            Angle between vectors (in radians or degrees).

        Example:
            >>> ops = VectorOperations()
            >>> ops.angle_between([1, 0], [0, 1], degrees=True)
            90.0
        """
        v1_arr = self._validate_vector(v1, "v1")
        v2_arr = self._validate_vector(v2, "v2")
        self._validate_same_dimension(v1_arr, v2_arr, "angle calculation")

        norm1 = self.norm(v1_arr)
        norm2 = self.norm(v2_arr)

        if norm1 < self.epsilon or norm2 < self.epsilon:
            raise ValueError("Cannot compute angle with zero vector")

        dot = self.dot_product(v1_arr, v2_arr)
        cos_theta = np.clip(dot / (norm1 * norm2), -1.0, 1.0)
        angle_rad = np.arccos(cos_theta)

        return np.degrees(angle_rad) if degrees else float(angle_rad)

    def are_orthogonal(self, v1: Vector, v2: Vector, tolerance: float = 1e-9) -> bool:
        """
        Check if two vectors are orthogonal (perpendicular).

        Vectors are orthogonal if their dot product is zero.

        Args:
            v1: First vector.
            v2: Second vector.
            tolerance: Numerical tolerance for zero check. Default: 1e-9.

        Returns:
            True if vectors are orthogonal.

        Example:
            >>> ops = VectorOperations()
            >>> ops.are_orthogonal([1, 0], [0, 1])
            True
        """
        return abs(self.dot_product(v1, v2)) < tolerance

    def are_parallel(self, v1: Vector, v2: Vector, tolerance: float = 1e-9) -> bool:
        """
        Check if two vectors are parallel.

        Args:
            v1: First vector.
            v2: Second vector.
            tolerance: Numerical tolerance. Default: 1e-9.

        Returns:
            True if vectors are parallel.

        Example:
            >>> ops = VectorOperations()
            >>> ops.are_parallel([1, 2, 3], [2, 4, 6])
            True
        """
        v1_arr = self._validate_vector(v1, "v1")
        v2_arr = self._validate_vector(v2, "v2")

        if v1_arr.shape == (3,):
            cross = np.cross(v1_arr, v2_arr)
            return self.norm(cross) < tolerance
        else:
            try:
                angle = self.angle_between(v1_arr, v2_arr)
                return angle < tolerance or abs(angle - np.pi) < tolerance
            except ValueError:
                return False

    def gram_schmidt(self, vectors: List[Vector]) -> List[np.ndarray]:
        """
        Apply Gram-Schmidt process to orthogonalize a set of vectors.

        The Gram-Schmidt process converts a set of linearly independent vectors
        into an orthonormal set spanning the same subspace.

        Args:
            vectors: List of linearly independent vectors.

        Returns:
            Orthonormal basis vectors.

        Raises:
            ValueError: If vectors are linearly dependent.

        Example:
            >>> ops = VectorOperations()
            >>> ortho = ops.gram_schmidt([[1, 1, 0], [1, 0, 1]])
            >>> [ops.norm(v) for v in ortho]
            [1.0, 1.0]
        """
        if len(vectors) == 0:
            raise ValueError("At least one vector is required")

        validated = [self._validate_vector(v, f"v{i}") for i, v in enumerate(vectors)]

        orthogonal = []
        for i, v in enumerate(validated):
            u = v.copy()
            for u_prev in orthogonal:
                proj = self.projection(u, u_prev)
                u = u - proj

            norm_u = self.norm(u)
            if norm_u < self.epsilon:
                raise ValueError(f"Vectors are linearly dependent at index {i}")

            orthogonal.append(u / norm_u)

        return orthogonal

    def distance(self, v1: Vector, v2: Vector, metric: str = "euclidean") -> float:
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
            Distance between vectors.

        Example:
            >>> ops = VectorOperations()
            >>> ops.distance([0, 0], [3, 4])
            5.0
        """
        v1_arr = self._validate_vector(v1, "v1")
        v2_arr = self._validate_vector(v2, "v2")
        self._validate_same_dimension(v1_arr, v2_arr, "distance calculation")

        diff = v1_arr - v2_arr

        if metric == "euclidean":
            return self.norm(diff, ord=2)
        elif metric == "manhattan":
            return self.norm(diff, ord=1)
        elif metric == "chebyshev":
            return self.norm(diff, ord=np.inf)
        elif metric == "cosine":
            norm1 = self.norm(v1_arr)
            norm2 = self.norm(v2_arr)
            if norm1 < self.epsilon or norm2 < self.epsilon:
                return 1.0
            cos_sim = self.dot_product(v1_arr, v2_arr) / (norm1 * norm2)
            return 1.0 - cos_sim
        else:
            raise ValueError(f"Unsupported metric: {metric}")

    def outer_product(self, v1: Vector, v2: Vector) -> np.ndarray:
        """
        Compute the outer product of two vectors.

        The outer product produces a matrix: (v1 ⊗ v2)[i,j] = v1[i] * v2[j]

        Args:
            v1: First vector (will form rows).
            v2: Second vector (will form columns).

        Returns:
            Outer product matrix of shape (len(v1), len(v2)).

        Example:
            >>> ops = VectorOperations()
            >>> ops.outer_product([1, 2], [3, 4, 5]).shape
            (2, 3)
        """
        v1_arr = self._validate_vector(v1, "v1")
        v2_arr = self._validate_vector(v2, "v2")

        return np.outer(v1_arr, v2_arr)


# Helper functions
def create_basis_vector(dimension: int, index: int) -> np.ndarray:
    """
    Create a standard basis vector (one-hot encoding).

    Args:
        dimension: Dimension of the vector space.
        index: Index where the value is 1 (0-indexed).

    Returns:
        Standard basis vector.

    Example:
        >>> create_basis_vector(3, 0)
        array([1., 0., 0.])
    """
    if index < 0 or index >= dimension:
        raise ValueError(f"Index {index} out of range for dimension {dimension}")

    result = np.zeros(dimension)
    result[index] = 1.0
    return result


def random_vector(
    dimension: int, low: float = 0.0, high: float = 1.0, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate a random vector with uniform distribution.

    Args:
        dimension: Dimension of the vector.
        low: Lower bound (inclusive). Default: 0.0.
        high: Upper bound (exclusive). Default: 1.0.
        seed: Random seed for reproducibility. Default: None.

    Returns:
        Random vector.

    Example:
        >>> random_vector(3, seed=42).shape
        (3,)
    """
    if seed is not None:
        np.random.seed(seed)

    return np.random.uniform(low, high, dimension)


def random_unit_vector(dimension: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a random unit vector (uniformly distributed on the unit sphere).

    Uses the method of normalizing a vector of Gaussian random variables.

    Args:
        dimension: Dimension of the vector space.
        seed: Random seed for reproducibility. Default: None.

    Returns:
        Random unit vector.

    Example:
        >>> v = random_unit_vector(3, seed=42)
        >>> np.isclose(np.linalg.norm(v), 1.0)
        True
    """
    if seed is not None:
        np.random.seed(seed)

    v = np.random.randn(dimension)
    ops = VectorOperations()
    return ops.normalize(v)


# Main execution
if __name__ == "__main__":
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
    print(f"L2 norm of v1: {ops.norm(v1)}")
    print(f"L1 norm of v1: {ops.norm(v1, ord=1)}")

    print(f"\nNormalized v1: {ops.normalize(v1)}")
    print(f"Projection of v1 onto v2: {ops.projection(v1, v2)}")

    print(f"\nAngle between v1 and v2: {ops.angle_between(v1, v2, degrees=True):.2f}°")
    print(f"Are orthogonal: {ops.are_orthogonal(v1, v2)}")
    print(f"Are parallel: {ops.are_parallel(v1, v2)}")

    print(f"\nDistance (Euclidean): {ops.distance(v1, v2, 'euclidean'):.4f}")
    print(f"Distance (Manhattan): {ops.distance(v1, v2, 'manhattan'):.4f}")

    # Linear combination
    print(
        f"\nLinear combination 2*v1 + 3*v2: {ops.linear_combination([v1, v2], [2, 3])}"
    )

    # Gram-Schmidt
    print("\nGram-Schmidt orthogonalization:")
    vectors = [[1, 1, 0], [1, 0, 1], [0, 1, 1]]
    orthonormal = ops.gram_schmidt(vectors)
    for i, v in enumerate(orthonormal):
        print(f"  u{i + 1} = {v}")

    # Basis vectors
    print("\nStandard basis vectors in R³:")
    for i in range(3):
        print(f"  e{i} = {create_basis_vector(3, i)}")

    print("\n" + "=" * 60)
