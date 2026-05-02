"""
Test Suite for ML Learning Module - Mathematical Foundations

Tests cover:
- Vector operations
- Matrix operations
- Calculus and optimization
- Probability distributions
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import modules we're testing
from ml_learning_module.module_01_math.vectors import VectorOperations
from ml_learning_module.module_01_math.matrices import MatrixOperations
from ml_learning_module.module_01_math.calculus import Optimizer, CalculusOperations
from ml_learning_module.module_01_math.probability import (
    Distribution,
    ProbabilityOperations,
)


class TestVectorOperations:
    """Test suite for vector operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ops = VectorOperations()
        self.epsilon = 1e-6

    def test_dot_product(self):
        """Test dot product calculation."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])

        result = self.ops.dot_product(v1, v2)
        expected = 1 * 4 + 2 * 5 + 3 * 6  # = 32

        assert abs(result - expected) < self.epsilon

    def test_dot_product_commutative(self):
        """Verify dot product is commutative."""
        v1 = np.array([1, 2, 3])
        v2 = np.array([4, 5, 6])

        assert self.ops.dot_product(v1, v2) == self.ops.dot_product(v2, v1)

    def test_norm_l2(self):
        """Test L2 norm calculation."""
        v = np.array([3, 4])

        result = self.ops.norm(v, ord=2)

        assert abs(result - 5.0) < self.epsilon

    def test_norm_l1(self):
        """Test L1 norm calculation."""
        v = np.array([3, 4])

        result = self.ops.norm(v, ord=1)

        assert abs(result - 7.0) < self.epsilon

    def test_norm_infinity(self):
        """Test infinity norm calculation."""
        v = np.array([3, -4, 2])

        result = self.ops.norm(v, ord=np.inf)

        assert abs(result - 4.0) < self.epsilon

    def test_normalize(self):
        """Test vector normalization."""
        v = np.array([3, 4])

        result = self.ops.normalize(v)

        # Check that norm is 1
        assert abs(self.ops.norm(result) - 1.0) < self.epsilon
        # Check direction is preserved
        assert abs(result[0] / result[1] - 3 / 4) < self.epsilon

    def test_projection(self):
        """Test vector projection."""
        v = np.array([3, 4])
        onto = np.array([1, 0])

        result = self.ops.projection(v, onto)

        assert abs(result[0] - 3.0) < self.epsilon
        assert abs(result[1] - 0.0) < self.epsilon

    def test_angle_between_perpendicular(self):
        """Test angle between perpendicular vectors."""
        v1 = np.array([1, 0])
        v2 = np.array([0, 1])

        angle = self.ops.angle_between(v1, v2, degrees=True)

        assert abs(angle - 90.0) < self.epsilon

    def test_angle_between_parallel(self):
        """Test angle between parallel vectors."""
        v1 = np.array([1, 2])
        v2 = np.array([2, 4])

        angle = self.ops.angle_between(v1, v2, degrees=True)

        assert abs(angle - 0.0) < self.epsilon

    def test_orthogonal(self):
        """Test orthogonal vectors check."""
        v1 = np.array([1, 0])
        v2 = np.array([0, 1])

        assert self.ops.are_orthogonal(v1, v2)

    def test_linear_combination(self):
        """Test linear combination of vectors."""
        v1 = np.array([1, 0])
        v2 = np.array([0, 1])

        result = self.ops.linear_combination([v1, v2], [3, 4])

        expected = np.array([3, 4])
        np.testing.assert_array_almost_equal(result, expected)

    def test_gram_schmidt(self):
        """Test Gram-Schmidt orthogonalization."""
        v1 = np.array([1, 1, 0])
        v2 = np.array([1, 0, 1])

        orthonormal = self.ops.gram_schmidt([v1, v2])

        # Check that resulting vectors are normalized
        for v in orthonormal:
            assert abs(self.ops.norm(v) - 1.0) < self.epsilon

        # Check orthogonality
        assert self.ops.are_orthogonal(orthonormal[0], orthonormal[1])

    def test_distance_euclidean(self):
        """Test Euclidean distance."""
        v1 = np.array([0, 0])
        v2 = np.array([3, 4])

        result = self.ops.distance(v1, v2, "euclidean")

        assert abs(result - 5.0) < self.epsilon

    def test_distance_manhattan(self):
        """Test Manhattan distance."""
        v1 = np.array([0, 0])
        v2 = np.array([3, 4])

        result = self.ops.distance(v1, v2, "manhattan")

        assert abs(result - 7.0) < self.epsilon


class TestMatrixOperations:
    """Test suite for matrix operations."""

    def setup_method(self):
        """Set up test fixtures."""
        self.ops = MatrixOperations()
        self.epsilon = 1e-6

    def test_matrix_addition(self):
        """Test matrix addition."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])

        result = self.ops.add(A, B)
        expected = np.array([[6, 8], [10, 12]])

        np.testing.assert_array_almost_equal(result, expected)

    def test_matrix_multiplication(self):
        """Test matrix multiplication."""
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])

        result = self.ops.matmul(A, B)
        expected = np.array([[19, 22], [43, 50]])

        np.testing.assert_array_almost_equal(result, expected)

    def test_transpose(self):
        """Test matrix transpose."""
        A = np.array([[1, 2, 3], [4, 5, 6]])

        result = self.ops.transpose(A)
        expected = np.array([[1, 4], [2, 5], [3, 6]])

        np.testing.assert_array_almost_equal(result, expected)

    def test_determinant_2x2(self):
        """Test determinant for 2x2 matrix."""
        A = np.array([[1, 2], [3, 4]])

        result = self.ops.determinant(A)

        assert abs(result - (-2.0)) < self.epsilon

    def test_determinant_3x3(self):
        """Test determinant for 3x3 matrix."""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        result = self.ops.determinant(A)

        assert abs(result - 0.0) < self.epsilon  # Singular matrix

    def test_inverse(self):
        """Test matrix inverse."""
        A = np.array([[1, 2], [3, 4]])

        result = self.ops.inverse(A)

        # Verify: A @ A^{-1} = I
        identity = result @ A
        np.testing.assert_array_almost_equal(identity, np.eye(2), decimal=5)

    def test_identity_creation(self):
        """Test identity matrix creation."""
        result = self.ops.identity(3)

        expected = np.eye(3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_zeros_creation(self):
        """Test zeros matrix creation."""
        result = self.ops.zeros(2, 3)

        expected = np.zeros((2, 3))
        np.testing.assert_array_almost_equal(result, expected)

    def test_rank_full_rank(self):
        """Test rank of full rank matrix."""
        A = np.array([[1, 0], [0, 1]])

        result = self.ops.rank(A)

        assert result == 2

    def test_rank_singular(self):
        """Test rank of singular matrix."""
        A = np.array([[1, 2], [2, 4]])

        result = self.ops.rank(A)

        assert result == 1

    def test_trace(self):
        """Test matrix trace."""
        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        result = self.ops.trace(A)

        assert result == 15  # 1 + 5 + 9


class TestCalculusOperations:
    """Test suite for calculus and optimization."""

    def setup_method(self):
        """Set up test fixtures."""
        self.calc = CalculusOperations()
        self.epsilon = 1e-4

    def test_derivative_polynomial(self):
        """Test derivative of polynomial."""

        def f(x):
            return x**3

        def expected_deriv(x):
            return 3 * x**2

        x = 2.0
        result = self.calc.derivative(f, x)
        expected = expected_deriv(x)

        assert abs(result - expected) < self.epsilon

    def test_gradient_2d(self):
        """Test gradient in 2D."""

        def f(x):
            return x[0] ** 2 + x[1] ** 2

        result = self.calc.gradient(f, np.array([3.0, 4.0]))
        expected = np.array([6.0, 8.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_partial_derivative(self):
        """Test partial derivative."""

        def f(x):
            return x[0] ** 2 * x[1]

        # ∂/∂x₁ of x₁²x₂ at (2, 3) = 2*2*3 = 12
        result = self.calc.partial_derivative(f, np.array([2.0, 3.0]), 0)

        assert abs(result - 12.0) < self.epsilon


class TestOptimizer:
    """Test suite for optimization algorithms."""

    def setup_method(self):
        """Set up test fixtures."""
        self.epsilon = 1e-3

    def test_gradient_descent_simple(self):
        """Test gradient descent on simple function."""
        # f(x) = (x - 3)², minimum at x = 3
        opt = Optimizer(learning_rate=0.1, method="gd")

        result = opt.minimize(
            lambda x: (x[0] - 3) ** 2, x0=np.array([0.0]), n_iterations=100
        )

        assert abs(result.x[0] - 3.0) < 0.1

    def test_gradient_descent_2d(self):
        """Test gradient descent in 2D."""
        # f(x,y) = (x-1)² + (y-2)², minimum at (1, 2)
        opt = Optimizer(learning_rate=0.1, method="gd")

        result = opt.minimize(
            lambda x: (x[0] - 1) ** 2 + (x[1] - 2) ** 2,
            x0=np.array([0.0, 0.0]),
            n_iterations=200,
        )

        assert abs(result.x[0] - 1.0) < 0.1
        assert abs(result.x[1] - 2.0) < 0.1

    def test_momentum(self):
        """Test momentum optimizer."""
        opt = Optimizer(learning_rate=0.1, method="momentum", momentum=0.9)

        result = opt.minimize(
            lambda x: (x[0] - 5) ** 2, x0=np.array([0.0]), n_iterations=100
        )

        assert abs(result.x[0] - 5.0) < 0.1

    def test_adam(self):
        """Test Adam optimizer."""
        opt = Optimizer(learning_rate=0.1, method="adam")

        result = opt.minimize(
            lambda x: (x[0] - 3) ** 2 + (x[1] - 4) ** 2,
            x0=np.array([0.0, 0.0]),
            n_iterations=100,
        )

        assert abs(result.x[0] - 3.0) < 0.1
        assert abs(result.x[1] - 4.0) < 0.1


class TestProbability:
    """Test suite for probability distributions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.epsilon = 1e-3

    def test_normal_distribution(self):
        """Test normal distribution properties."""
        dist = Distribution.normal(mean=0, std=1)

        # Mean should be approximately 0
        samples = dist.sample(10000)
        mean = np.mean(samples)

        assert abs(mean - 0.0) < 0.1

    def test_normal_variance(self):
        """Test normal distribution variance."""
        dist = Distribution.normal(mean=0, std=2)

        samples = dist.sample(10000)
        variance = np.var(samples)

        assert abs(variance - 4.0) < 0.2

    def test_uniform_distribution(self):
        """Test uniform distribution."""
        dist = Distribution.uniform(low=0, high=10)

        samples = dist.sample(10000)
        mean = np.mean(samples)

        assert abs(mean - 5.0) < 0.1

    def test_bernoulli(self):
        """Test Bernoulli distribution."""
        dist = Distribution.bernoulli(p=0.7)

        samples = dist.sample(1000)
        mean = np.mean(samples)

        assert abs(mean - 0.7) < 0.05


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_linear_regression_integration(self):
        """Test linear regression using math components."""
        from ml_learning_module.module_02_ml.linear_regression import LinearRegression

        # Generate sample data
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(100) * 0.1

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Check predictions
        predictions = model.predict(X[:5])

        # Predictions should be reasonable
        assert predictions.shape == (5,)

    def test_nlp_pipeline_integration(self):
        """Test NLP pipeline end-to-end."""
        from ml_learning_module.module_04_nlp.text_preprocessing import TextPreprocessor
        from ml_learning_module.module_04_nlp.vectorization import BagOfWords

        texts = [
            "This is a test document",
            "Another test with different words",
            "Machine learning is great",
        ]

        # Preprocess
        preprocessor = TextPreprocessor(lowercase=True)
        processed = [preprocessor.preprocess(t) for t in texts]

        # Vectorize
        vectorizer = BagOfWords()
        vectorizer.fit(processed)
        features = vectorizer.transform(processed)

        assert features.shape[0] == 3
        assert features.shape[1] > 0


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
