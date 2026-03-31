"""
Tests for Module 1.1: Mathematics for ML.
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from part1_fundamentals.module_1_1_mathematics.vectors import (
    VectorOperations,
    create_basis_vector,
    random_vector,
    random_unit_vector,
)
from part1_fundamentals.module_1_1_mathematics.matrices import (
    MatrixOperations,
    create_hadamard_matrix,
    create_vandermonde_matrix,
)
from part1_fundamentals.module_1_1_mathematics.calculus import (
    CalculusOperations,
    Optimizer,
    OptimizationMethod,
    numerical_integration,
    find_root,
)
from part1_fundamentals.module_1_1_mathematics.probability import (
    ProbabilityOperations,
    HypothesisTesting,
    Distribution,
    DistributionType,
)


class TestVectorOperations(unittest.TestCase):
    """Tests for vector operations."""
    
    def setUp(self):
        self.ops = VectorOperations()
    
    def test_dot_product(self):
        """Test dot product computation."""
        v1 = np.array([1.0, 2.0, 3.0])
        v2 = np.array([4.0, 5.0, 6.0])
        result = self.ops.dot_product(v1, v2)
        expected = 32.0  # 1*4 + 2*5 + 3*6
        self.assertAlmostEqual(result, expected)
    
    def test_cross_product(self):
        """Test cross product computation."""
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        result = self.ops.cross_product(v1, v2)
        expected = np.array([0.0, 0.0, 1.0])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_norm_l2(self):
        """Test L2 norm computation."""
        v = np.array([3.0, 4.0])
        result = self.ops.norm(v, ord=2)
        self.assertAlmostEqual(result, 5.0)
    
    def test_norm_l1(self):
        """Test L1 norm computation."""
        v = np.array([3.0, -4.0, 5.0])
        result = self.ops.norm(v, ord=1)
        self.assertAlmostEqual(result, 12.0)
    
    def test_normalize(self):
        """Test vector normalization."""
        v = np.array([3.0, 4.0])
        normalized = self.ops.normalize(v)
        self.assertAlmostEqual(self.ops.norm(normalized), 1.0)
    
    def test_projection(self):
        """Test vector projection."""
        v = np.array([3.0, 4.0])
        onto = np.array([1.0, 0.0])
        proj = self.ops.projection(v, onto)
        np.testing.assert_array_almost_equal(proj, np.array([3.0, 0.0]))
    
    def test_angle_between(self):
        """Test angle between vectors."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        angle = self.ops.angle_between(v1, v2, degrees=True)
        self.assertAlmostEqual(angle, 90.0)
    
    def test_orthogonal(self):
        """Test orthogonality check."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        self.assertTrue(self.ops.are_orthogonal(v1, v2))
    
    def test_linear_combination(self):
        """Test linear combination."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        result = self.ops.linear_combination([v1, v2], [3.0, 4.0])
        np.testing.assert_array_almost_equal(result, np.array([3.0, 4.0]))
    
    def test_gram_schmidt(self):
        """Test Gram-Schmidt orthogonalization."""
        vectors = [[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
        orthonormal = self.ops.gram_schmidt(vectors)
        
        # Check orthonormality
        for v in orthonormal:
            self.assertAlmostEqual(self.ops.norm(v), 1.0)
        
        for i in range(len(orthonormal)):
            for j in range(i + 1, len(orthonormal)):
                self.assertTrue(self.ops.are_orthogonal(orthonormal[i], orthonormal[j]))
    
    def test_distance(self):
        """Test distance computation."""
        v1 = np.array([0.0, 0.0])
        v2 = np.array([3.0, 4.0])
        
        euclidean = self.ops.distance(v1, v2, 'euclidean')
        self.assertAlmostEqual(euclidean, 5.0)
        
        manhattan = self.ops.distance(v1, v2, 'manhattan')
        self.assertAlmostEqual(manhattan, 7.0)


class TestMatrixOperations(unittest.TestCase):
    """Tests for matrix operations."""
    
    def setUp(self):
        self.ops = MatrixOperations()
    
    def test_multiply(self):
        """Test matrix multiplication."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = self.ops.multiply(A, B)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_transpose(self):
        """Test matrix transpose."""
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = self.ops.transpose(A)
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_determinant(self):
        """Test determinant computation."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = self.ops.determinant(A)
        self.assertAlmostEqual(result, -2.0)
    
    def test_inverse(self):
        """Test matrix inverse."""
        A = np.array([[4.0, 7.0], [2.0, 6.0]])
        A_inv = self.ops.inverse(A)
        I = self.ops.multiply(A, A_inv)
        np.testing.assert_array_almost_equal(I, np.eye(2))
    
    def test_eigenvalues(self):
        """Test eigenvalue computation."""
        A = np.array([[4.0, -2.0], [1.0, 1.0]])
        eigenvalues, eigenvectors = self.ops.eigenvalues(A)
        
        # Verify eigenpairs
        for i in range(len(eigenvalues)):
            Av = self.ops.multiply(A, eigenvectors[:, i])
            lv = eigenvalues[i] * eigenvectors[:, i]
            np.testing.assert_array_almost_equal(Av, lv)
    
    def test_svd(self):
        """Test SVD computation."""
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        U, S, Vt = self.ops.svd(A)
        
        # Reconstruct A
        Sigma = np.diag(S)
        A_reconstructed = self.ops.multiply(self.ops.multiply(U, Sigma), Vt)
        np.testing.assert_array_almost_equal(A, A_reconstructed)
    
    def test_rank(self):
        """Test matrix rank computation."""
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        rank = self.ops.rank(A)
        self.assertEqual(rank, 2)  # Third row is linear combination
    
    def test_cholesky(self):
        """Test Cholesky decomposition."""
        A = np.array([[4.0, 12.0, -16.0], [12.0, 37.0, -43.0], [-16.0, -43.0, 98.0]])
        L = self.ops.cholesky(A)
        LT = self.ops.transpose(L)
        A_reconstructed = self.ops.multiply(L, LT)
        np.testing.assert_array_almost_equal(A, A_reconstructed)
    
    def test_positive_definite(self):
        """Test positive definite check."""
        A = np.array([[2.0, -1.0], [-1.0, 2.0]])
        self.assertTrue(self.ops.is_positive_definite(A))
    
    def test_rotation_matrix(self):
        """Test rotation matrix creation."""
        R = self.ops.rotation_matrix_2d(np.pi / 2)
        v = np.array([1.0, 0.0])
        rotated = self.ops.multiply(R, v.reshape(2, 1)).flatten()
        np.testing.assert_array_almost_equal(rotated, np.array([0.0, 1.0]))


class TestCalculusOperations(unittest.TestCase):
    """Tests for calculus operations."""
    
    def setUp(self):
        self.calc = CalculusOperations()
    
    def test_numerical_derivative(self):
        """Test numerical derivative."""
        def f(x):
            return x ** 3
        
        result = self.calc.numerical_derivative(f, 2.0)
        expected = 12.0  # 3 * 2^2
        self.assertAlmostEqual(result, expected, places=4)
    
    def test_gradient(self):
        """Test gradient computation."""
        def f(x):
            return x[0] ** 2 + x[1] ** 2
        
        x = np.array([3.0, 4.0])
        grad = self.calc.numerical_gradient(f, x)
        expected = np.array([6.0, 8.0])
        np.testing.assert_array_almost_equal(grad, expected, decimal=4)
    
    def test_hessian(self):
        """Test Hessian computation."""
        def f(x):
            return x[0] ** 2 + 2 * x[0] * x[1] + x[1] ** 2
        
        x = np.array([1.0, 2.0])
        H = self.calc.hessian(f, x)
        expected = np.array([[2.0, 2.0], [2.0, 2.0]])
        np.testing.assert_array_almost_equal(H, expected, decimal=3)
    
    def test_optimization_gradient_descent(self):
        """Test gradient descent optimization."""
        def f(x):
            return (x[0] - 1) ** 2 + (x[1] - 2) ** 2
        
        opt = Optimizer(learning_rate=0.1, method=OptimizationMethod.GRADIENT_DESCENT)
        result = opt.minimize(f, x0=np.array([5.0, 5.0]), max_iterations=100)
        
        self.assertTrue(np.allclose(result.x, np.array([1.0, 2.0]), atol=0.1))
    
    def test_optimization_adam(self):
        """Test Adam optimization."""
        def f(x):
            return x[0] ** 2 + x[1] ** 2
        
        opt = Optimizer(learning_rate=0.1, method=OptimizationMethod.ADAM)
        result = opt.minimize(f, x0=np.array([5.0, 5.0]), max_iterations=100)
        
        self.assertTrue(np.allclose(result.x, np.array([0.0, 0.0]), atol=0.1))
    
    def test_numerical_integration(self):
        """Test numerical integration."""
        def f(x):
            return x ** 2
        
        result = numerical_integration(f, 0, 1, n=100, method='simpson')
        self.assertAlmostEqual(result, 1/3, places=4)
    
    def test_root_finding(self):
        """Test root finding."""
        def f(x):
            return x ** 2 - 2
        
        root = find_root(f, x0=1.5, method='newton')
        self.assertAlmostEqual(root, np.sqrt(2), places=6)


class TestProbabilityOperations(unittest.TestCase):
    """Tests for probability operations."""
    
    def setUp(self):
        self.ops = ProbabilityOperations()
        self.ht = HypothesisTesting()
    
    def test_bayes_theorem(self):
        """Test Bayes' theorem."""
        # Medical test example
        p_positive_given_disease = 0.99
        p_disease = 0.001
        p_positive_given_no_disease = 0.01
        
        result = self.ops.bayes_with_marginal(
            p_positive_given_disease,
            p_disease,
            p_positive_given_no_disease
        )
        
        # P(disease|positive) should be around 0.09
        self.assertAlmostEqual(result, 0.09, places=2)
    
    def test_expectation(self):
        """Test expectation computation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])  # Dice
        probs = np.array([1/6] * 6)
        result = self.ops.expectation(values, probs)
        self.assertAlmostEqual(result, 3.5)
    
    def test_variance(self):
        """Test variance computation."""
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = self.ops.variance(values, ddof=0)
        self.assertAlmostEqual(result, 2.0)
    
    def test_correlation(self):
        """Test correlation computation."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 4.0, 6.0, 8.0, 10.0])
        result = self.ops.correlation(x, y)
        self.assertAlmostEqual(result, 1.0)
    
    def test_kl_divergence(self):
        """Test KL divergence computation."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.4, 0.4, 0.2])
        result = self.ops.kl_divergence(p, q)
        self.assertGreater(result, 0)
    
    def test_entropy(self):
        """Test entropy computation."""
        fair_coin = np.array([0.5, 0.5])
        result = self.ops.entropy(fair_coin, base=2)
        self.assertAlmostEqual(result, 1.0)
    
    def test_normal_distribution(self):
        """Test normal distribution."""
        dist = Distribution.normal(mean=0, std=1)
        samples = dist.sample(10000, seed=42)
        
        self.assertAlmostEqual(np.mean(samples), 0, places=1)
        self.assertAlmostEqual(np.std(samples), 1, places=1)
    
    def test_confidence_interval(self):
        """Test confidence interval computation."""
        np.random.seed(42)
        data = np.random.normal(10, 2, 100)
        ci = self.ops.confidence_interval(data, confidence=0.95)
        
        # True mean should be in CI
        self.assertLess(ci[0], 10)
        self.assertGreater(ci[1], 10)
    
    def test_t_test(self):
        """Test t-test."""
        np.random.seed(42)
        sample1 = np.random.normal(10, 2, 50)
        sample2 = np.random.normal(10, 2, 50)  # Same mean
        
        t_stat, p_value = self.ht.two_sample_t_test(sample1, sample2)
        
        # Should not reject null hypothesis
        self.assertGreater(p_value, 0.05)


if __name__ == '__main__':
    unittest.main()
