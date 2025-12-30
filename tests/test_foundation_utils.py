"""
Unit Tests for Foundation Utilities
"""

import unittest
import numpy as np
from src.foundation_utils import (
    dot, magnitude, cosine_similarity, normalize, matrix_multiply,
    mean, variance, std, covariance, correlation,
    gradient_descent,
    linear_regression_closed_form, linear_regression_gradient,
    mse_loss, cross_entropy_loss, sigmoid, relu, softmax
)

class TestFoundationUtils(unittest.TestCase):
    
    # --- Linear Algebra ---
    def test_dot_product(self):
        v1, v2 = [1, 2, 3], [4, 5, 6]
        expected = 1*4 + 2*5 + 3*6
        self.assertEqual(dot(v1, v2), expected)

    def test_magnitude(self):
        v = [3, 4]
        self.assertEqual(magnitude(v), 5.0)

    def test_cosine_similarity(self):
        v1, v2 = [1, 0], [0, 1]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 0.0)
        v1, v2 = [1, 1], [1, 1]
        self.assertAlmostEqual(cosine_similarity(v1, v2), 1.0, places=4)

    def test_matrix_multiply(self):
        A = [[1, 2], [3, 4]]
        B = [[2, 0], [1, 2]]
        # [[1*2+2*1, 1*0+2*2], [3*2+4*1, 3*0+4*2]] = [[4, 4], [10, 8]]
        C = matrix_multiply(A, B)
        self.assertEqual(C, [[4, 4], [10, 8]])

    # --- Statistics ---
    def test_statistics(self):
        arr = [1, 2, 3, 4, 5]
        self.assertEqual(mean(arr), 3.0)
        self.assertEqual(variance(arr), 2.0) # Population variance of [1,2,3,4,5] is 2
        self.assertAlmostEqual(std(arr), np.sqrt(2.0))

    def test_correlation(self):
        x = [1, 2, 3]
        y = [2, 4, 6] # Perfect positive correlation
        self.assertAlmostEqual(correlation(x, y), 1.0)
        
        y_neg = [-1, -2, -3] # Perfect negative correlation
        self.assertAlmostEqual(correlation(x, y_neg), -1.0)

    # --- Optimization ---
    def test_gradient_descent(self):
        # Minimize f(x) = x^2, f'(x) = 2x
        f_prime = lambda x: 2*x
        min_x = gradient_descent(f_prime, lr=0.1, start=10.0, steps=100)
        self.assertAlmostEqual(min_x, 0.0, places=4)

    # --- Regression ---
    def test_linear_regression(self):
        # y = 2x + 1
        X = np.array([[1, 1], [1, 2], [1, 3]]) # Added bias term column
        y = np.array([3, 5, 7])
        
        w = linear_regression_closed_form(X, y)
        self.assertAlmostEqual(w[0], 1.0) # Bias
        self.assertAlmostEqual(w[1], 2.0) # Slope

    # --- Activations ---
    def test_activations(self):
        val = np.array([0.0])
        self.assertEqual(sigmoid(val)[0], 0.5)
        
        val = np.array([-5.0, 5.0])
        self.assertEqual(relu(val)[0], 0.0)
        self.assertEqual(relu(val)[1], 5.0)
        
        val = np.array([1.0, 1.0])
        sm = softmax(val)
        self.assertAlmostEqual(sm[0], 0.5)
        self.assertAlmostEqual(sm[1], 0.5)

if __name__ == '__main__':
    unittest.main()
