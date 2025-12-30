"""
Unit Tests for Classical ML Modules
"""

import unittest
import numpy as np
from src.ml.classical import LinearRegression, LogisticRegression, DecisionTree

class TestClassicalML(unittest.TestCase):
    
    def test_linear_regression(self):
        # f(x) = 2x + 1
        X = np.array([[1], [2], [3], [4]])
        y = np.array([3, 5, 7, 9])
        
        model = LinearRegression()
        model.fit(X, y, epochs=1000, lr=0.1)
        
        preds = model.predict(np.array([[5]]))
        self.assertAlmostEqual(preds[0], 11.0, delta=0.5)
        
    def test_logistic_regression(self):
        # AND gate
        X = np.array([[0,0], [0,1], [1,0], [1,1]])
        y = np.array([0, 0, 0, 1])
        
        model = LogisticRegression()
        model.fit(X, y, epochs=1000, lr=1.0)
        
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, y)
        
    def test_decision_tree(self):
        # Simple separation
        X = np.array([[1], [2], [10], [11]])
        y = np.array([0, 0, 1, 1])
        
        model = DecisionTree(max_depth=2)
        model.fit(X, y)
        
        preds = model.predict(X)
        np.testing.assert_array_equal(preds, y)

if __name__ == '__main__':
    unittest.main()
