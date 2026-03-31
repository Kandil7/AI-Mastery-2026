"""
Tests for Module 1.2: Python for ML.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from part1_fundamentals.module_1_2_python.data_processing import (
    ArrayOperations,
    DataProcessor,
)
from part1_fundamentals.module_1_2_python.ml_algorithms import (
    LinearRegression,
    LogisticRegression,
    DecisionTreeClassifier,
    RandomForestClassifier,
    KMeans,
    PCA,
)
from part1_fundamentals.module_1_2_python.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OneHotEncoder,
    SimpleImputer,
    train_test_split,
    FeatureSelector,
)


class TestArrayOperations(unittest.TestCase):
    """Tests for array operations."""
    
    def setUp(self):
        self.ops = ArrayOperations()
    
    def test_normalize_l2(self):
        """Test L2 normalization."""
        arr = np.array([[3.0, 4.0], [6.0, 8.0]])
        normalized = self.ops.normalize(arr, axis=1)
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_array_almost_equal(norms, np.array([1.0, 1.0]))
    
    def test_standardize(self):
        """Test standardization."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        standardized = self.ops.standardize(arr)
        
        np.testing.assert_array_almost_equal(np.mean(standardized, axis=0), 0, decimal=10)
        np.testing.assert_array_almost_equal(np.std(standardized, axis=0), 1, decimal=10)
    
    def test_min_max_scale(self):
        """Test min-max scaling."""
        arr = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        scaled = self.ops.min_max_scale(arr, feature_range=(0, 1))
        
        np.testing.assert_array_almost_equal(scaled.min(), 0)
        np.testing.assert_array_almost_equal(scaled.max(), 1)
    
    def test_one_hot_encode(self):
        """Test one-hot encoding."""
        labels = np.array([0, 1, 2, 1, 0])
        one_hot = self.ops.one_hot_encode(labels)
        
        self.assertEqual(one_hot.shape, (5, 3))
        np.testing.assert_array_equal(one_hot.sum(axis=1), 1)
    
    def test_rolling_window(self):
        """Test rolling window creation."""
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        windows = self.ops.rolling_window(arr, window_size=3)
        
        expected = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
        np.testing.assert_array_equal(windows, expected)


class TestDataProcessor(unittest.TestCase):
    """Tests for DataFrame processing."""
    
    def setUp(self):
        self.processor = DataProcessor()
        self.df = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': ['a', 'b', 'c', 'd', 'e'],
            'C': [1.0, 2.0, np.nan, 4.0, 5.0]
        })
    
    def test_handle_missing_mean(self):
        """Test missing value handling with mean."""
        cleaned = self.processor.handle_missing(self.df, strategy='mean', columns=['C'])
        self.assertFalse(cleaned['C'].isnull().any())
    
    def test_handle_missing_drop(self):
        """Test missing value handling with drop."""
        cleaned = self.processor.handle_missing(self.df, strategy='drop')
        self.assertEqual(len(cleaned), 4)
    
    def test_group_by(self):
        """Test group by aggregation."""
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1, 2, 3, 4]
        })
        result = self.processor.group_by(df, 'group', {'value': 'mean'})
        
        self.assertAlmostEqual(result.loc['A', 'value'], 1.5)
        self.assertAlmostEqual(result.loc['B', 'value'], 3.5)
    
    def test_get_dummies(self):
        """Test one-hot encoding."""
        df = pd.DataFrame({'category': ['A', 'B', 'A', 'C']})
        encoded = self.processor.get_dummies(df, columns=['category'])
        
        self.assertEqual(encoded.shape[1], 3)
    
    def test_train_test_split(self):
        """Test train/test split."""
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        self.assertEqual(len(X_train), 80)
        self.assertEqual(len(X_test), 20)


class TestMLAlgorithms(unittest.TestCase):
    """Tests for ML algorithms."""
    
    def test_linear_regression(self):
        """Test linear regression."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = 2 * X[:, 0] + 3 * X[:, 1] - 1.5 * X[:, 2] + np.random.randn(100) * 0.1
        
        model = LinearRegression(learning_rate=0.1, n_iterations=1000)
        model.fit(X, y)
        
        self.assertGreater(model.score(X, y), 0.9)
    
    def test_logistic_regression(self):
        """Test logistic regression."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = LogisticRegression(learning_rate=0.1, n_iterations=1000)
        model.fit(X, y)
        
        self.assertGreater(model.score(X, y), 0.85)
    
    def test_decision_tree(self):
        """Test decision tree classifier."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X, y)
        
        self.assertGreater(model.score(X, y), 0.85)
    
    def test_random_forest(self):
        """Test random forest classifier."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        model = RandomForestClassifier(n_estimators=10, max_depth=5, random_state=42)
        model.fit(X, y)
        
        self.assertGreater(model.score(X, y), 0.85)
    
    def test_kmeans(self):
        """Test K-means clustering."""
        np.random.seed(42)
        X = np.vstack([
            np.random.randn(100, 2) + [2, 2],
            np.random.randn(100, 2) + [-2, -2],
        ])
        
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(X)
        
        self.assertEqual(len(np.unique(labels)), 2)
        self.assertLess(kmeans.inertia_, 1000)
    
    def test_pca(self):
        """Test PCA."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        
        self.assertEqual(X_reduced.shape, (100, 2))
        self.assertGreater(np.sum(pca.explained_variance_ratio_), 0.1)


class TestPreprocessing(unittest.TestCase):
    """Tests for preprocessing utilities."""
    
    def test_standard_scaler(self):
        """Test standard scaler."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        np.testing.assert_array_almost_equal(np.mean(X_scaled, axis=0), 0, decimal=10)
        np.testing.assert_array_almost_equal(np.std(X_scaled, axis=0), 1, decimal=10)
    
    def test_min_max_scaler(self):
        """Test min-max scaler."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_scaled = scaler.fit_transform(X)
        
        np.testing.assert_array_almost_equal(X_scaled.min(), 0)
        np.testing.assert_array_almost_equal(X_scaled.max(), 1)
    
    def test_label_encoder(self):
        """Test label encoder."""
        labels = ['cat', 'dog', 'cat', 'bird', 'dog']
        
        encoder = LabelEncoder()
        encoded = encoder.fit_transform(labels)
        
        self.assertEqual(len(np.unique(encoded)), 3)
        
        # Test inverse transform
        decoded = encoder.inverse_transform(encoded)
        np.testing.assert_array_equal(decoded, labels)
    
    def test_one_hot_encoder(self):
        """Test one-hot encoder."""
        X = [['cat'], ['dog'], ['cat'], ['bird']]
        
        encoder = OneHotEncoder()
        encoded = encoder.fit_transform(X)
        
        self.assertEqual(encoded.shape[1], 3)
        np.testing.assert_array_equal(encoded.sum(axis=1), 1)
    
    def test_simple_imputer(self):
        """Test simple imputer."""
        X = np.array([[1, 2], [np.nan, 3], [7, np.nan], [4, 5]])
        
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X)
        
        self.assertFalse(np.isnan(X_imputed).any())
    
    def test_feature_selector_variance(self):
        """Test feature selection by variance."""
        X = np.array([[1, 0, 1], [2, 0, 2], [3, 0, 3]])
        
        selector = FeatureSelector()
        mask = selector.variance_threshold(X, threshold=0.1)
        
        # Second column has zero variance
        self.assertFalse(mask[1])


if __name__ == '__main__':
    unittest.main()
