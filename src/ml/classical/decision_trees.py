"""
Decision Trees Implementation

This module implements decision trees from scratch using NumPy,
including both classification and regression trees.
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Any
from collections import Counter
import matplotlib.pyplot as plt


class Node:
    """
    Node class for decision tree.
    """
    
    def __init__(self):
        self.feature_index = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None  # For leaf nodes
        self.is_leaf = False


class DecisionTree:
    """
    Base Decision Tree class.
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, str]] = None
    ):
        """
        Initialize Decision Tree.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required in a leaf node
            max_features: Number of features to consider when looking for best split
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.root = None
    
    def _gini_impurity(self, y: np.ndarray) -> float:
        """
        Calculate Gini impurity for a set of labels.
        
        Args:
            y: Array of labels
            
        Returns:
            Gini impurity value
        """
        if len(y) == 0:
            return 0.0
        
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        gini = 1.0 - np.sum(probabilities ** 2)
        return gini
    
    def _entropy(self, y: np.ndarray) -> float:
        """
        Calculate entropy for a set of labels.
        
        Args:
            y: Array of labels
            
        Returns:
            Entropy value
        """
        if len(y) == 0:
            return 0.0
        
        class_counts = np.bincount(y)
        probabilities = class_counts / len(y)
        # Avoid log(0) by using where condition
        entropy = -np.sum(np.where(probabilities > 0, probabilities * np.log2(probabilities), 0))
        return entropy
    
    def _calculate_information_gain(
        self, 
        y: np.ndarray, 
        left_y: np.ndarray, 
        right_y: np.ndarray
    ) -> float:
        """
        Calculate information gain for a split.
        
        Args:
            y: Original labels
            left_y: Left split labels
            right_y: Right split labels
            
        Returns:
            Information gain value
        """
        parent_impurity = self._impurity_function(y)
        
        n = len(y)
        n_left, n_right = len(left_y), len(right_y)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        weighted_impurity = (n_left / n) * self._impurity_function(left_y) + \
                           (n_right / n) * self._impurity_function(right_y)
        
        return parent_impurity - weighted_impurity
    
    def _impurity_function(self, y: np.ndarray) -> float:
        """
        Impurity function to use (to be overridden by subclasses).
        """
        raise NotImplementedError
    
    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best split for the current node.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Tuple of (best_feature_index, best_threshold, best_gain)
        """
        n_samples, n_features = X.shape
        best_gain = -1
        best_feature_idx = None
        best_threshold = None
        
        # Determine number of features to consider
        if self.max_features is None:
            feature_indices = range(n_features)
        elif isinstance(self.max_features, int):
            feature_indices = np.random.choice(n_features, self.max_features, replace=False)
        elif self.max_features == 'sqrt':
            n_sqrt = int(np.sqrt(n_features))
            feature_indices = np.random.choice(n_features, n_sqrt, replace=False)
        elif self.max_features == 'log2':
            n_log2 = int(np.log2(n_features))
            feature_indices = np.random.choice(n_features, n_log2, replace=False)
        else:
            raise ValueError(f"Invalid max_features value: {self.max_features}")
        
        for feature_idx in feature_indices:
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                left_y = y[left_mask]
                right_y = y[right_mask]
                
                gain = self._calculate_information_gain(y, left_y, right_y)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
        
        return best_feature_idx, best_threshold, best_gain
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """
        Recursively build the decision tree.
        
        Args:
            X: Feature matrix
            y: Target values
            depth: Current depth of the tree
            
        Returns:
            Root node of the subtree
        """
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        node = Node()
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
            node.is_leaf = True
            node.value = self._leaf_value(y)
            return node
        
        # Find best split
        best_feature_idx, best_threshold, best_gain = self._best_split(X, y)
        
        if best_feature_idx is None or best_gain <= 0:
            node.is_leaf = True
            node.value = self._leaf_value(y)
            return node
        
        # Split the data
        left_mask = X[:, best_feature_idx] <= best_threshold
        right_mask = ~left_mask
        
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            node.is_leaf = True
            node.value = self._leaf_value(y)
            return node
        
        # Create child nodes
        node.feature_index = best_feature_idx
        node.threshold = best_threshold
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return node
    
    def _leaf_value(self, y: np.ndarray) -> Any:
        """
        Calculate the value for a leaf node (to be overridden by subclasses).
        """
        raise NotImplementedError
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """
        Fit the decision tree to the training data.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self (for method chaining)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.root = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x: np.ndarray, node: Node) -> Any:
        """
        Predict a single sample.
        
        Args:
            x: Single sample features
            node: Current node in the tree
            
        Returns:
            Predicted value
        """
        if node.is_leaf:
            return node.value
        
        if x[node.feature_index] <= node.threshold:
            return self._predict_sample(x, node.left)
        else:
            return self._predict_sample(x, node.right)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples in X.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted target values
        """
        X = np.asarray(X)
        predictions = np.array([self._predict_sample(x, self.root) for x in X])
        return predictions


class DecisionTreeClassifier(DecisionTree):
    """
    Decision Tree Classifier implementation.
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, str]] = None,
        criterion: str = 'gini'
    ):
        """
        Initialize Decision Tree Classifier.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required in a leaf node
            max_features: Number of features to consider when looking for best split
            criterion: 'gini' or 'entropy'
        """
        super().__init__(max_depth, min_samples_split, min_samples_leaf, max_features)
        self.criterion = criterion
    
    def _impurity_function(self, y: np.ndarray) -> float:
        """
        Impurity function based on criterion.
        """
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Invalid criterion: {self.criterion}")
    
    def _leaf_value(self, y: np.ndarray) -> int:
        """
        Calculate the value for a leaf node (most common class).
        """
        # Return the most common class in y
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class probabilities
        """
        X = np.asarray(X)
        probas = []
        
        for x in X:
            # For simplicity, we'll use the leaf's class distribution
            # In practice, you might want to store class counts in each leaf
            node = self.root
            while not node.is_leaf:
                if x[node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            
            # Since we only store the predicted class in the leaf,
            # we'll return a one-hot encoded probability
            proba = np.zeros(len(np.unique(self._get_all_labels())))
            proba[node.value] = 1.0
            probas.append(proba)
        
        return np.array(probas)
    
    def _get_all_labels(self) -> np.ndarray:
        """
        Helper method to get all possible labels (used for predict_proba).
        """
        # This is a simplified implementation
        # In a more complex implementation, you'd store this during training
        pass


class DecisionTreeRegressor(DecisionTree):
    """
    Decision Tree Regressor implementation.
    """
    
    def __init__(
        self,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, str]] = None,
        criterion: str = 'mse'
    ):
        """
        Initialize Decision Tree Regressor.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required in a leaf node
            max_features: Number of features to consider when looking for best split
            criterion: 'mse' for mean squared error
        """
        super().__init__(max_depth, min_samples_split, min_samples_leaf, max_features)
        self.criterion = criterion
    
    def _impurity_function(self, y: np.ndarray) -> float:
        """
        Impurity function for regression (variance).
        """
        if self.criterion == 'mse':
            if len(y) <= 1:
                return 0.0
            return np.var(y)
        else:
            raise ValueError(f"Invalid criterion: {self.criterion}")
    
    def _leaf_value(self, y: np.ndarray) -> float:
        """
        Calculate the value for a leaf node (mean of values).
        """
        return np.mean(y)
    
    def _calculate_information_gain(
        self, 
        y: np.ndarray, 
        left_y: np.ndarray, 
        right_y: np.ndarray
    ) -> float:
        """
        Calculate information gain for regression (variance reduction).
        """
        parent_var = self._impurity_function(y)
        
        n = len(y)
        n_left, n_right = len(left_y), len(right_y)
        
        if n_left == 0 or n_right == 0:
            return 0.0
        
        weighted_var = (n_left / n) * self._impurity_function(left_y) + \
                      (n_right / n) * self._impurity_function(right_y)
        
        return parent_var - weighted_var


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy score."""
    return np.mean(y_true == y_pred)


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate R-squared score."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)