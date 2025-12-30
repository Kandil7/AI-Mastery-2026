"""
Ensemble Methods Implementation

This module implements ensemble methods from scratch using NumPy,
including Random Forest, AdaBoost, and Gradient Boosting.
"""

import numpy as np
from typing import Union, List, Tuple, Optional, Any
from src.ml.classical.decision_trees import DecisionTreeClassifier, DecisionTreeRegressor
from src.ml.classical.linear_regression import LinearRegression
import matplotlib.pyplot as plt


class RandomForest:
    """
    Random Forest implementation from scratch.
    
    This class implements Random Forest for both classification and regression
    using bootstrap aggregating (bagging) and random feature selection.
    """
    
    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, str]] = 'sqrt',
        bootstrap: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize Random Forest.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required in a leaf node
            max_features: Number of features to consider when looking for best split
            bootstrap: Whether to use bootstrap samples
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.trees = []
        self.is_fitted = False
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a bootstrap sample of the data.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Bootstrap sample of X and y
        """
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForest':
        """
        Fit the Random Forest model.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self (for method chaining)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.trees = []
        
        for i in range(self.n_estimators):
            # Create a decision tree with random feature selection
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            
            # Bootstrap sample
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y
            
            # Fit the tree
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples in X.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted target values
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        X = np.asarray(X)
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Take majority vote for classification
        predictions = []
        for i in range(X.shape[0]):
            # Get all predictions for this sample
            sample_predictions = tree_predictions[:, i]
            # Take majority vote
            unique, counts = np.unique(sample_predictions, return_counts=True)
            predictions.append(unique[np.argmax(counts)])
        
        return np.array(predictions)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for samples in X.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        X = np.asarray(X)
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Calculate probabilities based on votes
        n_samples = X.shape[0]
        n_classes = len(np.unique(np.concatenate([tree.predict(X) for tree in self.trees])))
        
        probas = np.zeros((n_samples, n_classes))
        
        for i in range(n_samples):
            sample_predictions = tree_predictions[:, i]
            unique, counts = np.unique(sample_predictions, return_counts=True)
            
            for j, cls in enumerate(unique):
                probas[i, cls] = counts[j] / len(sample_predictions)
        
        return probas


class RandomForestRegressor(RandomForest):
    """
    Random Forest Regressor implementation.
    """
    
    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, str]] = 'sqrt',
        bootstrap: bool = True,
        random_state: Optional[int] = None
    ):
        """
        Initialize Random Forest Regressor.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of the trees
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required in a leaf node
            max_features: Number of features to consider when looking for best split
            bootstrap: Whether to use bootstrap samples
            random_state: Random state for reproducibility
        """
        super().__init__(
            n_estimators, max_depth, min_samples_split, 
            min_samples_leaf, max_features, bootstrap, random_state
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RandomForestRegressor':
        """
        Fit the Random Forest Regressor model.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self (for method chaining)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        self.trees = []
        
        for i in range(self.n_estimators):
            # Create a decision tree regressor with random feature selection
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features
            )
            
            # Bootstrap sample
            if self.bootstrap:
                X_sample, y_sample = self._bootstrap_sample(X, y)
            else:
                X_sample, y_sample = X, y
            
            # Fit the tree
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict target values for samples in X.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted target values
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        X = np.asarray(X)
        
        # Get predictions from all trees
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])
        
        # Take average for regression
        predictions = np.mean(tree_predictions, axis=0)
        
        return predictions


class AdaBoostClassifier:
    """
    AdaBoost Classifier implementation from scratch.
    """
    
    def __init__(
        self,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: Optional[int] = None
    ):
        """
        Initialize AdaBoost Classifier.
        
        Args:
            n_estimators: Number of weak learners
            learning_rate: Learning rate shrinks the contribution of each classifier
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.estimators = []
        self.estimator_weights = []
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdaBoostClassifier':
        """
        Fit the AdaBoost model.
        
        Args:
            X: Training features
            y: Training targets with values {-1, +1}
            
        Returns:
            Self (for method chaining)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples = X.shape[0]
        
        # Initialize weights uniformly
        sample_weights = np.full(n_samples, (1 / n_samples))
        
        self.estimators = []
        self.estimator_weights = []
        
        for i in range(self.n_estimators):
            # Train weak classifier
            estimator = DecisionTreeClassifier(max_depth=1)  # Decision stump
            estimator.fit(X, y, sample_weights=sample_weights)
            
            # Make predictions
            predictions = estimator.predict(X)
            
            # Calculate error
            error = np.sum(sample_weights[predictions != y])
            
            # Calculate estimator weight
            estimator_weight = self.learning_rate * 0.5 * np.log((1 - error) / (error + 1e-10))
            
            # Update sample weights
            sample_weights *= np.exp(-estimator_weight * y * predictions)
            sample_weights /= np.sum(sample_weights)  # Normalize
            
            # Store estimator and its weight
            self.estimators.append(estimator)
            self.estimator_weights.append(estimator_weight)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict classes for samples in X.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted classes
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        X = np.asarray(X)
        
        # Calculate weighted predictions from all estimators
        weighted_predictions = np.zeros(X.shape[0])
        
        for estimator, weight in zip(self.estimators, self.estimator_weights):
            predictions = estimator.predict(X)
            weighted_predictions += weight * predictions
        
        # Return sign of weighted sum
        return np.sign(weighted_predictions).astype(int)


class GradientBoostingRegressor:
    """
    Gradient Boosting Regressor implementation from scratch.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
        loss: str = 'squared_error',
        random_state: Optional[int] = None
    ):
        """
        Initialize Gradient Boosting Regressor.
        
        Args:
            n_estimators: Number of boosting stages
            learning_rate: Shrinks the contribution of each tree
            max_depth: Maximum depth of individual regression estimators
            min_samples_split: Minimum samples required to split a node
            loss: Loss function to optimize ('squared_error', 'absolute_error')
            random_state: Random state for reproducibility
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.loss = loss
        self.random_state = random_state
        
        if random_state is not None:
            np.random.seed(random_state)
        
        self.estimators = []
        self.is_fitted = False
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GradientBoostingRegressor':
        """
        Fit the Gradient Boosting model.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Self (for method chaining)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Initialize predictions with mean of y
        self.initial_prediction = np.mean(y)
        current_predictions = np.full(X.shape[0], self.initial_prediction)
        
        self.estimators = []
        
        for i in range(self.n_estimators):
            # Calculate pseudo-residuals (negative gradient)
            residuals = y - current_predictions
            
            # Fit a tree to the residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split
            )
            tree.fit(X, residuals)
            
            # Update predictions
            tree_predictions = tree.predict(X)
            current_predictions += self.learning_rate * tree_predictions
            
            # Store the tree
            self.estimators.append(tree)
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict values for samples in X.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError("Model has not been fitted yet.")
        
        X = np.asarray(X)
        
        # Start with initial prediction
        predictions = np.full(X.shape[0], self.initial_prediction)
        
        # Add contributions from all trees
        for tree in self.estimators:
            predictions += self.learning_rate * tree.predict(X)
        
        return predictions


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


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))