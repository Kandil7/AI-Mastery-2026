"""
Linear Regression Implementation

This module implements linear regression from scratch using NumPy,
including both closed-form solution and gradient descent approaches.
"""

import numpy as np
from typing import Tuple, Optional, Union
from src.core.optimization import minimize
import matplotlib.pyplot as plt


class LinearRegression:
    """
    Linear Regression implementation from scratch.
    
    This class implements linear regression with options for:
    - Closed-form solution (Normal Equation)
    - Gradient descent optimization
    - L1/L2 regularization
    """
    
    def __init__(
        self, 
        method: str = 'normal', 
        learning_rate: float = 0.01, 
        n_iterations: int = 1000,
        regularization: Optional[str] = None,
        lambda_reg: float = 0.01
    ):
        """
        Initialize Linear Regression model.
        
        Args:
            method: 'normal' for closed-form solution, 'gd' for gradient descent
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations for gradient descent
            regularization: 'l1', 'l2', or None
            lambda_reg: Regularization strength
        """
        self.method = method
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Add bias term to feature matrix."""
        return np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearRegression':
        """
        Fit the linear regression model.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
            
        Returns:
            Self (for method chaining)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        if self.method == 'normal':
            # Closed-form solution: theta = (X^T * X)^(-1) * X^T * y
            # Add bias term
            X_with_bias = self._add_bias(X)
            
            # Regularization term
            reg_term = np.zeros((X_with_bias.shape[1], X_with_bias.shape[1]))
            if self.regularization == 'l2':
                # Ridge regression: add lambda * I to (X^T * X)
                reg_term[1:, 1:] = self.lambda_reg * np.eye(X_with_bias.shape[1] - 1)
            elif self.regularization == 'l1':
                # Lasso: we'll use gradient descent for L1
                self.method = 'gd'
        
        if self.method == 'normal' and self.regularization != 'l1':
            # Calculate weights using normal equation
            XtX_inv = np.linalg.inv(X_with_bias.T @ X_with_bias + reg_term)
            weights = XtX_inv @ X_with_bias.T @ y
            self.bias = weights[0]
            self.weights = weights[1:]
        elif self.method == 'gd' or self.regularization == 'l1':
            # Use gradient descent
            n_features = X.shape[1]
            self.weights = np.random.normal(0, 0.01, n_features)
            self.bias = 0.0
            
            for i in range(self.n_iterations):
                # Forward pass
                y_pred = self.predict(X)
                
                # Calculate gradients
                dw, db = self._compute_gradients(X, y, y_pred)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
                
                # Calculate and store cost
                cost = self._compute_cost(y, y_pred)
                self.cost_history.append(cost)
        
        return self
    
    def _compute_gradients(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        y_pred: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Compute gradients for weights and bias."""
        m = X.shape[0]
        error = y_pred - y
        
        # Gradient for weights
        dw = (1/m) * X.T @ error
        
        # Add regularization term
        if self.regularization == 'l2':
            dw += self.lambda_reg * self.weights
        elif self.regularization == 'l1':
            # Subgradient for L1: sign of weights
            dw += self.lambda_reg * np.sign(self.weights)
        
        # Gradient for bias
        db = (1/m) * np.sum(error)
        
        return dw, db
    
    def _compute_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean squared error cost."""
        m = y_true.shape[0]
        mse = (1/(2*m)) * np.sum((y_true - y_pred) ** 2)
        
        # Add regularization term
        if self.regularization == 'l2':
            mse += (self.lambda_reg / (2*m)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            mse += (self.lambda_reg / m) * np.sum(np.abs(self.weights))
        
        return mse
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predictions of shape (n_samples,)
        """
        X = np.asarray(X)
        return X @ self.weights + self.bias
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R-squared score.
        
        Args:
            X: Features of shape (n_samples, n_features)
            y: True targets of shape (n_samples,)
            
        Returns:
            R-squared score
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot)


class RidgeRegression(LinearRegression):
    """
    Ridge Regression (L2 regularization) implementation.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, lambda_reg: float = 0.01):
        super().__init__(
            method='gd',
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            regularization='l2',
            lambda_reg=lambda_reg
        )


class LassoRegression(LinearRegression):
    """
    Lasso Regression (L1 regularization) implementation.
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000, lambda_reg: float = 0.01):
        super().__init__(
            method='gd',
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            regularization='l1',
            lambda_reg=lambda_reg
        )


def polynomial_features(X: np.ndarray, degree: int) -> np.ndarray:
    """
    Generate polynomial features up to specified degree.
    
    Args:
        X: Input features of shape (n_samples, n_features)
        degree: Maximum polynomial degree
        
    Returns:
        Polynomial features of shape (n_samples, n_poly_features)
    """
    X = np.asarray(X)
    n_samples, n_features = X.shape
    
    # For now, implement for single feature
    if n_features == 1:
        poly_features = np.zeros((n_samples, degree + 1))
        for i in range(degree + 1):
            poly_features[:, i] = X[:, 0] ** i
        return poly_features
    else:
        raise NotImplementedError("Polynomial features for multiple features not implemented yet")


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean squared error."""
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate root mean squared error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate mean absolute error."""
    return np.mean(np.abs(y_true - y_pred))