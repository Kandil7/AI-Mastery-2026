"""
Logistic Regression Implementation

This module implements logistic regression from scratch using NumPy,
including both binary and multiclass classification.
"""

import numpy as np
from typing import Tuple, Optional, Union
from src.ml.classical.linear_regression import mean_squared_error
import matplotlib.pyplot as plt


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Sigmoid activation function.
    
    Args:
        z: Input values
        
    Returns:
        Sigmoid-transformed values
    """
    # Clip z to prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax activation function for multiclass classification.
    
    Args:
        z: Input values of shape (n_samples, n_classes)
        
    Returns:
        Softmax-transformed values
    """
    # Subtract max for numerical stability
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


class LogisticRegression:
    """
    Logistic Regression implementation from scratch.
    
    This class implements logistic regression for binary classification
    with options for regularization and different optimization methods.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: Optional[str] = None,
        lambda_reg: float = 0.01,
        tolerance: float = 1e-6
    ):
        """
        Initialize Logistic Regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations for gradient descent
            regularization: 'l1', 'l2', or None
            lambda_reg: Regularization strength
            tolerance: Convergence tolerance
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def _add_bias(self, X: np.ndarray) -> np.ndarray:
        """Add bias term to feature matrix."""
        return np.c_[np.ones(X.shape[0]), X]
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        """
        Fit the logistic regression model.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,) with binary values {0, 1}
            
        Returns:
            Self (for method chaining)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Initialize parameters
        n_features = X.shape[1]
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0.0
        
        prev_cost = float('inf')
        
        for i in range(self.n_iterations):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred_proba = sigmoid(z)
            
            # Calculate cost
            cost = self._compute_cost(y, y_pred_proba)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y, y_pred_proba)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Check for convergence
            if abs(prev_cost - cost) < self.tolerance:
                print(f"Converged after {i+1} iterations")
                break
            prev_cost = cost
        
        return self
    
    def _compute_cost(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute logistic regression cost (log-likelihood)."""
        # Prevent log(0) by clipping probabilities
        y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
        
        # Binary cross-entropy loss
        cost = -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))
        
        # Add regularization term
        if self.regularization == 'l2':
            cost += (self.lambda_reg / 2) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            cost += self.lambda_reg * np.sum(np.abs(self.weights))
        
        return cost
    
    def _compute_gradients(
        self, 
        X: np.ndarray, 
        y_true: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Compute gradients for weights and bias."""
        m = X.shape[0]
        
        # Gradient for weights
        dw = (1/m) * X.T @ (y_pred_proba - y_true)
        
        # Add regularization term
        if self.regularization == 'l2':
            dw += self.lambda_reg * self.weights
        elif self.regularization == 'l1':
            # Subgradient for L1: sign of weights
            dw += self.lambda_reg * np.sign(self.weights)
        
        # Gradient for bias
        db = (1/m) * np.sum(y_pred_proba - y_true)
        
        return dw, db
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities of shape (n_samples,)
        """
        X = np.asarray(X)
        z = X @ self.weights + self.bias
        return sigmoid(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predicted classes of shape (n_samples,) with values {0, 1}
        """
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Features of shape (n_samples, n_features)
            y: True targets of shape (n_samples,) with binary values {0, 1}
            
        Returns:
            Accuracy score
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class MultinomialLogisticRegression:
    """
    Multinomial Logistic Regression (Softmax Regression) implementation.
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        regularization: Optional[str] = None,
        lambda_reg: float = 0.01
    ):
        """
        Initialize Multinomial Logistic Regression model.
        
        Args:
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations for gradient descent
            regularization: 'l1', 'l2', or None
            lambda_reg: Regularization strength
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None  # Shape: (n_features, n_classes)
        self.biases = None   # Shape: (n_classes,)
        self.cost_history = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'MultinomialLogisticRegression':
        """
        Fit the multinomial logistic regression model.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,) with class labels {0, 1, ..., n_classes-1}
            
        Returns:
            Self (for method chaining)
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples, n_features = X.shape
        self.n_classes = len(np.unique(y))
        
        # Convert y to one-hot encoding
        y_onehot = np.eye(self.n_classes)[y]
        
        # Initialize parameters
        self.weights = np.random.normal(0, 0.01, (n_features, self.n_classes))
        self.biases = np.zeros(self.n_classes)
        
        for i in range(self.n_iterations):
            # Forward pass
            z = X @ self.weights + self.biases  # Broadcasting for bias
            y_pred_proba = softmax(z)
            
            # Calculate cost
            cost = self._compute_cost(y_onehot, y_pred_proba)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw, db = self._compute_gradients(X, y_onehot, y_pred_proba)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.biases -= self.learning_rate * db
        
        return self
    
    def _compute_cost(self, y_true_onehot: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Compute cross-entropy cost for multiclass classification."""
        # Prevent log(0) by clipping probabilities
        y_pred_proba = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
        
        # Cross-entropy loss
        cost = -np.mean(np.sum(y_true_onehot * np.log(y_pred_proba), axis=1))
        
        # Add regularization term
        if self.regularization == 'l2':
            cost += (self.lambda_reg / 2) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            cost += self.lambda_reg * np.sum(np.abs(self.weights))
        
        return cost
    
    def _compute_gradients(
        self, 
        X: np.ndarray, 
        y_true_onehot: np.ndarray, 
        y_pred_proba: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradients for weights and biases."""
        m = X.shape[0]
        
        # Gradient for weights
        dw = (1/m) * X.T @ (y_pred_proba - y_true_onehot)
        
        # Add regularization term
        if self.regularization == 'l2':
            dw += self.lambda_reg * self.weights
        elif self.regularization == 'l1':
            dw += self.lambda_reg * np.sign(self.weights)
        
        # Gradient for biases
        db = (1/m) * np.sum(y_pred_proba - y_true_onehot, axis=0)
        
        return dw, db
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predicted probabilities of shape (n_samples, n_classes)
        """
        X = np.asarray(X)
        z = X @ self.weights + self.biases
        return softmax(z)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make class predictions.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predicted classes of shape (n_samples,) with values {0, 1, ..., n_classes-1}
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Features of shape (n_samples, n_features)
            y: True targets of shape (n_samples,) with class labels {0, 1, ..., n_classes-1}
            
        Returns:
            Accuracy score
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate accuracy score."""
    return np.mean(y_true == y_pred)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """Calculate precision score for binary classification."""
    true_positives = np.sum((y_true == pos_label) & (y_pred == pos_label))
    predicted_positives = np.sum(y_pred == pos_label)
    
    if predicted_positives == 0:
        return 0.0
    
    return true_positives / predicted_positives


def recall_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """Calculate recall score for binary classification."""
    true_positives = np.sum((y_true == pos_label) & (y_pred == pos_label))
    actual_positives = np.sum(y_true == pos_label)
    
    if actual_positives == 0:
        return 0.0
    
    return true_positives / actual_positives


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, pos_label: int = 1) -> float:
    """Calculate F1 score for binary classification."""
    prec = precision_score(y_true, y_pred, pos_label)
    rec = recall_score(y_true, y_pred, pos_label)
    
    if prec + rec == 0:
        return 0.0
    
    return 2 * (prec * rec) / (prec + rec)


def log_loss(y_true: np.ndarray, y_pred_proba: np.ndarray, eps: float = 1e-15) -> float:
    """
    Calculate log loss (cross-entropy) for binary classification.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        eps: Small value to prevent log(0)
        
    Returns:
        Log loss value
    """
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred_proba) + (1 - y_true) * np.log(1 - y_pred_proba))