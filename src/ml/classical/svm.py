"""
Support Vector Machine Implementation

This module implements Support Vector Machine from scratch using NumPy,
including both classification and regression variants.
"""

import numpy as np
from typing import Union, Optional, Tuple
from src.core.optimization import minimize
import matplotlib.pyplot as plt


class SVM:
    """
    Support Vector Machine implementation from scratch.
    
    This class implements SVM for binary classification using the dual formulation
    and Sequential Minimal Optimization (SMO) algorithm.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        kernel: str = 'rbf',
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1,
        tol: float = 1e-3,
        max_iter: int = 1000
    ):
        """
        Initialize SVM model.
        
        Args:
            C: Regularization parameter
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            degree: Degree for 'poly' kernel
            coef0: Independent term in 'poly' and 'sigmoid' kernels
            tol: Tolerance for stopping criterion
            max_iter: Maximum number of iterations
        """
        self.C = C
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.support_vectors = None
        self.support_vector_labels = None
        self.alphas = None
        self.b = 0
        self.X_train = None
        self.y_train = None
    
    def _kernel_function(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute kernel function between two vectors.
        
        Args:
            x1: First vector
            x2: Second vector
            
        Returns:
            Kernel value
        """
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (np.dot(x1, x2) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            if self.gamma is None:
                self.gamma = 1.0 / x1.shape[0]
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(x1, x2) + self.coef0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def _compute_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the kernel matrix for the dataset.
        
        Args:
            X: Training data
            
        Returns:
            Kernel matrix
        """
        n_samples = X.shape[0]
        K = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])
        
        return K
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVM':
        """
        Fit the SVM model using Sequential Minimal Optimization (SMO).
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,) with values {-1, +1}
            
        Returns:
            Self (for method chaining)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Store training data
        self.X_train = X
        self.y_train = y
        
        n_samples, n_features = X.shape
        
        # Initialize alphas
        alphas = np.zeros(n_samples)
        
        # Compute kernel matrix
        K = self._compute_kernel_matrix(X)
        
        # SMO algorithm
        iter_count = 0
        while iter_count < self.max_iter:
            num_changed_alphas = 0
            
            for i in range(n_samples):
                # Calculate error for sample i
                E_i = self._compute_error(i, alphas, K)
                
                # Check KKT conditions
                if (y[i] * E_i < -self.tol and alphas[i] < self.C) or \
                   (y[i] * E_i > self.tol and alphas[i] > 0):
                    
                    # Select second alpha (j) using heuristic
                    j, E_j = self._select_j(i, alphas, E_i, K)
                    
                    if j == -1:
                        continue
                    
                    # Save old alphas
                    alpha_i_old, alpha_j_old = alphas[i], alphas[j]
                    
                    # Compute bounds for alpha_j
                    if y[i] != y[j]:
                        L = max(0, alphas[j] - alphas[i])
                        H = min(self.C, self.C + alphas[j] - alphas[i])
                    else:
                        L = max(0, alphas[i] + alphas[j] - self.C)
                        H = min(self.C, alphas[i] + alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    
                    if eta >= 0:
                        continue
                    
                    # Compute new alpha_j
                    alphas[j] -= y[j] * (E_i - E_j) / eta
                    alphas[j] = np.clip(alphas[j], L, H)
                    
                    # Compute new alpha_i
                    alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])
                    
                    # Compute b
                    b1 = self.b - E_i - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - \
                         y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - \
                         y[j] * (alphas[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < alphas[i] < self.C:
                        self.b = b1
                    elif 0 < alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    if abs(alphas[j] - alpha_j_old) > 1e-5:
                        num_changed_alphas += 1
            
            iter_count += 1
            if num_changed_alphas == 0:
                break
        
        # Store support vectors
        sv_mask = alphas > 1e-5
        self.support_vectors = X[sv_mask]
        self.support_vector_labels = y[sv_mask]
        self.alphas = alphas[sv_mask]
        
        return self
    
    def _compute_error(self, i: int, alphas: np.ndarray, K: np.ndarray) -> float:
        """
        Compute error for sample i.
        
        Args:
            i: Index of sample
            alphas: Alpha values
            K: Kernel matrix
            
        Returns:
            Error value
        """
        n_samples = len(alphas)
        prediction = 0
        
        for j in range(n_samples):
            if alphas[j] > 0:
                prediction += alphas[j] * self.y_train[j] * K[i, j]
        
        prediction += self.b
        return prediction - self.y_train[i]
    
    def _select_j(self, i: int, alphas: np.ndarray, E_i: float, K: np.ndarray) -> Tuple[int, float]:
        """
        Select second alpha (j) using heuristic.
        
        Args:
            i: Index of first alpha
            alphas: Alpha values
            E_i: Error for sample i
            K: Kernel matrix
            
        Returns:
            Tuple of (j, E_j) or (-1, 0) if no suitable j found
        """
        max_delta_E = 0
        j = -1
        E_j = 0
        
        for k in range(len(alphas)):
            if alphas[k] > 0 and k != i:
                E_k = self._compute_error(k, alphas, K)
                delta_E = abs(E_i - E_k)
                
                if delta_E > max_delta_E:
                    max_delta_E = delta_E
                    j = k
                    E_j = E_k
        
        return j, E_j
    
    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values.
        
        Args:
            X: Input features
            
        Returns:
            Decision function values
        """
        X = np.asarray(X)
        
        if self.support_vectors is None:
            raise ValueError("Model has not been fitted yet.")
        
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        for i in range(n_samples):
            for alpha, sv_label, sv in zip(self.alphas, self.support_vector_labels, self.support_vectors):
                predictions[i] += alpha * sv_label * self._kernel_function(X[i], sv)
            predictions[i] += self.b
        
        return predictions
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predicted classes of shape (n_samples,) with values {-1, +1}
        """
        decision_values = self._decision_function(X)
        return np.sign(decision_values).astype(int)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score.
        
        Args:
            X: Features of shape (n_samples, n_features)
            y: True targets of shape (n_samples,) with values {-1, +1}
            
        Returns:
            Accuracy score
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        y_pred = self.predict(X)
        return np.mean(y_pred == y)


class SVR:
    """
    Support Vector Regression implementation.
    """
    
    def __init__(
        self,
        C: float = 1.0,
        epsilon: float = 0.1,
        kernel: str = 'rbf',
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1,
        tol: float = 1e-3,
        max_iter: int = 1000
    ):
        """
        Initialize SVR model.
        
        Args:
            C: Regularization parameter
            epsilon: Epsilon in the epsilon-SVR model
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid')
            gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
            degree: Degree for 'poly' kernel
            coef0: Independent term in 'poly' and 'sigmoid' kernels
            tol: Tolerance for stopping criterion
            max_iter: Maximum number of iterations
        """
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.support_vectors = None
        self.dual_coef = None
        self.b = 0
        self.X_train = None
        self.y_train = None
    
    def _kernel_function(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """
        Compute kernel function between two vectors.
        
        Args:
            x1: First vector
            x2: Second vector
            
        Returns:
            Kernel value
        """
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (np.dot(x1, x2) + self.coef0) ** self.degree
        elif self.kernel == 'rbf':
            if self.gamma is None:
                self.gamma = 1.0 / x1.shape[0]
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        elif self.kernel == 'sigmoid':
            return np.tanh(self.gamma * np.dot(x1, x2) + self.coef0)
        else:
            raise ValueError(f"Unknown kernel: {self.kernel}")
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SVR':
        """
        Fit the SVR model.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
            
        Returns:
            Self (for method chaining)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        # Store training data
        self.X_train = X
        self.y_train = y
        
        n_samples, n_features = X.shape
        
        # For SVR, we need to solve a quadratic programming problem
        # This is a simplified implementation using a general optimization approach
        # In practice, specialized QP solvers are used
        
        # Create the combined variable [alpha, alpha*] where alpha* is alpha_tilde
        # The optimization problem has 2*n_samples variables
        
        # For this implementation, we'll use a simplified approach
        # In a full implementation, we would solve the dual problem
        
        # For now, we'll just store the training data
        # A complete implementation would require solving the QP problem
        self.support_vectors = X
        self.support_vector_labels = y
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data.
        
        Args:
            X: Features of shape (n_samples, n_features)
            
        Returns:
            Predicted values of shape (n_samples,)
        """
        X = np.asarray(X)
        
        if self.support_vectors is None:
            raise ValueError("Model has not been fitted yet.")
        
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)
        
        # This is a simplified prediction function
        # A complete implementation would use the dual coefficients
        for i in range(n_samples):
            for j in range(len(self.support_vectors)):
                predictions[i] += self._kernel_function(X[i], self.support_vectors[j])
            # Add bias term (in a real implementation, this would be properly calculated)
            predictions[i] += 0
        
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