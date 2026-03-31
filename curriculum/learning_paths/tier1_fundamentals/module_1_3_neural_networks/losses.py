"""
Neural Network Loss Functions Module.

This module provides comprehensive loss functions for training neural networks,
including MSE, Cross-Entropy, Binary Cross-Entropy, MAE, Huber loss, and more.

Each loss function includes:
- Forward pass (loss computation)
- Backward pass (gradient computation)
- Reduction options (mean, sum, none)

Example Usage:
    >>> import numpy as np
    >>> from losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
    >>> 
    >>> # MSE for regression
    >>> mse = MSELoss()
    >>> y_pred = np.array([[1.0], [2.0], [3.0]])
    >>> y_true = np.array([[1.1], [1.9], [3.1]])
    >>> loss = mse.forward(y_pred, y_true)
    >>> grad = mse.backward()
    >>> 
    >>> # Cross-Entropy for classification
    >>> ce = CrossEntropyLoss()
    >>> logits = np.array([[2.0, 1.0, 0.1], [0.1, 1.0, 2.0]])
    >>> labels = np.array([0, 2])
    >>> loss = ce.forward(logits, labels)
"""

from typing import Union, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike
import logging

logger = logging.getLogger(__name__)

ArrayLike2D = Union[np.ndarray, List]


class LossFunction:
    """Base class for loss functions."""
    
    def forward(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Compute loss value.
        
        Args:
            y_pred: Predicted values.
            y_true: Target values.
        
        Returns:
            float: Loss value.
        """
        raise NotImplementedError
    
    def backward(self) -> np.ndarray:
        """
        Compute gradient with respect to predictions.
        
        Returns:
            np.ndarray: Gradient dL/dy_pred.
        """
        raise NotImplementedError
    
    def __call__(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """Allow calling instance as function."""
        return self.forward(y_pred, y_true)


class MSELoss(LossFunction):
    """
    Mean Squared Error loss for regression.
    
    L = (1/n) * Σ(y_pred - y_true)²
    
    Properties:
    - Range: [0, ∞)
    - Sensitive to outliers
    - Use case: Regression problems
    
    Example:
        >>> mse = MSELoss()
        >>> y_pred = np.array([[1.0], [2.0], [3.0]])
        >>> y_true = np.array([[1.0], [2.0], [3.0]])
        >>> loss = mse.forward(y_pred, y_true)
        >>> loss
        0.0
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize MSELoss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none'). Default: 'mean'.
        """
        self.reduction = reduction
        self._y_pred: Optional[np.ndarray] = None
        self._y_true: Optional[np.ndarray] = None
    
    def forward(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Compute MSE loss.
        
        Args:
            y_pred: Predicted values.
            y_true: Target values.
        
        Returns:
            float: MSE loss.
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}")
        
        self._y_pred = y_pred
        self._y_true = y_true
        
        squared_errors = (y_pred - y_true) ** 2
        
        if self.reduction == 'mean':
            loss = float(np.mean(squared_errors))
        elif self.reduction == 'sum':
            loss = float(np.sum(squared_errors))
        else:  # 'none'
            loss = squared_errors
        
        logger.debug(f"MSE forward: loss={loss}, reduction={self.reduction}")
        return loss
    
    def backward(self) -> np.ndarray:
        """
        Compute gradient: dL/dy_pred = 2 * (y_pred - y_true) / n
        
        Returns:
            np.ndarray: Gradient.
        """
        if self._y_pred is None or self._y_true is None:
            raise ValueError("Must call forward() before backward()")
        
        n = self._y_pred.size
        
        if self.reduction == 'mean':
            grad = 2 * (self._y_pred - self._y_true) / n
        elif self.reduction == 'sum':
            grad = 2 * (self._y_pred - self._y_true)
        else:
            grad = 2 * (self._y_pred - self._y_true)
        
        logger.debug(f"MSE backward: gradient shape {grad.shape}")
        return grad


class MSELossStable(MSELoss):
    """
    Numerically stable MSE loss with optional clipping.
    
    Same as MSELoss but with input validation and clipping.
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        clip_value: Optional[float] = None
    ):
        """
        Initialize stable MSE loss.
        
        Args:
            reduction: Reduction method.
            clip_value: Clip predictions to [-clip_value, clip_value]. None for no clipping.
        """
        super().__init__(reduction)
        self.clip_value = clip_value
    
    def forward(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """Compute MSE with optional clipping."""
        y_pred = np.asarray(y_pred, dtype=np.float64)
        
        if self.clip_value is not None:
            y_pred = np.clip(y_pred, -self.clip_value, self.clip_value)
        
        return super().forward(y_pred, y_true)


class MAELoss(LossFunction):
    """
    Mean Absolute Error (L1) loss for regression.
    
    L = (1/n) * Σ|y_pred - y_true|
    
    Properties:
    - Range: [0, ∞)
    - Less sensitive to outliers than MSE
    - Non-differentiable at 0 (uses subgradient)
    - Use case: Regression with outliers
    
    Example:
        >>> mae = MAELoss()
        >>> y_pred = np.array([[1.0], [2.0], [3.0]])
        >>> y_true = np.array([[1.5], [2.0], [2.5]])
        >>> loss = mae.forward(y_pred, y_true)
        >>> loss
        0.333...
    """
    
    def __init__(self, reduction: str = 'mean'):
        """
        Initialize MAELoss.
        
        Args:
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        self.reduction = reduction
        self._y_pred: Optional[np.ndarray] = None
        self._y_true: Optional[np.ndarray] = None
        self._epsilon = 1e-10
    
    def forward(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Compute MAE loss.
        
        Args:
            y_pred: Predicted values.
            y_true: Target values.
        
        Returns:
            float: MAE loss.
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}")
        
        self._y_pred = y_pred
        self._y_true = y_true
        
        absolute_errors = np.abs(y_pred - y_true)
        
        if self.reduction == 'mean':
            loss = float(np.mean(absolute_errors))
        elif self.reduction == 'sum':
            loss = float(np.sum(absolute_errors))
        else:
            loss = absolute_errors
        
        logger.debug(f"MAE forward: loss={loss}")
        return loss
    
    def backward(self) -> np.ndarray:
        """
        Compute subgradient: sign(y_pred - y_true) / n
        
        Returns:
            np.ndarray: Subgradient.
        """
        if self._y_pred is None or self._y_true is None:
            raise ValueError("Must call forward() before backward()")
        
        diff = self._y_pred - self._y_true
        n = diff.size
        
        # Subgradient of |x| is sign(x), with sign(0) = 0
        sign = np.sign(diff)
        
        if self.reduction == 'mean':
            grad = sign / n
        elif self.reduction == 'sum':
            grad = sign
        else:
            grad = sign
        
        logger.debug(f"MAE backward: gradient shape {grad.shape}")
        return grad


class HuberLoss(LossFunction):
    """
    Huber loss for robust regression.
    
    Combines MSE (for small errors) and MAE (for large errors).
    
    L = 0.5 * (y_pred - y_true)²           if |error| <= δ
    L = δ * (|y_pred - y_true| - 0.5 * δ)  if |error| > δ
    
    Properties:
    - Range: [0, ∞)
    - Less sensitive to outliers than MSE
    - Differentiable everywhere
    - Use case: Robust regression
    
    Example:
        >>> huber = HuberLoss(delta=1.0)
        >>> y_pred = np.array([[1.0], [2.0], [5.0]])  # Last is outlier
        >>> y_true = np.array([[1.0], [2.0], [3.0]])
        >>> loss = huber.forward(y_pred, y_true)
    """
    
    def __init__(self, delta: float = 1.0, reduction: str = 'mean'):
        """
        Initialize HuberLoss.
        
        Args:
            delta: Threshold between MSE and MAE regions. Default: 1.0.
            reduction: Reduction method.
        """
        self.delta = delta
        self.reduction = reduction
        self._y_pred: Optional[np.ndarray] = None
        self._y_true: Optional[np.ndarray] = None
        self._error: Optional[np.ndarray] = None
    
    def forward(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Compute Huber loss.
        
        Args:
            y_pred: Predicted values.
            y_true: Target values.
        
        Returns:
            float: Huber loss.
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch")
        
        self._y_pred = y_pred
        self._y_true = y_true
        self._error = y_pred - y_true
        
        abs_error = np.abs(self._error)
        
        # Quadratic region
        quadratic_mask = abs_error <= self.delta
        quadratic_loss = 0.5 * self._error[quadratic_mask] ** 2
        
        # Linear region
        linear_mask = ~quadratic_mask
        linear_loss = self.delta * (abs_error[linear_mask] - 0.5 * self.delta)
        
        all_losses = np.zeros_like(self._error)
        all_losses[quadratic_mask] = quadratic_loss
        all_losses[linear_mask] = linear_loss
        
        if self.reduction == 'mean':
            loss = float(np.mean(all_losses))
        elif self.reduction == 'sum':
            loss = float(np.sum(all_losses))
        else:
            loss = all_losses
        
        logger.debug(f"Huber forward: loss={loss}, delta={self.delta}")
        return loss
    
    def backward(self) -> np.ndarray:
        """
        Compute gradient.
        
        dL/dy_pred = error           if |error| <= δ
        dL/dy_pred = δ * sign(error) if |error| > δ
        
        Returns:
            np.ndarray: Gradient.
        """
        if self._error is None:
            raise ValueError("Must call forward() before backward()")
        
        abs_error = np.abs(self._error)
        
        grad = np.zeros_like(self._error)
        
        # Quadratic region gradient
        quadratic_mask = abs_error <= self.delta
        grad[quadratic_mask] = self._error[quadratic_mask]
        
        # Linear region gradient
        linear_mask = ~quadratic_mask
        grad[linear_mask] = self.delta * np.sign(self._error[linear_mask])
        
        n = self._error.size
        if self.reduction == 'mean':
            grad = grad / n
        # For 'sum', no scaling needed
        
        logger.debug(f"Huber backward: gradient shape {grad.shape}")
        return grad


class BinaryCrossEntropyLoss(LossFunction):
    """
    Binary Cross-Entropy loss for binary classification.
    
    L = -[y * log(p) + (1-y) * log(1-p)]
    
    Where p = sigmoid(logits) if from_logits=True
    
    Properties:
    - Range: [0, ∞)
    - Use case: Binary classification
    
    Example:
        >>> bce = BinaryCrossEntropyLoss()
        >>> y_pred = np.array([[0.9], [0.1], [0.8]])
        >>> y_true = np.array([[1], [0], [1]])
        >>> loss = bce.forward(y_pred, y_true)
    """
    
    def __init__(
        self,
        from_logits: bool = False,
        reduction: str = 'mean',
        epsilon: float = 1e-15
    ):
        """
        Initialize BCE loss.
        
        Args:
            from_logits: If True, apply sigmoid to predictions. Default: False.
            reduction: Reduction method.
            epsilon: Small value for numerical stability.
        """
        self.from_logits = from_logits
        self.reduction = reduction
        self.epsilon = epsilon
        
        self._y_pred: Optional[np.ndarray] = None
        self._y_true: Optional[np.ndarray] = None
        self._probs: Optional[np.ndarray] = None
    
    def forward(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Compute BCE loss.
        
        Args:
            y_pred: Predicted probabilities or logits.
            y_true: Binary target values (0 or 1).
        
        Returns:
            float: BCE loss.
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch")
        
        self._y_true = y_true
        
        if self.from_logits:
            # Apply sigmoid with numerical stability
            self._probs = self._sigmoid_stable(y_pred)
        else:
            # Clip probabilities
            self._probs = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        self._y_pred = self._probs
        
        # BCE formula
        bce = -(y_true * np.log(self._probs) + (1 - y_true) * np.log(1 - self._probs))
        
        if self.reduction == 'mean':
            loss = float(np.mean(bce))
        elif self.reduction == 'sum':
            loss = float(np.sum(bce))
        else:
            loss = bce
        
        logger.debug(f"BCE forward: loss={loss}")
        return loss
    
    def _sigmoid_stable(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        positive_mask = x >= 0
        negative_mask = ~positive_mask
        
        result = np.zeros_like(x)
        result[positive_mask] = 1 / (1 + np.exp(-x[positive_mask]))
        exp_x = np.exp(x[negative_mask])
        result[negative_mask] = exp_x / (1 + exp_x)
        
        return result
    
    def backward(self) -> np.ndarray:
        """
        Compute gradient.
        
        For probabilities: dL/dp = (p - y) / (p * (1 - p))
        Simplified with sigmoid: dL/dlogits = p - y
        
        Returns:
            np.ndarray: Gradient.
        """
        if self._y_pred is None or self._y_true is None:
            raise ValueError("Must call forward() before backward()")
        
        if self.from_logits:
            # Gradient w.r.t. logits (simplified)
            grad = self._y_pred - self._y_true
        else:
            # Gradient w.r.t. probabilities
            grad = -(self._y_true / (self._y_pred + self.epsilon) - 
                    (1 - self._y_true) / (1 - self._y_pred + self.epsilon))
        
        n = self._y_pred.size
        if self.reduction == 'mean':
            grad = grad / n
        
        logger.debug(f"BCE backward: gradient shape {grad.shape}")
        return grad


class CrossEntropyLoss(LossFunction):
    """
    Cross-Entropy loss for multi-class classification.
    
    L = -Σ y_i * log(p_i)
    
    Where p = softmax(logits) if from_logits=True
    
    Properties:
    - Range: [0, ∞)
    - Use case: Multi-class classification
    
    Example:
        >>> ce = CrossEntropyLoss()
        >>> logits = np.array([[2.0, 1.0, 0.1], [0.1, 1.0, 2.0]])
        >>> labels = np.array([0, 2])  # Class indices
        >>> loss = ce.forward(logits, labels)
    """
    
    def __init__(
        self,
        from_logits: bool = True,
        reduction: str = 'mean',
        epsilon: float = 1e-15,
        axis: int = -1
    ):
        """
        Initialize CrossEntropyLoss.
        
        Args:
            from_logits: If True, apply softmax to predictions. Default: True.
            reduction: Reduction method.
            epsilon: Small value for numerical stability.
            axis: Class axis in predictions.
        """
        self.from_logits = from_logits
        self.reduction = reduction
        self.epsilon = epsilon
        self.axis = axis
        
        self._y_pred: Optional[np.ndarray] = None
        self._y_true: Optional[np.ndarray] = None
        self._probs: Optional[np.ndarray] = None
        self._logits: Optional[np.ndarray] = None
    
    def forward(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Compute Cross-Entropy loss.
        
        Args:
            y_pred: Predicted logits or probabilities.
            y_true: Target class indices (for sparse) or one-hot encoded.
        
        Returns:
            float: Cross-Entropy loss.
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.int32)
        
        self._logits = y_pred
        
        # Handle sparse labels (class indices)
        if y_true.ndim == 1 or (y_true.ndim == 2 and y_true.shape[1] == 1):
            y_true = y_true.flatten()
            n_classes = y_pred.shape[self.axis]
            # Convert to one-hot
            y_true_onehot = np.zeros_like(y_pred)
            if self.axis == -1 or self.axis == y_pred.ndim - 1:
                y_true_onehot[np.arange(len(y_true)), y_true] = 1
            self._y_true = y_true_onehot
            self._sparse_labels = y_true
        else:
            self._y_true = y_true.astype(np.float64)
            self._sparse_labels = None
        
        if self.from_logits:
            # Stable softmax
            self._probs = self._softmax_stable(y_pred)
        else:
            self._probs = np.clip(y_pred, self.epsilon, 1 - self.epsilon)
        
        self._y_pred = self._probs
        
        # Cross-entropy: -Σ y * log(p)
        ce = -np.sum(self._y_true * np.log(self._probs + self.epsilon), axis=self.axis)
        
        if self.reduction == 'mean':
            loss = float(np.mean(ce))
        elif self.reduction == 'sum':
            loss = float(np.sum(ce))
        else:
            loss = ce
        
        logger.debug(f"CrossEntropy forward: loss={loss}")
        return loss
    
    def _softmax_stable(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x_shifted = x - np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
    
    def backward(self) -> np.ndarray:
        """
        Compute gradient.
        
        For softmax + cross-entropy: dL/dlogits = probs - y_onehot
        
        Returns:
            np.ndarray: Gradient.
        """
        if self._y_pred is None or self._y_true is None:
            raise ValueError("Must call forward() before backward()")
        
        # Gradient for softmax + cross-entropy combination
        grad = self._probs - self._y_true
        
        n_samples = grad.shape[0] if grad.ndim > 1 else 1
        if self.reduction == 'mean':
            grad = grad / n_samples
        
        logger.debug(f"CrossEntropy backward: gradient shape {grad.shape}")
        return grad


class HingeLoss(LossFunction):
    """
    Hinge loss for SVM-style classification.
    
    L = max(0, 1 - y * f(x))
    
    Where y ∈ {-1, 1} and f(x) is the raw prediction.
    
    Properties:
    - Range: [0, ∞)
    - Use case: SVM, max-margin classification
    
    Example:
        >>> hinge = HingeLoss()
        >>> y_pred = np.array([[1.5], [-0.5], [2.0]])
        >>> y_true = np.array([[1], [-1], [1]])  # Must be -1 or 1
        >>> loss = hinge.forward(y_pred, y_true)
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        """
        Initialize HingeLoss.
        
        Args:
            margin: Margin parameter. Default: 1.0.
            reduction: Reduction method.
        """
        self.margin = margin
        self.reduction = reduction
        self._y_pred: Optional[np.ndarray] = None
        self._y_true: Optional[np.ndarray] = None
    
    def forward(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Compute Hinge loss.
        
        Args:
            y_pred: Raw predictions (scores).
            y_true: Labels in {-1, 1}.
        
        Returns:
            float: Hinge loss.
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch")
        
        self._y_pred = y_pred
        self._y_true = y_true
        
        # Hinge loss: max(0, margin - y * f(x))
        hinge = np.maximum(0, self.margin - y_true * y_pred)
        
        if self.reduction == 'mean':
            loss = float(np.mean(hinge))
        elif self.reduction == 'sum':
            loss = float(np.sum(hinge))
        else:
            loss = hinge
        
        logger.debug(f"Hinge forward: loss={loss}")
        return loss
    
    def backward(self) -> np.ndarray:
        """
        Compute gradient.
        
        dL/df = -y if y * f < margin, else 0
        
        Returns:
            np.ndarray: Gradient.
        """
        if self._y_pred is None or self._y_true is None:
            raise ValueError("Must call forward() before backward()")
        
        # Violated margin
        violated = self._y_true * self._y_pred < self.margin
        
        n = self._y_pred.size
        if self.reduction == 'mean':
            grad = -self._y_true * violated / n
        elif self.reduction == 'sum':
            grad = -self._y_true * violated
        else:
            grad = -self._y_true * violated
        
        logger.debug(f"Hinge backward: gradient shape {grad.shape}")
        return grad


class KLDivergenceLoss(LossFunction):
    """
    KL Divergence loss for distribution matching.
    
    L = Σ p * log(p / q)
    
    Where p is the target distribution and q is the predicted.
    
    Properties:
    - Range: [0, ∞)
    - Asymmetric: D_KL(P||Q) ≠ D_KL(Q||P)
    - Use case: VAEs, distribution matching
    
    Example:
        >>> kl = KLDivergenceLoss()
        >>> p = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
        >>> q = np.array([[0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])
        >>> loss = kl.forward(q, p)  # Note: q is prediction, p is target
    """
    
    def __init__(self, reduction: str = 'mean', epsilon: float = 1e-15):
        """
        Initialize KLDivergenceLoss.
        
        Args:
            reduction: Reduction method.
            epsilon: Small value for numerical stability.
        """
        self.reduction = reduction
        self.epsilon = epsilon
        
        self._y_pred: Optional[np.ndarray] = None
        self._y_true: Optional[np.ndarray] = None
    
    def forward(
        self,
        y_pred: np.ndarray,
        y_true: np.ndarray
    ) -> float:
        """
        Compute KL Divergence loss.
        
        Args:
            y_pred: Predicted distribution (q).
            y_true: Target distribution (p).
        
        Returns:
            float: KL divergence.
        """
        y_pred = np.asarray(y_pred, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.float64)
        
        if y_pred.shape != y_true.shape:
            raise ValueError(f"Shape mismatch")
        
        # Clip for numerical stability
        y_pred = np.clip(y_pred, self.epsilon, 1)
        y_true = np.clip(y_true, self.epsilon, 1)
        
        # Normalize to ensure valid distributions
        y_pred = y_pred / np.sum(y_pred, axis=-1, keepdims=True)
        y_true = y_true / np.sum(y_true, axis=-1, keepdims=True)
        
        self._y_pred = y_pred
        self._y_true = y_true
        
        # KL divergence: Σ p * log(p / q)
        kl = np.sum(y_true * np.log(y_true / y_pred), axis=-1)
        
        if self.reduction == 'mean':
            loss = float(np.mean(kl))
        elif self.reduction == 'sum':
            loss = float(np.sum(kl))
        else:
            loss = kl
        
        logger.debug(f"KL Divergence forward: loss={loss}")
        return loss
    
    def backward(self) -> np.ndarray:
        """
        Compute gradient.
        
        dL/dq = -p / q
        
        Returns:
            np.ndarray: Gradient.
        """
        if self._y_pred is None or self._y_true is None:
            raise ValueError("Must call forward() before backward()")
        
        n = self._y_pred.size
        if self.reduction == 'mean':
            grad = -self._y_true / (self._y_pred * n)
        elif self.reduction == 'sum':
            grad = -self._y_true / self._y_pred
        else:
            grad = -self._y_true / self._y_pred
        
        logger.debug(f"KL Divergence backward: gradient shape {grad.shape}")
        return grad


class CategoricalCrossEntropyLoss(CrossEntropyLoss):
    """
    Categorical Cross-Entropy for one-hot encoded labels.
    
    Alias for CrossEntropyLoss with from_logits=False.
    """
    
    def __init__(self, reduction: str = 'mean', epsilon: float = 1e-15):
        super().__init__(from_logits=False, reduction=reduction, epsilon=epsilon)


class SparseCategoricalCrossEntropyLoss(CrossEntropyLoss):
    """
    Sparse Categorical Cross-Entropy for integer labels.
    
    Alias for CrossEntropyLoss with from_logits=True.
    """
    
    def __init__(self, reduction: str = 'mean', epsilon: float = 1e-15):
        super().__init__(from_logits=True, reduction=reduction, epsilon=epsilon)


def get_loss(name: str, **kwargs) -> LossFunction:
    """
    Factory function to get loss by name.
    
    Args:
        name: Loss name ('mse', 'cross_entropy', 'bce', etc.).
        **kwargs: Additional arguments.
    
    Returns:
        LossFunction: Loss instance.
    
    Raises:
        ValueError: If name is not recognized.
    
    Example:
        >>> mse = get_loss('mse')
        >>> ce = get_loss('cross_entropy', from_logits=True)
        >>> bce = get_loss('bce', from_logits=False)
    """
    losses = {
        'mse': MSELoss,
        'mean_squared_error': MSELoss,
        'l2': MSELoss,
        'mae': MAELoss,
        'mean_absolute_error': MAELoss,
        'l1': MAELoss,
        'huber': HuberLoss,
        'bce': BinaryCrossEntropyLoss,
        'binary_crossentropy': BinaryCrossEntropyLoss,
        'cross_entropy': CrossEntropyLoss,
        'categorical_crossentropy': CategoricalCrossEntropyLoss,
        'sparse_categorical_crossentropy': SparseCategoricalCrossEntropyLoss,
        'hinge': HingeLoss,
        'kl_divergence': KLDivergenceLoss,
        'kld': KLDivergenceLoss,
    }
    
    name_lower = name.lower()
    if name_lower not in losses:
        raise ValueError(f"Unknown loss: {name}. Available: {list(losses.keys())}")
    
    return losses[name_lower](**kwargs)


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("Loss Functions Module - Demonstration")
    print("=" * 60)
    
    np.random.seed(42)
    
    # MSE Loss
    print("\n1. Mean Squared Error (Regression):")
    mse = MSELoss()
    y_pred_reg = np.array([[1.0], [2.0], [3.0], [4.0]])
    y_true_reg = np.array([[1.1], [1.9], [3.2], [3.8]])
    loss = mse.forward(y_pred_reg, y_true_reg)
    grad = mse.backward()
    print(f"   Predictions: {y_pred_reg.flatten()}")
    print(f"   Targets: {y_true_reg.flatten()}")
    print(f"   Loss: {loss:.6f}")
    print(f"   Gradient: {grad.flatten()}")
    
    # MAE Loss
    print("\n2. Mean Absolute Error (Regression):")
    mae = MAELoss()
    loss = mae.forward(y_pred_reg, y_true_reg)
    print(f"   Loss: {loss:.6f}")
    
    # Huber Loss
    print("\n3. Huber Loss (Robust Regression):")
    huber = HuberLoss(delta=1.0)
    y_pred_outlier = np.array([[1.0], [2.0], [10.0]])  # Outlier
    y_true_outlier = np.array([[1.0], [2.0], [3.0]])
    loss_huber = huber.forward(y_pred_outlier, y_true_outlier)
    loss_mse = mse.forward(y_pred_outlier, y_true_outlier)
    print(f"   With outlier prediction: {y_pred_outlier[2, 0]}")
    print(f"   Huber Loss: {loss_huber:.6f}")
    print(f"   MSE Loss: {loss_mse:.6f} (more sensitive to outlier)")
    
    # Binary Cross-Entropy
    print("\n4. Binary Cross-Entropy (Binary Classification):")
    bce = BinaryCrossEntropyLoss(from_logits=False)
    y_pred_bin = np.array([[0.9], [0.1], [0.8], [0.2]])
    y_true_bin = np.array([[1], [0], [1], [0]])
    loss = bce.forward(y_pred_bin, y_true_bin)
    print(f"   Predictions: {y_pred_bin.flatten()}")
    print(f"   Targets: {y_true_bin.flatten()}")
    print(f"   Loss: {loss:.6f}")
    
    # Cross-Entropy (Multi-class)
    print("\n5. Cross-Entropy (Multi-class Classification):")
    ce = CrossEntropyLoss(from_logits=True)
    logits = np.array([[2.0, 1.0, 0.1], [0.1, 1.0, 2.0], [1.0, 2.0, 0.5]])
    labels = np.array([0, 2, 1])
    loss = ce.forward(logits, labels)
    print(f"   Logits:\n{logits}")
    print(f"   Labels: {labels}")
    print(f"   Loss: {loss:.6f}")
    
    # Hinge Loss
    print("\n6. Hinge Loss (SVM):")
    hinge = HingeLoss()
    y_pred_svm = np.array([[1.5], [-0.5], [2.0], [-1.5]])
    y_true_svm = np.array([[1], [-1], [1], [-1]])
    loss = hinge.forward(y_pred_svm, y_true_svm)
    print(f"   Scores: {y_pred_svm.flatten()}")
    print(f"   Labels: {y_true_svm.flatten()}")
    print(f"   Loss: {loss:.6f}")
    
    # KL Divergence
    print("\n7. KL Divergence (Distribution Matching):")
    kl = KLDivergenceLoss()
    p = np.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
    q = np.array([[0.4, 0.4, 0.2], [0.3, 0.3, 0.4]])
    loss = kl.forward(q, p)
    print(f"   Target (p): {p}")
    print(f"   Predicted (q): {q}")
    print(f"   KL(p||q): {loss:.6f}")
    
    print("\n" + "=" * 60)
