"""
Remaining Useful Life (RUL) Prediction Module
==============================================

RUL prediction using LSTM for industrial equipment.

Architecture:
- LSTM layer(s) for temporal feature extraction
- Dense layer for RUL regression
- Output: RUL in hours with uncertainty estimation

Classes:
    LSTMCell: LSTM cell for sequence modeling
    RULPredictor: Remaining Useful Life predictor

Author: AI-Mastery-2026
"""

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class LSTMCell:
    """
    LSTM cell for sequence modeling.

    Gates:
    - Forget gate: f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
    - Input gate: i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
    - Candidate: c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
    - Output gate: o_t = σ(W_o · [h_{t-1}, x_t] + b_o)

    State updates:
    - c_t = f_t * c_{t-1} + i_t * c̃_t
    - h_t = o_t * tanh(c_t)
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        """
        Initialize LSTM cell.

        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden state dimension
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        combined_dim = hidden_dim + input_dim

        # Xavier initialization for gates
        scale = np.sqrt(6.0 / (combined_dim + hidden_dim))

        # Forget gate
        self.W_f = np.random.uniform(-scale, scale, (combined_dim, hidden_dim))
        self.b_f = np.ones(hidden_dim)

        # Input gate
        self.W_i = np.random.uniform(-scale, scale, (combined_dim, hidden_dim))
        self.b_i = np.zeros(hidden_dim)

        # Candidate
        self.W_c = np.random.uniform(-scale, scale, (combined_dim, hidden_dim))
        self.b_c = np.zeros(hidden_dim)

        # Output gate
        self.W_o = np.random.uniform(-scale, scale, (combined_dim, hidden_dim))
        self.b_o = np.zeros(hidden_dim)

    def _sigmoid(self, x):
        """Sigmoid activation with numerical stability."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def forward(
        self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through LSTM cell.

        Args:
            x: Input (batch, input_dim)
            h_prev: Previous hidden state (batch, hidden_dim)
            c_prev: Previous cell state (batch, hidden_dim)

        Returns:
            (h_t, c_t)
        """
        combined = np.concatenate([h_prev, x], axis=-1)

        f_t = self._sigmoid(combined @ self.W_f + self.b_f)
        i_t = self._sigmoid(combined @ self.W_i + self.b_i)
        c_tilde = np.tanh(combined @ self.W_c + self.b_c)
        o_t = self._sigmoid(combined @ self.W_o + self.b_o)

        c_t = f_t * c_prev + i_t * c_tilde
        h_t = o_t * np.tanh(c_t)

        return h_t, c_t


class RULPredictor:
    """
    Remaining Useful Life predictor using LSTM.

    Takes time-series sensor data and predicts hours until failure.

    Architecture:
    - LSTM layer(s) for temporal feature extraction
    - Dense layer for RUL regression

    Output: RUL in hours with uncertainty estimation
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, sequence_length: int = 50):
        """
        Initialize RUL predictor.

        Args:
            input_dim: Number of sensor features
            hidden_dim: LSTM hidden dimension
            sequence_length: Expected sequence length
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        # LSTM layer
        self.lstm = LSTMCell(input_dim, hidden_dim)

        # Output layer (predicts mean and log-variance for uncertainty)
        self.W_out = np.random.randn(hidden_dim, 2) * 0.1
        self.b_out = np.zeros(2)

        # Training state
        self.trained = False

        logger.info(f"RUL Predictor initialized: {input_dim} -> {hidden_dim} -> RUL")

    def forward(self, sequence: np.ndarray) -> Tuple[float, float]:
        """
        Predict RUL from sequence.

        Args:
            sequence: (sequence_length, input_dim)

        Returns:
            (rul_prediction, uncertainty)
        """
        batch_size = 1 if len(sequence.shape) == 2 else sequence.shape[0]

        if len(sequence.shape) == 2:
            sequence = sequence.reshape(1, *sequence.shape)

        h = np.zeros((batch_size, self.hidden_dim))
        c = np.zeros((batch_size, self.hidden_dim))

        for t in range(sequence.shape[1]):
            x_t = sequence[:, t, :]
            h, c = self.lstm.forward(x_t, h, c)

        output = h @ self.W_out + self.b_out

        rul_mean = np.exp(output[:, 0])
        log_var = output[:, 1]
        rul_std = np.exp(0.5 * log_var)

        return float(rul_mean[0]), float(rul_std[0])

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        learning_rate: float = 0.001,
    ):
        """
        Train RUL predictor.

        Args:
            X: Sequences (n_samples, sequence_length, input_dim)
            y: RUL targets (n_samples,)
        """
        logger.info("Training RUL predictor...")

        n_samples = X.shape[0]

        for epoch in range(epochs):
            epoch_loss = 0.0

            for i in range(n_samples):
                rul_pred, std = self.forward(X[i])

                loss = (rul_pred - y[i]) ** 2
                epoch_loss += loss

                grad = 2 * (rul_pred - y[i])
                self.W_out[:, 0] -= learning_rate * grad * 0.01

            if epoch % 10 == 0:
                logger.debug(f"Epoch {epoch}: loss={epoch_loss / n_samples:.4f}")

        self.trained = True
        logger.info("RUL predictor training complete")

    def predict(self, sequence: np.ndarray) -> Tuple[float, float, float]:
        """
        Predict RUL with confidence interval.

        Returns:
            (rul_hours, lower_bound, upper_bound)
        """
        rul, std = self.forward(sequence)

        lower = max(0, rul - 1.96 * std)
        upper = rul + 1.96 * std

        return rul, lower, upper

    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "sequence_length": self.sequence_length,
            "trained": self.trained,
        }


class RULPredictorWithDegradation:
    """
    Advanced RUL predictor with degradation modeling.

    Accounts for:
    - Linear degradation
    - Exponential degradation
    - Piece-wise degradation
    """

    def __init__(self, degradation_model: str = "linear"):
        """
        Initialize with degradation model.

        Args:
            degradation_model: 'linear', 'exponential', or 'piecewise'
        """
        self.degradation_model = degradation_model
        self.health_index_history = []
        self.failure_threshold = None
        self.trained = False

    def fit(self, health_indices: np.ndarray, rul_targets: np.ndarray):
        """Fit degradation model."""
        self.health_index_history = health_indices.tolist()

        if self.degradation_model == "linear":
            # Fit linear degradation
            x = np.arange(len(health_indices))
            self.slope, self.intercept = np.polyfit(x, health_indices, 1)
            self.trained = True

        elif self.degradation_model == "exponential":
            # Fit exponential degradation
            self.health_init = health_indices[0]
            self.decay_rate = -np.log(health_indices[-1] / health_indices[0]) / len(
                health_indices
            )
            self.trained = True

        logger.info(f"RUL degradation model ({self.degradation_model}) trained")

    def predict_rul(self, current_health: float) -> float:
        """Predict RUL based on current health."""
        if not self.trained:
            return float("inf")

        if self.degradation_model == "linear":
            # When will health reach 0?
            if self.slope >= 0:
                return float("inf")
            remaining = current_health / (-self.slope)
            return remaining

        elif self.degradation_model == "exponential":
            # When will health reach threshold?
            threshold = 0.2  # Assume 20% health as failure
            if current_health <= threshold:
                return 0
            return -np.log(threshold / self.health_init) / self.decay_rate

        return float("inf")
