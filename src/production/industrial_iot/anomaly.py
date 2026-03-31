"""
Anomaly Detection Module
=========================

Anomaly detection for industrial IoT systems.

Implements:
- Autoencoder: minimize ||x - D(E(x))||²
- Isolation Forest: anomaly score based on path length
- Ensemble: Combined detection

Classes:
    Autoencoder: Simple autoencoder for anomaly detection
    IsolationForest: Isolation Forest for anomaly detection
    AnomalyDetector: Ensemble anomaly detector

Author: AI-Mastery-2026
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class Autoencoder:
    """
    Simple autoencoder for anomaly detection.

    Architecture:
    - Encoder: Input -> Hidden1 -> Latent
    - Decoder: Latent -> Hidden2 -> Output

    Anomaly detection:
    - Train on normal data
    - High reconstruction error = anomaly

    Mathematical foundation:
    - minimize ||x - D(E(x))||² over normal data
    - anomaly score = reconstruction error
    """

    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 32):
        """
        Initialize autoencoder.

        Args:
            input_dim: Input dimension
            latent_dim: Bottleneck dimension
            hidden_dim: Hidden layer dimension
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Initialize weights
        self._init_weights()

        # Threshold for anomaly detection
        self.threshold = None

        logger.info(
            f"Autoencoder initialized: {input_dim} -> {latent_dim} -> {input_dim}"
        )

    def _init_weights(self):
        """He initialization for weights."""
        # Encoder
        self.W_enc1 = np.random.randn(self.input_dim, self.hidden_dim) * np.sqrt(
            2.0 / self.input_dim
        )
        self.b_enc1 = np.zeros(self.hidden_dim)

        self.W_enc2 = np.random.randn(self.hidden_dim, self.latent_dim) * np.sqrt(
            2.0 / self.hidden_dim
        )
        self.b_enc2 = np.zeros(self.latent_dim)

        # Decoder
        self.W_dec1 = np.random.randn(self.latent_dim, self.hidden_dim) * np.sqrt(
            2.0 / self.latent_dim
        )
        self.b_dec1 = np.zeros(self.hidden_dim)

        self.W_dec2 = np.random.randn(self.hidden_dim, self.input_dim) * np.sqrt(
            2.0 / self.hidden_dim
        )
        self.b_dec2 = np.zeros(self.input_dim)

    def _relu(self, x):
        """ReLU activation."""
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        """ReLU derivative."""
        return (x > 0).astype(float)

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Encode input to latent space."""
        h1 = self._relu(x @ self.W_enc1 + self.b_enc1)
        z = h1 @ self.W_enc2 + self.b_enc2
        return z

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode from latent space."""
        h1 = self._relu(z @ self.W_dec1 + self.b_dec1)
        x_hat = h1 @ self.W_dec2 + self.b_dec2
        return x_hat

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Full forward pass."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat

    def reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error (anomaly score).

        MSE per sample.
        """
        x_hat = self.forward(x)
        error = np.mean((x - x_hat) ** 2, axis=1)
        return error

    def fit(
        self,
        X: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.001,
        batch_size: int = 32,
    ):
        """
        Train autoencoder on normal data.

        Args:
            X: Training data (n_samples, n_features)
            epochs: Number of training epochs
            learning_rate: SGD learning rate
            batch_size: Mini-batch size
        """
        n_samples = X.shape[0]

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]

            epoch_loss = 0.0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuffled[start:end]

                # Forward pass
                h1_enc = self._relu(X_batch @ self.W_enc1 + self.b_enc1)
                z = h1_enc @ self.W_enc2 + self.b_enc2
                h1_dec = self._relu(z @ self.W_dec1 + self.b_dec1)
                x_hat = h1_dec @ self.W_dec2 + self.b_dec2

                # Compute loss
                loss = np.mean((X_batch - x_hat) ** 2)
                epoch_loss += loss * len(X_batch)

                # Backpropagation
                d_loss = 2 * (x_hat - X_batch) / len(X_batch)

                # Decoder gradients
                d_W_dec2 = h1_dec.T @ d_loss
                d_b_dec2 = d_loss.sum(axis=0)

                d_h1_dec = (
                    d_loss
                    @ self.W_dec2.T
                    * self._relu_derivative(z @ self.W_dec1 + self.b_dec1)
                )
                d_W_dec1 = z.T @ d_h1_dec
                d_b_dec1 = d_h1_dec.sum(axis=0)

                # Encoder gradients
                d_z = d_h1_dec @ self.W_dec1.T
                d_W_enc2 = h1_enc.T @ d_z
                d_b_enc2 = d_z.sum(axis=0)

                d_h1_enc = (
                    d_z
                    @ self.W_enc2.T
                    * self._relu_derivative(X_batch @ self.W_enc1 + self.b_enc1)
                )
                d_W_enc1 = X_batch.T @ d_h1_enc
                d_b_enc1 = d_h1_enc.sum(axis=0)

                # Update weights
                self.W_dec2 -= learning_rate * d_W_dec2
                self.b_dec2 -= learning_rate * d_b_dec2
                self.W_dec1 -= learning_rate * d_W_dec1
                self.b_dec1 -= learning_rate * d_b_dec1
                self.W_enc2 -= learning_rate * d_W_enc2
                self.b_enc2 -= learning_rate * d_b_enc2
                self.W_enc1 -= learning_rate * d_W_enc1
                self.b_enc1 -= learning_rate * d_b_enc1

            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}: loss={epoch_loss / n_samples:.6f}")

        # Set threshold based on training data
        train_errors = self.reconstruction_error(X)
        self.threshold = np.percentile(train_errors, 95)
        logger.info(f"Training complete. Threshold set to {self.threshold:.4f}")

    def detect(self, x: np.ndarray) -> Tuple[bool, float]:
        """
        Detect if input is anomalous.

        Args:
            x: Input sample(s)

        Returns:
            (is_anomaly, anomaly_score)
        """
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        error = self.reconstruction_error(x)[0]
        is_anomaly = error > self.threshold if self.threshold else False

        return is_anomaly, error


class IsolationForest:
    """
    Isolation Forest for anomaly detection.

    Key insight: Anomalies are easier to isolate
    (fewer random splits needed).

    Anomaly score = average path length across trees
    Shorter path = more anomalous
    """

    def __init__(
        self, n_trees: int = 100, sample_size: int = 256, contamination: float = 0.1
    ):
        """
        Initialize Isolation Forest.

        Args:
            n_trees: Number of isolation trees
            sample_size: Subsample size for each tree
            contamination: Expected proportion of anomalies
        """
        self.n_trees = n_trees
        self.sample_size = sample_size
        self.contamination = contamination

        self.trees: List[Dict] = []
        self.threshold: Optional[float] = None

        logger.info(f"Isolation Forest initialized: {n_trees} trees")

    def _build_tree(self, X: np.ndarray, depth: int = 0, max_depth: int = 10) -> Dict:
        """Build a single isolation tree recursively."""
        n_samples, n_features = X.shape

        # Termination conditions
        if depth >= max_depth or n_samples <= 1:
            return {"type": "leaf", "size": n_samples}

        # Random split
        feature_idx = np.random.randint(n_features)
        feature_values = X[:, feature_idx]

        min_val, max_val = feature_values.min(), feature_values.max()

        if min_val == max_val:
            return {"type": "leaf", "size": n_samples}

        split_value = np.random.uniform(min_val, max_val)

        # Split data
        left_mask = feature_values < split_value
        right_mask = ~left_mask

        return {
            "type": "node",
            "feature": feature_idx,
            "split": split_value,
            "left": self._build_tree(X[left_mask], depth + 1, max_depth),
            "right": self._build_tree(X[right_mask], depth + 1, max_depth),
        }

    def _path_length(self, x: np.ndarray, tree: Dict, depth: int = 0) -> float:
        """Compute path length for a single sample."""
        if tree["type"] == "leaf":
            c = self._c(tree["size"])
            return depth + c

        if x[tree["feature"]] < tree["split"]:
            return self._path_length(x, tree["left"], depth + 1)
        else:
            return self._path_length(x, tree["right"], depth + 1)

    def _c(self, n: int) -> float:
        """Average path length of unsuccessful search in BST."""
        if n <= 1:
            return 0
        return 2 * (np.log(n - 1) + 0.5772156649) - 2 * (n - 1) / n

    def fit(self, X: np.ndarray):
        """Build forest on training data."""
        n_samples = X.shape[0]

        for _ in range(self.n_trees):
            sample_indices = np.random.choice(
                n_samples, min(self.sample_size, n_samples), replace=False
            )
            X_sample = X[sample_indices]

            max_depth = int(np.ceil(np.log2(self.sample_size)))
            tree = self._build_tree(X_sample, max_depth=max_depth)
            self.trees.append(tree)

        scores = self.anomaly_score(X)
        self.threshold = np.percentile(scores, 100 * (1 - self.contamination))

        logger.info(f"Forest trained. Threshold: {self.threshold:.4f}")

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores."""
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        n_samples = X.shape[0]
        scores = np.zeros(n_samples)

        for i in range(n_samples):
            avg_path = np.mean([self._path_length(X[i], tree) for tree in self.trees])
            c_n = self._c(self.sample_size)
            scores[i] = 2 ** (-avg_path / c_n) if c_n > 0 else 0.5

        return scores

    def detect(self, x: np.ndarray) -> Tuple[bool, float]:
        """Detect if sample is anomalous."""
        score = self.anomaly_score(x)[0]
        is_anomaly = score > self.threshold if self.threshold else False
        return is_anomaly, score


class AnomalyDetector:
    """
    Ensemble anomaly detector combining multiple methods.

    Uses both Autoencoder and Isolation Forest for robustness.
    Final score is weighted combination.
    """

    def __init__(self, input_dim: int, ae_weight: float = 0.5, if_weight: float = 0.5):
        """
        Initialize ensemble detector.

        Args:
            input_dim: Feature dimension
            ae_weight: Weight for autoencoder score
            if_weight: Weight for isolation forest score
        """
        self.input_dim = input_dim
        self.ae_weight = ae_weight
        self.if_weight = if_weight

        self.autoencoder = Autoencoder(input_dim, latent_dim=max(8, input_dim // 8))
        self.isolation_forest = IsolationForest(n_trees=50)

        self.threshold = 0.5
        self.trained = False

        logger.info(f"Ensemble anomaly detector initialized: dim={input_dim}")

    def fit(self, X_normal: np.ndarray, epochs: int = 50):
        """Train on normal data."""
        logger.info("Training anomaly detector...")

        self.autoencoder.fit(X_normal, epochs=epochs)
        self.isolation_forest.fit(X_normal)

        ae_scores = self.autoencoder.reconstruction_error(X_normal)
        ae_scores_norm = ae_scores / (ae_scores.max() + 1e-10)

        if_scores = self.isolation_forest.anomaly_score(X_normal)

        combined = self.ae_weight * ae_scores_norm + self.if_weight * if_scores
        self.threshold = np.percentile(combined, 95)

        self.trained = True
        logger.info(f"Ensemble trained. Threshold: {self.threshold:.4f}")

    def detect(self, x: np.ndarray) -> Tuple[bool, float, Dict[str, float]]:
        """
        Detect anomaly using ensemble.

        Returns:
            (is_anomaly, combined_score, component_scores)
        """
        _, ae_score = self.autoencoder.detect(x)
        _, if_score = self.isolation_forest.detect(x)

        ae_norm = min(ae_score / (self.autoencoder.threshold + 1e-10), 2.0)

        combined = self.ae_weight * ae_norm + self.if_weight * if_score
        is_anomaly = combined > self.threshold

        return (
            is_anomaly,
            combined,
            {
                "autoencoder": ae_score,
                "isolation_forest": if_score,
                "combined": combined,
            },
        )
