"""
Embedding Drift Detection Module
=================================

Detect drift in embedding space (Arize/Fiddler pattern).

Monitors changes in embedding distributions over time to detect:
- Concept drift (semantics of queries changing)
- Data drift (input distribution changing)
- Model staleness (embeddings becoming less relevant)

Classes:
    EmbeddingDriftDetector: Monitor embedding distribution changes

Author: AI-Mastery-2026
"""

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class EmbeddingDriftDetector:
    """
    Detect drift in embedding space (Arize/Fiddler pattern).

    Monitors changes in embedding distributions over time to detect:
    - Concept drift (semantics of queries changing)
    - Data drift (input distribution changing)
    - Model staleness (embeddings becoming less relevant)

    Key Features:
    - Track embedding centroid over time
    - Compute distribution distances (cosine, euclidean)
    - Statistical tests for drift significance
    - UMAP visualization support

    Reference: Arize AI, Fiddler ML Observability
    """

    def __init__(
        self, embedding_dim: int, window_size: int = 1000, drift_threshold: float = 0.1
    ):
        """
        Initialize drift detector.

        Args:
            embedding_dim: Dimension of embeddings
            window_size: Number of embeddings to track per window
            drift_threshold: Threshold for drift alerts
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.drift_threshold = drift_threshold

        # Reference distribution (baseline)
        self._reference_embeddings: List[np.ndarray] = []
        self._reference_centroid: Optional[np.ndarray] = None
        self._reference_std: Optional[np.ndarray] = None

        # Current distribution
        self._current_embeddings: List[np.ndarray] = []
        self._current_centroid: Optional[np.ndarray] = None

        # Drift history
        self._drift_history: List[Dict[str, Any]] = []
        self._alerts: List[Dict[str, Any]] = []

        self._lock = threading.Lock()

    def set_reference(self, embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """
        Set the reference/baseline distribution.

        Args:
            embeddings: List of reference embeddings

        Returns:
            Reference statistics
        """
        with self._lock:
            self._reference_embeddings = embeddings[-self.window_size :]
            arr = np.array(self._reference_embeddings)
            self._reference_centroid = np.mean(arr, axis=0)
            self._reference_std = np.std(arr, axis=0)

        return {
            "reference_size": len(self._reference_embeddings),
            "centroid_norm": float(np.linalg.norm(self._reference_centroid)),
            "avg_std": float(np.mean(self._reference_std)),
        }

    def add_embedding(self, embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Add a new embedding and check for drift.

        Args:
            embedding: New embedding vector

        Returns:
            Drift alert if threshold exceeded, else None
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(
                f"Expected dim {self.embedding_dim}, got {embedding.shape[0]}"
            )

        with self._lock:
            self._current_embeddings.append(embedding)

            # Keep only window_size embeddings
            if len(self._current_embeddings) > self.window_size:
                self._current_embeddings = self._current_embeddings[-self.window_size :]

            # Update current centroid
            arr = np.array(self._current_embeddings)
            self._current_centroid = np.mean(arr, axis=0)

        # Check for drift if we have enough samples
        if len(self._current_embeddings) >= self.window_size // 2:
            return self._check_drift()

        return None

    def _check_drift(self) -> Optional[Dict[str, Any]]:
        """Check for drift between reference and current distributions."""
        if self._reference_centroid is None or self._current_centroid is None:
            return None

        # Compute drift metrics
        drift_metrics = self.compute_drift_metrics()

        # Record in history
        self._drift_history.append({"timestamp": time.time(), **drift_metrics})

        # Keep only last 1000 readings
        if len(self._drift_history) > 1000:
            self._drift_history = self._drift_history[-1000:]

        # Check if threshold exceeded
        if drift_metrics["cosine_distance"] > self.drift_threshold:
            alert = {
                "type": "embedding_drift",
                "timestamp": time.time(),
                "severity": (
                    "high"
                    if drift_metrics["cosine_distance"] > self.drift_threshold * 2
                    else "medium"
                ),
                "metrics": drift_metrics,
                "threshold": self.drift_threshold,
            }
            self._alerts.append(alert)
            return alert

        return None

    def compute_drift_metrics(self) -> Dict[str, float]:
        """
        Compute drift metrics between reference and current.

        Returns:
            Dictionary of drift metrics
        """
        if self._reference_centroid is None or self._current_centroid is None:
            return {"error": "Distributions not set"}

        # Cosine distance between centroids
        cosine_sim = np.dot(self._reference_centroid, self._current_centroid) / (
            np.linalg.norm(self._reference_centroid)
            * np.linalg.norm(self._current_centroid)
            + 1e-12
        )
        cosine_distance = 1 - cosine_sim

        # Euclidean distance between centroids
        euclidean_distance = float(
            np.linalg.norm(self._reference_centroid - self._current_centroid)
        )

        # Compute variance change
        current_std = np.std(np.array(self._current_embeddings), axis=0)
        std_change = (
            float(np.mean(np.abs(current_std - self._reference_std)))
            if self._reference_std is not None
            else 0.0
        )

        return {
            "cosine_distance": float(cosine_distance),
            "cosine_similarity": float(cosine_sim),
            "euclidean_distance": euclidean_distance,
            "std_change": std_change,
            "reference_size": len(self._reference_embeddings),
            "current_size": len(self._current_embeddings),
        }

    def get_drift_trend(self, window: int = 100) -> Dict[str, Any]:
        """
        Get drift trend over recent history.

        Args:
            window: Number of recent readings to analyze

        Returns:
            Trend analysis
        """
        if len(self._drift_history) < 2:
            return {"trend": "insufficient_data"}

        recent = self._drift_history[-window:]
        distances = [r["cosine_distance"] for r in recent]

        # Simple linear regression for trend
        x = np.arange(len(distances))
        slope = np.polyfit(x, distances, 1)[0] if len(distances) > 1 else 0

        return {
            "window_size": len(recent),
            "avg_drift": float(np.mean(distances)),
            "max_drift": float(np.max(distances)),
            "min_drift": float(np.min(distances)),
            "trend_slope": float(slope),
            "trend_direction": (
                "increasing"
                if slope > 0.001
                else ("decreasing" if slope < -0.001 else "stable")
            ),
            "above_threshold_count": sum(
                1 for d in distances if d > self.drift_threshold
            ),
        }

    def get_embeddings_for_umap(
        self,
        include_reference: bool = True,
        include_current: bool = True,
        sample_size: Optional[int] = 500,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get embeddings formatted for UMAP visualization.

        Args:
            include_reference: Include reference embeddings
            include_current: Include current embeddings
            sample_size: Max embeddings per distribution

        Returns:
            (embeddings_array, labels_list) for UMAP
        """
        embeddings = []
        labels = []

        if include_reference and self._reference_embeddings:
            ref = self._reference_embeddings
            if sample_size and len(ref) > sample_size:
                indices = np.random.choice(len(ref), sample_size, replace=False)
                ref = [ref[i] for i in indices]
            embeddings.extend(ref)
            labels.extend(["reference"] * len(ref))

        if include_current and self._current_embeddings:
            curr = self._current_embeddings
            if sample_size and len(curr) > sample_size:
                indices = np.random.choice(len(curr), sample_size, replace=False)
                curr = [curr[i] for i in indices]
            embeddings.extend(curr)
            labels.extend(["current"] * len(curr))

        if not embeddings:
            return np.array([]), []

        return np.array(embeddings), labels

    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent drift alerts."""
        return self._alerts[-limit:]

    def clear_alerts(self) -> int:
        """Clear alerts and return count."""
        count = len(self._alerts)
        self._alerts = []
        return count

    def reset_current(self) -> None:
        """Reset current distribution (e.g., after retraining)."""
        with self._lock:
            self._current_embeddings = []
            self._current_centroid = None
