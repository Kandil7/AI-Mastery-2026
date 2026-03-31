"""
Online Feature Server Module
============================

Low-latency feature serving for online inference.

Features:
- Fast key-value lookups
- Feature vector assembly
- TTL-based cache invalidation
- Fallback to default values

Classes:
    OnlineFeatureServer: Low-latency feature serving

Author: AI-Mastery-2026
"""

import logging
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np

from .registry import FeatureRegistry
from .types import FeatureValue

logger = logging.getLogger(__name__)


class OnlineFeatureServer:
    """
    Low-latency feature serving for online inference.

    Features:
    - Fast key-value lookups
    - Feature vector assembly
    - TTL-based cache invalidation
    - Fallback to default values
    """

    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        self.store: Dict[str, Dict[str, FeatureValue]] = defaultdict(dict)
        self.defaults: Dict[str, Any] = {}
        self._lock = threading.Lock()

    def set_default(self, feature_name: str, default_value: Any) -> None:
        """Set default value for a feature."""
        self.defaults[feature_name] = default_value

    def put(self, feature_value: FeatureValue) -> None:
        """Store a feature value."""
        with self._lock:
            self.store[feature_value.feature_name][feature_value.entity_id] = (
                feature_value
            )

    def put_batch(self, values: List[FeatureValue]) -> None:
        """Store multiple feature values."""
        with self._lock:
            for fv in values:
                self.store[fv.feature_name][fv.entity_id] = fv

    def get(self, feature_name: str, entity_id: str) -> Optional[Any]:
        """
        Get a feature value with TTL check.

        Returns None if not found or expired.
        """
        fv = self.store.get(feature_name, {}).get(entity_id)

        if not fv:
            return self.defaults.get(feature_name)

        # Check TTL
        definition = self.registry.get_feature(feature_name)
        if definition:
            if fv.is_expired(definition.ttl_seconds):
                return self.defaults.get(feature_name)

        return fv.value

    def get_feature_vector(
        self, entity_id: str, feature_names: List[str]
    ) -> np.ndarray:
        """
        Assemble a feature vector for an entity.

        Args:
            entity_id: Entity ID
            feature_names: Ordered list of feature names

        Returns:
            Numpy array of feature values
        """
        values = []

        for name in feature_names:
            value = self.get(name, entity_id)

            if value is None:
                # Use 0 as fallback if no default
                value = self.defaults.get(name, 0)

            # Handle different types
            if isinstance(value, (list, np.ndarray)):
                values.extend(value)
            else:
                values.append(float(value) if value is not None else 0.0)

        return np.array(values, dtype=np.float32)

    def get_batch(
        self, entity_ids: List[str], feature_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Get feature vectors for multiple entities.

        Args:
            entity_ids: List of entity IDs
            feature_names: Feature names to retrieve

        Returns:
            Dictionary mapping entity_id -> feature vector
        """
        result = {}

        for entity_id in entity_ids:
            result[entity_id] = self.get_feature_vector(entity_id, feature_names)

        return result

    def get_with_metadata(
        self, feature_name: str, entity_id: str
    ) -> Optional[FeatureValue]:
        """Get feature value with metadata."""
        return self.store.get(feature_name, {}).get(entity_id)

    def get_all_for_entity(
        self, entity_id: str, feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get all feature values for an entity.

        Args:
            entity_id: Entity ID
            feature_names: Optional list of features to get (None = all)

        Returns:
            Dictionary of feature_name -> value
        """
        if feature_names is None:
            feature_names = list(self.store.keys())

        result = {}
        for name in feature_names:
            value = self.get(name, entity_id)
            if value is not None:
                result[name] = value

        return result

    def clear_entity(self, entity_id: str) -> int:
        """Clear all features for an entity."""
        count = 0
        with self._lock:
            for feature_store in self.store.values():
                if entity_id in feature_store:
                    del feature_store[entity_id]
                    count += 1
        return count

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        total_entities = set()
        for feature_store in self.store.values():
            total_entities.update(feature_store.keys())

        return {
            "total_features": len(self.store),
            "total_entities": len(total_entities),
            "total_defaults": len(self.defaults),
        }
