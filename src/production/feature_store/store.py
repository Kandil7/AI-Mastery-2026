"""
Feature Store Main Module
==========================

Main Feature Store class combining all components.

Provides a unified interface for:
- Feature registration and discovery
- Batch and streaming computation
- Online serving
- Consistency validation

Classes:
    FeatureStore: Main feature store class

Example:
    >>> store = FeatureStore()
    >>>
    >>> # Register features
    >>> store.register_feature(FeatureDefinition(
    ...     name="user_avg_order_value",
    ...     feature_type=FeatureType.NUMERIC,
    ...     computation_type=ComputationType.BATCH,
    ...     description="Average order value in last 30 days",
    ...     entity_key="user"
    ... ))
    >>>
    >>> # Get online features
    >>> features = store.get_online_features("user_123", ["user_avg_order_value"])

Author: AI-Mastery-2026
"""

import logging
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .batch import BatchPipeline
from .online import OnlineFeatureServer
from .registry import FeatureRegistry
from .streaming import StreamingPipeline
from .transforms import FeatureTransform
from .types import FeatureDefinition, FeatureGroup, FeatureValue
from .validation import ConsistencyValidator

logger = logging.getLogger(__name__)


class FeatureStore:
    """
    Main Feature Store class combining all components.

    Provides a unified interface for:
    - Feature registration and discovery
    - Batch and streaming computation
    - Online serving
    - Consistency validation

    Example:
        >>> store = FeatureStore()
        >>>
        >>> # Register features
        >>> store.register_feature(FeatureDefinition(
        ...     name="user_avg_order_value",
        ...     feature_type="numeric",
        ...     computation_type="batch",
        ...     description="Average order value in last 30 days",
        ...     entity_key="user"
        ... ))
        >>>
        >>> # Get online features
        >>> features = store.get_online_features("user_123", ["user_avg_order_value"])
    """

    def __init__(self, validate_consistency: bool = True):
        self.registry = FeatureRegistry()
        self.batch_pipeline = BatchPipeline(self.registry)
        self.streaming_pipeline = StreamingPipeline(self.registry)
        self.online_server = OnlineFeatureServer(self.registry)
        self.validator = ConsistencyValidator()
        self.validate_consistency = validate_consistency

        # Subscribe streaming to update online server
        self.streaming_pipeline.subscribe(self.online_server.put)

    def register_feature(self, feature: FeatureDefinition) -> None:
        """Register a feature definition."""
        self.registry.register_feature(feature)

    def register_group(self, group: FeatureGroup) -> None:
        """Register a feature group."""
        self.registry.register_group(group)

    def register_batch_transform(self, transform: FeatureTransform) -> None:
        """Register a batch feature transform."""
        self.batch_pipeline.register_transform(transform)

    def register_streaming_transform(self, transform: FeatureTransform) -> None:
        """Register a streaming feature transform."""
        self.streaming_pipeline.register_transform(transform)

    def run_batch(
        self, entity_ids: List[str], raw_data_fn: Callable[[str], Dict[str, Any]]
    ) -> Dict[str, Dict[str, FeatureValue]]:
        """Run batch computation for all entities."""
        results = self.batch_pipeline.compute_all(entity_ids, raw_data_fn)

        # Sync to online server
        for feature_name, entity_values in results.items():
            for entity_id, fv in entity_values.items():
                self.online_server.put(fv)

        return results

    def process_event(self, event: Dict[str, Any]) -> List[FeatureValue]:
        """Process a streaming event."""
        return self.streaming_pipeline.on_event(event)

    def get_online_features(
        self, entity_id: str, feature_names: List[str]
    ) -> np.ndarray:
        """Get feature vector from online server."""
        return self.online_server.get_feature_vector(entity_id, feature_names)

    def get_online_features_batch(
        self, entity_ids: List[str], feature_names: List[str]
    ) -> Dict[str, np.ndarray]:
        """Get feature vectors for multiple entities."""
        return self.online_server.get_batch(entity_ids, feature_names)

    def validate(self, entity_ids: List[str]) -> Dict[str, Any]:
        """Validate online-offline consistency."""
        if not self.validate_consistency:
            return {"status": "skipped"}

        return self.validator.validate_batch(
            self.batch_pipeline.computed_values, self.online_server, entity_ids
        )

    def list_features(self, **filters) -> List[FeatureDefinition]:
        """List registered features."""
        return self.registry.list_features(**filters)

    def get_feature_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a feature."""
        return self.registry.get_feature_info(name)


def example_usage():
    """Demonstrate Feature Store usage."""
    from .transforms import AvgOrderValueTransform, SessionCountTransform

    # Create feature store
    store = FeatureStore()

    # Register transforms
    store.register_batch_transform(AvgOrderValueTransform())
    store.register_streaming_transform(SessionCountTransform())

    # Set defaults
    store.online_server.set_default("user_avg_order_value", 0.0)
    store.online_server.set_default("user_session_count_7d", 0)

    # Simulate batch computation
    def get_raw_data(entity_id: str) -> Dict:
        return {
            "orders": [{"value": 25.0}, {"value": 45.0}, {"value": 30.0}],
            "sessions": [{"ts": "2024-01-01"}, {"ts": "2024-01-02"}],
        }

    entity_ids = ["user_1", "user_2", "user_3"]

    # Run batch
    results = store.run_batch(entity_ids, get_raw_data)
    print(f"Batch computed {len(results)} features")

    # Get online features
    features = store.get_online_features("user_1", ["user_avg_order_value"])
    print(f"Online features for user_1: {features}")

    # Process streaming event
    event = {"entity_id": "user_1", "sessions": [{"ts": "now"}]}
    store.process_event(event)
    print("Processed streaming event")

    # Validate consistency
    report = store.validate(entity_ids)
    print(f"Validation report: {report['passed']}/{report['total_checks']} passed")

    # List features
    features = store.list_features()
    print(f"\nRegistered features: {[f.name for f in features]}")

    return store


if __name__ == "__main__":
    example_usage()
