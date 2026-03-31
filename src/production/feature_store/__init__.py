"""
Feature Store Submodule
=======================

Production-grade feature store for ML systems.

Inspired by Uber Michelangelo Palette architecture:
- Batch Pipeline: Historical feature computation
- Streaming Pipeline: Near-real-time features (interface)
- Feature Registry: Versioning and lineage tracking
- Online-Offline Consistency: Validation framework

Usage:
    from src.production.feature_store import (
        FeatureStore,
        FeatureRegistry,
        FeatureDefinition,
        FeatureValue,
        FeatureType,
        ComputationType,
    )

Example:
    >>> store = FeatureStore()
    >>> store.register_batch_transform(AvgOrderValueTransform())
    >>> results = store.run_batch(entity_ids, get_raw_data)
    >>> features = store.get_online_features("user_123", ["feature_name"])

Author: AI-Mastery-2026
Version: 2.0.0
"""

# Types and enums
from .types import (
    ComputationType,
    FeatureDefinition,
    FeatureGroup,
    FeatureType,
    FeatureValue,
    SerializationFormat,
    StreamingFeatureConfig,
    BatchResults,
    EntityFeatureMap,
    FeatureMap,
    FeatureVector,
)

# Registry
from .registry import FeatureRegistry

# Transforms
from .transforms import (
    FeatureTransform,
    AvgOrderValueTransform,
    SessionCountTransform,
    UserEngagementScoreTransform,
    ItemPopularityTransform,
)

# Pipelines
from .batch import BatchPipeline
from .streaming import (
    FeatureFreshnessTracker,
    GigascaleStreamingPipeline,
    StreamingPipeline,
)

# Online serving
from .online import OnlineFeatureServer

# Validation
from .validation import ConsistencyValidator

# Main class
from .store import FeatureStore, example_usage

__all__ = [
    # Types
    "FeatureType",
    "ComputationType",
    "FeatureDefinition",
    "FeatureValue",
    "FeatureGroup",
    "SerializationFormat",
    "StreamingFeatureConfig",
    # Type aliases
    "BatchResults",
    "EntityFeatureMap",
    "FeatureMap",
    "FeatureVector",
    # Registry
    "FeatureRegistry",
    # Transforms
    "FeatureTransform",
    "AvgOrderValueTransform",
    "SessionCountTransform",
    "UserEngagementScoreTransform",
    "ItemPopularityTransform",
    # Pipelines
    "BatchPipeline",
    "StreamingPipeline",
    "GigascaleStreamingPipeline",
    "FeatureFreshnessTracker",
    # Online serving
    "OnlineFeatureServer",
    # Validation
    "ConsistencyValidator",
    # Main
    "FeatureStore",
    "example_usage",
]

__version__ = "2.0.0"
