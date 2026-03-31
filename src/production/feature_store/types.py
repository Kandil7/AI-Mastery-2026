"""
Feature Store Type Definitions
===============================

Core data structures for the feature store.

Enums:
    FeatureType: Types of features (numeric, categorical, embedding, etc.)
    ComputationType: How features are computed (batch, streaming, on_demand)
    SerializationFormat: Serialization formats for feature values

Dataclasses:
    FeatureDefinition: Definition of a feature with metadata
    FeatureValue: A computed feature value with metadata
    FeatureGroup: A group of related features
    StreamingFeatureConfig: Configuration for streaming features

Author: AI-Mastery-2026
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class FeatureType(Enum):
    """Types of features."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    EMBEDDING = "embedding"
    TIMESTAMP = "timestamp"
    LIST = "list"


class ComputationType(Enum):
    """How the feature is computed."""

    BATCH = "batch"  # Computed periodically (daily, hourly)
    STREAMING = "streaming"  # Computed in near-real-time
    ON_DEMAND = "on_demand"  # Computed at request time


class SerializationFormat(Enum):
    """
    Serialization formats for feature values (DoorDash optimization).

    DoorDash achieved 3x faster serialization and 2x smaller storage
    by switching from JSON to Protobuf.
    """

    JSON = "json"  # Default, human-readable but slow
    MSGPACK = "msgpack"  # Fast, compact binary format
    PROTOBUF = "protobuf"  # Google's binary format (fastest)


@dataclass
class FeatureDefinition:
    """
    Definition of a feature with metadata.

    Attributes:
        name: Unique feature name
        feature_type: Data type of the feature
        computation_type: How the feature is computed
        description: Human-readable description
        entity_key: Primary entity (e.g., "user", "item")
        dependencies: Other features this depends on
        ttl_seconds: Time-to-live for cached values
        version: Feature version
        owner: Team or person responsible
        tags: Optional tags for categorization
    """

    name: str
    feature_type: FeatureType
    computation_type: ComputationType
    description: str
    entity_key: str
    dependencies: List[str] = field(default_factory=list)
    ttl_seconds: int = 86400  # 24 hours default
    version: str = "1.0.0"
    owner: str = "ml-platform"
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "feature_type": self.feature_type.value,
            "computation_type": self.computation_type.value,
            "description": self.description,
            "entity_key": self.entity_key,
            "dependencies": self.dependencies,
            "ttl_seconds": self.ttl_seconds,
            "version": self.version,
            "owner": self.owner,
            "tags": self.tags,
        }


@dataclass
class FeatureValue:
    """
    A computed feature value with metadata.

    Attributes:
        feature_name: Name of the feature
        entity_id: ID of the entity (e.g., user_id)
        value: The actual feature value
        timestamp: When the value was computed
        version: Feature version used
    """

    feature_name: str
    entity_id: str
    value: Any
    timestamp: datetime
    version: str = "1.0.0"

    @property
    def is_expired(self, ttl_seconds: int = 86400) -> bool:
        """Check if feature value has expired."""
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > ttl_seconds


@dataclass
class FeatureGroup:
    """
    A group of related features.

    Example: "user_engagement_features" containing
    avg_session_duration, pages_per_session, etc.
    """

    name: str
    description: str
    entity_key: str
    features: List[FeatureDefinition] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    owner: str = "ml-platform"

    def add_feature(self, feature: FeatureDefinition) -> None:
        """Add a feature to the group."""
        if feature.entity_key != self.entity_key:
            raise ValueError(
                f"Feature entity_key '{feature.entity_key}' doesn't match "
                f"group entity_key '{self.entity_key}'"
            )
        self.features.append(feature)

    def get_feature(self, name: str) -> Optional[FeatureDefinition]:
        """Get a feature by name."""
        for f in self.features:
            if f.name == name:
                return f
        return None


@dataclass
class StreamingFeatureConfig:
    """Configuration for streaming feature computation."""

    feature_name: str
    window_seconds: int = 300  # 5 minute window default
    aggregation: str = "count"  # count, sum, avg, max, min
    max_staleness_seconds: int = 60  # Alert if older than this


# Type aliases for common patterns
FeatureMap = Dict[str, Any]
EntityFeatureMap = Dict[
    str, Dict[str, FeatureValue]
]  # entity_id -> feature_name -> value
FeatureVector = List[float]
BatchResults = Dict[str, EntityFeatureMap]  # feature_name -> entity_features
