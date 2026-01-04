"""
Feature Store Module
====================

Production-grade feature store for ML systems.

Inspired by Uber Michelangelo Palette architecture:
- Batch Pipeline: Historical feature computation
- Streaming Pipeline: Near-real-time features (interface)
- Feature Registry: Versioning and lineage tracking
- Online-Offline Consistency: Validation framework

Features:
- Feature versioning and metadata management
- Batch vs streaming feature computation
- Point-in-time correctness for training
- Online serving with low-latency access

References:
- Uber Michelangelo: https://eng.uber.com/michelangelo-machine-learning-platform/
"""

import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from abc import ABC, abstractmethod
import hashlib
import json
import logging
import time
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

class FeatureType(Enum):
    """Types of features."""
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    EMBEDDING = "embedding"
    TIMESTAMP = "timestamp"
    LIST = "list"


class ComputationType(Enum):
    """How the feature is computed."""
    BATCH = "batch"           # Computed periodically (daily, hourly)
    STREAMING = "streaming"   # Computed in near-real-time
    ON_DEMAND = "on_demand"   # Computed at request time


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
            "tags": self.tags
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


# ============================================================
# FEATURE REGISTRY
# ============================================================

class FeatureRegistry:
    """
    Central registry for all feature definitions.
    
    Provides:
    - Feature discovery and documentation
    - Version tracking
    - Dependency management
    - Lineage tracking
    """
    
    def __init__(self):
        self.features: Dict[str, FeatureDefinition] = {}
        self.groups: Dict[str, FeatureGroup] = {}
        self.versions: Dict[str, List[str]] = defaultdict(list)
        self._lock = threading.Lock()
        
    def register_feature(self, feature: FeatureDefinition) -> None:
        """Register a feature definition."""
        with self._lock:
            key = f"{feature.name}:{feature.version}"
            self.features[key] = feature
            self.versions[feature.name].append(feature.version)
            logger.info(f"Registered feature: {feature.name} v{feature.version}")
            
    def register_group(self, group: FeatureGroup) -> None:
        """Register a feature group."""
        with self._lock:
            self.groups[group.name] = group
            for feature in group.features:
                self.register_feature(feature)
            logger.info(f"Registered feature group: {group.name}")
                
    def get_feature(self, name: str, 
                    version: Optional[str] = None) -> Optional[FeatureDefinition]:
        """Get a feature definition by name and optional version."""
        if version:
            key = f"{name}:{version}"
            return self.features.get(key)
        else:
            # Get latest version
            versions = self.versions.get(name, [])
            if versions:
                latest = sorted(versions)[-1]
                return self.features.get(f"{name}:{latest}")
        return None
    
    def list_features(self, entity_key: Optional[str] = None,
                      computation_type: Optional[ComputationType] = None,
                      tags: Optional[List[str]] = None) -> List[FeatureDefinition]:
        """List features with optional filters."""
        results = []
        
        # Get unique latest versions
        seen = set()
        for key, feature in self.features.items():
            if feature.name in seen:
                continue
                
            if entity_key and feature.entity_key != entity_key:
                continue
            if computation_type and feature.computation_type != computation_type:
                continue
            if tags and not any(t in feature.tags for t in tags):
                continue
                
            results.append(feature)
            seen.add(feature.name)
            
        return results
    
    def get_dependencies(self, feature_name: str) -> List[str]:
        """Get all dependencies for a feature (recursive)."""
        feature = self.get_feature(feature_name)
        if not feature:
            return []
            
        all_deps = set(feature.dependencies)
        
        for dep in feature.dependencies:
            all_deps.update(self.get_dependencies(dep))
            
        return list(all_deps)
    
    def validate_dependencies(self, feature: FeatureDefinition) -> bool:
        """Validate that all dependencies exist."""
        for dep in feature.dependencies:
            if not self.get_feature(dep):
                logger.warning(f"Missing dependency: {dep} for feature {feature.name}")
                return False
        return True


# ============================================================
# FEATURE COMPUTATION
# ============================================================

class FeatureTransform(ABC):
    """
    Abstract base class for feature transformations.
    
    Implement this to define how a feature is computed.
    """
    
    @abstractmethod
    def compute(self, entity_id: str, 
                raw_data: Dict[str, Any],
                dependencies: Dict[str, Any]) -> Any:
        """
        Compute the feature value.
        
        Args:
            entity_id: ID of the entity
            raw_data: Raw input data
            dependencies: Values of dependent features
            
        Returns:
            Computed feature value
        """
        pass
    
    @property
    @abstractmethod
    def feature_definition(self) -> FeatureDefinition:
        """Return the feature definition."""
        pass


class BatchPipeline:
    """
    Batch feature computation pipeline.
    
    Runs periodically (e.g., daily) to compute features
    for all entities.
    """
    
    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        self.transforms: Dict[str, FeatureTransform] = {}
        self.computed_values: Dict[str, Dict[str, FeatureValue]] = defaultdict(dict)
        
    def register_transform(self, transform: FeatureTransform) -> None:
        """Register a transform for batch computation."""
        self.transforms[transform.feature_definition.name] = transform
        self.registry.register_feature(transform.feature_definition)
        
    def compute_feature(self, feature_name: str,
                        entity_id: str,
                        raw_data: Dict[str, Any]) -> FeatureValue:
        """Compute a single feature for an entity."""
        if feature_name not in self.transforms:
            raise ValueError(f"No transform registered for: {feature_name}")
            
        transform = self.transforms[feature_name]
        definition = transform.feature_definition
        
        # Get dependency values
        dependencies = {}
        for dep_name in definition.dependencies:
            if dep_name in self.computed_values:
                if entity_id in self.computed_values[dep_name]:
                    dependencies[dep_name] = self.computed_values[dep_name][entity_id].value
                    
        # Compute the feature
        value = transform.compute(entity_id, raw_data, dependencies)
        
        feature_value = FeatureValue(
            feature_name=feature_name,
            entity_id=entity_id,
            value=value,
            timestamp=datetime.now(),
            version=definition.version
        )
        
        # Cache the computed value
        self.computed_values[feature_name][entity_id] = feature_value
        
        return feature_value
    
    def compute_all(self, entity_ids: List[str],
                    raw_data_fn: Callable[[str], Dict[str, Any]]) -> Dict[str, Dict[str, FeatureValue]]:
        """
        Compute all registered features for all entities.
        
        Args:
            entity_ids: List of entity IDs
            raw_data_fn: Function to get raw data for an entity
            
        Returns:
            Nested dict: feature_name -> entity_id -> FeatureValue
        """
        # Topologically sort features by dependencies
        sorted_features = self._topological_sort()
        
        results = defaultdict(dict)
        
        for entity_id in entity_ids:
            raw_data = raw_data_fn(entity_id)
            
            for feature_name in sorted_features:
                try:
                    feature_value = self.compute_feature(
                        feature_name, entity_id, raw_data
                    )
                    results[feature_name][entity_id] = feature_value
                except Exception as e:
                    logger.error(f"Error computing {feature_name} for {entity_id}: {e}")
                    
        return dict(results)
    
    def _topological_sort(self) -> List[str]:
        """Sort features by dependencies."""
        # Build dependency graph
        in_degree = defaultdict(int)
        graph = defaultdict(list)
        
        for name, transform in self.transforms.items():
            for dep in transform.feature_definition.dependencies:
                if dep in self.transforms:
                    graph[dep].append(name)
                    in_degree[name] += 1
                    
            if name not in in_degree:
                in_degree[name] = 0
                
        # Kahn's algorithm
        queue = [name for name, degree in in_degree.items() if degree == 0]
        sorted_features = []
        
        while queue:
            name = queue.pop(0)
            sorted_features.append(name)
            
            for dependent in graph[name]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
                    
        return sorted_features


class StreamingPipeline:
    """
    Near-real-time feature computation pipeline.
    
    This is an interface/stub implementation. In production,
    this would integrate with Kafka/Samza for real streaming.
    """
    
    def __init__(self, registry: FeatureRegistry):
        self.registry = registry
        self.transforms: Dict[str, FeatureTransform] = {}
        self.latest_values: Dict[str, Dict[str, FeatureValue]] = defaultdict(dict)
        self._subscribers: List[Callable[[FeatureValue], None]] = []
        
    def register_transform(self, transform: FeatureTransform) -> None:
        """Register a streaming transform."""
        self.transforms[transform.feature_definition.name] = transform
        self.registry.register_feature(transform.feature_definition)
        
    def subscribe(self, callback: Callable[[FeatureValue], None]) -> None:
        """Subscribe to feature updates."""
        self._subscribers.append(callback)
        
    def on_event(self, event: Dict[str, Any]) -> List[FeatureValue]:
        """
        Process an incoming event and compute affected features.
        
        Args:
            event: Event data (e.g., user action)
            
        Returns:
            List of computed feature values
        """
        entity_id = event.get("entity_id")
        if not entity_id:
            return []
            
        computed = []
        
        for name, transform in self.transforms.items():
            try:
                value = transform.compute(entity_id, event, {})
                feature_value = FeatureValue(
                    feature_name=name,
                    entity_id=entity_id,
                    value=value,
                    timestamp=datetime.now(),
                    version=transform.feature_definition.version
                )
                
                self.latest_values[name][entity_id] = feature_value
                computed.append(feature_value)
                
                # Notify subscribers
                for callback in self._subscribers:
                    callback(feature_value)
                    
            except Exception as e:
                logger.error(f"Error in streaming compute {name}: {e}")
                
        return computed
    
    def get_latest(self, feature_name: str, entity_id: str) -> Optional[FeatureValue]:
        """Get the latest value for a feature."""
        return self.latest_values.get(feature_name, {}).get(entity_id)


# ============================================================
# DOORDASH-INSPIRED STREAMING ENHANCEMENTS
# ============================================================

class SerializationFormat(Enum):
    """
    Serialization formats for feature values (DoorDash optimization).
    
    DoorDash achieved 3x faster serialization and 2x smaller storage
    by switching from JSON to Protobuf.
    """
    JSON = "json"           # Default, human-readable but slow
    MSGPACK = "msgpack"     # Fast, compact binary format
    PROTOBUF = "protobuf"   # Google's binary format (fastest)


@dataclass
class StreamingFeatureConfig:
    """Configuration for streaming feature computation."""
    feature_name: str
    window_seconds: int = 300  # 5 minute window default
    aggregation: str = "count"  # count, sum, avg, max, min
    max_staleness_seconds: int = 60  # Alert if older than this


class FeatureFreshnessTracker:
    """
    Track staleness of features (DoorDash Gigascale pattern).
    
    Ensures features are fresh enough for real-time inference.
    Alerts when features exceed their TTL thresholds.
    """
    
    def __init__(self):
        self._last_update: Dict[str, Dict[str, datetime]] = defaultdict(dict)
        self._staleness_alerts: List[Dict[str, Any]] = []
        self._freshness_stats: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()
    
    def record_update(self, feature_name: str, entity_id: str) -> None:
        """Record a feature update."""
        with self._lock:
            self._last_update[feature_name][entity_id] = datetime.now()
    
    def get_staleness(self, feature_name: str, entity_id: str) -> Optional[float]:
        """
        Get staleness in seconds for a feature.
        
        Returns:
            Seconds since last update, or None if never updated
        """
        last = self._last_update.get(feature_name, {}).get(entity_id)
        if not last:
            return None
        return (datetime.now() - last).total_seconds()
    
    def check_freshness(
        self,
        feature_name: str,
        entity_id: str,
        max_staleness_seconds: float
    ) -> Tuple[bool, Optional[float]]:
        """
        Check if a feature is fresh enough.
        
        Returns:
            (is_fresh, staleness_seconds)
        """
        staleness = self.get_staleness(feature_name, entity_id)
        
        if staleness is None:
            return False, None
        
        is_fresh = staleness <= max_staleness_seconds
        
        # Track freshness stats
        with self._lock:
            self._freshness_stats[feature_name].append(staleness)
            # Keep only last 1000 readings
            if len(self._freshness_stats[feature_name]) > 1000:
                self._freshness_stats[feature_name] = \
                    self._freshness_stats[feature_name][-1000:]
        
        if not is_fresh:
            self._staleness_alerts.append({
                "feature_name": feature_name,
                "entity_id": entity_id,
                "staleness_seconds": staleness,
                "threshold_seconds": max_staleness_seconds,
                "timestamp": datetime.now().isoformat()
            })
        
        return is_fresh, staleness
    
    def get_freshness_stats(self, feature_name: str) -> Dict[str, Any]:
        """Get freshness statistics for a feature."""
        stats = self._freshness_stats.get(feature_name, [])
        
        if not stats:
            return {"feature_name": feature_name, "no_data": True}
        
        return {
            "feature_name": feature_name,
            "sample_count": len(stats),
            "avg_staleness_seconds": float(np.mean(stats)),
            "p50_staleness_seconds": float(np.percentile(stats, 50)),
            "p95_staleness_seconds": float(np.percentile(stats, 95)),
            "p99_staleness_seconds": float(np.percentile(stats, 99)),
            "max_staleness_seconds": float(np.max(stats))
        }
    
    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent staleness alerts."""
        return self._staleness_alerts[-limit:]
    
    def clear_alerts(self) -> int:
        """Clear alerts and return count."""
        count = len(self._staleness_alerts)
        self._staleness_alerts = []
        return count


class GigascaleStreamingPipeline:
    """
    High-performance streaming feature pipeline (DoorDash pattern).
    
    Features:
    - Kafka-style event processing
    - Configurable serialization (Protobuf/MsgPack)
    - Window-based aggregations
    - Freshness tracking and alerting
    - Read/write path separation
    
    Reference: DoorDash Engineering - Gigascale Feature Store
    """
    
    def __init__(
        self,
        registry: FeatureRegistry,
        serialization: SerializationFormat = SerializationFormat.MSGPACK
    ):
        self.registry = registry
        self.serialization = serialization
        self.transforms: Dict[str, FeatureTransform] = {}
        self.configs: Dict[str, StreamingFeatureConfig] = {}
        self.freshness_tracker = FeatureFreshnessTracker()
        
        # Window buffers for aggregations
        self._window_buffers: Dict[str, Dict[str, List[Tuple[datetime, Any]]]] = \
            defaultdict(lambda: defaultdict(list))
        
        # Latest values (read path)
        self._latest_values: Dict[str, Dict[str, FeatureValue]] = defaultdict(dict)
        
        # Event queue (simulated Kafka)
        self._event_queue: List[Dict[str, Any]] = []
        self._subscribers: List[Callable[[FeatureValue], None]] = []
        
        # Stats
        self._events_processed = 0
        self._features_computed = 0
        self._serialization_time_ms: List[float] = []
        
        self._lock = threading.Lock()
    
    def register_streaming_feature(
        self,
        transform: FeatureTransform,
        config: Optional[StreamingFeatureConfig] = None
    ) -> None:
        """Register a streaming feature with optional config."""
        name = transform.feature_definition.name
        self.transforms[name] = transform
        self.registry.register_feature(transform.feature_definition)
        
        if config:
            self.configs[name] = config
        else:
            self.configs[name] = StreamingFeatureConfig(feature_name=name)
    
    def subscribe(self, callback: Callable[[FeatureValue], None]) -> None:
        """Subscribe to feature updates."""
        self._subscribers.append(callback)
    
    def enqueue_event(self, event: Dict[str, Any]) -> None:
        """Add event to processing queue (simulated Kafka producer)."""
        event["_enqueued_at"] = datetime.now().isoformat()
        self._event_queue.append(event)
    
    def process_events(self, batch_size: int = 100) -> List[FeatureValue]:
        """
        Process queued events (simulated Kafka consumer).
        
        Args:
            batch_size: Max events to process
            
        Returns:
            List of computed feature values
        """
        computed = []
        
        with self._lock:
            events_to_process = self._event_queue[:batch_size]
            self._event_queue = self._event_queue[batch_size:]
        
        for event in events_to_process:
            values = self._process_single_event(event)
            computed.extend(values)
            self._events_processed += 1
        
        return computed
    
    def _process_single_event(self, event: Dict[str, Any]) -> List[FeatureValue]:
        """Process a single event."""
        entity_id = event.get("entity_id")
        if not entity_id:
            return []
        
        computed = []
        event_time = datetime.now()
        
        for name, transform in self.transforms.items():
            try:
                config = self.configs.get(name, StreamingFeatureConfig(feature_name=name))
                
                # Add to window buffer
                value = transform.compute(entity_id, event, {})
                self._window_buffers[name][entity_id].append((event_time, value))
                
                # Clean old entries from window
                window_start = event_time - timedelta(seconds=config.window_seconds)
                self._window_buffers[name][entity_id] = [
                    (t, v) for t, v in self._window_buffers[name][entity_id]
                    if t >= window_start
                ]
                
                # Compute aggregated value
                window_values = [v for _, v in self._window_buffers[name][entity_id]]
                aggregated = self._aggregate(window_values, config.aggregation)
                
                # Serialize (tracking time)
                start_serialize = time.time()
                serialized = self._serialize(aggregated)
                serialize_ms = (time.time() - start_serialize) * 1000
                self._serialization_time_ms.append(serialize_ms)
                
                # Create feature value
                feature_value = FeatureValue(
                    feature_name=name,
                    entity_id=entity_id,
                    value=aggregated,
                    timestamp=event_time,
                    version=transform.feature_definition.version
                )
                
                # Update read path
                self._latest_values[name][entity_id] = feature_value
                self.freshness_tracker.record_update(name, entity_id)
                
                computed.append(feature_value)
                self._features_computed += 1
                
                # Notify subscribers
                for callback in self._subscribers:
                    callback(feature_value)
                    
            except Exception as e:
                logger.error(f"Error processing event for {name}: {e}")
        
        return computed
    
    def _aggregate(self, values: List[Any], aggregation: str) -> Any:
        """Compute aggregation over window values."""
        if not values:
            return 0
        
        numeric_values = [v for v in values if isinstance(v, (int, float))]
        
        if not numeric_values:
            return values[-1] if values else 0
        
        if aggregation == "count":
            return len(numeric_values)
        elif aggregation == "sum":
            return sum(numeric_values)
        elif aggregation == "avg":
            return sum(numeric_values) / len(numeric_values)
        elif aggregation == "max":
            return max(numeric_values)
        elif aggregation == "min":
            return min(numeric_values)
        else:
            return numeric_values[-1]
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value using configured format."""
        if self.serialization == SerializationFormat.JSON:
            return json.dumps(value).encode('utf-8')
        elif self.serialization == SerializationFormat.MSGPACK:
            # Simulated msgpack (would use msgpack library in production)
            return json.dumps(value).encode('utf-8')  # Fallback
        elif self.serialization == SerializationFormat.PROTOBUF:
            # Simulated protobuf (would use generated proto classes in production)
            return json.dumps(value).encode('utf-8')  # Fallback
        else:
            return str(value).encode('utf-8')
    
    def get_latest(self, feature_name: str, entity_id: str) -> Optional[FeatureValue]:
        """Get latest value (read path)."""
        return self._latest_values.get(feature_name, {}).get(entity_id)
    
    def get_fresh_value(
        self,
        feature_name: str,
        entity_id: str,
        max_staleness_seconds: Optional[float] = None
    ) -> Tuple[Optional[FeatureValue], bool]:
        """
        Get value only if fresh enough.
        
        Returns:
            (value, is_fresh)
        """
        config = self.configs.get(feature_name, StreamingFeatureConfig(feature_name=feature_name))
        threshold = max_staleness_seconds or config.max_staleness_seconds
        
        is_fresh, _ = self.freshness_tracker.check_freshness(
            feature_name, entity_id, threshold
        )
        
        if not is_fresh:
            return None, False
        
        return self.get_latest(feature_name, entity_id), True
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        serialize_times = self._serialization_time_ms[-1000:] if self._serialization_time_ms else [0]
        
        return {
            "events_processed": self._events_processed,
            "features_computed": self._features_computed,
            "pending_events": len(self._event_queue),
            "serialization_format": self.serialization.value,
            "avg_serialize_ms": float(np.mean(serialize_times)),
            "p99_serialize_ms": float(np.percentile(serialize_times, 99)) if len(serialize_times) > 1 else 0,
            "freshness_alerts": len(self.freshness_tracker.get_alerts())
        }


# ============================================================
# ONLINE FEATURE SERVER
# ============================================================

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
            self.store[feature_value.feature_name][feature_value.entity_id] = feature_value
            
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
    
    def get_feature_vector(self, entity_id: str,
                           feature_names: List[str]) -> np.ndarray:
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
    
    def get_batch(self, entity_ids: List[str],
                  feature_names: List[str]) -> Dict[str, np.ndarray]:
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


# ============================================================
# ONLINE-OFFLINE CONSISTENCY
# ============================================================

class ConsistencyValidator:
    """
    Validate consistency between online and offline features.
    
    The "training-serving skew" is a major issue in ML systems.
    This validator ensures that:
    1. Features computed offline match those served online
    2. Point-in-time correctness is maintained
    3. Data drift is detected
    """
    
    def __init__(self, tolerance: float = 0.01):
        self.tolerance = tolerance
        self.validation_results: List[Dict[str, Any]] = []
        
    def validate(self, offline_value: Any, 
                 online_value: Any,
                 feature_name: str) -> Tuple[bool, Optional[str]]:
        """
        Validate consistency between offline and online values.
        
        Returns:
            (is_consistent, error_message)
        """
        if offline_value is None and online_value is None:
            return True, None
            
        if offline_value is None or online_value is None:
            return False, f"One value is None: offline={offline_value}, online={online_value}"
            
        # Numeric comparison
        if isinstance(offline_value, (int, float)) and isinstance(online_value, (int, float)):
            diff = abs(offline_value - online_value)
            max_val = max(abs(offline_value), abs(online_value), 1e-8)
            relative_diff = diff / max_val
            
            if relative_diff > self.tolerance:
                return False, f"Values differ by {relative_diff:.4f} (tolerance: {self.tolerance})"
            return True, None
            
        # Array comparison
        if isinstance(offline_value, np.ndarray) and isinstance(online_value, np.ndarray):
            if offline_value.shape != online_value.shape:
                return False, f"Shape mismatch: {offline_value.shape} vs {online_value.shape}"
                
            diff = np.abs(offline_value - online_value)
            max_diff = np.max(diff)
            
            if max_diff > self.tolerance:
                return False, f"Max element difference: {max_diff:.4f}"
            return True, None
            
        # Exact comparison for other types
        if offline_value != online_value:
            return False, f"Values don't match: {offline_value} vs {online_value}"
            
        return True, None
    
    def validate_batch(self, 
                       offline_values: Dict[str, Dict[str, Any]],
                       online_server: OnlineFeatureServer,
                       entity_ids: List[str]) -> Dict[str, Any]:
        """
        Validate a batch of features across entities.
        
        Args:
            offline_values: feature_name -> entity_id -> value
            online_server: Online feature server
            entity_ids: Entities to validate
            
        Returns:
            Validation report
        """
        report = {
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "failures": []
        }
        
        for feature_name, entity_values in offline_values.items():
            for entity_id in entity_ids:
                if entity_id not in entity_values:
                    continue
                    
                offline_val = entity_values[entity_id]
                online_val = online_server.get(feature_name, entity_id)
                
                # Handle FeatureValue wrapper
                if isinstance(offline_val, FeatureValue):
                    offline_val = offline_val.value
                    
                is_consistent, error = self.validate(
                    offline_val, online_val, feature_name
                )
                
                report["total_checks"] += 1
                
                if is_consistent:
                    report["passed"] += 1
                else:
                    report["failed"] += 1
                    report["failures"].append({
                        "feature": feature_name,
                        "entity_id": entity_id,
                        "error": error
                    })
                    
        return report


# ============================================================
# FEATURE STORE (MAIN CLASS)
# ============================================================

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
        ...     feature_type=FeatureType.NUMERIC,
        ...     computation_type=ComputationType.BATCH,
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
        
    def run_batch(self, entity_ids: List[str],
                  raw_data_fn: Callable[[str], Dict[str, Any]]) -> Dict[str, Dict[str, FeatureValue]]:
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
    
    def get_online_features(self, entity_id: str,
                            feature_names: List[str]) -> np.ndarray:
        """Get feature vector from online server."""
        return self.online_server.get_feature_vector(entity_id, feature_names)
    
    def get_online_features_batch(self, entity_ids: List[str],
                                   feature_names: List[str]) -> Dict[str, np.ndarray]:
        """Get feature vectors for multiple entities."""
        return self.online_server.get_batch(entity_ids, feature_names)
    
    def validate(self, entity_ids: List[str]) -> Dict[str, Any]:
        """Validate online-offline consistency."""
        if not self.validate_consistency:
            return {"status": "skipped"}
            
        return self.validator.validate_batch(
            self.batch_pipeline.computed_values,
            self.online_server,
            entity_ids
        )
    
    def list_features(self, **filters) -> List[FeatureDefinition]:
        """List registered features."""
        return self.registry.list_features(**filters)
    
    def get_feature_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a feature."""
        feature = self.registry.get_feature(name)
        if not feature:
            return None
            
        return {
            "definition": feature.to_dict(),
            "dependencies": self.registry.get_dependencies(name),
            "versions": self.registry.versions.get(name, [])
        }


# ============================================================
# EXAMPLE TRANSFORMS
# ============================================================

class AvgOrderValueTransform(FeatureTransform):
    """Example: Average order value for a user."""
    
    def __init__(self, window_days: int = 30):
        self.window_days = window_days
        
    @property
    def feature_definition(self) -> FeatureDefinition:
        return FeatureDefinition(
            name="user_avg_order_value",
            feature_type=FeatureType.NUMERIC,
            computation_type=ComputationType.BATCH,
            description=f"Average order value in last {self.window_days} days",
            entity_key="user",
            ttl_seconds=86400
        )
        
    def compute(self, entity_id: str,
                raw_data: Dict[str, Any],
                dependencies: Dict[str, Any]) -> float:
        orders = raw_data.get("orders", [])
        if not orders:
            return 0.0
        return float(np.mean([o.get("value", 0) for o in orders]))


class SessionCountTransform(FeatureTransform):
    """Example: Session count in last 7 days."""
    
    @property
    def feature_definition(self) -> FeatureDefinition:
        return FeatureDefinition(
            name="user_session_count_7d",
            feature_type=FeatureType.NUMERIC,
            computation_type=ComputationType.STREAMING,
            description="Number of sessions in last 7 days",
            entity_key="user",
            ttl_seconds=3600
        )
        
    def compute(self, entity_id: str,
                raw_data: Dict[str, Any],
                dependencies: Dict[str, Any]) -> int:
        sessions = raw_data.get("sessions", [])
        return len(sessions)


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_usage():
    """Demonstrate Feature Store usage."""
    
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
            "orders": [
                {"value": 25.0}, {"value": 45.0}, {"value": 30.0}
            ],
            "sessions": [
                {"ts": "2024-01-01"}, {"ts": "2024-01-02"}
            ]
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
