"""
Streaming Feature Pipeline Module
==================================

Near-real-time feature computation pipeline.

This is an interface/stub implementation. In production,
this would integrate with Kafka/Samza for real streaming.

Classes:
    StreamingPipeline: Basic streaming pipeline
    GigascaleStreamingPipeline: High-performance streaming (DoorDash pattern)
    FeatureFreshnessTracker: Track staleness of features

Author: AI-Mastery-2026
"""

import logging
import threading
import time
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .registry import FeatureRegistry
from .transforms import FeatureTransform
from .types import FeatureValue, SerializationFormat, StreamingFeatureConfig

logger = logging.getLogger(__name__)


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
        self, feature_name: str, entity_id: str, max_staleness_seconds: float
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
                self._freshness_stats[feature_name] = self._freshness_stats[
                    feature_name
                ][-1000:]

        if not is_fresh:
            self._staleness_alerts.append(
                {
                    "feature_name": feature_name,
                    "entity_id": entity_id,
                    "staleness_seconds": staleness,
                    "threshold_seconds": max_staleness_seconds,
                    "timestamp": datetime.now().isoformat(),
                }
            )

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
            "max_staleness_seconds": float(np.max(stats)),
        }

    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent staleness alerts."""
        return self._staleness_alerts[-limit:]

    def clear_alerts(self) -> int:
        """Clear alerts and return count."""
        count = len(self._staleness_alerts)
        self._staleness_alerts = []
        return count


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
                    version=transform.feature_definition.version,
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
        serialization: SerializationFormat = SerializationFormat.MSGPACK,
    ):
        self.registry = registry
        self.serialization = serialization
        self.transforms: Dict[str, FeatureTransform] = {}
        self.configs: Dict[str, StreamingFeatureConfig] = {}
        self.freshness_tracker = FeatureFreshnessTracker()

        # Window buffers for aggregations
        self._window_buffers: Dict[str, Dict[str, List[Tuple[datetime, Any]]]] = (
            defaultdict(lambda: defaultdict(list))
        )

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
        config: Optional[StreamingFeatureConfig] = None,
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
                config = self.configs.get(
                    name, StreamingFeatureConfig(feature_name=name)
                )

                # Add to window buffer
                value = transform.compute(entity_id, event, {})
                self._window_buffers[name][entity_id].append((event_time, value))

                # Clean old entries from window
                window_start = event_time - timedelta(seconds=config.window_seconds)
                self._window_buffers[name][entity_id] = [
                    (t, v)
                    for t, v in self._window_buffers[name][entity_id]
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
                    version=transform.feature_definition.version,
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
        import json

        if self.serialization == SerializationFormat.JSON:
            return json.dumps(value).encode("utf-8")
        elif self.serialization == SerializationFormat.MSGPACK:
            # Simulated msgpack (would use msgpack library in production)
            return json.dumps(value).encode("utf-8")  # Fallback
        elif self.serialization == SerializationFormat.PROTOBUF:
            # Simulated protobuf (would use generated proto classes in production)
            return json.dumps(value).encode("utf-8")  # Fallback
        else:
            return str(value).encode("utf-8")

    def get_latest(self, feature_name: str, entity_id: str) -> Optional[FeatureValue]:
        """Get latest value (read path)."""
        return self._latest_values.get(feature_name, {}).get(entity_id)

    def get_fresh_value(
        self,
        feature_name: str,
        entity_id: str,
        max_staleness_seconds: Optional[float] = None,
    ) -> Tuple[Optional[FeatureValue], bool]:
        """
        Get value only if fresh enough.

        Returns:
            (value, is_fresh)
        """
        config = self.configs.get(
            feature_name, StreamingFeatureConfig(feature_name=feature_name)
        )
        threshold = max_staleness_seconds or config.max_staleness_seconds

        is_fresh, _ = self.freshness_tracker.check_freshness(
            feature_name, entity_id, threshold
        )

        if not is_fresh:
            return None, False

        return self.get_latest(feature_name, entity_id), True

    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline performance statistics."""
        serialize_times = (
            self._serialization_time_ms[-1000:] if self._serialization_time_ms else [0]
        )

        return {
            "events_processed": self._events_processed,
            "features_computed": self._features_computed,
            "pending_events": len(self._event_queue),
            "serialization_format": self.serialization.value,
            "avg_serialize_ms": float(np.mean(serialize_times)),
            "p99_serialize_ms": (
                float(np.percentile(serialize_times, 99))
                if len(serialize_times) > 1
                else 0
            ),
            "freshness_alerts": len(self.freshness_tracker.get_alerts()),
        }
