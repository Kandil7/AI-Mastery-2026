"""
Batch Feature Pipeline Module
==============================

Batch feature computation pipeline.

Runs periodically (e.g., daily) to compute features for all entities.

Classes:
    BatchPipeline: Batch feature computation pipeline

Author: AI-Mastery-2026
"""

import logging
from collections import defaultdict
from typing import Any, Callable, Dict, List

from .registry import FeatureRegistry
from .transforms import FeatureTransform
from .types import FeatureValue

logger = logging.getLogger(__name__)


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

    def compute_feature(
        self, feature_name: str, entity_id: str, raw_data: Dict[str, Any]
    ) -> FeatureValue:
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
                    dependencies[dep_name] = self.computed_values[dep_name][
                        entity_id
                    ].value

        # Compute the feature
        value = transform.compute(entity_id, raw_data, dependencies)

        from datetime import datetime

        feature_value = FeatureValue(
            feature_name=feature_name,
            entity_id=entity_id,
            value=value,
            timestamp=datetime.now(),
            version=definition.version,
        )

        # Cache the computed value
        self.computed_values[feature_name][entity_id] = feature_value

        return feature_value

    def compute_all(
        self, entity_ids: List[str], raw_data_fn: Callable[[str], Dict[str, Any]]
    ) -> Dict[str, Dict[str, FeatureValue]]:
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

    def get_computed_values(
        self, feature_name: str, entity_id: str
    ) -> Optional[FeatureValue]:
        """Get computed feature value."""
        return self.computed_values.get(feature_name, {}).get(entity_id)

    def clear_cache(self) -> None:
        """Clear computed values cache."""
        self.computed_values.clear()
        logger.info("Cleared batch pipeline cache")
