"""
Feature Registry Module
========================

Central registry for all feature definitions.

Provides:
- Feature discovery and documentation
- Version tracking
- Dependency management
- Lineage tracking

Classes:
    FeatureRegistry: Central registry for feature definitions

Author: AI-Mastery-2026
"""

import logging
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

from .types import (
    ComputationType,
    FeatureDefinition,
    FeatureGroup,
)

logger = logging.getLogger(__name__)


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

    def get_feature(
        self, name: str, version: Optional[str] = None
    ) -> Optional[FeatureDefinition]:
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

    def list_features(
        self,
        entity_key: Optional[str] = None,
        computation_type: Optional[ComputationType] = None,
        tags: Optional[List[str]] = None,
    ) -> List[FeatureDefinition]:
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

    def get_feature_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get detailed info about a feature."""
        feature = self.get_feature(name)
        if not feature:
            return None

        return {
            "definition": feature.to_dict(),
            "dependencies": self.get_dependencies(name),
            "versions": self.versions.get(name, []),
        }

    def list_groups(self) -> List[FeatureGroup]:
        """List all feature groups."""
        return list(self.groups.values())

    def get_group(self, name: str) -> Optional[FeatureGroup]:
        """Get a feature group by name."""
        return self.groups.get(name)
