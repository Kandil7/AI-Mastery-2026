"""
Online-Offline Consistency Validation Module
===============================================

Validate consistency between online and offline features.

The "training-serving skew" is a major issue in ML systems.
This validator ensures that:
1. Features computed offline match those served online
2. Point-in-time correctness is maintained
3. Data drift is detected

Classes:
    ConsistencyValidator: Validate consistency between online/offline

Author: AI-Mastery-2026
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .online import OnlineFeatureServer
from .types import EntityFeatureMap, FeatureValue

logger = logging.getLogger(__name__)


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

    def validate(
        self, offline_value: Any, online_value: Any, feature_name: str
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate consistency between offline and online values.

        Returns:
            (is_consistent, error_message)
        """
        if offline_value is None and online_value is None:
            return True, None

        if offline_value is None or online_value is None:
            return (
                False,
                f"One value is None: offline={offline_value}, online={online_value}",
            )

        # Numeric comparison
        if isinstance(offline_value, (int, float)) and isinstance(
            online_value, (int, float)
        ):
            diff = abs(offline_value - online_value)
            max_val = max(abs(offline_value), abs(online_value), 1e-8)
            relative_diff = diff / max_val

            if relative_diff > self.tolerance:
                return (
                    False,
                    f"Values differ by {relative_diff:.4f} (tolerance: {self.tolerance})",
                )
            return True, None

        # Array comparison
        if isinstance(offline_value, np.ndarray) and isinstance(
            online_value, np.ndarray
        ):
            if offline_value.shape != online_value.shape:
                return (
                    False,
                    f"Shape mismatch: {offline_value.shape} vs {online_value.shape}",
                )

            diff = np.abs(offline_value - online_value)
            max_diff = np.max(diff)

            if max_diff > self.tolerance:
                return False, f"Max element difference: {max_diff:.4f}"
            return True, None

        # Exact comparison for other types
        if offline_value != online_value:
            return False, f"Values don't match: {offline_value} vs {online_value}"

        return True, None

    def validate_batch(
        self,
        offline_values: Dict[str, Dict[str, Any]],
        online_server: OnlineFeatureServer,
        entity_ids: List[str],
    ) -> Dict[str, Any]:
        """
        Validate a batch of features across entities.

        Args:
            offline_values: feature_name -> entity_id -> value
            online_server: Online feature server
            entity_ids: Entities to validate

        Returns:
            Validation report
        """
        report = {"total_checks": 0, "passed": 0, "failed": 0, "failures": []}

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
                    report["failures"].append(
                        {
                            "feature": feature_name,
                            "entity_id": entity_id,
                            "error": error,
                        }
                    )

        # Store result
        self.validation_results.append(report)

        return report

    def validate_single_feature(
        self,
        feature_name: str,
        offline_values: Dict[str, Any],
        online_server: OnlineFeatureServer,
    ) -> Dict[str, Any]:
        """
        Validate a single feature across all entities.

        Args:
            feature_name: Name of feature to validate
            offline_values: entity_id -> value
            online_server: Online feature server

        Returns:
            Validation report for this feature
        """
        report = {
            "feature_name": feature_name,
            "total_checks": 0,
            "passed": 0,
            "failed": 0,
            "failures": [],
        }

        for entity_id, offline_val in offline_values.items():
            online_val = online_server.get(feature_name, entity_id)

            # Handle FeatureValue wrapper
            if isinstance(offline_val, FeatureValue):
                offline_val = offline_val.value

            is_consistent, error = self.validate(offline_val, online_val, feature_name)

            report["total_checks"] += 1

            if is_consistent:
                report["passed"] += 1
            else:
                report["failed"] += 1
                report["failures"].append(
                    {
                        "entity_id": entity_id,
                        "error": error,
                    }
                )

        return report

    def get_validation_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get validation history."""
        return self.validation_results[-limit:]

    def get_drift_report(
        self, offline_values: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a data drift report across all features.

        Args:
            offline_values: feature_name -> entity_id -> value

        Returns:
            Drift report with statistics per feature
        """
        report = {}

        for feature_name, entity_values in offline_values.items():
            values = []
            for val in entity_values.values():
                if isinstance(val, FeatureValue):
                    val = val.value
                if isinstance(val, (int, float)):
                    values.append(val)

            if values:
                report[feature_name] = {
                    "count": len(values),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)),
                    "p99": float(np.percentile(values, 99)),
                }

        return report
