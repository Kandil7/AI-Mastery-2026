"""
Model A/B Testing Module
========================
Infrastructure for running A/B tests on ML models.

Features:
- Traffic splitting
- Metrics collection per variant
- Statistical significance testing
- Automated winner selection

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from scipy import stats
import hashlib


@dataclass
class ModelVariant:
    """Represents a model variant in an A/B test."""
    name: str
    model: any  # The actual model object
    traffic_percentage: float  # 0.0 to 1.0
    metrics: Dict[str, List[float]] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {"predictions": [], "latencies": [], "errors": []}


class ABTest:
    """
    A/B test manager for ML models.
    
    Example:
        test = ABTest("churn_model_test")
        test.add_variant("control", model_v1, 0.5)
        test.add_variant("treatment", model_v2, 0.5)
        
        variant = test.get_variant(user_id)
        prediction = variant.model.predict(features)
        test.record_prediction(variant.name, prediction, actual_label)
        
        results = test.analyze()
    """
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.variants: Dict[str, ModelVariant] = {}
        self.started_at = datetime.now()
    
    def add_variant(self, name: str, model: any, traffic_percentage: float):
        """
        Add a model variant to the test.
        
        Args:
            name: Variant name (e.g., "control", "treatment")
            model: Model object
            traffic_percentage: Fraction of traffic (0.0-1.0)
        """
        self.variants[name] = ModelVariant(name, model, traffic_percentage)
    
    def get_variant(self, user_id: str) -> ModelVariant:
        """
        Assign user to a variant (consistent hashing).
        
        Args:
            user_id: User identifier
        
        Returns:
            Assigned variant
        """
        # Consistent hashing: same user always gets same variant
        hash_value = int(hashlib.md5(f"{user_id}:{self.test_name}".encode()).hexdigest(), 16)
        threshold = (hash_value % 100) / 100.0
        
        cumulative = 0.0
        for variant in self.variants.values():
            cumulative += variant.traffic_percentage
            if threshold < cumulative:
                return variant
        
        # Fallback to first variant
        return list(self.variants.values())[0]
    
    def record_prediction(self, variant_name: str, prediction: float, 
                         actual: Optional[float] = None, latency: Optional[float] = None):
        """
        Record prediction metrics.
        
        Args:
            variant_name: Name of variant
            prediction: Model prediction
            actual: Actual label (if available)
            latency: Prediction latency in ms
        """
        variant = self.variants[variant_name]
        variant.metrics["predictions"].append(prediction)
        
        if actual is not None:
            error = abs(prediction - actual)
            variant.metrics["errors"].append(error)
        
        if latency is not None:
            variant.metrics["latencies"].append(latency)
    
    def analyze(self, metric: str = "error") -> Dict:
        """
        Analyze A/B test results.
        
        Args:
            metric: Metric to analyze ("error" or "latency")
        
        Returns:
            Analysis results with statistical tests
        """
        if len(self.variants) != 2:
            return {"error": "A/B test requires exactly 2 variants"}
        
        control_name, treatment_name = list(self.variants.keys())
        control = self.variants[control_name]
        treatment = self.variants[treatment_name]
        
        control_data = np.array(control.metrics[metric + "s"])
        treatment_data = np.array(treatment.metrics[metric + "s"])
        
        if len(control_data) == 0 or len(treatment_data) == 0:
            return {"error": "Insufficient data for analysis"}
        
        # T-test for statistical significance
        t_stat, p_value = stats.ttest_ind(control_data, treatment_data)
        
        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            ((len(control_data) - 1) * np.var(control_data) + 
             (len(treatment_data) - 1) * np.var(treatment_data)) /
            (len(control_data) + len(treatment_data) - 2)
        )
        cohens_d = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std
        
        # Confidence intervals (95%)
        control_ci = stats.t.interval(
            0.95, len(control_data) - 1,
            loc=np.mean(control_data),
            scale=stats.sem(control_data)
        )
        treatment_ci = stats.t.interval(
            0.95, len(treatment_data) - 1,
            loc=np.mean(treatment_data),
            scale=stats.sem(treatment_data)
        )
        
        # Determine winner
        is_significant = p_value < 0.05
        if metric == "error":
            # Lower is better for error
            winner = treatment_name if np.mean(treatment_data) < np.mean(control_data) else control_name
            improvement = (np.mean(control_data) - np.mean(treatment_data)) / np.mean(control_data)
        else:
            # Lower is better for latency
            winner = treatment_name if np.mean(treatment_data) < np.mean(control_data) else control_name
            improvement = (np.mean(control_data) - np.mean(treatment_data)) / np.mean(control_data)
        
        return {
            "test_name": self.test_name,
            "control": {
                "name": control_name,
                "mean": float(np.mean(control_data)),
                "std": float(np.std(control_data)),
                "n": len(control_data),
                "ci_95": [float(control_ci[0]), float(control_ci[1])]
            },
            "treatment": {
                "name": treatment_name,
                "mean": float(np.mean(treatment_data)),
                "std": float(np.std(treatment_data)),
                "n": len(treatment_data),
                "ci_95": [float(treatment_ci[0]), float(treatment_ci[1])]
            },
            "statistical_test": {
                "t_statistic": float(t_stat),
                "p_value": float(p_value),
                "is_significant": is_significant,
                "cohens_d": float(cohens_d),
                "effect_size": "small" if abs(cohens_d) < 0.5 else ("medium" if abs(cohens_d) < 0.8 else "large")
            },
            "conclusion": {
                "winner": winner if is_significant else "inconclusive",
                "improvement_pct": float(improvement * 100),
                "recommendation": f"Deploy {winner}" if is_significant and improvement > 0.02 else "Continue testing"
            }
        }
    
    def get_summary(self) -> Dict:
        """Get test summary."""
        return {
            "test_name": self.test_name,
            "started_at": self.started_at.isoformat(),
            "variants": {
                name: {
                    "traffic": variant.traffic_percentage,
                    "samples": len(variant.metrics["predictions"])
                }
                for name, variant in self.variants.items()
            }
        }


# ============================================================
# USAGE EXAMPLE
# ============================================================

if __name__ == "__main__":
    from sklearn.linear_model import LogisticRegression
    import numpy as np
    
    # Create two model variants
    model_v1 = LogisticRegression()  # Control
    model_v2 = LogisticRegression()  # Treatment (better model)
    
    # Setup A/B test
    test = ABTest("conversion_model_v1_vs_v2")
    test.add_variant("control", model_v1, 0.5)
    test.add_variant("treatment", model_v2, 0.5)
    
    # Simulate traffic
    np.random.seed(42)
    for i in range(1000):
        user_id = f"user_{i}"
        
        # Get variant for this user
        variant = test.get_variant(user_id)
        
        # Simulate prediction (control has higher error)
        if variant.name == "control":
            prediction = np.random.rand()
            actual = np.random.rand() * 1.2  # Higher error
        else:
            prediction = np.random.rand()
            actual = np.random.rand() * 0.8  # Lower error
        
        latency = np.random.uniform(10, 50)  # ms
        
        # Record metrics
        test.record_prediction(variant.name, prediction, actual, latency)
    
    # Analyze results
    print("\n" + "="*60)
    print("A/B TEST RESULTS")
    print("="*60)
    
    summary = test.get_summary()
    print(f"\nTest: {summary['test_name']}")
    print(f"Started: {summary['started_at']}")
    
    results = test.analyze(metric="error")
    
    print(f"\nControl: {results['control']['name']}")
    print(f"  Mean Error: {results['control']['mean']:.4f} ± {results['control']['std']:.4f}")
    print(f"  95% CI: [{results['control']['ci_95'][0]:.4f}, {results['control']['ci_95'][1]:.4f}]")
    print(f"  Samples: {results['control']['n']}")
    
    print(f"\nTreatment: {results['treatment']['name']}")
    print(f"  Mean Error: {results['treatment']['mean']:.4f} ± {results['treatment']['std']:.4f}")
    print(f"  95% CI: [{results['treatment']['ci_95'][0]:.4f}, {results['treatment']['ci_95'][1]:.4f}]")
    print(f"  Samples: {results['treatment']['n']}")
    
    print(f"\nStatistical Test:")
    print(f"  P-value: {results['statistical_test']['p_value']:.6f}")
    print(f"  Significant: {results['statistical_test']['is_significant']}")
    print(f"  Effect Size (Cohen's d): {results['statistical_test']['cohens_d']:.4f} ({results['statistical_test']['effect_size']})")
    
    print(f"\nConclusion:")
    print(f"  Winner: {results['conclusion']['winner']}")
    print(f"  Improvement: {results['conclusion']['improvement_pct']:.2f}%")
    print(f"  Recommendation: {results['conclusion']['recommendation']}")
    
    print("\n" + "="*60)
    print("✅ A/B testing framework ready for production!")

