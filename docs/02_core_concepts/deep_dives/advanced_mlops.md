# Advanced MLOps Guide

Advanced MLOps patterns for production AI systems.

---

## 1. Drift Detection

### Data Drift Detection

```python
import numpy as np
from scipy import stats

class DriftDetector:
    """Detect statistical drift in input data"""
    
    def __init__(self, reference_data, threshold=0.05):
        self.reference = reference_data
        self.threshold = threshold
        self.reference_stats = self._compute_stats(reference_data)
    
    def _compute_stats(self, data):
        return {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'quantiles': np.percentile(data, [25, 50, 75], axis=0)
        }
    
    def detect_drift(self, new_data) -> dict:
        """Kolmogorov-Smirnov test for drift"""
        results = {'drifted': False, 'features': []}
        
        for i in range(new_data.shape[1]):
            stat, p_value = stats.ks_2samp(
                self.reference[:, i], 
                new_data[:, i]
            )
            if p_value < self.threshold:
                results['drifted'] = True
                results['features'].append({
                    'index': i,
                    'p_value': p_value,
                    'statistic': stat
                })
        
        return results

# Usage
detector = DriftDetector(X_train)
drift_result = detector.detect_drift(new_batch)
if drift_result['drifted']:
    alert(f"Data drift detected in features: {drift_result['features']}")
```

### Concept Drift (Model Performance)

```python
class ConceptDriftMonitor:
    """Monitor model performance over time"""
    
    def __init__(self, window_size=1000, threshold=0.1):
        self.window_size = window_size
        self.threshold = threshold
        self.predictions = []
        self.actuals = []
        self.baseline_accuracy = None
    
    def add_sample(self, prediction, actual):
        self.predictions.append(prediction)
        self.actuals.append(actual)
        
        # Keep rolling window
        if len(self.predictions) > self.window_size:
            self.predictions.pop(0)
            self.actuals.pop(0)
    
    def check_drift(self) -> dict:
        if len(self.predictions) < self.window_size:
            return {'drift': False, 'reason': 'Insufficient data'}
        
        current_accuracy = np.mean(
            np.array(self.predictions) == np.array(self.actuals)
        )
        
        if self.baseline_accuracy is None:
            self.baseline_accuracy = current_accuracy
            return {'drift': False, 'accuracy': current_accuracy}
        
        degradation = self.baseline_accuracy - current_accuracy
        
        if degradation > self.threshold:
            return {
                'drift': True,
                'baseline': self.baseline_accuracy,
                'current': current_accuracy,
                'degradation': degradation
            }
        
        return {'drift': False, 'accuracy': current_accuracy}
```

---

## 2. Model Versioning

### Version Tracking

```python
import hashlib
import json
from datetime import datetime

class ModelRegistry:
    """Track model versions and metadata"""
    
    def __init__(self, registry_path='models/registry.json'):
        self.registry_path = registry_path
        self.registry = self._load_registry()
    
    def _load_registry(self):
        try:
            with open(self.registry_path) as f:
                return json.load(f)
        except FileNotFoundError:
            return {'models': {}}
    
    def register_model(self, name, model_path, metrics, tags=None):
        version = self._next_version(name)
        model_hash = self._compute_hash(model_path)
        
        entry = {
            'version': version,
            'path': model_path,
            'hash': model_hash,
            'metrics': metrics,
            'tags': tags or [],
            'created_at': datetime.now().isoformat(),
            'stage': 'staging'
        }
        
        if name not in self.registry['models']:
            self.registry['models'][name] = []
        
        self.registry['models'][name].append(entry)
        self._save_registry()
        
        return f"{name}:{version}"
    
    def promote_to_production(self, name, version):
        """Promote model version to production"""
        for model in self.registry['models'].get(name, []):
            if model['version'] == version:
                model['stage'] = 'production'
            elif model['stage'] == 'production':
                model['stage'] = 'archived'
        
        self._save_registry()
    
    def get_production_model(self, name):
        for model in self.registry['models'].get(name, []):
            if model['stage'] == 'production':
                return model
        return None
```

### Git-based Versioning

```bash
# Tag model versions with git
git tag -a model-v1.0.0 -m "Classification model v1.0.0, accuracy 92%"
git push origin model-v1.0.0

# DVC for large model files
dvc add models/classification_model.joblib
git add models/classification_model.joblib.dvc
git commit -m "Add model v1.0.0"
```

---

## 3. A/B Testing

```python
import random
from collections import defaultdict

class ABTestManager:
    """Manage A/B tests for model comparison"""
    
    def __init__(self):
        self.experiments = {}
        self.results = defaultdict(lambda: {'a': [], 'b': []})
    
    def create_experiment(self, name, model_a, model_b, traffic_split=0.5):
        self.experiments[name] = {
            'model_a': model_a,
            'model_b': model_b,
            'traffic_split': traffic_split,
            'active': True
        }
    
    def get_model(self, experiment_name, user_id=None):
        exp = self.experiments[experiment_name]
        
        # Deterministic assignment based on user_id
        if user_id:
            random.seed(hash(user_id))
        
        if random.random() < exp['traffic_split']:
            return 'a', exp['model_a']
        else:
            return 'b', exp['model_b']
    
    def record_result(self, experiment_name, variant, success):
        self.results[experiment_name][variant].append(success)
    
    def get_statistics(self, experiment_name):
        a_results = self.results[experiment_name]['a']
        b_results = self.results[experiment_name]['b']
        
        a_rate = sum(a_results) / len(a_results) if a_results else 0
        b_rate = sum(b_results) / len(b_results) if b_results else 0
        
        # Statistical significance test
        from scipy.stats import chi2_contingency
        
        table = [[sum(a_results), len(a_results) - sum(a_results)],
                 [sum(b_results), len(b_results) - sum(b_results)]]
        
        chi2, p_value, _, _ = chi2_contingency(table)
        
        return {
            'model_a': {'rate': a_rate, 'samples': len(a_results)},
            'model_b': {'rate': b_rate, 'samples': len(b_results)},
            'p_value': p_value,
            'significant': p_value < 0.05
        }
```

---

## 4. Shadow Mode Deployment

```python
class ShadowDeployment:
    """Run new model in shadow mode alongside production"""
    
    def __init__(self, production_model, shadow_model):
        self.production = production_model
        self.shadow = shadow_model
        self.comparisons = []
    
    async def predict(self, input_data):
        # Production prediction (returned to user)
        prod_result = self.production.predict(input_data)
        
        # Shadow prediction (logged only)
        try:
            shadow_result = self.shadow.predict(input_data)
            self._log_comparison(input_data, prod_result, shadow_result)
        except Exception as e:
            logger.error(f"Shadow model failed: {e}")
        
        return prod_result
    
    def _log_comparison(self, input_data, prod_result, shadow_result):
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'input_hash': hash(str(input_data)),
            'production': prod_result,
            'shadow': shadow_result,
            'match': prod_result == shadow_result
        }
        self.comparisons.append(comparison)
    
    def get_agreement_rate(self):
        if not self.comparisons:
            return None
        matches = sum(1 for c in self.comparisons if c['match'])
        return matches / len(self.comparisons)
```

---

## 5. Feature Store

```python
from datetime import datetime
import pandas as pd

class FeatureStore:
    """Store and retrieve precomputed features"""
    
    def __init__(self, storage_path='features/'):
        self.storage_path = storage_path
        self.feature_definitions = {}
    
    def register_feature(self, name, compute_fn, description):
        self.feature_definitions[name] = {
            'compute': compute_fn,
            'description': description,
            'created_at': datetime.now().isoformat()
        }
    
    def compute_and_store(self, feature_name, entity_ids, raw_data):
        compute_fn = self.feature_definitions[feature_name]['compute']
        features = compute_fn(raw_data)
        
        df = pd.DataFrame({
            'entity_id': entity_ids,
            feature_name: features,
            'computed_at': datetime.now()
        })
        
        path = f"{self.storage_path}/{feature_name}.parquet"
        df.to_parquet(path)
        return path
    
    def get_features(self, feature_names, entity_ids):
        result = pd.DataFrame({'entity_id': entity_ids})
        
        for name in feature_names:
            path = f"{self.storage_path}/{name}.parquet"
            feature_df = pd.read_parquet(path)
            result = result.merge(feature_df, on='entity_id', how='left')
        
        return result
```

---

## 6. Automated Retraining Pipeline

```python
class RetrainingPipeline:
    """Automated model retraining based on triggers"""
    
    def __init__(self, model_class, training_config):
        self.model_class = model_class
        self.config = training_config
        self.drift_detector = DriftDetector(training_config['reference_data'])
        self.performance_monitor = ConceptDriftMonitor()
    
    def should_retrain(self, new_data, predictions, actuals):
        # Check data drift
        drift = self.drift_detector.detect_drift(new_data)
        if drift['drifted']:
            return True, 'data_drift'
        
        # Check performance drift
        for pred, actual in zip(predictions, actuals):
            self.performance_monitor.add_sample(pred, actual)
        
        perf = self.performance_monitor.check_drift()
        if perf['drift']:
            return True, 'performance_drift'
        
        # Check time-based trigger
        if self._days_since_last_training() > self.config['max_age_days']:
            return True, 'scheduled'
        
        return False, None
    
    def retrain(self, training_data):
        model = self.model_class(**self.config['model_params'])
        X, y = training_data
        
        # Train
        model.fit(X, y)
        
        # Validate
        metrics = self._evaluate(model, self.config['validation_data'])
        
        if metrics['accuracy'] >= self.config['min_accuracy']:
            # Register and deploy
            registry.register_model('model', model, metrics)
            return True, metrics
        
        return False, metrics
```

---

## Quick Reference

| Pattern | Use When |
|---------|----------|
| **Drift Detection** | Monitor production data quality |
| **A/B Testing** | Compare model versions |
| **Shadow Mode** | Validate before production |
| **Feature Store** | Share features across models |
| **Auto Retraining** | Maintain model freshness |
