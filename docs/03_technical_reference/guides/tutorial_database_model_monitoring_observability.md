# Database Integration with AI/ML Model Monitoring and Observability Tutorial

## Overview

This tutorial focuses on integrating databases with AI/ML model monitoring and observability systems. We'll cover real-time model performance monitoring, data drift detection, concept drift detection, model explainability tracking, and production observability specifically for senior AI/ML engineers building robust ML systems.

## Prerequisites
- Python 3.8+
- Prometheus, Grafana, or similar monitoring tools
- PostgreSQL/MySQL with proper indexing for time-series data
- MLflow, Evidently AI, or similar observability tools
- Basic understanding of ML monitoring concepts

## Tutorial Structure
1. **Real-time Model Performance Monitoring** - Latency, throughput, error rates
2. **Data Drift Detection** - Statistical drift detection with database storage
3. **Concept Drift Detection** - Model performance drift detection
4. **Model Explainability Tracking** - SHAP, LIME integration with database
5. **Production Observability** - Comprehensive monitoring dashboards
6. **Alerting and Anomaly Detection** - Proactive issue detection
7. **Performance Benchmarking** - Monitoring overhead analysis

## Section 1: Real-time Model Performance Monitoring

### Step 1: Database-backed model metrics storage
```python
import psycopg2
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List

class ModelPerformanceMonitor:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_monitoring_tables(self):
        """Create tables for model performance monitoring"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Model metrics (time-series)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_metrics (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            metric_value FLOAT NOT NULL,
            metric_unit VARCHAR(50),
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        # Prediction logs
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            user_id VARCHAR(255),
            input_data_hash VARCHAR(64),
            prediction_result JSONB,
            confidence FLOAT,
            latency_ms INTEGER NOT NULL,
            status VARCHAR(20) NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        # Model health checks
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_health_checks (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            status VARCHAR(20) NOT NULL,
            health_score FLOAT,
            issues JSONB,
            checked_at TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "Monitoring tables created successfully"
    
    def log_model_metric(self, model_name: str, version: str, environment: str,
                        metric_name: str, metric_value: float, 
                        metric_unit: str = "", timestamp: datetime = None):
        """Log model metric to database"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO model_metrics (
            model_name, version, environment, metric_name, metric_value,
            metric_unit, timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            model_name, version, environment, metric_name, metric_value,
            metric_unit, timestamp
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    
    def log_prediction(self, model_name: str, version: str, environment: str,
                      user_id: str = None, input_data_hash: str = "",
                      prediction_result: Dict = None, confidence: float = 0.0,
                      latency_ms: int = 0, status: str = "success",
                      timestamp: datetime = None):
        """Log prediction to database"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO prediction_logs (
            model_name, version, environment, user_id, input_data_hash,
            prediction_result, confidence, latency_ms, status, timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            model_name, version, environment, user_id, input_data_hash,
            json.dumps(prediction_result) if prediction_result else None,
            confidence, latency_ms, status, timestamp
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    
    def log_health_check(self, model_name: str, version: str, environment: str,
                        status: str, health_score: float = 1.0,
                        issues: List[Dict] = None, timestamp: datetime = None):
        """Log model health check"""
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO model_health_checks (
            model_name, version, environment, status, health_score,
            issues, checked_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(insert_query, (
            model_name, version, environment, status, health_score,
            json.dumps(issues) if issues else None, timestamp
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True

# Usage example
monitor = ModelPerformanceMonitor(db_config)

# Create monitoring tables
monitor.create_monitoring_tables()

# Log model metrics
monitor.log_model_metric(
    model_name="user_engagement_predictor",
    version="v1.2.3",
    environment="production",
    metric_name="prediction_latency",
    metric_value=120.5,
    metric_unit="ms"
)

monitor.log_model_metric(
    model_name="user_engagement_predictor",
    version="v1.2.3",
    environment="production",
    metric_name="requests_per_second",
    metric_value=45.2,
    metric_unit="req/s"
)

# Log prediction
monitor.log_prediction(
    model_name="user_engagement_predictor",
    version="v1.2.3",
    environment="production",
    user_id="123",
    input_data_hash="abc123def456",
    prediction_result={"prediction": 0.85, "class": "high_engagement"},
    confidence=0.92,
    latency_ms=120,
    status="success"
)

# Log health check
monitor.log_health_check(
    model_name="user_engagement_predictor",
    version="v1.2.3",
    environment="production",
    status="healthy",
    health_score=0.95,
    issues=[{"severity": "warning", "message": "Latency increased by 10%"}]
)
```

### Step 2: Real-time monitoring dashboard queries
```python
class MonitoringDashboardQueries:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def get_recent_metrics(self, model_name: str, hours: int = 1):
        """Get recent metrics for a model"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        query = """
        SELECT 
            metric_name,
            metric_value,
            metric_unit,
            timestamp
        FROM model_metrics
        WHERE model_name = %s 
        AND timestamp >= NOW() - INTERVAL '%s hours'
        ORDER BY timestamp DESC
        LIMIT 100
        """
        
        cursor.execute(query, (model_name, hours))
        results = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                'metric_name': row[0],
                'metric_value': row[1],
                'metric_unit': row[2],
                'timestamp': row[3].isoformat()
            }
            for row in results
        ]
    
    def get_prediction_summary(self, model_name: str, hours: int = 1):
        """Get prediction summary statistics"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        query = """
        SELECT 
            COUNT(*) as total_predictions,
            COUNT(CASE WHEN status = 'success' THEN 1 END) as successful_predictions,
            COUNT(CASE WHEN status = 'error' THEN 1 END) as error_predictions,
            AVG(latency_ms) as avg_latency_ms,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) as p95_latency_ms,
            MIN(latency_ms) as min_latency_ms,
            MAX(latency_ms) as max_latency_ms
        FROM prediction_logs
        WHERE model_name = %s 
        AND timestamp >= NOW() - INTERVAL '%s hours'
        """
        
        cursor.execute(query, (model_name, hours))
        result = cursor.fetchone()
        
        conn.close()
        
        return {
            'total_predictions': result[0],
            'successful_predictions': result[1],
            'error_predictions': result[2],
            'avg_latency_ms': result[3],
            'p95_latency_ms': result[4],
            'min_latency_ms': result[5],
            'max_latency_ms': result[6]
        }
    
    def get_model_health_trend(self, model_name: str, days: int = 7):
        """Get model health trend over time"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        query = """
        SELECT 
            DATE(checked_at) as date,
            AVG(health_score) as avg_health_score,
            COUNT(CASE WHEN status = 'healthy' THEN 1 END) as healthy_count,
            COUNT(*) as total_checks
        FROM model_health_checks
        WHERE model_name = %s 
        AND checked_at >= NOW() - INTERVAL '%s days'
        GROUP BY DATE(checked_at)
        ORDER BY date
        """
        
        cursor.execute(query, (model_name, days))
        results = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                'date': row[0].isoformat(),
                'avg_health_score': row[1],
                'healthy_count': row[2],
                'total_checks': row[3]
            }
            for row in results
        ]

# Usage example
dashboard_queries = MonitoringDashboardQueries(db_config)

# Get recent metrics
recent_metrics = dashboard_queries.get_recent_metrics("user_engagement_predictor", hours=1)
print(f"Recent metrics: {recent_metrics}")

# Get prediction summary
summary = dashboard_queries.get_prediction_summary("user_engagement_predictor", hours=1)
print(f"Prediction summary: {summary}")

# Get health trend
health_trend = dashboard_queries.get_model_health_trend("user_engagement_predictor", days=7)
print(f"Health trend: {health_trend}")
```

## Section 2: Data Drift Detection

### Step 1: Statistical drift detection with database storage
```python
import numpy as np
from scipy import stats
from typing import List, Dict, Optional

class DataDriftDetector:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_drift_tables(self):
        """Create tables for drift detection"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Drift detection results
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS data_drift_results (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            feature_name VARCHAR(100) NOT NULL,
            reference_distribution JSONB,
            current_distribution JSONB,
            statistical_test VARCHAR(50) NOT NULL,
            test_statistic FLOAT,
            p_value FLOAT,
            drift_detected BOOLEAN NOT NULL,
            threshold FLOAT,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        # Drift alerts
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS drift_alerts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            drift_result_id UUID REFERENCES data_drift_results(id),
            severity VARCHAR(20) NOT NULL,
            message TEXT NOT NULL,
            acknowledged BOOLEAN DEFAULT FALSE,
            acknowledged_at TIMESTAMP,
            acknowledged_by VARCHAR(255),
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "Drift detection tables created successfully"
    
    def calculate_drift(self, reference_data: List[float], current_data: List[float],
                       feature_name: str, test_type: str = "ks") -> Dict:
        """Calculate drift between reference and current data"""
        if test_type == "ks":
            # Kolmogorov-Smirnov test
            stat, p_value = stats.ks_2samp(reference_data, current_data)
            threshold = 0.05
            drift_detected = p_value < threshold
        elif test_type == "chi2":
            # Chi-square test (for categorical data)
            # Simplified implementation
            stat, p_value = stats.chisquare(
                np.histogram(current_data, bins=10)[0],
                np.histogram(reference_data, bins=10)[0]
            )
            threshold = 0.05
            drift_detected = p_value < threshold
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        
        return {
            'statistic': stat,
            'p_value': p_value,
            'threshold': threshold,
            'drift_detected': drift_detected,
            'feature_name': feature_name,
            'test_type': test_type
        }
    
    def log_drift_result(self, model_name: str, version: str, environment: str,
                        feature_name: str, reference_data: List[float],
                        current_data: List[float], test_type: str = "ks"):
        """Log drift detection result to database"""
        drift_info = self.calculate_drift(reference_data, current_data, feature_name, test_type)
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO data_drift_results (
            model_name, version, environment, feature_name,
            reference_distribution, current_distribution,
            statistical_test, test_statistic, p_value,
            drift_detected, threshold, timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        RETURNING id
        """
        
        cursor.execute(insert_query, (
            model_name, version, environment, feature_name,
            json.dumps({
                'mean': np.mean(reference_data),
                'std': np.std(reference_data),
                'count': len(reference_data),
                'histogram': list(np.histogram(reference_data, bins=10)[0])
            }),
            json.dumps({
                'mean': np.mean(current_data),
                'std': np.std(current_data),
                'count': len(current_data),
                'histogram': list(np.histogram(current_data, bins=10)[0])
            }),
            drift_info['test_type'],
            drift_info['statistic'],
            drift_info['p_value'],
            drift_info['drift_detected'],
            drift_info['threshold']
        ))
        
        result_id = cursor.fetchone()[0]
        
        # Create alert if drift detected
        if drift_info['drift_detected']:
            alert_query = """
            INSERT INTO drift_alerts (
                drift_result_id, severity, message, created_at
            ) VALUES (%s, %s, %s, NOW())
            """
            
            severity = "high" if drift_info['p_value'] < 0.01 else "medium"
            message = f"Data drift detected for feature '{feature_name}': p-value={drift_info['p_value']:.4f}"
            
            cursor.execute(alert_query, (result_id, severity, message))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return drift_info
    
    def get_recent_drift_alerts(self, model_name: str, hours: int = 24):
        """Get recent drift alerts"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        query = """
        SELECT 
            d.severity,
            d.message,
            dr.feature_name,
            dr.p_value,
            dr.drift_detected,
            d.created_at
        FROM drift_alerts d
        JOIN data_drift_results dr ON d.drift_result_id = dr.id
        WHERE dr.model_name = %s 
        AND d.created_at >= NOW() - INTERVAL '%s hours'
        ORDER BY d.created_at DESC
        """
        
        cursor.execute(query, (model_name, hours))
        results = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                'severity': row[0],
                'message': row[1],
                'feature_name': row[2],
                'p_value': row[3],
                'drift_detected': row[4],
                'created_at': row[5].isoformat()
            }
            for row in results
        ]

# Usage example
drift_detector = DataDriftDetector(db_config)

# Create drift tables
drift_detector.create_drift_tables()

# Generate sample data
reference_data = np.random.normal(0.5, 0.1, 1000).tolist()
current_data = np.random.normal(0.6, 0.15, 1000).tolist()

# Detect and log drift
drift_result = drift_detector.log_drift_result(
    model_name="user_engagement_predictor",
    version="v1.2.3",
    environment="production",
    feature_name="engagement_score",
    reference_data=reference_data,
    current_data=current_data,
    test_type="ks"
)

print(f"Drift detection result: {drift_result}")

# Get recent drift alerts
alerts = drift_detector.get_recent_drift_alerts("user_engagement_predictor", hours=24)
print(f"Recent drift alerts: {alerts}")
```

### Step 2: Feature importance drift detection
```python
class FeatureImportanceDriftDetector:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_feature_importance_tables(self):
        """Create tables for feature importance tracking"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Feature importance history
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_importance_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            feature_name VARCHAR(100) NOT NULL,
            importance_value FLOAT NOT NULL,
            importance_rank INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        # Feature importance drift
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_importance_drift (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            feature_name VARCHAR(100) NOT NULL,
            previous_importance FLOAT,
            current_importance FLOAT,
            importance_change FLOAT,
            rank_change INTEGER,
            drift_detected BOOLEAN NOT NULL,
            threshold FLOAT,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "Feature importance tables created successfully"
    
    def log_feature_importance(self, model_name: str, version: str, environment: str,
                             feature_importances: Dict[str, float]):
        """Log feature importance values"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Get current timestamp
        timestamp = datetime.utcnow()
        
        # Insert feature importance values
        for i, (feature_name, importance) in enumerate(sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)):
            insert_query = """
            INSERT INTO feature_importance_history (
                model_name, version, environment, feature_name,
                importance_value, importance_rank, timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                model_name, version, environment, feature_name,
                importance, i + 1, timestamp
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    
    def detect_feature_importance_drift(self, model_name: str, version: str, environment: str,
                                      current_importances: Dict[str, float],
                                      reference_version: str = None):
        """Detect drift in feature importance"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Get reference feature importance (previous version or latest)
        if reference_version:
            query = """
            SELECT feature_name, importance_value
            FROM feature_importance_history
            WHERE model_name = %s AND version = %s AND environment = %s
            AND timestamp = (
                SELECT MAX(timestamp) 
                FROM feature_importance_history 
                WHERE model_name = %s AND version = %s AND environment = %s
            )
            """
            cursor.execute(query, (model_name, reference_version, environment, model_name, reference_version, environment))
        else:
            query = """
            SELECT feature_name, importance_value
            FROM feature_importance_history
            WHERE model_name = %s AND environment = %s
            AND timestamp = (
                SELECT MAX(timestamp) 
                FROM feature_importance_history 
                WHERE model_name = %s AND environment = %s
            )
            """
            cursor.execute(query, (model_name, environment, model_name, environment))
        
        reference_importances = {row[0]: row[1] for row in cursor.fetchall()}
        
        # Calculate drift for each feature
        drift_results = []
        threshold = 0.1  # 10% change threshold
        
        for feature_name, current_importance in current_importances.items():
            reference_importance = reference_importances.get(feature_name, 0.0)
            importance_change = abs(current_importance - reference_importance)
            drift_detected = importance_change > threshold
            
            drift_results.append({
                'feature_name': feature_name,
                'previous_importance': reference_importance,
                'current_importance': current_importance,
                'importance_change': importance_change,
                'drift_detected': drift_detected,
                'threshold': threshold
            })
        
        # Log drift results
        for result in drift_results:
            insert_query = """
            INSERT INTO feature_importance_drift (
                model_name, version, environment, feature_name,
                previous_importance, current_importance, importance_change,
                drift_detected, threshold, timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """
            
            cursor.execute(insert_query, (
                model_name, version, environment, result['feature_name'],
                result['previous_importance'], result['current_importance'],
                result['importance_change'], result['drift_detected'],
                result['threshold']
            ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return drift_results

# Usage example
feature_drift_detector = FeatureImportanceDriftDetector(db_config)

# Create feature importance tables
feature_drift_detector.create_feature_importance_tables()

# Log initial feature importance
initial_importances = {
    "engagement_score": 0.45,
    "session_count": 0.30,
    "age": 0.15,
    "user_id": 0.10
}
feature_drift_detector.log_feature_importance(
    model_name="user_engagement_predictor",
    version="v1.2.3",
    environment="production",
    feature_importances=initial_importances
)

# Simulate new feature importance (with drift)
new_importances = {
    "engagement_score": 0.35,
    "session_count": 0.40,
    "age": 0.10,
    "user_id": 0.15
}
drift_results = feature_drift_detector.detect_feature_importance_drift(
    model_name="user_engagement_predictor",
    version="v1.2.4",
    environment="production",
    current_importances=new_importances,
    reference_version="v1.2.3"
)

print(f"Feature importance drift results: {drift_results}")
```

## Section 3: Concept Drift Detection

### Step 1: Model performance drift detection
```python
class ConceptDriftDetector:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_concept_drift_tables(self):
        """Create tables for concept drift detection"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Model performance history
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance_history (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            metric_value FLOAT NOT NULL,
            metric_unit VARCHAR(50),
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        # Concept drift detection results
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS concept_drift_results (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            metric_name VARCHAR(100) NOT NULL,
            reference_window_start TIMESTAMP NOT NULL,
            reference_window_end TIMESTAMP NOT NULL,
            current_window_start TIMESTAMP NOT NULL,
            current_window_end TIMESTAMP NOT NULL,
            reference_mean FLOAT,
            current_mean FLOAT,
            mean_difference FLOAT,
            z_score FLOAT,
            drift_detected BOOLEAN NOT NULL,
            threshold FLOAT,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "Concept drift tables created successfully"
    
    def calculate_concept_drift(self, reference_values: List[float], current_values: List[float],
                               metric_name: str, alpha: float = 0.05) -> Dict:
        """Calculate concept drift using statistical tests"""
        # Calculate means
        ref_mean = np.mean(reference_values)
        curr_mean = np.mean(current_values)
        
        # Calculate standard deviations
        ref_std = np.std(reference_values)
        curr_std = np.std(current_values)
        
        # Calculate pooled standard deviation
        n1 = len(reference_values)
        n2 = len(current_values)
        pooled_std = np.sqrt(((n1 - 1) * ref_std**2 + (n2 - 1) * curr_std**2) / (n1 + n2 - 2))
        
        # Calculate t-statistic
        t_stat = (ref_mean - curr_mean) / (pooled_std * np.sqrt(1/n1 + 1/n2))
        
        # Calculate critical value for two-tailed test
        df = n1 + n2 - 2
        critical_value = stats.t.ppf(1 - alpha/2, df)
        
        # Determine if drift detected
        drift_detected = abs(t_stat) > critical_value
        
        return {
            'reference_mean': ref_mean,
            'current_mean': curr_mean,
            'mean_difference': ref_mean - curr_mean,
            't_statistic': t_stat,
            'critical_value': critical_value,
            'drift_detected': drift_detected,
            'alpha': alpha,
            'metric_name': metric_name
        }
    
    def log_concept_drift(self, model_name: str, version: str, environment: str,
                         metric_name: str, reference_values: List[float],
                         current_values: List[float], 
                         reference_window_start: datetime,
                         reference_window_end: datetime,
                         current_window_start: datetime,
                         current_window_end: datetime):
        """Log concept drift detection result"""
        drift_info = self.calculate_concept_drift(reference_values, current_values, metric_name)
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO concept_drift_results (
            model_name, version, environment, metric_name,
            reference_window_start, reference_window_end,
            current_window_start, current_window_end,
            reference_mean, current_mean, mean_difference,
            z_score, drift_detected, threshold, timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """
        
        cursor.execute(insert_query, (
            model_name, version, environment, metric_name,
            reference_window_start, reference_window_end,
            current_window_start, current_window_end,
            drift_info['reference_mean'], drift_info['current_mean'],
            drift_info['mean_difference'], drift_info['t_statistic'],
            drift_info['drift_detected'], drift_info['alpha']
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return drift_info
    
    def get_concept_drift_alerts(self, model_name: str, days: int = 7):
        """Get concept drift alerts"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        query = """
        SELECT 
            metric_name,
            reference_mean,
            current_mean,
            mean_difference,
            drift_detected,
            timestamp
        FROM concept_drift_results
        WHERE model_name = %s 
        AND timestamp >= NOW() - INTERVAL '%s days'
        AND drift_detected = TRUE
        ORDER BY timestamp DESC
        """
        
        cursor.execute(query, (model_name, days))
        results = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                'metric_name': row[0],
                'reference_mean': row[1],
                'current_mean': row[2],
                'mean_difference': row[3],
                'drift_detected': row[4],
                'timestamp': row[5].isoformat()
            }
            for row in results
        ]

# Usage example
concept_drift_detector = ConceptDriftDetector(db_config)

# Create concept drift tables
concept_drift_detector.create_concept_drift_tables()

# Generate sample performance data
reference_accuracy = np.random.normal(0.92, 0.01, 100).tolist()
current_accuracy = np.random.normal(0.88, 0.02, 100).tolist()

# Log concept drift
drift_info = concept_drift_detector.log_concept_drift(
    model_name="user_engagement_predictor",
    version="v1.2.3",
    environment="production",
    metric_name="accuracy",
    reference_values=reference_accuracy,
    current_values=current_accuracy,
    reference_window_start=datetime.utcnow() - timedelta(days=7),
    reference_window_end=datetime.utcnow() - timedelta(days=1),
    current_window_start=datetime.utcnow() - timedelta(hours=24),
    current_window_end=datetime.utcnow()
)

print(f"Concept drift detection result: {drift_info}")

# Get concept drift alerts
alerts = concept_drift_detector.get_concept_drift_alerts("user_engagement_predictor", days=7)
print(f"Concept drift alerts: {alerts}")
```

## Section 4: Model Explainability Tracking

### Step 1: SHAP and LIME integration with database
```python
import shap
import numpy as np
from typing import List, Dict, Optional

class ModelExplainabilityTracker:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_explainability_tables(self):
        """Create tables for model explainability tracking"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # SHAP values
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS shap_values (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            prediction_id UUID,
            feature_name VARCHAR(100) NOT NULL,
            shap_value FLOAT NOT NULL,
            feature_value FLOAT,
            base_value FLOAT,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        # LIME explanations
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS lime_explanations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            prediction_id UUID,
            feature_name VARCHAR(100) NOT NULL,
            weight FLOAT NOT NULL,
            feature_value FLOAT,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        # Explanation metadata
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS explanation_metadata (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            prediction_id UUID,
            explanation_type VARCHAR(50) NOT NULL,
            explanation_method VARCHAR(50) NOT NULL,
            complexity_score FLOAT,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "Explainability tables created successfully"
    
    def log_shap_values(self, model_name: str, version: str, environment: str,
                       prediction_id: str, feature_names: List[str],
                       shap_values: List[float], feature_values: List[float],
                       base_value: float):
        """Log SHAP values to database"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Insert SHAP values
        for i, (feature_name, shap_value, feature_value) in enumerate(zip(feature_names, shap_values, feature_values)):
            insert_query = """
            INSERT INTO shap_values (
                model_name, version, environment, prediction_id,
                feature_name, shap_value, feature_value, base_value, timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW())
            """
            
            cursor.execute(insert_query, (
                model_name, version, environment, prediction_id,
                feature_name, shap_value, feature_value, base_value
            ))
        
        # Insert explanation metadata
        metadata_query = """
        INSERT INTO explanation_metadata (
            model_name, version, environment, prediction_id,
            explanation_type, explanation_method, complexity_score, timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """
        
        complexity_score = np.std(shap_values)  # Simple complexity measure
        cursor.execute(metadata_query, (
            model_name, version, environment, prediction_id,
            "shap", "tree_shap", complexity_score
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    
    def log_lime_explanation(self, model_name: str, version: str, environment: str,
                            prediction_id: str, feature_names: List[str],
                            weights: List[float], feature_values: List[float]):
        """Log LIME explanation to database"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Insert LIME explanations
        for i, (feature_name, weight, feature_value) in enumerate(zip(feature_names, weights, feature_values)):
            insert_query = """
            INSERT INTO lime_explanations (
                model_name, version, environment, prediction_id,
                feature_name, weight, feature_value, timestamp
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
            """
            
            cursor.execute(insert_query, (
                model_name, version, environment, prediction_id,
                feature_name, weight, feature_value
            ))
        
        # Insert explanation metadata
        metadata_query = """
        INSERT INTO explanation_metadata (
            model_name, version, environment, prediction_id,
            explanation_type, explanation_method, complexity_score, timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
        """
        
        complexity_score = np.std(weights)  # Simple complexity measure
        cursor.execute(metadata_query, (
            model_name, version, environment, prediction_id,
            "lime", "linear_regression", complexity_score
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return True
    
    def get_shap_values_for_prediction(self, prediction_id: str):
        """Get SHAP values for a specific prediction"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        query = """
        SELECT 
            feature_name,
            shap_value,
            feature_value,
            base_value
        FROM shap_values
        WHERE prediction_id = %s
        ORDER BY ABS(shap_value) DESC
        """
        
        cursor.execute(query, (prediction_id,))
        results = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                'feature_name': row[0],
                'shap_value': row[1],
                'feature_value': row[2],
                'base_value': row[3]
            }
            for row in results
        ]

# Usage example
explainability_tracker = ModelExplainabilityTracker(db_config)

# Create explainability tables
explainability_tracker.create_explainability_tables()

# Generate sample SHAP values
feature_names = ["engagement_score", "session_count", "age", "user_id"]
shap_values = [0.35, 0.25, -0.15, 0.05]
feature_values = [0.8, 5, 25, 123]
base_value = 0.5

# Log SHAP values
explainability_tracker.log_shap_values(
    model_name="user_engagement_predictor",
    version="v1.2.3",
    environment="production",
    prediction_id="pred_123",
    feature_names=feature_names,
    shap_values=shap_values,
    feature_values=feature_values,
    base_value=base_value
)

# Get SHAP values for prediction
shap_values_pred = explainability_tracker.get_shap_values_for_prediction("pred_123")
print(f"SHAP values for prediction: {shap_values_pred}")
```

### Step 2: Explainability drift detection
```python
class ExplainabilityDriftDetector:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_explainability_drift_tables(self):
        """Create tables for explainability drift detection"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Explainability drift results
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS explainability_drift_results (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            feature_name VARCHAR(100) NOT NULL,
            reference_shap_mean FLOAT,
            current_shap_mean FLOAT,
            shap_mean_difference FLOAT,
            drift_detected BOOLEAN NOT NULL,
            threshold FLOAT,
            timestamp TIMESTAMP NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "Explainability drift tables created successfully"
    
    def detect_explainability_drift(self, model_name: str, version: str, environment: str,
                                  feature_name: str, reference_shap_values: List[float],
                                  current_shap_values: List[float], threshold: float = 0.1):
        """Detect drift in SHAP values"""
        ref_mean = np.mean(reference_shap_values)
        curr_mean = np.mean(current_shap_values)
        mean_diff = abs(ref_mean - curr_mean)
        drift_detected = mean_diff > threshold
        
        return {
            'reference_shap_mean': ref_mean,
            'current_shap_mean': curr_mean,
            'shap_mean_difference': mean_diff,
            'drift_detected': drift_detected,
            'threshold': threshold,
            'feature_name': feature_name
        }
    
    def log_explainability_drift(self, model_name: str, version: str, environment: str,
                                feature_name: str, reference_shap_values: List[float],
                                current_shap_values: List[float], threshold: float = 0.1):
        """Log explainability drift detection result"""
        drift_info = self.detect_explainability_drift(
            model_name, version, environment, feature_name,
            reference_shap_values, current_shap_values, threshold
        )
        
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO explainability_drift_results (
            model_name, version, environment, feature_name,
            reference_shap_mean, current_shap_mean, shap_mean_difference,
            drift_detected, threshold, timestamp
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW())
        """
        
        cursor.execute(insert_query, (
            model_name, version, environment, feature_name,
            drift_info['reference_shap_mean'], drift_info['current_shap_mean'],
            drift_info['shap_mean_difference'], drift_info['drift_detected'],
            drift_info['threshold']
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return drift_info

# Usage example
explainability_drift_detector = ExplainabilityDriftDetector(db_config)

# Create explainability drift tables
explainability_drift_detector.create_explainability_drift_tables()

# Generate sample SHAP values for drift detection
reference_shap = np.random.normal(0.35, 0.05, 100).tolist()
current_shap = np.random.normal(0.25, 0.08, 100).tolist()

# Detect and log explainability drift
drift_result = explainability_drift_detector.log_explainability_drift(
    model_name="user_engagement_predictor",
    version="v1.2.4",
    environment="production",
    feature_name="engagement_score",
    reference_shap_values=reference_shap,
    current_shap_values=current_shap,
    threshold=0.1
)

print(f"Explainability drift result: {drift_result}")
```

## Section 5: Production Observability

### Step 1: Comprehensive observability dashboard
```python
class ProductionObservabilityDashboard:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
        self.monitor = ModelPerformanceMonitor(db_config)
        self.drift_detector = DataDriftDetector(db_config)
        self.concept_drift_detector = ConceptDriftDetector(db_config)
        self.explainability_tracker = ModelExplainabilityTracker(db_config)
    
    def get_overall_model_health(self, model_name: str, hours: int = 24):
        """Get overall model health summary"""
        # Get prediction summary
        prediction_summary = self.monitor.get_prediction_summary(model_name, hours)
        
        # Get recent metrics
        recent_metrics = self.monitor.get_recent_metrics(model_name, hours)
        
        # Get drift alerts
        drift_alerts = self.drift_detector.get_recent_drift_alerts(model_name, hours)
        
        # Get concept drift alerts
        concept_drift_alerts = self.concept_drift_detector.get_concept_drift_alerts(model_name, hours)
        
        # Calculate health score
        health_score = 1.0
        
        # Prediction success rate
        success_rate = prediction_summary['successful_predictions'] / prediction_summary['total_predictions'] if prediction_summary['total_predictions'] > 0 else 0
        health_score *= success_rate
        
        # Latency impact
        if prediction_summary['avg_latency_ms'] > 200:
            health_score *= 0.8  # 20% penalty for high latency
        
        # Drift impact
        if len(drift_alerts) > 0:
            health_score *= 0.7  # 30% penalty for drift detected
        
        # Concept drift impact
        if len(concept_drift_alerts) > 0:
            health_score *= 0.6  # 40% penalty for concept drift
        
        return {
            'model_name': model_name,
            'health_score': max(0.0, min(1.0, health_score)),
            'prediction_summary': prediction_summary,
            'recent_metrics': recent_metrics,
            'drift_alerts': drift_alerts,
            'concept_drift_alerts': concept_drift_alerts,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    def get_model_performance_trend(self, model_name: str, days: int = 7):
        """Get model performance trend over time"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        query = """
        SELECT 
            DATE(timestamp) as date,
            AVG(metric_value) as avg_accuracy,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY metric_value) as p95_accuracy,
            COUNT(*) as sample_count
        FROM model_metrics
        WHERE model_name = %s 
        AND metric_name = 'accuracy'
        AND timestamp >= NOW() - INTERVAL '%s days'
        GROUP BY DATE(timestamp)
        ORDER BY date
        """
        
        cursor.execute(query, (model_name, days))
        results = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                'date': row[0].isoformat(),
                'avg_accuracy': row[1],
                'p95_accuracy': row[2],
                'sample_count': row[3]
            }
            for row in results
        ]
    
    def get_drift_trend(self, model_name: str, days: int = 7):
        """Get drift trend over time"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        query = """
        SELECT 
            DATE(timestamp) as date,
            COUNT(CASE WHEN drift_detected = TRUE THEN 1 END) as drift_count,
            COUNT(*) as total_checks,
            AVG(CASE WHEN drift_detected = TRUE THEN p_value ELSE 1 END) as avg_p_value
        FROM data_drift_results
        WHERE model_name = %s 
        AND timestamp >= NOW() - INTERVAL '%s days'
        GROUP BY DATE(timestamp)
        ORDER BY date
        """
        
        cursor.execute(query, (model_name, days))
        results = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                'date': row[0].isoformat(),
                'drift_count': row[1],
                'total_checks': row[2],
                'avg_p_value': row[3]
            }
            for row in results
        ]

# Usage example
observability_dashboard = ProductionObservabilityDashboard(db_config)

# Get overall model health
health_summary = observability_dashboard.get_overall_model_health("user_engagement_predictor", hours=24)
print(f"Overall model health: {health_summary}")

# Get performance trend
performance_trend = observability_dashboard.get_model_performance_trend("user_engagement_predictor", days=7)
print(f"Performance trend: {performance_trend}")

# Get drift trend
drift_trend = observability_dashboard.get_drift_trend("user_engagement_predictor", days=7)
print(f"Drift trend: {drift_trend}")
```

## Section 6: Alerting and Anomaly Detection

### Step 1: Proactive alerting system
```python
class ModelAlertingSystem:
    def __init__(self, db_config: Dict):
        self.db_config = db_config
    
    def create_alert_tables(self):
        """Create tables for alerting system"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Alerts
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_alerts (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_name VARCHAR(255) NOT NULL,
            version VARCHAR(50) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            alert_type VARCHAR(100) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            message TEXT NOT NULL,
            details JSONB,
            triggered_at TIMESTAMP NOT NULL,
            acknowledged BOOLEAN DEFAULT FALSE,
            acknowledged_at TIMESTAMP,
            acknowledged_by VARCHAR(255),
            resolved BOOLEAN DEFAULT FALSE,
            resolved_at TIMESTAMP,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """)
        
        # Alert rules
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS alert_rules (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            model_name VARCHAR(255) NOT NULL,
            environment VARCHAR(50) NOT NULL,
            alert_type VARCHAR(100) NOT NULL,
            severity VARCHAR(20) NOT NULL,
            condition JSONB NOT NULL,
            enabled BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT NOW(),
            created_by VARCHAR(255)
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return "Alerting tables created successfully"
    
    def create_alert_rule(self, model_name: str, environment: str,
                         alert_type: str, severity: str, condition: Dict,
                         created_by: str = "system"):
        """Create alert rule"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO alert_rules (
            model_name, environment, alert_type, severity, condition, created_by
        ) VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id
        """
        
        cursor.execute(insert_query, (model_name, environment, alert_type, severity, json.dumps(condition), created_by))
        rule_id = cursor.fetchone()[0]
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return rule_id
    
    def evaluate_alert_rules(self, model_name: str, environment: str,
                            metrics: Dict[str, float], 
                            prediction_stats: Dict = None):
        """Evaluate alert rules against current metrics"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Get active alert rules
        cursor.execute("""
            SELECT id, alert_type, severity, condition
            FROM alert_rules
            WHERE model_name = %s AND environment = %s AND enabled = TRUE
        """, (model_name, environment))
        
        rules = cursor.fetchall()
        
        alerts = []
        
        for rule_id, alert_type, severity, condition_json in rules:
            condition = json.loads(condition_json)
            
            # Evaluate condition
            should_trigger = self._evaluate_condition(condition, metrics, prediction_stats)
            
            if should_trigger:
                message = f"Alert triggered: {alert_type} for {model_name} ({environment})"
                details = {
                    'rule_id': rule_id,
                    'condition': condition,
                    'metrics': metrics,
                    'prediction_stats': prediction_stats
                }
                
                # Create alert
                insert_query = """
                INSERT INTO model_alerts (
                    model_name, version, environment, alert_type, severity,
                    message, details, triggered_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, NOW())
                """
                
                cursor.execute(insert_query, (
                    model_name, "latest", environment, alert_type, severity,
                    message, json.dumps(details)
                ))
                
                alerts.append({
                    'alert_type': alert_type,
                    'severity': severity,
                    'message': message,
                    'details': details
                })
        
        conn.commit()
        cursor.close()
        conn.close()
        
        return alerts
    
    def _evaluate_condition(self, condition: Dict, metrics: Dict, prediction_stats: Dict = None) -> bool:
        """Evaluate condition against metrics"""
        if condition['type'] == 'metric_threshold':
            metric_name = condition['metric']
            operator = condition['operator']
            threshold = condition['threshold']
            
            if metric_name not in metrics:
                return False
            
            value = metrics[metric_name]
            
            if operator == '>':
                return value > threshold
            elif operator == '<':
                return value < threshold
            elif operator == '>=':
                return value >= threshold
            elif operator == '<=':
                return value <= threshold
            elif operator == '==':
                return value == threshold
            elif operator == '!=':
                return value != threshold
        
        elif condition['type'] == 'drift_detected':
            # Check if drift was detected
            return condition.get('drift_detected', False)
        
        return False
    
    def get_active_alerts(self, model_name: str, environment: str, severity: str = None):
        """Get active alerts"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        query = """
        SELECT 
            alert_type,
            severity,
            message,
            details,
            triggered_at,
            acknowledged,
            resolved
        FROM model_alerts
        WHERE model_name = %s AND environment = %s AND resolved = FALSE
        """
        
        params = [model_name, environment]
        if severity:
            query += " AND severity = %s"
            params.append(severity)
        
        query += " ORDER BY triggered_at DESC"
        
        cursor.execute(query, params)
        results = cursor.fetchall()
        
        conn.close()
        
        return [
            {
                'alert_type': row[0],
                'severity': row[1],
                'message': row[2],
                'details': json.loads(row[3]) if row[3] else {},
                'triggered_at': row[4].isoformat(),
                'acknowledged': row[5],
                'resolved': row[6]
            }
            for row in results
        ]

# Usage example
alerting_system = ModelAlertingSystem(db_config)

# Create alerting tables
alerting_system.create_alert_tables()

# Create alert rules
alerting_system.create_alert_rule(
    model_name="user_engagement_predictor",
    environment="production",
    alert_type="high_latency",
    severity="warning",
    condition={
        "type": "metric_threshold",
        "metric": "prediction_latency",
        "operator": ">",
        "threshold": 200.0
    }
)

alerting_system.create_alert_rule(
    model_name="user_engagement_predictor",
    environment="production",
    alert_type="low_success_rate",
    severity="critical",
    condition={
        "type": "metric_threshold",
        "metric": "success_rate",
        "operator": "<",
        "threshold": 0.90
    }
)

# Evaluate alert rules
metrics = {
    "prediction_latency": 250.0,
    "success_rate": 0.85,
    "requests_per_second": 45.2
}

alerts = alerting_system.evaluate_alert_rules(
    model_name="user_engagement_predictor",
    environment="production",
    metrics=metrics
)

print(f"Generated alerts: {alerts}")

# Get active alerts
active_alerts = alerting_system.get_active_alerts("user_engagement_predictor", "production")
print(f"Active alerts: {active_alerts}")
```

## Section 7: Performance Benchmarking

### Step 1: Monitoring overhead benchmarking
```python
import time
import pandas as pd
from typing import List, Dict, Callable

class MonitoringBenchmark:
    def __init__(self):
        self.results = []
    
    def benchmark_monitoring_overhead(self, methods: List[Callable], 
                                    operations: List[str] = ["prediction", "training", "inference"]):
        """Benchmark monitoring overhead"""
        for method in methods:
            for operation in operations:
                start_time = time.time()
                
                try:
                    method(operation)
                    duration = time.time() - start_time
                    
                    self.results.append({
                        'benchmark': 'monitoring_overhead',
                        'method': method.__name__,
                        'operation': operation,
                        'duration_seconds': duration,
                        'overhead_percentage': duration / 0.01 * 100 if duration > 0 else 0  # baseline 10ms
                    })
                except Exception as e:
                    self.results.append({
                        'benchmark': 'monitoring_overhead',
                        'method': method.__name__,
                        'operation': operation,
                        'duration_seconds': float('inf'),
                        'error': str(e)
                    })
    
    def benchmark_drift_detection(self, methods: List[Callable],
                                 data_sizes: List[int] = [100, 1000, 10000]):
        """Benchmark drift detection performance"""
        for method in methods:
            for size in data_sizes:
                start_time = time.time()
                
                try:
                    method(size)
                    duration = time.time() - start_time
                    
                    self.results.append({
                        'benchmark': 'drift_detection',
                        'method': method.__name__,
                        'data_size': size,
                        'duration_seconds': duration,
                        'throughput_operations_per_second': size / duration if duration > 0 else 0
                    })
                except Exception as e:
                    self.results.append({
                        'benchmark': 'drift_detection',
                        'method': method.__name__,
                        'data_size': size,
                        'duration_seconds': float('inf'),
                        'error': str(e)
                    })
    
    def generate_monitoring_benchmark_report(self):
        """Generate comprehensive monitoring benchmark report"""
        df = pd.DataFrame(self.results)
        
        # Calculate statistics
        summary = df.groupby(['benchmark', 'method']).agg({
            'duration_seconds': ['mean', 'std', 'min', 'max'],
            'overhead_percentage': ['mean', 'std'],
            'throughput_operations_per_second': ['mean', 'std']
        }).round(2)
        
        # Generate recommendations
        recommendations = []
        
        # Best monitoring overhead
        if 'monitoring_overhead' in df['benchmark'].values:
            best_overhead = df[df['benchmark'] == 'monitoring_overhead'].loc[
                df[df['benchmark'] == 'monitoring_overhead']['duration_seconds'].idxmin()
            ]
            recommendations.append(
                f"Best monitoring overhead: {best_overhead['method']} "
                f"({best_overhead['duration_seconds']:.4f}s for {best_overhead['operation']} operation)"
            )
        
        # Best drift detection
        if 'drift_detection' in df['benchmark'].values:
            best_drift = df[df['benchmark'] == 'drift_detection'].loc[
                df[df['benchmark'] == 'drift_detection']['duration_seconds'].idxmin()
            ]
            recommendations.append(
                f"Best drift detection: {best_drift['method']} "
                f"({best_drift['duration_seconds']:.4f}s for {best_drift['data_size']} records)"
            )
        
        return {
            'summary': summary,
            'detailed_results': df,
            'recommendations': recommendations,
            'monitoring_tips': [
                "Use sampling for large datasets to reduce drift detection overhead",
                "Implement incremental drift detection for real-time monitoring",
                "Cache frequently accessed monitoring data",
                "Optimize database indexes for time-series monitoring queries",
                "Batch monitoring operations when possible",
                "Monitor monitoring overhead itself to ensure it doesn't impact production",
                "Consider trade-offs between monitoring frequency and system performance",
                "Implement alert suppression to reduce noise in high-volume systems"
            ]
        }

# Usage example
benchmark = MonitoringBenchmark()

# Define test methods
def test_monitoring_overhead(operation: str):
    """Test monitoring overhead"""
    time.sleep(0.015)  # Simulate 15ms overhead

def test_drift_detection(size: int):
    """Test drift detection"""
    time.sleep(0.002 * size)  # Simulate O(n) complexity

# Run benchmarks
benchmark.benchmark_monitoring_overhead(
    [test_monitoring_overhead],
    ["prediction", "training", "inference"]
)

benchmark.benchmark_drift_detection(
    [test_drift_detection],
    [100, 1000, 10000]
)

report = benchmark.generate_monitoring_benchmark_report()
print("Monitoring Benchmark Report:")
print(report['summary'])
print("\nRecommendations:")
for rec in report['recommendations']:
    print(f"- {rec}")
```

## Hands-on Exercises

### Exercise 1: Real-time monitoring implementation
1. Set up database-backed model metrics storage
2. Implement prediction logging
3. Create monitoring dashboard queries
4. Test with simulated production traffic

### Exercise 2: Data drift detection
1. Implement statistical drift detection
2. Set up feature importance drift detection
3. Configure drift alerting
4. Test with synthetic drift scenarios

### Exercise 3: Concept drift detection
1. Implement model performance drift detection
2. Set up concept drift alerting
3. Integrate with monitoring system
4. Test with performance degradation scenarios

### Exercise 4: Model explainability tracking
1. Integrate SHAP with database storage
2. Implement LIME explanation tracking
3. Set up explainability drift detection
4. Test with model updates that change feature importance

### Exercise 5: Production observability
1. Build comprehensive observability dashboard
2. Implement health scoring system
3. Create performance trend analysis
4. Test with real-world model deployment scenarios

### Exercise 6: Alerting system
1. Set up alert rules engine
2. Implement proactive alerting
3. Configure escalation policies
4. Test alert suppression and deduplication

### Exercise 7: Performance optimization
1. Benchmark monitoring overhead
2. Optimize database queries for monitoring
3. Implement caching strategies
4. Test scalability with increasing data volumes

## Best Practices Summary

1. **Comprehensive Monitoring**: Track both technical metrics (latency, throughput) and business metrics (accuracy, business impact)
2. **Drift Detection**: Implement both data drift and concept drift detection for complete monitoring coverage
3. **Explainability Tracking**: Store and monitor model explanations to understand model behavior changes
4. **Proactive Alerting**: Use intelligent alerting with appropriate thresholds and suppression
5. **Performance Considerations**: Monitor and optimize monitoring overhead to avoid impacting production systems
6. **Database Optimization**: Use appropriate indexing and partitioning for time-series monitoring data
7. **Automation**: Automate monitoring setup and alert configuration
8. **Documentation**: Maintain comprehensive documentation of monitoring metrics and alerting logic

This tutorial provides practical, hands-on experience with database integration for AI/ML model monitoring and observability. Complete all exercises to master these critical skills for building robust, observable ML systems.