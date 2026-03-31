# Database Data Quality Management Framework

This comprehensive guide covers data quality management practices for production database systems, focusing on accuracy, completeness, consistency, and reliability.

## Table of Contents
1. [Introduction to Data Quality]
2. [Data Profiling and Anomaly Detection]
3. [Schema Validation and Contract Enforcement]
4. [Data Lineage Tracking Implementation]
5. [Quality SLAs and Monitoring]
6. [Automated Data Validation Pipelines]
7. [Implementation Examples]
8. [Common Anti-Patterns and Solutions]

---

## 1. Introduction to Data Quality

Data quality is the foundation of reliable AI/ML systems. Poor data quality leads to inaccurate models, unreliable insights, and business failures.

### Data Quality Dimensions
- **Accuracy**: Data correctly represents real-world entities
- **Completeness**: All required data is present
- **Consistency**: Data is uniform across systems
- **Timeliness**: Data is available when needed
- **Validity**: Data conforms to defined formats and rules
- **Uniqueness**: No duplicate records exist
- **Integrity**: Relationships between data are maintained

### Business Impact of Poor Data Quality
| Issue | Impact | Cost Example |
|-------|--------|--------------|
| Inaccurate customer data | Failed marketing campaigns | $10K-$100K per campaign |
| Missing transaction data | Revenue leakage | 1-5% revenue loss |
| Inconsistent product data | Customer confusion | 15-30% cart abandonment |
| Duplicate records | Wasted resources | 10-20% operational inefficiency |
| Stale data | Poor decision making | 20-50% reduced ROI |

### Data Quality Maturity Model
| Level | Characteristics | Tools/Techniques |
|-------|----------------|------------------|
| Reactive | Fix issues after they cause problems | Manual validation, error logs |
| Proactive | Prevent issues before they occur | Automated validation, monitoring |
| Predictive | Anticipate quality issues | ML-based anomaly detection |
| Prescriptive | Automatically fix quality issues | Auto-remediation, workflow integration |

---

## 2. Data Profiling and Anomaly Detection

### Comprehensive Data Profiling

#### Statistical Profiling
```sql
-- Example: Comprehensive data profiling query
SELECT 
    column_name,
    COUNT(*) as total_records,
    COUNT(column_name) as non_null_count,
    COUNT(DISTINCT column_name) as distinct_count,
    MIN(column_name) as min_value,
    MAX(column_name) as max_value,
    AVG(column_name) as avg_value,
    STDDEV(column_name) as stddev_value,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY column_name) as median_value,
    -- Null percentage
    (COUNT(*) - COUNT(column_name)) * 100.0 / COUNT(*) as null_percentage,
    -- Uniqueness ratio
    COUNT(DISTINCT column_name) * 100.0 / COUNT(*) as uniqueness_ratio
FROM information_schema.columns c
JOIN your_table t ON true
WHERE c.table_name = 'your_table'
GROUP BY column_name;
```

#### Pattern-Based Profiling
```python
class PatternProfiler:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def profile_patterns(self, table_name: str, column_name: str):
        """Profile common patterns in data"""
        # Email pattern analysis
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        
        # Phone number patterns
        phone_patterns = [
            r'^\+?1?[-.\s]?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})$',
            r'^\+?[0-9]{1,3}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}$'
        ]
        
        # Date format analysis
        date_formats = [
            '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y/%m/%d',
            '%B %d, %Y', '%d %B %Y'
        ]
        
        results = {
            'email_validity': self._check_pattern(table_name, column_name, email_pattern),
            'phone_validity': self._check_multiple_patterns(table_name, column_name, phone_patterns),
            'date_format_consistency': self._analyze_date_formats(table_name, column_name, date_formats),
            'numeric_range_analysis': self._analyze_numeric_ranges(table_name, column_name)
        }
        
        return results
    
    def _check_pattern(self, table_name: str, column_name: str, pattern: str):
        """Check pattern validity"""
        query = f"""
        SELECT 
            COUNT(*) as total,
            COUNT(CASE WHEN {column_name} ~ '{pattern}' THEN 1 END) as matches
        FROM {table_name}
        """
        result = self.db.execute(query)
        return {
            'total_records': result[0][0],
            'valid_records': result[0][1],
            'validity_rate': result[0][1] / result[0][0'] if result[0][0] > 0 else 0
        }
```

### Anomaly Detection Techniques

#### Statistical Anomaly Detection
- **Z-score analysis**: Identify outliers based on standard deviations
- **IQR method**: Interquartile range for outlier detection
- **Moving averages**: Detect anomalies in time series data
- **Control charts**: Statistical process control for data streams

```python
def detect_statistical_anomalies(data: list, method: str = 'z_score'):
    """Detect anomalies using statistical methods"""
    if method == 'z_score':
        mean = np.mean(data)
        std = np.std(data)
        z_scores = [(x - mean) / std for x in data]
        return [i for i, z in enumerate(z_scores) if abs(z) > 3]  # 3 sigma rule
    
    elif method == 'iqr':
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        return [i for i, x in enumerate(data) if x < lower_bound or x > upper_bound]
    
    elif method == 'moving_avg':
        window_size = 10
        moving_averages = []
        for i in range(len(data)):
            if i >= window_size:
                ma = np.mean(data[i-window_size:i])
                moving_averages.append(ma)
        
        anomalies = []
        for i in range(window_size, len(data)):
            if abs(data[i] - moving_averages[i-window_size]) > 2 * np.std(moving_averages):
                anomalies.append(i)
        
        return anomalies
```

#### Machine Learning-Based Anomaly Detection
- **Isolation Forest**: Efficient for high-dimensional data
- **Autoencoders**: Deep learning for complex patterns
- **One-Class SVM**: For novelty detection
- **LSTM-based**: For time series anomaly detection

```python
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class MLAnomalyDetector:
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.scaler = StandardScaler()
    
    def fit(self, data: np.ndarray):
        """Fit anomaly detection model"""
        scaled_data = self.scaler.fit_transform(data)
        self.model.fit(scaled_data)
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """Predict anomalies (1 = normal, -1 = anomaly)"""
        scaled_data = self.scaler.transform(data)
        return self.model.predict(scaled_data)
    
    def anomaly_scores(self, data: np.ndarray) -> np.ndarray:
        """Get anomaly scores"""
        scaled_data = self.scaler.transform(data)
        return self.model.decision_function(scaled_data)
```

### Real-Time Anomaly Detection
```python
class RealTimeAnomalyDetector:
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.data_window = deque(maxlen=window_size)
        self.stats = {
            'mean': 0.0,
            'std': 0.0,
            'count': 0
        }
    
    def update(self, value: float) -> dict:
        """Update with new value and detect anomalies"""
        self.data_window.append(value)
        
        # Update statistics
        if len(self.data_window) == 1:
            self.stats['mean'] = value
            self.stats['std'] = 0.0
            self.stats['count'] = 1
        else:
            old_mean = self.stats['mean']
            self.stats['count'] += 1
            self.stats['mean'] = old_mean + (value - old_mean) / self.stats['count']
            
            # Update variance (Welford's algorithm)
            self.stats['std'] = self.stats['std'] + (value - old_mean) * (value - self.stats['mean'])
        
        # Calculate anomaly score
        if self.stats['count'] > 1:
            z_score = (value - self.stats['mean']) / (self.stats['std'] / np.sqrt(self.stats['count']))
            is_anomaly = abs(z_score) > 3.0
            
            return {
                'value': value,
                'z_score': z_score,
                'is_anomaly': is_anomaly,
                'threshold': 3.0,
                'timestamp': datetime.now()
            }
        else:
            return {'value': value, 'is_anomaly': False, 'timestamp': datetime.now()}
```

---

## 3. Schema Validation and Contract Enforcement

### Schema Definition Standards

#### JSON Schema for Data Contracts
```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Customer Profile",
  "type": "object",
  "properties": {
    "customer_id": {
      "type": "string",
      "pattern": "^[A-Z0-9]{8}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{4}-[A-Z0-9]{12}$",
      "description": "UUID v4 format"
    },
    "name": {
      "type": "string",
      "minLength": 1,
      "maxLength": 100,
      "pattern": "^[a-zA-Z\\s'-]+$"
    },
    "email": {
      "type": "string",
      "format": "email",
      "maxLength": 254
    },
    "created_at": {
      "type": "string",
      "format": "date-time"
    },
    "status": {
      "type": "string",
      "enum": ["active", "inactive", "suspended"]
    },
    "preferences": {
      "type": "object",
      "properties": {
        "marketing_emails": {"type": "boolean"},
        "sms_notifications": {"type": "boolean"},
        "language": {"type": "string", "enum": ["en", "es", "fr", "de"]}
      },
      "required": ["marketing_emails", "sms_notifications"]
    }
  },
  "required": ["customer_id", "name", "email", "created_at", "status"],
  "additionalProperties": false
}
```

#### SQL Schema Validation
```sql
-- PostgreSQL example with constraints
CREATE TABLE customers (
    customer_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) NOT NULL CHECK (name !~ '[^a-zA-Z\s''-]'),
    email VARCHAR(254) NOT NULL UNIQUE CHECK (email ~* '^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$'),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status VARCHAR(20) NOT NULL CHECK (status IN ('active', 'inactive', 'suspended')),
    preferences JSONB NOT NULL CHECK (
        preferences ? 'marketing_emails' AND
        preferences ? 'sms_notifications' AND
        (preferences->>'marketing_emails')::boolean IS NOT NULL AND
        (preferences->>'sms_notifications')::boolean IS NOT NULL AND
        (preferences->>'language') IN ('en', 'es', 'fr', 'de')
    ),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Create trigger for automatic updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
   NEW.updated_at = NOW();
   RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_customers_updated_at 
BEFORE UPDATE ON customers 
FOR EACH ROW 
EXECUTE PROCEDURE update_updated_at_column();
```

### Data Contract Enforcement

#### Contract Validation Middleware
```python
class DataContractValidator:
    def __init__(self, schema_registry: dict):
        self.schemas = schema_registry
        self.validator = Draft7Validator(schema_registry)
    
    def validate(self, data: dict, contract_name: str) -> dict:
        """Validate data against contract"""
        if contract_name not in self.schemas:
            raise ValueError(f"Contract {contract_name} not found")
        
        try:
            # Validate against JSON Schema
            errors = list(self.validator.iter_errors(data))
            
            if errors:
                return {
                    'valid': False,
                    'errors': [str(error) for error in errors],
                    'contract': contract_name,
                    'timestamp': datetime.now()
                }
            else:
                return {
                    'valid': True,
                    'contract': contract_name,
                    'timestamp': datetime.now(),
                    'data_hash': hashlib.sha256(json.dumps(data).encode()).hexdigest()
                }
                
        except Exception as e:
            return {
                'valid': False,
                'error': str(e),
                'contract': contract_name,
                'timestamp': datetime.now()
            }
    
    def enforce_contract(self, data: dict, contract_name: str, strict: bool = True):
        """Enforce contract with optional strict mode"""
        validation_result = self.validate(data, contract_name)
        
        if not validation_result['valid']:
            if strict:
                raise DataContractViolationError(
                    f"Data validation failed for contract {contract_name}: {validation_result['errors']}"
                )
            else:
                # Log warning but allow processing
                logger.warning(
                    f"Data quality issue for contract {contract_name}: {validation_result['errors']}",
                    extra={'data': data, 'contract': contract_name}
                )
        
        return validation_result
```

#### Schema Evolution Management
```python
class SchemaEvolutionManager:
    def __init__(self, db_connection):
        self.db = db_connection
        self.version_history = []
    
    def register_schema_version(self, contract_name: str, schema: dict, version: str):
        """Register new schema version"""
        self.db.execute("""
            INSERT INTO schema_versions 
            (contract_name, version, schema_json, created_at, created_by)
            VALUES (%s, %s, %s, %s, %s)
        """, (contract_name, version, json.dumps(schema), datetime.now(), get_current_user()))
        
        self.version_history.append({
            'contract_name': contract_name,
            'version': version,
            'timestamp': datetime.now(),
            'schema_hash': hashlib.sha256(json.dumps(schema).encode()).hexdigest()
        })
    
    def get_compatible_schemas(self, contract_name: str, data_version: str):
        """Get schemas compatible with given data version"""
        # Backward compatibility: newer schemas can read older data
        # Forward compatibility: older schemas can read newer data (with defaults)
        
        versions = self.db.execute("""
            SELECT version, schema_json 
            FROM schema_versions 
            WHERE contract_name = %s 
            ORDER BY created_at DESC
        """, (contract_name,))
        
        compatible = []
        for version, schema_json in versions:
            schema = json.loads(schema_json)
            # Check compatibility rules
            if self._is_backward_compatible(schema, data_version):
                compatible.append({'version': version, 'schema': schema})
        
        return compatible
    
    def _is_backward_compatible(self, new_schema: dict, old_version: str) -> bool:
        """Check if new schema is backward compatible with old version"""
        # Backward compatible if:
        # 1. No required fields removed
        # 2. No field types changed to incompatible types
        # 3. New optional fields added
        # 4. Enum values expanded (not restricted)
        
        # Simplified check - in practice use more sophisticated logic
        return True
```

---

## 4. Data Lineage Tracking Implementation

### Lineage Tracking Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Raw Data Sources│───▶│  Ingestion Layer │───▶│  Processing Layer│
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Lineage Events │◀──▶│  Metadata Store │◀──▶│  Transformation │
│  (Event Stream) │    │  (Neo4j/PostgreSQL)│    │  Definitions   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │
        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│  Lineage Query  │    │  Impact Analysis│
│  Engine         │    │  Engine        │
└─────────────────┘    └─────────────────┘
```

### Lineage Event Collection
```python
class LineageEventCollector:
    def __init__(self, event_bus: KafkaProducer, metadata_store: Neo4jConnection):
        self.event_bus = event_bus
        self.metadata_store = metadata_store
    
    def collect_lineage_event(self, operation: str, source: str, target: str, 
                            transformation: str = None, metadata: dict = None):
        """Collect lineage event for data movement"""
        event = {
            'event_id': str(uuid.uuid4()),
            'operation': operation,
            'source': source,
            'target': target,
            'transformation': transformation,
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
            'producer': get_current_service(),
            'version': '1.0'
        }
        
        # Send to event bus for real-time processing
        self.event_bus.send('lineage_events', value=json.dumps(event).encode())
        
        # Store in metadata store
        self._store_in_metadata_store(event)
    
    def _store_in_metadata_store(self, event: dict):
        """Store lineage event in graph database"""
        # Neo4j Cypher query
        query = """
        MERGE (s:Dataset {name: $source})
        MERGE (t:Dataset {name: $target})
        CREATE (e:Event {
            id: $event_id,
            operation: $operation,
            timestamp: $timestamp,
            producer: $producer,
            version: $version
        })
        CREATE (s)-[r:TRANSFORMED_TO {transformation: $transformation}]->(t)
        CREATE (e)-[:TRIGGERED]->(r)
        WITH e, s, t
        UNWIND $metadata_keys AS key
        WITH e, s, t, key, $metadata[key] AS value
        SET e[key] = value
        """
        
        params = {
            'event_id': event['event_id'],
            'operation': event['operation'],
            'source': event['source'],
            'target': event['target'],
            'transformation': event.get('transformation'),
            'timestamp': event['timestamp'],
            'producer': event['producer'],
            'version': event['version'],
            'metadata': event.get('metadata', {}),
            'metadata_keys': list(event.get('metadata', {}).keys()) if event.get('metadata') else []
        }
        
        self.metadata_store.run(query, **params)
```

### Lineage Query Patterns

#### Impact Analysis
```python
class LineageQueryEngine:
    def __init__(self, graph_db: Neo4jConnection):
        self.db = graph_db
    
    def find_impact_of_change(self, dataset_name: str, field_name: str = None):
        """Find all downstream dependencies of a dataset or field"""
        if field_name:
            # Field-level impact analysis
            query = """
            MATCH path = (n:Dataset {name: $dataset})-[*..5]->(m)
            WHERE n.name = $dataset AND ANY(label IN labels(m) WHERE label = 'Field' AND m.name = $field)
            RETURN DISTINCT m, relationships(path)
            """
        else:
            # Dataset-level impact analysis
            query = """
            MATCH path = (n:Dataset {name: $dataset})-[*..5]->(m:Dataset)
            RETURN DISTINCT m.name as downstream_dataset, 
                   length(path) as distance,
                   [r in relationships(path) | r.transformation] as transformations
            ORDER BY distance
            """
        
        result = self.db.run(query, dataset=dataset_name, field=field_name)
        return [record for record in result]
    
    def trace_data_origin(self, dataset_name: str):
        """Trace data back to original sources"""
        query = """
        MATCH path = (n:Dataset {name: $dataset})<-[*..5]-(m:Dataset)
        WHERE NOT ()<-[]-(m)
        RETURN DISTINCT m.name as source_dataset,
               [r in relationships(path) | r.transformation] as transformations,
               length(path) as distance
        ORDER BY distance
        """
        
        result = self.db.run(query, dataset=dataset_name)
        return [record for record in result]
    
    def analyze_quality_propagation(self, dataset_name: str):
        """Analyze how quality issues propagate through lineage"""
        query = """
        MATCH path = (n:Dataset {name: $dataset})-[*..5]->(m)
        OPTIONAL MATCH (n)-[q:HAS_QUALITY]->(quality)
        RETURN m.name as dataset,
               quality.score as quality_score,
               quality.timestamp as quality_timestamp,
               length(path) as distance
        ORDER BY distance
        """
        
        result = self.db.run(query, dataset=dataset_name)
        return [record for record in result]
```

### Lineage Visualization
```python
class LineageVisualizer:
    def __init__(self, lineage_engine: LineageQueryEngine):
        self.engine = lineage_engine
    
    def generate_lineage_graph(self, dataset_name: str, depth: int = 3):
        """Generate visualization-ready lineage graph"""
        # Get upstream and downstream lineage
        upstream = self.engine.trace_data_origin(dataset_name)
        downstream = self.engine.find_impact_of_change(dataset_name)
        
        # Build graph structure
        nodes = {}
        edges = []
        
        # Add central node
        nodes[dataset_name] = {
            'id': dataset_name,
            'label': dataset_name,
            'type': 'dataset',
            'color': '#4CAF50',
            'size': 20
        }
        
        # Add upstream nodes
        for item in upstream:
            source_name = item['source_dataset']
            nodes[source_name] = {
                'id': source_name,
                'label': source_name,
                'type': 'source',
                'color': '#2196F3',
                'size': 16
            }
            edges.append({
                'from': source_name,
                'to': dataset_name,
                'label': '→',
                'arrows': 'to',
                'color': '#9E9E9E'
            })
        
        # Add downstream nodes
        for item in downstream:
            target_name = item['downstream_dataset']
            nodes[target_name] = {
                'id': target_name,
                'label': target_name,
                'type': 'destination',
                'color': '#FF5722',
                'size': 16
            }
            edges.append({
                'from': dataset_name,
                'to': target_name,
                'label': '→',
                'arrows': 'to',
                'color': '#9E9E9E'
            })
        
        return {
            'nodes': list(nodes.values()),
            'edges': edges,
            'title': f'Lineage Graph: {dataset_name}',
            'timestamp': datetime.now().isoformat()
        }
```

---

## 5. Quality SLAs and Monitoring

### Data Quality SLA Framework

#### SLA Definition Template
```yaml
# data_quality_sla.yaml
customer_data:
  accuracy: 99.9%
  completeness: 99.5%
  consistency: 99.8%
  timeliness: 95% within 5 minutes
  validity: 99.9%
  uniqueness: 99.99%
  integrity: 99.9%

transaction_data:
  accuracy: 99.99%
  completeness: 99.9%
  consistency: 99.95%
  timeliness: 99% within 1 minute
  validity: 99.99%
  uniqueness: 99.999%
  integrity: 99.99%

analytics_data:
  accuracy: 99.5%
  completeness: 98%
  consistency: 99%
  timeliness: 90% within 1 hour
  validity: 99%
  uniqueness: 99.5%
  integrity: 99%
```

### Quality Monitoring Dashboard

#### Key Metrics Dashboard
```python
class DataQualityDashboard:
    def __init__(self, metrics_client: PrometheusClient):
        self.metrics = metrics_client
    
    def get_quality_summary(self):
        """Get overall data quality summary"""
        metrics = {
            'overall_quality_score': self._calculate_overall_score(),
            'critical_issues': self._get_critical_issues(),
            'trend': self._get_quality_trend(),
            'sla_compliance': self._get_sla_compliance(),
            'top_problem_areas': self._get_top_problems()
        }
        
        return metrics
    
    def _calculate_overall_score(self):
        """Calculate weighted overall quality score"""
        weights = {
            'accuracy': 0.3,
            'completeness': 0.2,
            'consistency': 0.2,
            'timeliness': 0.1,
            'validity': 0.1,
            'uniqueness': 0.05,
            'integrity': 0.05
        }
        
        scores = {}
        for dimension, weight in weights.items():
            score = self.metrics.get(f"data_quality_{dimension}_score")
            scores[dimension] = score or 0.0
        
        return sum(scores[dimension] * weight for dimension, weight in weights.items())
    
    def _get_critical_issues(self):
        """Get critical data quality issues"""
        return self.metrics.query("""
            SELECT metric, value, timestamp 
            FROM data_quality_alerts 
            WHERE severity = 'CRITICAL' 
            AND timestamp > NOW() - INTERVAL '24 hours'
            ORDER BY timestamp DESC
            LIMIT 10
        """)
    
    def _get_quality_trend(self):
        """Get quality trend over time"""
        return self.metrics.query("""
            SELECT 
                DATE(timestamp) as date,
                AVG(overall_quality_score) as avg_score
            FROM data_quality_metrics 
            WHERE timestamp > NOW() - INTERVAL '30 days'
            GROUP BY DATE(timestamp)
            ORDER BY date
        """)
    
    def _get_sla_compliance(self):
        """Get SLA compliance rates"""
        return self.metrics.query("""
            SELECT 
                sla_metric,
                compliance_rate,
                target_rate,
                CASE 
                    WHEN compliance_rate >= target_rate THEN 'COMPLIANT'
                    ELSE 'NON_COMPLIANT'
                END as status
            FROM data_quality_sla_compliance
            WHERE timestamp = (SELECT MAX(timestamp) FROM data_quality_sla_compliance)
        """)
```

### Alerting Strategy

#### Alert Classification
| Severity | Threshold | Response Time | Notification |
|----------|-----------|---------------|--------------|
| Critical | <95% quality | 5 minutes | SMS + Email + PagerDuty |
| High | <98% quality | 15 minutes | Email + Slack |
| Medium | <99% quality | 1 hour | Slack + Email digest |
| Low | <99.5% quality | 24 hours | Email digest |

#### Alert Generation
```python
class QualityAlertGenerator:
    def __init__(self, alert_system: AlertManager, sla_config: dict):
        self.alert_system = alert_system
        self.sla_config = sla_config
    
    def check_quality_thresholds(self, dataset: str, quality_metrics: dict):
        """Check quality metrics against SLA thresholds"""
        alerts = []
        
        for metric, current_value in quality_metrics.items():
            if metric in self.sla_config.get(dataset, {}):
                sla_target = self.sla_config[dataset][metric]
                
                # Calculate deviation
                deviation = (sla_target - current_value) / sla_target * 100
                
                # Determine severity
                if deviation > 10:
                    severity = 'CRITICAL'
                elif deviation > 5:
                    severity = 'HIGH'
                elif deviation > 2:
                    severity = 'MEDIUM'
                else:
                    continue  # Within acceptable range
                
                alert = {
                    'dataset': dataset,
                    'metric': metric,
                    'current_value': current_value,
                    'sla_target': sla_target,
                    'deviation_percent': deviation,
                    'severity': severity,
                    'timestamp': datetime.now(),
                    'recommendation': self._get_recommendation(metric, deviation)
                }
                
                alerts.append(alert)
        
        return alerts
    
    def _get_recommendation(self, metric: str, deviation: float) -> str:
        """Get remediation recommendations"""
        recommendations = {
            'accuracy': {
                '>10%': 'Investigate data source quality and validation rules',
                '>5%': 'Review data transformation logic and validation',
                '>2%': 'Monitor closely, investigate root cause'
            },
            'completeness': {
                '>10%': 'Check data ingestion pipelines and source availability',
                '>5%': 'Review ETL job failures and retry mechanisms',
                '>2%': 'Verify data source connectivity and permissions'
            },
            'consistency': {
                '>10%': 'Audit data transformation logic across systems',
                '>5%': 'Review schema evolution and migration processes',
                '>2%': 'Check for recent schema changes or deployments'
            }
        }
        
        for threshold, recommendation in recommendations.get(metric, {}).items():
            if deviation > float(threshold.replace('>', '')):
                return recommendation
        
        return 'Monitor and investigate potential causes'
```

---

## 6. Automated Data Validation Pipelines

### Validation Pipeline Architecture
```
┌─────────────────┐    ┌─────────────────┐
│  Data Ingestion │───▶│  Validation Gate │
└─────────────────┘    └─────────────────┘
        │                        │
        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│  Pre-validation │    │  Core Validation│
│  Checks         │    │  (Schema, Rules)│
└─────────────────┘    └─────────────────┘
        │                        │
        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│  Advanced       │    │  Quality Scoring│
│  Validation     │    │  & Anomaly Det. │
└─────────────────┘    └─────────────────┘
        │                        │
        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐
│  Remediation    │◀──▶│  Alerting &     │
│  Workflow1      │    │  Reporting      │
└─────────────────┘    └─────────────────┘
```

### Pipeline Implementation

#### Validation Gateway
```python
class ValidationGateway:
    def __init__(self, validators: list, config: dict):
        self.validators = validators
        self.config = config
        self.metrics = PrometheusClient()
    
    async def validate_data(self, data: dict, context: dict = None) -> dict:
        """Validate data through pipeline"""
        start_time = time.time()
        
        result = {
            'timestamp': datetime.now(),
            'data_id': context.get('data_id', str(uuid.uuid4())),
            'status': 'PENDING',
            'validation_results': [],
            'summary': {
                'total_checks': 0,
                'passed': 0,
                'failed': 0,
                'warnings': 0
            }
        }
        
        try:
            # Run validators in sequence
            for validator in self.validators:
                validator_start = time.time()
                
                try:
                    validation_result = await validator.validate(data, context)
                    
                    # Record metrics
                    self.metrics.increment(f'validation.{validator.name}.total')
                    if validation_result['status'] == 'PASS':
                        self.metrics.increment(f'validation.{validator.name}.pass')
                    elif validation_result['status'] == 'FAIL':
                        self.metrics.increment(f'validation.{validator.name}.fail')
                    elif validation_result['status'] == 'WARN':
                        self.metrics.increment(f'validation.{validator.name}.warn')
                    
                    result['validation_results'].append(validation_result)
                    result['summary']['total_checks'] += 1
                    
                    if validation_result['status'] == 'PASS':
                        result['summary']['passed'] += 1
                    elif validation_result['status'] == 'FAIL':
                        result['summary']['failed'] += 1
                    elif validation_result['status'] == 'WARN':
                        result['summary']['warnings'] += 1
                        
                except Exception as e:
                    error_result = {
                        'validator': validator.name,
                        'status': 'ERROR',
                        'message': str(e),
                        'timestamp': datetime.now()
                    }
                    result['validation_results'].append(error_result)
                    result['summary']['failed'] += 1
                
                validator_time = time.time() - validator_start
                self.metrics.observe(f'validation.{validator.name}.duration_seconds', validator_time)
            
            # Determine overall status
            if result['summary']['failed'] > 0:
                result['status'] = 'FAILED'
            elif result['summary']['warnings'] > 0:
                result['status'] = 'WARNING'
            else:
                result['status'] = 'PASSED'
            
            # Record overall metrics
            self.metrics.observe('validation.duration_seconds', time.time() - start_time)
            self.metrics.set('validation.status', 1 if result['status'] == 'PASSED' else 0)
            
            return result
            
        except Exception as e:
            result['status'] = 'ERROR'
            result['error'] = str(e)
            return result
```

#### Validator Implementations
```python
class SchemaValidator:
    def __init__(self, schema_registry: dict):
        self.schema_registry = schema_registry
        self.validator = Draft7Validator(schema_registry)
    
    async def validate(self, data: dict, context: dict) -> dict:
        contract_name = context.get('contract_name')
        if not contract_name:
            return {'status': 'ERROR', 'message': 'Contract name required'}
        
        if contract_name not in self.schema_registry:
            return {'status': 'ERROR', 'message': f'Contract {contract_name} not found'}
        
        try:
            errors = list(self.validator.iter_errors(data))
            
            if errors:
                return {
                    'validator': 'schema',
                    'status': 'FAIL',
                    'errors': [str(error) for error in errors],
                    'timestamp': datetime.now()
                }
            else:
                return {
                    'validator': 'schema',
                    'status': 'PASS',
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            return {'validator': 'schema', 'status': 'ERROR', 'message': str(e)}

class BusinessRuleValidator:
    def __init__(self, rules: list):
        self.rules = rules
    
    async def validate(self, data: dict, context: dict) -> dict:
        failed_rules = []
        
        for rule in self.rules:
            try:
                # Evaluate rule expression
                if not self._evaluate_rule(rule, data, context):
                    failed_rules.append({
                        'rule_id': rule['id'],
                        'description': rule['description'],
                        'expected': rule.get('expected'),
                        'actual': self._get_actual_value(rule, data)
                    })
            except Exception as e:
                failed_rules.append({
                    'rule_id': rule['id'],
                    'description': rule['description'],
                    'error': str(e)
                })
        
        if failed_rules:
            return {
                'validator': 'business_rules',
                'status': 'FAIL',
                'failed_rules': failed_rules,
                'timestamp': datetime.now()
            }
        else:
            return {
                'validator': 'business_rules',
                'status': 'PASS',
                'timestamp': datetime.now()
            }
    
    def _evaluate_rule(self, rule: dict, data: dict, context: dict) -> bool:
        """Evaluate business rule expression"""
        # Simple rule evaluation - in practice use more sophisticated engine
        if rule['type'] == 'field_comparison':
            field_value = self._get_field_value(rule['field'], data)
            return eval(f"{field_value} {rule['operator']} {rule['value']}")
        
        elif rule['type'] == 'conditional':
            condition = self._evaluate_rule(rule['condition'], data, context)
            if condition:
                return self._evaluate_rule(rule['then'], data, context)
            elif 'else' in rule:
                return self._evaluate_rule(rule['else'], data, context)
            else:
                return True
    
    def _get_field_value(self, field_path: str, data: dict) -> any:
        """Get value from nested field path"""
        keys = field_path.split('.')
        value = data
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return None
        return value
```

### Integration with Data Pipelines
```python
# Apache Airflow example
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

def validate_and_load(**context):
    """Validate data and load to destination"""
    ti = context['task_instance']
    data = ti.xcom_pull(task_ids='extract_data')
    
    # Validate data
    validation_result = validation_gateway.validate_data(data, context)
    
    if validation_result['status'] == 'FAILED':
        raise ValueError(f"Data validation failed: {validation_result}")
    
    # Load to destination
    load_result = destination_loader.load_data(data, validation_result)
    
    # Record validation metrics
    ti.xcom_push(key='validation_result', value=validation_result)
    ti.xcom_push(key='load_result', value=load_result)

# DAG definition
default_args = {
    'owner': 'data_engineering',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'customer_data_pipeline',
    default_args=default_args,
    description='Customer data pipeline with validation',
    schedule_interval=timedelta(hours=1),
    catchup=False,
)

extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_customer_data,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_and_load,
    provide_context=True,
    dag=dag,
)

load_task = PythonOperator(
    task_id='load_data',
    python_callable=load_to_warehouse,
    provide_context=True,
    dag=dag,
)

extract_task >> validate_task >> load_task
```

---

## 7. Implementation Examples

### Example 1: Real-Time Data Quality Monitoring
```python
class RealTimeQualityMonitor:
    def __init__(self, kafka_consumer: KafkaConsumer, quality_engine: QualityAlertGenerator):
        self.consumer = kafka_consumer
        self.quality_engine = quality_engine
        self.metrics = PrometheusClient()
    
    async def monitor_stream(self):
        """Monitor data stream in real-time"""
        for message in self.consumer:
            try:
                data = json.loads(message.value)
                context = {
                    'topic': message.topic,
                    'partition': message.partition,
                    'offset': message.offset,
                    'timestamp': message.timestamp
                }
                
                # Validate data
                validation_result = await self.quality_engine.validate_data(data, context)
                
                # Update metrics
                self.metrics.set('data_quality.validation_status', 
                               1 if validation_result['status'] == 'PASSED' else 0)
                self.metrics.increment('data_quality.validation_total')
                
                # Generate alerts if needed
                if validation_result['status'] == 'FAILED':
                    alerts = self.quality_engine.check_quality_thresholds(
                        'customer_data',
                        validation_result['summary']
                    )
                    for alert in alerts:
                        self.quality_engine.alert_system.send_alert(alert)
                
                # Log validation result
                logger.info("Data validation result", 
                          extra={
                              'data_id': context.get('data_id'),
                              'status': validation_result['status'],
                              'checks_passed': validation_result['summary']['passed'],
                              'checks_failed': validation_result['summary']['failed']
                          })
                
            except Exception as e:
                logger.error("Error processing message", extra={'error': str(e)})
                self.metrics.increment('data_quality.processing_errors')
```

### Example 2: Data Quality Scorecard
```python
class DataQualityScorecard:
    def __init__(self, db_connection):
        self.db = db_connection
    
    def generate_scorecard(self, dataset_name: str, period: str = 'daily'):
        """Generate comprehensive data quality scorecard"""
        scorecard = {
            'dataset': dataset_name,
            'period': period,
            'generated_at': datetime.now().isoformat(),
            'metrics': {},
            'summary': {},
            'trends': {},
            'recommendations': []
        }
        
        # Get quality metrics
        metrics = self._get_quality_metrics(dataset_name, period)
        scorecard['metrics'] = metrics
        
        # Calculate summary
        scorecard['summary'] = self._calculate_summary(metrics)
        
        # Analyze trends
        scorecard['trends'] = self._analyze_trends(dataset_name, period)
        
        # Generate recommendations
        scorecard['recommendations'] = self._generate_recommendations(metrics)
        
        return scorecard
    
    def _get_quality_metrics(self, dataset_name: str, period: str):
        """Get quality metrics for dataset"""
        query = """
        SELECT 
            metric_name,
            AVG(value) as avg_value,
            MIN(value) as min_value,
            MAX(value) as max_value,
            COUNT(*) as sample_size,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY value) as median_value
        FROM data_quality_metrics
        WHERE dataset_name = %s 
        AND period = %s
        AND timestamp > NOW() - INTERVAL '7 days'
        GROUP BY metric_name
        """
        
        results = self.db.execute(query, (dataset_name, period))
        return {row[0]: {
            'avg': row[1],
            'min': row[2],
            'max': row[3],
            'count': row[4],
            'median': row[5]
        } for row in results}
    
    def _calculate_summary(self, metrics: dict):
        """Calculate overall summary"""
        weights = {
            'accuracy': 0.3,
            'completeness': 0.2,
            'consistency': 0.2,
            'timeliness': 0.1,
            'validity': 0.1,
            'uniqueness': 0.05,
            'integrity': 0.05
        }
        
        total_weight = sum(weights.values())
        score = 0.0
        
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric]['avg'] * (weight / total_weight)
        
        return {
            'overall_score': round(score, 2),
            'grade': self._get_grade(score),
            'status': 'GOOD' if score >= 95 else 'FAIR' if score >= 90 else 'POOR'
        }
    
    def _get_grade(self, score: float):
        """Convert score to letter grade"""
        if score >= 95: return 'A'
        elif score >= 90: return 'B'
        elif score >= 85: return 'C'
        elif score >= 80: return 'D'
        else: return 'F'
    
    def _generate_recommendations(self, metrics: dict):
        """Generate actionable recommendations"""
        recommendations = []
        
        if metrics.get('completeness', {}).get('avg', 100) < 95:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'Completeness',
                'issue': 'Missing data in critical fields',
                'recommendation': 'Implement data completeness checks in ingestion pipeline',
                'impact': 'High - affects downstream analytics accuracy'
            })
        
        if metrics.get('accuracy', {}).get('avg', 100) < 98:
            recommendations.append({
                'priority': 'HIGH',
                'area': 'Accuracy',
                'issue': 'Data accuracy below target',
                'recommendation': 'Review data source quality and validation rules',
                'impact': 'High - affects business decisions'
            })
        
        return recommendations
```

### Example 3: Automated Remediation Workflow
```python
class AutomatedRemediationWorkflow:
    def __init__(self, validation_engine: ValidationGateway, alert_system: AlertSystem):
        self.validation_engine = validation_engine
        self.alert_system = alert_system
    
    async def handle_validation_failure(self, validation_result: dict):
        """Handle validation failures with automated remediation"""
        dataset = validation_result.get('context', {}).get('dataset')
        failed_checks = [r for r in validation_result['validation_results'] 
                        if r['status'] in ['FAIL', 'ERROR']]
        
        for check in failed_checks:
            # Determine remediation strategy
            remediation = self._determine_remediation_strategy(check, dataset)
            
            if remediation['action'] == 'auto_fix':
                await self._auto_fix_issue(check, remediation['parameters'])
            elif remediation['action'] == 'manual_review':
                await self._create_review_ticket(check, remediation['parameters'])
            elif remediation['action'] == 'alert_only':
                await self._send_alert_only(check, remediation['parameters'])
        
        return {'status': 'processed', 'remediation_actions': len(failed_checks)}
    
    def _determine_remediation_strategy(self, check: dict, dataset: str):
        """Determine appropriate remediation strategy"""
        # Rule-based decision engine
        if check['validator'] == 'schema' and 'null' in check.get('message', ''):
            return {'action': 'auto_fix', 'parameters': {'replace_with': 'UNKNOWN'}}
        
        elif check['validator'] == 'business_rules' and 'outlier' in check.get('message', ''):
            return {'action': 'manual_review', 'parameters': {'severity': 'medium'}}
        
        elif check['validator'] == 'anomaly_detection' and check.get('score', 0) > 0.8:
            return {'action': 'alert_only', 'parameters': {'threshold': 0.8}}
        
        else:
            return {'action': 'manual_review', 'parameters': {'severity': 'low'}}
    
    async def _auto_fix_issue(self, check: dict, parameters: dict):
        """Automatically fix data issues"""
        # Example: Replace null values with defaults
        if parameters.get('replace_with') == 'UNKNOWN':
            # Update database
            await self._update_database_null_values(
                check['context']['table'],
                check['context']['column'],
                'UNKNOWN'
            )
        
        # Log remediation
        logger.info("Auto-fixed data issue", extra={
            'check_id': check.get('id'),
            'remediation': 'replaced_null_with_unknown',
            'timestamp': datetime.now()
        })
    
    async def _create_review_ticket(self, check: dict, parameters: dict):
        """Create ticket for manual review"""
        ticket = {
            'title': f"Data Quality Issue: {check['validator']} - {check.get('message', 'Unknown')}",
            'description': f"Validation failed for dataset {check.get('context', {}).get('dataset')}\n\n"
                         f"Check: {check['validator']}\n"
                         f"Error: {check.get('message', 'No message')}\n"
                         f"Severity: {parameters.get('severity', 'medium')}",
            'priority': parameters.get('severity', 'medium'),
            'assignee': 'data_quality_team',
            'labels': ['data-quality', 'validation-failure'],
            'due_date': datetime.now() + timedelta(days=2)
        }
        
        # Create Jira ticket
        jira_response = await self._create_jira_ticket(ticket)
        
        logger.info("Created review ticket", extra={
            'ticket_id': jira_response.get('key'),
            'status': 'created'
        })
```

---

## 8. Common Anti-Patterns and Solutions

### Anti-Pattern 1: Quality as an Afterthought
**Symptom**: Data quality issues discovered late in the pipeline
**Root Cause**: Validation only at end of pipeline
**Solution**: Shift-left validation - validate at ingestion and each transformation step

### Anti-Pattern 2: One-Size-Fits-All Validation
**Symptom**: Same validation rules applied to all data
**Root Cause**: Lack of data classification and context awareness
**Solution**: Context-aware validation with different rules per data type and sensitivity

### Anti-Pattern 3: Manual Quality Checks
**Symptom**: Time-consuming, inconsistent manual validation
**Root Cause**: No automated validation infrastructure
**Solution**: Automated validation pipelines with CI/CD integration

### Anti-Pattern 4: Ignoring Data Lineage
**Symptom**: Can't trace quality issues to root cause
**Root Cause**: No lineage tracking
**Solution**: Comprehensive lineage tracking with impact analysis

### Anti-Pattern 5: No Quality SLAs
**Symptom**: Unclear quality expectations and accountability
**Root Cause**: Lack of measurable quality targets
**Solution**: Define and enforce data quality SLAs with business alignment

---

## Next Steps

1. **Assess current data quality**: Conduct baseline assessment of key datasets
2. **Define quality requirements**: Establish SLAs aligned with business needs
3. **Implement validation gateway**: Start with schema and basic rule validation
4. **Build lineage tracking**: Implement basic lineage collection
5. **Set up monitoring**: Configure quality dashboards and alerting
6. **Automate remediation**: Start with simple auto-fix capabilities

Data quality management is essential for reliable AI/ML systems. By implementing these patterns, you'll build robust data pipelines that produce trustworthy, high-quality data for your applications.