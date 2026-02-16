# Change Data Capture: Real-time Data Pipelines for AI/ML Systems

## Overview

Change Data Capture (CDC) is a technique for capturing and propagating changes to database records in real-time. In AI/ML systems, CDC enables real-time feature engineering, model monitoring, and streaming analytics by providing immediate access to data mutations.

## CDC Architecture Patterns

### 1. Log-Based CDC
- **Mechanism**: Reads database transaction logs (WAL, binlog, redo log)
- **Examples**: Debezium (PostgreSQL, MySQL), Maxwell (MySQL), Oracle GoldenGate
- **Advantages**: Low overhead, no application changes required
- **Disadvantages**: Database-specific, complex setup

### 2. Trigger-Based CDC
- **Mechanism**: Database triggers capture INSERT/UPDATE/DELETE operations
- **Examples**: Custom PostgreSQL triggers, SQL Server CDC
- **Advantages**: Simple implementation, works with any database
- **Disadvantages**: Performance impact, maintenance overhead

### 3. Application-Level CDC
- **Mechanism**: Application code captures changes before persistence
- **Examples**: Event sourcing patterns, domain events
- **Advantages**: Full control, rich context available
- **Disadvantages**: Requires application changes, consistency challenges

## Implementation Deep Dive

### Debezium with Kafka (Production-Ready)

#### Architecture
```
Database → Debezium Connector → Kafka Topics → Stream Processing → ML Systems
     ↑              ↑                   ↑
  Transaction Log  Schema Registry   Consumer Groups
```

#### Configuration Example
```yaml
# debezium-postgres-connector.json
{
  "name": "postgres-connector",
  "config": {
    "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
    "database.hostname": "postgres-prod",
    "database.port": "5432",
    "database.user": "debezium",
    "database.password": "password",
    "database.dbname": "ml_feature_store",
    "database.server.name": "feature-store",
    "topic.prefix": "ml",
    "slot.name": "debezium_slot",
    "plugin.name": "pgoutput",
    "transforms": "unwrap",
    "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
    "transforms.unwrap.drop.tombstones": "false"
  }
}
```

### Real-Time Feature Engineering Pipeline

```python
# Stream processing with Apache Flink
class RealTimeFeatureProcessor:
    def __init__(self):
        self.feature_store = RedisCluster()
        self.model_registry = ModelRegistry()
    
    def process_change_event(self, event):
        if event.operation == "INSERT":
            self._create_features(event.data)
        elif event.operation == "UPDATE":
            self._update_features(event.data, event.old_data)
        elif event.operation == "DELETE":
            self._delete_features(event.data)
    
    def _create_features(self, record):
        # Extract features from raw data
        features = self.feature_extractor.extract(record)
        
        # Apply transformations
        transformed = self.transformer.transform(features)
        
        # Store in feature store with TTL
        self.feature_store.set(
            f"feature:{record['user_id']}:{record['timestamp']}",
            json.dumps(transformed),
            ex=3600  # 1 hour TTL
        )
        
        # Publish to model inference topic
        self.kafka_producer.send(
            "realtime-inference-requests",
            key=str(record['user_id']).encode(),
            value=json.dumps({
                "features": transformed,
                "model_version": self.model_registry.get_latest_version()
            })
        )
```

## AI/ML Specific Use Cases

### Real-Time Model Monitoring
- **Drift detection**: Monitor feature distribution changes in real-time
- **Performance tracking**: Capture prediction outcomes immediately
- **Anomaly detection**: Identify outliers as they occur

### Streaming Feature Engineering
- **Session-based features**: Build features across user sessions
- **Temporal aggregation**: Calculate rolling statistics (last 5 minutes, 1 hour)
- **Cross-entity features**: Join related entities in stream processing

### Online Learning Systems
- **Incremental updates**: Update model parameters with new data points
- **Feedback loops**: Capture user feedback and retrain models
- **A/B testing**: Route traffic based on real-time feature values

## Performance Optimization Techniques

### Batch Processing vs Streaming
| Approach | Latency | Throughput | Complexity | Use Case |
|----------|---------|------------|------------|----------|
| Batch CDC | Minutes | High | Low | Offline training |
| Micro-batch | Seconds | Medium | Medium | Near-real-time |
| True streaming | Milliseconds | Variable | High | Real-time inference |

### Compression and Filtering
- **Delta encoding**: Only send changed fields
- **Field filtering**: Subscribe only to relevant columns
- **Compression**: Use Snappy or Zstandard for network efficiency

### Backpressure Handling
```python
class CDCBackpressureManager:
    def __init__(self, max_queue_size=10000):
        self.queue = deque(maxlen=max_queue_size)
        self.watermark = 0.8  # 80% capacity threshold
    
    def enqueue(self, change_event):
        if len(self.queue) > len(self.queue) * self.watermark:
            self._throttle()
        self.queue.append(change_event)
    
    def _throttle(self):
        # Implement exponential backoff or circuit breaker
        time.sleep(0.1 * (2 ** self.throttle_level))
        self.throttle_level += 1
```

## Real-World Production Examples

### Airbnb's Real-time Pricing Engine
- Uses CDC to capture booking changes in real-time
- Updates pricing models within 100ms of data change
- Handles 50K+ events per second during peak times

### Spotify's Recommendation System
- CDC captures user interaction events (plays, skips, likes)
- Real-time feature updates for collaborative filtering
- Achieves sub-50ms latency for feature updates

### Tesla's Autopilot Training
- CDC streams sensor data changes to training pipelines
- Real-time anomaly detection for autonomous driving
- Processes 2TB/day of streaming data

## Debugging CDC Systems

### Common Failure Modes
1. **Log lag**: CDC connector falling behind transaction log
2. **Data loss**: Network partitions causing message drops
3. **Schema evolution**: Breaking changes in source schema
4. **Duplicate processing**: Exactly-once semantics failures

### Diagnostic Tools
- **Lag monitoring**: Track consumer group lag in Kafka
- **Schema validation**: Validate CDC output against expected schema
- **End-to-end tracing**: Correlate source changes with downstream processing
- **Replay capabilities**: Test CDC pipeline with historical data

### Metrics to Monitor
- **CDC lag**: Time difference between source commit and CDC processing
- **Throughput**: Events processed per second
- **Error rate**: Failed events vs total events
- **Latency percentiles**: p50, p95, p99 processing latency

## AI/ML Integration Patterns

### Feature Store Integration
```python
class CDCFeatureStoreAdapter:
    def __init__(self, feature_store, cdc_source):
        self.feature_store = feature_store
        self.cdc_source = cdc_source
    
    def handle_cdc_event(self, event):
        if event.table == "user_interactions":
            self._update_user_features(event)
        elif event.table == "model_predictions":
            self._update_prediction_metrics(event)
        elif event.table == "training_jobs":
            self._update_training_status(event)
    
    def _update_user_features(self, event):
        # Extract user ID and interaction type
        user_id = event.data["user_id"]
        interaction_type = event.data["interaction_type"]
        
        # Update real-time features
        current_features = self.feature_store.get(f"user:{user_id}")
        updated_features = self.feature_updater.update(
            current_features, 
            interaction_type,
            event.timestamp
        )
        
        # Store with versioning
        self.feature_store.set(
            f"user:{user_id}:v{int(time.time())}",
            json.dumps(updated_features),
            ex=86400  # 24 hours
        )
```

### Model Versioning with CDC
- **Automated retraining**: Trigger retraining when data distribution shifts
- **Version promotion**: Promote models based on real-time performance
- **Canary deployment**: Route traffic based on CDC-derived metrics

## Best Practices for Senior Engineers

1. **Start with log-based CDC**: Lowest overhead for production systems
2. **Implement exactly-once semantics**: Critical for ML training data
3. **Monitor CDC lag continuously**: Set alerts for abnormal lag
4. **Design for schema evolution**: Handle breaking changes gracefully
5. **Test with production-like data volumes**: Simulate peak loads

## Related Resources
- [Database Case Study: Uber's Real-time Feature Pipeline](../06_case_studies/uber_realtime_pipeline.md)
- [System Design: Streaming ML Infrastructure](../03_system_design/streaming_ml_infrastructure.md)
- [Debugging Patterns: CDC Lag Analysis](../05_interview_prep/database_debugging_patterns.md)

---
*Last updated: February 2026 | Target audience: Senior AI/ML Engineers*