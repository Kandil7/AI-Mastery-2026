# Database Integration with Real-time Streaming Platforms Tutorial

## Overview

This tutorial focuses on integrating databases with real-time streaming platforms: Apache Kafka, Redis Streams, and modern event-driven architectures. We'll cover real-time feature engineering, stream processing, anomaly detection, and production deployment patterns specifically for senior AI/ML engineers building real-time AI systems.

## Prerequisites
- Python 3.8+
- Apache Kafka (with confluent-kafka-python)
- Redis (with redis-py)
- PostgreSQL/MySQL with CDC capabilities
- Basic understanding of stream processing

## Tutorial Structure
1. **Kafka Integration** - Database-to-Kafka pipelines
2. **Redis Streams Integration** - Real-time feature storage
3. **CDC (Change Data Capture) Integration** - Real-time database changes
4. **Stream Processing with AI** - Real-time ML inference
5. **Real-time Anomaly Detection** - Database-backed monitoring
6. **Performance Benchmarking** - Streaming platform comparison

## Section 1: Kafka Integration

### Step 1: Database-to-Kafka CDC pipeline
```python
from confluent_kafka import Producer, Consumer, KafkaError
import psycopg2
import json
import time
from typing import Dict, List

class KafkaDatabaseIntegration:
    def __init__(self, kafka_config: Dict, db_config: Dict):
        self.kafka_config = kafka_config
        self.db_config = db_config
        self.producer = Producer(kafka_config)
    
    def setup_kafka_producer(self):
        """Setup Kafka producer with proper configuration"""
        # Producer configuration
        config = {
            'bootstrap.servers': self.kafka_config['bootstrap_servers'],
            'client.id': 'database-producer',
            'linger.ms': 10,
            'batch.num.messages': 1000,
            'compression.type': 'snappy',
            'enable.idempotence': True,
            'max.in.flight.requests.per.connection': 5
        }
        
        return Producer(config)
    
    def create_cdc_listener(self, table_name: str, topic_name: str):
        """Create CDC listener for database changes"""
        # In practice, use Debezium or similar CDC tool
        # This is a simplified simulation
        
        def delivery_report(err, msg):
            if err is not None:
                print(f'Message delivery failed: {err}')
            else:
                print(f'Message delivered to {msg.topic()} [{msg.partition()}]')
        
        # Simulate CDC processing
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        # Listen for changes (simplified)
        try:
            while True:
                # Simulate getting changes
                changes = self._get_simulated_changes(cursor, table_name)
                
                for change in changes:
                    # Format as CDC message
                    cdc_message = {
                        'op': change['operation'],  # 'insert', 'update', 'delete'
                        'table': table_name,
                        'before': change.get('before'),
                        'after': change.get('after'),
                        'ts_ms': int(time.time() * 1000),
                        'source': {
                            'server': 'postgres-server',
                            'ts_ms': int(time.time() * 1000)
                        }
                    }
                    
                    # Send to Kafka
                    self.producer.produce(
                        topic_name,
                        key=str(change['id']).encode('utf-8'),
                        value=json.dumps(cdc_message).encode('utf-8'),
                        callback=delivery_report
                    )
                    self.producer.poll(0)
                
                time.sleep(1)  # Simulate polling interval
                
        except KeyboardInterrupt:
            pass
        finally:
            self.producer.flush()
            cursor.close()
            conn.close()
    
    def _get_simulated_changes(self, cursor, table_name: str) -> List[Dict]:
        """Simulate getting database changes"""
        # In real implementation, this would use logical replication or triggers
        return [
            {
                'id': 123,
                'operation': 'insert',
                'before': None,
                'after': {
                    'user_id': 123,
                    'event_type': 'click',
                    'timestamp': '2024-01-01T12:00:00Z',
                    'data': {'page': '/home', 'duration': 30}
                }
            }
        ]
    
    def create_kafka_consumer(self, topic_name: str, group_id: str):
        """Create Kafka consumer for real-time processing"""
        consumer_config = {
            'bootstrap.servers': self.kafka_config['bootstrap_servers'],
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,
            'max.poll.interval.ms': 300000,
            'session.timeout.ms': 10000
        }
        
        consumer = Consumer(consumer_config)
        consumer.subscribe([topic_name])
        
        return consumer
    
    def process_real_time_events(self, consumer, processor_func):
        """Process real-time events from Kafka"""
        try:
            while True:
                msg = consumer.poll(1.0)
                
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        print(f"Error: {msg.error()}")
                        break
                
                # Process message
                try:
                    value = json.loads(msg.value().decode('utf-8'))
                    result = processor_func(value)
                    
                    # Commit offset
                    consumer.commit(asynchronous=False)
                    
                except Exception as e:
                    print(f"Processing error: {e}")
                    # Handle error (retry, dead letter queue, etc.)
        
        except KeyboardInterrupt:
            pass
        finally:
            consumer.close()

# Usage example
kafka_config = {
    'bootstrap_servers': 'localhost:9092'
}

db_config = {
    'host': 'localhost',
    'database': 'ai_db',
    'user': 'postgres',
    'password': 'password'
}

kafka_integration = KafkaDatabaseIntegration(kafka_config, db_config)

# Create CDC listener (in practice, use Debezium)
# kafka_integration.create_cdc_listener("user_events", "user-events-cdc")
```

### Step 2: Real-time feature engineering with Kafka
```python
from datetime import datetime, timedelta
from collections import defaultdict
import json

class KafkaFeatureEngineer:
    def __init__(self, kafka_config: Dict):
        self.kafka_config = kafka_config
        self.feature_store = defaultdict(dict)
        self.window_sizes = [60, 300, 3600]  # 1min, 5min, 1hr
    
    def create_feature_processor(self, topic_name: str, output_topic: str):
        """Create feature processor for real-time feature engineering"""
        consumer = self._create_kafka_consumer(topic_name, "feature-processor")
        producer = Producer(self.kafka_config)
        
        def process_event(event: Dict):
            """Process single event and compute features"""
            user_id = event['after']['user_id']
            event_type = event['after']['event_type']
            timestamp = datetime.fromisoformat(event['after']['timestamp'].replace('Z', '+00:00'))
            
            # Update feature store
            user_features = self.feature_store[user_id]
            
            # Update counters
            if 'event_count' not in user_features:
                user_features['event_count'] = 0
            user_features['event_count'] += 1
            
            # Update time-based windows
            for window_size in self.window_sizes:
                window_key = f"window_{window_size}"
                if window_key not in user_features:
                    user_features[window_key] = {
                        'count': 0,
                        'last_reset': timestamp
                    }
                
                # Check if window needs reset
                if timestamp - user_features[window_key]['last_reset'] > timedelta(seconds=window_size):
                    user_features[window_key]['count'] = 0
                    user_features[window_key]['last_reset'] = timestamp
                
                user_features[window_key]['count'] += 1
            
            # Compute real-time features
            real_time_features = {
                'user_id': user_id,
                'current_event_type': event_type,
                'event_count_1min': user_features.get('window_60', {}).get('count', 0),
                'event_count_5min': user_features.get('window_300', {}).get('count', 0),
                'event_count_1hr': user_features.get('window_3600', {}).get('count', 0),
                'total_event_count': user_features.get('event_count', 0),
                'timestamp': timestamp.isoformat()
            }
            
            # Send to output topic
            producer.produce(
                output_topic,
                key=str(user_id).encode('utf-8'),
                value=json.dumps(real_time_features).encode('utf-8')
            )
            
            return real_time_features
        
        return process_event
    
    def _create_kafka_consumer(self, topic_name: str, group_id: str):
        """Create Kafka consumer"""
        consumer_config = {
            'bootstrap.servers': self.kafka_config['bootstrap_servers'],
            'group.id': group_id,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False
        }
        
        return Consumer(consumer_config)

# Usage example
feature_engineer = KafkaFeatureEngineer(kafka_config)

# Create feature processor
processor = feature_engineer.create_feature_processor(
    "user-events-cdc",
    "real-time-features"
)

# In real implementation, this would be run in a Kafka consumer loop
```

## Section 2: Redis Streams Integration

### Step 1: Redis Streams for real-time feature storage
```python
import redis
import json
import time
from typing import Dict, List

class RedisStreamsIntegration:
    def __init__(self, redis_config: Dict):
        self.redis_client = redis.Redis(**redis_config)
    
    def create_stream_for_features(self, stream_name: str):
        """Create Redis stream for real-time features"""
        # Redis streams are created automatically on first write
        return stream_name
    
    def ingest_event_to_stream(self, stream_name: str, event_data: Dict):
        """Ingest event to Redis stream"""
        # Convert to Redis hash format
        stream_entry = {
            'event_id': event_data.get('event_id', str(int(time.time() * 1000))),
            'user_id': str(event_data.get('user_id')),
            'event_type': event_data.get('event_type', 'unknown'),
            'timestamp': event_data.get('timestamp', time.time()),
            'data': json.dumps(event_data.get('data', {}))
        }
        
        # Add to stream
        result = self.redis_client.xadd(stream_name, stream_entry)
        
        return result
    
    def read_from_stream(self, stream_name: str, last_id: str = '$'):
        """Read from Redis stream"""
        # Read new messages
        messages = self.redis_client.xread({stream_name: last_id}, count=10, block=1000)
        
        if messages:
            stream_key, messages_list = messages[0]
            return [
                {
                    'id': msg_id,
                    'data': {k.decode('utf-8'): v.decode('utf-8') for k, v in msg_data.items()}
                }
                for msg_id, msg_data in messages_list
            ]
        return []
    
    def create_real_time_feature_store(self, feature_stream_name: str):
        """Create real-time feature store using Redis streams"""
        # Use Redis hashes for feature storage
        def update_user_features(user_id: str, features: Dict):
            """Update user features in Redis"""
            # Use Redis hash for user features
            feature_key = f"user_features:{user_id}"
            
            # Update hash
            self.redis_client.hset(feature_key, mapping=features)
            
            # Set expiration for real-time features
            self.redis_client.expire(feature_key, 3600)  # 1 hour expiration
            
            return True
        
        def get_user_features(user_id: str):
            """Get user features from Redis"""
            feature_key = f"user_features:{user_id}"
            features = self.redis_client.hgetall(feature_key)
            
            return {k.decode('utf-8'): v.decode('utf-8') for k, v in features.items()}
        
        return update_user_features, get_user_features

# Usage example
redis_config = {
    'host': 'localhost',
    'port': 6379,
    'db': 0
}

redis_integration = RedisStreamsIntegration(redis_config)

# Create feature store
update_features, get_features = redis_integration.create_real_time_feature_store("user-features")

# Update features
update_features("123", {
    "event_count_1min": "5",
    "event_count_5min": "12",
    "engagement_score": "0.85"
})

# Get features
features = get_features("123")
print("User features:", features)
```

### Step 2: Real-time inference with Redis features
```python
class RealTimeInferenceWithRedis:
    def __init__(self, redis_client: redis.Redis, model_path: str = None):
        self.redis_client = redis_client
        self.model_path = model_path
        self.model = self._load_model(model_path)
    
    def _load_model(self, model_path: str):
        """Load model (simplified)"""
        # In practice, load actual ML model
        return "loaded_model"
    
    def get_real_time_features(self, user_id: str):
        """Get real-time features from Redis"""
        feature_key = f"user_features:{user_id}"
        features = self.redis_client.hgetall(feature_key)
        
        if not features:
            # Return default features for new users
            return {
                'event_count_1min': '0',
                'event_count_5min': '0',
                'event_count_1hr': '0',
                'engagement_score': '0.0'
            }
        
        return {k.decode('utf-8'): v.decode('utf-8') for k, v in features.items()}
    
    def predict_with_real_time_features(self, user_id: str):
        """Make prediction using real-time features"""
        # Get features
        features = self.get_real_time_features(user_id)
        
        # Convert to numerical values
        numeric_features = {
            'event_count_1min': float(features.get('event_count_1min', '0')),
            'event_count_5min': float(features.get('event_count_5min', '0')),
            'event_count_1hr': float(features.get('event_count_1hr', '0')),
            'engagement_score': float(features.get('engagement_score', '0.0'))
        }
        
        # Make prediction (simplified)
        # In practice, use loaded model
        prediction = (
            0.3 * numeric_features['event_count_1min'] +
            0.2 * numeric_features['event_count_5min'] +
            0.1 * numeric_features['event_count_1hr'] +
            0.4 * numeric_features['engagement_score']
        )
        
        return {
            'user_id': user_id,
            'prediction': min(max(prediction, 0.0), 1.0),  # Clamp between 0-1
            'features_used': numeric_features,
            'timestamp': time.time()
        }

# Usage example
inference_engine = RealTimeInferenceWithRedis(redis_integration.redis_client)

# Make real-time prediction
result = inference_engine.predict_with_real_time_features("123")
print("Real-time prediction:", result)
```

## Section 3: CDC (Change Data Capture) Integration

### Step 1: Database CDC with Debezium
```python
import requests
import json
import time

class DebeziumCDCIntegration:
    def __init__(self, debezium_url: str, db_config: Dict):
        self.debezium_url = debezium_url
        self.db_config = db_config
    
    def setup_debezium_connector(self, connector_name: str, 
                               database_server_name: str,
                               database_name: str):
        """Setup Debezium connector for CDC"""
        connector_config = {
            "name": connector_name,
            "config": {
                "connector.class": "io.debezium.connector.postgresql.PostgresConnector",
                "tasks.max": "1",
                "database.server.name": database_server_name,
                "database.hostname": self.db_config['host'],
                "database.port": str(self.db_config.get('port', 5432)),
                "database.user": self.db_config['user'],
                "database.password": self.db_config['password'],
                "database.dbname": database_name,
                "database.include.schema.changes": "false",
                "plugin.name": "pgoutput",
                "slot.name": "debezium_slot",
                "publication.name": "dbz_publication",
                "snapshot.mode": "initial",
                "transforms": "unwrap",
                "transforms.unwrap.type": "io.debezium.transforms.ExtractNewRecordState",
                "transforms.unwrap.drop.tombstones": "false"
            }
        }
        
        # Create connector
        response = requests.post(
            f"{self.debezium_url}/connectors",
            headers={"Content-Type": "application/json"},
            data=json.dumps(connector_config)
        )
        
        return response.json()
    
    def consume_cdc_events(self, topic_name: str, consumer_group: str):
        """Consume CDC events from Kafka"""
        # In practice, use Kafka consumer
        # This is a simplified simulation
        
        def process_cdc_event(event: Dict):
            """Process CDC event"""
            # Extract operation type
            op = event['op']
            
            if op == 'c':  # create
                return self._handle_insert(event)
            elif op == 'u':  # update
                return self._handle_update(event)
            elif op == 'd':  # delete
                return self._handle_delete(event)
        
        return process_cdc_event
    
    def _handle_insert(self, event: Dict):
        """Handle insert operation"""
        after = event['after']
        table = event['source']['table']
        
        # Extract relevant fields
        user_id = after.get('user_id')
        event_type = after.get('event_type')
        
        # Generate real-time features
        real_time_features = {
            'user_id': user_id,
            'event_type': event_type,
            'timestamp': event['source']['ts_ms'],
            'operation': 'insert'
        }
        
        return real_time_features
    
    def _handle_update(self, event: Dict):
        """Handle update operation"""
        before = event['before']
        after = event['after']
        table = event['source']['table']
        
        # Calculate changes
        changes = {}
        for key in after:
            if key in before and before[key] != after[key]:
                changes[key] = {'before': before[key], 'after': after[key]}
        
        return {
            'user_id': after.get('user_id'),
            'table': table,
            'changes': changes,
            'timestamp': event['source']['ts_ms'],
            'operation': 'update'
        }

# Usage example
debezium_integration = DebeziumCDCIntegration(
    "http://localhost:8083",
    db_config
)

# Setup connector
# connector_result = debezium_integration.setup_debezium_connector(
#     "postgres-connector",
#     "postgres-server",
#     "ai_db"
# )
```

## Section 4: Stream Processing with AI

### Step 1: Real-time ML inference with Kafka streams
```python
from confluent_kafka import Producer, Consumer
import torch
import numpy as np

class KafkaStreamMLProcessor:
    def __init__(self, kafka_config: Dict, model_path: str = None):
        self.kafka_config = kafka_config
        self.model = self._load_model(model_path)
        self.producer = Producer(kafka_config)
    
    def _load_model(self, model_path: str):
        """Load ML model"""
        # In practice, load actual model
        return torch.nn.Linear(4, 1)  # Simplified model
    
    def create_inference_processor(self, input_topic: str, output_topic: str):
        """Create real-time inference processor"""
        consumer_config = {
            'bootstrap.servers': self.kafka_config['bootstrap_servers'],
            'group.id': 'ml-inference-group',
            'auto.offset.reset': 'latest',
            'enable.auto.commit': False
        }
        
        consumer = Consumer(consumer_config)
        consumer.subscribe([input_topic])
        
        def process_message(msg):
            """Process single message and make inference"""
            try:
                value = json.loads(msg.value().decode('utf-8'))
                
                # Extract features
                features = [
                    float(value.get('event_count_1min', 0)),
                    float(value.get('event_count_5min', 0)),
                    float(value.get('event_count_1hr', 0)),
                    float(value.get('engagement_score', 0.0))
                ]
                
                # Convert to tensor
                input_tensor = torch.tensor([features], dtype=torch.float32)
                
                # Make prediction
                with torch.no_grad():
                    prediction = torch.sigmoid(self.model(input_tensor)).item()
                
                # Prepare output
                output = {
                    'user_id': value.get('user_id'),
                    'prediction': prediction,
                    'features': features,
                    'timestamp': time.time(),
                    'source': 'real-time-inference'
                }
                
                # Send to output topic
                self.producer.produce(
                    output_topic,
                    key=str(value.get('user_id')).encode('utf-8'),
                    value=json.dumps(output).encode('utf-8')
                )
                self.producer.poll(0)
                
                return output
                
            except Exception as e:
                print(f"Processing error: {e}")
                return None
        
        return consumer, process_message
    
    def start_stream_processing(self, input_topic: str, output_topic: str):
        """Start stream processing"""
        consumer, processor = self.create_inference_processor(input_topic, output_topic)
        
        try:
            while True:
                msg = consumer.poll(1.0)
                
                if msg is None:
                    continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        continue
                    else:
                        print(f"Error: {msg.error()}")
                        break
                
                result = processor(msg)
                if result:
                    print(f"Processed: {result['user_id']} -> {result['prediction']:.3f}")
        
        except KeyboardInterrupt:
            pass
        finally:
            consumer.close()
            self.producer.flush()

# Usage example
stream_processor = KafkaStreamMLProcessor(kafka_config)

# Start processing (in practice, run in background)
# stream_processor.start_stream_processing("real-time-features", "predictions")
```

## Section 5: Real-time Anomaly Detection

### Step 1: Database-backed anomaly detection
```python
import numpy as np
from collections import deque
import time

class RealTimeAnomalyDetector:
    def __init__(self, window_size: int = 100, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.feature_windows = {}
        self.anomaly_log = []
    
    def update_feature_window(self, user_id: str, feature_name: str, value: float):
        """Update feature window for anomaly detection"""
        if user_id not in self.feature_windows:
            self.feature_windows[user_id] = {}
        
        if feature_name not in self.feature_windows[user_id]:
            self.feature_windows[user_id][feature_name] = deque(maxlen=self.window_size)
        
        self.feature_windows[user_id][feature_name].append(value)
    
    def detect_anomaly(self, user_id: str, feature_name: str, current_value: float):
        """Detect anomaly using statistical methods"""
        if user_id not in self.feature_windows or \
           feature_name not in self.feature_windows[user_id]:
            return False, 0.0
        
        window = self.feature_windows[user_id][feature_name]
        if len(window) < 10:  # Need minimum samples
            return False, 0.0
        
        # Calculate statistics
        mean = np.mean(window)
        std = np.std(window)
        
        # Z-score
        z_score = abs((current_value - mean) / std) if std > 0 else 0
        
        is_anomaly = z_score > self.threshold
        
        return is_anomaly, z_score
    
    def process_real_time_event(self, event: Dict):
        """Process real-time event for anomaly detection"""
        user_id = event.get('user_id')
        timestamp = event.get('timestamp')
        
        # Extract features for anomaly detection
        features_to_check = [
            ('event_count_1min', event.get('event_count_1min', 0)),
            ('event_count_5min', event.get('event_count_5min', 0)),
            ('engagement_score', event.get('engagement_score', 0.0'))
        ]
        
        anomalies = []
        for feature_name, value in features_to_check:
            is_anomaly, z_score = self.detect_anomaly(user_id, feature_name, float(value))
            if is_anomaly:
                anomalies.append({
                    'feature': feature_name,
                    'value': value,
                    'z_score': z_score,
                    'threshold': self.threshold
                })
        
        # Log anomalies
        if anomalies:
            self.anomaly_log.append({
                'user_id': user_id,
                'timestamp': timestamp,
                'anomalies': anomalies,
                'total_anomalies': len(anomalies)
            })
            
            return {
                'user_id': user_id,
                'is_anomalous': True,
                'anomalies': anomalies,
                'timestamp': timestamp
            }
        
        return {
            'user_id': user_id,
            'is_anomalous': False,
            'anomalies': [],
            'timestamp': timestamp
        }

# Usage example
anomaly_detector = RealTimeAnomalyDetector(window_size=50, threshold=2.5)

# Process events
event = {
    'user_id': '123',
    'event_count_1min': '15',
    'event_count_5min': '25',
    'engagement_score': '0.95',
    'timestamp': time.time()
}

result = anomaly_detector.process_real_time_event(event)
print("Anomaly detection result:", result)
```

### Step 2: Database-integrated anomaly monitoring
```python
class DatabaseAnomalyMonitor:
    def __init__(self, db_config: Dict, redis_client: redis.Redis = None):
        self.db_config = db_config
        self.redis_client = redis_client
        self.anomaly_detector = RealTimeAnomalyDetector()
    
    def log_anomaly_to_database(self, anomaly_data: Dict):
        """Log anomaly to database"""
        conn = psycopg2.connect(**self.db_config)
        cursor = conn.cursor()
        
        insert_query = """
        INSERT INTO anomalies (
            user_id, 
            is_anomalous, 
            anomaly_count,
            anomaly_details,
            created_at
        ) VALUES (%s, %s, %s, %s, NOW())
        """
        
        cursor.execute(insert_query, (
            anomaly_data['user_id'],
            anomaly_data['is_anomalous'],
            anomaly_data['total_anomalies'],
            json.dumps(anomaly_data['anomalies'])
        ))
        
        conn.commit()
        cursor.close()
        conn.close()
    
    def monitor_real_time_streams(self, stream_name: str):
        """Monitor real-time streams for anomalies"""
        # In practice, integrate with Kafka/Redis consumer
        # This is a simplified simulation
        
        def process_event(event: Dict):
            # Detect anomalies
            anomaly_result = self.anomaly_detector.process_real_time_event(event)
            
            # Log to database if anomalous
            if anomaly_result['is_anomalous']:
                self.log_anomaly_to_database(anomaly_result)
            
            # Store in Redis for real-time dashboard
            if self.redis_client:
                key = f"anomaly:{anomaly_result['user_id']}:{int(time.time())}"
                self.redis_client.setex(key, 3600, json.dumps(anomaly_result))
            
            return anomaly_result
        
        return process_event

# Usage example
db_anomaly_monitor = DatabaseAnomalyMonitor(db_config, redis_integration.redis_client)

# Process events (in real implementation, connected to stream processor)
# event_processor = db_anomaly_monitor.monitor_real_time_streams("real-time-features")
```

## Section 6: Performance Benchmarking

### Step 1: Streaming platform benchmarking
```python
import time
import pandas as pd
from typing import List, Dict, Callable

class StreamingBenchmark:
    def __init__(self):
        self.results = []
    
    def benchmark_event_processing(self, methods: List[Callable], 
                                 event_counts: List[int] = [1000, 10000, 100000]):
        """Benchmark event processing performance"""
        for method in methods:
            for count in event_counts:
                start_time = time.time()
                
                try:
                    method(count)
                    duration = time.time() - start_time
                    
                    self.results.append({
                        'benchmark': 'event_processing',
                        'method': method.__name__,
                        'event_count': count,
                        'duration_seconds': duration,
                        'throughput_events_per_second': count / duration if duration > 0 else 0
                    })
                except Exception as e:
                    self.results.append({
                        'benchmark': 'event_processing',
                        'method': method.__name__,
                        'event_count': count,
                        'duration_seconds': float('inf'),
                        'error': str(e)
                    })
    
    def benchmark_real_time_inference(self, methods: List[Callable],
                                    query_types: List[str] = ["simple", "complex"]):
        """Benchmark real-time inference latency"""
        for method in methods:
            for q_type in query_types:
                start_time = time.time()
                
                try:
                    method(q_type)
                    duration = time.time() - start_time
                    
                    self.results.append({
                        'benchmark': 'real_time_inference',
                        'method': method.__name__,
                        'query_type': q_type,
                        'duration_seconds': duration
                    })
                except Exception as e:
                    self.results.append({
                        'benchmark': 'real_time_inference',
                        'method': method.__name__,
                        'query_type': q_type,
                        'duration_seconds': float('inf'),
                        'error': str(e)
                    })
    
    def generate_streaming_benchmark_report(self):
        """Generate comprehensive streaming benchmark report"""
        df = pd.DataFrame(self.results)
        
        # Calculate statistics
        summary = df.groupby(['benchmark', 'method']).agg({
            'duration_seconds': ['mean', 'std', 'min', 'max'],
            'throughput_events_per_second': ['mean', 'std']
        }).round(2)
        
        # Generate recommendations
        recommendations = []
        
        # Best event processing
        if 'event_processing' in df['benchmark'].values:
            best_processing = df[df['benchmark'] == 'event_processing'].loc[
                df[df['benchmark'] == 'event_processing']['duration_seconds'].idxmin()
            ]
            recommendations.append(
                f"Best event processing: {best_processing['method']} "
                f"({best_processing['duration_seconds']:.2f}s for {best_processing['event_count']} events)"
            )
        
        # Best real-time inference
        if 'real_time_inference' in df['benchmark'].values:
            best_inference = df[df['benchmark'] == 'real_time_inference'].loc[
                df[df['benchmark'] == 'real_time_inference']['duration_seconds'].idxmin()
            ]
            recommendations.append(
                f"Best real-time inference: {best_inference['method']} "
                f"({best_inference['duration_seconds']:.2f}s for {best_inference['query_type']} queries)"
            )
        
        return {
            'summary': summary,
            'detailed_results': df,
            'recommendations': recommendations,
            'streaming_tips': [
                "Use Kafka for high-throughput, durable event processing",
                "Use Redis Streams for low-latency, real-time feature storage",
                "Combine CDC with stream processing for comprehensive real-time AI",
                "Implement proper error handling and dead letter queues",
                "Monitor stream lag and processing latency in production"
            ]
        }

# Usage example
benchmark = StreamingBenchmark()

# Define test methods
def test_kafka_processing(count: int):
    """Test Kafka event processing"""
    time.sleep(0.001 * count)

def test_redis_processing(count: int):
    """Test Redis stream processing"""
    time.sleep(0.0005 * count)

def test_real_time_inference(q_type: str):
    """Test real-time inference"""
    time.sleep(0.01)

# Run benchmarks
benchmark.benchmark_event_processing(
    [test_kafka_processing, test_redis_processing],
    [1000, 10000, 100000]
)

benchmark.benchmark_real_time_inference(
    [test_real_time_inference],
    ["simple", "complex"]
)

report = benchmark.generate_streaming_benchmark_report()
print("Streaming Platform Benchmark Report:")
print(report['summary'])
print("\nRecommendations:")
for rec in report['recommendations']:
    print(f"- {rec}")
```

## Hands-on Exercises

### Exercise 1: Kafka CDC integration
1. Set up Debezium with PostgreSQL
2. Create Kafka topics for CDC events
3. Implement real-time feature engineering
4. Test with simulated database changes

### Exercise 2: Redis Streams integration
1. Set up Redis Streams for real-time features
2. Implement feature store with expiration
3. Build real-time inference engine
4. Test latency and throughput

### Exercise 3: Real-time anomaly detection
1. Implement statistical anomaly detection
2. Integrate with database logging
3. Create real-time dashboard
4. Test with synthetic anomaly data

### Exercise 4: Stream processing with AI
1. Build Kafka stream processor
2. Integrate with ML model
3. Deploy as microservice
4. Monitor performance and reliability

## Best Practices Summary

1. **CDC Implementation**: Use Debezium for reliable database change capture
2. **Stream Processing**: Choose Kafka for durability, Redis for low latency
3. **Real-time Features**: Use appropriate window sizes for your use case
4. **Anomaly Detection**: Combine statistical methods with domain knowledge
5. **Error Handling**: Implement robust error handling and retry mechanisms
6. **Monitoring**: Track stream lag, processing latency, and error rates
7. **Scalability**: Design for horizontal scaling of stream processors
8. **Security**: Secure stream connections and implement proper authentication

This tutorial provides practical, hands-on experience with database integration for real-time streaming platforms. Complete all exercises to master these critical skills for building real-time AI systems.