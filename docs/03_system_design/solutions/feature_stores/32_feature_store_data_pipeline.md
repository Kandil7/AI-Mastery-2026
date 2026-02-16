# System Design Solution: Feature Stores and Data Pipelines for ML

## Problem Statement

Design a comprehensive feature store and data pipeline architecture that can handle petabytes of data, support real-time and batch feature computation, provide feature consistency across training and inference, enable feature discovery and sharing, implement versioning and lineage tracking, and ensure data quality with monitoring and validation. The system should support multiple data sources, handle schema evolution, and provide low-latency feature serving for online inference.

## Solution Overview

This system design presents a comprehensive architecture for feature stores and data pipelines that addresses the critical need for unified, reliable, and scalable feature management in machine learning workflows. The solution implements a hybrid offline-online architecture with robust data quality controls and comprehensive monitoring.

## 1. High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Data Sources │────│  Ingestion      │────│  Feature Store │
│   (DB, Files,  │    │  Layer          │    │  (Offline &    │
│   Streams)     │    │  (Kafka, Spark) │    │  Online)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Data Quality  │────│  Transformation │────│  Feature       │
│  Monitoring    │    │  Engine         │    │  Serving       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌────────────────────────────────┼─────────────────────────────────┐
│                    Feature Management Infrastructure           │
│  ┌─────────────────┐    ┌──────────────────┐    ┌──────────┐  │
│  │  Metadata      │────│  Lineage        │────│  Access  │  │
│  │  Store         │    │  Tracking       │    │  Control │  │
│  └─────────────────┘    └──────────────────┘    └──────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## 2. Core Components

### 2.1 Feature Store Core System
```python
import asyncio
import aioredis
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
import json
from enum import Enum

class FeatureType(Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    TEXT = "text"
    EMBEDDING = "embedding"
    BOOLEAN = "boolean"
    TIMESTAMP = "timestamp"

class StorageType(Enum):
    OFFLINE = "offline"
    ONLINE = "online"

@dataclass
class FeatureSpec:
    name: str
    feature_type: FeatureType
    storage_type: StorageType
    transformation: Optional[str] = None
    default_value: Optional[Any] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None

@dataclass
class FeatureValue:
    entity_key: str
    feature_name: str
    value: Any
    timestamp: datetime
    version: str

class FeatureStore:
    """
    Core feature store implementation
    """
    def __init__(self, offline_storage_path: str, online_redis_url: str = "redis://localhost:6379"):
        self.offline_storage_path = offline_storage_path
        self.online_redis_url = online_redis_url
        self.redis = None
        self.feature_specs: Dict[str, FeatureSpec] = {}
        self.entity_registry = {}
        
    async def initialize(self):
        """
        Initialize the feature store
        """
        self.redis = await aioredis.from_url(self.online_redis_url)
    
    async def register_feature(self, spec: FeatureSpec) -> bool:
        """
        Register a new feature specification
        """
        try:
            # Validate feature spec
            if not self._validate_feature_spec(spec):
                return False
            
            # Store feature spec
            self.feature_specs[spec.name] = spec
            
            # Create metadata entry
            metadata = {
                'name': spec.name,
                'type': spec.feature_type.value,
                'storage_type': spec.storage_type.value,
                'transformation': spec.transformation,
                'default_value': spec.default_value,
                'description': spec.description,
                'tags': spec.tags,
                'created_at': datetime.utcnow().isoformat()
            }
            
            await self.redis.set(f"feature_meta:{spec.name}", json.dumps(metadata))
            return True
            
        except Exception as e:
            print(f"Error registering feature {spec.name}: {str(e)}")
            return False
    
    def _validate_feature_spec(self, spec: FeatureSpec) -> bool:
        """
        Validate feature specification
        """
        # Check if feature name is valid
        if not spec.name or not spec.name.replace('_', '').replace('-', '').isalnum():
            return False
        
        # Check if storage type is valid
        if spec.storage_type not in [StorageType.OFFLINE, StorageType.ONLINE]:
            return False
        
        return True
    
    async def put_features(self, entity_key: str, features: Dict[str, Any], 
                          timestamp: datetime = None) -> bool:
        """
        Store features for an entity
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        try:
            for feature_name, value in features.items():
                if feature_name not in self.feature_specs:
                    raise ValueError(f"Feature {feature_name} not registered")
                
                spec = self.feature_specs[feature_name]
                
                # Store in appropriate storage based on type
                if spec.storage_type == StorageType.ONLINE:
                    await self._store_online_feature(entity_key, feature_name, value, timestamp)
                else:
                    await self._store_offline_feature(entity_key, feature_name, value, timestamp)
            
            return True
            
        except Exception as e:
            print(f"Error putting features for entity {entity_key}: {str(e)}")
            return False
    
    async def _store_online_feature(self, entity_key: str, feature_name: str, 
                                  value: Any, timestamp: datetime):
        """
        Store feature in online storage (Redis)
        """
        feature_key = f"feature:{entity_key}:{feature_name}"
        feature_value = {
            'value': value,
            'timestamp': timestamp.isoformat(),
            'version': '1.0'  # In production, implement proper versioning
        }
        
        await self.redis.setex(
            feature_key, 
            86400,  # 24 hours expiry
            json.dumps(feature_value)
        )
    
    async def _store_offline_feature(self, entity_key: str, feature_name: str, 
                                   value: Any, timestamp: datetime):
        """
        Store feature in offline storage (Parquet files)
        """
        # Create partition path based on date
        partition_date = timestamp.strftime('%Y/%m/%d')
        partition_path = f"{self.offline_storage_path}/{partition_date}"
        
        # Create feature record
        record = {
            'entity_key': entity_key,
            'feature_name': feature_name,
            'value': value,
            'timestamp': timestamp.isoformat(),
            'version': '1.0'
        }
        
        # Append to parquet file (in production, use proper batch writing)
        file_path = f"{partition_path}/{feature_name}.parquet"
        df = pd.DataFrame([record])
        
        # Write to parquet (append mode)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, file_path, append=True)
    
    async def get_features(self, entity_key: str, feature_names: List[str], 
                          timestamp: datetime = None) -> Dict[str, Any]:
        """
        Retrieve features for an entity
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        features = {}
        
        for feature_name in feature_names:
            if feature_name not in self.feature_specs:
                features[feature_name] = None
                continue
            
            spec = self.feature_specs[feature_name]
            
            # Try to get from online storage first
            if spec.storage_type == StorageType.ONLINE:
                value = await self._get_online_feature(entity_key, feature_name)
                if value is not None:
                    features[feature_name] = value
                else:
                    # Fallback to offline storage
                    value = await self._get_offline_feature(entity_key, feature_name, timestamp)
                    features[feature_name] = value
            else:
                # Get from offline storage
                value = await self._get_offline_feature(entity_key, feature_name, timestamp)
                features[feature_name] = value
        
        return features
    
    async def _get_online_feature(self, entity_key: str, feature_name: str) -> Optional[Any]:
        """
        Get feature from online storage
        """
        feature_key = f"feature:{entity_key}:{feature_name}"
        raw_value = await self.redis.get(feature_key)
        
        if raw_value:
            value_dict = json.loads(raw_value)
            return value_dict['value']
        
        return None
    
    async def _get_offline_feature(self, entity_key: str, feature_name: str, 
                                 timestamp: datetime) -> Optional[Any]:
        """
        Get feature from offline storage
        """
        # In production, implement efficient querying
        # For now, return default value or None
        spec = self.feature_specs.get(feature_name)
        if spec:
            return spec.default_value
        return None
    
    async def get_historical_features(self, entity_keys: List[str], 
                                    feature_names: List[str], 
                                    start_time: datetime, 
                                    end_time: datetime) -> pd.DataFrame:
        """
        Get historical features for entities over time period
        """
        # This would query the offline storage system
        # For now, return empty DataFrame
        columns = ['entity_key', 'timestamp'] + feature_names
        return pd.DataFrame(columns=columns)

class DataIngestionPipeline:
    """
    Data ingestion pipeline for feature store
    """
    def __init__(self, kafka_bootstrap_servers: str, feature_store: FeatureStore):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.feature_store = feature_store
        self.consumers = {}
        self.producers = {}
    
    async def setup_source_connection(self, source_config: Dict[str, Any]):
        """
        Setup connection to data source
        """
        source_type = source_config.get('type')
        
        if source_type == 'kafka':
            from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
            topic = source_config['topic']
            
            consumer = AIOKafkaConsumer(
                topic,
                bootstrap_servers=self.kafka_bootstrap_servers,
                value_deserializer=lambda x: json.loads(x.decode('utf-8'))
            )
            
            await consumer.start()
            self.consumers[topic] = consumer
            
        elif source_type == 'database':
            # Setup database connection
            pass
        elif source_type == 'file':
            # Setup file watcher
            pass
    
    async def process_stream(self, topic: str, batch_size: int = 1000):
        """
        Process streaming data from Kafka
        """
        consumer = self.consumers.get(topic)
        if not consumer:
            raise ValueError(f"No consumer for topic {topic}")
        
        batch = []
        async for msg in consumer:
            record = msg.value
            batch.append(record)
            
            if len(batch) >= batch_size:
                await self._process_batch(batch)
                batch = []
    
    async def _process_batch(self, batch: List[Dict[str, Any]]):
        """
        Process a batch of records
        """
        for record in batch:
            entity_key = record.get('entity_key')
            features = {k: v for k, v in record.items() if k != 'entity_key'}
            
            if entity_key:
                await self.feature_store.put_features(entity_key, features)

class FeatureTransformationEngine:
    """
    Engine for applying transformations to features
    """
    def __init__(self):
        self.transformations = {
            'normalize': self._normalize,
            'standardize': self._standardize,
            'categorical_encode': self._categorical_encode,
            'aggregate': self._aggregate
        }
    
    def apply_transformation(self, data: pd.Series, transform_type: str, 
                           params: Dict[str, Any] = None) -> pd.Series:
        """
        Apply transformation to data
        """
        if transform_type not in self.transformations:
            raise ValueError(f"Unknown transformation: {transform_type}")
        
        return self.transformations[transform_type](data, params)
    
    def _normalize(self, data: pd.Series, params: Dict[str, Any] = None) -> pd.Series:
        """
        Normalize data to 0-1 range
        """
        min_val = params.get('min', data.min()) if params else data.min()
        max_val = params.get('max', data.max()) if params else data.max()
        
        return (data - min_val) / (max_val - min_val)
    
    def _standardize(self, data: pd.Series, params: Dict[str, Any] = None) -> pd.Series:
        """
        Standardize data (z-score normalization)
        """
        mean_val = params.get('mean', data.mean()) if params else data.mean()
        std_val = params.get('std', data.std()) if params else data.std()
        
        return (data - mean_val) / std_val
    
    def _categorical_encode(self, data: pd.Series, params: Dict[str, Any] = None) -> pd.Series:
        """
        Encode categorical data
        """
        if params and 'mapping' in params:
            mapping = params['mapping']
        else:
            # Create mapping from unique values
            unique_vals = data.unique()
            mapping = {val: idx for idx, val in enumerate(unique_vals)}
        
        return data.map(mapping).fillna(-1)  # -1 for unseen categories
    
    def _aggregate(self, data: pd.Series, params: Dict[str, Any] = None) -> pd.Series:
        """
        Apply aggregation function
        """
        agg_func = params.get('function', 'mean') if params else 'mean'
        
        if agg_func == 'mean':
            return pd.Series([data.mean()] * len(data))
        elif agg_func == 'sum':
            return pd.Series([data.sum()] * len(data))
        elif agg_func == 'count':
            return pd.Series([len(data)] * len(data))
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")

class DataQualityMonitor:
    """
    Monitor data quality in the feature store
    """
    def __init__(self, feature_store: FeatureStore):
        self.feature_store = feature_store
        self.quality_rules = {}
        self.alerts = []
    
    def add_quality_rule(self, feature_name: str, rule_type: str, 
                        threshold: float, description: str = ""):
        """
        Add a data quality rule
        """
        rule = {
            'feature_name': feature_name,
            'type': rule_type,
            'threshold': threshold,
            'description': description,
            'created_at': datetime.utcnow().isoformat()
        }
        
        if feature_name not in self.quality_rules:
            self.quality_rules[feature_name] = []
        
        self.quality_rules[feature_name].append(rule)
    
    async def check_quality(self, entity_key: str, features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Check quality of features
        """
        violations = []
        
        for feature_name, value in features.items():
            if feature_name in self.quality_rules:
                for rule in self.quality_rules[feature_name]:
                    violation = self._check_single_rule(value, rule)
                    if violation:
                        violations.append(violation)
        
        return violations
    
    def _check_single_rule(self, value: Any, rule: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Check if a single rule is violated
        """
        rule_type = rule['type']
        threshold = rule['threshold']
        
        if rule_type == 'min_value' and value < threshold:
            return {
                'feature_name': rule['feature_name'],
                'rule_type': rule_type,
                'threshold': threshold,
                'actual_value': value,
                'violation': f"Value {value} is below minimum {threshold}"
            }
        elif rule_type == 'max_value' and value > threshold:
            return {
                'feature_name': rule['feature_name'],
                'rule_type': rule_type,
                'threshold': threshold,
                'actual_value': value,
                'violation': f"Value {value} is above maximum {threshold}"
            }
        elif rule_type == 'null_check' and value is None:
            return {
                'feature_name': rule['feature_name'],
                'rule_type': rule_type,
                'threshold': threshold,
                'actual_value': value,
                'violation': f"Value is null but should not be"
            }
        
        return None
```

### 2.2 Feature Discovery and Metadata Management
```python
class FeatureDiscoveryService:
    """
    Service for discovering and exploring features
    """
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = None
        self.redis_url = redis_url
    
    async def initialize(self):
        """
        Initialize Redis connection
        """
        self.redis = await aioredis.from_url(self.redis_url)
    
    async def search_features(self, query: str, tags: List[str] = None, 
                            limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for features by name, description, or tags
        """
        # In production, use proper search indexing
        # For now, return all features that match
        all_features = await self.redis.keys("feature_meta:*")
        results = []
        
        for feature_key in all_features:
            feature_meta_raw = await self.redis.get(feature_key)
            if feature_meta_raw:
                feature_meta = json.loads(feature_meta_raw)
                
                # Check if query matches
                matches = (
                    query.lower() in feature_meta['name'].lower() or
                    (feature_meta.get('description') and query.lower() in feature_meta['description'].lower())
                )
                
                # Check tags if provided
                if tags and feature_meta.get('tags'):
                    tag_match = any(tag in feature_meta['tags'] for tag in tags)
                    matches = matches and tag_match
                
                if matches:
                    results.append(feature_meta)
        
        # Sort by relevance (simplified)
        results.sort(key=lambda x: x['name'])
        return results[:limit]
    
    async def get_feature_lineage(self, feature_name: str) -> Dict[str, Any]:
        """
        Get lineage information for a feature
        """
        lineage_key = f"lineage:{feature_name}"
        lineage_data = await self.redis.get(lineage_key)
        
        if lineage_data:
            return json.loads(lineage_data)
        
        # If not found, create basic lineage
        lineage = {
            'feature_name': feature_name,
            'sources': [],  # Would be populated from actual data sources
            'transformations': [],  # Would be populated from transformation logs
            'dependents': [],  # Models or other features that depend on this
            'created_at': datetime.utcnow().isoformat()
        }
        
        return lineage
    
    async def get_feature_statistics(self, feature_name: str, 
                                   time_window_hours: int = 24) -> Dict[str, Any]:
        """
        Get statistics for a feature
        """
        # This would aggregate statistics from the feature store
        # For now, return placeholder
        return {
            'feature_name': feature_name,
            'time_window_hours': time_window_hours,
            'sample_size': 1000,
            'mean': 0.0,
            'std_dev': 1.0,
            'min': 0.0,
            'max': 1.0,
            'null_percentage': 0.0,
            'updated_at': datetime.utcnow().isoformat()
        }

class FeatureLineageTracker:
    """
    Track feature lineage and dependencies
    """
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = None
        self.redis_url = redis_url
    
    async def initialize(self):
        """
        Initialize Redis connection
        """
        self.redis = await aioredis.from_url(self.redis_url)
    
    async def record_transformation(self, input_features: List[str], 
                                  output_feature: str, 
                                  transformation_details: Dict[str, Any]):
        """
        Record a transformation that creates an output feature from input features
        """
        transformation_record = {
            'input_features': input_features,
            'output_feature': output_feature,
            'transformation_details': transformation_details,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Store transformation record
        trans_key = f"transformation:{output_feature}:{datetime.utcnow().isoformat()}"
        await self.redis.set(trans_key, json.dumps(transformation_record))
        
        # Update lineage for output feature
        lineage_key = f"lineage:{output_feature}"
        current_lineage = await self.redis.get(lineage_key)
        if current_lineage:
            lineage = json.loads(current_lineage)
        else:
            lineage = {
                'feature_name': output_feature,
                'sources': [],
                'transformations': [],
                'dependents': []
            }
        
        # Add input features as sources
        for input_feat in input_features:
            if input_feat not in lineage['sources']:
                lineage['sources'].append(input_feat)
        
        # Add transformation
        lineage['transformations'].append(transformation_details)
        
        await self.redis.set(lineage_key, json.dumps(lineage))
    
    async def get_impact_analysis(self, feature_name: str) -> Dict[str, Any]:
        """
        Get impact analysis for a feature (which downstream features/models are affected)
        """
        # This would traverse the dependency graph
        # For now, return placeholder
        return {
            'feature_name': feature_name,
            'affected_features': [],
            'affected_models': [],
            'impact_level': 'low'  # low, medium, high
        }

class FeatureAccessControl:
    """
    Control access to features
    """
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = None
        self.redis_url = redis_url
    
    async def initialize(self):
        """
        Initialize Redis connection
        """
        self.redis = await aioredis.from_url(self.redis_url)
    
    async def grant_access(self, user_id: str, feature_names: List[str], 
                          permissions: List[str]):
        """
        Grant access permissions to user for features
        """
        for feature_name in feature_names:
            permission_key = f"permissions:{user_id}:{feature_name}"
            await self.redis.sadd(permission_key, *permissions)
    
    async def check_access(self, user_id: str, feature_name: str, 
                          permission: str) -> bool:
        """
        Check if user has permission for feature
        """
        permission_key = f"permissions:{user_id}:{feature_name}"
        user_permissions = await self.redis.smembers(permission_key)
        
        return permission.encode() in user_permissions
    
    async def get_user_features(self, user_id: str) -> List[str]:
        """
        Get list of features accessible to user
        """
        # This would scan all permission keys for the user
        # For now, return placeholder
        all_permissions = await self.redis.keys(f"permissions:{user_id}:*")
        feature_names = []
        
        for perm_key in all_permissions:
            parts = perm_key.decode().split(':')
            if len(parts) >= 3:
                feature_names.append(parts[2])
        
        return list(set(feature_names))
```

### 2.3 Batch and Real-time Processing Pipelines
```python
import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.io import ReadFromKafka, WriteToBigQuery

class BatchFeaturePipeline:
    """
    Apache Beam pipeline for batch feature computation
    """
    def __init__(self, project_id: str, temp_location: str):
        self.project_id = project_id
        self.temp_location = temp_location
    
    def create_pipeline(self, input_topic: str, output_table: str):
        """
        Create batch processing pipeline
        """
        options = PipelineOptions([
            '--project', self.project_id,
            '--temp_location', self.temp_location,
            '--runner', 'DataflowRunner',  # Or DirectRunner for local testing
            '--streaming', 'false'
        ])
        
        with beam.Pipeline(options=options) as pipeline:
            # Read from Kafka
            messages = (
                pipeline
                | 'Read from Kafka' >> ReadFromKafka(
                    consumer_config={'bootstrap.servers': 'localhost:9092'},
                    topics=[input_topic]
                )
            )
            
            # Parse and transform data
            parsed_data = (
                messages
                | 'Parse JSON' >> beam.Map(lambda x: json.loads(x[1].decode('utf-8')))
                | 'Transform Features' >> beam.Map(self._transform_features)
            )
            
            # Write to BigQuery or other storage
            parsed_data | 'Write to BigQuery' >> WriteToBigQuery(
                output_table,
                schema='entity_key:STRING, feature_name:STRING, value:FLOAT, timestamp:TIMESTAMP',
                create_disposition=beam.io.BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=beam.io.BigQueryDisposition.WRITE_APPEND
            )
    
    def _transform_features(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform raw record into feature format
        """
        # Apply business logic and transformations
        transformed = {
            'entity_key': record.get('entity_key'),
            'feature_name': record.get('feature_name'),
            'value': self._apply_business_logic(record),
            'timestamp': datetime.utcnow().isoformat()
        }
        return transformed
    
    def _apply_business_logic(self, record: Dict[str, Any]) -> Any:
        """
        Apply business logic to compute features
        """
        # This would contain actual feature computation logic
        # For example: aggregations, ratios, derived metrics
        return record.get('value', 0)

class RealTimeFeaturePipeline:
    """
    Real-time feature computation pipeline
    """
    def __init__(self, kafka_bootstrap_servers: str, feature_store: FeatureStore):
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.feature_store = feature_store
        self.kafka_consumer = None
        self.kafka_producer = None
    
    async def initialize(self):
        """
        Initialize Kafka connections
        """
        from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
        
        self.kafka_consumer = AIOKafkaConsumer(
            'raw_events',
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
        
        self.kafka_producer = AIOKafkaProducer(
            bootstrap_servers=self.kafka_bootstrap_servers,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        
        await self.kafka_consumer.start()
        await self.kafka_producer.start()
    
    async def start_processing(self):
        """
        Start real-time feature processing
        """
        async for msg in self.kafka_consumer:
            try:
                raw_event = msg.value
                
                # Compute real-time features
                features = await self._compute_real_time_features(raw_event)
                
                # Store features
                entity_key = raw_event.get('entity_key', 'unknown')
                await self.feature_store.put_features(entity_key, features)
                
                # Optionally, publish computed features to another topic
                feature_msg = {
                    'entity_key': entity_key,
                    'features': features,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                await self.kafka_producer.send('computed_features', feature_msg)
                
            except Exception as e:
                print(f"Error processing event: {str(e)}")
    
    async def _compute_real_time_features(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute real-time features from event
        """
        features = {}
        
        # Example real-time features:
        # 1. Count features (events in last N minutes)
        if 'event_type' in event:
            features['event_count_last_5min'] = await self._get_event_count(
                event.get('entity_key'), 
                event.get('event_type'), 
                minutes=5
            )
        
        # 2. Aggregation features (sum, avg, etc.)
        if 'value' in event:
            features['rolling_sum_10_events'] = await self._get_rolling_sum(
                event.get('entity_key'), 
                event.get('value'), 
                count=10
            )
        
        # 3. Behavioral features
        features['is_weekend'] = datetime.utcnow().weekday() >= 5
        features['hour_of_day'] = datetime.utcnow().hour
        
        return features
    
    async def _get_event_count(self, entity_key: str, event_type: str, minutes: int) -> int:
        """
        Get count of events for entity in last N minutes
        """
        # This would query a time-series database or use Redis
        # For now, return placeholder
        return 1
    
    async def _get_rolling_sum(self, entity_key: str, value: float, count: int) -> float:
        """
        Get rolling sum of values for entity
        """
        # This would maintain sliding window sums
        # For now, return placeholder
        return value
```

## 3. Deployment Architecture

### 3.1 Kubernetes Configuration
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: feature-store-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: feature-store-api
  template:
    metadata:
      labels:
        app: feature-store-api
    spec:
      containers:
      - name: feature-store-api
        image: feature-store-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_URL
          value: "redis://redis-cluster:6379"
        - name: OFFLINE_STORAGE_PATH
          value: "/data/features"
        - name: KAFKA_BOOTSTRAP_SERVERS
          value: "kafka-cluster:9092"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: feature-store-service
spec:
  selector:
    app: feature-store-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: feature-store-config
data:
  feature-store-config.yaml: |
    offline_storage:
      path: "/data/features"
      format: "parquet"
      retention_days: 90
    online_storage:
      redis_url: "redis://redis-cluster:6379"
      ttl_seconds: 86400
    kafka:
      bootstrap_servers: "kafka-cluster:9092"
      consumer_group: "feature-store-consumer"
    monitoring:
      enabled: true
      metrics_port: 9090
```

### 3.2 Data Storage Configuration
```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: feature-store-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 1Ti  # Adjust based on data volume needs

---
apiVersion: v1
kind: Service
metadata:
  name: minio-service
spec:
  selector:
    app: minio
  ports:
    - protocol: TCP
      port: 9000
      targetPort: 9000
  type: ClusterIP

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: minio-feature-store
spec:
  replicas: 1
  selector:
    matchLabels:
      app: minio
  template:
    metadata:
      labels:
        app: minio
    spec:
      volumes:
      - name: storage
        persistentVolumeClaim:
          claimName: feature-store-pvc
      containers:
      - name: minio
        image: minio/minio:latest
        args:
        - server
        - /data
        - --console-address
        - ":9001"
        env:
        - name: MINIO_ROOT_USER
          value: "admin"
        - name: MINIO_ROOT_PASSWORD
          value: "password123"
        ports:
        - containerPort: 9000
        - containerPort: 9001
        volumeMounts:
        - name: storage
          mountPath: "/data"
```

## 4. Security Considerations

### 4.1 Data Encryption and Access Control
```python
from cryptography.fernet import Fernet
import jwt

class FeatureStoreSecurity:
    """
    Security layer for feature store
    """
    def __init__(self, encryption_key: str, jwt_secret: str):
        self.cipher_suite = Fernet(encryption_key.encode())
        self.jwt_secret = jwt_secret
    
    def encrypt_feature_value(self, value: Any) -> bytes:
        """
        Encrypt sensitive feature values
        """
        value_str = json.dumps(value)
        encrypted_value = self.cipher_suite.encrypt(value_str.encode())
        return encrypted_value
    
    def decrypt_feature_value(self, encrypted_value: bytes) -> Any:
        """
        Decrypt sensitive feature values
        """
        decrypted_str = self.cipher_suite.decrypt(encrypted_value).decode()
        return json.loads(decrypted_str)
    
    def generate_access_token(self, user_id: str, features: List[str], 
                            expiration_hours: int = 24) -> str:
        """
        Generate JWT token for feature access
        """
        payload = {
            'user_id': user_id,
            'features': features,
            'exp': datetime.utcnow() + timedelta(hours=expiration_hours)
        }
        
        token = jwt.encode(payload, self.jwt_secret, algorithm='HS256')
        return token
    
    def verify_access_token(self, token: str, required_feature: str) -> bool:
        """
        Verify JWT token and check feature access
        """
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=['HS256'])
            return required_feature in payload.get('features', [])
        except jwt.ExpiredSignatureError:
            return False
        except jwt.InvalidTokenError:
            return False

class DataMaskingService:
    """
    Service for masking sensitive data in features
    """
    def __init__(self):
        self.masking_rules = {}
    
    def add_masking_rule(self, feature_name: str, mask_type: str, 
                        params: Dict[str, Any] = None):
        """
        Add a data masking rule for a feature
        """
        self.masking_rules[feature_name] = {
            'type': mask_type,
            'params': params or {}
        }
    
    def apply_masking(self, feature_name: str, value: Any) -> Any:
        """
        Apply masking to feature value
        """
        if feature_name not in self.masking_rules:
            return value
        
        rule = self.masking_rules[feature_name]
        mask_type = rule['type']
        
        if mask_type == 'hash':
            return hashlib.sha256(str(value).encode()).hexdigest()
        elif mask_type == 'truncate':
            truncate_len = rule['params'].get('length', 4)
            str_val = str(value)
            return str_val[:truncate_len] + '*' * max(0, len(str_val) - truncate_len)
        elif mask_type == 'nullify':
            return None
        else:
            return value
```

## 5. Performance Optimization

### 5.1 Caching and Indexing Strategies
```python
import redis.asyncio as redis
from typing import Tuple

class FeatureCache:
    """
    Multi-level caching for features
    """
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = None
        self.redis_url = redis_url
        self.local_cache = {}  # In-memory cache
        self.cache_ttl = 300  # 5 minutes
    
    async def initialize(self):
        """
        Initialize Redis connection
        """
        self.redis = redis.from_url(self.redis_url)
    
    async def get(self, entity_key: str, feature_name: str) -> Optional[Any]:
        """
        Get feature from cache hierarchy
        """
        # Check local cache first
        local_key = f"{entity_key}:{feature_name}"
        if local_key in self.local_cache:
            value, timestamp = self.local_cache[local_key]
            if time.time() - timestamp < 60:  # 1 minute local cache
                return value
        
        # Check Redis cache
        redis_key = f"feature_cache:{entity_key}:{feature_name}"
        cached_value = await self.redis.get(redis_key)
        if cached_value:
            cached_value = json.loads(cached_value)
            # Update local cache
            self.local_cache[local_key] = (cached_value, time.time())
            return cached_value
        
        return None
    
    async def set(self, entity_key: str, feature_name: str, value: Any):
        """
        Set feature in cache hierarchy
        """
        local_key = f"{entity_key}:{feature_name}"
        self.local_cache[local_key] = (value, time.time())
        
        redis_key = f"feature_cache:{entity_key}:{feature_name}"
        await self.redis.setex(
            redis_key, 
            self.cache_ttl, 
            json.dumps(value)
        )

class FeatureIndex:
    """
    Index for efficient feature lookup
    """
    def __init__(self):
        self.entity_index = {}  # entity_key -> [feature_names]
        self.feature_index = {}  # feature_name -> [entity_keys]
        self.tag_index = {}  # tag -> [feature_names]
    
    def add_feature(self, entity_key: str, feature_name: str, tags: List[str] = None):
        """
        Add feature to indexes
        """
        # Update entity index
        if entity_key not in self.entity_index:
            self.entity_index[entity_key] = []
        if feature_name not in self.entity_index[entity_key]:
            self.entity_index[entity_key].append(feature_name)
        
        # Update feature index
        if feature_name not in self.feature_index:
            self.feature_index[feature_name] = []
        if entity_key not in self.feature_index[feature_name]:
            self.feature_index[feature_name].append(entity_key)
        
        # Update tag index
        if tags:
            for tag in tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = []
                if feature_name not in self.tag_index[tag]:
                    self.tag_index[tag].append(feature_name)
    
    def get_entity_features(self, entity_key: str) -> List[str]:
        """
        Get all features for an entity
        """
        return self.entity_index.get(entity_key, [])
    
    def get_feature_entities(self, feature_name: str) -> List[str]:
        """
        Get all entities that have a feature
        """
        return self.feature_index.get(feature_name, [])
    
    def get_features_by_tag(self, tag: str) -> List[str]:
        """
        Get all features with a tag
        """
        return self.tag_index.get(tag, [])
```

## 6. Testing and Validation

### 6.1 Comprehensive Testing Suite
```python
import unittest
import asyncio
from unittest.mock import Mock, AsyncMock, patch

class TestFeatureStore(unittest.TestCase):
    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
    
    def test_feature_registration(self):
        """Test feature registration functionality"""
        async def run_test():
            store = FeatureStore("/tmp/test_features")
            store.redis = Mock()
            store.redis.set = AsyncMock(return_value=True)
            
            spec = FeatureSpec(
                name="user_age",
                feature_type=FeatureType.NUMERIC,
                storage_type=StorageType.ONLINE,
                default_value=0,
                description="Age of the user"
            )
            
            result = await store.register_feature(spec)
            self.assertTrue(result)
            
            # Verify Redis was called correctly
            store.redis.set.assert_called_once()
        
        self.loop.run_until_complete(run_test())
    
    def test_feature_storage_retrieval(self):
        """Test storing and retrieving features"""
        async def run_test():
            store = FeatureStore("/tmp/test_features")
            store.redis = Mock()
            store.redis.setex = AsyncMock(return_value=True)
            store.redis.get = AsyncMock(return_value=json.dumps({
                'value': 25,
                'timestamp': '2023-01-01T00:00:00Z',
                'version': '1.0'
            }))
            
            # Test storing feature
            result = await store.put_features("user_123", {"user_age": 25})
            self.assertTrue(result)
            
            # Test retrieving feature
            features = await store.get_features("user_123", ["user_age"])
            self.assertEqual(features["user_age"], 25)
        
        self.loop.run_until_complete(run_test())
    
    def test_data_quality_monitoring(self):
        """Test data quality monitoring"""
        async def run_test():
            store = FeatureStore("/tmp/test_features")
            monitor = DataQualityMonitor(store)
            
            # Add quality rule
            monitor.add_quality_rule("user_age", "min_value", 0, "Age must be non-negative")
            monitor.add_quality_rule("user_age", "max_value", 150, "Age must be realistic")
            
            # Test valid data
            violations = await monitor.check_quality("user_123", {"user_age": 25})
            self.assertEqual(len(violations), 0)
            
            # Test invalid data
            violations = await monitor.check_quality("user_123", {"user_age": -5})
            self.assertEqual(len(violations), 1)
            self.assertEqual(violations[0]["rule_type"], "min_value")
        
        self.loop.run_until_complete(run_test())
    
    def test_feature_transformation(self):
        """Test feature transformation engine"""
        engine = FeatureTransformationEngine()
        
        # Test normalization
        data = pd.Series([1, 2, 3, 4, 5])
        normalized = engine.apply_transformation(data, "normalize")
        self.assertAlmostEqual(normalized.min(), 0.0)
        self.assertAlmostEqual(normalized.max(), 1.0)
        
        # Test standardization
        standardized = engine.apply_transformation(data, "standardize")
        self.assertAlmostEqual(standardized.mean(), 0.0, places=5)
        self.assertAlmostEqual(standardized.std(), 1.0, places=5)

if __name__ == '__main__':
    unittest.main()
```

## 7. Implementation Plan

### Phase 1: Core Infrastructure (Weeks 1-3)
- Set up basic feature store with offline and online storage
- Implement feature registration and basic CRUD operations
- Create data ingestion pipeline
- Basic data quality monitoring

### Phase 2: Advanced Features (Weeks 4-6)
- Implement feature discovery and search
- Add lineage tracking and metadata management
- Create transformation engine
- Implement access control

### Phase 3: Scalability and Performance (Weeks 7-9)
- Optimize for large-scale data processing
- Implement caching and indexing strategies
- Add monitoring and alerting
- Performance testing and optimization

### Phase 4: Production Readiness (Weeks 10-12)
- Security hardening
- Disaster recovery and backup
- Documentation and client libraries
- Integration testing

## 8. Success Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| Feature Storage Latency (p95) | < 50ms | Redis metrics |
| Batch Processing Throughput | 1M records/hour | Pipeline monitoring |
| Data Freshness | < 5 minutes | Timestamp comparison |
| Feature Availability | 99.9% | Health checks |
| Storage Efficiency | < 10% overhead | Size comparison |
| Data Quality Score | > 95% | Validation metrics |

This comprehensive feature store and data pipeline architecture provides a robust foundation for managing machine learning features at scale with high performance, reliability, and data quality.