# MongoDB Tutorial for Machine Learning Applications

This tutorial provides hands-on MongoDB fundamentals specifically designed for ML engineers who need to work with flexible, schema-less data for model metadata, experiment tracking, and feature storage.

## Why MongoDB for ML Workflows?

MongoDB excels in ML applications because:
- **Flexible schema**: Perfect for evolving ML data structures
- **Rich document model**: Natural fit for nested ML metadata
- **High write throughput**: Handles high-volume training data logging
- **Aggregation framework**: Powerful for ML analytics and feature engineering
- **Horizontal scalability**: Sharding for large-scale ML datasets
- **Geospatial and text search**: Built-in support for specialized queries

## Setting Up MongoDB for ML Development

### Installation Options
```bash
# Docker (recommended for development)
docker run -d \
  --name mongodb-ml \
  -e MONGO_INITDB_ROOT_USERNAME=ml_user \
  -e MONGO_INITDB_ROOT_PASSWORD=ml_password \
  -p 27017:27017 \
  mongo:6.0

# With WiredTiger storage engine (default) and journaling enabled
docker run -d \
  --name mongodb-ml-prod \
  -e MONGO_INITDB_ROOT_USERNAME=ml_user \
  -e MONGO_INITDB_ROOT_PASSWORD=ml_password \
  -v /data/mongodb:/data/db \
  -p 27017:27017 \
  mongo:6.0 --storageEngine wiredTiger --journalCommitIntervalMs 100
```

### Essential Configuration for ML Workloads
```javascript
// mongod.conf for ML workloads
storage:
  engine: "wiredTiger"
  wiredTiger:
    engineConfig:
      cacheSizeGB: 8  # Adjust based on available RAM
      journalCompressor: zlib
  journal:
    commitIntervalMs: 100  # Lower for higher durability

operationProfiling:
  mode: "slowOp"
  slowOpThresholdMs: 100

replication:
  oplogSizeMB: 10240  # Larger oplog for high-write ML workloads
```

## Core MongoDB Concepts for ML Engineers

### Document Structure for ML Data

#### Model Registry Schema
```javascript
// models collection
{
  "_id": ObjectId("65a1b2c3d4e5f67890abcdef"),
  "name": "ResNet-50-v2",
  "description": "Improved ResNet-50 with better regularization",
  "created_at": ISODate("2026-02-15T10:30:00Z"),
  "updated_at": ISODate("2026-02-15T11:45:00Z"),
  "status": "production",
  "owner": "data_science_team",
  
  "training_config": {
    "epochs": 100,
    "batch_size": 32,
    "optimizer": "AdamW",
    "learning_rate": 0.001,
    "scheduler": "CosineAnnealing",
    "loss_function": "CrossEntropy"
  },
  
  "metrics": {
    "accuracy": 0.942,
    "precision": 0.938,
    "recall": 0.941,
    "f1_score": 0.939,
    "training_time_seconds": 18432,
    "inference_latency_ms": 45.2
  },
  
  "artifacts": [
    {
      "name": "model_weights.h5",
      "size_bytes": 24576000,
      "storage_url": "s3://models/resnet50-v2/weights.h5",
      "checksum": "sha256:abc123def456...",
      "uploaded_at": ISODate("2026-02-15T11:30:00Z")
    }
  ],
  
  "tags": ["computer_vision", "image_classification", "production"],
  "version": "1.2.0"
}
```

#### Experiment Tracking Schema
```javascript
// experiments collection
{
  "_id": ObjectId("65a1b2c3d4e5f67890abcdee"),
  "model_id": ObjectId("65a1b2c3d4e5f67890abcdef"),
  "experiment_name": "hyperparameter_tuning_v3",
  "created_at": ISODate("2026-02-15T09:15:00Z"),
  "started_at": ISODate("2026-02-15T09:20:00Z"),
  "completed_at": ISODate("2026-02-15T10:45:00Z"),
  "status": "completed",
  
  "parameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "dropout_rate": 0.5,
    "weight_decay": 0.01,
    "augmentation_strength": 0.8
  },
  
  "results": {
    "train_loss": 0.058,
    "val_loss": 0.062,
    "train_accuracy": 0.965,
    "val_accuracy": 0.942,
    "best_epoch": 87,
    "convergence_time_minutes": 85
  },
  
  "metrics_history": [
    { "epoch": 1, "train_loss": 1.25, "val_loss": 1.32, "accuracy": 0.65 },
    { "epoch": 10, "train_loss": 0.45, "val_loss": 0.52, "accuracy": 0.82 },
    // ... more epochs
  ],
  
  "hardware": {
    "gpu_type": "NVIDIA A100",
    "gpu_count": 4,
    "cpu_cores": 32,
    "memory_gb": 256
  }
}
```

## MongoDB Query Patterns for ML Workflows

### Basic CRUD Operations
```javascript
// Insert a new model
db.models.insertOne({
  name: "Transformer-Large",
  description: "Large transformer model for NLP tasks",
  status: "draft",
  created_at: new Date(),
  training_config: { epochs: 50, batch_size: 16 },
  metrics: { accuracy: 0.89 }
});

// Find models by status and minimum accuracy
db.models.find({
  status: "production",
  "metrics.accuracy": { $gt: 0.9 }
}).sort({ "metrics.accuracy": -1 }).limit(10);

// Update model status and add metadata
db.models.updateOne(
  { _id: ObjectId("65a1b2c3d4e5f67890abcdef") },
  {
    $set: {
      status: "staging",
      "updated_at": new Date(),
      "deployment_info.endpoint": "https://api.ml-platform.com/models/resnet-50-v2",
      "deployment_info.deployed_at": new Date()
    }
  }
);
```

### Aggregation Framework for ML Analytics

#### Training Progress Analysis
```javascript
// Calculate moving averages for training metrics
db.experiments.aggregate([
  { $match: { "model_id": ObjectId("65a1b2c3d4e5f67890abcdef") } },
  { $unwind: "$metrics_history" },
  { $sort: { "metrics_history.epoch": 1 } },
  { $group: {
    _id: "$_id",
    epochs: { $push: "$metrics_history.epoch" },
    train_losses: { $push: "$metrics_history.train_loss" },
    val_losses: { $push: "$metrics_history.val_loss" },
    accuracies: { $push: "$metrics_history.accuracy" }
  } },
  { $project: {
    _id: 1,
    epoch_stats: {
      $map: {
        input: { $range: [0, { $size: "$epochs" }] },
        as: "i",
        in: {
          epoch: { $arrayElemAt: ["$epochs", "$$i"] },
          train_loss: { $arrayElemAt: ["$train_losses", "$$i"] },
          val_loss: { $arrayElemAt: ["$val_losses", "$$i"] },
          accuracy: { $arrayElemAt: ["$accuracies", "$$i"] },
          // Calculate 5-epoch moving average
          train_loss_ma: {
            $avg: {
              $slice: [
                "$train_losses",
                { $subtract: ["$$i", 2] },
                5
              ]
            }
          }
        }
      }
    }
  } }
]);
```

#### Model Performance Comparison
```javascript
// Compare models by category and performance
db.models.aggregate([
  { $match: { status: "production" } },
  { $addFields: {
    category: {
      $switch: {
        branches: [
          { case: { $in: ["$tags", ["computer_vision"]] }, then: "computer_vision" },
          { case: { $in: ["$tags", ["nlp"]] }, then: "nlp" },
          { case: { $in: ["$tags", ["recommendation"]] }, then: "recommendation" }
        ],
        default: "other"
      }
    }
  } },
  { $group: {
    _id: "$category",
    count: { $sum: 1 },
    avg_accuracy: { $avg: "$metrics.accuracy" },
    max_accuracy: { $max: "$metrics.accuracy" },
    min_accuracy: { $min: "$metrics.accuracy" },
    models: { $push: { name: "$name", accuracy: "$metrics.accuracy", _id: "$_id" } }
  } },
  { $sort: { avg_accuracy: -1 } }
]);
```

### Text Search for ML Metadata
```javascript
// Create text index on model descriptions and names
db.models.createIndex(
  { "name": "text", "description": "text", "tags": "text" },
  { name: "model_search_index" }
);

// Full-text search across ML metadata
db.models.find(
  { $text: { $search: "image classification resnet" } },
  { score: { $meta: "textScore" } }
).sort({ score: { $meta: "textScore" } }).limit(5);

// Fuzzy matching with regex (for partial matches)
db.models.find({
  $or: [
    { name: { $regex: /resnet/i } },
    { "tags": { $in: ["computer_vision"] } },
    { "training_config.optimizer": { $regex: /adam/i } }
  ]
});
```

## Indexing Strategies for ML Performance

### Essential Indexes for ML Workloads

#### Compound Indexes
```javascript
// Common query patterns for ML systems
db.models.createIndex({ "status": 1, "metrics.accuracy": -1 });
db.models.createIndex({ "created_at": -1, "status": 1 });
db.models.createIndex({ "training_config.batch_size": 1, "metrics.accuracy": -1 });

// For time-series metrics
db.experiments.createIndex({ "completed_at": -1, "model_id": 1 });
db.experiments.createIndex({ "parameters.learning_rate": 1, "results.val_accuracy": -1 });
```

#### Multikey Indexes for Arrays
```javascript
// Index array elements (tags, artifacts)
db.models.createIndex({ "tags": 1 });
db.models.createIndex({ "artifacts.name": 1 });
db.models.createIndex({ "artifacts.uploaded_at": -1 });

// Query using indexed array fields
db.models.find({ "tags": "computer_vision" });
db.models.find({ "artifacts.name": "weights.h5" });
```

#### TTL Indexes for Auto-expiration
```javascript
// Auto-delete old experiment data
db.experiments.createIndex({ "completed_at": 1 }, { expireAfterSeconds: 2592000 }); // 30 days

// Auto-delete temporary model versions
db.models.createIndex({ "created_at": 1 }, { 
  expireAfterSeconds: 604800, // 7 days
  partialFilterExpression: { status: "draft" }
});
```

## MongoDB for Feature Stores

### Designing a Feature Store Schema
```javascript
// entities collection (users, products, sessions)
{
  "_id": ObjectId("65a1b2c3d4e5f67890abcdf0"),
  "entity_type": "user",
  "entity_id": "user_12345",
  "created_at": ISODate("2026-02-10T08:15:00Z"),
  "metadata": {
    "signup_date": ISODate("2025-01-15"),
    "region": "us-west",
    "tier": "premium"
  }
}

// features collection (feature definitions)
{
  "_id": ObjectId("65a1b2c3d4e5f67890abcdf1"),
  "feature_name": "user_engagement_score",
  "description": "Composite engagement score based on activity",
  "data_type": "float",
  "created_at": ISODate("2026-02-01T10:00:00Z"),
  "tags": ["user", "engagement", "realtime"]
}

// feature_values collection (time-series feature values)
{
  "_id": ObjectId("65a1b2c3d4e5f67890abcdf2"),
  "entity_id": ObjectId("65a1b2c3d4e5f67890abcdf0"), // reference to entity
  "feature_id": ObjectId("65a1b2c3d4e5f67890abcdf1"), // reference to feature
  "timestamp": ISODate("2026-02-15T10:30:00Z"),
  "value": 0.87,
  "source": "realtime_processor",
  "version": "v1.2"
}
```

### Efficient Feature Retrieval Patterns
```javascript
// Get latest feature values for a user
db.feature_values.aggregate([
  { $match: { 
      "entity_id": ObjectId("65a1b2c3d4e5f67890abcdf0"),
      "timestamp": { $lte: new Date() }
    } },
  { $sort: { "timestamp": -1 } },
  { $group: {
    _id: "$feature_id",
    latest: { $first: "$$ROOT" }
  } },
  { $replaceRoot: { newRoot: "$latest" } },
  { $lookup: {
    from: "features",
    localField: "feature_id",
    foreignField: "_id",
    as: "feature_info"
  } },
  { $unwind: "$feature_info" },
  { $project: {
    _id: 0,
    feature_name: "$feature_info.feature_name",
    value: 1,
    timestamp: 1,
    source: 1
  } }
]);

// Get feature values over time range
db.feature_values.find({
  "entity_id": ObjectId("65a1b2c3d4e5f67890abcdf0"),
  "feature_id": { $in: [
    ObjectId("65a1b2c3d4e5f67890abcdf1"), // engagement score
    ObjectId("65a1b2c3d4e5f67890abcdf3")  // session duration
  ] },
  "timestamp": { $gte: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000) } // last 7 days
}).sort({ "timestamp": 1 });
```

## Performance Optimization for ML Workloads

### Write Optimization
```javascript
// Use bulk operations for high-volume data ingestion
const bulkOps = db.experiments.initializeUnorderedBulkOp();
for (let i = 0; i < 1000; i++) {
  bulkOps.insert({
    model_id: ObjectId("65a1b2c3d4e5f67890abcdef"),
    experiment_name: `batch_${i}`,
    parameters: { learning_rate: 0.001 + Math.random() * 0.002 },
    results: { accuracy: 0.85 + Math.random() * 0.1 },
    created_at: new Date()
  });
}
bulkOps.execute();

// Use write concern for durability
db.models.insertOne(doc, { writeConcern: { w: "majority", j: true } });
```

### Read Optimization
```javascript
// Use projection to limit returned data
db.models.find(
  { status: "production" },
  { name: 1, "metrics.accuracy": 1, "created_at": 1, _id: 0 }
);

// Use hint to force index usage
db.models.find({ "tags": "computer_vision" })
  .hint({ "tags": 1 })
  .limit(10);

// Use cursor batchSize for large result sets
db.models.find({ status: "production" }).batchSize(100);
```

### Memory and Storage Optimization
```javascript
// Use compression (WiredTiger default is snappy)
// Configure in mongod.conf:
storage:
  wiredTiger:
    collectionConfig:
      blockCompressor: zlib
    indexConfig:
      prefixCompression: true

// Use capped collections for time-series data
db.createCollection("training_logs", {
  capped: true,
  size: 1000000000, // 1GB
  max: 100000 // max documents
});
```

## Common MongoDB Pitfalls for ML Engineers

### 1. Document Size Limits
- **16MB document limit**: Plan for large embeddings or metadata
- **Solution**: Use references or split large documents
- **Example**: Store large feature vectors in separate collection

### 2. Array Growth Issues
- **Array field growth**: Can cause document moves and fragmentation
- **Solution**: Pre-allocate array sizes or use separate collections
- **Example**: Store metrics_history in separate collection instead of embedded array

### 3. Index Overhead
- **Too many indexes**: Slows down writes significantly
- **Solution**: Monitor index usage with `db.collection.getIndexes()`
- **Best practice**: Start with essential indexes, add others as needed

### 4. Sharding Complexity
- **Shard key selection**: Critical for performance
- **Common mistakes**: Using monotonically increasing keys (like timestamps)
- **Better choices**: Hashed shard keys or compound keys with high cardinality

## Visual Diagrams

### MongoDB Architecture for ML Systems
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │───▶│  MongoDB Driver │───▶│  MongoDB       │
│ (ML Training,   │    │ (Node.js, Python)│    │  Replica Set    │
│  API, Dashboard)│    └─────────────────┘    │  (Primary +     │
└─────────────────┘                           │   Secondaries)  │
                                              └────────┬────────┘
                                                       │
                                              ┌─────────────────┐
                                              │  Sharded Cluster│
                                              │  (for large data)│
                                              └─────────────────┘
```

### Feature Store Data Flow
```
Training Pipeline → [Feature Engineering] → MongoDB Feature Store
       ↑                    │                         │
[Data Sources] ←─────────────┘                         ▼
                                                      ┌─────────────────┐
                                                      │  Model Training │
                                                      │  (reads features)│
                                                      └─────────────────┘
                                                              │
                                                              ▼
                                                  ┌─────────────────────┐
                                                  │  Real-time Inference│
                                                  │  (online features)  │
                                                  └─────────────────────┘
```

## Hands-on Exercises

### Exercise 1: Build a Model Registry
1. Create collections for models, versions, and deployments
2. Insert sample model data with nested structures
3. Write aggregation queries for model performance analysis
4. Create appropriate indexes for common query patterns

### Exercise 2: Experiment Tracking System
1. Design schema for hyperparameter tuning experiments
2. Implement time-series metrics storage
3. Write queries for comparing experiment results
4. Set up TTL indexes for automatic cleanup

### Exercise 3: Feature Store Implementation
1. Create schema for entities, features, and feature values
2. Implement efficient queries for latest feature values
3. Add indexing for time-based queries
4. Test performance with simulated ML workload

## Best Practices Summary

1. **Use embedded documents** for related data that's frequently accessed together
2. **Normalize when documents grow too large** (> 1MB) or arrays grow excessively
3. **Start with minimal indexes** and add based on query patterns
4. **Use aggregation framework** for complex ML analytics instead of application-side processing
5. **Monitor performance** with `db.currentOp()` and `db.profile`
6. **Plan for sharding** early if you expect large data volumes
7. **Use transactions** for critical ML metadata operations (MongoDB 4.0+)
8. **Version your schemas** and implement migration strategies

This tutorial provides the foundation for effectively using MongoDB in machine learning applications, from model registry to feature stores and experiment tracking.