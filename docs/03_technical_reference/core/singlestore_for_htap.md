# SingleStore for Hybrid Transactional/Analytical Processing (HTAP)

## Overview
SingleStore (formerly MemSQL) is a distributed SQL database designed for hybrid transactional/analytical processing (HTAP). It combines the speed of in-memory processing with the durability of disk-based storage, enabling real-time analytics on operational data without ETL pipelines.

## Core Architecture Principles

### Dual-Engine Architecture
- **Rowstore**: Optimized for OLTP workloads with ACID transactions
- **Columnstore**: Optimized for OLAP workloads with columnar compression
- **Automatic data movement**: Hot data in rowstore, cold data in columnstore
- **Unified query engine**: Single SQL interface for both workloads

### Distributed Query Processing
- Shared-nothing architecture with coordinator and leaf nodes
- Parallel query execution across multiple nodes
- Adaptive query optimization based on data distribution
- In-memory processing with disk spill for large datasets

### Real-time Data Ingestion
- High-throughput ingestion capabilities (1M+ rows/sec per node)
- Support for Kafka, Kinesis, and other streaming sources
- Change data capture (CDC) for real-time data synchronization
- Built-in data transformation capabilities

## Performance Characteristics

| Workload | SingleStore | PostgreSQL | ClickHouse |
|----------|-------------|------------|------------|
| OLTP throughput | 150K ops/sec | 50K ops/sec | N/A |
| OLAP query latency | 100ms-1s | 5-30s | 50-500ms |
| Mixed workload | Excellent | Poor | Good (OLAP only) |
| Data freshness | Real-time | Batch delay | Near real-time |
| Memory efficiency | High (hybrid) | Medium | Low (columnar) |

*Test environment: 4-node cluster (16 vCPUs, 64GB RAM each), 100M rows of synthetic e-commerce data*

## AI/ML Specific Use Cases

### Real-time ML Feature Engineering
- Generate features directly from operational data
- Combine transactional and analytical queries in single workflows
- Support for window functions and time-series analysis

```sql
-- Real-time feature engineering example
SELECT 
    user_id,
    -- Transactional features
    COUNT(*) as total_purchases,
    SUM(amount) as total_spent,
    -- Analytical features
    AVG(amount) OVER (PARTITION BY user_id ORDER BY purchase_time ROWS BETWEEN 30 PRECEDING AND CURRENT ROW) as rolling_avg_30d,
    -- ML-specific features
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY amount) as p95_spend,
    CASE 
        WHEN COUNT(*) > 10 AND SUM(amount) > 1000 THEN 'high_value'
        ELSE 'standard'
    END as customer_segment
FROM purchases 
WHERE purchase_time > NOW() - INTERVAL 7 DAY
GROUP BY user_id;
```

### Online Model Training
- Train models on fresh operational data without ETL
- Support for incremental model updates
- Real-time feedback loops for model retraining

### Operational Analytics for ML Systems
- Monitor ML system performance in real-time
- Track model drift and concept drift metrics
- Correlate business metrics with ML performance

## Implementation Patterns

### Schema Design Best Practices
- **Hybrid tables**: Use rowstore for frequently updated data, columnstore for historical data
- **Partitioning**: Time-based partitioning for time-series data
- **Indexing**: Composite indexes for multi-dimensional queries
- **Materialized views**: Pre-compute complex aggregations

### Query Optimization Techniques
- **Query hints**: Force execution plans for critical queries
- **Resource pools**: Isolate ML workloads from operational workloads
- **Adaptive execution**: Automatic switching between rowstore/columnstore
- **Vectorized execution**: SIMD optimizations for analytical queries

### Integration with ML Frameworks
- **Direct SQL-to-model**: Export query results to pandas/scikit-learn
- **Feature store integration**: Serve as backend for real-time feature stores
- **Model serving**: Store model parameters and metadata
- **Experiment tracking**: Track ML experiments with ACID guarantees

```python
# Python integration example
import singlestoredb as s2

# Connect to SingleStore
conn = s2.connect('mysql://user:pass@localhost:3306/ml_db')

# Execute real-time feature query
query = """
    SELECT user_id, feature1, feature2, feature3, target
    FROM (
        SELECT 
            user_id,
            AVG(transaction_amount) as feature1,
            COUNT(CASE WHEN category = 'electronics' THEN 1 END) * 1.0 / COUNT(*) as feature2,
            STDDEV(transaction_amount) as feature3,
            LAG(target, 1) OVER (PARTITION BY user_id ORDER BY timestamp) as target
        FROM transactions
        WHERE timestamp > NOW() - INTERVAL 1 HOUR
        GROUP BY user_id, DATE(timestamp)
    )
    WHERE target IS NOT NULL
"""

# Load into pandas for ML training
df = pd.read_sql(query, conn)
X = df.drop('target', axis=1)
y = df['target']
```

## Trade-offs and Limitations

### Strengths
- **Real-time analytics**: Eliminate ETL latency for ML workflows
- **Unified platform**: Single system for OLTP and OLAP
- **Scalability**: Linear scaling for both transactional and analytical workloads
- **Familiar SQL**: Standard SQL interface with PostgreSQL compatibility

### Limitations
- **Cost**: Higher licensing costs than open-source alternatives
- **Complexity**: Requires understanding of hybrid architecture
- **Ecosystem**: Smaller community than PostgreSQL or MySQL
- **Feature parity**: Some advanced analytical functions still developing

## Production Examples

### Shopify's Real-time Analytics
- Powers real-time merchant analytics dashboard
- Processes 10M+ transactions per hour
- Enables real-time fraud detection with live data

### Capital One's Risk Management
- Real-time credit risk assessment
- Combines transactional data with analytical models
- Reduced decision latency from hours to seconds

### Uber's Dynamic Pricing
- Real-time demand forecasting and pricing
- Combines ride requests with traffic and weather data
- Supports A/B testing with immediate feedback

## AI/ML Specific Optimizations

### Vector Operations
- Native support for vector similarity search
- Integration with ML models for hybrid recommendation systems
- Example: `COSINE_SIMILARITY(vector1, vector2)` function

### Time-Series Analysis
- Specialized time-series functions for temporal feature extraction
- Efficient window functions for rolling statistics
- Built-in functions for seasonal decomposition

### Feature Store Patterns
- Multi-version feature storage with real-time updates
- Support for feature lineage and provenance tracking
- Integration with MLflow and other MLOps tools

## Getting Started Guide

### Installation Options
- Docker: `docker run -p 3306:3306 -p 8080:8080 singlestore/singlestore-db`
- Kubernetes: Official Helm chart available
- Cloud: SingleStore Cloud (managed service)
- Bare metal: RPM/DEB packages for major Linux distributions

### Basic Setup
```sql
-- Create a hybrid table for ML workloads
CREATE TABLE user_transactions (
    user_id BIGINT,
    transaction_id BIGINT,
    amount DECIMAL(10,2),
    category VARCHAR(50),
    timestamp DATETIME,
    metadata JSON,
    PRIMARY KEY (user_id, transaction_id)
) ENGINE=InnoDB;

-- Create columnstore table for historical analysis
CREATE TABLE user_transactions_history (
    user_id BIGINT,
    month DATE,
    total_amount DECIMAL(12,2),
    transaction_count BIGINT,
    avg_amount DECIMAL(10,2),
    categories JSON,
    INDEX (user_id, month)
) ENGINE=Columnstore;

-- Set up automatic data movement
CREATE PIPELINE transactions_pipeline AS
LOAD DATA FROM S3 's3://bucket/transactions/'
INTO TABLE user_transactions
SETTINGS
    batch_size = 10000,
    max_errors = 100;
```

## Related Resources
- [SingleStore Documentation](https://docs.singlestore.com/)
- [HTAP for ML Engineers](https://www.singlestore.com/blog/htap-machine-learning/)
- [Case Study: Real-time ML Workflows with SingleStore](../06_case_studies/singlestore_htap_ml.md)
- [System Design: Unified Database for AI Applications](../03_system_design/solutions/database_architecture_patterns_ai.md)