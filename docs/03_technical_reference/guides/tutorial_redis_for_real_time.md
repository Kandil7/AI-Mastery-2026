# Redis Tutorial for Real-Time AI Systems

This tutorial provides hands-on Redis fundamentals specifically designed for AI/ML engineers building real-time inference systems, feature serving, and low-latency ML applications.

## Why Redis for Real-Time AI?

Redis excels in real-time AI systems because:
- **Sub-millisecond latency**: Critical for real-time inference
- **In-memory storage**: Ultra-fast access to hot data
- **Rich data structures**: Hashes, lists, sets, sorted sets for different patterns
- **Pub/Sub and streams**: Real-time event processing
- **Lua scripting**: Atomic operations for complex logic
- **High availability**: Sentinel and Cluster modes

## Setting Up Redis for AI Workloads

### Installation Options
```bash
# Docker (recommended for development)
docker run -d \
  --name redis-ai \
  -p 6379:6379 \
  -e REDIS_PASSWORD=ai_password \
  redis:7.2 --requirepass ai_password

# With persistence configuration for production
docker run -d \
  --name redis-ai-prod \
  -v /data/redis:/data \
  -p 6379:6379 \
  redis:7.2 \
  --requirepass ai_password \
  --save 60 1000 \          # Save if 1000 keys change in 60s
  --appendonly yes \        # Enable AOF persistence
  --appendfsync everysec \  # Balance durability and performance
  --maxmemory 8gb \         # Memory limit
  --maxmemory-policy allkeys-lru  # LRU eviction policy
```

### Essential Configuration for Real-Time AI
```conf
# redis.conf for AI workloads
bind 0.0.0.0
protected-mode yes
port 6379
tcp-backlog 511
timeout 0
tcp-keepalive 30

# Persistence
save 900 1      # Save if 1 key changes in 900s
save 300 10     # Save if 10 keys change in 300s
save 60 10000   # Save if 10000 keys change in 60s
appendonly yes
appendfsync everysec
no-appendfsync-on-rewrite yes

# Memory management
maxmemory 8gb
maxmemory-policy allkeys-lru

# Performance
hash-max-ziplist-entries 512
hash-max-ziplist-value 64
list-max-ziplist-size -2
set-max-intset-entries 512
zset-max-ziplist-entries 128
zset-max-ziplist-value 64

# Security
requirepass ai_password
```

## Core Redis Data Structures for AI Systems

### Hashes for Feature Serving
```bash
# Store user features as hash
HSET user:12345:features \
  age 28 \
  gender "female" \
  region "us-west" \
  engagement_score 0.87 \
  last_login "2026-02-15T10:30:00Z" \
  session_count 142

# Get specific features
HGET user:12345:features age
HGET user:12345:features engagement_score

# Get all features
HGETALL user:12345:features

# Update multiple features atomically
HMSET user:12345:features \
  engagement_score 0.89 \
  last_login "2026-02-15T10:45:00Z" \
  session_count 143
```

### Sorted Sets for Real-Time Ranking
```bash
# Store recommendation scores
ZADD recommendations:12345 0.92 item_123
ZADD recommendations:12345 0.87 item_456
ZADD recommendations:12345 0.85 item:789

# Get top 10 recommendations
ZREVRANGE recommendations:12345 0 9 WITHSCORES

# Update scores in real-time
ZINCRBY recommendations:12345 0.01 item_123  # Boost score

# Remove expired recommendations
ZREMRANGEBYSCORE recommendations:12345 -inf 0.7  # Remove low scores
```

### Lists for Real-Time Event Processing
```bash
# Store recent user events
LPUSH user:12345:events \
  '{"type":"click","item":"product_123","timestamp":"2026-02-15T10:30:00Z"}' \
  '{"type":"view","item":"category_456","timestamp":"2026-02-15T10:29:00Z"}'

# Get recent 10 events
LRANGE user:12345:events 0 9

# Trim to keep only last 100 events
LTRIM user:12345:events 0 99
```

### Sets for User Cohorts and Segmentation
```bash
# Store user cohorts
SADD cohort:premium_users user:12345 user:67890 user:24680
SADD cohort:active_users user:12345 user:56789 user:34567

# Intersection for targeted campaigns
SINTER cohort:premium_users cohort:active_users

# Union for broader targeting
SUNION cohort:premium_users cohort:free_users

# Check membership
SISMEMBER cohort:premium_users user:12345
```

## Real-Time AI Patterns

### Online Feature Serving
```bash
# Pattern: Real-time feature lookup for inference
# Key format: feature:{entity_type}:{entity_id}:{feature_name}

# Store features
HSET feature:user:12345:engagement_score 0.87
HSET feature:user:12345:session_duration 184
HSET feature:user:12345:conversion_rate 0.12

# Batch feature retrieval for model inference
# Use pipeline for multiple features
MULTI
HGET feature:user:12345:engagement_score
HGET feature:user:12345:session_duration  
HGET feature:user:12345:conversion_rate
EXEC

# Result: [0.87, 184, 0.12] for model input
```

### Real-Time Model Scoring Cache
```bash
# Cache model predictions to reduce compute load
# Key format: prediction:{model_id}:{input_hash}

# Store prediction with TTL
SET prediction:resnet-50-v2:abc123 '{"score":0.942,"class":"cat","timestamp":"2026-02-15T10:30:00Z"}' EX 300

# Get cached prediction
GET prediction:resnet-50-v2:abc123

# Check if exists and get
EXISTS prediction:resnet-50-v2:abc123
```

### Rate Limiting for API Endpoints
```bash
# Token bucket algorithm for rate limiting
# Key format: rate_limit:{user_id}:{endpoint}

# Initialize counter (only if doesn't exist)
SETNX rate_limit:user_12345:/predict 0
EXPIRE rate_limit:user_12345:/predict 60

# Atomic increment and check
EVAL "
  local count = redis.call('INCR', KEYS[1])
  if count == 1 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
  end
  if count > tonumber(ARGV[2]) then
    return 0
  end
  return count
" 1 rate_limit:user_12345:/predict 60 100

# Returns current count or 0 if rate limited
```

### Real-Time Anomaly Detection
```bash
# Track recent metrics for anomaly detection
# Key format: metric:{metric_name}:{entity_id}

# Store recent values
LPUSH metric:latency:user_12345 45.2 42.8 47.1 43.5 46.8
LTRIM metric:latency:user_12345 0 99  # Keep last 100 samples

# Calculate moving average and standard deviation
# Use Lua script for atomic calculation
EVAL "
  local values = redis.call('LRANGE', KEYS[1], 0, -1)
  local sum = 0
  local sum_sq = 0
  for i=1,#values do
    local val = tonumber(values[i])
    sum = sum + val
    sum_sq = sum_sq + (val * val)
  end
  local count = #values
  local mean = sum / count
  local variance = (sum_sq / count) - (mean * mean)
  local std_dev = math.sqrt(variance)
  return {mean, std_dev}
" 1 metric:latency:user_12345
```

## Advanced Redis Features for AI Systems

### Redis Streams for Event Processing
```bash
# Create stream for real-time events
XADD events * \
  user_id 12345 \
  event_type "prediction" \
  model_id "resnet-50-v2" \
  input_hash "abc123" \
  prediction_score 0.942 \
  timestamp "2026-02-15T10:30:00Z"

# Consumer groups for parallel processing
XGROUP CREATE events ml_consumers 0 MKSTREAM
XREADGROUP GROUP ml_consumers consumer_1 COUNT 10 BLOCK 1000 STREAMS events >

# Acknowledge processed messages
XACK events ml_consumers <message_id>
```

### Redis Modules for AI Workloads

#### RedisJSON for Nested Data
```bash
# Store complex model metadata
JSON.SET model:12345 $ '{"name":"ResNet-50-v2","metrics":{"accuracy":0.942},"artifacts":[{"name":"weights.h5"}]}'

# Query nested fields
JSON.GET model:12345 $.metrics.accuracy
JSON.GET model:12345 $.artifacts[0].name

# Update nested fields
JSON.SET model:12345 $.status '"production"'
```

#### RedisSearch for Hybrid Search
```bash
# Create index for model search
FT.CREATE models_idx ON HASH PREFIX 1 model: SCHEMA \
  name TEXT SORTABLE \
  status TEXT SORTABLE \
  accuracy NUMERIC SORTABLE \
  tags TAG \
  description TEXT

# Add model to index
HSET model:12345 \
  name "ResNet-50-v2" \
  status "production" \
  accuracy 0.942 \
  tags "computer_vision,image_classification" \
  description "Improved ResNet-50 with better regularization"

# Hybrid search
FT.SEARCH models_idx "@tags:{computer_vision} @accuracy:[0.9 inf]" SORTBY accuracy DESC
```

#### RedisAI for In-Process Inference
```bash
# Load model into Redis
AI.MODELSTORE resnet50 TF CPU INPUTS input OUTPUTS output BLOB $model_bytes

# Run inference
AI.MODELEXECUTE resnet50 INPUTS input_tensor OUTPUTS output_tensor

# Get results
AI.TENSORGET output_tensor VALUES
```

## Performance Optimization for Real-Time AI

### Connection Management
```python
# Python example with connection pooling
import redis
from redis.connection import ConnectionPool

pool = ConnectionPool(
    host='localhost',
    port=6379,
    password='ai_password',
    max_connections=100,
    decode_responses=True
)

redis_client = redis.Redis(connection_pool=pool)

# Use pipeline for batch operations
pipe = redis_client.pipeline()
for i in range(100):
    pipe.hget(f'user:{i}:features', 'engagement_score')
results = pipe.execute()
```

### Memory Optimization
```bash
# Use appropriate data types
# Instead of string: "12345" → use integer: 12345
SET user:12345:count 12345  # Integer encoding

# Use hashes for objects instead of separate keys
# Instead of:
# SET user:12345:name "John"
# SET user:12345:age "28"
# Use:
HSET user:12345 name "John" age 28

# Use bitmaps for boolean flags
SETBIT user_flags:12345 0 1  # user is active
SETBIT user_flags:12345 1 0  # user is not premium
```

### Latency Optimization
```bash
# Use pipelining for multiple commands
PIPELINE
HGET user:12345:features age
HGET user:12345:features engagement_score
HGET user:12345:features region
EXEC

# Use MULTI/EXEC for atomic operations
MULTI
HINCRBY user:12345:features session_count 1
HSET user:12345:features last_activity "2026-02-15T10:30:00Z"
EXEC
```

## Common Redis Pitfalls for AI Engineers

### 1. Memory Bloat from Large Keys
- **Problem**: Storing large embeddings or feature vectors as strings
- **Solution**: Use compression, quantization, or external storage
- **Example**: Store 768-dim float32 vector as 3KB instead of 3KB+ overhead

### 2. Hot Key Problems
- **Problem**: Single key receiving excessive traffic
- **Solution**: Shard hot keys, use client-side caching
- **Example**: Instead of `counter:total_requests`, use `counter:total_requests:{shard}`

### 3. Blocking Operations
- **Problem**: Long-running operations blocking other clients
- **Solution**: Use async operations, avoid KEYS command
- **Example**: Use SCAN instead of KEYS for large datasets

### 4. Persistence Overhead
- **Problem**: RDB/AOF causing latency spikes
- **Solution**: Tune persistence settings, use replica for backups
- **Example**: `save ""` to disable RDB, rely on AOF with `everysec`

## Visual Diagrams

### Redis Architecture for Real-Time AI
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Inference     │───▶│  Redis Client   │───▶│    Redis        │
│   Service       │    │  (Application)  │    │   (In-Memory)   │
└─────────────────┘    └────────┬────────┘    └────────┬────────┘
                                │                         │
                                ▼                         ▼
                      ┌─────────────────┐       ┌─────────────────┐
                      │  Connection     │       │  Data Structures  │
                      │  Pooling        │       │  • Hashes         │
                      └─────────────────┘       │  • Sorted Sets    │
                                                │  • Lists          │
                                                │  • Sets, Streams  │
                                                └─────────────────┘
```

### Real-Time Feature Serving Pipeline
```
User Request → [API Gateway] → [Feature Service] → Redis (Features)
       ↑              │                │                │
       │              ▼                ▼                ▼
       └── [Model Inference] ← [Feature Vector] ← [Online Features]
                                 ▲
                                 │
                         [Real-time Event Processor]
                                 │
                         [Data Pipeline → Redis Streams]
```

## Hands-on Exercises

### Exercise 1: Real-Time Feature Store
1. Implement hash-based feature storage for users
2. Create pipeline for batch feature retrieval
3. Implement TTL-based expiration for stale features
4. Test latency with different data sizes

### Exercise 2: Rate Limiting System
1. Build token bucket rate limiter using Redis
2. Implement sliding window algorithm
3. Test with simulated high-concurrency requests
4. Monitor Redis memory usage during stress testing

### Exercise 3: Real-Time Recommendation Engine
1. Implement sorted set for personalized recommendations
2. Create Lua script for real-time scoring updates
3. Build stream processor for user behavior events
4. Test end-to-end latency from event to recommendation

## Best Practices Summary

1. **Use hashes for objects**: More memory-efficient than separate keys
2. **Leverage pipelines**: Reduce round trips for batch operations
3. **Monitor memory usage**: Set appropriate maxmemory policies
4. **Use appropriate data structures**: Choose based on access patterns
5. **Implement proper error handling**: Redis failures should not crash AI services
6. **Plan for clustering**: Scale horizontally as traffic grows
7. **Use Redis modules strategically**: JSON, Search, AI for specialized needs
8. **Test under load**: Simulate production traffic patterns

This tutorial provides the foundation for building high-performance real-time AI systems using Redis, from feature serving to real-time inference and event processing.