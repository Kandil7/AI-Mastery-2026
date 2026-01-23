# Scaling

This section provides comprehensive information about scaling the Production RAG System to handle increased load, data volume, and user demand.

## Scaling Overview

The Production RAG System is designed with horizontal and vertical scaling capabilities to accommodate growing demands. The system follows microservices principles with clear separation of concerns to enable independent scaling of different components.

## Scaling Strategies

### 1. Horizontal Scaling

#### API Layer Scaling
The API layer can be scaled horizontally by running multiple instances behind a load balancer:

```yaml
# Example Kubernetes deployment for API scaling
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
spec:
  replicas: 3  # Start with 3 instances
  selector:
    matchLabels:
      app: rag-api
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      containers:
      - name: rag-api
        image: rag-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
---
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
spec:
  selector:
    app: rag-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

#### Load Balancer Configuration
```nginx
# Example Nginx load balancer configuration
upstream rag_api_backend {
    least_conn;  # Distribute requests based on least connections
    server rag-api-1:8000 max_fails=3 fail_timeout=30s;
    server rag-api-2:8000 max_fails=3 fail_timeout=30s;
    server rag-api-3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    
    location / {
        proxy_pass http://rag_api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Connection pooling
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

### 2. Vertical Scaling

#### Resource Allocation
Increase resources for individual components:

```yaml
# Example resource allocation for high-load scenario
resources:
  requests:
    memory: "2Gi"    # Increased memory request
    cpu: "1000m"     # Increased CPU request
  limits:
    memory: "4Gi"    # Increased memory limit
    cpu: "2000m"     # Increased CPU limit
```

#### Database Scaling
Scale database resources:

```yaml
# MongoDB resource scaling
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mongodb
spec:
  serviceName: mongodb
  replicas: 1
  template:
    spec:
      containers:
      - name: mongodb
        image: mongo:6.0
        resources:
          requests:
            memory: "4Gi"
            cpu: "1000m"
          limits:
            memory: "8Gi"
            cpu: "2000m"
        env:
        - name: MONGO_INITDB_ROOT_USERNAME
          valueFrom:
            secretKeyRef:
              name: mongodb-secret
              key: username
        - name: MONGO_INITDB_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mongodb-secret
              key: password
```

## Component-Specific Scaling

### 1. API Layer Scaling

#### FastAPI Configuration for Scaling
```python
# api.py - Optimized for scaling
from fastapi import FastAPI
import uvicorn
from src.config import settings

app = FastAPI(
    title="Production RAG API",
    description="Scaled RAG API with load balancing support",
    version="1.0.0",
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
)

# Add health check for load balancer
@app.get("/health")
async def health_check():
    return {"status": "healthy", "instance": settings.host}

if __name__ == "__main__":
    # Run with multiple workers for scaling
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        workers=4,  # Number of worker processes
        log_level="info",
        timeout_keep_alive=120,  # Keep-alive timeout
    )
```

#### Worker Process Management
```python
# Scaling configuration
WORKER_PROCESSES = 4  # Number of worker processes per instance
MAX_WORKER_CONNECTIONS = 1000  # Max connections per worker
TIMEOUT_KEEP_ALIVE = 120  # Keep-alive timeout in seconds
```

### 2. Database Scaling

#### MongoDB Sharding
For very large datasets, implement MongoDB sharding:

```yaml
# MongoDB sharding configuration
sharding:
  clusterRole: shardsvr
  replication:
    replSetName: rs0

shards:
  - name: shard0001
    size: 3  # Replica set size
    storage:
      size: 100Gi  # Storage size per shard
      class: fast-ssd
  - name: shard0002
    size: 3
    storage:
      size: 100Gi
      class: fast-ssd

config_servers:
  size: 3  # Odd number for consensus
  storage:
    size: 50Gi
    class: fast-ssd
```

#### Connection Pooling
Optimize database connection pooling:

```python
# src/ingestion/mongo_storage.py - Optimized connection pooling
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient

class MongoConnectionManager:
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.sync_client: Optional[MongoClient] = None
        self.logger = logging.getLogger(__name__)

    async def connect(self):
        """Establish connection with optimized settings for scaling."""
        try:
            self.client = AsyncIOMotorClient(
                settings.database.url,
                username=settings.database.username,
                password=settings.database.password,
                maxPoolSize=settings.database.pool_size,  # Configurable pool size
                minPoolSize=settings.database.pool_size // 2,  # Maintain minimum connections
                maxIdleTimeMS=30000,  # Close idle connections after 30s
                serverSelectionTimeoutMS=5000,  # 5 seconds timeout
                connectTimeoutMS=10000,  # 10 seconds connection timeout
                socketTimeoutMS=20000,  # 20 seconds socket timeout
                heartbeatFrequencyMS=10000,  # Heartbeat every 10s
            )
```

### 3. Vector Database Scaling

#### ChromaDB Scaling
```python
# src/retrieval/vector_store.py - ChromaDB scaling configuration
class ChromaVectorStore(BaseVectorStore):
    def __init__(self, config: VectorConfig):
        super().__init__(config)
        
        if not CHROMA_AVAILABLE:
            raise RuntimeError("ChromaDB is not available. Install with 'pip install chromadb'")
        
        self.client = None
        self.collection = None
        self.settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=config.persist_directory,
            anonymized_telemetry=False,
            # Scaling settings
            is_persistent=True,
            persist_n_threads=4,  # Number of threads for persistence
        )
```

#### FAISS for High Performance
For high-performance scenarios, consider FAISS:

```python
# FAISS configuration for scaling
class FAISSVectorStore(BaseVectorStore):
    def __init__(self, config: VectorConfig):
        super().__init__(config)
        
        # Configure FAISS for performance
        self.index = faiss.IndexFlatIP(config.dimension)
        
        # Use multiple indexes for sharding
        self.shard_count = 4  # Number of shards
        self.shards = [faiss.IndexFlatIP(config.dimension) for _ in range(self.shard_count)]
```

### 4. Model Scaling

#### Model Serving Scaling
```python
# Model serving with horizontal scaling
class ScaledModelService:
    def __init__(self, model_configs: List[Dict]):
        self.model_instances = []
        for config in model_configs:
            # Create model instance with specific resource allocation
            model = self.create_scaled_model(config)
            self.model_instances.append(model)
    
    def distribute_load(self, query: str):
        """Distribute load across model instances."""
        # Round-robin distribution
        instance = self.model_instances[hash(query) % len(self.model_instances)]
        return instance.generate(query)
```

#### Model Caching and Sharing
```python
# Shared model caching across instances
from functools import lru_cache
import threading

class ModelCache:
    def __init__(self):
        self.cache = {}
        self.lock = threading.RLock()
    
    @lru_cache(maxsize=10)  # Cache up to 10 models
    def get_model(self, model_name: str):
        """Get cached model instance."""
        from transformers import pipeline
        return pipeline("text-generation", model=model_name)
```

## Auto-Scaling Configuration

### Kubernetes Auto-Scaling

#### Horizontal Pod Autoscaler (HPA)
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

#### Custom Metrics Scaling
```yaml
# Scale based on custom metrics like query rate
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-custom-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 2
  maxReplicas: 20
  metrics:
  - type: Pods
    pods:
      metric:
        name: queries_per_second
      target:
        type: AverageValue
        averageValue: "100"  # Scale when > 100 QPS per pod
```

### Cloud Provider Scaling

#### AWS Auto Scaling
```json
{
  "AutoScalingGroupName": "rag-api-asg",
  "MinSize": 2,
  "MaxSize": 20,
  "DesiredCapacity": 3,
  "TargetTrackingConfiguration": {
    "TargetValue": 70.0,
    "PredefinedMetricSpecification": {
      "PredefinedMetricType": "ASGAverageCPUUtilization"
    },
    "ScaleOutCooldown": 300,
    "ScaleInCooldown": 300
  }
}
```

## Scaling Patterns

### 1. Database Read Replicas
```yaml
# MongoDB read replicas for scaling reads
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: mongodb-replica
spec:
  replicas: 3  # Primary + 2 replicas
  template:
    spec:
      containers:
      - name: mongodb
        image: mongo:6.0
        env:
        - name: MONGO_REPLICA_SET_NAME
          value: "rs0"
        - name: MONGO_INITDB_ROOT_USERNAME
          valueFrom:
            secretKeyRef:
              name: mongodb-secret
              key: username
        - name: MONGO_INITDB_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: mongodb-secret
              key: password
```

### 2. Caching Layer Scaling
```yaml
# Redis cluster for distributed caching
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: redis-cluster
spec:
  serviceName: redis-cluster
  replicas: 6  # 3 master + 3 slave nodes
  template:
    spec:
      containers:
      - name: redis
        image: redis:7-alpine
        command:
        - "redis-server"
        - "/etc/redis/redis.conf"
        ports:
        - containerPort: 6379
        volumeMounts:
        - name: redis-config
          mountPath: /etc/redis
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### 3. CDN and Edge Caching
```yaml
# CDN configuration for static assets
apiVersion: cloud.google.com/v1
kind: BackendConfig
metadata:
  name: cdn-config
spec:
  cdn:
    enabled: true
    cachePolicy:
      cacheMode: USE_ORIGIN_HEADERS
      defaultTtl: 3600
      maxTtl: 86400
      clientTtl: 3600
```

## Scaling Monitoring

### Key Scaling Metrics
```python
# src/observability/__init__.py - Scaling metrics
class ScalingMetrics:
    def __init__(self):
        self.metrics = {
            "cpu_usage": Gauge("cpu_usage_percentage", "CPU usage percentage"),
            "memory_usage": Gauge("memory_usage_bytes", "Memory usage in bytes"),
            "active_connections": Gauge("active_connections", "Active API connections"),
            "requests_per_second": Counter("requests_per_second", "Requests per second"),
            "average_response_time": Histogram("response_time_ms", "Response time in milliseconds"),
            "queue_size": Gauge("queue_size", "Size of processing queue"),
            "model_loading_time": Histogram("model_loading_time_ms", "Time to load models"),
        }
    
    def record_scaling_metrics(self):
        """Record metrics relevant to scaling decisions."""
        # CPU usage
        cpu_percent = psutil.cpu_percent()
        self.metrics["cpu_usage"].set(cpu_percent)
        
        # Memory usage
        memory_info = psutil.virtual_memory()
        self.metrics["memory_usage"].set(memory_info.used)
        
        # Active connections
        active_conns = len(active_connections)
        self.metrics["active_connections"].set(active_conns)
```

### Scaling Alerts
```yaml
# Prometheus alerting rules for scaling
groups:
- name: scaling_alerts
  rules:
  - alert: HighCPUUsage
    expr: cpu_usage_percentage > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage detected"
      description: "CPU usage is above 80% for more than 5 minutes"
  
  - alert: HighMemoryUsage
    expr: memory_usage_bytes / machine_memory_bytes > 0.85
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High memory usage detected"
      description: "Memory usage is above 85% for more than 5 minutes"
  
  - alert: HighResponseTime
    expr: histogram_quantile(0.95, response_time_ms_bucket) > 1000
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is above 1000ms"
```

## Scaling Best Practices

### 1. Gradual Scaling
- Scale gradually to avoid overwhelming the system
- Monitor metrics during scaling operations
- Implement scaling delays to allow for stabilization

### 2. Resource Planning
- Plan resources based on projected growth
- Consider seasonal variations in demand
- Account for peak usage periods

### 3. Testing at Scale
- Test scaling configurations in staging
- Perform load testing at expected peak loads
- Validate failover scenarios

### 4. Cost Optimization
- Use spot instances for non-critical workloads
- Implement auto-scaling to reduce costs during low demand
- Monitor and optimize resource utilization

### 5. Monitoring and Alerting
- Implement comprehensive monitoring
- Set up alerts for scaling triggers
- Monitor scaling effectiveness

## Multi-Region Scaling

### Global Deployment
```yaml
# Multi-region deployment configuration
apiVersion: v1
kind: ConfigMap
metadata:
  name: global-config
data:
  regions: |
    - name: us-east
      endpoint: https://us-east.rag-api.example.com
      weight: 40
    - name: eu-west
      endpoint: https://eu-west.rag-api.example.com
      weight: 35
    - name: ap-south
      endpoint: https://ap-south.rag-api.example.com
      weight: 25
```

### Traffic Distribution
```nginx
# Geographic load balancing
geo $country_code {
    default 0;
    US 1;
    GB 1;
    DE 1;
    IN 1;
}

map $country_code $backend_region {
    0 eu-west;  # Default to EU
    1 $geoip_country_code;
}

upstream rag_us_east {
    server us-east-rag-api-1:8000 weight=1;
    server us-east-rag-api-2:8000 weight=1;
}

upstream rag_eu_west {
    server eu-west-rag-api-1:8000 weight=1;
    server eu-west-rag-api-2:8000 weight=1;
}

upstream rag_ap_south {
    server ap-south-rag-api-1:8000 weight=1;
    server ap-south-rag-api-2:8000 weight=1;
}

server {
    listen 80;
    
    location / {
        # Route to region based on geography
        proxy_pass http://rag_$backend_region;
    }
}
```

## Scaling Considerations

### 1. Data Consistency
- Implement eventual consistency where appropriate
- Use distributed transactions when needed
- Consider CAP theorem trade-offs

### 2. Session Management
- Use stateless API design
- Implement distributed session storage
- Consider sticky sessions for certain workloads

### 3. Data Partitioning
- Implement sharding strategies
- Use consistent hashing for partitioning
- Plan for rebalancing as data grows

### 4. Network Latency
- Optimize for geographic distribution
- Implement CDN for static assets
- Consider edge computing for low-latency requirements

This scaling documentation provides a comprehensive guide to scaling the Production RAG System to meet growing demands while maintaining performance and reliability.