# System Design: ML Model Serving at Scale (10K req/s)

## Problem Statement

Design a production ML model serving system that can:
- Serve predictions at 10,000 requests per second
- Support multiple ML models simultaneously (versioning)
- Maintain <50ms p95 latency for inference
- Enable safe deployment (A/B testing, canary, blue-green)
- Handle model updates without downtime
- Scale horizontally as traffic grows

---

## High-Level Architecture

```
┌────────────────┐
│  Client Apps   │
└───────┬────────┘
        │
        ▼
┌──────────────────────────────────────┐
│   API Gateway + Load Balancer        │
│   (NGINX / Kong / AWS ALB)           │
│   - Rate limiting                    │
│   - Authentication                   │
│   - Traffic routing                  │
└────────┬─────────────────────────────┘
         │
         ├──────────────┬──────────────┐
         ▼              ▼              ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐
   │Inference│    │Inference│    │Inference│
   │ Server  │    │ Server  │    │ Server  │
   │  Pod 1  │    │  Pod 2  │    │  Pod N  │
   └────┬────┘    └────┬────┘    └────┬────┘
        │              │              │
        └──────────────┴──────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
  ┌──────────┐  ┌──────────┐  ┌──────────┐
  │  Redis   │  │  Model   │  │Monitoring│
  │  Cache   │  │ Registry │  │(Prom +   │
  │          │  │ (MLflow) │  │ Grafana) │
  └──────────┘  └──────────┘  └──────────┘
```

---

## Component Deep Dive

### 1. Model Serving Framework

**Options Comparison**:

| Framework | Pros | Cons | Best For |
|-----------|------|------|----------|
| **TorchServe** | PyTorch native, dynamic batching | Limited ecosystem | PyTorch models |
| **TensorFlow Serving** | Mature, gRPC support | TF-only | TensorFlow models |
| **Triton** | Multi-framework, GPU optimization | Complex setup | Mixed frameworks |
| **FastAPI + Custom** | Full control, flexible | Manual batching | Simple models |

**Recommended**: **Triton Inference Server** (multi-framework support) + **FastAPI** (orchestration)

**Triton Configuration**:
```protobuf
# model_config.pbtxt
name: "my_model"
platform: "pytorch_libtorch"
max_batch_size: 32
input [
  {
    name: "input__0"
    data_type: TYPE_FP32
    dims: [ 768 ]
  }
]
output [
  {
    name: "output__0"
    data_type: TYPE_FP32
    dims: [ 10 ]
  }
]
dynamic_batching {
  preferred_batch_size: [ 8, 16, 32 ]
  max_queue_delay_microseconds: 5000  # Wait 5ms for batch
}
```

---

### 2. Dynamic Batching Strategy

**Goal**: Increase throughput without hurting latency

```python
class BatchingPredictor:
    def __init__(self, model, max_batch_size=32, max_wait_ms=5):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue = asyncio.Queue()
        
    async def predict(self, input_data):
        # Add to queue with future
        future = asyncio.Future()
        await self.queue.put((input_data, future))
        return await future  # Wait for batch processing
    
    async def _batch_worker(self):
        while True:
            batch = []
            futures = []
            
            # Collect batch (max size or timeout)
            deadline = time.time() + self.max_wait_ms / 1000
            while len(batch) < self.max_batch_size and time.time() < deadline:
                try:
                    data, future = await asyncio.wait_for(
                        self.queue.get(), 
                        timeout=deadline - time.time()
                    )
                    batch.append(data)
                    futures.append(future)
                except asyncio.TimeoutError:
                    break
            
            if not batch:
                continue
            
            # Run inference on batch
            results = self.model.predict_batch(batch)
            
            # Return results to individual futures
            for future, result in zip(futures, results):
                future.set_result(result)
```

**Performance Impact**:
- Single request: 10ms per prediction
- Batch of 32: 15ms per batch = **0.47ms per prediction** (21x improvement)

---

### 3. Model Versioning & Registry

**MLflow Model Registry**:

```python
import mlflow

class ModelManager:
    def __init__(self):
        self.registry = mlflow.tracking.MlflowClient()
        self.loaded_models = {}  # model_name:version -> model object
        
    def register_model(self, model_path, model_name):
        """Register new model version"""
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, model_name)
        
    def promote_to_production(self, model_name, version):
        """Promote model version to production stage"""
        self.registry.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production"
        )
        
    def load_production_model(self, model_name):
        """Load current production version"""
        versions = self.registry.get_latest_versions(
            model_name, 
            stages=["Production"]
        )
        if not versions:
            raise ValueError(f"No production model for {model_name}")
        
        model_uri = versions[0].source
        return mlflow.pytorch.load_model(model_uri)
```

**Version Tracking**:
- Each model deployed with version tag: `model_name:v1.2.3`
- Metadata stored: training metrics, dataset hash, hyperparameters
- Rollback capability: Instantly revert to previous version

---

### 4. Deployment Strategies

#### Blue-Green Deployment

```
┌──────────────┐
│   Traffic    │
│   (100%)     │
└──────┬───────┘
       │
       ▼
┌──────────────┐       ┌──────────────┐
│ Blue (v1.0)  │       │ Green (v2.0) │
│  Current     │       │   Standby    │
│  ✓ Serving   │       │   ❌ Testing  │
└──────────────┘       └──────────────┘

        Switch Traffic (instant)
                ↓

┌──────────────┐       ┌──────────────┐
│ Blue (v1.0)  │       │ Green (v2.0) │
│  Standby     │       │   Serving    │
│  ❌ Draining │       │   ✓ Active   │
└──────────────┘       └──────────────┘
```

**Implementation**:
```yaml
# Kubernetes Service
apiVersion: v1
kind: Service
metadata:
  name: model-service
spec:
  selector:
    app: model-server
    version: blue  # Switch to 'green' for deployment
  ports:
    - port: 8080
```

#### Canary Deployment

```python
class CanaryRouter:
    def __init__(self, canary_percent=10):
        self.canary_percent = canary_percent
        self.stable_model = load_model("stable")
        self.canary_model = load_model("canary")
        
    async def predict(self, request):
        # Route based on hash (consistent per user)
        if hash(request.user_id) % 100 < self.canary_percent:
            model = self.canary_model
            version = "canary"
        else:
            model = self.stable_model
            version = "stable"
        
        result = await model.predict(request.data)
        
        # Log for comparison
        self._log_metrics(version, result, request)
        
        return result
```

**Canary Rollout**:
1. Start: 5% traffic to new model
2. Monitor: Latency, error rate, business metrics
3. Gradual increase: 5% → 10% → 25% → 50% → 100%
4. Each step: 30 minutes monitoring
5. Rollback: If error rate > baseline + 1%

---

### 5. Caching Layer

**Multi-Tier Caching**:

```python
class PredictionCache:
    def __init__(self):
        self.redis = redis.Redis(decode_responses=True)
        self.local_cache = TTLCache(maxsize=1000, ttl=60)  # 1 min
        
    async def get_prediction(self, input_hash):
        # L1: Local in-memory cache (1ms)
        if result := self.local_cache.get(input_hash):
            return result
        
        # L2: Redis cache (5ms)
        if cached := self.redis.get(f"pred:{input_hash}"):
            result = json.loads(cached)
            self.local_cache[input_hash] = result
            return result
        
        return None  # Cache miss
    
    async def set_prediction(self, input_hash, result, ttl=3600):
        # Store in both layers
        self.local_cache[input_hash] = result
        self.redis.setex(
            f"pred:{input_hash}", 
            ttl, 
            json.dumps(result)
        )
```

**Cache Hit Rate**: 30-40% for common predictions (e.g., spam detection)

---

### 6. Horizontal Scaling

**Kubernetes HPA (Horizontal Pod Autoscaler)**:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: model-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: model-server
  minReplicas: 5
  maxReplicas: 50
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Pods
    pods:
      metric:
        name: requests_per_second
      target:
        type: AverageValue
        averageValue: "200"  # Scale when >200 RPS per pod
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
```

**Capacity Planning** (10K req/s):
- Assume 200 req/s per pod (with batching)
- Required pods: 10,000 / 200 = **50 pods**
- Buffer for spikes: 50 × 1.5 = **75 max pods**
- Cost: ~$0.10/hour per pod = **$360/month** (sustained 50 pods)

---

### 7. Monitoring & Observability

**Key Metrics** (Prometheus):

```python
from prometheus_client import Histogram, Counter, Gauge

# Latency
INFERENCE_LATENCY = Histogram(
    'model_inference_latency_seconds',
    'Time spent in model inference',
    ['model_name', 'model_version']
)

# Throughput
PREDICTIONS_TOTAL = Counter(
    'model_predictions_total',
    'Total predictions served',
    ['model_name', 'model_version', 'status']
)

# Batch size distribution
BATCH_SIZE = Histogram(
    'model_batch_size',
    'Distribution of batch sizes',
    buckets=[1, 2, 4, 8, 16, 32, 64]
)

# GPU utilization (if applicable)
GPU_MEMORY = Gauge(
    'gpu_memory_usage_bytes',
    'GPU memory usage',
    ['gpu_id']
)

# Model staleness
MODEL_AGE_DAYS = Gauge(
    'model_age_days',
    'Days since model was trained',
    ['model_name']
)
```

**Grafana Dashboard Panels**:
1. **Latency heatmap**: p50, p95, p99 by model version
2. **Throughput**: Requests/sec over time
3. **Error rate**: 4xx, 5xx errors
4. **Batch efficiency**: Avg batch size, batching utilization
5. **Cache hit rate**: L1, L2 hit rates

**Alerts**:
```yaml
groups:
  - name: model_serving
    rules:
      - alert: HighInferenceLatency
        expr: histogram_quantile(0.95, model_inference_latency_seconds) > 0.050
        for: 2m
        annotations:
          summary: "p95 latency > 50ms"
          
      - alert: HighErrorRate
        expr: rate(model_predictions_total{status="error"}[5m]) > 0.01
        for: 1m
        annotations:
          summary: "Error rate > 1%"
          
      - alert: ModelStaleness
        expr: model_age_days > 30
        annotations:
          summary: "Model hasn't been retrained in 30 days"
```

---

### 8. A/B Testing Integration

**Experiment Configuration**:

```python
class ABExperiment:
    def __init__(self, experiment_id, variants):
        self.experiment_id = experiment_id
        self.variants = variants  # {"control": 0.9, "treatment": 0.1}
        
    def assign_variant(self, user_id):
        """Consistent hash-based assignment"""
        hash_val = hashlib.md5(f"{user_id}{self.experiment_id}".encode())
        bucket = int(hash_val.hexdigest(), 16) % 100
        
        cumulative = 0
        for variant, percentage in self.variants.items():
            cumulative += percentage * 100
            if bucket < cumulative:
                return variant
        
        return "control"

@app.post("/predict")
async def predict(request: PredictRequest):
    # Assign experiment variant
    variant = experiment.assign_variant(request.user_id)
    
    # Route to appropriate model
    model = "model_v1" if variant == "control" else "model_v2"
    result = await models[model].predict(request.data)
    
    # Log for analysis
    analytics.log_event(
        user_id=request.user_id,
        experiment_id="model_v2_test",
        variant=variant,
        prediction=result,
        timestamp=datetime.utcnow()
    )
    
    return result
```

**Statistical Significance Testing**:
```python
from scipy.stats import ttest_ind

def analyze_experiment(experiment_id):
    """Compare treatment vs control"""
    control_metrics = get_metrics(experiment_id, "control")
    treatment_metrics = get_metrics(experiment_id, "treatment")
    
    # Two-sample t-test
    t_stat, p_value = ttest_ind(control_metrics, treatment_metrics)
    
    # Calculate confidence interval
    control_mean = np.mean(control_metrics)
    treatment_mean = np.mean(treatment_metrics)
    uplift_pct = (treatment_mean - control_mean) / control_mean * 100
    
    return {
        "control_mean": control_mean,
        "treatment_mean": treatment_mean,
        "uplift_pct": uplift_pct,
        "p_value": p_value,
        "is_significant": p_value < 0.05
    }
```

---

### 9. Latency Optimization Techniques

#### GPU Optimization
```python
# Mixed precision inference (FP16)
model.half()  # Convert to FP16
input_tensor = input_tensor.half()

# Speedup: 2-3x on V100/A100 GPUs
```

#### Model Quantization
```python
import torch.quantization

# Post-training quantization (INT8)
model_int8 = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Speedup: 2-4x, Model size: -75%
```

#### TorchScript Compilation
```python
# JIT compilation
scripted_model = torch.jit.script(model)
scripted_model.save("model_scripted.pt")

# Speedup: 1.5-2x (removes Python overhead)
```

**Latency Breakdown** (Target: <50ms p95):

| Component | Latency | Optimization |
|-----------|---------|--------------|
| Input preprocessing | 2ms | Batch normalization |
| Model inference | 25ms | INT8 quantization + TorchScript |
| Postprocessing | 3ms | Vectorized ops |
| Network overhead | 10ms | HTTP/2, gRPC |
| Batching wait | 5ms | Max 5ms queue delay |
| **Total** | **45ms** | Within target ✓ |

---

### 10. Cost Optimization

**Estimated Monthly Cost** (10K req/s, 24/7):

| Component | Cost | Notes |
|-----------|------|-------|
| Compute (50 pods) | $3,600 | c5.2xlarge spot instances |
| GPU (if needed) | $2,000 | 5x g4dn.xlarge (T4 GPUs) |
| Load Balancer | $200 | AWS ALB |
| Redis Cache | $100 | ElastiCache r6g.large |
| Model Registry | $50 | S3 storage + MLflow |
| Monitoring | $400 | Prometheus + Grafana Cloud |
| Data Transfer | $500 | Egress costs |
| **Total (CPU)** | **~$4,850/month** | Without GPU |
| **Total (GPU)** | **~$6,850/month** | With GPU |

**Cost Reduction Strategies**:
1. **Spot instances**: -70% compute cost (use with autoscaling)
2. **ONNX runtime**: Framework-agnostic, faster inference
3. **Distillation**: Smaller model, same accuracy (-80% inference cost)
4. **Aggressive caching**: 40% hit rate = -40% compute cost
5. **Regional routing**: Serve from cheaper regions

---

## Trade-offs & Decisions

| Decision | Option A | Option B | Choice |
|----------|----------|----------|--------|
| Serving Framework | Custom FastAPI | Triton | Triton (GPU support) |
| Batching | Static (fixed size) | Dynamic (timeout) | Dynamic (lower latency) |
| Deployment | Blue-Green (instant) | Canary (gradual) | Canary (safer) |
| Hardware | CPU (cheap) | GPU (fast) | CPU for simple models |
| Caching | None (always fresh) | Redis (faster) | Redis (40% savings) |

---

## Interview Discussion Points

1. **How would you handle model warm-up latency?**
   - Pre-load models on startup
   - Health check endpoint that exercises model
   - Keep "warm" instances in pool

2. **What if a model version has a critical bug?**
   - Instant rollback to previous version (blue-green)
   - Circuit breaker: Auto-disable if error rate spikes
   - Shadow deployment: Test new version without serving traffic

3. **How to optimize for cost vs latency?**
   - Use CPU for simple models (<5ms inference)
   - GPU only for large transformers (>100ms CPU time)
   - Batch size tuning: Larger batches = higher throughput, higher latency

4. **Multi-tenancy concerns?**
   - Resource quotas per tenant (CPU, memory)
   - Separate model instances for premium customers
   - QoS: Priority queue for high-value requests

5. **How to handle model drift in production?**
   - Monitor prediction distribution shifts
   - Alert if input data diverges from training data
   - Automated retraining triggers

---

## Conclusion

This design serves 10K requests/second with <50ms p95 latency by:
- **Dynamic batching** (21x throughput improvement)
- **Horizontal scaling** (50 pods with autoscaling)
- **Multi-tier caching** (40% cache hit rate)
- **Safe deployments** (canary with statistical testing)

**Production-ready checklist**:
- ✅ Handles traffic spikes (autoscaling to 75 pods)
- ✅ Fault-tolerant (circuit breaker, retries)
- ✅ Observable (latency, throughput, error metrics)
- ✅ Cost-effective (<$5K/month for CPU, <$7K with GPU)
- ✅ A/B testing ready (variant assignment + metrics)

**Typical Interview Flow**:
1. Start with requirements (QPS, latency, models)
2. Draw high-level architecture (5 min)
3. Deep dive on 2-3 components (batching, caching, deployment)
4. Discuss trade-offs and optimizations (10 min)
5. Cost estimation and scaling (5 min)
