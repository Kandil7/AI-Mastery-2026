# System Design: ML Model Serving at Scale (10K Requests/Second)

## Problem Statement

Design a model serving system that can:
- Handle **10,000 requests/second** (10K QPS)
- Serve multiple ML models (TensorFlow, PyTorch, scikit-learn)
- Maintain **<50ms p95 latency** for inference
- Support **A/B testing** and gradual rollouts
- Provide **horizontal scalability** and high availability
- Enable **real-time model updates** without downtime

---

## High-Level Architecture

```
Client Requests (10K QPS)
        │
        ▼
┌───────────────────────┐
│  Global Load Balancer │
│  (GSLB + CDN)         │
└──────────┬────────────┘
           │
    ┌──────┴──────┐
    ▼             ▼
┌─────────┐ ┌─────────┐
│ Region  │ │ Region  │
│  US     │ │  EU     │
└────┬────┘ └────┬────┘
     │           │
     ▼           ▼
┌──────────────────────────┐
│  API Gateway (Kong)      │
│  - Routing               │
│  - Rate Limiting         │
│  - Authentication        │
└────────┬─────────────────┘
         │
    ┌────┴─────┬──────────┬────────┐
    ▼          ▼          ▼        ▼
┌────────┐ ┌────────┐ ┌────────┐ ...
│ Model  │ │ Model  │ │ Model  │
│Server 1│ │Server 2│ │Server N│
│(PyTorch│ │(TF)    │ │(sklearn)
└───┬────┘ └───┬────┘ └───┬────┘
    │          │           │
    └──────────┴───────────┘
           │
           ▼
    ┌─────────────┐
    │  Model      │
    │  Registry   │
    │  (S3/GCS)   │
    └─────────────┘
           │
           ▼
    ┌─────────────┐
    │  Monitoring │
    │  (Prometheus│
    │   + Grafana)│
    └─────────────┘
```

---

## Component Deep Dive

### 1. Model Server Framework Selection

**Options Comparison**:

| Framework | Pros | Cons | Best For |
|-----------|------|------|----------|
| **TensorFlow Serving** | Optimized for TF, batching | TF-only | TensorFlow models |
| **TorchServe** | PyTorch native, good docs | PyTorch-only | PyTorch models |
| **NVIDIA Triton** | Multi-framework, GPU-optimized | Complex setup | Mixed models + GPU |
| **FastAPI + Custom** | Full control, lightweight | Manual optimization | Custom needs |
| **BentoML** | Easy deployment, multi-framework | Smaller community | Rapid prototyping |

**Recommended**: **NVIDIA Triton Inference Server**
- Supports TensorFlow, PyTorch, ONNX, scikit-learn
- Dynamic batching out-of-the-box
- GPU optimization
- Model ensemble support

---

### 2. Triton Model Server Setup

**Model Repository Structure**:
```
models/
├── fraud_detector_v1/
│   ├── config.pbtxt
│   └── 1/
│       └── model.pt
├── recommendation_v2/
│   ├── config.pbtxt
│   └── 2/
│       └── model.savedmodel/
└── text_classifier/
    ├── config.pbtxt
    └── 1/
        └── model.onnx
```

**Configuration Example** (`config.pbtxt`):
```protobuf
name: "fraud_detector_v1"
platform: "pytorch_libtorch"
max_batch_size: 128
input [
  {
    name: "input_features"
    data_type: TYPE_FP32
    dims: [ 64 ]
  }
]
output [
  {
    name: "fraud_probability"
    data_type: TYPE_FP32
    dims: [ 1 ]
  }
]

# Dynamic batching for throughput
dynamic_batching {
  max_queue_delay_microseconds: 100
  preferred_batch_size: [ 32, 64, 128 ]
}

# Instance group (GPU/CPU allocation)
instance_group [
  {
    count: 2
    kind: KIND_GPU
    gpus: [ 0 ]
  }
]

# Model warmup
model_warmup {
  name: "warmup_sample"
  batch_size: 32
  inputs {
    key: "input_features"
    value: {
      data_type: TYPE_FP32
      dims: [ 64 ]
      zero_data: true
    }
  }
}
```

---

### 3. Dynamic Batching Strategy

**Problem**: Individual requests are inefficient (GPU underutilization)

**Solution**: Batch multiple requests together

**Implementation**:
```python
class DynamicBatcher:
    def __init__(self, max_batch_size=128, max_delay_ms=10):
        self.max_batch_size = max_batch_size
        self.max_delay_ms = max_delay_ms
        self.queue = []
        self.lock = threading.Lock()
    
    async def add_request(self, request):
        """Add request to batch queue."""
        future = asyncio.Future()
        
        with self.lock:
            self.queue.append((request, future))
            
            # Trigger batch if full
            if len(self.queue) >= self.max_batch_size:
                await self._process_batch()
        
        # Wait for result
        return await future
    
    async def _process_batch(self):
        """Process accumulated requests as a batch."""
        with self.lock:
            if not self.queue:
                return
            
            batch = self.queue[:self.max_batch_size]
            self.queue = self.queue[self.max_batch_size:]
        
        # Extract inputs
        requests = [req for req, _ in batch]
        futures = [fut for _, fut in batch]
        
        # Batch inference
        inputs = np.stack([r.features for r in requests])
        predictions = await model.predict_batch(inputs)
        
        # Return results to individual requests
        for future, pred in zip(futures, predictions):
            future.set_result(pred)
```

**Benefits**:
- **10-50x throughput increase** on GPU
- Automatic queue management
- Configurable delay vs throughput trade-off

---

### 4. Model Optimization Techniques

#### A. Quantization (INT8)

**FP32 → INT8**: 4x memory reduction, 2-4x speedup

```python
import torch

# Post-training quantization
model_fp32 = torch.load('model.pt')
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32,
    {torch.nn.Linear},  # Layers to quantize
    dtype=torch.qint8
)

# Save quantized model
torch.save(model_int8.state_dict(), 'model_int8.pt')
```

**Expected Impact**:
- Latency: 50ms → 15ms
- Memory: 400MB → 100MB
- Accuracy drop: <1% typically

---

#### B. ONNX Conversion

**Benefits**: Optimized runtime, cross-framework compatibility

```python
import torch
import onnx
from onnxruntime import InferenceSession

# Export PyTorch to ONNX
dummy_input = torch.randn(1, 64)
torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={'input': {0: 'batch_size'}}
)

# Optimize ONNX graph
from onnxruntime.transformers import optimizer
optimized_model = optimizer.optimize_model(
    "model.onnx",
    model_type='bert',
    num_heads=12,
    hidden_size=768
)
optimized_model.save_model_to_file("model_optimized.onnx")

# Inference with ONNX Runtime
session = InferenceSession("model_optimized.onnx")
output = session.run(None, {'input': input_data})
```

---

#### C. TensorRT (NVIDIA GPUs)

**Extreme optimization for NVIDIA cards**:

```python
import tensorrt as trt

# Convert ONNX to TensorRT
logger = trt.Logger(trt.Logger.WARNING)
builder = trt.Builder(logger)
network = builder.create_network()
parser = trt.OnnxParser(network, logger)

# Parse ONNX model
with open('model.onnx', 'rb') as f:
    parser.parse(f.read())

# Build optimized engine
config = builder.create_builder_config()
config.max_workspace_size = 1 << 30  # 1GB
config.set_flag(trt.BuilderFlag.FP16)  # FP16 precision

engine = builder.build_engine(network, config)

# Save engine
with open('model.trt', 'wb') as f:
    f.write(engine.serialize())
```

**Performance Gain**: 5-10x speedup on NVIDIA GPUs

---

### 5. A/B Testing & Canary Deployments

**Traffic Splitting Strategy**:

```python
class ModelRouter:
    def __init__(self):
        self.models = {
            'model_v1': {'weight': 90, 'endpoint': 'triton-v1:8000'},
            'model_v2': {'weight': 10, 'endpoint': 'triton-v2:8000'}  # Canary
        }
    
    def route_request(self, request):
        """Route request to model based on weights."""
        # Hash user ID for consistent routing
        hash_val = hash(request.user_id) % 100
        
        cumulative_weight = 0
        for model_id, config in self.models.items():
            cumulative_weight += config['weight']
            if hash_val < cumulative_weight:
                return config['endpoint']
        
        return self.models['model_v1']['endpoint']  # Fallback
```

**Gradual Rollout**:
1. Deploy v2 with 5% traffic
2. Monitor for 24h
3. If metrics good, increase to 25%
4. Continue until 100% or rollback

**Monitoring Metrics**:
```python
from prometheus_client import Counter, Histogram

MODEL_REQUESTS = Counter('model_requests_total', 'Total requests', ['model_version'])
MODEL_LATENCY = Histogram('model_latency_seconds', 'Latency', ['model_version'])
MODEL_ERRORS = Counter('model_errors_total', 'Errors', ['model_version'])

@app.post("/predict")
async def predict(request: PredictionRequest):
    model_version = router.select_model(request)
    
    with MODEL_LATENCY.labels(model_version=model_version).time():
        try:
            result = await client.infer(model_version, request.features)
            MODEL_REQUESTS.labels(model_version=model_version).inc()
            return result
        except Exception as e:
            MODEL_ERRORS.labels(model_version=model_version).inc()
            raise
```

---

### 6. Horizontal Scaling Architecture

**Auto-Scaling Strategy**:

```yaml
# Kubernetes HorizontalPodAutoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: triton-autoscaler
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: triton-inference-server
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
        name: inference_requests_per_second
      target:
        type: AverageValue
        averageValue: "200"  # 200 RPS per pod
```

**Load Balancing** (NGINX):
```nginx
upstream triton_cluster {
    least_conn;  # Route to least busy server
    
    server triton-1:8000 max_fails=3 fail_timeout=30s;
    server triton-2:8000 max_fails=3 fail_timeout=30s;
    server triton-3:8000 max_fails=3 fail_timeout=30s;
    # ... up to 50 pods
    
    keepalive 32;  # Connection pooling
}

server {
    listen 80;
    
    location /v2/models {
        proxy_pass http://triton_cluster;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 30s;
    }
}
```

---

### 7. Caching Layer

**Multi-Level Caching**:

```
┌──────────────────┐
│ L1: In-Memory    │  5ms (LRU, 10K requests)
│    (Redis)       │
└────────┬─────────┘
         │ Miss (2%)
         ▼
┌──────────────────┐
│ L2: Feature Hash │  15ms (100K requests)
│    (Redis)       │
└────────┬─────────┘
         │ Miss (10%)
         ▼
┌──────────────────┐
│ L3: Model Infer  │  30-50ms
│    (Triton)      │
└──────────────────┘
```

**Implementation**:
```python
import hashlib
import redis

redis_client = redis.Redis(host='localhost', port=6379)

def predict_with_cache(features):
    # Create cache key from features
    feature_hash = hashlib
.md5(features.tobytes()).hexdigest()
    
    # L1: Check exact match cache
    cache_key = f"pred:{feature_hash}"
    cached = redis_client.get(cache_key)
    
    if cached:
        return json.loads(cached)
    
    # L2: Check similar features (LSH - Locality Sensitive Hashing)
    similar_key = lsh.find_similar(features, threshold=0.99)
    if similar_key:
        cached = redis_client.get(f"pred:{similar_key}")
        if cached:
            return json.loads(cached)
    
    # L3: Perform actual inference
    prediction = model.predict(features)
    
    # Cache result (TTL = 1 hour)
    redis_client.setex(cache_key, 3600, json.dumps(prediction))
    
    return prediction
```

**Cache Hit Rate**: 85-90% for common patterns

---

### 8. Latency Budget Breakdown

**Target: <50ms p95 for 10K QPS**

| Component | Latency | Optimization |
|-----------|---------|--------------|
| Load Balancer | 2ms | Connection pooling |
| API Gateway | 3ms | Kubernetes service mesh |
| Cache Lookup | 5ms | Redis cluster (3 nodes) |
| Model Inference | 25ms | INT8 quantization + batching |
| Response Serialization | 2ms | Protocol Buffers |
| Network Overhead | 8ms | Regional deployment |
| **Total** | **45ms** | ✅ Within budget |

---

### 9. Capacity Planning

**For 10,000 QPS**:

**GPU Instances** (NVIDIA T4):
- Per-GPU throughput: 500 RPS (with batching)
- Required: 10,000 / 500 = **20 GPUs**
- AWS p3.2xlarge: 1 GPU each → **20 instances**
- With 2x redundancy: **40 instances**

**CPU Instances** (fallback, no GPU):
- Per-CPU throughput: 50 RPS
- Required: 10,000 / 50 = **200 instances**
- Much more expensive than GPUs!

**Cost Comparison**:
| Setup | Monthly Cost | Latency p95 |
|-------|--------------|-------------|
| 40x p3.2xlarge (GPU) | $45,000 | 30ms |
| 200x c5.4xlarge (CPU) | $120,000 | 80ms |

**Verdict**: GPUs are 2.7x cheaper AND 2.7x faster!

---

### 10. Monitoring & Observability

**Key Metrics** (Prometheus + Grafana):

```python
# Latency
INFERENCE_LATENCY = Histogram(
    'inference_latency_seconds',
    'Model inference latency',
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0]
)

# Throughput
REQUESTS_PER_SECOND = Gauge('requests_per_second', 'Current RPS')

# Error Rate
ERROR_RATE = Counter('inference_errors_total', 'Total errors', ['error_type'])

# GPU Utilization
GPU_UTILIZATION = Gauge('gpu_utilization_percent', 'GPU usage', ['gpu_id'])

# Queue Depth
BATCH_QUEUE_SIZE = Gauge('batch_queue_size', 'Requests waiting for batching')
```

**Alerting Rules**:
```yaml
groups:
- name: model_serving_alerts
  rules:
  - alert: HighLatency
    expr: histogram_quantile(0.95, inference_latency_seconds) > 0.05
    for: 5m
    annotations:
      summary: "P95 latency >50ms"
  
  - alert: HighErrorRate
    expr: rate(inference_errors_total[5m]) / rate(inference_requests_total[5m]) > 0.01
    for: 2m
    annotations:
      summary: "Error rate >1%"
  
  - alert: GPUDown
    expr: gpu_utilization_percent == 0
    for: 1m
    annotations:
      summary: "GPU appears offline"
```

---

### 11. Model Update Strategy

**Zero-Downtime Deployment**:

```
1. Upload new model to registry
2. Triton hot-loads model (no restart)
3. Warm up new model (send dummy requests)
4. Gradually shift traffic:
   - 0% → 5% → 25% → 50% → 100%
5. Monitor metrics for regressions
6. Rollback if needed (instant)
```

**Implementation**:
```python
import requests

def deploy_model_version(model_name, version):
    # Load new version
    triton_admin_url = "http://triton:8001/v2/repository/models/load"
    response = requests.post(triton_admin_url, json={
        "model_name": model_name,
        "parameters": {"version": version}
    })
    
    # Warm up
    warmup_requests = generate_warmup_data(n=1000)
    for req in warmup_requests:
        _ = infer(model_name, req, version=version)
    
    # Gradually shift traffic
    for traffic_pct in [5, 25, 50, 75, 100]:
        update_traffic_split(model_name, old_version, version, traffic_pct)
        time.sleep(300)  # Wait 5 min
        
        # Check metrics
        if check_regression(model_name, version):
            rollback(model_name, old_version)
            return False
    
    # Unload old version
    unload_model_version(model_name, old_version)
    return True
```

---

### 12. Cost Optimization

**Monthly Cost** (10K QPS, 99.9% SLA):

| Component | Cost | Notes |
|-----------|------|-------|
| Triton Servers (40x p3.2xlarge) | $45,000 | GPUs |
| Load Balancers (2x) | $500 | AWS ALB |
| Redis Cache (r5.4xlarge) | $2,000 | 3-node cluster |
| S3 Model Storage | $100 | ~500GB |
| Data Transfer | $3,000 | Egress |
| Monitoring (Grafana Cloud) | $200 | |
| **Total** | **~$50,700/month** | |

**Optimization Strategies**:
1. **Spot Instances**: -70% cost (use for non-critical replicas)
2. **Model Quantization**: Fewer GPUs needed
3. **Aggressive Caching**: -30% inference requests
4. **Regional Deployment**: Cheaper egress

**Optimized Cost**: ~$30,000/month

---

## Interview Discussion Points

**Q: How to handle spiky traffic?**
- **Auto-scaling**: HPA scales pods based on queue depth
- **Burst capacity**: Overprovision by 20% for sudden spikes
- **Queue throttling**: Return 429 (Too Many Requests) if overloaded
- **CDN caching**: Cache common predictions at edge

**Q: What if a model becomes slow?**
- **Circuit breaker**: Fail fast after 3 consecutive timeouts
- **Fallback model**: Route to older, faster version
- **Alert on tail latency**: p99 >100ms triggers investigation
- **Isolate bad model**: Remove from rotation automatically

**Q: How to ensure model quality in production?**
- **Shadow deployment**: Run new model without serving traffic
- **A/B testing**: Compare metrics (accuracy, engagement)
- **Data drift detection**: Monitor input distribution shifts
- **Feedback loop**: User corrections retrain model

---

## Conclusion

This design achieves:
- ✅ **10,000 QPS** with horizontal scaling
- ✅ **<50ms p95 latency** via GPU optimization
- ✅ **Multi-framework support** (TensorFlow, PyTorch, sklearn)
- ✅ **Zero-downtime updates** with canary deployments
- ✅ **Cost-effective** (~$30K/month optimized)

**Key Technologies**:
- NVIDIA Triton for serving
- Dynamic batching for throughput
- INT8 quantization for speedup
- Kubernetes for orchestration
- Prometheus for monitoring
