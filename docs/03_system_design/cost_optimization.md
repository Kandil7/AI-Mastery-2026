# Cloud Cost Optimization Guide

This guide covers strategies to minimize cloud costs while running AI-Mastery-2026 in production.

---

## Table of Contents

1. [Cost Overview](#cost-overview)
2. [GPU Optimization](#gpu-optimization)
3. [Storage Strategies](#storage-strategies)
4. [Compute Optimization](#compute-optimization)
5. [Model Quantization](#model-quantization)
6. [Auto-scaling](#auto-scaling)
7. [Monthly Cost Examples](#monthly-cost-examples)

---

## Cost Overview

### Major Cost Drivers

| Component | Cost Range | Optimization Potential |
|-----------|------------|------------------------|
| GPU Instances | $0.50-$30/hour | High (use spot, quantization) |
| Model Inference | $0.001-$0.10/request | High (batching, caching) |
| Storage | $0.02-$0.10/GB/month | Medium (compression, tiering) |
| Bandwidth | $0.05-$0.12/GB | Medium (edge caching) |
| Vector DB | $0.05-$1/hour | Medium (self-hosted vs managed) |

---

## GPU Optimization

### 1. Use Spot Instances (70-90% savings)

```bash
# AWS Spot Instance example
aws ec2 run-instances \
    --instance-type g4dn.xlarge \
    --instance-market-options MarketType=spot \
    --spot-options MaxPrice=0.50
```

**Spot Instance Prices (approx):**
| Instance | On-Demand | Spot | Savings |
|----------|-----------|------|---------|
| AWS g4dn.xlarge | $0.526/hr | $0.16/hr | 70% |
| AWS p3.2xlarge | $3.06/hr | $0.92/hr | 70% |
| GCP n1-standard-4+T4 | $0.95/hr | $0.29/hr | 69% |

### 2. Right-size GPU Selection

```python
# GPU selection guide
GPU_RECOMMENDATIONS = {
    'inference_small': 'T4',           # 16GB, $0.50/hr
    'inference_medium': 'A10G',        # 24GB, $1.00/hr
    'fine_tuning_lora': 'A10G',        # 24GB, sufficient for LoRA
    'fine_tuning_full': 'A100-40GB',   # 40GB, $3.00/hr
    'training_large': 'A100-80GB',     # 80GB, $5.00/hr
}

# Model to GPU mapping
MODEL_GPU_REQUIREMENTS = {
    'all-MiniLM-L6-v2': 'CPU or T4',      # 80MB, CPU sufficient
    'llama-2-7b': 'T4 16GB',              # 14GB (int8)
    'llama-2-13b': 'A10G 24GB',           # 26GB (int8)
    'llama-2-70b': 'A100 80GB or 2xA10G', # 140GB (int8)
}
```

### 3. GPU Scheduling

```python
# Only load GPU model when needed
class LazyGPUModel:
    def __init__(self, model_path):
        self.model_path = model_path
        self._model = None
        self._last_used = None
        self.timeout = 300  # 5 minutes
    
    @property
    def model(self):
        if self._model is None:
            self._model = load_model(self.model_path)
        self._last_used = time.time()
        return self._model
    
    def cleanup(self):
        """Unload model after timeout to free GPU memory"""
        if self._model and time.time() - self._last_used > self.timeout:
            del self._model
            torch.cuda.empty_cache()
            self._model = None
```

---

## Storage Strategies

### 1. Model Storage Tiers

```yaml
# S3 Lifecycle Policy
storage_tiers:
  hot:
    description: "Active models (< 7 days old)"
    storage_class: STANDARD
    cost: $0.023/GB/month
  
  warm:
    description: "Previous versions (7-30 days)"
    storage_class: STANDARD_IA
    cost: $0.0125/GB/month
  
  cold:
    description: "Archive (> 30 days)"
    storage_class: GLACIER
    cost: $0.004/GB/month
```

### 2. Compress Model Checkpoints

```python
import lzma
import pickle

def save_model_compressed(model, path):
    """Save model with LZMA compression (50-70% smaller)"""
    with lzma.open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model_compressed(path):
    with lzma.open(path, 'rb') as f:
        return pickle.load(f)
```

### 3. Vector DB Cost Comparison

| Solution | Cost | Use Case |
|----------|------|----------|
| FAISS (self-hosted) | $0.05/hr (compute only) | Full control |
| Pinecone | $0.096/hr + $0.025/1M queries | Managed, low-ops |
| Qdrant Cloud | $0.0825/hr | Balance |
| ChromaDB (free tier) | Free (100K vectors) | Development |

---

## Compute Optimization

### 1. Request Batching

```python
from collections import defaultdict
import asyncio

class BatchProcessor:
    def __init__(self, batch_size=32, max_wait_ms=100):
        self.batch_size = batch_size
        self.max_wait = max_wait_ms / 1000
        self.queue = asyncio.Queue()
        self.results = {}
    
    async def add_request(self, request_id, data):
        """Add request to batch queue"""
        future = asyncio.Future()
        await self.queue.put((request_id, data, future))
        return await future
    
    async def process_batch(self, model):
        """Process accumulated batch"""
        batch = []
        start_time = time.time()
        
        while len(batch) < self.batch_size:
            try:
                timeout = self.max_wait - (time.time() - start_time)
                item = await asyncio.wait_for(self.queue.get(), timeout=max(0, timeout))
                batch.append(item)
            except asyncio.TimeoutError:
                break
        
        if batch:
            # Process entire batch at once (much faster)
            inputs = [item[1] for item in batch]
            results = model.predict_batch(inputs)
            
            for (request_id, _, future), result in zip(batch, results):
                future.set_result(result)
```

### 2. Response Caching

```python
from functools import lru_cache
import hashlib

class PredictionCache:
    def __init__(self, maxsize=10000):
        self.cache = {}
        self.maxsize = maxsize
    
    def _hash_input(self, features):
        return hashlib.md5(str(features).encode()).hexdigest()
    
    def get_or_compute(self, features, compute_fn):
        key = self._hash_input(features)
        if key in self.cache:
            return self.cache[key]
        
        result = compute_fn(features)
        
        if len(self.cache) >= self.maxsize:
            # Remove oldest entry
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[key] = result
        return result
```

---

## Model Quantization

Reduce model size and inference cost by 4-8x.

### 1. INT8 Quantization (PyTorch)

```python
import torch

def quantize_model(model):
    """Quantize model to INT8 (4x smaller, 2x faster)"""
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},  # Layers to quantize
        dtype=torch.qint8
    )
    return quantized

# Memory comparison
# FP32: 1GB → INT8: 250MB
```

### 2. GGUF Quantization (LLMs)

```bash
# Using llama.cpp for LLM quantization
python convert.py model_path --outtype q4_k_m

# Quantization levels:
# q4_0: 4-bit (smallest, some quality loss)
# q4_k_m: 4-bit with k-means (good balance)
# q8_0: 8-bit (larger, minimal quality loss)
```

### 3. Cost Impact

| Model | Full Precision | Quantized (INT8) | Savings |
|-------|----------------|------------------|---------|
| BERT-base | 440MB | 110MB | 75% |
| LLaMA-7B | 14GB | 4GB | 71% |
| LLaMA-70B | 140GB | 35GB | 75% |

---

## Auto-scaling

### 1. Kubernetes HPA

```yaml
# kubernetes/hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ml-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ml-api
  minReplicas: 1
  maxReplicas: 10
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
        name: http_requests_per_second
      target:
        type: AverageValue
        averageValue: 100
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300  # Wait 5 min before scaling down
```

### 2. Time-based Scaling

```python
# Scale down during off-hours
SCALING_SCHEDULE = {
    'weekday_business': {'min': 3, 'max': 10, 'hours': '09:00-18:00'},
    'weekday_evening': {'min': 1, 'max': 5, 'hours': '18:00-09:00'},
    'weekend': {'min': 1, 'max': 3, 'hours': 'all'},
}

# Saves ~40% on compute costs
```

---

## Monthly Cost Examples

### Development Environment
| Component | Configuration | Monthly Cost |
|-----------|--------------|--------------|
| Compute | 1x t3.medium (spot) | $15 |
| Storage | 50GB EBS | $5 |
| Vector DB | ChromaDB (local) | $0 |
| **Total** | | **$20/month** |

### Small Production (< 1000 requests/day)
| Component | Configuration | Monthly Cost |
|-----------|--------------|--------------|
| API Instance | 1x t3.large | $60 |
| GPU Instance | g4dn.xlarge (8hrs/day, spot) | $40 |
| Database | RDS t3.small | $30 |
| Storage | 200GB + S3 | $15 |
| Monitoring | Prometheus/Grafana | $0 |
| **Total** | | **$145/month** |

### Medium Production (10K-100K requests/day)
| Component | Configuration | Monthly Cost |
|-----------|--------------|--------------|
| API Cluster | 3x c5.xlarge | $315 |
| GPU Inference | 2x g4dn.xlarge (spot) | $240 |
| Load Balancer | ALB | $20 |
| Database | RDS r5.large | $150 |
| Vector DB | Pinecone Starter | $70 |
| CDN | CloudFront | $50 |
| **Total** | | **$845/month** |

---

## Cost Monitoring

```python
# Track cost per prediction
COST_PER_GPU_SECOND = 0.0001  # Approximate

@app.post("/predict")
async def predict(request: PredictionRequest):
    start = time.time()
    
    result = model.predict(request.features)
    
    duration = time.time() - start
    cost = duration * COST_PER_GPU_SECOND
    
    # Log for cost tracking
    metrics.record_cost(
        endpoint="/predict",
        duration=duration,
        cost=cost
    )
    
    return result
```

---

## Key Takeaways

1. **Use spot instances** for training and non-critical inference (70% savings)
2. **Quantize models** to INT8/INT4 (4-8x cost reduction)
3. **Batch requests** to maximize GPU utilization
4. **Cache predictions** for repeated inputs
5. **Right-size GPUs** - don't use A100 when T4 is sufficient
6. **Auto-scale** based on demand, scale to zero when idle
7. **Use storage tiering** for model checkpoints
