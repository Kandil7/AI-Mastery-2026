# System Design Question: LLM Infrastructure

## Problem Statement

Design the infrastructure to serve a large language model (LLM) at scale for a conversational AI product with 10M daily active users.

---

## Requirements

### Functional Requirements
1. Serve LLM inference with streaming responses
2. Support multiple model versions and sizes
3. Enable prompt caching and context management
4. Provide usage metering for billing
5. Support fine-tuned models per customer

### Non-Functional Requirements
1. P50 time-to-first-token < 200ms
2. Throughput: 100K requests/minute peak
3. 99.9% availability
4. Cost optimization (GPU utilization > 70%)
5. Context window up to 32K tokens

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     API Gateway / CDN                            │
│  Rate limiting, Auth, Request validation, Streaming support      │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                    Request Router / Load Balancer                │
│  Model routing, Token estimation, Queue management               │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         │                        │                        │
         ▼                        ▼                        ▼
┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│   Small Model    │   │  Medium Model    │   │   Large Model    │
│   Cluster        │   │  Cluster         │   │   Cluster        │
│   (7B params)    │   │  (13B params)    │   │   (70B params)   │
└──────────────────┘   └──────────────────┘   └──────────────────┘
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  │
┌─────────────────────────────────▼───────────────────────────────┐
│                     Shared Services                              │
│  • KV Cache Service (Redis)                                      │
│  • Prompt Cache                                                  │
│  • Usage Metering                                                │
│  • Model Registry                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Inference Stack

### Option 1: vLLM (Recommended)

**Advantages**:
- PagedAttention for efficient KV cache management
- Continuous batching for high throughput
- OpenAI-compatible API

**Configuration**:
```yaml
model: llama-2-70b
tensor_parallel_size: 8  # 8x A100 GPUs
max_num_batched_tokens: 8192
gpu_memory_utilization: 0.9
```

### Option 2: TensorRT-LLM

**Advantages**:
- Optimized NVIDIA kernels
- Lower latency (10-20% faster)
- INT8/FP8 quantization

**When to use**: Latency-critical applications

### Option 3: Text Generation Inference (TGI)

**Advantages**:
- Flash Attention 2
- Speculative decoding
- Watermarking support

---

## Key Optimizations

### 1. KV Cache Management

**Problem**: KV cache grows linearly with context length

**Solutions**:
```
┌─────────────────────────────────────────┐
│           PagedAttention                 │
│  • Allocate KV cache in fixed pages     │
│  • Share pages across similar prompts   │
│  • Reclaim unused pages immediately     │
└─────────────────────────────────────────┘
```

Memory per request = `num_layers × 2 × hidden_dim × seq_len × dtype_size`

For 70B model with 32K context: ~32GB per request!

### 2. Continuous Batching

**Traditional batching**: Wait for batch to fill → high latency

**Continuous batching**: 
- Add new requests to running batch
- Remove completed requests immediately
- 2-3x throughput improvement

### 3. Speculative Decoding

**Concept**: Use small "draft" model to predict multiple tokens, verify with large model

```
Draft model (7B) generates: "The quick brown fox"
Large model (70B) verifies in parallel → accepts 3/4 tokens
Net: 3 tokens in 1 forward pass
```

**Speedup**: 2-3x for greedy decoding

### 4. Prefix Caching

**Scenario**: System prompts repeated across requests

**Solution**:
- Cache KV values for common prefixes
- Hash prefix → retrieve cached KV
- Reduces TTFT by 30-50%

---

## Scaling Strategy

### GPU Cluster Design

| Model Size | GPU Config | Tensor Parallel | Instances |
|------------|------------|-----------------|-----------|
| 7B | 1x A100-40G | 1 | 50 |
| 13B | 2x A100-40G | 2 | 30 |
| 70B | 8x A100-80G | 8 | 20 |

### Auto-Scaling

```python
def calculate_replicas(queue_depth, avg_latency, target_latency):
    if avg_latency > target_latency * 1.5:
        return current_replicas * 1.5  # Scale up fast
    elif queue_depth < 10 and avg_latency < target_latency * 0.5:
        return current_replicas * 0.8  # Scale down slow
    return current_replicas
```

**Signals**:
- Request queue depth
- GPU memory utilization
- Inference latency P99
- Token throughput

---

## Cost Optimization

### 1. Model Routing

Route to smallest capable model:
```python
def route_request(prompt, requirements):
    if len(prompt) < 1000 and not requirements.complex_reasoning:
        return "model-7b"
    elif requirements.coding or requirements.long_context:
        return "model-70b"
    else:
        return "model-13b"
```

### 2. Quantization

| Precision | Memory | Speed | Quality |
|-----------|--------|-------|---------|
| FP16 | 1x | 1x | Baseline |
| INT8 | 0.5x | 1.3x | -0.5% |
| INT4 | 0.25x | 1.5x | -2% |

### 3. Spot/Preemptible Instances

- Use for batch inference workloads
- Stateless design for quick restart
- 60-70% cost savings

---

## Reliability

### Failure Modes

1. **GPU failure**: Hot standby, health checks
2. **OOM**: Request rejection, graceful degradation
3. **Model loading failure**: Rollback, health validation

### Disaster Recovery

- Multi-region deployment
- Model artifacts in S3 (replicated)
- Stateless inference servers

---

## Monitoring

### Metrics

```python
metrics = {
    # Latency
    "ttft_p50": "Time to first token (ms)",
    "ttft_p99": "Time to first token P99",
    "tpot": "Time per output token",
    
    # Throughput  
    "tokens_per_second": "Output tokens/sec",
    "requests_per_second": "Requests/sec",
    
    # Efficiency
    "gpu_utilization": "GPU compute %",
    "kv_cache_hit_rate": "Prefix cache hits",
    "batch_size_avg": "Avg concurrent requests",
    
    # Quality
    "generation_length_avg": "Avg output length",
    "error_rate": "Failed requests %",
}
```

---

## Trade-offs Discussion

| Question | Consideration |
|----------|---------------|
| vLLM vs TGI? | vLLM for throughput, TGI for features |
| Self-host vs API? | Self-host for data privacy, API for simplicity |
| Single large vs multiple small? | Large for quality, small for cost/latency |
| Tensor vs Pipeline parallel? | Tensor for latency, Pipeline for memory |

---

## Sample Interview Questions

1. **How do you handle long-running requests?**
   - Streaming, checkpointing, timeout policies

2. **How to ensure fairness across customers?**
   - Per-tenant rate limits, priority queues

3. **How to A/B test a new model?**
   - Shadow traffic, gradual rollout, quality metrics

4. **What happens when GPU memory is exhausted?**
   - Request queuing, rejection, graceful degradation
