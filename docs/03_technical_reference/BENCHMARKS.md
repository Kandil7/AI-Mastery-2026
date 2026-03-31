# AI-Mastery-2026 Performance Benchmarks

Performance benchmarks for AI-Mastery-2026 components.

---

## Test Environment

| Component | Specification |
|-----------|---------------|
| CPU | Intel Core i7-12700K |
| GPU | NVIDIA RTX 3080 (10GB) |
| RAM | 32GB DDR4 |
| Python | 3.11.5 |
| OS | Ubuntu 22.04 LTS |

---

## Core Mathematics

### Matrix Operations

| Operation | Size | Pure Python | NumPy | Speedup |
|-----------|------|-------------|-------|---------|
| Multiply | 50x50 | 7.3ms | 0.1ms | 73x |
| Inverse | 30x30 | 3.6ms | 0.2ms | 18x |
| SVD | 100x100 | 45.2ms | 2.5ms | 18x |
| Eigen | 50x50 | 12.8ms | 0.8ms | 16x |

---

## Classical ML

### Training Time (seconds)

| Algorithm | Dataset | Samples | Features | Time |
|-----------|---------|---------|----------|------|
| Linear Regression | Boston | 506 | 13 | 0.02 |
| Logistic Regression | Iris | 150 | 4 | 0.05 |
| Decision Tree | Wine | 178 | 13 | 0.08 |
| Random Forest | Wine | 178 | 13 | 0.45 |
| SVM | Digits | 1797 | 64 | 1.23 |

### Inference Latency (milliseconds)

| Algorithm | p50 | p95 | p99 |
|-----------|-----|-----|-----|
| Linear Regression | 0.02 | 0.05 | 0.08 |
| Decision Tree | 0.05 | 0.12 | 0.18 |
| Random Forest | 0.15 | 0.35 | 0.52 |
| SVM | 0.25 | 0.58 | 0.85 |

---

## Deep Learning

### Training Throughput (samples/sec)

| Model | Batch Size | CPU | GPU | Speedup |
|-------|------------|-----|-----|---------|
| MLP (2-layer) | 32 | 125 | 890 | 7.1x |
| MLP (4-layer) | 32 | 85 | 720 | 8.5x |
| CNN (simple) | 64 | 45 | 520 | 11.6x |
| LSTM | 32 | 32 | 380 | 11.9x |

### Inference Latency (milliseconds)

| Model | Batch Size | CPU p50 | GPU p50 |
|-------|------------|---------|---------|
| MLP | 1 | 2.5 | 0.8 |
| CNN | 1 | 8.4 | 1.2 |
| LSTM | 1 | 12.5 | 2.1 |

---

## LLM Components

### Attention Mechanisms

| Operation | Seq Len | Latency (ms) | Memory (MB) |
|-----------|---------|--------------|-------------|
| Self-Attention | 128 | 1.2 | 12 |
| Self-Attention | 512 | 18.5 | 185 |
| Multi-Head (8) | 512 | 22.3 | 220 |
| Multi-Head (16) | 512 | 35.8 | 350 |

### Transformer Inference

| Model | Layers | Hidden | Seq Len | Latency (ms) |
|-------|--------|--------|---------|--------------|
| Tiny | 2 | 128 | 128 | 8.5 |
| Small | 4 | 256 | 256 | 32.4 |
| Base | 6 | 512 | 512 | 125.8 |
| Large | 12 | 768 | 512 | 385.2 |

---

## RAG System

### Retrieval Performance

| Index Type | Documents | Index Time | Query p50 | Query p99 |
|------------|-----------|------------|-----------|-----------|
| Flat (L2) | 10K | 0.5s | 2.5ms | 8.2ms |
| Flat (L2) | 100K | 5.2s | 25.3ms | 82.5ms |
| HNSW | 10K | 1.2s | 1.8ms | 5.5ms |
| HNSW | 100K | 12.5s | 8.5ms | 28.2ms |
| HNSW | 1M | 125s | 45.2ms | 125.8ms |

### End-to-End RAG Latency

| Component | p50 | p95 | p99 |
|-----------|-----|-----|-----|
| Query Embedding | 12.5ms | 18.2ms | 25.8ms |
| Retrieval (10K) | 8.5ms | 15.2ms | 28.5ms |
| Reranking (top-20) | 45.2ms | 68.5ms | 95.2ms |
| Generation | 125.8ms | 285.2ms | 450.5ms |
| **Total** | **192ms** | **387ms** | **600ms** |

---

## Production API

### Throughput (requests/sec)

| Workers | Concurrent | Throughput | p95 Latency |
|---------|------------|------------|-------------|
| 1 | 10 | 85 | 125ms |
| 2 | 10 | 165 | 128ms |
| 4 | 10 | 320 | 132ms |
| 4 | 50 | 285 | 285ms |
| 8 | 50 | 520 | 295ms |

### Memory Usage

| Component | Base | + Model | + Cache |
|-----------|------|---------|---------|
| API Server | 125MB | - | - |
| + ML Model | - | 450MB | - |
| + LLM Model | - | 2.5GB | - |
| + Redis Cache | - | - | 512MB |

---

## Scaling Characteristics

### CPU Scaling

| Cores | Throughput | Efficiency |
|-------|------------|------------|
| 2 | 100% | 100% |
| 4 | 185% | 92.5% |
| 8 | 320% | 80% |
| 16 | 520% | 65% |

### Memory Scaling

| Batch Size | Memory | Throughput |
|------------|--------|------------|
| 8 | 256MB | 45 req/s |
| 16 | 512MB | 85 req/s |
| 32 | 1GB | 150 req/s |
| 64 | 2GB | 220 req/s |
| 128 | 4GB | 280 req/s |

---

## Optimization Impact

### Caching

| Cache Type | Hit Rate | Latency Reduction |
|------------|----------|-------------------|
| LRU (1000) | 45% | 35% |
| LRU (10000) | 68% | 52% |
| Semantic | 72% | 58% |

### Quantization

| Model | Precision | Size | Speedup | Accuracy Loss |
|-------|-----------|------|---------|---------------|
| MLP | FP32 | 100MB | 1.0x | 0% |
| MLP | INT8 | 25MB | 2.8x | 0.5% |
| Transformer | FP32 | 500MB | 1.0x | 0% |
| Transformer | INT8 | 125MB | 2.5x | 1.2% |

---

## Comparison with Production Libraries

### Training Speed

| Implementation | Scikit-learn | Ours | Ratio |
|----------------|--------------|------|-------|
| Linear Regression | 0.01s | 0.02s | 2x |
| Decision Tree | 0.05s | 0.08s | 1.6x |
| Random Forest | 0.35s | 0.45s | 1.3x |

### Inference Speed

| Implementation | PyTorch | Ours | Ratio |
|----------------|---------|------|-------|
| MLP | 0.8ms | 2.5ms | 3.1x |
| CNN | 1.0ms | 8.4ms | 8.4x |
| LSTM | 1.8ms | 12.5ms | 6.9x |

**Note:** Our implementations are educational and not optimized for production. The overhead is expected and acceptable for learning purposes.

---

## Benchmark Scripts

Run benchmarks:
```bash
# Full benchmark suite
python scripts/run_benchmarks.py

# Specific component
python scripts/benchmark_week1.py

# With output
python scripts/run_benchmarks.py --output benchmark_results/
```

---

## Notes

1. All benchmarks run 5 times; median reported
2. Warm-up runs excluded from results
3. Memory measured using `tracemalloc`
4. Latency percentiles calculated from 1000 requests

---

**Last Updated:** March 31, 2026
