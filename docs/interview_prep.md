# AI Engineer Interview Preparation Guide

A comprehensive guide covering technical interviews for AI/ML Engineer roles, based on the AI-Mastery-2026 curriculum.

---

## 📊 Machine Learning Fundamentals

### Bias-Variance Tradeoff

**Q: Explain the bias-variance tradeoff.**

**A:** The total error of a model can be decomposed into:
- **Bias**: Error from overly simplistic assumptions → underfitting
- **Variance**: Error from sensitivity to training data → overfitting
- **Irreducible error**: Noise inherent in the problem

```
Total Error = Bias² + Variance + Irreducible Error
```

High bias → model too simple (e.g., linear regression for non-linear data)
High variance → model too complex (e.g., deep network on small dataset)

**Follow-up: How do you reduce each?**
- Reduce bias: More complex model, more features, less regularization
- Reduce variance: More data, regularization, ensemble methods, dropout

---

### Regularization

**Q: What's the difference between L1 and L2 regularization?**

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|------------|------------|
| Penalty | Σ\|w\| | Σw² |
| Effect | Sparse weights (feature selection) | Small weights |
| Geometry | Diamond constraint | Circular constraint |
| Use case | Feature selection | All features relevant |

**Q: When does L1 produce exactly zero weights?**
The diamond shape of L1 constraint has corners on axes. The optimal point often lands exactly on a corner, zeroing some weights.

---

### Optimization

**Q: Why does Adam tend to generalize worse than SGD in some cases?**

Adam adapts learning rates per-parameter, which can lead to:
1. Sharp minima that generalize poorly
2. Faster convergence to "convenient" but suboptimal solutions

SGD with momentum often finds flatter minima that generalize better. This is why many SOTA vision models still use SGD.

**Q: What is the purpose of bias correction in Adam?**

The first and second moment estimates (m, v) are initialized at 0. Early in training, these estimates are biased toward 0. Bias correction divides by (1 - β^t) to compensate, especially important when β is close to 1.

---

## 🧠 Deep Learning

### Backpropagation

**Q: Derive the gradient for a softmax + cross-entropy loss.**

For softmax: `p_i = exp(z_i) / Σ exp(z_j)`
Cross-entropy: `L = -Σ y_i log(p_i)`

The gradient simplifies elegantly:
```
∂L/∂z_i = p_i - y_i
```

This is why softmax + cross-entropy is so common—the gradient is simple!

---

### Attention Mechanism

**Q: What is the computational complexity of self-attention?**

For sequence length N and dimension D:
- Attention matrix computation: O(N² × D)
- Memory: O(N²)

This quadratic scaling is why Transformers struggle with long sequences. Solutions include:
- Sparse attention (Longformer): O(N × k)
- Linear attention (Performer): O(N × D)
- Sliding window + global tokens

**Q: Why do we scale by √d_k in attention?**

Without scaling, for large d_k, the dot products grow large, pushing softmax into regions with extremely small gradients (saturation). Scaling by √d_k keeps variance constant.

---

### Transformers

**Q: What happens if we remove positional encodings?**

The model becomes permutation-invariant—it treats "The dog bit the man" the same as "The man bit the dog." Position encodings inject ordering information.

**Q: Why layer normalization over batch normalization in Transformers?**

1. LayerNorm normalizes across features (per-sample), BatchNorm across batch (per-feature)
2. BatchNorm requires batch statistics during inference (problematic for variable-length sequences)
3. LayerNorm is independent of batch size, crucial for language models

---

## 🔤 LLM Engineering

### RAG (Retrieval Augmented Generation)

**Q: How would you improve RAG retrieval quality?**

1. **Hybrid retrieval**: Combine dense (semantic) + sparse (keyword) with reciprocal rank fusion
2. **Reranking**: Use cross-encoder after initial retrieval
3. **Query expansion**: Generate multiple query variations
4. **Chunking strategy**: Overlap chunks, semantic chunking vs. fixed-size
5. **Metadata filtering**: Pre-filter by date, source, category

**Q: What's the trade-off between chunk size in RAG?**

| Small chunks | Large chunks |
|-------------|--------------|
| More precise retrieval | More context per chunk |
| May miss cross-chunk info | May dilute relevance |
| More chunks to search | Fewer chunks |

---

### Fine-tuning

**Q: Explain LoRA and why it's efficient.**

LoRA (Low-Rank Adaptation) adds trainable low-rank matrices to attention weights:
```
W' = W + BA  where B∈ℝ^{d×r}, A∈ℝ^{r×k}, r << min(d,k)
```

Benefits:
- Original weights frozen (no forgetting)
- Only r×(d+k) parameters vs. d×k
- Rank r=8-32 often sufficient
- Easy to swap adapters for different tasks

---

## 🏗️ System Design

### ML System Design Template

1. **Clarify requirements**: Latency, throughput, accuracy, freshness
2. **Data**: Sources, volume, labeling, preprocessing
3. **Model**: Architecture, training, serving
4. **Infrastructure**: Batch vs. streaming, compute, storage
5. **Monitoring**: Metrics, drift detection, alerts

---

### Design: Real-time Recommendation System

**Context**: Spotify-like music recommendations

**Requirements**:
- <100ms latency for real-time recommendations
- Handle 100M+ users, 50M+ songs
- Incorporate user history, context (time, location), catalog

**Architecture**:

```
User Request → API Gateway
                   ↓
         Feature Store (user embeddings, recent history)
                   ↓
         Candidate Generation (ANN search on song embeddings)
                   ↓
         Ranking Model (cross features between user + candidates)
                   ↓
         Business Rules (diversity, freshness, licensing)
                   ↓
         Response
```

**Key Components**:
- **Embeddings**: Two-tower model trained on implicit feedback
- **ANN Index**: HNSW for <10ms retrieval of top-1000 candidates
- **Ranker**: Light GBM or small neural net on candidate features
- **Serving**: Model distillation for latency

---

### Design: Fraud Detection System

**Requirements**:
- Process 10,000 transactions/second
- <50ms latency decision
- 99.9% availability

**Architecture**:

```
Transaction → Stream Processing (Kafka)
                    ↓
         Feature Engineering (Flink)
         - Aggregations (last 1h, 24h, 7d)
         - Velocity features
         - Device fingerprinting
                    ↓
         Rule Engine (fast, interpretable rules)
                    ↓
         ML Model (gradient boosting)
                    ↓
         Decision + Logging
```

**Monitoring**:
- Precision/Recall on labeled chargebacks
- Distribution drift on input features
- Model confidence calibration

---

## 💻 Coding Interview Patterns

### Complexity Cheat Sheet

| Operation | Array | Hash | BST | Heap |
|-----------|-------|------|-----|------|
| Search | O(n) | O(1) | O(log n) | O(n) |
| Insert | O(n) | O(1) | O(log n) | O(log n) |
| Delete | O(n) | O(1) | O(log n) | O(log n) |
| Min/Max | O(n) | O(n) | O(log n) | O(1) |

### Common ML Coding Questions

1. **Implement gradient descent from scratch**
2. **K-means clustering**
3. **Binary cross-entropy loss with numerical stability**
4. **Softmax with temperature**
5. **Cosine similarity for batch of vectors**

---

## 🎯 Behavioral Questions

### STAR Method

**S**ituation → **T**ask → **A**ction → **R**esult

### Common Questions

**Q: Tell me about a time you dealt with ambiguous requirements.**

*Example structure*:
- S: New recommendation system, vague success metrics
- T: Define metrics and build system
- A: Stakeholder interviews, proposed A/B framework, iterative approach
- R: 15% engagement lift, adopted metrics framework

**Q: How do you handle disagreement with a coworker?**

Focus on:
- Data-driven discussions
- Seeking to understand their perspective
- Finding common ground
- Escalating appropriately if needed

---

## 📚 Quick Reference

### Key Formulas

| Concept | Formula |
|---------|---------|
| Cross-Entropy | -Σ y log(p) |
| KL Divergence | Σ p log(p/q) |
| Attention | softmax(QK^T/√d)V |
| Adam update | m/(√v + ε) with bias correction |
| LoRA | W + BA where r << d |

### Performance Benchmarks to Know

| Metric | Good Target |
|--------|-------------|
| API latency (p95) | <100ms |
| Model serving | <50ms |
| Embedding search (1M vectors) | <10ms |
| RAG retrieval | <500ms end-to-end |
