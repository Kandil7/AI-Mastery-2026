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

---

## 🔬 Advanced Integration Methods

### Neural ODEs

**Q: What is the adjoint method in Neural ODEs, and why is it important?**

The adjoint method computes gradients in O(1) memory regardless of integration steps:
- Instead of storing all intermediate states (backprop through solver), we solve an augmented ODE backward in time
- This makes training practical for long time horizons
- Trade-off: More compute (solve ODE twice) for less memory

**Q: How do Neural ODEs handle uncertainty quantification?**

Two main approaches:
1. **MC Dropout**: Keep dropout active during inference, run multiple passes
2. **Input perturbation**: Add small noise to initial state, measure trajectory variance

Both capture epistemic uncertainty—what the model doesn't know.

---

### Federated Learning

**Q: What are the main aggregation strategies in Federated Learning?**

| Strategy | Formula | Best When |
|----------|---------|-----------|
| FedAvg | Σ(n_k/n) × w_k | IID data across clients |
| Inverse variance | Σ(1/σ²) × w_k | Known uncertainties |
| Bayesian | Σ(n_k/σ²) × w_k | Non-IID with uncertainty |

**Q: How does differential privacy work in federated settings?**

- Add calibrated noise to gradients before aggregation
- Noise scale: σ ∝ sensitivity / ε (privacy budget)
- Trade-off: More privacy → more noise → lower accuracy
- Apple uses ε ≈ 8 for HealthKit analytics

---

### AI Ethics & Fairness

**Q: What is disparate impact, and how do you measure it?**

Disparate impact occurs when a neutral policy has unequal effects on protected groups:
```
Disparate Impact Ratio = (Approval Rate_minority) / (Approval Rate_majority)
```
- Ratio < 0.8 triggers legal scrutiny (80% rule)
- Also measure: FPR parity, FNR parity, calibration

**Q: Name three bias mitigation strategies.**

1. **Pre-processing**: Resampling, reweighting, representation learning
2. **In-processing**: Fairness constraints, adversarial debiasing
3. **Post-processing**: Threshold adjustment, calibration

**Q: What's the Impossibility Theorem in fairness?**

You cannot simultaneously satisfy:
- Calibration (P(Y=1|score=s) same across groups)
- FPR parity
- FNR parity

...unless base rates are equal or predictor is perfect. Must choose trade-offs.

---

### Integration Interview Coding Challenge

**Q: Implement Bayesian weighted aggregation for federated estimates.**

```python
def bayesian_aggregate(estimates):
    """
    estimates: list of {'risk': float, 'uncertainty': float, 'n': int}
    """
    weights = [e['n'] / (e['uncertainty']**2 + 1e-8) for e in estimates]
    total_weight = sum(weights)
    
    global_risk = sum(w * e['risk'] for w, e in zip(weights, estimates)) / total_weight
    global_unc = 1 / np.sqrt(sum(e['n'] / (e['uncertainty']**2 + 1e-8) for e in estimates))
    
    return global_risk, global_unc
```

---

## 🚀 Hardware Acceleration & Adaptive Integration

### GPU Acceleration

**Q: When does GPU acceleration provide the most benefit for Monte Carlo integration?**

**A:** GPU acceleration excels when:
1. **Large sample sizes** (>50,000) - Parallelism overcomes kernel launch overhead
2. **Complex function evaluations** - More compute per sample amortizes memory transfer
3. **Batch processing** - Multiple integrals computed simultaneously

The **break-even point** is typically around 50,000 samples; below this, Numba/CPU may be faster.

**Q: Explain the trade-off between Numba and GPU acceleration.**

| Aspect | Numba | GPU |
|--------|-------|-----|
| Startup | Fast | Slow (kernel compilation) |
| Best for | Medium problems (10K-1M) | Large problems (>1M) |
| Memory | CPU RAM | GPU VRAM (limited) |
| Flexibility | Any Python code | Needs framework (PyTorch/TF) |

---

### PPL Comparison

**Q: Compare PyMC3, TensorFlow Probability, and Stan for Bayesian inference.**

| Library | Best For |
|---------|----------|
| **PyMC3** | Rapid prototyping of complex hierarchical models |
| **TFP** | Production systems integrated with deep learning |
| **Stan** | Rigorous statistical research, maximum accuracy |

**Q: What is the ELBO and why is it important for variational inference?**

Evidence Lower BOund:
$$\text{ELBO} = \mathbb{E}_{q(z)}[\log p(x,z) - \log q(z)]$$

Key properties:
1. Maximizing ELBO ≈ minimizing KL divergence to true posterior
2. Tractable when posterior is intractable
3. Enables gradient-based optimization (vs. sampling)

---

### Adaptive Integration

**Q: How would you design an adaptive integrator that selects methods automatically?**

Key components:
1. **Feature extraction**: Analyze function smoothness, modes, sharp transitions
2. **Method library**: Gaussian quadrature, Monte Carlo, Bayesian quadrature, Simpson
3. **Selection model**: Random Forest classifier trained on function-method pairs
4. **Fallback strategy**: If selected method fails, try alternatives in order

Critical features:
- **Smoothness** = 1 / mean(|gradient|)
- **Modality** = number of peaks
- **Sharp transitions** = proportion of extreme gradients

**Q: Implement a simple function feature extractor.**

```python
def extract_features(f, a, b, n_samples=1000):
    """Extract features for method selection."""
    x = np.linspace(a, b, n_samples)
    y = np.array([f(xi) for xi in x])
    gradients = np.gradient(y, x)
    
    return {
        'smoothness': 1.0 / (np.mean(np.abs(gradients)) + 1e-8),
        'num_modes': len(find_peaks(y / y.max())[0]),
        'variance': np.var(y),
        'sharp_transitions': np.sum(np.abs(gradients) > np.percentile(np.abs(gradients), 95)) / n_samples
    }
```

---

## 🎮 Reinforcement Learning Integration

### Policy Gradient

**Q: Explain how Monte Carlo integration is used in REINFORCE.**

**A**: REINFORCE estimates the policy gradient via Monte Carlo sampling:

$$\nabla J(\theta) = \mathbb{E}\left[\sum_t \nabla \log \pi_\theta(a_t|s_t) \cdot G_t\right]$$

We sample trajectories and average gradient estimates - this is Monte Carlo integration over the trajectory distribution.

**Q: What is the exploration-exploitation tradeoff in MCTS?**

MCTS uses UCB to balance:
```
Q(s,a) + c · P(s,a) · √(N(s)) / (1 + N(s,a))
```
- **Exploitation**: Q(s,a) favors high-value actions
- **Exploration**: Second term favors rarely-visited actions

---

## 🔬 Causal Inference

### Treatment Effect Estimation

**Q: Why is Doubly Robust estimation preferred over IPW?**

Doubly Robust is consistent if EITHER propensity OR outcome model is correct:

```python
tau_dr = mu1(X) - mu0(X) + T*(Y-mu1(X))/e(X) - (1-T)*(Y-mu0(X))/(1-e(X))
```

This "double protection" makes it robust to model misspecification.

**Q: Implement propensity score trimming.**

```python
def trim_propensity_scores(ps, min_val=0.05, max_val=0.95):
    """Trim extreme PS to reduce variance in IPW."""
    return np.clip(ps, min_val, max_val)
```

**Q: What is the fundamental problem of causal inference?**

We can never observe both Y(1) AND Y(0) for the same individual. This is a missing data problem - we use statistical assumptions (ignorability, overlap) to estimate effects.

