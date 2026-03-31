# Optimization Techniques - Interview Questions

A comprehensive collection of optimization and training questions for AI/ML interviews.

---

## Table of Contents
1. [Gradient Descent Variants](#1-gradient-descent-variants)
2. [Learning Rate Strategies](#2-learning-rate-strategies)
3. [Advanced Optimizers](#3-advanced-optimizers)
4. [Training Challenges](#4-training-challenges)
5. [Hyperparameter Tuning](#5-hyperparameter-tuning)

---

## 1. Gradient Descent Variants

### Q1.1: Explain the gradient descent algorithm and its variants.

**Answer:**
Gradient descent minimizes a loss function by moving in the direction of steepest descent:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$$

**Variants:**

| Variant | Batch Size | Update Frequency | Characteristics |
|---------|------------|------------------|-----------------|
| **Batch GD** | Full dataset | Once per epoch | Stable but slow |
| **Stochastic GD** | 1 sample | Every sample | Noisy but fast |
| **Mini-batch GD** | 32-256 | Every batch | Best of both |

**Trade-offs:**
- Larger batch → More stable gradients, better GPU utilization
- Smaller batch → More noise → Can escape local minima, better generalization

---

### Q1.2: What is momentum and why does it help?

**Answer:**
Momentum accelerates convergence by accumulating velocity in consistent gradient directions.

**Update Rule:**
```
v_t = β * v_{t-1} + ∇L(θ)     # Accumulate velocity
θ = θ - η * v_t               # Update parameters
```

**Benefits:**
1. **Accelerates through flat regions** - Builds up speed
2. **Dampens oscillations** - Cancels opposing gradients
3. **Escapes shallow local minima** - Momentum carries past

**Typical value:** β = 0.9

**Visualization:**
```
Without momentum:   ←→←→←→→  (oscillates)
With momentum:      →→→→→→   (smooth path)
```

---

### Q1.3: What is Nesterov Momentum?

**Answer:**
Nesterov Accelerated Gradient (NAG) "looks ahead" before computing the gradient.

**Standard Momentum:**
```
v = β*v + ∇L(θ)
θ = θ - η*v
```

**Nesterov Momentum:**
```
v = β*v + ∇L(θ - β*v)    # Gradient at "lookahead" position
θ = θ - η*v
```

**Why It's Better:**
- Anticipates where you're going
- More responsive to changes in gradient
- Converges faster in practice

---

## 2. Learning Rate Strategies

### Q2.1: How do you find a good learning rate?

**Answer:**

**Methods:**

1. **Learning Rate Range Test:**
   - Start with tiny LR (1e-7)
   - Increase exponentially while training
   - Plot loss vs. LR
   - Choose LR where loss decreases fastest

2. **Grid/Random Search:**
   - Try values like [0.1, 0.01, 0.001, 0.0001]
   - Log-scale is important

3. **Adaptive Methods:**
   - Use Adam (self-adjusting LR per parameter)

**Rules of Thumb:**
- SGD: Start with 0.1, divide by 10 when stuck
- Adam: Start with 3e-4 or 1e-3
- Fine-tuning: Use 10-100x smaller than pretraining

---

### Q2.2: Explain common learning rate schedules.

**Answer:**

| Schedule | Description | Formula/Example |
|----------|-------------|-----------------|
| **Step Decay** | Reduce by factor at fixed epochs | LR × 0.1 at epochs 30, 60 |
| **Exponential** | Continuous decay | LR × decay^epoch |
| **Cosine Annealing** | Smooth decrease | LR × (1 + cos(πt/T))/2 |
| **Warmup** | Start low, increase | Linear increase for N steps |
| **ReduceOnPlateau** | Reduce when metric stagnates | Divide by 10 if no improvement |
| **One-Cycle** | Increase then decrease | Triangle or cosine shape |

**Best Practices:**
- Use warmup for transformers (1-5% of training)
- Cosine schedule often works well
- Monitor validation loss to detect when to reduce

---

### Q2.3: What is warmup and why is it important for transformers?

**Answer:**
**Warmup** gradually increases learning rate from near-zero to target value.

**Why Needed for Transformers:**
1. **Adam's adaptive LR needs time** - Running averages are inaccurate initially
2. **Large gradients early** - Randomly initialized attention can have extreme gradients
3. **Prevents divergence** - High LR at start causes instability

**Implementation:**
```python
if step < warmup_steps:
    lr = base_lr * step / warmup_steps
else:
    lr = base_lr * decay_schedule(step)
```

**Typical Values:**
- Warmup for 1-10% of total training steps
- BERT often uses 10,000 warmup steps

---

## 3. Advanced Optimizers

### Q3.1: Explain Adam optimizer in detail.

**Answer:**
Adam = Adaptive Moment Estimation. Combines momentum with adaptive learning rates.

**Algorithm:**
```
# Compute gradients
g_t = ∇L(θ)

# Update biased first moment (momentum)
m_t = β₁ * m_{t-1} + (1 - β₁) * g_t

# Update biased second moment (RMSprop-like)
v_t = β₂ * v_{t-1} + (1 - β₂) * g_t²

# Bias correction (important early in training)
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)

# Update parameters
θ = θ - η * m̂_t / (√v̂_t + ε)
```

**Default Hyperparameters:**
- η = 0.001 (learning rate)
- β₁ = 0.9 (momentum)
- β₂ = 0.999 (RMSprop)
- ε = 1e-8 (numerical stability)

**Why Bias Correction?**
- m and v are initialized to 0
- Early estimates are biased toward 0
- Correction scales up early values

---

### Q3.2: What is AdamW and why is it better than Adam?

**Answer:**
AdamW applies weight decay correctly (decoupled from gradient).

**Problem with L2 in Adam:**
```
# L2 regularization adds to gradient
g = ∇L(θ) + λθ

# Adam then scales this by adaptive learning rate
# This couples regularization with learning rate!
```

**AdamW Solution:**
```
# Apply Adam to loss gradient only
θ_adam = θ - η * m̂ / (√v̂ + ε)

# Apply weight decay separately
θ = θ_adam - η * λ * θ
```

**Result:**
- More consistent regularization
- Better generalization
- Preferred for transformers and modern models

---

### Q3.3: When should you use SGD vs Adam?

**Answer:**

| Aspect | SGD+Momentum | Adam |
|--------|--------------|------|
| **Convergence speed** | Slower | Faster |
| **Final performance** | Often better | Good |
| **Hyperparameter sensitivity** | Higher | Lower |
| **Memory usage** | Lower | Higher (stores m, v) |

**When to Use What:**

**Use SGD+Momentum:**
- When training for maximum performance (final few %)
- CNNs on image classification
- When you have time to tune LR schedule
- ResNet, VGG style networks

**Use Adam/AdamW:**
- Quick experimentation
- Transformers and NLP models
- Reinforcement learning
- When schedule tuning is difficult
- Default choice for most cases

**Hybrid Approach:**
- Start with Adam for fast initial training
- Switch to SGD for final refinement

---

## 4. Training Challenges

### Q4.1: How do you diagnose and fix underfitting?

**Answer:**

**Symptoms:**
- High training loss
- High validation loss
- Both losses similar (low variance)

**Causes & Solutions:**

| Cause | Solution |
|-------|----------|
| Model too simple | Increase capacity (layers, neurons) |
| Not enough training | Train longer, more epochs |
| High regularization | Reduce dropout, L2 weight |
| Bad learning rate | Usually too low, increase |
| Poor features | Better feature engineering |
| Bad architecture | Try different architecture |

**Debugging Steps:**
1. Overfit on small subset first (proves model can learn)
2. Gradually add complexity
3. Monitor training loss closely

---

### Q4.2: How do you diagnose and fix overfitting?

**Answer:**

**Symptoms:**
- Low training loss
- High validation loss
- Large gap between train/val (high variance)

**Causes & Solutions:**

| Solution | Description |
|----------|-------------|
| **More data** | Best solution if available |
| **Data augmentation** | Artificially expand dataset |
| **Dropout** | Random neuron dropping |
| **L2 regularization** | Weight penalty |
| **Early stopping** | Stop when val loss increases |
| **Reduce model size** | Fewer layers/neurons |
| **Batch normalization** | Regularization effect |

**Key Insight:**
Overfitting = model memorizing training data instead of learning patterns.

---

### Q4.3: What is the bias-variance tradeoff?

**Answer:**

**Error Decomposition:**
```
Error = Bias² + Variance + Irreducible Noise
```

| Component | Description | Model Type |
|-----------|-------------|------------|
| **High Bias** | Consistently wrong | Underfitting (too simple) |
| **High Variance** | Sensitive to training data | Overfitting (too complex) |

**The Tradeoff:**
- Simple models: High bias, low variance
- Complex models: Low bias, high variance
- Goal: Find optimal complexity

**Visualization:**
```
Error
  │    Total
  │     ╱╲
  │    ╱  ╲____
  │   ╱        ╲
  │ Variance   Bias
  └─────────────────→ Model Complexity
         ↑
    Sweet Spot
```

---

## 5. Hyperparameter Tuning

### Q5.1: What are the most important hyperparameters to tune?

**Answer:**

**Priority Order:**

| Priority | Hyperparameter | Typical Range |
|----------|---------------|---------------|
| 🔴 High | Learning rate | 1e-5 to 1e-1 |
| 🔴 High | Batch size | 16, 32, 64, 128, 256 |
| 🟡 Medium | Network depth | 2-100 layers |
| 🟡 Medium | Hidden units | 64-2048 |
| 🟡 Medium | Dropout rate | 0.1-0.5 |
| 🟢 Low | Weight decay | 1e-5 to 1e-2 |
| 🟢 Low | Momentum | 0.9-0.99 |

**Tuning Strategy:**
1. Start with known good defaults
2. Tune learning rate first (most impact)
3. Then batch size and architecture
4. Fine-tune regularization last

---

### Q5.2: Compare Grid Search, Random Search, and Bayesian Optimization.

**Answer:**

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Grid Search** | Try all combinations | Exhaustive | Exponential time |
| **Random Search** | Random sampling | More efficient | Can miss optimal |
| **Bayesian** | Build surrogate model | Most efficient | Complex to implement |

**Random Search Insight:**
- Often finds good solutions faster than grid
- Not all hyperparameters equally important
- Random better explores important dimensions

**When to Use:**

| Scenario | Method |
|----------|--------|
| Few hyperparameters (<3) | Grid search |
| Many hyperparameters | Random search |
| Expensive evaluations | Bayesian optimization |
| Quick exploration | Random search |

---

### Q5.3: Explain early stopping as regularization.

**Answer:**
**Early Stopping:** Stop training when validation loss stops improving.

**Why It Works:**
1. Training loss keeps decreasing
2. Validation loss eventually increases (overfitting)
3. Stopping at minimum val loss = optimal generalization

**Implementation:**
```python
best_val_loss = float('inf')
patience = 10
patience_counter = 0

for epoch in range(max_epochs):
    val_loss = evaluate(model)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(model)
        patience_counter = 0
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        break  # Early stop
```

**Key Hyperparameters:**
- `patience`: How many epochs to wait (typically 5-20)
- Often combined with model checkpointing

---

## 📚 Key Takeaways

1. **Adam is the safe default** - Works well without tuning
2. **Learning rate is most important** - Start with LR range test
3. **Warmup helps transformers** - First ~5% of training
4. **AdamW > Adam** - Better weight decay handling
5. **Early stopping is free regularization** - Use it!
6. **Random search > grid search** - More efficient
