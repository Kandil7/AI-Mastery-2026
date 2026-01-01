# Deep Learning Fundamentals - Interview Questions

A comprehensive collection of deep learning theory questions and answers for AI/ML interviews.

---

## Table of Contents
1. [Neural Network Basics](#1-neural-network-basics)
2. [Backpropagation](#2-backpropagation)
3. [Activation Functions](#3-activation-functions)
4. [Regularization](#4-regularization)
5. [Optimization](#5-optimization)
6. [Architectures](#6-architectures)

---

## 1. Neural Network Basics

### Q1.1: What is a neural network and how does it learn?

**Answer:**
A neural network is a computational model inspired by biological neurons, consisting of:
- **Layers**: Input, hidden, and output layers
- **Neurons**: Compute weighted sum of inputs + bias, apply activation
- **Connections**: Weighted edges between neurons

**Learning Process:**
1. **Forward Pass**: Compute predictions from inputs
2. **Loss Calculation**: Measure error between prediction and target
3. **Backward Pass**: Compute gradients via backpropagation
4. **Weight Update**: Adjust weights to minimize loss

```
y = f(Wx + b)

Where:
- x = input
- W = weights (learned)
- b = bias (learned)
- f = activation function
```

---

### Q1.2: What is the Universal Approximation Theorem?

**Answer:**
The Universal Approximation Theorem states that a feedforward neural network with:
- At least one hidden layer
- A sufficient number of neurons
- A non-linear activation function

Can approximate any continuous function to arbitrary accuracy on a compact domain.

**Practical Implications:**
- Neural networks are flexible function approximators
- Deeper networks often need fewer neurons (more efficient)
- Having the capacity doesn't guarantee learning

---

### Q1.3: Explain the difference between shallow and deep networks.

**Answer:**

| Aspect | Shallow (1-2 layers) | Deep (many layers) |
|--------|---------------------|-------------------|
| **Capacity** | Limited | High |
| **Features** | Hand-crafted | Learned hierarchically |
| **Training** | Easier | Harder (vanishing gradients) |
| **Data needs** | Less | More |
| **Representations** | Simple | Hierarchical (edges → shapes → objects) |

**Why Depth Matters:**
- Each layer learns increasingly abstract features
- Depth enables exponentially more efficient function representation
- Example: Vision - edges → textures → parts → objects

---

## 2. Backpropagation

### Q2.1: Explain backpropagation step by step.

**Answer:**
Backpropagation computes gradients of the loss with respect to each weight using the chain rule.

**Algorithm:**
1. **Forward Pass**: Compute all activations layer by layer
2. **Compute Output Gradient**: ∂L/∂ŷ
3. **Backward Pass**: For each layer (output to input):
   ```
   ∂L/∂W = ∂L/∂a × ∂a/∂z × ∂z/∂W
   
   Where:
   - a = activation output
   - z = pre-activation (Wx + b)
   ```
4. **Update Weights**: W = W - η × ∂L/∂W

**Key Insight:** Chain rule allows computing gradients efficiently in O(n) time where n = number of parameters.

---

### Q2.2: What is the vanishing gradient problem and how do you solve it?

**Answer:**
**Problem:** In deep networks, gradients become very small when propagating backward, causing:
- Early layers learn very slowly
- Training stagnates
- Network doesn't learn deep representations

**Causes:**
- Sigmoid/tanh activations saturate (gradient ≈ 0)
- Multiple small gradients multiply together

**Solutions:**

| Solution | How It Helps |
|----------|--------------|
| **ReLU activation** | Gradient = 1 for positive inputs |
| **Residual connections** | Skip connections allow gradient flow |
| **Batch Normalization** | Keeps activations in good range |
| **Proper initialization** | Xavier/He initialization |
| **LSTM/GRU gates** | Control gradient flow in RNNs |

---

### Q2.3: What is the exploding gradient problem?

**Answer:**
**Problem:** Gradients grow exponentially large during backpropagation, causing:
- Weights become NaN
- Training diverges
- Unstable learning

**Solutions:**
1. **Gradient Clipping**: Limit gradient magnitude
   ```python
   if ||g|| > threshold:
       g = threshold × g / ||g||
   ```
2. **Proper weight initialization**
3. **Lower learning rate**
4. **Batch normalization**

---

## 3. Activation Functions

### Q3.1: Compare ReLU, Sigmoid, and Tanh.

**Answer:**

| Function | Formula | Range | Pros | Cons |
|----------|---------|-------|------|------|
| **Sigmoid** | 1/(1+e^(-x)) | (0, 1) | Smooth, probabilistic | Vanishing gradient, not zero-centered |
| **Tanh** | (e^x - e^(-x))/(e^x + e^(-x)) | (-1, 1) | Zero-centered | Vanishing gradient |
| **ReLU** | max(0, x) | [0, ∞) | No vanishing gradient, fast | Dead neurons, not zero-centered |

**When to Use:**
- **Sigmoid**: Output layer for binary classification
- **Tanh**: RNNs (historically)
- **ReLU**: Hidden layers (default choice)
- **Softmax**: Multi-class classification output

---

### Q3.2: What is the dying ReLU problem and how do you fix it?

**Answer:**
**Problem:** ReLU neurons can "die" when they output 0 for all inputs:
- If input is always negative → gradient is always 0
- Weight never updates → neuron is permanently dead

**Solutions:**

| Variant | Formula | How It Helps |
|---------|---------|--------------|
| **Leaky ReLU** | max(0.01x, x) | Small gradient for negative inputs |
| **PReLU** | max(αx, x) | Learnable α |
| **ELU** | x if x>0, α(e^x-1) otherwise | Smooth, negative values |
| **GELU** | x·Φ(x) | Used in transformers |

---

### Q3.3: Why is ReLU not differentiable at 0? Is this a problem?

**Answer:**
ReLU is not differentiable at exactly x=0 because the left derivative (0) ≠ right derivative (1).

**In Practice:**
- This is NOT a significant problem
- We use subgradients (any value between 0 and 1)
- The probability of x being exactly 0 is essentially 0
- Implementations typically use 0 or 1 at x=0

---

## 4. Regularization

### Q4.1: Explain L1 vs L2 regularization.

**Answer:**

| Aspect | L1 (Lasso) | L2 (Ridge) |
|--------|-----------|-----------|
| **Penalty** | Σ\|w\| | Σw² |
| **Effect on weights** | Sparse (many zeros) | Small but non-zero |
| **Geometric view** | Diamond constraint | Circle constraint |
| **Feature selection** | Yes (implicit) | No |
| **Bayesian view** | Laplace prior | Gaussian prior |

**When to Use:**
- **L1**: When you expect many features are irrelevant
- **L2**: Default choice, prevents overfitting
- **Elastic Net**: Combine both (L1 + L2)

---

### Q4.2: How does dropout work and why is it effective?

**Answer:**
**Mechanism:**
- During training: Randomly set p% of neurons to 0
- During inference: Scale outputs by (1-p) or scale during training

**Why It Works:**
1. **Prevents co-adaptation**: Neurons can't rely on specific other neurons
2. **Ensemble effect**: Like training many sub-networks
3. **Noise injection**: Adds regularization noise

**Implementation:**
```python
# Training
mask = (np.random.rand(*x.shape) < keep_prob) / keep_prob
output = x * mask

# Inference
output = x  # No dropout, already scaled
```

---

### Q4.3: What is batch normalization and why does it help?

**Answer:**
**Mechanism:**
Normalize activations within a mini-batch, then apply learnable scale (γ) and shift (β):

```
x̂ = (x - μ) / √(σ² + ε)
y = γx̂ + β
```

**Benefits:**
1. **Reduces internal covariate shift**: Layer inputs stay stable
2. **Enables higher learning rates**: More stable training
3. **Regularization effect**: Noise from batch statistics
4. **Reduces sensitivity to initialization**

**Key Points:**
- Use different statistics for train (batch) vs. inference (running average)
- Usually applied before activation
- Layer Norm is alternative for transformers

---

## 5. Optimization

### Q5.1: Compare SGD, Momentum, and Adam.

**Answer:**

| Optimizer | Update Rule | Characteristics |
|-----------|-------------|-----------------|
| **SGD** | w = w - η∇L | Simple, noisy, slow |
| **Momentum** | v = βv + ∇L; w = w - ηv | Smooths updates, faster convergence |
| **Adam** | Combines momentum + adaptive LR | Fast, works well with noisy gradients |

**Adam Details:**
```
m = β₁m + (1-β₁)g          # First moment (momentum)
v = β₂v + (1-β₂)g²         # Second moment (RMSprop)
m̂ = m / (1-β₁ᵗ)           # Bias correction
v̂ = v / (1-β₂ᵗ)
w = w - η × m̂ / (√v̂ + ε)
```

**When to Use:**
- **SGD + Momentum**: Best final performance, needs tuning
- **Adam**: Fast convergence, less tuning, good default
- **AdamW**: Adam + proper weight decay

---

### Q5.2: What is the learning rate and how do you choose it?

**Answer:**
**Definition:** Step size for gradient descent updates.

**Effects:**
- Too high → Divergence, oscillation
- Too low → Slow convergence, stuck in local minima
- Just right → Smooth, fast convergence

**Finding Good LR:**
1. **Learning rate range test**: Try 1e-7 to 10, plot loss vs LR
2. **Common starting points**: 1e-3 (Adam), 0.1 (SGD)
3. **Warmup**: Start low, increase gradually

**Learning Rate Schedules:**
- **Step decay**: Divide by 10 at epochs 30, 60
- **Cosine annealing**: Smooth decrease
- **One-cycle**: Increase then decrease
- **ReduceOnPlateau**: Reduce when metric stagnates

---

### Q5.3: Explain the difference between batch, mini-batch, and stochastic gradient descent.

**Answer:**

| Type | Batch Size | Pros | Cons |
|------|------------|------|------|
| **Batch GD** | Full dataset | Stable gradient | Slow, memory-heavy |
| **Stochastic GD** | 1 sample | Fast updates, escape local minima | Very noisy |
| **Mini-batch GD** | 16-256 samples | Best of both | Standard choice |

**Batch Size Considerations:**
- **Larger batches**: More stable, better GPU utilization
- **Smaller batches**: More regularization, may generalize better
- **Typical values**: 32, 64, 128, 256

---

## 6. Architectures

### Q6.1: Explain CNNs and their key components.

**Answer:**
Convolutional Neural Networks are designed for spatial/grid-like data (images).

**Key Components:**

| Component | Purpose | Example |
|-----------|---------|---------|
| **Convolution** | Extract local features | 3×3 filter for edges |
| **Pooling** | Reduce spatial size, add invariance | 2×2 max pooling |
| **Stride** | Control output size | Stride 2 halves dimensions |
| **Padding** | Preserve spatial dimensions | 'same' padding |

**Why CNNs Work for Images:**
1. **Local connectivity**: Pixels relate to neighbors
2. **Weight sharing**: Same filter applied everywhere
3. **Translation invariance**: Features detected anywhere
4. **Hierarchical features**: Low→high level representations

---

### Q6.2: Explain RNNs, LSTMs, and the vanishing gradient problem in sequences.

**Answer:**
**RNNs** process sequences by maintaining hidden state:
```
hₜ = f(Wₕhₜ₋₁ + Wₓxₜ + b)
```

**Problem:** Gradients vanish/explode over long sequences because gradients are multiplied at each step.

**LSTMs** solve this with gates:
- **Forget gate**: What to discard from cell state
- **Input gate**: What new information to store
- **Output gate**: What to output

**Key Insight:** Cell state provides a highway for gradients to flow through time with minimal transformation.

---

### Q6.3: What are Transformers and why did they replace RNNs?

**Answer:**
Transformers use self-attention instead of recurrence.

**Key Innovation - Self-Attention:**
```
Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
```

**Advantages over RNNs:**

| Aspect | RNN | Transformer |
|--------|-----|-------------|
| **Parallelization** | Sequential (slow) | Fully parallel |
| **Long dependencies** | Struggles | Direct attention |
| **Training time** | O(n) sequential | O(1) parallel |
| **Memory** | Fixed hidden state | Attention over all |

**Components:**
- Multi-head attention
- Position encodings (since no recurrence)
- Layer normalization
- Feedforward layers

---

## 📚 Study Tips

1. **Understand the "why"**: Know why techniques work, not just how
2. **Draw diagrams**: Visualize architectures and data flow
3. **Code from scratch**: Implement backprop, simple networks
4. **Know trade-offs**: Every technique has pros and cons
5. **Stay current**: Transformers, attention, recent advances
