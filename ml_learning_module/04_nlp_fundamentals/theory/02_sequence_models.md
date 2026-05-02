# Sequence Models: RNN, LSTM, and GRU

## Introduction to Sequence Modeling

Sequence models are neural networks designed to process sequential data where the order of elements matters. Unlike traditional feedforward networks that process fixed-size inputs independently, sequence models maintain internal state to capture temporal dependencies.

**Why Sequence Models?**

Traditional ML algorithms assume independent samples, but many real-world problems involve sequential dependencies:

| Problem Type | Examples | Challenge |
|--------------|----------|-----------|
| NLP | Text generation, translation | Long-range word dependencies |
| Time Series | Stock prediction, weather | Temporal patterns |
| Speech | ASR, TTS | Audio waveform sequences |
| Video | Action recognition | Frame-by-frame dynamics |

---

## 1. Recurrent Neural Networks (RNN)

### 1.1 Basic RNN Architecture

An RNN processes sequences by maintaining a **hidden state** that gets updated at each time step.

```
Time Step t:
┌─────────────────────────────────────┐
│                                     │
│    x_t ──┬──► [Hidden State h_t] ──┼──► y_t
│         ╱                            │
│        ╱  ┌──────────────┐          │
│        │  │  h_{t-1}     │          │
│        └──►│ (previous   │          │
│             │  hidden)    │          │
│             └──────────────┘          │
└─────────────────────────────────────┘
```

### 1.2 Mathematical Formulation

The RNN forward pass at each time step:

$$h_t = \tanh(W_{xh} \cdot x_t + W_{hh} \cdot h_{t-1} + b_h)$$

$$y_t = W_{hy} \cdot h_t + b_y$$

Where:
- $x_t$: Input at time $t$ (shape: $d_{in}$)
- $h_t$: Hidden state at time $t$ (shape: $d_{hidden}$)
- $h_{t-1}$: Previous hidden state
- $W_{xh}$: Input-to-hidden weights (shape: $d_{hidden} \times d_{in}$)
- $W_{hh}$: Hidden-to-hidden (recurrent) weights (shape: $d_{hidden} \times d_{hidden}$)
- $W_{hy}$: Hidden-to-output weights (shape: $d_{out} \times d_{hidden}$)
- $b_h, b_y$: Bias terms

**Unrolled (Expanded) View:**

```
x_0 → [RNN] → h_0 → [RNN] → h_1 → [RNN] → h_2 → ... → y_t
       │         │         │
       h_{-1}    h_0       h_1
```

### 1.3 RNN Variants

#### Many-to-Many (Sequence to Sequence)
```
Input: [x_0, x_1, ..., x_T]  →  Output: [y_0, y_1, ..., y_T]
```
- Use case: POS tagging, language modeling

#### Many-to-One
```
Input: [x_0, x_1, ..., x_T]  →  Output: y_T
```
- Use case: Sentiment classification, video classification

#### One-to-Many
```
Input: x_0  →  Output: [y_0, y_1, ..., y_T]
```
- Use case: Image captioning, music generation

#### Encoder-Decoder (Seq2Seq)
```
Input: [x_0, ..., x_T] ──► [Encoder] ──► [Context] ──► [Decoder] ──► [y_0, ..., y_T']
```
- Use case: Machine translation, summarization

### 1.4 Backpropagation Through Time (BPTT)

RNNs are trained using BPTT, which unrolls the network through time and computes gradients.

**The BPTT Algorithm:**

1. **Forward pass**: Compute all hidden states and outputs
2. **Backward pass**: Propagate gradients back through each time step

**Gradient computation for $W_{hh}$:**

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=0}^{T} \frac{\partial L}{\partial h_t} \cdot \frac{\partial h_t}{\partial W_{hh}}$$

Using the chain rule:

$$\frac{\partial h_t}{\partial h_{t-1}} = W_{hh}^T \cdot \text{diag}(\tanh'(W_{xh}x_t + W_{hh}h_{t-1}))$$

**The Vanishing Gradient Problem:**

As we backpropagate through many time steps, gradients often become very small:

$$\left\|\frac{\partial h_t}{\partial h_{t-k}}\right\| \approx \lambda^k \quad \text{where } |\lambda| < 1$$

This means:
- Long-term dependencies are difficult to learn
- Earlier time steps receive tiny gradient updates
- The network "forgets" distant information

**Gradient Clipping** (Practical Solution):

To prevent exploding gradients:

```python
# Pseudocode
if gradient_norm > max_norm:
    gradient = gradient * max_norm / gradient_norm
```

---

## 2. Long Short-Term Memory (LSTM)

### 2.1 LSTM Motivation

LSTMs were introduced by Hochreiter & Schmidhuber (1997) to solve the vanishing gradient problem. They use a **gating mechanism** to control information flow.

### 2.2 LSTM Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         LSTM Cell                                        │
│                                                                          │
│   x_t ──► ┌─────────┐                                                    │
│           │  σ      │──► forget_gate ──┐                                │
│           └─────────┘                   │                                │
│                                        ┌──▼──────────────┐              │
│   h_{t-1} ──► ┌─────────┐              │  (multiply)     │              │
│               │  σ      │──► input_gate ─┤                 │              │
│               └─────────┘              │                 ▼              │
│                                       │  ┌─────────┐    ┌─────────┐    │
│               ┌─────────┐             └──│  h_{t-1}├───►│   C_t   │    │
│   x_t ──►     │  σ      │──► output_gate  │  * f_t   │    │  (cell) │    │
│               └─────────┘                │          │    │  + i_t  │    │
│                                        │          └──►│   * g_t  │    │
│   h_{t-1} ──► ┌─────────┐               │              └────┬────┘    │
│               │  tanh   │──► candidate ───┘                    │         │
│               └─────────┘                                       │         │
│                                                                ▼         │
│                                                   ┌─────────┐           │
│   x_t ────────────────────────────────────────────►│ tanh(C) │──► h_t   │
│                                                   └─────────┘           │
└──────────────────────────────────────────────────────────────────────────┘
```

### 2.3 LSTM Equations

**Forget Gate** (decides what to discard):
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**Input Gate** (decides what to store):
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

**Candidate Cell** (new information):
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**Cell Update** (combine old and new):
$$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$

**Output Gate** (decides what to output):
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

**Hidden State** (output):
$$h_t = o_t * \tanh(C_t)$$

### 2.4 Why LSTMs Solve Vanishing Gradients

The **cell state** ($C_t$) acts as a "highway" for gradients:

$$C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$$

- **Forget gate close to 1**: Information preserved
- **Forget gate close to 0**: Information forgotten
- Gradients can flow through the cell state unchanged (forget gate = 1)

**Intuition**: LSTMs have **additive** gradient flow rather than multiplicative, making it easier to maintain gradients over long sequences.

---

## 3. Gated Recurrent Unit (GRU)

### 3.1 GRU Overview

GRU (Cho et al., 2014) is a simplified version of LSTM with fewer gates:

- **Update gate**: Combines forget and input gates
- **Reset gate**: Controls how much past information to forget

### 3.2 GRU Equations

**Reset Gate**:
$$r_t = \sigma(W_r \cdot [h_{t-1}, x_t])$$

**Update Gate**:
$$z_t = \sigma(W_z \cdot [h_{t-1}, x_t])$$

**Candidate Hidden State**:
$$\tilde{h}_t = \tanh(W \cdot [r_t * h_{t-1}, x_t])$$

**Hidden State Update**:
$$h_t = (1 - z_t) * h_{t-1} + z_t * \tilde{h}_t$$

### 3.3 LSTM vs GRU Comparison

| Feature | LSTM | GRU |
|---------|------|-----|
| Gates | 3 (forget, input, output) | 2 (reset, update) |
| Cell State | Yes | No (merged with hidden) |
| Parameters | More | Fewer |
| Training Speed | Slower | Faster |
| Performance | Often slightly better | Often comparable |
| When to Use | Complex long dependencies | Simpler sequences, less data |

---

## 4. Bidirectional RNNs

### 4.1 Motivation

Sometimes we need context from both past AND future:

- Sentence: "The cat ___ on the mat" → Need to see "cat" AND "mat"
-语音识别: Need future context for disambiguation

### 4.2 Bidirectional Architecture

```
Forward:  x_0 ─→ x_1 ─→ x_2 ─→ ... ─→ x_T
             ↓       ↓       ↓
          h_f0    h_f1    h_f2        h_fT

Backward: x_T ←── x_{T-1} ←── x_{T-2} ←─ ... ←─ x_0
             ↓       ↓       ↓
          h_b0    h_b1    h_b2        h_bT

Combine: [h_f_t, h_b_t] ──► Output at time t
```

**Final hidden state at time t**:
$$h_t = [h_{f,t}; h_{b,t}]$$

---

## 5. Multi-Layer RNNs (Stacked RNNs)

### 5.1 Deep RNNs

Stacking multiple RNN layers allows learning more complex patterns:

```
Layer 2: h^{(2)}_t = RNN(h^{(2)}_{t-1}, h^{(1)}_t)
Layer 1: h^{(1)}_t = RNN(h^{(1)}_{t-1}, x_t)

Output: y_t = W * h^{(2)}_t
```

### 5.2 Common Architectures

| Layers | Use Case |
|--------|----------|
| 1-2 | Simple sequences, small datasets |
| 2-4 | Standard NLP tasks |
| 4-6 | Complex sequence tasks, large datasets |

**Tip**: Start with 2 layers, increase if underfitting.

---

## 6. Practical Considerations

### 6.1 Sequence Padding

Neural networks require fixed-length inputs:

```python
# Pad sequences to same length
padded_sequences = pad_sequences(
    sequences, 
    maxlen=max_length, 
    padding='post',
    truncating='post'
)
```

### 6.2 Truncated BPTT

For very long sequences, truncate backpropagation:

```python
# Only backprop 20 steps
truncated_backprop_length = 20
```

### 6.3 Dropout in RNNs

Apply dropout between layers (not within same time step):

```
Dropout layer between RNN layers
(apply same mask at all timesteps for consistency)
```

### 6.4 Learning Rate Scheduling

RNNs are sensitive to learning rate:

```python
# Learning rate schedule
lr = initial_lr * decay_factor ^ (epoch / decay_steps)

# Or use optimizers like Adam
optimizer = Adam(lr=0.001)
```

---

## 7. Implementation from Scratch

### 7.1 Simple RNN Cell

```python
import numpy as np

class RNNCell:
    """Basic RNN cell implementation."""
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Xavier initialization
        scale = np.sqrt(2.0 / (input_size + hidden_size))
        self.W_xh = np.random.randn(hidden_size, input_size) * scale
        self.W_hh = np.random.randn(hidden_size, hidden_size) * scale
        self.b_h = np.zeros(hidden_size)
        
    def forward(self, x, h_prev):
        """Single time step forward pass."""
        self.cache = (x, h_prev)
        h = np.tanh(self.W_xh @ x + self.W_hh @ h_prev + self.b_h)
        return h
    
    def backward(self, dh_next, dh_cur):
        """Backward pass for single time step."""
        x, h_prev = self.cache
        
        # Gradient of tanh
        dh_raw = dh_cur * (1 - h_prev ** 2)
        
        # Gradients
        dW_xh = np.outer(dh_raw, x)
        dW_hh = np.outer(dh_raw, h_prev)
        db_h = dh_raw
        
        dx = self.W_xh.T @ dh_raw
        dh_prev = self.W_hh.T @ dh_raw
        
        return dx, dh_prev
```

### 7.2 LSTM Implementation

```python
class LSTMCell:
    """LSTM cell implementation."""
    
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Combined weight matrix for efficiency
        self.W = np.random.randn(4 * hidden_size, input_size + hidden_size)
        self.b = np.zeros(4 * hidden_size)
        
    def forward(self, x, h_prev, c_prev):
        """LSTM forward pass."""
        # Concatenate for gate computation
        combined = np.concatenate([h_prev, x])
        
        # Compute gates
        gates = self.W @ combined + self.b
        
        # Split into gates
        f = self._sigmoid(gates[:self.hidden_size])
        i = self._sigmoid(gates[self.hidden_size:2*self.hidden_size])
        o = self._sigmoid(gates[2*self.hidden_size:3*self.hidden_size])
        g = np.tanh(gates[3*self.hidden_size:])
        
        # Cell state and hidden state
        c = f * c_prev + i * g
        h = o * np.tanh(c)
        
        self.cache = (x, h_prev, c_prev, f, i, o, g, c)
        return h, c
    
    def _sigmoid(self, x):
        """Numerically stable sigmoid."""
        return np.where(x >= 0, 
                       1 / (1 + np.exp(-x)), 
                       np.exp(x) / (1 + np.exp(x)))
```

---

## 8. Applications

### 8.1 Text Classification

```
Input: "This movie is amazing!"
Process: Tokenize → Embed → LSTM → Dense → Output: Positive

Architecture: Many-to-One
- Input: Sequence of word embeddings
- Output: Single sentiment label
```

### 8.2 Language Modeling

```
Input: "The cat sat on the"
Process: Embed → LSTM → Dense → Output: "mat"

Architecture: Many-to-Many (shifted)
- Predict next token given previous tokens
- Used for text generation
```

### 8.3 Machine Translation (with attention)

```
English: "I love cats" → Japanese: "私は猫が好きです"

Architecture: Encoder-Decoder with Attention
- Encoder: Bi-LSTM processes English
- Decoder: LSTM generates Japanese with attention
- Attention: Focus on relevant source words
```

### 8.4 Named Entity Recognition (NER)

```
Input: "Apple Inc. was founded in Cupertino"
Output: [ORG, ORG, O, O, O, LOC]

Architecture: Many-to-Many
- Tag each word with entity type
- BIO tagging scheme common
```

---

## 9. Summary

| Model | Key Feature | Pros | Cons |
|-------|-------------|------|------|
| **RNN** | Basic recurrence | Simple, interpretable | Vanishing gradients |
| **LSTM** | Cell + 3 gates | Long-term memory | Complex, slow |
| **GRU** | Cell + 2 gates | Faster, fewer params | May miss complex patterns |

**Key Takeaways:**
1. RNNs process sequences by maintaining hidden state
2. LSTMs solve vanishing gradients via gating mechanism
3. Bidirectional models capture past and future context
4. Stacking layers increases representational power
5. Gradient clipping prevents exploding gradients
6. Truncated BPTT enables processing long sequences

**Modern Note**: Transformers have largely replaced RNNs/LSTMs for most NLP tasks due to parallelization and attention advantages, but RNNs remain important for understanding sequence modeling fundamentals.