# Transformer Architecture

## Introduction

The Transformer, introduced by Vaswani et al. in the landmark paper "Attention Is All You Need" (2017), revolutionized natural language processing. Unlike RNNs that process sequentially, Transformers use **self-attention** to process entire sequences in parallel.

### Why Transformers?

| Aspect | RNNs | Transformers |
|--------|------|--------------|
| Parallelization | Sequential (slow) | Fully parallel (fast) |
| Long-range dependencies | Difficult (vanishing gradient) | Direct attention |
| Memory | Hidden state (limited) | All positions |
| Training | BPTT (complex) | Simple backprop |

---

## 1. Overall Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TRANSFORMER                                      │
│                                                                          │
│  Input: "The cat sat"                                                    │
│       ↓                                                                  │
│  ┌─────────────┐      ┌─────────────────────────────────────────────┐   │
│  │ Embedding  │      │              ENCODER STACK                  │   │
│  │ + Positional│      │  ┌─────────────────────────────────────┐    │   │
│  │ Encoding   │──────►│  │  Multi-Head Attention + Add+Norm  │    │   │
│  └─────────────┘      │  └─────────────────────────────────────┘    │   │
│                       │              ↓                                 │   │
│                       │  ┌─────────────────────────────────────┐    │   │
│                       │  │  Feed-Forward + Add+Norm            │    │   │
│                       │  └─────────────────────────────────────┘    │   │
│                       │         (×N layers)                        │   │
│                       └─────────────────────────────────────────────┘   │
│                                    ↓                                    │
│                       ┌─────────────────────────────────────────────┐   │
│                       │              DECODER STACK                  │   │
│                       │  ┌─────────────────────────────────────┐    │   │
│                       │  │  Masked Multi-Head Attention       │    │   │
│                       │  └─────────────────────────────────────┘    │   │
│                       │              ↓                               │   │
│                       │  ┌─────────────────────────────────────┐    │   │
│                       │  │  Encoder-Decoder Attention         │    │   │
│                       │  └─────────────────────────────────────┘    │   │
│                       │              ↓                               │   │
│                       │  ┌─────────────────────────────────────┐    │   │
│                       │  │  Feed-Forward + Add+Norm           │    │   │
│                       │  └─────────────────────────────────────┘    │   │
│                       │         (×N layers)                        │   │
│                       └─────────────────────────────────────────────┘   │
│                                    ↓                                    │
│                               Linear + Softmax                          │
│                                    ↓                                    │
│                         Output: "on the mat"                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Self-Attention Mechanism

### 2.1 The Core Idea

Self-attention allows each position in a sequence to attend to all positions in the previous layer. It computes a weighted sum where weights depend on the similarity between query and key vectors.

```
Position i attends to all positions:
      Query(q_i) compares with Keys(k_1, k_2, ..., k_n)
                  ↓
            Attention weights
                  ↓
      Weighted sum of Values(v_1, v_2, ..., v_n)
                  ↓
           Output for position i
```

### 2.2 Scaled Dot-Product Attention

The core computation:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Step-by-step:**

1. **Compute dot products**: $QK^T$ gives similarity between queries and keys
2. **Scale**: Divide by $\sqrt{d_k}$ (key dimension) to prevent large gradients
3. **Softmax**: Convert to probabilities (attention weights)
4. **Weighted sum**: Multiply by values

```
Q (queries), K (keys), V (values):
┌─────────┐   ┌─────────┐   ┌─────────┐
│    Q    │ × │    K^T  │ → │ Scores │
└─────────┘   └─────────┘   └─────────┘
                               ↓
                         Scale + Softmax
                               ↓
┌─────────┐   ┌─────────┐   ┌─────────┐
│  Weights│ × │    V    │ → │ Output │
└─────────┘   └─────────┘   └─────────┘
```

**Why scale by $\sqrt{d_k}$?**
- Without scaling, dot products grow with $d_k$
- Large dot products → softmax pushes to extreme values
- $\sqrt{d_k}$ normalization keeps softmax "soft"

### 2.3 Multi-Head Attention

Instead of single attention function, use multiple "heads" that learn different attention patterns:

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

Where each head:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

```
Input X:
    │
    ├───────────┬───────────┬───────────┐
    ↓           ↓           ↓           ↓
  Linear     Linear     Linear     Linear
  (WQ_0)     (WQ_1)     (WQ_2)     (WQ_3)
    ↓           ↓           ↓           ↓
  Split into 8 heads (parallel attention)
    ↓           ↓           ↓           ↓
  Attention(Q,K,V) for each head
    ↓           ↓           ↓           ↓
  Concatenate outputs
    ↓
  Final linear projection
```

**Why Multi-Head?**
- Each head can learn different relationships
- One head: syntax ( grammatical)
- Another: semantics ( meaning)
- Another: locality ( nearby words)

---

## 3. Positional Encoding

### 3.1 The Problem

Transformers process all positions simultaneously, losing order information. We need to inject position information.

### 3.2 Sinusoidal Positional Encoding

Using sine and cosine functions of different frequencies:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

Where:
- $pos$: Position in sequence (0, 1, 2, ...)
- $i$: Dimension index (0 to $d-1$)
- $d$: Embedding dimension

**Properties:**
- Each dimension has unique frequency
- Can represent any position
- Derivatives allow model to learn relative positions
- Works well empirically

### 3.3 Alternative: Learned Positional Embeddings

```python
# Simpler alternative
position_embeddings = nn.Embedding(max_seq_len, d_model)
positions = torch.arange(max_seq_len)
pos_encoded = position_embeddings(positions)
```

---

## 4. Feed-Forward Networks

Each transformer layer contains a position-wise feed-forward network:

$$\text{FFN}(x) = W_2 \cdot \sigma(W_1 \cdot x + b_1) + b_2$$

Where $\sigma$ is ReLU activation.

```
Input x (d_model):
    │
    Linear(d_model → d_ff)  (typically d_ff = 4 × d_model)
    │
    ReLU
    │
    Linear(d_ff → d_model)
    │
Output x
```

**Key points:**
- Same weights applied to each position (independent)
- Allows complex transformations per position
- Often the largest component by parameter count

---

## 5. Residual Connections and Layer Normalization

### 5.1 Add & Norm

Each sub-layer has residual connection followed by layer normalization:

$$\text{Output} = \text{LayerNorm}(x + \text{SubLayer}(x))$$

```
Input x:
    │
    SubLayer (attention or FFN)
    │
    + (residual connection)
    │
    LayerNorm
    │
Output
```

**Why?**
- **Residual connections**: Enable gradient flow (helps deep networks)
- **Layer normalization**: Stabilizes training, faster convergence

### 5.2 Layer Normalization

For each sample (across features):
$$\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

Where:
- $\mu = \frac{1}{d} \sum x_i$: Mean
- $\sigma^2 = \frac{1}{d} \sum (x_i - \mu)^2$: Variance
- $\gamma, \beta$: Learnable scale and shift

---

## 6. Encoder Component

### 6.1 Encoder Architecture

Each encoder layer:
1. **Multi-Head Self-Attention**: All positions attend to all positions
2. **Add & Norm**
3. **Feed-Forward Network**
4. **Add & Norm**

### 6.2 Encoder Output

The encoder produces a representation for each position that incorporates information from the entire sequence.

**Use cases:**
- Classification: Use [CLS] token representation
- Extraction: Use per-position representations
- Fine-tuning: Transfer to downstream tasks

---

## 7. Decoder Component

### 7.1 Decoder Architecture

Each decoder layer:
1. **Masked Multi-Head Self-Attention**: Only attend to previous positions
2. **Add & Norm**
3. **Encoder-Decoder Attention**: Attend to encoder output
4. **Add & Norm**
5. **Feed-Forward Network**

### 7.2 Masked Attention

During training, we mask future positions:

```python
# Create mask
mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 0
# Apply to attention scores before softmax
scores = scores.masked_fill(mask == 0, -1e9)
```

This ensures the model can only see previous positions when predicting.

---

## 8. The Full Transformer

### 8.1 Architecture Summary

```
Input: "The cat sat"
  │
  ├─ Embedding Layer (word IDs → vectors)
  │
  ├─ Positional Encoding (add position info)
  │
  ├─ N × Encoder Layers
  │     ├─ Multi-Head Self-Attention
  │     ├─ Add & Norm
  │     ├─ Feed-Forward
  │     └─ Add & Norm
  │
  ├─ N × Decoder Layers
  │     ├─ Masked Multi-Head Self-Attention
  │     ├─ Add & Norm
  │     ├─ Encoder-Decoder Attention
  │     ├─ Add & Norm
  │     ├─ Feed-Forward
  │     └─ Add & Norm
  │
  └─ Linear + Softmax (predict next token)
```

### 8.2 Standard Sizes

| Model | Layers | Hidden | Heads | Parameters |
|-------|--------|--------|-------|------------|
| Base | 6 | 512 | 8 | ~65M |
| Large | 6 | 768 | 12 | ~125M |
| XL | 12 | 1024 | 16 | ~360M |
| XXL | 24 | 2048 | 32 | ~1.5B |

---

## 9. Implementation from Scratch

### 9.1 Scaled Dot-Product Attention

```python
import numpy as np

def scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute scaled dot-product attention."""
    d_k = Q.shape[-1]
    
    # Compute attention scores
    scores = Q @ K.T / np.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    
    # Softmax
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    
    # Weighted sum of values
    output = attention_weights @ V
    
    return output, attention_weights
```

### 9.2 Multi-Head Attention

```python
class MultiHeadAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.1
        self.W_k = np.random.randn(d_model, d_model) * 0.1
        self.W_v = np.random.randn(d_model, d_model) * 0.1
        self.W_o = np.random.randn(d_model, d_model) * 0.1
    
    def split_heads(self, x):
        """Split d_model into num_heads of d_k."""
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)  # (batch, heads, seq, d_k)
    
    def forward(self, Q, K, V, mask=None):
        # Linear projections
        Q = Q @ self.W_q
        K = K @ self.W_k
        V = V @ self.W_v
        
        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Scaled dot-product attention
        attention, _ = scaled_dot_product_attention(Q, K, V, mask)
        
        # Combine heads
        attention = attention.transpose(0, 2, 1, 3).reshape(-1, self.d_model)
        
        # Final linear
        output = attention @ self.W_o
        
        return output
```

### 9.3 Positional Encoding

```python
def positional_encoding(seq_len, d_model):
    """Generate sinusoidal positional encodings."""
    pe = np.zeros((seq_len, d_model))
    
    for pos in range(seq_len):
        for i in range(d_model // 2):
            # Even dimensions: sin
            pe[pos, 2*i] = np.sin(pos / np.power(10000, 2*i / d_model))
            # Odd dimensions: cos
            pe[pos, 2*i + 1] = np.cos(pos / np.power(10000, 2*i / d_model))
    
    return pe
```

---

## 10. Training Transformers

### 10.1 Loss Function

For language modeling (next token prediction):
$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{1:t-1})$$

Cross-entropy loss between predicted and actual tokens.

### 10.2 Optimizations

| Technique | Purpose |
|-----------|---------|
| Adam Optimizer | Adaptive learning rates |
| Warmup Steps | Stable early training |
| Gradient Clipping | Prevent exploding gradients |
| Label Smoothing | Prevent overconfidence |
| Dropout | Regularization |

### 10.3 Beam Search

At inference time, instead of greedy selection, use beam search:

```
Greedy: Always pick highest probability token
Beam Search: Keep top-k candidates, explore all paths
```

---

## 11. Transformer Variants

### 11.1 BERT (Bidirectional Encoder Representations)

- **Architecture**: Encoder-only
- **Training**: Masked Language Modeling + Next Sentence Prediction
- **Use**: Text classification, NER, question answering
- **Key innovation**: Bidirectional attention

### 11.2 GPT (Generative Pre-training)

- **Architecture**: Decoder-only
- **Training**: Causal (left-to-right) language modeling
- **Use**: Text generation, completion
- **Key innovation**: Autoregressive generation

### 11.3 T5 (Text-to-Text Transfer Transformer)

- **Architecture**: Encoder-Decoder
- **Training**: Text-to-text (everything as generation)
- **Use**: Translation, summarization, QA
- **Key innovation**: Unified text-to-text framework

---

## 12. Key Takeaways

1. **Self-Attention**: Core mechanism for modeling dependencies
2. **Multi-Head**: Parallel attention for multiple relationship types
3. **Positional Encoding**: Injecting order information
4. **Residual Connections**: Enable deep networks
5. **LayerNorm**: Stabilize training
6. **Parallelization**: Major advantage over RNNs

---

## 13. Further Reading

- **Original Paper**: "Attention Is All You Need" (Vaswani et al., 2017)
- **BERT**: "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2018)
- **GPT**: "Improving Language Understanding by Generative Pre-Training" (Radford et al.)
- **Transformer-XL**: "Transformer-XL: Attentive Language Modeling Beyond Fixed-Length Context"

---

## 14. Common Interview Questions

| Question | Answer |
|----------|--------|
| Why scale by √d? | Prevents large dot products from pushing softmax to extreme values |
| How does self-attention capture dependencies? | Direct attention to any position, regardless of distance |
| What's the difference between encoder and decoder attention? | Encoder: self-attention; Decoder: masked self + encoder-decoder |
| Why use sinusoidal position encoding? | Can represent any position, model learns relative positions |
| What are the main advantages over RNNs? | Parallelizable, better long-range modeling, simpler to train |