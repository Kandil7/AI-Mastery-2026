# Chapter 1: Attention & Transformers

In this chapter, we dive into the fundamental unit of modern AI: the **Transformer**. We will implement the core components using NumPy and PyTorch, following the code in `src/llm/`.

## 1. Attention: All You Need?

The power of Transformers comes from **Attention**, which allows every token in a sequence to "look" at every other token to find context.

### 1.1 Scaled Dot-Product Attention
We use **Scaled** dot-product to prevent gradients from vanishing/exploding when $d_k$ is large.

**Mathematical Proof (Hands-on):**
If $Q, K \sim N(0, 1)$, then $Q \cdot K^T$ has variance $d_k$. By dividing by $\sqrt{d_k}$, we bring the variance back to 1, ensuring the softmax function operates in a region where it has meaningful gradients.

```python
# From src/llm/attention.py
def scaled_dot_product_attention(query, key, value, mask=None):
    d_k = query.shape[-1]
    scores = np.matmul(query, key.swapaxes(-2, -1)) / np.sqrt(d_k)
    if mask is not None:
        scores += (mask * -1e9)
    weights = softmax(scores)
    return np.matmul(weights, value), weights
```

### 1.2 Multi-Head Attention (MHA)
Why use multiple heads? A single head might focus too much on one specific relationship (e.g., the next word). Multiple heads allow the model to simultaneously focus on:
-   **Syntactic relationships** (Subject-Verb)
-   **Semantic relationships** (Entity-Action)
-   **Coreference** (He -> John)

**Hands-on Exercise**: Open `src/llm/attention.py` and modify the `num_heads` parameter. Observe how the output shape remains the same while the internal representation capacity changes.

---

## 2. Encoder vs Decoder

### 2.1 BERT (The Understanding Machine)
BERT is a **Bidirectional** encoder. It sees the whole sentence at once. This is great for:
-   Sentiment Analysis
-   Named Entity Recognition (NER)
-   Question Answering (Reading Comprehension)

### 2.2 GPT (The Generative Machine)
GPT is an **Autoregressive** decoder. It uses **Causal Masking** to ensure it cannot "see the future."

```python
# Causal Mask implementation from src/llm/transformer.py
causal_mask = np.triu(np.ones((seq_len, seq_len)) * -1e9, k=1)
```

---

## 🚀 Lab: Building a Transformer Block
Try to assemble a `TransformerEncoderLayer` manually:

```python
from src.llm.transformer import MultiHeadAttention, LayerNorm, Dense

# 1. Input
x = np.random.randn(1, 10, 512)

# 2. Multi-Head Attention + Residual
attn = MultiHeadAttention(d_model=512, num_heads=8)
norm1 = LayerNorm(512)
x_attn = attn.forward(x, x, x)
x = norm1.forward(x + x_attn)

# 3. Feed Forward + Residual
ffn = Dense(512, 2048)
norm2 = LayerNorm(512)
x_ffn = ffn.forward(x)
x = norm2.forward(x + x_ffn)
```

## ❓ Troubleshooting
-   **Shape Mismatch**: Check if `d_model` is divisible by `num_heads`.
-   **NaN Loss**: Ensure you are using `LayerNorm` after every major operation.

---
[Next Chapter: Enterprise RAG](./02_enterprise_rag.md)
