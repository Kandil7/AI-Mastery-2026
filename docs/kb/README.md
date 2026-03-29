# Knowledge Base - AI-Mastery-2026

<div align="center">

![Articles](https://img.shields.io/badge/articles-50+-blue.svg)
![Last Updated](https://img.shields.io/badge/updated-March%2028%2C%202026-green.svg)

**Comprehensive knowledge base for AI engineering concepts and best practices**

[Concepts](#-core-concepts) • [Best Practices](#-best-practices) • [Troubleshooting](#-troubleshooting) • [FAQ](../faq/)

</div>

---

## 📚 Core Concepts

### Transformer Architecture

<details>
<summary><strong>What is a Transformer?</strong></summary>

A Transformer is a deep learning architecture based on the self-attention mechanism, introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017).

**Key Components:**
- **Self-Attention:** Allows the model to weigh the importance of different tokens
- **Positional Encoding:** Provides sequence order information
- **Feed-Forward Networks:** Process attention outputs
- **Layer Normalization:** Stabilizes training
- **Residual Connections:** Enables deep architectures

**Why Transformers?**
- Parallel processing (unlike RNNs)
- Long-range dependency capture
- State-of-the-art results on NLP tasks
- Scalable to billions of parameters

</details>

<details>
<summary><strong>How does Self-Attention work?</strong></summary>

Self-attention computes attention scores between all token pairs:

```python
# Scaled Dot-Product Attention
def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / sqrt(d_k)  # Attention scores
    weights = softmax(scores)      # Normalize
    output = weights @ V           # Weighted sum
    return output
```

**Process:**
1. **Query, Key, Value:** Each token is projected into Q, K, V vectors
2. **Score Calculation:** Compute dot product between Q and K
3. **Scaling:** Divide by sqrt(d_k) to stabilize gradients
4. **Softmax:** Normalize scores to probabilities
5. **Weighted Sum:** Combine values based on attention weights

</details>

### RAG Fundamentals

<details>
<summary><strong>What is RAG?</strong></summary>

**Retrieval-Augmented Generation (RAG)** combines retrieval-based and generation-based approaches:

```
Query → Retrieval → Context + Query → Generation → Answer
```

**Components:**
1. **Retriever:** Finds relevant documents from a knowledge base
2. **Generator:** LLM that produces answers using retrieved context

**Benefits:**
- Access to up-to-date information
- Reduced hallucinations
- Explainable through source citations
- Cost-effective (smaller models with retrieval)

</details>

<details>
<summary><strong>How does Vector Search work?</strong></summary>

Vector search finds similar items by comparing embeddings:

**Process:**
1. **Embedding:** Convert text to vectors (e.g., 1536 dimensions)
2. **Indexing:** Store vectors in a vector database
3. **Query:** Embed the query and find nearest neighbors
4. **Similarity:** Use cosine similarity or dot product

**Algorithms:**
- **Exact Search:** Compare with all vectors (accurate but slow)
- **Approximate (ANN):** HNSW, IVF, PQ (fast with slight accuracy loss)

```python
# Cosine Similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

</details>

### Embedding Models

<details>
<summary><strong>What are Embeddings?</strong></summary>

Embeddings are dense vector representations that capture semantic meaning:

**Properties:**
- Similar concepts have similar vectors
- Vector arithmetic captures relationships
- Dimensionality: 128 to 4096+ dimensions

**Popular Models:**
| Model | Dimensions | Max Tokens | Use Case |
|-------|------------|------------|----------|
| text-embedding-ada-002 | 1536 | 8191 | General purpose |
| all-MiniLM-L6-v2 | 384 | 512 | Fast, lightweight |
| bge-large-en | 1024 | 512 | High quality |
| m3e-base | 768 | 512 | Multilingual |

</details>

### LLM Fine-Tuning

<details>
<summary><strong>What is Fine-Tuning?</strong></summary>

Fine-tuning adapts a pre-trained model to specific tasks:

**Approaches:**

1. **Full Fine-Tuning:**
   - Update all parameters
   - Requires significant compute
   - Best for domain adaptation

2. **Parameter-Efficient (PEFT):**
   - **LoRA:** Low-Rank Adaptation
   - **QLoRA:** Quantized LoRA
   - **Prefix Tuning:** Learnable prefixes
   - Updates only small subset of parameters

**When to Fine-Tune:**
- Domain-specific language
- Specialized tasks
- Style/tone adaptation
- When prompting is insufficient

</details>

<details>
<summary><strong>How does LoRA work?</strong></summary>

**LoRA (Low-Rank Adaptation)** freezes base weights and adds low-rank matrices:

```
W' = W + ΔW = W + BA

where:
W ∈ ℝ^(d×k) - frozen weights
B ∈ ℝ^(d×r) - trainable
A ∈ ℝ^(r×k) - trainable
r << d (e.g., r=8, d=4096)
```

**Benefits:**
- 10,000x fewer parameters to train
- No inference latency
- Easy task switching
- Memory efficient

</details>

---

## 🎯 Best Practices

### Prompt Engineering

<details>
<summary><strong>What makes a good prompt?</strong></summary>

**CLEAR Framework:**

- **C**oncise: Be brief and specific
- **L**ogical: Structure your request clearly
- **E**xplicit: State requirements explicitly
- **A**dapptive: Adjust based on results
- **R**efined: Iterate and improve

**Example:**

❌ Bad:
```
Tell me about AI
```

✅ Good:
```
Explain the key differences between supervised and unsupervised learning.
Include:
1. Definition of each
2. 2-3 examples
3. When to use each approach

Target audience: Beginner with basic programming knowledge
Format: Bullet points with clear headings
```

</details>

<details>
<summary><strong>Advanced Prompting Techniques</strong></summary>

**1. Chain-of-Thought (CoT):**
```
Let's solve this step by step. First, we need to...
```

**2. Few-Shot Learning:**
```
Example 1:
Input: "I loved the movie!"
Sentiment: Positive

Example 2:
Input: "The service was terrible."
Sentiment: Negative

Now classify: "The food was okay but expensive."
```

**3. Role Prompting:**
```
You are an expert Python developer with 10 years of experience.
Review this code for performance issues...
```

**4. Self-Consistency:**
```
Generate 3 different solutions, then select the best one.
```

</details>

### RAG Optimization

<details>
<summary><strong>How to improve RAG accuracy?</strong></summary>

**1. Chunking Strategy:**
- **Size:** 256-512 tokens (balance context vs. precision)
- **Overlap:** 50-100 tokens (maintain coherence)
- **Semantic:** Split by meaning, not just length

**2. Retrieval Optimization:**
- **Hybrid Search:** Combine dense + sparse retrieval
- **Re-ranking:** Re-rank top results with cross-encoder
- **Query Expansion:** Add synonyms and related terms

**3. Context Management:**
- **Compression:** Summarize retrieved chunks
- **Selection:** Filter irrelevant chunks
- **Ordering:** Put most relevant first

```python
# Hybrid retrieval example
results_dense = vector_db.search(query_vector, k=20)
results_sparse = bm25.search(query_text, k=20)
combined = reciprocal_rank_fusion(results_dense, results_sparse)
reranked = cross_encoder.rerank(query, combined[:50])
```

</details>

<details>
<summary><strong>Common RAG Issues</strong></summary>

| Issue | Cause | Solution |
|-------|-------|----------|
| Irrelevant results | Poor embeddings | Use domain-specific embeddings |
| Missing context | Bad chunking | Adjust chunk size/overlap |
| Lost in middle | Long context | Put key info at start/end |
| Contradictions | Multiple sources | Add source verification |
| Slow retrieval | Large index | Use ANN, add filters |

</details>

### LLM Evaluation

<details>
<summary><strong>How to evaluate LLM outputs?</strong></summary>

**Metrics:**

1. **Quality Metrics:**
   - **Accuracy:** Correctness of facts
   - **Relevance:** Staying on topic
   - **Coherence:** Logical flow
   - **Fluency:** Natural language

2. **Safety Metrics:**
   - **Toxicity:** Harmful content
   - **Bias:** Unfair representations
   - **Privacy:** PII leakage

3. **Performance Metrics:**
   - **Latency:** Response time
   - **Throughput:** Requests/second
   - **Cost:** Token usage

**Evaluation Methods:**
- **Human Evaluation:** Gold standard but expensive
- **LLM-as-Judge:** Use stronger LLM to evaluate
- **Automated Metrics:** ROUGE, BLEU, BERTScore
- **Task-Specific:** Accuracy on benchmarks

</details>

### Cost Optimization

<details>
<summary><strong>How to reduce LLM costs?</strong></summary>

**1. Model Selection:**
- Use smaller models for simple tasks
- Route complex queries to larger models
- Consider open-source alternatives

**2. Token Optimization:**
- Optimize prompts (remove unnecessary tokens)
- Use shorter contexts
- Implement token caching

**3. Caching:**
```python
# Cache identical queries
cache_key = hash(query + context)
if cache_key in cache:
    return cache[cache_key]
```

**4. Batch Processing:**
- Combine multiple requests
- Use async processing
- Schedule non-urgent tasks

**5. Self-Hosting:**
- For high-volume use cases
- Use vLLM, TGI for efficiency
- Consider spot instances

</details>

---

## 🐛 Troubleshooting

### Common Issues

<details>
<summary><strong>Out of Memory Errors</strong></summary>

**Symptoms:**
```
CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions:**

1. **Reduce Batch Size:**
```python
batch_size = 4  # Instead of 32
```

2. **Gradient Accumulation:**
```python
# Accumulate gradients over multiple steps
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
```

3. **Mixed Precision:**
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
```

4. **Model Parallelism:**
```python
# Split model across GPUs
model = nn.DataParallel(model, device_ids=[0, 1])
```

</details>

<details>
<summary><strong>Poor RAG Performance</strong></summary>

**Symptoms:**
- Irrelevant retrieved documents
- Hallucinated answers
- Missing key information

**Debugging Steps:**

1. **Check Retrieval:**
```python
# Log retrieved chunks
results = retriever.search(query)
for i, chunk in enumerate(results):
    print(f"Chunk {i}: {chunk.score}")
    print(chunk.text[:200])
```

2. **Evaluate Embeddings:**
```python
# Test similarity
similarity = cosine_similarity(query_emb, doc_emb)
print(f"Similarity: {similarity}")  # Should be > 0.5 for relevant
```

3. **Analyze Chunks:**
- Are chunks too large/small?
- Is key information split across chunks?
- Is metadata being used for filtering?

4. **Test Different Retrievers:**
- Try different embedding models
- Add hybrid search
- Implement re-ranking

</details>

<details>
<summary><strong>Slow Inference</strong></summary>

**Symptoms:**
- High latency (>5s for response)
- Low throughput (<10 req/s)

**Optimization:**

1. **Model Optimization:**
```bash
# Use quantization
python -m transformers.quantize --model llama-7b --bits 4

# Use vLLM for serving
vllm serve llama-7b --tensor-parallel-size 4
```

2. **Caching:**
```python
# KV cache for repeated prompts
model.generate(..., use_cache=True)
```

3. **Batching:**
```python
# Dynamic batching
batch = collect_requests(timeout=100ms)
outputs = model.generate(batch)
```

4. **Hardware:**
- Use GPU acceleration
- Consider inference chips (TPU, Inferentia)
- Optimize memory bandwidth

</details>

---

## 📖 Glossary

| Term | Definition |
|------|------------|
| **Attention** | Mechanism that weighs importance of different input parts |
| **Batch Size** | Number of samples processed before model update |
| **Beam Search** | Decoding strategy that explores multiple sequences |
| **Checkpoint** | Saved model state during training |
| **Embedding** | Vector representation of text |
| **Fine-Tuning** | Adapting pre-trained model to specific task |
| **Gradient** | Derivative of loss with respect to parameters |
| **Hallucination** | Model generating false or fabricated information |
| **Inference** | Using trained model to make predictions |
| **LoRA** | Low-Rank Adaptation for efficient fine-tuning |
| **Perplexity** | Metric for language model quality |
| **Quantization** | Reducing numerical precision for efficiency |
| **RAG** | Retrieval-Augmented Generation |
| **Temperature** | Controls randomness in generation |
| **Token** | Unit of text (word/subword) |
| **Transformer** | Neural network architecture based on attention |
| **Vector DB** | Database optimized for vector similarity search |
| **Zero-Shot** | Model performing task without examples |

---

## 🔗 Related Resources

- [FAQ](../faq/) - Frequently asked questions
- [Tutorials](../tutorials/) - Step-by-step guides
- [API Reference](../api/) - Complete API docs
- [Glossary](../reference/glossary.md) - Terms and definitions

---

**Last Updated:** March 28, 2026  
**Articles:** 50+  
**Contributors:** AI-Mastery-2026 Team
