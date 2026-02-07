# ✂️ Deep Dive: Chunking Strategies

> How to split documents into "A-Grade" chunks for maximum LLM performance.

---

## Why Chunk?

Large Language Models (LLMs) have limited context windows. Even if a model has 128k tokens, filling it with 10 entire PDFs will:
1.  **Increase Cost**: You pay per token.
2.  **Dilute Context**: The model might "lose" the specific answer in a sea of irrelevant text (Lost-in-the-Middle phenomenon).
3.  **Decrease Accuracy**: Retrieval is more precise on smaller, focused units of text.

---

## 🏗️ Chunking Levels

| level | Method | Pros | Cons |
|-------|--------|------|------|
| **1. Fixed-Size** | Every 500 characters | Simple, fast | Cuts sentences in half |
| **2. Token-Aware** | Every 500 tokens | Optimal for LLMs | Still cuts concepts |
| **3. Semantic** | NLP-based split | High consistency | Slow, heavy processing |
| **4. Recursive** | Split by md/para/sent | Best balance | Complex implementation |

---

## ⚡ The RAG Engine Mini Approach

We use **Token-Aware Recursive Chunking** with `tiktoken`.

### 1. Why Tokens and not Characters?
OpenAI (and others) charge and think in tokens. 500 tokens is a consistent "measurement" of meaning across languages, whereas 500 characters might be 150 words in English but only 100 in Arabic.

### 2. Overlap is Mandatory
If you split exactly at token 500, the answer to a question might be split:
*   *Chunk 1*: "The secret key is..."
*   *Chunk 2*: "...42-ABC-XYZ."

Without overlap, neither chunk knows the full secret.
**Wait, what about the overlap?**
In RAG Engine Mini, we use a default of **512 tokens** with a **50 token overlap** (~10% buffer).

### 3. Header Awareness (Recursive)
We prioritize splitting at double newlines (`\n\n`), then single newlines (`\n`), then spaces. This keeps paragraphs together as much as possible.

---

## 🛠️ Code Logic

Our logic in `src/application/services/chunking.py`:

```python
def chunk_text(text, max_tokens=512, overlap=50):
    tokens = encoding.encode(text)
    chunks = []
    
    for i in range(0, len(tokens), max_tokens - overlap):
        chunk_tokens = tokens[i : i + max_tokens]
        chunks.append(encoding.decode(chunk_tokens))
        
        if i + max_tokens >= len(tokens):
            break
            
    return chunks
```

---

## 🌟 Pro Tips for Chunking

1.  **Small Chunks (128-256 tokens)**: Best for specific facts, troubleshooting steps, and fact-checking.
2.  **Large Chunks (512-1024 tokens)**: Best for summarization, complex logic extraction, and narrative analysis.
3.  **Context Injection**: In Stage 2 of this project, we plan to add "Parent-Child" chunking where small chunks are retrieved, but the LLM receives the larger surrounding context.

---

## 📚 Further Reading
- [Pinecone Guide to Chunking](https://www.pinecone.io/learn/chunking-strategies/)
- [OpenAI Tokenizer](https://platform.openai.com/tokenizer)
