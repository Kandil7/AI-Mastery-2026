# RAG (Retrieval-Augmented Generation) Fundamentals

**Last Updated:** March 28, 2026  
**Level:** Intermediate  
**Prerequisites:** Basic understanding of LLMs and embeddings

---

## 📋 Table of Contents

- [What is RAG?](#-what-is-rag)
- [Why Use RAG?](#-why-use-rag)
- [RAG Architecture](#-rag-architecture)
- [Core Components](#-core-components)
- [Implementation Guide](#-implementation-guide)
- [Best Practices](#-best-practices)
- [Advanced Patterns](#-advanced-patterns)
- [Evaluation](#-evaluation)
- [Troubleshooting](#-troubleshooting)

---

## 🎯 What is RAG?

**Retrieval-Augmented Generation (RAG)** is an AI architecture that combines:

1. **Retrieval:** Finding relevant information from a knowledge base
2. **Generation:** Using an LLM to generate answers based on retrieved context

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────────┐
│   Query     │ ──→ │  Retriever   │ ──→ │  Context +  │ ──→ │   Answer    │
│  "What is   │     │  (Search     │     │   Query     │     │  (LLM      │
│   ML?"      │     │  documents)  │     │             │     │  generates) │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
                                     ┌──────────────┐
                                     │  Knowledge   │
                                     │   Base       │
                                     └──────────────┘
```

### Key Benefits

| Benefit | Description |
|---------|-------------|
| **Up-to-date** | Access current information without retraining |
| **Reduced Hallucination** | Ground responses in retrieved facts |
| **Explainable** | Cite sources for answers |
| **Cost-Effective** | Use smaller models with retrieval |
| **Domain-Specific** | Add custom knowledge easily |

---

## 🤔 Why Use RAG?

### When to Use RAG

✅ **Good fit for:**
- Question answering over documents
- Customer support chatbots
- Knowledge base search
- Research assistants
- Compliance and legal Q&A

❌ **Not ideal for:**
- Creative writing
- Code generation (usually)
- General conversation
- Tasks requiring no external knowledge

### RAG vs. Fine-Tuning

| Aspect | RAG | Fine-Tuning |
|--------|-----|-------------|
| **Knowledge Source** | External documents | Model weights |
| **Update Frequency** | Instant (add docs) | Requires retraining |
| **Cost** | Lower (API calls) | Higher (GPU hours) |
| **Hallucination** | Reduced | Possible |
| **Best For** | Dynamic knowledge | Style adaptation |

**Recommendation:** Use both when possible for best results!

---

## 🏗️ RAG Architecture

### Basic RAG Flow

```python
from src.rag import RAGSystem

# Initialize
rag = RAGSystem(
    embedding_model="all-MiniLM-L6-v2",
    vector_db="chroma",
    llm="gpt-3.5-turbo"
)

# Add documents
rag.add_documents([
    {"id": "doc1", "content": "Machine learning is a subset of AI..."},
    {"id": "doc2", "content": "Deep learning uses neural networks..."}
])

# Query
result = rag.query("What is machine learning?")
print(f"Answer: {result.answer}")
print(f"Sources: {result.sources}")
```

### Detailed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Query Processing                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Query      │  │  Query      │  │  Query      │             │
│  │  Parsing    │  │  Expansion  │  │  Embedding  │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Retrieval Layer                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Dense      │  │  Sparse     │  │  Hybrid     │             │
│  │  Retrieval  │  │  Retrieval  │  │  Fusion     │             │
│  │  (Vector)   │  │  (BM25)     │  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Post-Processing                            │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Re-ranking │  │  Filtering  │  │  Context    │             │
│  │  (Cross-    │  │  (Metadata) │  │  Building   │             │
│  │   Encoder)  │  │             │  │             │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Generation Layer                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Prompt     │  │  LLM        │  │  Response   │             │
│  │  Building   │  │  Inference  │  │  Parsing    │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 Core Components

### 1. Document Processing

#### Chunking Strategies

**Fixed-Size Chunking:**
```python
def chunk_by_size(text: str, chunk_size: int = 512, overlap: int = 50):
    """Split text into fixed-size chunks with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
```

**Semantic Chunking:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " "]  # Split by semantic boundaries
)

chunks = text_splitter.split_text(document)
```

**Best Practices:**
- Chunk size: 256-512 tokens (balance context vs. precision)
- Overlap: 50-100 tokens (maintain coherence)
- Split by semantic boundaries (paragraphs, sections)

### 2. Embedding Models

**Popular Choices:**

| Model | Dimensions | Max Tokens | Speed | Quality |
|-------|------------|------------|-------|---------|
| text-embedding-ada-002 | 1536 | 8191 | Fast | High |
| all-MiniLM-L6-v2 | 384 | 512 | Very Fast | Good |
| bge-large-en-v1.5 | 1024 | 512 | Medium | Very High |
| m3e-base | 768 | 512 | Fast | Good (multilingual) |

**Selection Criteria:**
- **Quality:** bge-large for best accuracy
- **Speed:** all-MiniLM for fastest
- **Multilingual:** m3e-base for multiple languages
- **Cost:** Self-hosted models for high volume

### 3. Vector Databases

**Comparison:**

| Database | Type | Best For | Scale |
|----------|------|----------|-------|
| **Chroma** | Embedded | Development | <1M vectors |
| **FAISS** | Library | Local, fast | <10M vectors |
| **Pinecone** | Managed | Production | 100M+ vectors |
| **Weaviate** | Self-hosted | Feature-rich | 10M+ vectors |
| **Qdrant** | Self-hosted | Performance | 10M+ vectors |

**Usage Example:**
```python
import chromadb

# Initialize
client = chromadb.Client()
collection = client.create_collection("documents")

# Add vectors
collection.add(
    documents=["Document 1", "Document 2"],
    embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
    ids=["doc1", "doc2"],
    metadatas=[{"source": "wiki"}, {"source": "docs"}]
)

# Query
results = collection.query(
    query_embeddings=[[0.15, 0.25, ...]],
    n_results=5
)
```

### 4. Retrieval Strategies

#### Dense Retrieval

```python
# Vector similarity search
query_embedding = embedding_model.encode(query)
results = vector_db.search(query_embedding, k=10)
```

#### Sparse Retrieval

```python
# BM25 keyword search
from rank_bm25 import BM25Okapi

bm25 = BM25Okapi(tokenized_documents)
scores = bm25.get_scores(tokenized_query)
```

#### Hybrid Retrieval

```python
def hybrid_search(query, dense_results, sparse_results, alpha=0.5):
    """Combine dense and sparse retrieval using Reciprocal Rank Fusion."""
    # Reciprocal Rank Fusion
    fused_scores = {}
    
    for i, doc in enumerate(dense_results):
        fused_scores[doc.id] = fused_scores.get(doc.id, 0) + 1 / (i + 1) * alpha
    
    for i, doc in enumerate(sparse_results):
        fused_scores[doc.id] = fused_scores.get(doc.id, 0) + 1 / (i + 1) * (1 - alpha)
    
    # Sort by fused score
    ranked = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked[:10]
```

### 5. Re-ranking

```python
from sentence_transformers import CrossEncoder

# Initialize re-ranker
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Re-rank top 50 results
pairs = [[query, doc.text] for doc in retrieved_docs[:50]]
scores = reranker.predict(pairs)

# Sort by score
ranked_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
top_k = ranked_docs[:10]
```

### 6. Prompt Building

```python
def build_rag_prompt(query: str, context: List[str]) -> str:
    """Build prompt for RAG generation."""
    context_text = "\n\n".join([
        f"[Document {i+1}]\n{doc}" 
        for i, doc in enumerate(context)
    ])
    
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.

Context:
{context_text}

Question: {query}

Answer the question based on the context above. If the answer cannot be found in the context, say "I don't have enough information to answer this question."

Answer:"""
    
    return prompt
```

---

## 🛠️ Implementation Guide

### Step 1: Setup

```bash
# Install dependencies
pip install chromadb sentence-transformers langchain openai
```

### Step 2: Initialize RAG System

```python
from src.rag import RAGSystem

rag = RAGSystem(
    embedding_model="BAAI/bge-large-en-v1.5",
    vector_db="chroma",
    llm="gpt-3.5-turbo",
    collection_name="knowledge_base"
)
```

### Step 3: Add Documents

```python
documents = [
    {
        "id": "doc1",
        "content": "Machine learning is a subset of artificial intelligence...",
        "metadata": {"source": "textbook", "topic": "ml"}
    },
    {
        "id": "doc2",
        "content": "Deep learning uses neural networks with multiple layers...",
        "metadata": {"source": "research_paper", "topic": "dl"}
    }
]

rag.add_documents(documents)
```

### Step 4: Query

```python
result = rag.query(
    query="What is the difference between ML and DL?",
    top_k=5,
    use_reranking=True
)

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Sources:")
for source in result.sources:
    print(f"  - {source.metadata['source']}: {source.text[:100]}...")
```

---

## ✅ Best Practices

### 1. Document Preparation

```python
# ✅ Good: Clean and chunk properly
def prepare_documents(raw_docs):
    cleaned = []
    for doc in raw_docs:
        # Remove special characters
        text = clean_text(doc["content"])
        # Split into chunks
        chunks = semantic_chunk(text, chunk_size=512)
        # Add metadata
        for chunk in chunks:
            cleaned.append({
                "id": f"{doc['id']}_{chunks.index(chunk)}",
                "content": chunk,
                "metadata": {**doc["metadata"], "chunk_index": chunks.index(chunk)}
            })
    return cleaned
```

### 2. Query Optimization

```python
# ✅ Good: Expand query with synonyms
def expand_query(query: str) -> str:
    synonyms = {
        "ML": "machine learning",
        "DL": "deep learning",
        "AI": "artificial intelligence",
        "LLM": "large language model"
    }
    
    expanded = query
    for abbr, full in synonyms.items():
        expanded = expanded.replace(abbr, f"{abbr} ({full})")
    
    return expanded
```

### 3. Context Management

```python
# ✅ Good: Compress and filter context
def optimize_context(retrieved_docs, max_tokens=2000):
    # Filter low-scoring docs
    filtered = [doc for doc in retrieved_docs if doc.score > 0.5]
    
    # Sort by relevance
    sorted_docs = sorted(filtered, key=lambda x: x.score, reverse=True)
    
    # Truncate to fit token limit
    context = []
    total_tokens = 0
    for doc in sorted_docs:
        doc_tokens = count_tokens(doc.text)
        if total_tokens + doc_tokens <= max_tokens:
            context.append(doc.text)
            total_tokens += doc_tokens
        else:
            # Truncate document
            remaining = max_tokens - total_tokens
            context.append(truncate_text(doc.text, remaining))
            break
    
    return context
```

### 4. Answer Quality

```python
# ✅ Good: Add confidence scoring
def calculate_confidence(answer: str, sources: List[Document]) -> float:
    # Factor 1: Source relevance
    avg_relevance = sum(doc.score for doc in sources) / len(sources)
    
    # Factor 2: Answer length (not too short)
    length_score = min(len(answer) / 100, 1.0)
    
    # Factor 3: Source diversity
    unique_sources = len(set(doc.metadata["source"] for doc in sources))
    diversity_score = min(unique_sources / 3, 1.0)
    
    # Weighted average
    confidence = (
        0.5 * avg_relevance +
        0.3 * length_score +
        0.2 * diversity_score
    )
    
    return confidence
```

---

## 🚀 Advanced Patterns

### 1. Multi-Hop RAG

```python
def multi_hop_rag(query: str, max_hops: int = 3):
    """Perform multi-hop retrieval for complex queries."""
    context = []
    current_query = query
    
    for hop in range(max_hops):
        # Retrieve
        results = rag.retrieve(current_query, k=3)
        context.extend(results)
        
        # Check if we have enough information
        if has_answer(results, query):
            break
        
        # Generate next query
        current_query = generate_follow_up(query, results)
    
    # Generate final answer
    return rag.generate(query, context)
```

### 2. Self-Query RAG

```python
def self_query_rag(query: str):
    """RAG that extracts filters from query."""
    # Extract metadata filters
    filters = extract_filters(query)
    # Example: "Documents from 2024 about AI" → {"year": 2024, "topic": "AI"}
    
    # Query with filters
    results = rag.retrieve(
        query=query,
        filters=filters,
        k=10
    )
    
    return rag.generate(query, results)
```

### 3. Ensemble RAG

```python
def ensemble_rag(query: str):
    """Combine multiple RAG systems."""
    # Query multiple retrievers
    results_dense = dense_retriever.search(query, k=20)
    results_sparse = sparse_retriever.search(query, k=20)
    results_graph = graph_retriever.search(query, k=20)
    
    # Combine results
    all_results = combine_results(results_dense, results_sparse, results_graph)
    
    # Re-rank
    reranked = reranker.rerank(query, all_results[:50])
    
    return rag.generate(query, reranked[:10])
```

---

## 📊 Evaluation

### RAG Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Retrieval Precision** | % of retrieved docs that are relevant | >80% |
| **Recall@K** | % of relevant docs in top K | >90% |
| **Answer Accuracy** | % of answers that are correct | >85% |
| **Faithfulness** | % of answers grounded in context | >95% |
| **Latency** | Time to generate answer | <3s |

### Evaluation Code

```python
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Evaluate RAG system
results = evaluate(
    dataset=eval_dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall
    ]
)

print(results)
```

---

## 🐛 Troubleshooting

### Issue: Irrelevant Retrieved Documents

**Solutions:**
1. Improve chunking (smaller chunks, better boundaries)
2. Use better embeddings (bge-large instead of default)
3. Add re-ranking with cross-encoder
4. Implement hybrid search

### Issue: Lost in the Middle

**Problem:** Model ignores middle content in long context

**Solutions:**
1. Put most relevant chunks first and last
2. Use shorter context (top 5 instead of top 10)
3. Summarize middle content

### Issue: Hallucinations

**Solutions:**
1. Add prompt instruction: "Answer only from context"
2. Include confidence scoring
3. Add source citations
4. Use smaller, focused context

---

## 📚 Related Resources

- [Vector Databases](vector-db.md) - Deep dive into vector DBs
- [Embedding Models](embeddings.md) - Embedding model comparison
- [Advanced RAG](advanced-rag.md) - Advanced patterns
- [RAG Evaluation](rag-evaluation.md) - Evaluation techniques

---

**Last Updated:** March 28, 2026  
**Level:** Intermediate  
**Estimated Read Time:** 30 minutes
