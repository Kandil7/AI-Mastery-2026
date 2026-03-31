# 🎓 Educational Layer - AI Engineering Curriculum

Welcome to the complete AI Engineering Curriculum for RAG Engine Mini! This curriculum is designed to take you from beginner to expert in production-ready RAG systems, following the **White-Box Approach** where you understand the mathematics, implement from scratch, and consider production aspects throughout.

## 📚 Table of Contents

1. [Foundation Concepts](#foundation-concepts)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Implementation from Scratch](#implementation-from-scratch)
4. [Production Considerations](#production-considerations)
5. [Advanced Topics](#advanced-topics)
6. [Evaluation & Monitoring](#evaluation--monitoring)
7. [Troubleshooting & Debugging](#troubleshooting--debugging)
8. [Industry Best Practices](#industry-best-practices)

---

## Foundation Concepts

### 1. Understanding RAG Systems

**What is RAG?**
- Retrieval-Augmented Generation combines information retrieval with language model generation
- Addresses hallucination by grounding responses in actual documents
- Enables LLMs to access proprietary or up-to-date information

**Core Components:**
- **Retriever**: Finds relevant documents/chunks from knowledge base
- **Generator**: Creates responses based on retrieved information
- **Index**: Preprocessed document database for fast retrieval

**Mathematical Definition:**
```
P(y|x, D) = Σ_{d∈retrieve(x,D)} P(y|x,d) × relevance(d,x)
```
Where:
- x = user query
- D = document collection
- d = retrieved document
- y = generated response

### 2. Vector Embeddings

**Concept:**
- Convert text to dense vectors in high-dimensional space
- Similar texts have similar vector representations
- Distance metrics measure semantic similarity

**Common Embedding Models:**
- OpenAI: text-embedding-ada-002, text-embedding-3-small
- Sentence Transformers: all-MiniLM-L6-v2, all-mpnet-base-v2
- Local models: nomic-embed-text, mxbai-embed-large

**Distance Metrics:**
- Cosine similarity: `cos(θ) = (A·B)/(||A||×||B||)`
- Euclidean distance: `√Σ(Ai-Bi)²`
- Dot product: `A·B`

### 3. Hybrid Search

**Why Hybrid?**
- Vector search: Good for semantic similarity but misses exact matches
- Keyword search: Good for exact matches but misses synonyms
- Hybrid: Combines both approaches for optimal recall and precision

**RRF (Reciprocal Rank Fusion):**
```
score(doc) = Σ(1/(k + rank_i(doc)))
```
Where k is typically 60, and ranks come from different search methods

---

## Mathematical Foundations

### 1. Embedding Mathematics

**Tokenization Process:**
```
text → tokens → token_ids → embeddings
```

**Transformer Embeddings:**
- Input embeddings: `E = Σ(token_emb + pos_emb + seg_emb)`
- Attention mechanism: `Attention(Q,K,V) = softmax(QK^T/√d_k)V`
- Final representation: Concatenated layer outputs

**Similarity Calculations:**
- Cosine similarity: `sim(A,B) = (A·B)/(||A||₂ × ||B||₂)`
- Angular distance: `d = arccos(sim(A,B))/π`
- Euclidean distance: `d = ||A-B||₂`

### 2. Retrieval Mathematics

**TF-IDF (Term Frequency-Inverse Document Frequency):**
```
tfidf(t,d) = tf(t,d) × idf(t)
idf(t) = log(N/df(t))
```
Where:
- t = term
- d = document
- N = total documents
- df(t) = documents containing term t

**BM25 (Best Matching 25):**
```
score(D,Q) = Σ(wi ∈ Q) IDF(wi) × (tf(wi,D) × (k+1))/(tf(wi,D) + k × (1 - b + b × |D|/avgdl))
```

### 3. Cross-Encoder Reranking

**Cross-Encoder Model:**
- Takes query and document pair as input
- Outputs probability of relevance
- More accurate than bi-encoders but slower

**Mathematical Formulation:**
```
P(relevant|q,d) = sigmoid(W_classifier × f([q;d]) + b_classifier)
```
Where f is the transformer encoder output

---

## Implementation from Scratch

### 1. Basic Embedding Implementation

```python
import numpy as np
from typing import List

class BasicEmbedder:
    def __init__(self, vocab_size: int, embedding_dim: int):
        self.embedding_matrix = np.random.normal(0, 0.1, (vocab_size, embedding_dim))
    
    def encode(self, tokens: List[int]) -> np.ndarray:
        """Average pooling of token embeddings"""
        token_embeddings = self.embedding_matrix[tokens]
        return np.mean(token_embeddings, axis=0)
    
    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embeddings"""
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        return dot_product / (norm1 * norm2) if norm1 != 0 and norm2 != 0 else 0
```

### 2. Vector Store Implementation

```python
import heapq
from typing import List, Tuple

class BasicVectorStore:
    def __init__(self):
        self.vectors = []
        self.metadata = []
    
    def add_vector(self, vector: np.ndarray, meta: dict):
        self.vectors.append(vector)
        self.metadata.append(meta)
    
    def search(self, query_vector: np.ndarray, k: int = 10) -> List[Tuple[float, dict]]:
        """Find k most similar vectors using cosine similarity"""
        similarities = []
        for i, vec in enumerate(self.vectors):
            sim = self.cosine_similarity(query_vector, vec)
            similarities.append((sim, self.metadata[i]))
        
        # Return top-k results sorted by similarity
        return heapq.nlargest(k, similarities, key=lambda x: x[0])
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b) if norm_a != 0 and norm_b != 0 else 0
```

### 3. RRF Fusion Implementation

```python
def reciprocal_rank_fusion(results_list: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    """
    Fuse multiple ranked lists using Reciprocal Rank Fusion
    
    Args:
        results_list: List of ranked results [(doc_id, score), ...]
        k: Smoothing parameter (typically 60)
    
    Returns:
        Fused ranked list with combined scores
    """
    doc_scores = {}
    
    for results in results_list:
        for rank, (doc_id, _) in enumerate(results, 1):
            if doc_id not in doc_scores:
                doc_scores[doc_id] = 0
            doc_scores[doc_id] += 1.0 / (k + rank)
    
    # Sort by RRF score in descending order
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_docs
```

### 4. Basic Reranker Implementation

```python
import torch
from transformers import AutoTokenizer, AutoModel

class BasicCrossEncoder:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def predict(self, query: str, documents: List[str]) -> List[float]:
        """Score query-document pairs for relevance"""
        scores = []
        
        for doc in documents:
            # Tokenize query and document together
            inputs = self.tokenizer.encode_plus(
                query, doc, 
                max_length=512, 
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Take the CLS token representation
                logits = outputs.last_hidden_state[:, 0, :].mean(dim=1)
                # Convert to probability (this is simplified)
                score = torch.sigmoid(logits).item()
                scores.append(score)
        
        return scores
```

---

## Production Considerations

### 1. Scalability Patterns

**Horizontal Scaling:**
- Load balancers distribute requests across multiple API instances
- Stateful components (vector stores, databases) scaled separately
- Caching layers reduce load on expensive operations

**Batch Processing:**
- Process multiple documents simultaneously for efficiency
- Queue systems (Celery) handle long-running tasks
- Memory management prevents out-of-memory errors

**Caching Strategies:**
- Redis for frequently accessed embeddings
- LRU cache for recent queries
- Document-level caching for repeated requests

### 2. Security Measures

**Authentication & Authorization:**
- API keys for access control
- JWT tokens for session management
- Rate limiting to prevent abuse

**Data Privacy:**
- PII detection and redaction
- Encryption at rest and in transit
- Secure document upload/download

**Input Sanitization:**
- Validate file types and sizes
- Sanitize user queries to prevent injection
- Limit query complexity to prevent resource exhaustion

### 3. Monitoring & Observability

**Metrics Collection:**
- Request latency and throughput
- Error rates and types
- Resource utilization (CPU, memory, disk)

**Logging:**
- Structured logs with correlation IDs
- Performance metrics per operation
- Error tracking and alerting

**Health Checks:**
- Service availability monitoring
- Dependency health (database, vector store)
- Performance degradation alerts

---

## Advanced Topics

### 1. Query Expansion Techniques

**Zero-shot Prompting:**
```
Expand this query into related terms:
Query: "machine learning algorithms"
Expanded: "ML algorithms, neural networks, supervised learning, unsupervised learning, deep learning, classification, regression"
```

**Synonym Generation:**
- Use WordNet, ConceptNet, or embedding-based approaches
- Expand query with semantically similar terms
- Increase recall without sacrificing precision

### 2. Multi-Modal RAG

**Image Processing:**
- Extract text from images using OCR
- Generate image embeddings for visual similarity
- Combine text and visual features

**Table Understanding:**
- Parse structured data from documents
- Convert tables to searchable text
- Maintain structural relationships

### 3. Agentic RAG

**ReAct Pattern:**
- Reasoning + Acting in iterative cycles
- Tool usage for external information retrieval
- Self-correction based on intermediate results

**Planning & Execution:**
- Decompose complex queries into subtasks
- Execute multiple retrieval steps
- Synthesize results from multiple sources

---

## Evaluation & Monitoring

### 1. RAG-Specific Metrics

**Faithfulness:**
- Does the response align with retrieved context?
- Measured using entailment models
- Target: >90% faithfulness

**Answer Relevance:**
- Is the answer relevant to the query?
- Measured using similarity to query
- Target: >85% relevance

**Context Relevance:**
- Are retrieved documents relevant to query?
- Measured using human evaluation or models
- Target: >90% relevance

### 2. RAGAS Framework

**Components:**
- Faithfulness: Evaluates factual consistency
- Answer Relevance: Measures answer-query alignment
- Context Precision: Assesses retrieval quality
- Context Recall: Measures retrieval completeness

**Implementation:**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_precision, context_recall

dataset = Dataset.from_dict({
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truth": ground_truths
})

scores = evaluate(
    dataset = dataset,
    metrics=[
        faithfulness,
        answer_relevance,
        context_precision,
        context_recall,
    ],
)
```

### 3. Synthetic Data Generation

**Flywheel Approach:**
- Generate questions from documents using LLMs
- Create synthetic QA pairs for evaluation
- Continuously improve system with generated data

**Prompt for QA Generation:**
```
Generate 5 diverse questions based on this document:
Document: {document_content}
Questions:
1. {question_1}
2. {question_2}
...
```

---

## Troubleshooting & Debugging

### 1. Common Issues & Solutions

**Low Recall:**
- Symptoms: System says "I don't know" when answer exists
- Causes: Wrong chunk size, vector-only retrieval, poor embeddings
- Solutions: Enable hybrid search, adjust chunk size, improve embeddings

**High Latency:**
- Symptoms: Slow response times (>5 seconds)
- Causes: Large reranking batches, inefficient queries, network issues
- Solutions: Optimize batch sizes, add caching, profile bottlenecks

**Hallucinations:**
- Symptoms: Confident but incorrect answers
- Causes: Poor grounding, insufficient context, model bias
- Solutions: Strengthen retrieval, add fact-checking, use self-correction

### 2. Debugging Strategies

**Query Analysis:**
- Log query expansion results
- Track retrieval scores and rankings
- Monitor for query patterns causing issues

**Component Isolation:**
- Test retrieval independently
- Evaluate generation separately
- Profile individual pipeline components

**A/B Testing:**
- Compare different embedding models
- Test various chunk sizes
- Evaluate reranking effectiveness

---

## Industry Best Practices

### 1. Architecture Patterns

**Clean Architecture:**
- Separate business logic from infrastructure
- Use dependency inversion principle
- Maintain clear boundaries between layers

**Ports & Adapters:**
- Define interfaces for external dependencies
- Implement multiple adapters for flexibility
- Enable easy testing and mocking

**Event-Driven Design:**
- Handle document ingestion asynchronously
- Use message queues for long-running tasks
- Implement circuit breakers for resilience

### 2. Performance Optimization

**Caching Hierarchies:**
- L1: In-memory cache (LRU)
- L2: Redis cache for shared resources
- L3: Database query results

**Indexing Strategies:**
- Vector indices for semantic search
- Text indices for keyword search
- Composite indices for multi-field queries

**Resource Management:**
- Connection pooling for databases
- Memory-efficient processing pipelines
- Graceful degradation under load

### 3. Deployment & Operations

**Containerization:**
- Docker for consistent environments
- Kubernetes for orchestration
- Health checks for service discovery

**CI/CD Pipelines:**
- Automated testing for each commit
- Staging environment validation
- Blue-green deployments for zero downtime

**Monitoring & Alerting:**
- Real-time performance dashboards
- Anomaly detection for metrics
- Automated incident response

---

## 🎯 Learning Path

### Beginner Level (Weeks 1-2)
- Understand RAG fundamentals
- Implement basic vector search
- Learn about embeddings and similarity

### Intermediate Level (Weeks 3-4)
- Build hybrid search system
- Implement chunking strategies
- Add basic evaluation metrics

### Advanced Level (Weeks 5-6)
- Develop reranking capabilities
- Implement agentic patterns
- Add monitoring and observability

### Expert Level (Weeks 7-8)
- Optimize for production scale
- Implement advanced evaluation
- Master troubleshooting techniques

---

## 📚 Additional Resources

### Recommended Reading
- "Hands-On Large Language Models" by Rex Yap
- "Designing Machine Learning Systems" by Chip Huyen
- "Building Systems with the ChatGPT API" by AI21 Labs

### Online Courses
- Stanford CS224N: Natural Language Processing
- Hugging Face Course: Natural Language Processing
- Fast.ai: Practical Deep Learning for Coders

### Communities
- Papers with Code: Latest research implementations
- Reddit r/MachineLearning: Discussions and news
- Stack Overflow: Technical Q&A

---

## 🏆 Certification Requirements

To complete this curriculum and earn the "Production-Ready RAG Engineer" certification, you must:

1. Successfully implement all core components from scratch
2. Deploy a working RAG system with monitoring
3. Achieve >80% on RAGAS evaluation metrics
4. Demonstrate troubleshooting skills with common issues
5. Document your implementation decisions with ADRs

---

*This curriculum represents the state-of-the-art in 2026 RAG engineering. Technology evolves rapidly, so continue learning and adapting these principles to new developments.*