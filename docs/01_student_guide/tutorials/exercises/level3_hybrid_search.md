# 🧪 Exercise 3: Implementing and Evaluating Hybrid Search

## 🎯 Objective
Implement and compare hybrid search approaches (vector + keyword) to understand their benefits over single-method retrieval.

## 📋 Prerequisites
- Understanding of embeddings and chunking (Exercises 1 & 2)
- Access to both vector and keyword search implementations
- Sample dataset with ground truth queries

## 🧪 Exercise Tasks

### Task 1: Implement Hybrid Search
1. Locate the hybrid search implementation in `src/app/application/services/fusion.py`
2. Implement both vector-only and keyword-only search functions
3. Combine results using Reciprocal Rank Fusion (RRF)
4. Compare the results of each approach

### Task 2: Evaluate Different Fusion Methods
1. Implement RRF (Reciprocal Rank Fusion) as described in the fusion service
2. Compare with simple weighted fusion
3. Test with various weight combinations
4. Measure the effectiveness of each approach

### Task 3: Analyze Retrieval Performance
1. Create a test set of queries with known relevant documents
2. Measure recall@K and precision@K for each method:
   - Vector-only search
   - Keyword-only search
   - Hybrid search (RRF)
3. Identify scenarios where each method excels
4. Document the improvement achieved by hybrid search

## 🛠️ Implementation Hints
```python
from src.app.application.services.fusion import rrf_fusion
from src.adapters.vector.qdrant_store import QdrantVectorStore
from src.adapters.persistence.postgres.keyword_store import KeywordStore

# Initialize search components
vector_store = QdrantVectorStore()
keyword_store = KeywordStore()

# Perform individual searches
vector_results = vector_store.search(query_embedding, top_k=10)
keyword_results = keyword_store.search(query_text, top_k=10)

# Fuse results using RRF
fused_results = rrf_fusion(vector_results, keyword_results)

# Evaluate performance
def evaluate_retrieval(retrieved_docs, ground_truth_docs):
    # Calculate recall, precision, etc.
    recall_at_k = len(set(retrieved_docs) & set(ground_truth_docs)) / len(ground_truth_docs)
    return recall_at_k

# Compare methods
recall_vector_only = evaluate_retrieval(vector_results, ground_truth_docs)
recall_keyword_only = evaluate_retrieval(keyword_results, ground_truth_docs)
recall_hybrid = evaluate_retrieval(fused_results, ground_truth_docs)

print(f"Recall - Vector Only: {recall_vector_only}")
print(f"Recall - Keyword Only: {recall_keyword_only}")
print(f"Recall - Hybrid: {recall_hybrid}")
```

## 🧠 Reflection Questions
1. In what scenarios does vector search outperform keyword search?
2. When does keyword search provide better results than vector search?
3. How does RRF address the limitations of each individual method?
4. What are the computational trade-offs of hybrid search?

## 📊 Success Criteria
- Successfully implemented hybrid search combining vector and keyword results
- Demonstrated improvement in retrieval metrics compared to single-method approaches
- Identified specific use cases where each method excels
- Quantified the improvement achieved by hybrid search

## 🚀 Challenge Extension
Experiment with different fusion methods (e.g., Densification, Query Likelihood) and compare their effectiveness against RRF.