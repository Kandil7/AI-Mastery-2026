# 🧠 Deep Dive: Hybrid Search & RRF Fusion

> Understanding the math and mechanics behind combining Vector and Keyword search.

---

## The Problem: Vector vs. Keyword

In a RAG system, neither Vector nor Keyword search is perfect:

### 1. Vector Search (Semantic)
*   **Strength**: Understands synonyms and intent ("How to fix it?" matches "Troubleshooting guide").
*   **Weakness**: Struggles with exact IDs, specific technical terms, or rare words (e.g., "Error code CR-404").

### 2. Keyword Search (Lexical)
*   **Strength**: Perfect matching for names, product codes, and specific terminology.
*   **Weakness**: Doesn't understand synonyms. If you search "salary" it won't find "compensation".

---

## The Solution: Hybrid Search

Hybrid search runs both queries in parallel and merges the results. However, merging them is difficult because their scores are on different scales:
*   **Qdrant scores**: Usually 0.0 to 1.0 (Cosine similarity).
*   **Postgres FTS scores**: Rank-based (usually much larger numbers).

**You cannot simply add them together.**

---

## 🚀 Reciprocal Rank Fusion (RRF)

RRF is a simple yet powerful algorithm to merge ranked lists without needing to normalize scores.

### The Formula

$$score(d) = \sum_{r \in R} \frac{1}{k + rank(d, r)}$$

Where:
*   $d$ is a document.
*   $R$ is the set of rankings (Vector and Keyword).
*   $k$ is a smoothing constant (usually 60).
*   $rank(d, r)$ is the position of document $d$ in ranking $r$ (1, 2, 3...).

### Why is it brilliant?
1.  **Normalization Free**: It only cares about the *order* of results, not the raw scores.
2.  **Smoothing ($k$)**: The constant $60$ prevents items at the very top from dominating too heavily.
3.  **High Agreement**: An item that is in the top 10 of BOTH lists will score much higher than an item that is #1 in only one list but missing from the other.

---

## 🛠️ Implementation in RAG Engine Mini

Our implementation in `src/application/services/fusion.py` follows this logic:

```python
def rrf_fusion(vector_hits, keyword_hits, k=60):
    scores = {}
    
    # Process vector hits
    for rank, hit in enumerate(vector_hits, start=1):
        scores[hit.id] = 1 / (k + rank)
        
    # Process keyword hits
    for rank, hit in enumerate(keyword_hits, start=1):
        if hit.id in scores:
            scores[hit.id] += 1 / (k + rank)
        else:
            scores[hit.id] = 1 / (k + rank)
            
    # Sort and return
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)
```

---

## 🎯 When to use RRF?

- ✅ Use RRF when you want a reliable, "out-of-the-box" hybrid search without tuning complex weights.
- ❌ Avoid simple RRF if one search provider is significantly more reliable than the other (in that case, use weighted additive fusion).

---

## 📚 Further Reading
- [Reciprocal Rank Fusion (Original Paper)](https://plg.uwaterloo.ca/~gvcormac/pyserini/RecursiveRankFusion.pdf)
- [Qdrant Hybrid Search Guide](https://qdrant.tech/articles/hybrid-search/)
