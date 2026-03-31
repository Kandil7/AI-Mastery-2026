# Knowledge Checks - Module 1: Adaptive Multimodal RAG

## Instructions

Answer the following questions to test your understanding of adaptive multimodal RAG concepts. Each question has a single best answer.

---

## Question 1: Modality Detection

**Which approach provides the MOST robust modality detection for ambiguous content?**

A) File extension matching only
B) Pattern-based keyword matching only  
C) Hybrid approach combining pattern matching with embedding similarity
D) Manual classification by users

<details>
<summary><strong>Click for Answer</strong></summary>

**Answer: C**

**Explanation:** A hybrid approach combining pattern matching (fast, interpretable) with embedding similarity (semantic understanding) provides the most robust detection. Pattern matching alone fails on ambiguous content, while embedding-based approaches can be computationally expensive. The hybrid approach uses pattern matching for clear cases and falls back to embeddings for ambiguous content.

</details>

---

## Question 2: Adaptive Routing

**In a rule-based router, what happens when multiple rules match a query?**

A) All matching rules are applied simultaneously
B) The first matching rule (by priority) is applied
C) The last matching rule is applied
D) Rules are averaged together

<details>
<summary><strong>Click for Answer</strong></summary>

**Answer: B**

**Explanation:** In a priority-based rule system, rules are evaluated in priority order (highest first). When a high-priority rule matches (typically priority >= 10 for "definitive" rules), it takes precedence and evaluation stops. This ensures that specific, confident rules override general fallback rules.

</details>

---

## Question 3: Reciprocal Rank Fusion

**What is the primary advantage of Reciprocal Rank Fusion (RRF) over score-based fusion?**

A) RRF is faster to compute
B) RRF doesn't require score normalization across different indexes
C) RRF produces more results
D) RRF works only with text data

<details>
<summary><strong>Click for Answer</strong></summary>

**Answer: B**

**Explanation:** RRF's key advantage is that it operates on rank positions rather than raw scores. This means it can combine results from different indexes that may have completely different score scales (e.g., one index scores 0-1, another scores 0-100). The formula `1/(k+rank)` naturally normalizes contributions regardless of the original score distribution.

</details>

---

## Question 4: Query Complexity

**Which query would have the HIGHEST complexity score?**

A) "What is the revenue?"
B) "Show me the Q1 and Q2 revenue comparison table with year-over-year growth percentages, but only for products that exceeded targets"
C) "Revenue table"
D) "Tell me about revenue"

<details>
<summary><strong>Click for Answer</strong></summary>

**Answer: B**

**Explanation:** Query B has the highest complexity because it:
- Is longer (more words)
- Has multiple parts (Q1 AND Q2, comparison AND growth)
- Contains constraints ("only for products that exceeded targets")
- Has multiple entities (Q1, Q2, products, targets)
- Specifies modality (table)

Complexity estimation considers length, vocabulary diversity, structure indicators, entity count, and constraints.

</details>

---

## Question 5: Diversity Optimization

**What is the effect of increasing the diversity factor (λ) in Maximal Marginal Relevance (MMR)?**

A) More diverse results, potentially less relevant
B) More relevant results, potentially less diverse
C) No change in results
D) Faster retrieval

<details>
<summary><strong>Click for Answer</strong></summary>

**Answer: B**

**Explanation:** In MMR, the formula is: `MMR = λ * relevance - (1-λ) * max_similarity`

- When λ = 1: Pure relevance ranking (no diversity consideration)
- When λ = 0: Pure diversity (ignores relevance)
- When λ = 0.7: Balanced (70% relevance, 30% diversity)

Increasing λ puts more weight on relevance and less on diversity, producing more relevant but potentially less diverse results.

</details>

---

## Question 6: Cross-Modal Scoring

**When a user queries "Show me the architecture diagram", how should cross-modal scoring adjust results?**

A) Decrease scores for image results
B) Increase scores for image results
C) Keep all scores unchanged
D) Only return image results

<details>
<summary><strong>Click for Answer</strong></summary>

**Answer: B**

**Explanation:** Cross-modal scoring detects modality signals in the query ("diagram", "show me") and boosts results that match the expected modality. Image results should receive a higher alignment score because they match the user's apparent intent for visual content. However, other modalities aren't excluded entirely—they're just ranked lower.

</details>

---

## Question 7: Embedding Strategies

**What is the main trade-off between separate vs. unified embedding spaces for multimodal RAG?**

A) Separate spaces are always better
B) Unified spaces are always better
C) Separate: optimal per-modality quality; Unified: cross-modal retrieval capability
D) No significant difference

<details>
<summary><strong>Click for Answer</strong></summary>

**Answer: C**

**Explanation:** 
- **Separate embedding spaces**: Each modality uses its optimal embedding model, maximizing retrieval quality within that modality. However, you cannot directly compare scores across modalities or do cross-modal retrieval (text query → image results).

- **Unified embedding spaces**: All modalities map to the same space (e.g., CLIP), enabling cross-modal retrieval and simpler fusion. However, embeddings may be suboptimal for specific modalities.

The choice depends on whether cross-modal retrieval is required.

</details>

---

## Question 8: Intent Classification

**Which intent best matches the query: "Fix the error in the database connection"?**

A) Informational
B) Navigational
C) Transactional
D) Troubleshooting

<details>
<summary><strong>Click for Answer</strong></summary>

**Answer: D**

**Explanation:** The query contains clear troubleshooting signals:
- "Fix" - indicates a problem needs solving
- "error" - explicit problem indicator
- The overall intent is to resolve an issue, not learn information or navigate somewhere

Troubleshooting queries should trigger retrieval strategies that prioritize recent content (bugs may be fixed) and practical solutions.

</details>

---

## Question 9: Production Considerations

**What is the recommended P95 latency target for production RAG retrieval?**

A) < 50ms
B) < 500ms
C) < 2000ms
D) < 10000ms

<details>
<summary><strong>Click for Answer</strong></summary>

**Answer: B**

**Explanation:** For production RAG systems, the recommended latency targets are:
- P50 (median): < 100ms
- P95: < 500ms
- P99: < 1000ms

These targets balance user experience (fast responses) with the computational cost of embedding generation, vector search, and reranking.

</details>

---

## Question 10: Hybrid Routing

**When should a hybrid router prefer ML-based routing over rule-based routing?**

A) Always prefer ML-based routing
B) Always prefer rule-based routing
C) When ML confidence exceeds a threshold (e.g., 0.7)
D) Never use both together

<details>
<summary><strong>Click for Answer</strong></summary>

**Answer: C**

**Explanation:** Hybrid routing combines the interpretability and control of rule-based routing with the pattern-learning capability of ML-based routing. The typical strategy is:

1. Always evaluate rule-based routing (provides fallback)
2. Get ML predictions and confidence scores
3. If ML confidence >= threshold (e.g., 0.7), use ML decision
4. Otherwise, use rule-based decision

This ensures ML is only trusted when confident, while rules provide reliable fallback behavior.

</details>

---

## Scoring

- **9-10 correct**: Excellent understanding! Ready for Module 2.
- **7-8 correct**: Good understanding. Review weak areas before proceeding.
- **5-6 correct**: Fair understanding. Re-read theory sections.
- **Below 5**: Needs improvement. Complete labs again and review theory.
