# 🧪 Exercise 4: Re-ranking Techniques and Effectiveness

## 🎯 Objective
Implement and evaluate different re-ranking approaches to understand their impact on result relevance.

## 📋 Prerequisites
- Understanding of retrieval methods (Exercise 3)
- Access to re-ranking implementations
- Sample dataset with ground truth relevance judgments

## 🧪 Exercise Tasks

### Task 1: Implement Re-ranking Pipeline
1. Locate the re-ranking service in `src/app/application/services/reranker.py`
2. Set up both cross-encoder and LLM-based re-ranking (if available)
3. Create a pipeline that retrieves initial results and then re-ranks them
4. Compare results before and after re-ranking

### Task 2: Compare Re-ranking Approaches
1. Test cross-encoder re-ranking on retrieved results
2. Compare with LLM-based re-ranking (if available)
3. Measure the shift in result rankings
4. Evaluate the computational cost vs. quality improvement

### Task 3: Evaluate Re-ranking Effectiveness
1. Create test queries with known relevant documents at different ranks
2. Measure Normalized Discounted Cumulative Gain (NDCG) before/after re-ranking
3. Assess the improvement in top-k relevance
4. Document scenarios where re-ranking is most beneficial

## 🛠️ Implementation Hints
```python
from src.adapters.rerank.cross_encoder import CrossEncoderReranker
from src.adapters.rerank.llm_reranker import LLMReranker

# Initialize re-rankers
cross_encoder_reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
llm_reranker = LLMReranker(llm_client=llm_client)

# Get initial retrieval results
initial_results = retrieve_initial_results(query, top_k=20)

# Apply cross-encoder re-ranking
cross_enc_reranked = cross_encoder_reranker.rerank(query, initial_results)

# Apply LLM re-ranking
llm_reranked = llm_reranker.rerank(query, initial_results)

# Evaluate effectiveness
def calculate_ndcg(ranked_list, ground_truth_relevance):
    # Implementation of NDCG calculation
    # ...
    pass

ndcg_before = calculate_ndcg(initial_results, ground_truth_relevance)
ndcg_cross_enc = calculate_ndcg(cross_enc_reranked, ground_truth_relevance)
ndcg_llm = calculate_ndcg(llm_reranked, ground_truth_relevance)

print(f"NDCG - Before: {ndcg_before}")
print(f"NDCG - Cross-Encoder: {ndcg_cross_enc}")
print(f"NDCG - LLM: {ndcg_llm}")
```

## 🧠 Reflection Questions
1. When is cross-encoder re-ranking most effective?
2. What are the cost-benefit trade-offs of LLM re-ranking?
3. How does re-ranking impact overall system latency?
4. For which types of queries is re-ranking most beneficial?

## 📊 Success Criteria
- Successfully implemented re-ranking pipeline
- Demonstrated improvement in result relevance after re-ranking
- Compared effectiveness of different re-ranking approaches
- Quantified the cost-benefit trade-offs of re-ranking

## 🚀 Challenge Extension
Implement a learning-to-rank approach that adapts the re-ranking model based on user feedback and evaluate its effectiveness over time.