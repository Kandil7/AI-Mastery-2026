# ADR-003: Cross-Encoder vs LLM Re-ranking

## Status
Accepted

## Context
Re-ranking is crucial for improving the relevance of retrieved documents in RAG systems. After initial retrieval from vector and keyword stores, we can re-rank the top candidates to improve the quality of the final results sent to the LLM.

Options for re-ranking include:
- **Cross-Encoder Models**: Specialized models that score query-document pairs
- **LLM-based Re-ranking**: Using LLMs to assess relevance
- **No Re-ranking**: Skipping re-ranking for performance

## Decision
Use local Cross-Encoder models as the default re-ranking approach, with LLM-based re-ranking as an option.

## Alternatives Considered
### Cross-Encoder Re-ranking
- Pros: Fast, efficient, purpose-built for re-ranking, good performance
- Cons: Requires GPU resources, additional model to manage

### LLM-based Re-ranking
- Pros: Flexible, can incorporate complex reasoning, no additional models
- Cons: Expensive, slow, inconsistent scoring

### No Re-ranking
- Pros: Fastest, lowest cost
- Cons: Potentially lower quality results

### Cross-Encoder (Primary Choice)
- Pros: Good balance of performance and quality, efficient inference
- Cons: Requires computational resources for model hosting

## Rationale
1. **Performance**: Cross-encoders are significantly faster than LLM-based approaches
2. **Cost**: Much more economical than using LLMs for re-ranking
3. **Quality**: Proven effectiveness for re-ranking tasks
4. **Control**: Deterministic, consistent results

## Consequences
### Positive
- Better performance than LLM re-ranking
- Lower operational costs
- Consistent, predictable behavior
- Faster response times

### Negative
- Requires additional computational resources
- Need to manage another model deployment
- Less flexible than LLM-based approaches

## Implementation
Cross-encoder models take the query and document as input and output a relevance score:

```
score = cross_encoder_model(query, document)
```

The documents are then re-ranked based on these scores.

## Validation
Cross-encoders have been extensively validated in information retrieval tasks and consistently outperform other approaches in terms of speed-quality tradeoffs for re-ranking.