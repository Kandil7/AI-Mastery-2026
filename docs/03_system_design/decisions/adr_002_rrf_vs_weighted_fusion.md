# ADR-002: RRF vs Weighted Fusion for Hybrid Search

## Status
Accepted

## Context
In hybrid search systems, we need to combine results from vector search (semantic) and keyword search (lexical). The effectiveness of our retrieval depends heavily on how well we combine these different ranking systems.

Common approaches include:
- **Weighted Score Fusion**: Linear combination of normalized scores
- **Reciprocal Rank Fusion (RRF)**: Non-linear fusion based on ranks
- **Learned Fusion**: ML-based combination of features

## Decision
Use Reciprocal Rank Fusion (RRF) as the default fusion method for hybrid search.

## Alternatives Considered
### Weighted Score Fusion
- Pros: Intuitive, customizable weights, easy to understand
- Cons: Requires parameter tuning, sensitive to score distribution differences

### Reciprocal Rank Fusion (RRF) (Chosen)
- Pros: Parameter-free, robust to score distribution differences, theoretically sound
- Cons: Less intuitive, fixed weighting scheme

### Learned Fusion
- Pros: Optimal for specific datasets, adaptive
- Cons: Requires training data, complex implementation, overfitting risk

## Rationale
1. **Parameter-Free**: RRF doesn't require tuning of weights, making it easier to deploy
2. **Robustness**: Works well regardless of the underlying score distributions
3. **Theoretical Foundation**: Based on probability theory, proven effectiveness
4. **Industry Adoption**: Widely adopted in search engines and RAG systems

## Consequences
### Positive
- No parameter tuning required
- Consistent performance across different domains
- Robust to score normalization issues
- Well-established technique with known properties

### Negative
- Less flexibility than weighted approaches
- May not be optimal for all datasets
- Less intuitive than simple weighted averaging

## Formula
```
RRF_score(d) = Σ(1 / (k + rank_i(d)))
```
Where k is a smoothing constant (typically 60), and rank_i(d) is the rank of document d in result set i.

## Validation
RRF has been extensively tested in information retrieval literature and performs comparably to tuned weighted approaches while requiring no parameter adjustment.