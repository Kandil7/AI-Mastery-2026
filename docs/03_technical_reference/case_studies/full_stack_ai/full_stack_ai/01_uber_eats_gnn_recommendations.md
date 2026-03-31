# Uber Eats: GNN-Based Restaurant Recommendations

## Business Context

**Challenge**: Uber Eats needed to recommend restaurants that users would not just click on, but actually order from repeatedly. Traditional methods struggled with:
- Sparse interaction data for new users/restaurants
- Distinguishing between "clicked once" vs "orders frequently"
- Cold-start for new restaurants in the marketplace

**Solution**: Graph Neural Network with custom loss functions that understand preference intensity.

---

## Technical Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  User-Restaurant│────▶│   GraphSAGE     │────▶│   Two-Tower     │
│  Bipartite Graph│     │   Embeddings    │     │   Ranker        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
  ┌──────────────┐      ┌──────────────┐       ┌──────────────┐
  │ Edge Weights │      │ Neighbor     │       │ Hinge Loss   │
  │ = order_count│      │ Aggregation  │       │ w/ Low-Rank  │
  └──────────────┘      └──────────────┘       │ Positives    │
                                               └──────────────┘
```

### Key Innovation: Low-Rank Positive Handling

Standard ranking losses treat all positives equally. Uber's innovation:

```python
def hinge_loss_with_low_rank_positives(scores, positive_mask, rank_weights, margin=1.0):
    """
    Penalizes not just negative > positive,
    but also "sometimes ordered" > "frequently ordered"
    """
    # Loss for positives vs negatives (standard)
    for pos_score, pos_weight in zip(positive_scores, positive_weights):
        for neg_score in negative_scores:
            hinge = max(0, margin - (pos_score - neg_score))
            total_loss += pos_weight * hinge
            
    # NEW: Loss for low-rank vs high-rank positives
    for i, (score_i, weight_i) in enumerate(zip(positive_scores, positive_weights)):
        for j, (score_j, weight_j) in enumerate(zip(positive_scores, positive_weights)):
            if weight_i > weight_j:  # i should rank higher
                hinge = max(0, margin - (score_i - score_j))
                total_loss += (weight_i - weight_j) * hinge
```

### GraphSAGE for Inductive Learning

Why GraphSAGE instead of traditional GCN?

| Aspect | GCN | GraphSAGE |
|--------|-----|-----------|
| **New nodes** | Requires retraining | Inductive: works immediately |
| **Scalability** | Full graph in memory | Mini-batch friendly |
| **Cold-start** | Poor | Can embed from features |

---

## Production Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Order Conversion Rate | 12.3% | 15.7% | **+27.6%** |
| Repeat Order Rate | 34% | 41% | **+20.6%** |
| New Restaurant Discovery | 8% | 14% | **+75%** |
| Recommendation Latency | 45ms | 38ms | -15.5% |

---

## Implementation in This Project

See: [`src/ml/gnn_recommender.py`](file:///k:/learning/technical/ai-ml/AI-Mastery-2026/src/ml/gnn_recommender.py)

**Key Classes**:
- `BipartiteGraph`: User-item graph with weighted edges
- `GraphSAGELayer`: Neighbor aggregation with mean/pool/LSTM
- `TwoTowerRanker`: Dual-tower scoring architecture
- `RankingLoss`: BPR, Hinge with low-rank positives, InfoNCE
- `ColdStartHandler`: Demographic + content-based fallbacks

---

## Code Snippet: Training Loop

```python
from src.ml.gnn_recommender import GNNRecommender, TwoTowerRanker, RankingLoss

# Build graph from order data
graph = BipartiteGraph()
for user_id, orders in user_orders.items():
    graph.add_node(user_id, NodeType.USER, user_features[user_id])
    for restaurant_id, order_count in orders.items():
        if restaurant_id not in graph.nodes:
            graph.add_node(restaurant_id, NodeType.ITEM, restaurant_features[restaurant_id])
        graph.add_edge(user_id, restaurant_id, weight=order_count)

# Generate embeddings
recommender = GNNRecommender(feature_dim=128, embedding_dim=256, num_layers=2)
user_embs, item_embs = recommender.generate_embeddings(graph, training=True)

# Train with low-rank positive loss
for batch in dataloader:
    scores = ranker.score(batch.user_features, batch.item_features)
    loss = RankingLoss.hinge_loss_with_low_rank_positives(
        scores, batch.positive_mask, batch.order_counts
    )
```

---

## Lessons Learned

1. **Edge weights matter**: Encoding interaction strength (not just binary) improved results by 15%
2. **Inductive > Transductive**: Daily new restaurants required real-time embedding capability
3. **Multi-loss objectives**: Combining BPR + low-rank positive loss outperformed either alone
4. **Cold-start blending**: New users get 70% popular + 30% demographic-based recommendations

---

## References

- [Uber Engineering: Restaurant Recommendations with GNNs](https://eng.uber.com/)
- Hamilton et al., "Inductive Representation Learning on Large Graphs" (GraphSAGE paper)
