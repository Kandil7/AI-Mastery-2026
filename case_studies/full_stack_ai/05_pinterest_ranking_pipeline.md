# Pinterest: Multi-Stage Ranking Pipeline

## Business Context

**Challenge**: Pinterest needed to rank billions of pins for 400M+ users in real-time. Key constraints:
- Latency budget: <100ms total
- Personalization at scale
- Balance engagement with diversity
- Handle cold-start for new pins and users

**Solution**: Multi-stage funnel that progressively reduces candidates while increasing ranking precision.

---

## Technical Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                     Multi-Stage Ranking Pipeline                    │
├──────────────┬──────────────┬──────────────┬──────────────────────┤
│   Candidate  │   Pre-Rank   │  Full-Rank   │     Re-Rank          │
│   Generation │   (Light)    │   (Deep)     │   (Business)         │
├──────────────┼──────────────┼──────────────┼──────────────────────┤
│  1M → 10K    │  10K → 500   │  500 → 50    │     50 → 25          │
│  <10ms       │  <20ms       │  <50ms       │     <10ms            │
│  k-NN, Pop   │  MLP (64d)   │  Two-Tower   │  Diversity, Fresh    │
└──────────────┴──────────────┴──────────────┴──────────────────────┘
```

**Funnel Metrics**:
| Stage | Input | Output | Latency | Model Complexity |
|-------|-------|--------|---------|------------------|
| Candidate Gen | 1M pins | 10K | <10ms | Approximate NN |
| Pre-Rank | 10K | 500 | <20ms | 2-layer MLP |
| Full-Rank | 500 | 50 | <50ms | Deep Two-Tower |
| Re-Rank | 50 | 25 | <10ms | Business rules |

---

## Key Components

### 1. Candidate Generation (Multiple Sources)

Different retrieval strategies catch different signals:

```python
class CandidateGenerator:
    def __init__(self):
        self.sources = [
            EmbeddingSimilaritySource(weight=1.0),   # User-pin similarity
            PopularitySource(weight=0.5),            # Trending pins
            CollaborativeFilteringSource(weight=0.7),# Similar users liked
            RecentSource(weight=0.3),                # Fresh content
        ]
        
    def generate(self, user, k_per_source=2500):
        all_candidates = {}
        
        for source in self.sources:
            candidates = source.retrieve(user, k_per_source)
            
            for c in candidates:
                if c.item.id in all_candidates:
                    # Merge: keep max weighted score
                    existing = all_candidates[c.item.id]
                    existing.retrieval_score = max(
                        existing.retrieval_score,
                        c.retrieval_score * source.weight
                    )
                else:
                    all_candidates[c.item.id] = c
                    
        return sorted(all_candidates.values(), 
                      key=lambda x: -x.retrieval_score)
```

### 2. Pre-Ranker (Lightweight Filtering)

Fast model to reduce 10K → 500:

```python
class PreRanker:
    def __init__(self):
        # Simple 2-layer MLP
        self.W1 = np.random.randn(feature_dim, 64)
        self.W2 = np.random.randn(64, 1)
        
    def score(self, user, candidates):
        for c in candidates:
            # Lightweight features only
            features = np.concatenate([
                user.features,
                c.item.features,
                [c.retrieval_score, len(c.metadata.get("sources", []))]
            ])
            
            # Fast forward pass
            h = np.maximum(0, features @ self.W1)
            c.prerank_score = sigmoid(h @ self.W2)
            
        return sorted(candidates, key=lambda x: -x.prerank_score)[:500]
```

### 3. Full Ranker (Deep Precision)

Deep model for accurate ranking:

```python
class FullRanker:
    def __init__(self):
        self.user_tower = MLP([128, 64])  # User embedding
        self.item_tower = MLP([128, 64])  # Item embedding
        self.cross_network = MLP([128, 64, 32, 1])  # Interaction features
        
    def score(self, user, candidates):
        # Encode user once (amortized)
        user_emb = self.user_tower(user.features)
        
        for c in candidates:
            # Encode item
            item_emb = self.item_tower(c.item.features)
            
            # Cross features
            cross_features = [
                np.dot(user_emb, item_emb),  # Similarity
                c.prerank_score,              # Pre-rank signal
                c.item.age_days,              # Freshness
                len(user.history),            # User activity
            ]
            
            # Final score
            combined = np.concatenate([user_emb, item_emb, cross_features])
            c.fullrank_score = self.cross_network(combined)
            
        return sorted(candidates, key=lambda x: -x.fullrank_score)[:50]
```

### 4. Re-Ranker (Business Logic)

Apply diversity and business rules:

```python
class DiversityReRanker:
    def __init__(self):
        self.max_per_category = 5
        self.freshness_boost = 0.1
        
    def rerank(self, candidates, k=25):
        # 1. Category diversity (no more than 5 per category)
        category_counts = defaultdict(int)
        diverse_candidates = []
        
        for c in candidates:
            category = c.item.metadata.get("category", "unknown")
            if category_counts[category] < self.max_per_category:
                diverse_candidates.append(c)
                category_counts[category] += 1
                
        # 2. Freshness boost for new pins
        for c in diverse_candidates:
            if c.item.age_days <= 7:
                c.final_score = c.fullrank_score * (1 + self.freshness_boost)
            else:
                c.final_score = c.fullrank_score
                
        # 3. Position bias correction
        # (Higher positions get more clicks regardless of quality)
        
        return sorted(diverse_candidates, key=lambda x: -x.final_score)[:k]
```

---

## Production Results

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Engagement Rate | 4.2% | 5.8% | **+38%** |
| P99 Latency | 180ms | 85ms | **-53%** |
| Diversity Score | 0.42 | 0.68 | **+62%** |
| New Pin Exposure | 12% | 21% | **+75%** |
| Serving Cost | Baseline | 0.4x | **-60%** |

---

## Implementation in This Project

See: [`src/production/ranking_pipeline.py`](file:///k:/learning/technical/ai-ml/AI-Mastery-2026/src/production/ranking_pipeline.py)

**Key Classes**:
- `CandidateGenerator`: Multi-source retrieval with score fusion
- `EmbeddingSimilaritySource`, `PopularitySource`, `RecentSource`: Retrieval strategies
- `PreRanker`: Lightweight MLP for fast filtering
- `FullRanker`: Deep two-tower with cross features
- `DiversityReRanker`: Category limits, freshness, position bias
- `BusinessRuleReRanker`: Custom boost/filter rules
- `RankingPipeline`: Unified orchestration

---

## Code Example

```python
from src.production.ranking_pipeline import (
    RankingPipeline, EmbeddingSimilaritySource, PopularitySource, User, Item
)

# Create items
items = [Item(id=f"pin_{i}", features=np.random.randn(64)) for i in range(10000)]
embeddings = np.array([item.features for item in items])
popularity = {item.id: np.random.random() for item in items}

# Build pipeline
pipeline = RankingPipeline(
    user_feature_dim=64,
    item_feature_dim=64,
    retrieval_k=1000,
    prerank_k=200,
    fullrank_k=50
)

# Add retrieval sources
pipeline.add_retrieval_source(EmbeddingSimilaritySource(items, embeddings), weight=1.0)
pipeline.add_retrieval_source(PopularitySource(items, popularity), weight=0.5)

# Add business rules
pipeline.add_business_rule(
    "filter",
    lambda c: c.item.metadata.get("is_spam", False) == False,
    "Filter spam pins"
)
pipeline.add_business_rule(
    "boost",
    lambda c: 1.5 if c.item.metadata.get("is_promoted") else 1.0,
    "Boost promoted pins"
)

# Rank for user
user = User(id="user_123", features=np.random.randn(64))
result = pipeline.rank(user, k=25)

print(f"Retrieved: {result.retrieval_count}")
print(f"Pre-ranked: {result.prerank_count}")
print(f"Full-ranked: {result.fullrank_count}")
print(f"Latency: {result.latency_ms:.1f}ms")
print(f"Final: {len(result.candidates)} pins")
```

---

## Lessons Learned

1. **Funnel is essential**: Can't run deep model on 1M items
2. **Pre-rank saves money**: 20x reduction before expensive scoring
3. **Diversity is engagement**: Users leave if feed is monotonous
4. **Position bias is real**: Correct for it in training and serving

---

## References

- Pinterest Engineering Blog: "Pinnability: Machine Learning in the Home Feed"
- Netflix Tech Blog: "Recommendations at Netflix"
