# Case Study 3: Personalized Recommender System for Streaming Platform

## Executive Summary

**Problem**: Streaming platform struggling with user engagement - 45% of users never watched beyond homepage.

**Solution**: Built personalized recommendation system using collaborative filtering + deep learning hybrid approach.

**Impact**: +32% watch time, +18% subscriber retention, $12M incremental revenue.

---

## Business Context

### Company Profile
- **Industry**: Video Streaming (similar to Netflix/Disney+)
- **Subscribers**: 8M active users
- **Content Library**: 15K titles (movies + TV shows)
- **Problem**: Generic homepage → poor content discovery → churn

### Key Challenges
1. **Cold Start**: New users have no watch history
2. **Catalog Diversity**: Need to balance popular vs niche content
3. **Latency**: Homepage must load in <500ms
4. **Scale**: 8M users × 15K items = 120B possible combinations

---

## Recommendation Approach

### Hybrid Architecture

```
┌─────────────── User Context ───────────────┐
│  Watch History  Device  Time  Location    │
└────────────────────┬────────────────────────┘
                     ↓
    ┌───────────────┴───────────────┐
    ↓                               ↓
Collaborative Filtering         Content-Based
(Matrix Factorization)          (Metadata + NLP)
    │                               │
    └────────┬──────────┬───────────┘
             ↓          ↓
       Candidate      Re-Ranking
       Generation     (Neural Net)
          (Fast)      (Personalized)
             │              │
             └──────┬───────┘
                    ↓
            Final Top-20 Recs
```

### Three-Stage Pipeline

**Stage 1: Candidate Generation** (retrieve 200 candidates, <50ms)
- Collaborative Filtering: User-based + Item-based
- Content-Based: Genre/category matching
- Trending: Popularity in last 7 days
- Continuation: Next episodes for in-progress shows

**Stage 2: Feature Enrichment** (<100ms)
- User features: demographics, watch patterns, device
- Item features: genre, cast, release year, ratings
- Context features: time of day, day of week
- Interaction features: user-item affinity scores

**Stage 3: Re-Ranking** (<200ms)
- Deep neural network scores all 200 candidates
- Final sort by predicted watch probability
- Diversity post-processing (avoid genre clustering)

**Total Latency**: p95 < 400ms ✅

---

## Model Details

### Collaborative Filtering (Matrix Factorization)

**Approach**: SVD (Singular Value Decomposition)

```python
# User-Item matrix R (8M×15K, very sparse)
# Decompose: R ≈ U × Σ × V^T
# U: user embeddings (8M × 256)
# V: item embeddings (15K × 256)

# Predict rating: r_ui = user_embedding_u · item_embedding_i
```

**Training**:
- Implicit feedback (watch event = 1, no watch = 0)
- Weighted matrix factorization (longer watch time → higher weight)
- Negative sampling (randomly sample unwatched items)

**Performance**:
- Recall@20: 0.28 (of next 5 watched items, 28% in top 20)
- Training time: 4 hours on 16 CPU cores

### Content-Based Filtering

**Features**:
- **Structured**: Genre, cast, director, year, rating
- **Unstructured**: Title + description embeddings (BERT)

```python
# Title/description embedding
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
title_embedding = model.encode("Stranger Things: Sci-fi horror series...")

# Cosine similarity to user's historical watches
similar_items = cosine_similarity(title_embedding, user_favorite_embeddings)
```

**Performance**:
- Handles cold-start (new items immediately have embeddings)
- Recall@20: 0.19 (lower than collaborative, but complements it)

### Deep Re-Ranking Model

**Architecture**: Two-tower neural network

```
User Tower                    Item Tower
    ↓                             ↓
[Embedding Layers]        [Embedding Layers]
    ↓                             ↓
[Dense 256 → 128]         [Dense 256 → 128]
    ↓                             ↓
    └────────── Dot Product ──────┘
                    ↓
            Watch Probability
```

**Input Features**:
- User: ID embedding, demographics, watch time distribution by genre
- Item: ID embedding, genre, recency, popularity
- Context: hour of day, device type
- Interaction: collaborative score, content score

**Training**:
- Positive: Watched >80% of video
- Negative: Shown but not clicked, or clicked but watched <20%
- Loss: Binary cross-entropy
- Batch size: 512, 10 epochs

**Performance**:
- AUC: 0.76 (predicting watch vs no-watch)
- Significantly better than collaborative filtering alone for ranking

---

## Production Architecture

### Offline Components (Batch Processing)

```
Daily Airflow DAG:
  1. Extract watch events from data warehouse (Snowflake)
  2. Train Matrix Factorization model (Spark)
  3. Compute item embeddings for new content (BERT)
  4. Pre-compute candidate sets per user (store in Redis)
  5. Update re-ranking model weekly
```

**Storage**:
- User embeddings: Redis (8M × 256 floats × 4 bytes = 8.2GB)
- Item embeddings: Redis (15K × 256 floats = 15MBSmall enough for in-memory)
- Pre-computed candidates: Cassandra (8M users × 200 candidates × 4 bytes = 6.4GB)

### Online Serving (Real-Time)

```python
@app.get("/recommendations/{user_id}")
async def get_recommendations(user_id: int, device: str, time: datetime):
    # 1. Retrieve pre-computed candidates (50ms)
    candidates = await cassandra.get_candidates(user_id)  # 200 items
    
    # 2. Enrich with real-time context (30ms)
    user_embedding = await redis.get(f"user_emb:{user_id}")
    item_embeddings = await redis.mget([f"item_emb:{item_id}" for item_id in candidates])
    
    # 3. Re-rank with neural net (120ms)
    features = create_features(user_id, candidates, device, time)
    scores = reranking_model.predict(features)
    
    # 4. Sort and diversify (10ms)
    ranked = sort_by_score(candidates, scores)
    final = apply_diversity(ranked, top_k=20)
    
    return {"recommendations": final}
```

**Latency Breakdown** (p95):
- Candidate retrieval: 55ms
- Feature fetching: 32ms
- Re-ranking: 128ms
- Post-processing: 12ms
- **Total**: 227ms ✅ (target: <400ms)

---

## Results & Impact

### A/B Test Results (4 weeks, 800K users)

| Metric | Control | Treatment | Improvement |
|--------|---------|-----------|-------------|
| **Click-Through Rate** | 18.2% | 24.1% | **+32%** |
| **Watch Time (hr/user/week)** | 8.4 | 11.1 | **+32%** |
| **Session Duration** | 42 min | 51 min | **+21%** |
| **Content Diversity** | 2.3 genres/week | 3.1 genres/week | **+35%** |
| **7-Day Retention** | 68% | 73% | **+5pp** |
| **30-Day Retention** | 82% | 85% | **+3pp** |

### Business Impact (12 months)

**Revenue**:
- Watch time increase → lower churn → more subscribers
- Estimated subscriber lift: +120K (from improved retention)
- Revenue impact: 120K × $12/month × 12 months = **$17.3M**
- Engineering cost: $500K (team of 5 for 4 months)
- Net benefit: **$16.8M**

**Content Strategy**:
- Identified underutilized catalog (30% of content had <1% views)
- Recommendations surfaced niche content → better ROI on content acquisition

---

## Handling Cold-Start Problems

### New Users (No Watch History)

**1. Onboarding Quiz**:
- "Pick 5 titles you like" during signup
- Instant collaborative filtering based on similar users

**2. Demographic-Based**:
- Age, country → default genre preferences
- Example: Users 18-24 in US → lean towards action/comedy

**3. Popularity Fallback**:
- Show trending content until 5+ watches

**Performance**: Even for cold users, CTR = 15% (vs 24% for warm users)

### New Content (No Watch History)

**1. Content-Based**:
- BERT embeddings available immediately
- Recommend to users who watched similar genres/cast

**2. Exploration Boost**:
- New releases get +0.2 score boost for first 7 days
- Ensures new content gets initial visibility

**3. Monitoring**:
- Track "time-to-1000-views" for new content
- Alert if not reached within 48 hours (may indicate metadata issue)

---

## Diversity & Fairness

### Problem: Filter Bubble

Without intervention, model creates "echo chambers":
- Action fans only see action
- Reduces serendipity, limits content ROI

### Solution: Diversity Post-Processing

```python
def apply_diversity(ranked_items: List[int], top_k: int = 20) -> List[int]:
    """
    Ensure genre diversity in final recommendations.
    Max 5 items per genre in top 20.
    """
    selected = []
    genre_counts = {}
    
    for item in ranked_items:
        genre = get_genre(item)
        
        if genre_counts.get(genre, 0) < 5:
            selected.append(item)
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        if len(selected) == top_k:
            break
    
    return selected
```

**Impact**:
- Genres per week: 2.3 → 3.1 (+35%)
- User feedback: "Discovered content I wouldn't have found"

### Fairness: Content Creator Exposure

**Problem**: Popular content gets 80% of views (Pareto principle)

**Solution**: Explore/Exploit Balance
- 80% of recommendations: Exploit (highest predicted watch probability)
- 20% of recommendations: Explore (lower-ranked but diverse)

**Result**: Long-tail content views increased 40%

---

## Model Monitoring

### Metrics Tracked

**Real-Time** (every hour):
- Latency (p50, p95, p99)
- Error rate (failed recommendation requests)
- CTR (click-through rate)
- Diversity (unique genres in top 20)

**Batch** (daily):
- Recall@20, NDCG@20 (using next-day watches as ground truth)
- Coverage (% of catalog recommended at least once)
- Popularity bias (avg item popularity rank)

### Alerts

- **Performance**: p95 latency > 400ms for 10 minutes
- **Model Drift**: CTR drops >10% week-over-week
- **Coverage**: <70% of catalog recommended in 7 days
- **Bias**: >50% of рекомendations from top 100 most popular items

### Dashboard

```python
# Daily model quality report
metrics = {
    'recall@20': 0.28,
    'ndcg@20': 0.34,
    'ctr': 0.241,
    'diversity': 3.1,  # Avg genres per user per week
    'coverage_7d': 0.82,  # 82% of catalog recommended
    'latency_p95': 227  # ms
}

# Alert if deviations from baseline
for metric, value in metrics.items():
    if abs(value - baseline[metric]) > threshold[metric]:
        send_alert(f"{metric} drift detected: {value} vs {baseline[metric]}")
```

---

## Lessons Learned

### What Worked

1. **Hybrid > Pure Collaborative**:
   - Collaborative alone: Recall@20 = 0.28
   - + Content-based: Recall@20 = 0.35 (+25%)
   - + Deep re-ranking: NDCG@20 = 0.34 → 0.41 (+21%)

2. **Pre-Computation Critical for Latency**:
   - Computing candidates online: 800ms (too slow)
   - Pre-compute daily, store in Cassandra: 55ms ✅

3. **Diversity Post-Processing Essential**:
   - Without: Users plateau at 2.3 genres/week
   - With: 3.1 genres/week → better engagement

### What Didn't Work

1. **Graph Neural Networks**:
   - Tried GNN on user-item interaction graph
   - Training time: 18 hours (vs 4 hours for MF)
   - Recall improvement: +2% (not worth complexity)

2. **Real-Time Model Updates**:
   - Attempted online learning (update embeddings after each watch)
   - Embeddings became unstable, degraded quality
   - Daily batch updates sufficient

3. **Session-Based RNNs**:
   - RNN to model within-session behavior
   - Slight improvement (NDCG +0.02) but 3x latency
   - Abandoned for simpler re-rankning network

---

## Technical Implementation

### Matrix Factorization Training

```python
import implicit  # Fast collaborative filtering library

# Create user-item matrix
from scipy.sparse import coo_matrix
user_ids = np.array([...])  # 8M entries
item_ids = np.array([...])
weights = np.array([...])  # Watch time / max_watch_time

R = coo_matrix((weights, (user_ids, item_ids)), shape=(8_000_000, 15_000))

# Train ALS (Alternating Least Squares)
model = implicit.als.AlternatingLeastSquares(
    factors=256,
    regularization=0.01,
    iterations=15,
    use_gpu=False  # CPU with 16 cores faster for our scale
)

model.fit(R.T.tocsr())  # Transpose because library expects item-user

# Get user embedding
user_embedding = model.user_factors[user_id]

# Get top 200 recommendations
recommendations, scores = model.recommend(
    user_id,
    R[user_id],
    N=200,
    filter_already_liked_items=True
)
```

### Deep Re-Ranking Model (PyTorch)

```python
import torch
import torch.nn as nn

class TwoTowerRanker(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=128):
        super().__init__()
        
        # User tower
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_features = nn.Sequential(
            nn.Linear(embedding_dim + 10, 256),  # +10 for demographics/context
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Item tower
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.item_features = nn.Sequential(
            nn.Linear(embedding_dim + 5, 256),  # +5 for genre/recency/popularity
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
    
    def forward(self, user_ids, item_ids, user_feats, item_feats):
        # User tower
        user_emb = self.user_embedding(user_ids)
        user_input = torch.cat([user_emb, user_feats], dim=1)
        user_vec = self.user_features(user_input)
        
        # Item tower
        item_emb = self.item_embedding(item_ids)
        item_input = torch.cat([item_emb, item_feats], dim=1)
        item_vec = self.item_features(item_input)
        
        # Dot product + sigmoid
        scores = torch.sigmoid((user_vec * item_vec).sum(dim=1))
        return scores
```

---

## Next Steps

### Q1 2026
- [ ] Add sequential models (RNN/Transformer) for session-based recommendations
- [ ] Incorporate social signals (what friends are watching)
- [ ] Multi-objective optimization (balance watch time + finishing rate)

### Q2 2026
- [ ] Causal models to understand *why* users watch (not just predict)
- [ ] Reinforcement learning to optimize long-term engagement
- [ ] Cross-platform recommendations (web + mobile + TV)

---

## Conclusion

This recommendation system demonstrates ML at scale:
- **Hybrid Approach**: Collaborative + content-based + deep learning
- **Production-Ready**: <400ms latency, 8M users, 15K items
- **Impactful**: +32% watch time, +18% retention, $17M revenue

**Key Takeaway**: Combining multiple signals (collaborative, content, context) and optimizing for business metrics (not just accuracy) drives real impact.

---

**Implementation**: See `src/ml/recommender.py` and `notebooks/case_studies/recommender_system.ipynb`
