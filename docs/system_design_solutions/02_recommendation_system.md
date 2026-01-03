# System Design: Recommendation System at Scale

## Problem Statement

Design a recommendation system for an e-commerce platform with:
- **100M users**, **10M products**
- **1B+ historical interactions** (clicks, purchases, ratings)
- **Real-time personalization** (<100ms p95 latency)
- **Cold start handling** (new users/products)
- Support both **collaborative filtering** and **content-based** recommendations

---

## High-Level Architecture

```
┌────────────────────────────────────────────────────────────┐
│                    User Request                           │
└──────────────────────┬─────────────────────────────────────┘
                       │
                       ▼
        ┌──────────────────────────────┐
        │   API Gateway (Kong/NGINX)   │
        └──────────────┬─────────────────┘
                       │
                       ▼
    ┌──────────────────────────────────────┐
    │  Recommendation Service (FastAPI)    │
    │  - Feature engineering                │
    │  - Model serving                      │
    │  - A/B testing logic                  │
    └──────┬───────────────────────────────┘
           │
           ├────────────┬─────────────┬──────────┐
           ▼            ▼             ▼          ▼
    ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐
    │   MF     │ │Content-  │ │ Deep    │ │ Bandit  │
    │  Model   │ │  Based   │ │Learning │ │(Explore)│
    │(ALS/SVD) │ │(TF-IDF)  │ │(2-Tower)│ │         │
    └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬────┘
         │            │             │            │
         └────────────┴─────────────┴────────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Ensemble/Reranking  │
            │  (Business Rules)    │
            └──────────┬───────────┘
                       │
                       ▼
            ┌──────────────────────┐
            │  Cache (Redis)       │
            │  + Precomputed Recs  │
            └──────────────────────┘
```

---

## Component Deep Dive

### 1. Data Layer

**Data Schema**:

```sql
-- User table
CREATE TABLE users (
    user_id BIGINT PRIMARY KEY,
    created_at TIMESTAMP,
    demographics JSONB,  -- age, gender, location
    embedding VECTOR(128)  -- Learned user embedding
);

-- Products table
CREATE TABLE products (
    product_id BIGINT PRIMARY KEY,
    category VARCHAR(100),
    price DECIMAL(10,2),
    features JSONB,  -- brand, attributes
    embedding VECTOR(128),
    popularity_score FLOAT
);

-- Interactions table (partitioned by date)
CREATE TABLE interactions (
    user_id BIGINT,
    product_id BIGINT,
    interaction_type VARCHAR(20),  -- view, click, purchase
    timestamp TIMESTAMP,
    context JSONB  -- device, session_id, etc
) PARTITION BY RANGE (timestamp);

CREATE INDEX idx_user_interactions ON interactions(user_id, timestamp DESC);
CREATE INDEX idx_product_interactions ON interactions(product_id, timestamp DESC);
```

**Data Pipeline**:
```
User Interaction
       │
       ▼
┌──────────────┐
│ Kafka Stream │  (Real-time)
└──────┬───────┘
       │
       ├──► Feature Store (Feast)
       ├──► OLAP (ClickHouse) - Analytics
       └──► Model Training (Spark)
```

---

### 2. Model Architecture - Multi-Strategy Approach

#### A. Matrix Factorization (Baseline)

**ALS (Alternating Least Squares)** for implicit feedback:

```python
from pyspark.ml.recommendation import ALS

# Train on historical interactions
als = ALS(
    rank=128,               # Embedding dimension
    maxIter=10,
    regParam=0.1,
    implicitPrefs=True,     # Clicks = implicit feedback
    alpha=40,               # Confidence scaling
    userCol="user_id",
    itemCol="product_id",
    ratingCol="score"       # Normalized interaction score
)

model = als.fit(interaction_df)

# Extract embeddings
user_embeddings = model.userFactors  # (100M x 128)
item_embeddings = model.itemFactors  # (10M x 128)
```

**Pros**:
- Fast inference (<10ms)
- Explainable (dot product similarity)
- Works well for established users

**Cons**:
- Cold start problem
- Doesn't use item features

---

#### B. Content-Based Filtering

**Product Feature Engineering**:

```python
class ProductEmbedder:
    def __init__(self):
        self.text_encoder = SentenceTransformer()
        self.category_encoder = OneHotEncoder()
        
    def embed(self, product: Product) -> np.ndarray:
        # Text features (title + description)
        text_emb = self.text_encoder.encode(product.description)
        
        # Categorical features
        cat_emb = self.category_encoder.transform([[
            product.category,
            product.brand,
            product.price_bucket
        ]])
        
        # Concatenate
        return np.concatenate([text_emb, cat_emb.flatten()])
```

**Recommendation Logic**:
```python
def recommend_content_based(user_history: List[int], top_k: int = 10):
    # Get user's purchase history embeddings
    purchased_items = [product_embeddings[pid] for pid in user_history]
    
    # User profile = average of purchased items
    user_profile = np.mean(purchased_items, axis=0)
    
    # Find similar products (cosine similarity)
    similarities = cosine_similarity([user_profile], product_embeddings)
    top_indices = np.argsort(-similarities[0])[:top_k]
    
    return top_indices
```

**Use Case**: Cold start for new users (no interaction history)

---

#### C. Two-Tower Deep Learning Model

**Architecture**:

```
User Features                    Item Features
     │                                │
     ▼                                ▼
┌─────────┐                      ┌─────────┐
│  User   │                      │  Item   │
│  Tower  │                      │  Tower  │
│  (DNN)  │                      │  (DNN)  │
└────┬────┘                      └────┬────┘
     │                                │
     ▼                                ▼
  [128-dim]                        [128-dim]
     └────────────┬───────────────────┘
                  │
                  ▼
          Dot Product / Cosine Sim
                  │
                  ▼
            Probability(click)
```

**PyTorch Implementation**:

```python
import torch
import torch.nn as nn

class TwoTowerModel(nn.Module):
    def __init__(self, user_features, item_features, embedding_dim=128):
        super().__init__()
        
        # User tower
        self.user_tower = nn.Sequential(
            nn.Linear(user_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
        # Item tower
        self.item_tower = nn.Sequential(
            nn.Linear(item_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, user_x, item_x):
        user_emb = self.user_tower(user_x)
        item_emb = self.item_tower(item_x)
        
        # Cosine similarity
        score = F.cosine_similarity(user_emb, item_emb, dim=1)
        return torch.sigmoid(score)
    
    def get_embeddings(self, x, tower='user'):
        if tower == 'user':
            return self.user_tower(x)
        return self.item_tower(x)
```

**Training**:
- **Loss**: Binary cross-entropy (click vs no-click)
- **Negative sampling**: 4 negatives per positive
- **Batch size**: 2048 (for efficiency)

**Advantages**:
- Uses rich features (demographics, context)
- Handles cold start naturally
- Can update embeddings incrementally

---

### 3. Ensemble & Reranking

**Combine Multiple Models**:

```python
class EnsembleRecommender:
    def __init__(self):
        self.weights = {
            'mf': 0.4,        # Matrix factorization
            'content': 0.2,   # Content-based
            'two_tower': 0.4  # Deep learning
        }
        
    def recommend(self, user_id: int, top_k: int = 10):
        # Get candidates from each model
        mf_scores = self.mf_model.predict(user_id, k=50)
        content_scores = self.content_model.predict(user_id, k=50)
        two_tower_scores = self.two_tower_model.predict(user_id, k=50)
        
        # Aggregate scores (weighted sum)
        all_items = set(mf_scores.keys()) | set(content_scores.keys()) | set(two_tower_scores.keys())
        
        final_scores = {}
        for item in all_items:
            final_scores[item] = (
                self.weights['mf'] * mf_scores.get(item, 0) +
                self.weights['content'] * content_scores.get(item, 0) +
                self.weights['two_tower'] * two_tower_scores.get(item, 0)
            )
        
        # Business rules reranking
        return self._apply_business_rules(final_scores, user_id, top_k)
    
    def _apply_business_rules(self, scores, user_id, top_k):
        # Filter out purchased items
        purchased = self.get_user_purchases(user_id)
        scores = {k: v for k, v in scores.items() if k not in purchased}
        
        # Boost popular items slightly (avoid filter bubble)
        for item in scores:
            scores[item] *= (1 + 0.1 * self.popularity[item])
        
        # Diversity: ensure different categories in top-K
        return self._diversify(scores, top_k)
```

---

### 4. Caching & Precomputation

**3-Tier Caching**:

```
┌──────────────────────────────────────┐
│ L1: User-specific cache (Redis)     │  <-- 80% hit rate
│    TTL: 1 hour                       │
└──────────────┬───────────────────────┘
               │ Miss
               ▼
┌──────────────────────────────────────┐
│ L2: Batch precomputed recs           │  <-- 15% hit rate
│    (Updated nightly for all users)   │
└──────────────┬───────────────────────┘
               │ Miss
               ▼
┌──────────────────────────────────────┐
│ L3: Real-time computation            │  <-- 5% (new/active users)
│    (Expensive, <100ms target)        │
└──────────────────────────────────────┘
```

**Implementation**:

```python
class CachedRecommender:
    def __init__(self):
        self.redis = redis.Redis()
        self.model = EnsembleRecommender()
        
    async def get_recommendations(self, user_id: int, k: int = 10):
        # L1: Check user-specific cache
        cache_key = f"recs:{user_id}:{k}"
        if cached := await self.redis.get(cache_key):
            return json.loads(cached)
        
        # L2: Check batch precomputed
        batch_key = f"batch_recs:{user_id}"
        if batch_recs := await self.redis.get(batch_key):
            recs = json.loads(batch_recs)[:k]
            await self.redis.setex(cache_key, 3600, json.dumps(recs))
            return recs
        
        # L3: Compute in real-time
        recs = await self.model.recommend(user_id, k)
        await self.redis.setex(cache_key, 3600, json.dumps(recs))
        return recs
```

**Precomputation Job** (Nightly Spark):
```python
# Generate recs for all users (distributed)
user_recs = (
    spark.table("users")
    .rdd
    .map(lambda user: (user.id, model.recommend(user.id, k=100)))
    .persist()
)

# Store in Redis
user_recs.foreach(lambda ur: redis.set(f"batch_recs:{ur[0]}", json.dumps(ur[1])))
```

---

### 5. A/B Testing Framework

**Multi-Armed Bandit for Online Learning**:

```python
from scipy.stats import beta

class ThompsonSamplingBandit:
    def __init__(self, arms=['mf', 'content', 'two_tower']):
        self.arms = arms
        # Prior: Beta(1, 1) = uniform
        self.successes = {arm: 1 for arm in arms}
        self.failures = {arm: 1 for arm in arms}
        
    def select_arm(self):
        # Sample from Beta distribution for each arm
        samples = {
            arm: beta.rvs(self.successes[arm], self.failures[arm])
            for arm in self.arms
        }
        return max(samples, key=samples.get)
    
    def update(self, arm, reward):
        if reward > 0:
            self.successes[arm] += 1
        else:
            self.failures[arm] += 1
```

**Metrics to Track**:
- **CTR** (Click-Through Rate): % of recommended items clicked
- **Conversion Rate**: % leading to purchase
- **Engagement**: Time spent on recommended products
- **Diversity**: Number of unique categories shown

---

### 6. Scaling & Performance

**Latency Breakdown** (Target: <100ms p95):

| Component | Latency | Optimization |
|-----------|---------|--------------|
| Cache lookup | 5ms | Redis cluster |
| Feature retrieval | 10ms | Feature store (Feast) |
| Model inference | 20ms | TorchServe batch inference |
| Reranking | 15ms | Numba JIT |
| **Total** | **50ms** | ✓ Within budget |

**Throughput** (1M QPS peak):
- **API Pods**: 100 pods @ 10K QPS each
- **Redis**: 10-node cluster (sharded by user_id)
- **Model Serving**: 20 GPU instances (TensorRT optimized)

---

### 7. Cold Start Strategies

**New User (No History)**:
1. **Popularity-based**: Show top trending products
2. **Demographic-based**: Match similar users by age/location
3. **Contextual bandits**: Explore different categories

**New Product**:
1. **Content-based**: Match to similar existing products
2. **Boost in search**: Increase visibility for first 7 days
3. **Targeted ads**: Show to users who bought similar items

---

### 8. Cost Estimation

**Monthly Cost** (100M users, 10M products):

| Component | Cost | Notes |
|-----------|------|-------|
| Compute (100 API pods) | $5,000 | c5.2xlarge |
| Redis Cluster | $2,000 | r5.large |
| Model Training (Spark) | $3,000 | Monthly retrain |
| GPU Inference | $8,000 | 20x g4dn.xlarge |
| Storage (S3/EBS) | $1,000 | Embeddings, logs |
| **Total** | **~$19,000/month** | |

**Optimization**:
- Use **quantized embeddings** (float32 → int8): -75% storage
- **Batch inference** instead of real-time for non-critical users
- **Spot instances** for training: -70% cost

---

## Interview Discussion Points

1. **How to handle filter bubbles?**
   - Add exploration (ε-greedy, Thompson Sampling)
   - Diversity constraint in reranking
   - Show "new to you" category

2. **How to measure success?**
   - Online: CTR, conversion rate, revenue
   - Offline: NDCG, MAP, recall@k
   - User surveys: perceived quality

3. **What if a model degrades?**
   - Auto-rollback if CTR drops >5%
   - Gradual rollout (canary deployment)
   - Multi-model ensemble as safety net

4. **Privacy considerations?**
   - Federated learning for sensitive data
   - Differential privacy in embeddings
   - User opt-out of personalization

---

## Conclusion

This design provides:
- ✅ **Scalability**: Handles 100M users, 1M QPS
- ✅ **Low Latency**: <100ms p95 (50ms avg)
- ✅ **Cold Start**: Multiple fallback strategies
- ✅ **Adaptability**: A/B testing, online learning
- ✅ **Cost-Effective**: ~$19K/month

**Production Checklist**:
- ✓ Multi-model ensemble for robustness
- ✓ 3-tier caching for performance
- ✓ A/B testing for continuous improvement
- ✓ Cold start handling for all scenarios
