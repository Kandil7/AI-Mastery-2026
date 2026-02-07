# Case Study 17: Advanced Recommendation Systems for E-Commerce Platform

## Executive Summary

**Problem**: An e-commerce platform with 5M users and 2M products experienced 2.1% conversion rate and 34% cart abandonment, losing $12M in potential revenue annually.

**Solution**: Implemented a hybrid recommendation system combining collaborative filtering, content-based filtering, and deep learning with real-time personalization achieving 4.8% conversion rate and 18% cart abandonment.

**Impact**: Increased revenue by $28M annually, improved user engagement by 67%, and reduced cart abandonment by 47% while maintaining sub-200ms recommendation latency.

**System design snapshot** (full design in `docs/system_design_solutions/17_recommendation_system.md`):
- SLOs: p99 <200ms recommendation latency; 85% click-through rate; 99.9% availability during peak hours.
- Scale: ~50M recommendations/day; 5M users; 2M products; real-time personalization.
- Cost guardrails: < $0.001 per recommendation; infrastructure costs under $8K/month.
- Data quality gates: freshness SLA <1 hour; coverage metrics for cold start users/items.
- Reliability: blue/green deploys with A/B testing; auto rollback if CTR drops >5%.

---

## Business Context

### Company Profile
- **Industry**: E-Commerce Retail
- **User Base**: 5M registered users
- **Product Catalog**: 2M products across 500 categories
- **Conversion Rate**: 2.1% (industry average 3.5%)
- **Cart Abandonment**: 34% (industry average 69%)
- **Problem**: Poor personalization leading to low conversions

### Key Challenges
1. Cold start problem for new users and products
2. Scalability with growing user base and product catalog
3. Real-time personalization requirements
4. Need for diverse and serendipitous recommendations
5. Multi-objective optimization (clicks, conversions, revenue)

---

## Technical Approach

### Architecture Overview

```
User Session -> Feature Engineering -> Candidate Generation -> Scoring -> Ranking -> Diversity -> Recommendations
                    |                       |                    |         |         |           |
                    v                       v                    v         v         v           v
             User/Item Features      Matrix Factorization    Deep NN   Gradient Boosting  MMR Algorithm
                                    Content-Based Filter    Features      Model      Re-ranking
```

### Data Collection and Preprocessing

**Interaction Data**:
- User-item interactions: views, clicks, purchases, ratings
- User features: demographics, browsing history, purchase history
- Item features: category, price, brand, description, images
- Context features: time, device, location, season

**Dataset Creation**:
- 50M user-item interactions over 12 months
- 5M user profiles with demographic and behavioral features
- 2M item profiles with categorical and numerical features
- Temporal splits for evaluation (80/10/10 train/validation/test)

```python
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.preprocessing import LabelEncoder

def create_interaction_matrix(user_ids, item_ids, ratings, n_users, n_items):
    """Create sparse interaction matrix for collaborative filtering"""
    user_encoder = LabelEncoder()
    item_encoder = LabelEncoder()
    
    u_indices = user_encoder.fit_transform(user_ids)
    i_indices = item_encoder.fit_transform(item_ids)
    
    # Create sparse matrix
    interactions = csr_matrix(
        (ratings, (u_indices, i_indices)), 
        shape=(n_users, n_items)
    )
    
    return interactions, user_encoder, item_encoder

def extract_user_features(user_df, interaction_df):
    """Extract comprehensive user features"""
    user_features = user_df.copy()
    
    # Interaction-based features
    user_stats = interaction_df.groupby('user_id').agg({
        'rating': ['mean', 'std', 'count'],
        'timestamp': ['min', 'max']
    }).reset_index()
    
    user_stats.columns = ['user_id', 'avg_rating', 'rating_std', 'interaction_count',
                         'first_interaction', 'last_interaction']
    
    # Engagement features
    user_features = user_features.merge(user_stats, on='user_id', how='left')
    
    # Calculate recency
    user_features['days_since_last_interaction'] = (
        pd.Timestamp.now() - pd.to_datetime(user_features['last_interaction'])
    ).dt.days
    
    # Category preferences
    category_preferences = interaction_df.groupby(['user_id', 'category']).size().reset_index(name='cat_count')
    category_totals = category_preferences.groupby('user_id')['cat_count'].sum().reset_index(name='total_interactions')
    category_preferences = category_preferences.merge(category_totals, on='user_id')
    category_preferences['cat_preference'] = category_preferences['cat_count'] / category_preferences['total_interactions']
    
    # Pivot to create feature matrix
    cat_pref_pivot = category_preferences.pivot(index='user_id', columns='category', values='cat_preference').fillna(0)
    
    user_features = user_features.merge(cat_pref_pivot, left_on='user_id', right_index=True, how='left')
    
    return user_features

def extract_item_features(item_df, interaction_df):
    """Extract comprehensive item features"""
    item_features = item_df.copy()
    
    # Popularity features
    popularity = interaction_df.groupby('item_id').agg({
        'rating': ['mean', 'count', 'std'],
        'timestamp': 'max'
    }).reset_index()
    
    popularity.columns = ['item_id', 'avg_rating', 'popularity', 'rating_std', 'last_interaction']
    item_features = item_features.merge(popularity, on='item_id', how='left')
    
    # Price position relative to category
    item_features['price_percentile_in_category'] = item_features.groupby('category')['price'].rank(pct=True)
    
    # Temporal features
    item_features['days_since_launch'] = (
        pd.Timestamp.now() - pd.to_datetime(item_features['launch_date'])
    ).dt.days
    
    return item_features
```

### Model Architecture

**Hybrid Recommendation System**:
```python
from src.ml.recommendation import MatrixFactorization, ContentBasedFilter
from src.ml.deep_learning import NeuralNetwork, Dense, Embedding
import numpy as np

class HybridRecommender:
    def __init__(self, n_users, n_items, n_factors=100, embedding_dim=128):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.embedding_dim = embedding_dim
        
        # Collaborative filtering component
        self.cf_model = MatrixFactorization(n_users, n_items, n_factors)
        
        # Content-based component
        self.cb_model = ContentBasedFilter()
        
        # Deep learning component
        self.dl_model = DeepRecommender(n_users, n_items, embedding_dim)
        
        # Ensemble weights
        self.weights = {
            'collaborative': 0.4,
            'content_based': 0.3,
            'deep_learning': 0.3
        }
    
    def fit(self, interactions, user_features, item_features):
        """Fit all components of the hybrid model"""
        
        # Fit collaborative filtering
        self.cf_model.fit(interactions)
        
        # Fit content-based model
        self.cb_model.fit(item_features)
        
        # Fit deep learning model
        self.dl_model.fit(interactions, user_features, item_features)
        
    def predict(self, user_id, item_id):
        """Get prediction for user-item pair"""
        
        # Get predictions from each component
        cf_score = self.cf_model.predict(user_id, item_id)
        cb_score = self.cb_model.predict(user_id, item_id)
        dl_score = self.dl_model.predict(user_id, item_id)
        
        # Ensemble prediction
        final_score = (
            self.weights['collaborative'] * cf_score +
            self.weights['content_based'] * cb_score +
            self.weights['deep_learning'] * dl_score
        )
        
        return final_score
    
    def recommend(self, user_id, n_recommendations=10, exclude_seen=True):
        """Generate recommendations for user"""
        
        # Get user's seen items
        if exclude_seen:
            seen_items = self.get_user_interactions(user_id)
        else:
            seen_items = set()
        
        # Score all items
        scores = []
        for item_id in range(self.n_items):
            if item_id not in seen_items:
                score = self.predict(user_id, item_id)
                scores.append((item_id, score))
        
        # Sort and return top recommendations
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n_recommendations]

class DeepRecommender:
    def __init__(self, n_users, n_items, embedding_dim=128):
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Embedding layers
        self.user_embedding = Embedding(n_users, embedding_dim)
        self.item_embedding = Embedding(n_items, embedding_dim)
        
        # Neural network layers
        self.dense1 = Dense(embedding_dim * 2, 256)
        self.dense2 = Dense(256, 128)
        self.output = Dense(128, 1)
        
    def forward(self, user_ids, item_ids):
        """Forward pass through the deep recommender"""
        
        # Get embeddings
        user_emb = self.user_embedding.forward(user_ids)
        item_emb = self.item_embedding.forward(item_ids)
        
        # Concatenate embeddings
        concat_emb = np.concatenate([user_emb, item_emb], axis=1)
        
        # Forward through neural network
        x = self.dense1.forward(concat_emb)
        x = np.maximum(0, x)  # ReLU
        x = self.dense2.forward(x)
        x = np.maximum(0, x)  # ReLU
        output = self.output.forward(x)
        
        return output
```

---

## Model Development

### Approach Comparison

| Model | Precision@10 | Recall@10 | NDCG@10 | Coverage | Latency (ms) | Notes |
|-------|--------------|-----------|---------|----------|--------------|-------|
| Popularity Baseline | 0.12 | 0.08 | 0.15 | 0.05 | <1 | Simple but not personalized |
| Collaborative Filtering | 0.28 | 0.19 | 0.32 | 0.25 | 15 | Good for warm users/items |
| Content-Based | 0.22 | 0.15 | 0.26 | 0.65 | 25 | Good for cold start |
| Matrix Factorization | 0.31 | 0.23 | 0.37 | 0.35 | 20 | Better than basic CF |
| Deep Learning | 0.34 | 0.26 | 0.41 | 0.40 | 45 | Captures complex patterns |
| **Hybrid System** | **0.42** | **0.31** | **0.48** | **0.55** | **85** | **Selected** |

**Selected Model**: Hybrid Recommendation System
- **Reason**: Best balance of accuracy, coverage, and personalization
- **Architecture**: Ensemble of CF, CB, and DL models with diversity re-ranking

### Hyperparameter Tuning

```python
best_params = {
    'n_factors': 150,  # For matrix factorization
    'embedding_dim': 128,  # For deep learning model
    'learning_rate': 0.001,
    'regularization': 0.01,
    'batch_size': 256,
    'epochs': 100,
    'dropout_rate': 0.2,
    'ensemble_weights': [0.4, 0.3, 0.3]  # CF, CB, DL
}
```

### Training Process

```python
def train_hybrid_recommender(model, train_interactions, user_features, item_features, 
                            val_interactions, epochs=100):
    """Training loop for hybrid recommender"""
    
    for epoch in range(epochs):
        # Train collaborative filtering component
        model.cf_model.fit(train_interactions)
        
        # Train deep learning component
        model.dl_model.train_epoch(train_interactions, user_features, item_features)
        
        # Validate
        val_metrics = evaluate_recommender(model, val_interactions)
        
        print(f'Epoch {epoch+1}/{epochs}, '
              f'Val Precision@10: {val_metrics["precision@10"]:.4f}, '
              f'Val NDCG@10: {val_metrics["ndcg@10"]:.4f}')
    
    return model

def train_deep_recommender(model, interactions, user_features, item_features, epochs=100):
    """Training loop for deep learning component"""
    
    optimizer = Adam(learning_rate=0.001)
    
    for epoch in range(epochs):
        total_loss = 0.0
        
        # Create batches
        for batch in create_interaction_batches(interactions, batch_size=256):
            user_ids, item_ids, ratings = batch
            
            # Forward pass
            predictions = model.forward(user_ids, item_ids)
            
            # Compute loss
            loss = mse_loss(predictions, ratings)
            
            # Backward pass
            gradients = compute_gradients(loss, model)
            optimizer.update(model, gradients)
            
            total_loss += loss
        
        print(f'Deep Model Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}')
```

### Cross-Validation
- **Strategy**: Temporal cross-validation to respect chronological order
- **Validation Metrics**: Precision@10: 0.418 +/- 0.015, NDCG@10: 0.476 +/- 0.012
- **Test Metrics**: Precision@10: 0.423, NDCG@10: 0.481

---

## Production Deployment

### Infrastructure

**Cloud Architecture**:
- Kubernetes cluster with auto-scaling
- Redis for real-time feature caching
- Apache Kafka for streaming user events
- PostgreSQL for user/item metadata
- Elasticsearch for content-based search

### Software Architecture

```
User Session -> Event Stream -> Feature Store -> Model Inference -> Ranking -> Diversity -> API Response
                    |               |                |              |         |           |
                    v               v                v              v         v           v
            Kafka Producer   Redis Cache      Candidate Gen   Scoring    MMR Re-ranking  FastAPI
```

### Real-Time Recommendation Pipeline

```python
import asyncio
import redis
import numpy as np
from typing import List, Tuple, Dict
import json
from concurrent.futures import ThreadPoolExecutor

class RealTimeRecommender:
    def __init__(self, model_path, cache_ttl=3600):
        self.model = load_model(model_path)
        self.executor = ThreadPoolExecutor(max_workers=20)
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        self.cache_ttl = cache_ttl
        
    async def get_recommendations(self, user_id: int, n_recommendations: int = 10, 
                                 context: Dict = None) -> List[Tuple[int, float]]:
        """Get real-time recommendations for user"""
        
        # Check cache first
        cache_key = f"recs:{user_id}:{n_recommendations}"
        cached_recs = self.redis_client.get(cache_key)
        
        if cached_recs:
            return json.loads(cached_recs)
        
        # Get user features
        user_features = await self._get_user_features(user_id)
        
        # Generate candidates
        candidates = await self._generate_candidates(user_id, n_recommendations * 5)
        
        # Score candidates
        scored_candidates = await self._score_candidates(user_id, candidates, user_features)
        
        # Apply diversity re-ranking
        diversified_recs = await self._apply_diversity(
            scored_candidates, n_recommendations, user_features
        )
        
        # Cache results
        self.redis_client.setex(cache_key, self.cache_ttl, json.dumps(diversified_recs))
        
        return diversified_recs
    
    async def _get_user_features(self, user_id: int) -> Dict:
        """Get user features from cache/database"""
        # Check Redis first
        features_key = f"user_features:{user_id}"
        cached_features = self.redis_client.get(features_key)
        
        if cached_features:
            return json.loads(cached_features)
        
        # Fetch from database
        user_features = await fetch_user_features_from_db(user_id)
        
        # Cache for 1 hour
        self.redis_client.setex(features_key, 3600, json.dumps(user_features))
        
        return user_features
    
    async def _generate_candidates(self, user_id: int, n_candidates: int) -> List[int]:
        """Generate candidate items using multiple strategies"""
        
        candidates = set()
        
        # Collaborative filtering candidates
        cf_candidates = await self._get_cf_candidates(user_id, n_candidates // 3)
        candidates.update(cf_candidates)
        
        # Content-based candidates
        cb_candidates = await self._get_cb_candidates(user_id, n_candidates // 3)
        candidates.update(cb_candidates)
        
        # Trending/popular candidates
        trending_candidates = await self._get_trending_candidates(n_candidates // 3)
        candidates.update(trending_candidates)
        
        # Cold start fallback
        if len(candidates) < n_candidates:
            cold_start_candidates = await self._get_cold_start_candidates(
                user_id, n_candidates - len(candidates)
            )
            candidates.update(cold_start_candidates)
        
        return list(candidates)[:n_candidates]
    
    async def _score_candidates(self, user_id: int, candidates: List[int], 
                               user_features: Dict) -> List[Tuple[int, float]]:
        """Score candidates using the hybrid model"""
        
        loop = asyncio.get_event_loop()
        
        # Prepare input for model
        user_ids = [user_id] * len(candidates)
        
        # Run inference in thread pool
        scores = await loop.run_in_executor(
            self.executor,
            self._run_model_inference,
            user_ids, candidates
        )
        
        # Combine with candidate IDs
        scored_candidates = [(candidate, score) for candidate, score 
                            in zip(candidates, scores)]
        
        # Sort by score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return scored_candidates
    
    def _run_model_inference(self, user_ids, item_ids):
        """Run model inference (executed in thread pool)"""
        with torch.no_grad():
            scores = self.model.predict_batch(user_ids, item_ids)
        return scores.tolist()
    
    async def _apply_diversity(self, scored_candidates: List[Tuple[int, float]], 
                              n_recommendations: int, user_features: Dict) -> List[Tuple[int, float]]:
        """Apply MMR (Maximal Marginal Relevance) for diversity"""
        
        if len(scored_candidates) <= n_recommendations:
            return scored_candidates
        
        # Get item features for diversity calculation
        selected = []
        remaining = scored_candidates[:]
        
        # First, select the highest-scoring item
        if remaining:
            best = remaining.pop(0)
            selected.append(best)
        
        # Iteratively select diverse items
        while len(selected) < n_recommendations and remaining:
            best_candidate = None
            best_score = float('-inf')
            
            for i, (item_id, score) in enumerate(remaining):
                # Calculate diversity score (MMR formula)
                relevance = score
                
                # Calculate diversity penalty
                diversity_penalty = 0
                for sel_item_id, _ in selected:
                    similarity = await self._calculate_similarity(item_id, sel_item_id)
                    diversity_penalty = max(diversity_penalty, similarity)
                
                mmr_score = 0.7 * relevance - 0.3 * diversity_penalty
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_candidate = i
            
            if best_candidate is not None:
                selected_item = remaining.pop(best_candidate)
                selected.append(selected_item)
            else:
                # If no suitable candidate found, just take the next best
                selected.append(remaining.pop(0))
        
        return selected[:n_recommendations]
    
    async def _calculate_similarity(self, item1_id: int, item2_id: int) -> float:
        """Calculate similarity between two items"""
        # Get item features
        item1_features = await self._get_item_features(item1_id)
        item2_features = await self._get_item_features(item2_id)
        
        # Calculate cosine similarity
        sim = cosine_similarity(item1_features, item2_features)
        return sim

# API Implementation
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="E-Commerce Recommendation API")

class RecommendationRequest(BaseModel):
    user_id: int
    n_recommendations: int = 10
    context: dict = {}

class RecommendationResponse(BaseModel):
    user_id: int
    recommendations: List[Dict[str, any]]
    processing_time_ms: float

recommender = RealTimeRecommender(model_path="hybrid_recommender_v1.pkl")

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    try:
        start_time = time.time()
        
        recs = await recommender.get_recommendations(
            request.user_id, 
            request.n_recommendations,
            request.context
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Format response
        formatted_recs = []
        for item_id, score in recs:
            item_info = await get_item_details(item_id)
            formatted_recs.append({
                'item_id': item_id,
                'score': score,
                'title': item_info.get('title'),
                'price': item_info.get('price'),
                'category': item_info.get('category')
            })
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=formatted_recs,
            processing_time_ms=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}
```

### Online Learning and Updates

```python
import asyncio
from kafka import KafkaConsumer
import json

class OnlineUpdater:
    def __init__(self, recommender, kafka_topic="user_events"):
        self.recommender = recommender
        self.kafka_topic = kafka_topic
        self.consumer = KafkaConsumer(
            kafka_topic,
            bootstrap_servers=['localhost:9092'],
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))
        )
    
    async def start_streaming_updates(self):
        """Process streaming user events for online learning"""
        
        for message in self.consumer:
            event = message.value
            
            # Process different event types
            if event['type'] == 'view':
                await self._process_view_event(event)
            elif event['type'] == 'click':
                await self._process_click_event(event)
            elif event['type'] == 'purchase':
                await self._process_purchase_event(event)
            elif event['type'] == 'cart_add':
                await self._process_cart_event(event)
    
    async def _process_view_event(self, event):
        """Process view event"""
        user_id = event['user_id']
        item_id = event['item_id']
        timestamp = event['timestamp']
        
        # Update user features
        await update_user_features(user_id, {
            'last_viewed_item': item_id,
            'view_count': increment_counter(f"user:{user_id}:views"),
            'view_timestamp': timestamp
        })
        
        # Update item features
        await update_item_features(item_id, {
            'view_count': increment_counter(f"item:{item_id}:views"),
            'last_viewed': timestamp
        })
    
    async def _process_purchase_event(self, event):
        """Process purchase event"""
        user_id = event['user_id']
        item_id = event['item_id']
        rating = event.get('rating', 5)  # Assume 5-star if no explicit rating
        
        # Update interaction matrix
        await add_interaction(user_id, item_id, rating, 'purchase')
        
        # Update user preferences
        item_category = await get_item_category(item_id)
        await update_user_category_preference(user_id, item_category, 1.0)
        
        # Update item popularity
        await update_item_popularity(item_id, 1.0)
        
        # Trigger model update if needed
        if await should_update_model(user_id, item_id):
            await self._trigger_incremental_update(user_id, item_id)
    
    async def _trigger_incremental_update(self, user_id, item_id):
        """Trigger incremental model update"""
        # Update embeddings for the specific user/item
        await self.recommender.incremental_update(user_id, item_id)
        
        # Update cache
        await self._invalidate_user_cache(user_id)
```

### Operational SLOs and Runbook
- **Latency**: p99 <200ms; auto-scale if exceeded
- **Availability**: 99.9% during peak hours; 99.5% off-peak
- **Accuracy**: Maintain >85% CTR; trigger retraining if below 83%
- **Runbook Highlights**:
  - Model drift: monitor CTR daily, retrain weekly
  - Traffic spikes: auto-scale based on request rate
  - Data quality: validate feature distributions, alert on anomalies

### Monitoring and Alerting
- **Metrics**: CTR, conversion rate, revenue per user, recommendation diversity
- **Alerts**: Page if CTR drops below 83% or latency exceeds 200ms
- **A/B Testing**: Compare model versions with statistical significance

---

## Results & Impact

### Model Performance in Production

**Overall Performance**:
- **Precision@10**: 0.423
- **Recall@10**: 0.311
- **NDCG@10**: 0.481
- **Coverage**: 0.55 (fraction of items that can be recommended)
- **Diversity**: 0.73 (entropy of recommended categories)

**Per-User Segment Performance**:
| User Segment | Precision@10 | CTR | Conversion Rate |
|--------------|--------------|-----|-----------------|
| New Users | 0.18 | 1.2% | 0.8% |
| Active Users | 0.45 | 4.2% | 3.1% |
| Power Users | 0.52 | 6.8% | 5.2% |
| Inactive Users | 0.22 | 1.8% | 1.2% |

### Business Impact (12 months post-launch)

| Metric | Before ML | After ML | Improvement |
|--------|-----------|----------|-------------|
| **Conversion Rate** | 2.1% | 4.8% | **+128.6%** |
| **Cart Abandonment** | 34% | 18% | **-47.1%** |
| **Click-Through Rate** | 1.8% | 4.2% | **+133.3%** |
| **Revenue per User** | $42 | $98 | **+133.3%** |
| **Session Duration** | 3.2 min | 5.4 min | **+68.8%** |
| **Return Visits** | 28% | 45% | **+60.7%** |
| **Annual Revenue Impact** | - | - | **+$28M** |

### Cost-Benefit Analysis

**Annual Benefits**:
- Increased revenue: $28M
- Reduced cart abandonment: $3.2M
- Improved user retention: $2.1M
- **Total Annual Benefit**: $33.3M

**Investment**:
- Model development: $800K
- Infrastructure: $400K
- Integration: $300K
- **Total Investment**: $1.5M

**ROI**: 2120% in first year ($33.3M/$1.5M)

### Key Insights from Analysis

**Most Effective Recommendation Strategies**:
1. **Collaborative Filtering**: 40% of clicks (warm users/items)
2. **Content-Based**: 25% of clicks (cold start scenarios)
3. **Trending Items**: 20% of clicks (new users)
4. **Cross-Sell**: 15% of clicks (purchase context)

**User Behavior Patterns**:
- **Time-based**: Evening recommendations perform 23% better
- **Device-based**: Mobile users prefer visual recommendations
- **Seasonal**: Holiday season drives 45% more engagement
- **Category-based**: Electronics have highest conversion rates

---

## Challenges & Solutions

### Challenge 1: Cold Start Problem
- **Problem**: Difficulty recommending for new users and products
- **Solution**:
  - Content-based fallback for new users
  - Demographic-based recommendations initially
  - Rapid learning from first few interactions
  - Trending items for new products

### Challenge 2: Scalability
- **Problem**: 5M users and 2M products require efficient computation
- **Solution**:
  - Two-stage approach: candidate generation + ranking
  - Approximate nearest neighbor search
  - Caching popular recommendations
  - Distributed computing with Spark

### Challenge 3: Real-Time Personalization
- **Problem**: Need to incorporate recent user behavior instantly
- **Solution**:
  - Streaming architecture with Kafka
  - Incremental model updates
  - Feature store with real-time updates
  - Session-based recommendations

### Challenge 4: Multi-Objective Optimization
- **Problem**: Balancing clicks, conversions, revenue, and user satisfaction
- **Solution**:
  - Multi-task learning approach
  - Reward shaping for different objectives
  - A/B testing framework for objective weights
  - Feedback loops for continuous improvement

---

## Lessons Learned

### What Worked

1. **Hybrid Approach**
   - Combining multiple recommendation strategies improved performance
   - Each approach handles different scenarios effectively
   - Ensemble methods provided robustness

2. **Real-Time Updates**
   - Streaming architecture enabled rapid adaptation
   - Incremental learning maintained model freshness
   - Session-based personalization increased engagement

3. **Diversity and Serendipity**
   - MMR algorithm improved user experience
   - Exploration-exploitation balance maintained novelty
   - Category diversity prevented filter bubbles

### What Didn't Work

1. **Single Model Approach**
   - Pure collaborative filtering failed for new users
   - Content-based alone missed complex patterns
   - Deep learning alone was too slow for real-time

2. **Batch-Only Processing**
   - Daily updates too slow for user behavior changes
   - Missed opportunities for immediate personalization
   - Switched to streaming + batch hybrid

---

## Technical Implementation

### Matrix Factorization Implementation

```python
import numpy as np
from scipy.sparse import csr_matrix
import random

class MatrixFactorization:
    def __init__(self, n_users, n_items, n_factors=100, learning_rate=0.01, reg=0.01, epochs=100):
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg = reg
        self.epochs = epochs
        
        # Initialize user and item latent factors
        self.user_factors = np.random.normal(0, 0.1, (n_users, n_factors))
        self.item_factors = np.random.normal(0, 0.1, (n_items, n_factors))
        
        # Initialize biases
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_bias = 0.0
    
    def fit(self, interactions):
        """Fit the matrix factorization model using SGD"""
        # Calculate global bias
        self.global_bias = interactions.data.mean()
        
        # Get non-zero indices
        rows, cols = interactions.nonzero()
        
        for epoch in range(self.epochs):
            # Shuffle the data
            indices = list(range(len(rows)))
            random.shuffle(indices)
            
            for idx in indices:
                i, j = rows[idx], cols[idx]
                rating = interactions[i, j]
                
                # Current prediction
                prediction = self.global_bias + self.user_bias[i] + self.item_bias[j] + \
                             np.dot(self.user_factors[i, :], self.item_factors[j, :])
                
                # Error
                error = rating - prediction
                
                # Update factors
                self.user_factors[i, :] += self.learning_rate * (
                    error * self.item_factors[j, :] - self.reg * self.user_factors[i, :]
                )
                
                self.item_factors[j, :] += self.learning_rate * (
                    error * self.user_factors[i, :] - self.reg * self.item_factors[j, :]
                )
                
                # Update biases
                self.user_bias[i] += self.learning_rate * (error - self.reg * self.user_bias[i])
                self.item_bias[j] += self.learning_rate * (error - self.reg * self.item_bias[j])
    
    def predict(self, user_id, item_id):
        """Predict rating for user-item pair"""
        prediction = self.global_bias + self.user_bias[user_id] + self.item_bias[item_id] + \
                     np.dot(self.user_factors[user_id, :], self.item_factors[item_id, :])
        return prediction
    
    def recommend(self, user_id, n_recommendations=10, exclude_seen=True):
        """Generate recommendations for user"""
        user_vector = self.user_factors[user_id, :]
        
        # Calculate scores for all items
        scores = self.global_bias + self.user_bias + self.item_bias + \
                 np.dot(user_vector, self.item_factors.T)
        
        # Exclude items user has already interacted with
        if exclude_seen:
            seen_items = self.get_user_interactions(user_id)
            scores[list(seen_items)] = -np.inf
        
        # Get top recommendations
        top_items = np.argsort(scores)[::-1][:n_recommendations]
        top_scores = scores[top_items]
        
        return list(zip(top_items, top_scores))
    
    def get_user_interactions(self, user_id):
        """Get items user has interacted with"""
        # This would typically come from the interaction matrix
        # Implementation depends on how interactions are stored
        pass

# Content-Based Filtering Implementation
class ContentBasedFilter:
    def __init__(self, similarity_metric='cosine'):
        self.similarity_metric = similarity_metric
        self.item_features = None
        self.feature_matrix = None
    
    def fit(self, item_features):
        """Fit the content-based model"""
        self.item_features = item_features
        self.feature_matrix = self._normalize_features(item_features)
    
    def _normalize_features(self, features):
        """Normalize item features"""
        # Implementation depends on feature types
        # Could include TF-IDF for text, min-max scaling for numeric
        pass
    
    def predict(self, user_id, item_id):
        """Predict score based on user profile similarity to item"""
        # Get user profile (average of liked items' features)
        user_profile = self._get_user_profile(user_id)
        
        # Calculate similarity between user profile and item
        item_features = self.feature_matrix[item_id]
        similarity = self._calculate_similarity(user_profile, item_features)
        
        return similarity
    
    def _get_user_profile(self, user_id):
        """Get user profile based on past interactions"""
        # Average features of items user has interacted with positively
        pass
    
    def _calculate_similarity(self, vec1, vec2):
        """Calculate similarity between two vectors"""
        if self.similarity_metric == 'cosine':
            dot_product = np.dot(vec1, vec2)
            norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
            return dot_product / (norm_product + 1e-8)  # Add small epsilon to avoid division by zero
        # Add other similarity metrics as needed
```

### Deep Learning Component

```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim=128):
        super(NeuralCollaborativeFiltering, self).__init__()
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # Neural network layers
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, user_ids, item_ids):
        user_embeds = self.user_embedding(user_ids)
        item_embeds = self.item_embedding(item_ids)
        
        # Concatenate user and item embeddings
        concat_embeds = torch.cat([user_embeds, item_embeds], dim=1)
        
        # Pass through neural network
        output = self.fc_layers(concat_embeds)
        
        return output.squeeze()

class DeepRecommenderTrainer:
    def __init__(self, model, learning_rate=0.001):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.criterion = nn.BCELoss()  # Binary cross-entropy for implicit feedback
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        
        for batch in train_loader:
            user_ids, item_ids, ratings = batch
            
            self.optimizer.zero_grad()
            
            predictions = self.model(user_ids, item_ids)
            loss = self.criterion(predictions, ratings.float())
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def evaluate(self, test_loader):
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in test_loader:
                user_ids, item_ids, ratings = batch
                
                predictions = self.model(user_ids, item_ids)
                loss = self.criterion(predictions, ratings.float())
                
                # Convert predictions to binary (threshold at 0.5)
                binary_preds = (predictions > 0.5).float()
                correct_predictions += (binary_preds == ratings.float()).sum().item()
                total_predictions += len(ratings)
                
                total_loss += loss.item()
        
        accuracy = correct_predictions / total_predictions
        avg_loss = total_loss / len(test_loader)
        
        return avg_loss, accuracy
```

---

## Next Steps & Future Improvements

### Short-Term (Q1 2026)
- [ ] Implement contextual bandits for exploration-exploitation
- [ ] Add visual similarity features for fashion items
- [ ] Enhance diversity algorithm with category constraints

### Medium-Term (Q2-Q3 2026)
- [ ] Extend to session-based recommendations
- [ ] Implement graph neural networks for social recommendations
- [ ] Add multi-modal features (images, text, reviews)

### Long-Term (2027)
- [ ] Develop reinforcement learning for long-term value optimization
- [ ] Implement federated learning for privacy preservation
- [ ] Add conversational recommendations with NLP

---

## Mathematical Foundations

### Matrix Factorization
The objective function for regularized matrix factorization:
```
min ∑(i,j)∈R (r_{ij} - μ - b_i - c_j - p_i^T q_j)^2 + λ(||p_i||^2 + ||q_j||^2 + b_i^2 + c_j^2)
```
Where:
- r_{ij} is the rating of user i for item j
- μ is the global average rating
- b_i and c_j are user and item biases
- p_i and q_j are user and item latent factor vectors
- λ is the regularization parameter

### Neural Collaborative Filtering
The generalized matrix factorization (GMF) component:
```
y_{ui} = h^T (p_u ⊙ q_i)
```
Where ⊙ denotes element-wise multiplication.

The multi-layer perceptron (MLP) component:
```
f^{(1)} = activation(W^{(1)}[p_u, q_i] + b^{(1)})
...
f^{(L)} = activation(W^{(L)}f^{(L-1)} + b^{(L)})
y_{ui} = σ(h^T f^{(L)})
```

### Information Retrieval Metrics
Precision@K:
```
P@K = (1/K) * Σ_{i=1}^{K} rel(i)
```
Where rel(i) = 1 if the i-th item is relevant, 0 otherwise.

NDCG@K (Normalized Discounted Cumulative Gain):
```
NDCG@K = DCG@K / IDCG@K
DCG@K = Σ_{i=1}^{K} (2^{rel_i} - 1) / log_2(i + 1)
```

### Diversity Metrics
Coverage measures the fraction of items that can be recommended:
```
Coverage = |∪_u R(u)| / |I|
```
Where R(u) is the set of items recommended to user u, and I is the set of all items.

Diversity can be measured using intra-list diversity:
```
ILD(R) = (2/(|R|(|R|-1))) * Σ_{i,j∈R, i≠j} dist(i,j)
```
Where dist(i,j) is the distance between items i and j.

---

## Production Considerations

### Scalability
- **Distributed Training**: Use Spark for large-scale matrix factorization
- **Approximate Algorithms**: LSH for fast nearest neighbor search
- **Caching Strategy**: Redis for frequently accessed recommendations
- **Load Balancing**: Distribute requests across multiple model instances

### Security
- **Data Privacy**: Anonymize user data and use differential privacy
- **Access Control**: API keys and role-based access to recommendation API
- **Audit Logging**: Track all recommendation requests and outcomes

### Reliability
- **Redundancy**: Multiple model instances across availability zones
- **Graceful Degradation**: Fallback to simpler models if primary fails
- **Monitoring**: Real-time performance and accuracy tracking

### Performance Optimization
- **Model Compression**: Quantization and pruning for faster inference
- **Batch Processing**: Process multiple requests together when possible
- **Feature Caching**: Pre-compute and cache user/item features

---

## Conclusion

This advanced recommendation system demonstrates sophisticated ML engineering:
- **Hybrid Architecture**: Combines multiple recommendation strategies
- **Real-Time Processing**: Streaming updates and personalization
- **Scalable Infrastructure**: Handles millions of users and items
- **Business Impact**: $33.3M annual revenue increase

**Key takeaway**: Effective recommendation systems require combining multiple approaches with careful attention to scalability and real-time requirements.

Architecture and ops blueprint: `docs/system_design_solutions/17_recommendation_system.md`.

---

**Contact**: Implementation details in `src/recommendation/hybrid_recommender.py`.
Notebooks: `notebooks/case_studies/recommendation_systems.ipynb`