# System Design: Real-Time Recommendation System

## 1. Problem Definition
Design a recommendation system for a video streaming platform (like YouTube or Netflix) that personalizes the "Home Feed" for users.

### 1.1 Requirements
**Functional:**
*   Users should see a list of personalized video recommendations.
*   The feed should update in near real-time based on user interactions (clicks, views).
*   Support for "New" (cold-start) content and users.

**Non-Functional:**
*   **Latency:** < 200ms for generating the feed.
*   **Scalability:** 100M+ Daily Active Users (DAU), 1B+ videos.
*   **Availability:** 99.99%.
*   **Freshness:** User actions should influence recommendations within seconds.

---

## 2. High-Level Architecture
We follow the standard **Candidate Generation → Scoring → Re-ranking** funnel.

```mermaid
graph TD
    User[User] -->|Request Feed| API[API Gateway]
    
    subgraph "Online Serving"
        API --> Orchestrator[Recommendation Service]
        Orchestrator --> CG[Candidate Generation]
        Orchestrator --> Ranker[Ranking Service]
        Orchestrator --> Reranker[Re-ranking/Policy]
    end
    
    subgraph "Data & Training"
        Logs[Event Logs] -->|Stream| FeatureStore[Feature Store]
        Logs -->|Batch| DataLake[Data Lake]
        DataLake --> Training[Model Training Pipeline]
        Training -->|Model Weights| ModelRegistry[Model Registry]
    end
    
    subgraph "Storage"
        FeatureStore <--> Ranker
        ANN[Vector DB (ANN)] <--> CG
    end
```

---

## 3. Detailed Components

### 3.1 Candidate Generation (Retrieval)
**Goal:** Narrow down 1B videos to ~1,000 candidates relevant to the user.
**Fast & Coarse.**

*   **Two-Tower Embedding Model:**
    *   **User Tower:** Encodes user history, demographics into a vector $u$.
    *   **Item Tower:** Encodes video metadata, transcripts into a vector $v$.
    *   **Similarity:** Dot product $u \cdot v$.
    *   **Serving:** Approximate Nearest Neighbor (ANN) search (e.g., FAISS, ScaNN, HNSW).
*   **Sources:**
    *   **Collaborative Filtering:** "Users who watched X also watched Y" (Item-to-Item).
    *   **Content-Based:** Similarity tags/embeddings.
    *   **Trending/Popular:** Global top lists.

### 3.2 Ranking (Scoring)
**Goal:** Score the 1,000 candidates to find the most likely clicks/views.
**Precise & Heavy.**

*   **Model:** Multi-Task Learning (MTL) Deep Neural Network (e.g., DLRM or Wide & Deep).
*   **Objectives (Tasks):** Predict p(Click), p(Watch > 30s), p(Share).
*   **Final Score:** weighted_sum = $w_1 \cdot p(Click) + w_2 \cdot p(Watch)$.
*   **Features:**
    *   **User:** Age, location, device, past 50 interactions embedding.
    *   **Item:** ID embedding, category, creator freq.
    *   **Context:** Time of day, day of week.
    *   **Interaction:** User-Item cross features (e.g., "User liked this creator before?").

### 3.3 Re-ranking & Policy Layer
**Goal:** Business logic and diversity.
*   **Diversity filters:** Don't show 10 videos from the same creator.
*   **Bloom Filters:** Remove already watched videos.
*   **Exploration:** Insert random/new content (Epsilon-Greedy or bandit).

### 3.4 Feature Store (Real-Time)
Need low-latency access to features for Ranking.
*   **Technologies:** Redis, Cassandra, or specialized stores like Tecton/Feast.
*   **User Profile:** Updated essentially immediately after a view event (via Kafka pipeline).

---

## 4. Data Pipeline & Training

### 4.1 Training Data
*   **Positive Labels:** Clicked, Watched > 30s.
*   **Negative Labels:** Impressed but not clicked (Implicit feedback).
*   **Point-in-Time Correctness:** Must log features exactly as they were at inference time to avoid "data leakage."

### 4.2 Handling Cold Start
*   **New Items:** Use content embeddings (BERT on title, ResNet on thumbnail) to map to vector space immediately. Use "Bandits" to boost exposure initially.
*   **New Users:** Rely on broad demographics (geo, device) + popular items until interaction history builds up.

---

## 5. Scaling Estimates

*   **100M DAU**, assume 5 refreshes/day = 500M requests/day.
*   **QPS:** 500M / 86400 ≈ ~6,000 QPS average, peak ~20,000 QPS.
*   **Storage:** 
    *   1B videos x 1KB metadata = 1TB (RAM/SSD capable).
    *   Embeddings (1B x 64 dims x 4 bytes) ≈ 256GB (Fit in highly optimized RAM index or distributed).

## 6. Trade-offs
*   **Accuracy vs. Latency:** Heavier ranking models (Transformers) are better but slower. Solution: Distillation or smaller student models for online inference.
*   **Freshness vs. Stability:** Updating embeddings instantly can cause feedback loops. Solution: Batch updates hourly/daily, but real-time features (last 10 clicks) updated instantly.
