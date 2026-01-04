"""
Multi-Stage Ranking Pipeline
============================

Production ranking pipeline for recommendation systems.

Inspired by Pinterest/Netflix multi-stage architecture:
- Candidate Generation: Fast retrieval (k-NN, graph-based)
- Pre-Ranking: Lightweight scoring for filtering
- Full Ranking: Deep model for precise ranking  
- Re-Ranking: Business rules, diversity, freshness

Features:
- Multiple retrieval strategies
- Score calibration across sources
- Diversity and freshness controls
- Position bias correction

References:
- Pinterest Pre-Ranking architecture
- Netflix Recommendation System
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
from abc import ABC, abstractmethod
import logging
from enum import Enum
import heapq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Item:
    """An item to be ranked."""
    id: str
    features: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    
    
@dataclass
class User:
    """A user requesting recommendations."""
    id: str
    features: np.ndarray
    history: List[str] = field(default_factory=list)  # Item IDs
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    """A candidate item with scores from different stages."""
    item: Item
    retrieval_score: float = 0.0
    retrieval_source: str = ""
    prerank_score: float = 0.0
    fullrank_score: float = 0.0
    final_score: float = 0.0
    position: int = -1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RankingResult:
    """Result from the ranking pipeline."""
    candidates: List[Candidate]
    retrieval_count: int
    prerank_count: int
    fullrank_count: int
    latency_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# CANDIDATE GENERATION
# ============================================================

class RetrievalSource(ABC):
    """Abstract base class for retrieval sources."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this retrieval source."""
        pass
    
    @abstractmethod
    def retrieve(self, user: User, k: int) -> List[Candidate]:
        """Retrieve candidate items for a user."""
        pass


class EmbeddingSimilaritySource(RetrievalSource):
    """Retrieve items similar to user embedding."""
    
    def __init__(self, items: List[Item], item_embeddings: np.ndarray):
        self.items = items
        self.item_embeddings = item_embeddings
        self._name = "embedding_similarity"
        
    @property
    def name(self) -> str:
        return self._name
    
    def retrieve(self, user: User, k: int) -> List[Candidate]:
        """Retrieve top-k items by embedding similarity."""
        if len(self.items) == 0:
            return []
            
        # Normalize for cosine similarity
        user_norm = user.features / (np.linalg.norm(user.features) + 1e-8)
        item_norms = self.item_embeddings / (
            np.linalg.norm(self.item_embeddings, axis=1, keepdims=True) + 1e-8
        )
        
        # Compute similarities
        similarities = item_norms @ user_norm
        
        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]
        
        candidates = []
        for idx in top_k_indices:
            candidates.append(Candidate(
                item=self.items[idx],
                retrieval_score=float(similarities[idx]),
                retrieval_source=self.name
            ))
            
        return candidates


class PopularitySource(RetrievalSource):
    """Retrieve popular items."""
    
    def __init__(self, items: List[Item], popularity_scores: Dict[str, float]):
        self.items = items
        self.popularity_scores = popularity_scores
        self._name = "popularity"
        
    @property
    def name(self) -> str:
        return self._name
    
    def retrieve(self, user: User, k: int) -> List[Candidate]:
        """Retrieve top-k popular items."""
        # Sort by popularity
        sorted_items = sorted(
            self.items,
            key=lambda x: self.popularity_scores.get(x.id, 0),
            reverse=True
        )
        
        candidates = []
        for item in sorted_items[:k]:
            score = self.popularity_scores.get(item.id, 0)
            candidates.append(Candidate(
                item=item,
                retrieval_score=score,
                retrieval_source=self.name
            ))
            
        return candidates


class RecentSource(RetrievalSource):
    """Retrieve recently added items."""
    
    def __init__(self, items: List[Item]):
        self.items = items
        self._name = "recent"
        
    @property
    def name(self) -> str:
        return self._name
    
    def retrieve(self, user: User, k: int) -> List[Candidate]:
        """Retrieve most recent items."""
        # Sort by creation date
        items_with_date = [
            (item, item.created_at or datetime.min)
            for item in self.items
        ]
        sorted_items = sorted(items_with_date, key=lambda x: x[1], reverse=True)
        
        candidates = []
        for item, created_at in sorted_items[:k]:
            # Score based on recency (decay over time)
            if created_at != datetime.min:
                age_days = (datetime.now() - created_at).days
                score = 1.0 / (1 + age_days * 0.1)
            else:
                score = 0.5
                
            candidates.append(Candidate(
                item=item,
                retrieval_score=score,
                retrieval_source=self.name
            ))
            
        return candidates


class CollaborativeFilteringSource(RetrievalSource):
    """Retrieve items based on similar users' preferences."""
    
    def __init__(self, items: List[Item], 
                 user_item_matrix: Dict[str, Dict[str, float]]):
        self.items = items
        self.item_map = {item.id: item for item in items}
        self.user_item_matrix = user_item_matrix
        self._name = "collaborative"
        
    @property
    def name(self) -> str:
        return self._name
    
    def retrieve(self, user: User, k: int) -> List[Candidate]:
        """Retrieve items that similar users liked."""
        if user.id not in self.user_item_matrix:
            return []
            
        user_ratings = self.user_item_matrix[user.id]
        
        # Find similar users
        similar_users = self._find_similar_users(user.id, n=10)
        
        # Aggregate scores from similar users
        item_scores: Dict[str, float] = defaultdict(float)
        
        for similar_user_id, similarity in similar_users:
            if similar_user_id not in self.user_item_matrix:
                continue
            for item_id, rating in self.user_item_matrix[similar_user_id].items():
                if item_id not in user_ratings:  # Don't recommend seen items
                    item_scores[item_id] += similarity * rating
                    
        # Get top-k
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        
        candidates = []
        for item_id, score in sorted_items[:k]:
            if item_id in self.item_map:
                candidates.append(Candidate(
                    item=self.item_map[item_id],
                    retrieval_score=score,
                    retrieval_source=self.name
                ))
                
        return candidates
    
    def _find_similar_users(self, user_id: str, n: int) -> List[Tuple[str, float]]:
        """Find n most similar users."""
        user_ratings = self.user_item_matrix.get(user_id, {})
        if not user_ratings:
            return []
            
        similarities = []
        
        for other_id, other_ratings in self.user_item_matrix.items():
            if other_id == user_id:
                continue
                
            # Compute cosine similarity
            common_items = set(user_ratings.keys()) & set(other_ratings.keys())
            if not common_items:
                continue
                
            vec1 = [user_ratings[i] for i in common_items]
            vec2 = [other_ratings[i] for i in common_items]
            
            dot = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = sum(a ** 2 for a in vec1) ** 0.5
            norm2 = sum(b ** 2 for b in vec2) ** 0.5
            
            if norm1 > 0 and norm2 > 0:
                sim = dot / (norm1 * norm2)
                similarities.append((other_id, sim))
                
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]


class CandidateGenerator:
    """
    Generate candidates from multiple sources.
    
    Merges results from different retrieval strategies
    and deduplicates by item ID.
    """
    
    def __init__(self):
        self.sources: List[RetrievalSource] = []
        self.source_weights: Dict[str, float] = {}
        
    def add_source(self, source: RetrievalSource, weight: float = 1.0) -> None:
        """Add a retrieval source."""
        self.sources.append(source)
        self.source_weights[source.name] = weight
        
    def generate(self, user: User, k_per_source: int = 100) -> List[Candidate]:
        """
        Generate candidates from all sources.
        
        Args:
            user: User to generate for
            k_per_source: Number of candidates per source
            
        Returns:
            Deduplicated list of candidates
        """
        # Collect from all sources
        all_candidates: Dict[str, Candidate] = {}
        
        for source in self.sources:
            try:
                candidates = source.retrieve(user, k_per_source)
                weight = self.source_weights.get(source.name, 1.0)
                
                for candidate in candidates:
                    item_id = candidate.item.id
                    
                    if item_id in all_candidates:
                        # Merge scores using weighted max
                        existing = all_candidates[item_id]
                        existing.retrieval_score = max(
                            existing.retrieval_score,
                            candidate.retrieval_score * weight
                        )
                        existing.metadata["sources"] = existing.metadata.get("sources", []) + [source.name]
                    else:
                        candidate.retrieval_score *= weight
                        candidate.metadata["sources"] = [source.name]
                        all_candidates[item_id] = candidate
                        
            except Exception as e:
                logger.error(f"Error in source {source.name}: {e}")
                
        # Sort by retrieval score
        candidates = list(all_candidates.values())
        candidates.sort(key=lambda c: c.retrieval_score, reverse=True)
        
        return candidates


# ============================================================
# PRE-RANKER
# ============================================================

class PreRanker:
    """
    Lightweight ranking for filtering candidates.
    
    Uses a simple model (e.g., logistic regression, small neural net)
    to quickly score candidates and reduce the set for full ranking.
    """
    
    def __init__(self, 
                 feature_dim: int,
                 hidden_dim: int = 64,
                 num_features: int = 10):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Simple MLP for scoring
        scale = np.sqrt(2.0 / feature_dim)
        self.W1 = np.random.randn(feature_dim * 2 + num_features, hidden_dim) * scale
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * scale
        self.b2 = np.zeros(1)
        
    def score(self, user: User, candidates: List[Candidate]) -> List[Candidate]:
        """
        Score candidates with lightweight model.
        
        Args:
            user: User features
            candidates: Candidates to score
            
        Returns:
            Candidates with prerank_score set
        """
        user_feat = user.features
        
        for candidate in candidates:
            item_feat = candidate.item.features
            
            # Simple features
            extra_features = np.array([
                candidate.retrieval_score,
                len(candidate.metadata.get("sources", [])),
                1.0 if candidate.item.id in user.history else 0.0,
                # Add more lightweight features here
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Padding
            ])[:10]
            
            # Concatenate features
            combined = np.concatenate([user_feat, item_feat, extra_features])
            
            # Forward pass
            h = np.maximum(0, combined @ self.W1 + self.b1)
            score = float(1 / (1 + np.exp(-(h @ self.W2 + self.b2))))
            
            candidate.prerank_score = score
            
        return candidates
    
    def filter(self, candidates: List[Candidate], k: int) -> List[Candidate]:
        """
        Filter to top-k by prerank score.
        
        Args:
            candidates: Scored candidates
            k: Number to keep
            
        Returns:
            Top-k candidates
        """
        candidates.sort(key=lambda c: c.prerank_score, reverse=True)
        return candidates[:k]


# ============================================================
# FULL RANKER
# ============================================================

class FullRanker:
    """
    Deep ranking model for precise scoring.
    
    Uses a more complex model with rich features
    for accurate ranking of the reduced candidate set.
    """
    
    def __init__(self,
                 user_feature_dim: int,
                 item_feature_dim: int,
                 embedding_dim: int = 64,
                 hidden_dims: List[int] = None):
        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [128, 64]
        
        # Build layers
        self._build_model()
        
    def _build_model(self):
        """Build the ranking model."""
        scale = lambda dim: np.sqrt(2.0 / dim)
        
        # User tower
        self.user_W1 = np.random.randn(self.user_feature_dim, self.hidden_dims[0]) * scale(self.user_feature_dim)
        self.user_b1 = np.zeros(self.hidden_dims[0])
        self.user_W2 = np.random.randn(self.hidden_dims[0], self.embedding_dim) * scale(self.hidden_dims[0])
        self.user_b2 = np.zeros(self.embedding_dim)
        
        # Item tower
        self.item_W1 = np.random.randn(self.item_feature_dim, self.hidden_dims[0]) * scale(self.item_feature_dim)
        self.item_b1 = np.zeros(self.hidden_dims[0])
        self.item_W2 = np.random.randn(self.hidden_dims[0], self.embedding_dim) * scale(self.hidden_dims[0])
        self.item_b2 = np.zeros(self.embedding_dim)
        
        # Cross features network
        cross_dim = self.embedding_dim * 2 + 10  # +10 for cross features
        self.cross_W = np.random.randn(cross_dim, 32) * scale(cross_dim)
        self.cross_b = np.zeros(32)
        self.final_W = np.random.randn(32, 1) * scale(32)
        self.final_b = np.zeros(1)
        
    def score(self, user: User, candidates: List[Candidate]) -> List[Candidate]:
        """
        Score candidates with full model.
        
        Args:
            user: User with features
            candidates: Candidates to score
            
        Returns:
            Candidates with fullrank_score set
        """
        # Encode user once
        user_h1 = np.maximum(0, user.features @ self.user_W1 + self.user_b1)
        user_emb = user_h1 @ self.user_W2 + self.user_b2
        user_emb = user_emb / (np.linalg.norm(user_emb) + 1e-8)
        
        for candidate in candidates:
            # Encode item
            item_h1 = np.maximum(0, candidate.item.features @ self.item_W1 + self.item_b1)
            item_emb = item_h1 @ self.item_W2 + self.item_b2
            item_emb = item_emb / (np.linalg.norm(item_emb) + 1e-8)
            
            # Cross features
            cross_features = np.array([
                candidate.retrieval_score,
                candidate.prerank_score,
                float(np.dot(user_emb, item_emb)),
                1.0 if candidate.item.id in user.history else 0.0,
                len(user.history) / 100.0,  # Normalized
                # More cross features
                0.0, 0.0, 0.0, 0.0, 0.0
            ])
            
            # Combine
            combined = np.concatenate([user_emb, item_emb, cross_features])
            
            # Final scoring
            h = np.maximum(0, combined @ self.cross_W + self.cross_b)
            score = float(1 / (1 + np.exp(-(h @ self.final_W + self.final_b))))
            
            candidate.fullrank_score = score
            
        return candidates


# ============================================================
# RE-RANKER
# ============================================================

class DiversityReRanker:
    """
    Re-rank for diversity and business rules.
    
    Implements:
    - Category diversity
    - Freshness boost
    - Position bias correction
    - Business rule application
    """
    
    def __init__(self,
                 diversity_weight: float = 0.3,
                 freshness_weight: float = 0.1,
                 max_per_category: int = 5):
        self.diversity_weight = diversity_weight
        self.freshness_weight = freshness_weight
        self.max_per_category = max_per_category
        
    def rerank(self, candidates: List[Candidate], k: int) -> List[Candidate]:
        """
        Apply re-ranking rules.
        
        Args:
            candidates: Ranked candidates
            k: Final number to return
            
        Returns:
            Re-ranked candidates
        """
        # Sort by fullrank score first
        candidates.sort(key=lambda c: c.fullrank_score, reverse=True)
        
        # Apply diversity
        reranked = self._apply_diversity(candidates, k)
        
        # Apply freshness boost
        reranked = self._apply_freshness_boost(reranked)
        
        # Calculate final scores
        for i, candidate in enumerate(reranked):
            candidate.position = i
            candidate.final_score = self._calculate_final_score(candidate)
            
        return reranked[:k]
    
    def _apply_diversity(self, candidates: List[Candidate], k: int) -> List[Candidate]:
        """
        Apply diversity constraint using MMR-like approach.
        
        Ensures no more than max_per_category from same category.
        """
        selected = []
        category_counts: Dict[str, int] = defaultdict(int)
        
        for candidate in candidates:
            category = candidate.item.metadata.get("category", "unknown")
            
            if category_counts[category] >= self.max_per_category:
                continue
                
            selected.append(candidate)
            category_counts[category] += 1
            
            if len(selected) >= k:
                break
                
        return selected
    
    def _apply_freshness_boost(self, candidates: List[Candidate]) -> List[Candidate]:
        """Apply freshness boost to new items."""
        for candidate in candidates:
            if candidate.item.created_at:
                age_days = (datetime.now() - candidate.item.created_at).days
                if age_days <= 7:
                    # Boost new items
                    boost = (7 - age_days) / 7 * self.freshness_weight
                    candidate.fullrank_score *= (1 + boost)
                    
        # Re-sort
        candidates.sort(key=lambda c: c.fullrank_score, reverse=True)
        return candidates
    
    def _calculate_final_score(self, candidate: Candidate) -> float:
        """Calculate final blended score."""
        base_score = candidate.fullrank_score
        position_bias = 1.0 / (1 + candidate.position * 0.05)  # Mild decay
        
        return base_score * position_bias


class BusinessRuleReRanker:
    """Apply business-specific re-ranking rules."""
    
    def __init__(self):
        self.rules: List[Callable[[Candidate], Tuple[bool, float]]] = []
        self.boost_rules: List[Callable[[Candidate], float]] = []
        self.filter_rules: List[Callable[[Candidate], bool]] = []
        
    def add_boost_rule(self, 
                       rule: Callable[[Candidate], float],
                       description: str = "") -> None:
        """
        Add a score boost rule.
        
        Rule should return a multiplicative boost factor.
        """
        self.boost_rules.append(rule)
        
    def add_filter_rule(self,
                        rule: Callable[[Candidate], bool],
                        description: str = "") -> None:
        """
        Add a filter rule.
        
        Rule should return True to keep, False to remove.
        """
        self.filter_rules.append(rule)
        
    def apply(self, candidates: List[Candidate]) -> List[Candidate]:
        """Apply all rules to candidates."""
        # Apply filters
        filtered = []
        for candidate in candidates:
            keep = all(rule(candidate) for rule in self.filter_rules)
            if keep:
                filtered.append(candidate)
                
        # Apply boosts
        for candidate in filtered:
            total_boost = 1.0
            for rule in self.boost_rules:
                total_boost *= rule(candidate)
            candidate.fullrank_score *= total_boost
            
        # Re-sort
        filtered.sort(key=lambda c: c.fullrank_score, reverse=True)
        return filtered


# ============================================================
# RANKING PIPELINE
# ============================================================

class RankingPipeline:
    """
    Multi-stage ranking pipeline.
    
    Orchestrates:
    1. Candidate Generation (fast, broad)
    2. Pre-Ranking (lightweight filtering)
    3. Full Ranking (deep scoring)
    4. Re-Ranking (diversity, business rules)
    
    Example:
        >>> pipeline = RankingPipeline(user_feature_dim=64, item_feature_dim=64)
        >>> pipeline.add_retrieval_source(EmbeddingSimilaritySource(...))
        >>> results = pipeline.rank(user, k=20)
    """
    
    def __init__(self,
                 user_feature_dim: int,
                 item_feature_dim: int,
                 retrieval_k: int = 1000,
                 prerank_k: int = 200,
                 fullrank_k: int = 50):
        self.candidate_generator = CandidateGenerator()
        self.pre_ranker = PreRanker(feature_dim=item_feature_dim)
        self.full_ranker = FullRanker(
            user_feature_dim=user_feature_dim,
            item_feature_dim=item_feature_dim
        )
        self.diversity_reranker = DiversityReRanker()
        self.business_reranker = BusinessRuleReRanker()
        
        self.retrieval_k = retrieval_k
        self.prerank_k = prerank_k
        self.fullrank_k = fullrank_k
        
    def add_retrieval_source(self, source: RetrievalSource, 
                             weight: float = 1.0) -> None:
        """Add a candidate retrieval source."""
        self.candidate_generator.add_source(source, weight)
        
    def add_business_rule(self,
                          rule_type: str,
                          rule: Callable,
                          description: str = "") -> None:
        """Add a business rule."""
        if rule_type == "boost":
            self.business_reranker.add_boost_rule(rule, description)
        elif rule_type == "filter":
            self.business_reranker.add_filter_rule(rule, description)
            
    def rank(self, user: User, k: int = 20) -> RankingResult:
        """
        Run the full ranking pipeline.
        
        Args:
            user: User to rank for
            k: Number of final recommendations
            
        Returns:
            RankingResult with ranked candidates
        """
        import time
        start_time = time.time()
        
        # Stage 1: Candidate Generation
        candidates = self.candidate_generator.generate(
            user, k_per_source=self.retrieval_k // max(1, len(self.candidate_generator.sources))
        )
        retrieval_count = len(candidates)
        logger.debug(f"Retrieval: {retrieval_count} candidates")
        
        if not candidates:
            return RankingResult(
                candidates=[],
                retrieval_count=0,
                prerank_count=0,
                fullrank_count=0,
                latency_ms=0.0
            )
            
        # Stage 2: Pre-Ranking
        candidates = self.pre_ranker.score(user, candidates)
        candidates = self.pre_ranker.filter(candidates, self.prerank_k)
        prerank_count = len(candidates)
        logger.debug(f"Pre-rank: {prerank_count} candidates")
        
        # Stage 3: Full Ranking
        candidates = self.full_ranker.score(user, candidates)
        candidates.sort(key=lambda c: c.fullrank_score, reverse=True)
        candidates = candidates[:self.fullrank_k]
        fullrank_count = len(candidates)
        logger.debug(f"Full-rank: {fullrank_count} candidates")
        
        # Stage 4: Re-Ranking
        candidates = self.business_reranker.apply(candidates)
        candidates = self.diversity_reranker.rerank(candidates, k)
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RankingResult(
            candidates=candidates,
            retrieval_count=retrieval_count,
            prerank_count=prerank_count,
            fullrank_count=fullrank_count,
            latency_ms=latency_ms,
            metadata={
                "num_sources": len(self.candidate_generator.sources),
                "final_count": len(candidates)
            }
        )


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_usage():
    """Demonstrate Ranking Pipeline usage."""
    
    # Create sample items
    items = []
    for i in range(500):
        items.append(Item(
            id=f"item_{i}",
            features=np.random.randn(64),
            metadata={
                "category": f"category_{i % 10}",
                "price": np.random.uniform(10, 100)
            },
            created_at=datetime.now()
        ))
        
    item_embeddings = np.array([item.features for item in items])
    
    # Popularity scores
    popularity = {item.id: np.random.uniform(0, 1) for item in items}
    
    # Create pipeline
    pipeline = RankingPipeline(
        user_feature_dim=64,
        item_feature_dim=64,
        retrieval_k=200,
        prerank_k=50,
        fullrank_k=20
    )
    
    # Add retrieval sources
    pipeline.add_retrieval_source(
        EmbeddingSimilaritySource(items, item_embeddings),
        weight=1.0
    )
    pipeline.add_retrieval_source(
        PopularitySource(items, popularity),
        weight=0.5
    )
    pipeline.add_retrieval_source(
        RecentSource(items),
        weight=0.3
    )
    
    # Add business rules
    pipeline.add_business_rule(
        "filter",
        lambda c: c.item.metadata.get("price", 0) < 80,  # Filter expensive items
        "Filter items over $80"
    )
    pipeline.add_business_rule(
        "boost",
        lambda c: 1.2 if c.item.metadata.get("price", 50) < 30 else 1.0,  # Boost cheap items
        "Boost items under $30"
    )
    
    # Create user
    user = User(
        id="user_123",
        features=np.random.randn(64),
        history=["item_1", "item_5", "item_10"],
        metadata={"segment": "premium"}
    )
    
    # Run pipeline
    result = pipeline.rank(user, k=10)
    
    print(f"Ranking Pipeline Results:")
    print(f"  Retrieval: {result.retrieval_count} candidates")
    print(f"  Pre-rank: {result.prerank_count} candidates")
    print(f"  Full-rank: {result.fullrank_count} candidates")
    print(f"  Latency: {result.latency_ms:.1f}ms")
    print()
    print(f"Top 10 Recommendations:")
    for i, candidate in enumerate(result.candidates[:10]):
        print(f"  {i+1}. {candidate.item.id}")
        print(f"     Score: {candidate.final_score:.4f}")
        print(f"     Category: {candidate.item.metadata.get('category')}")
        print(f"     Price: ${candidate.item.metadata.get('price', 0):.2f}")
        print()
        
    return pipeline


if __name__ == "__main__":
    example_usage()
