"""
Graph Neural Network Recommender Module
========================================

Implements graph-based recommendation systems inspired by:
- Uber Eats: GraphSAGE for three-sided marketplace optimization
- Pinterest: PinSage for web-scale visual discovery

Features:
- GraphSAGE: Sample and aggregate neighborhoods
- Bipartite Graph Construction: User-Item relationships with edge weights
- Custom Hinge Loss: With low-rank positives for ranking quality
- Two-Tower Architecture: With optional layer sharing
- Cold Start Handling: Content-based fallback for new users/items

References:
- Hamilton et al., "Inductive Representation Learning on Large Graphs" (GraphSAGE)
- Ying et al., "Graph Convolutional Neural Networks for Web-Scale Recommender Systems" (PinSage)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import random
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# DATA STRUCTURES
# ============================================================

class NodeType(Enum):
    """Types of nodes in the bipartite graph."""
    USER = "user"
    ITEM = "item"


@dataclass
class Node:
    """Represents a node in the graph."""
    id: str
    node_type: NodeType
    features: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Represents a weighted edge between nodes."""
    source_id: str
    target_id: str
    weight: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# BIPARTITE GRAPH
# ============================================================

class BipartiteGraph:
    """
    Bipartite graph for user-item relationships.
    
    Supports weighted edges representing interaction strength
    (e.g., watch time, purchase count, rating).
    
    Example:
        >>> graph = BipartiteGraph()
        >>> graph.add_node("user_1", NodeType.USER, np.random.randn(64))
        >>> graph.add_node("item_1", NodeType.ITEM, np.random.randn(64))
        >>> graph.add_edge("user_1", "item_1", weight=0.8)
        >>> neighbors = graph.get_neighbors("user_1")
    """
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: Dict[str, List[Edge]] = defaultdict(list)
        self.reverse_edges: Dict[str, List[Edge]] = defaultdict(list)
        
    def add_node(self, node_id: str, node_type: NodeType, 
                 features: np.ndarray, metadata: Optional[Dict] = None) -> None:
        """Add a node to the graph."""
        self.nodes[node_id] = Node(
            id=node_id,
            node_type=node_type,
            features=features,
            metadata=metadata or {}
        )
        
    def add_edge(self, source_id: str, target_id: str, 
                 weight: float = 1.0, metadata: Optional[Dict] = None) -> None:
        """Add a weighted edge between nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            raise ValueError(f"Both nodes must exist: {source_id}, {target_id}")
            
        edge = Edge(
            source_id=source_id,
            target_id=target_id,
            weight=weight,
            metadata=metadata or {}
        )
        self.edges[source_id].append(edge)
        
        # Reverse edge for bidirectional traversal
        reverse_edge = Edge(
            source_id=target_id,
            target_id=source_id,
            weight=weight,
            metadata=metadata or {}
        )
        self.reverse_edges[target_id].append(reverse_edge)
        
    def get_neighbors(self, node_id: str, 
                      direction: str = "outgoing") -> List[Tuple[str, float]]:
        """
        Get neighbors with edge weights.
        
        Args:
            node_id: Source node ID
            direction: "outgoing", "incoming", or "both"
            
        Returns:
            List of (neighbor_id, weight) tuples
        """
        neighbors = []
        
        if direction in ("outgoing", "both"):
            for edge in self.edges.get(node_id, []):
                neighbors.append((edge.target_id, edge.weight))
                
        if direction in ("incoming", "both"):
            for edge in self.reverse_edges.get(node_id, []):
                neighbors.append((edge.target_id, edge.weight))
                
        return neighbors
    
    def sample_neighbors(self, node_id: str, k: int, 
                         weighted: bool = True) -> List[str]:
        """
        Sample k neighbors using random walks.
        
        This implements importance sampling where higher-weight
        edges are more likely to be sampled.
        
        Args:
            node_id: Source node ID
            k: Number of neighbors to sample
            weighted: If True, sample proportional to edge weights
            
        Returns:
            List of sampled neighbor IDs
        """
        neighbors = self.get_neighbors(node_id, direction="both")
        
        if not neighbors:
            return []
            
        if len(neighbors) <= k:
            return [n_id for n_id, _ in neighbors]
            
        if weighted:
            # Importance sampling based on edge weights
            weights = np.array([w for _, w in neighbors])
            weights = weights / weights.sum()
            indices = np.random.choice(
                len(neighbors), size=k, replace=False, p=weights
            )
            return [neighbors[i][0] for i in indices]
        else:
            # Uniform sampling
            sampled = random.sample(neighbors, k)
            return [n_id for n_id, _ in sampled]
    
    def get_node_features(self, node_id: str) -> np.ndarray:
        """Get feature vector for a node."""
        if node_id not in self.nodes:
            raise ValueError(f"Node not found: {node_id}")
        return self.nodes[node_id].features
    
    def get_users(self) -> List[str]:
        """Get all user node IDs."""
        return [n_id for n_id, node in self.nodes.items() 
                if node.node_type == NodeType.USER]
    
    def get_items(self) -> List[str]:
        """Get all item node IDs."""
        return [n_id for n_id, node in self.nodes.items() 
                if node.node_type == NodeType.ITEM]
    
    def random_walk(self, start_node: str, walk_length: int) -> List[str]:
        """
        Perform a random walk from start node.
        
        Used for neighborhood discovery in PinSage.
        
        Args:
            start_node: Starting node ID
            walk_length: Number of steps in the walk
            
        Returns:
            List of visited node IDs
        """
        walk = [start_node]
        current = start_node
        
        for _ in range(walk_length):
            neighbors = self.get_neighbors(current, direction="both")
            if not neighbors:
                break
                
            # Weighted random selection
            weights = np.array([w for _, w in neighbors])
            weights = weights / weights.sum()
            next_idx = np.random.choice(len(neighbors), p=weights)
            current = neighbors[next_idx][0]
            walk.append(current)
            
        return walk


# ============================================================
# GRAPHSAGE LAYER
# ============================================================

class GraphSAGEAggregator(Enum):
    """Aggregation methods for GraphSAGE."""
    MEAN = "mean"
    POOL = "pool"
    LSTM = "lstm"


class GraphSAGELayer:
    """
    GraphSAGE layer: Sample and Aggregate.
    
    Implements the GraphSAGE algorithm for inductive
    representation learning on graphs.
    
    Forward pass:
    1. Sample neighbors for each node
    2. Aggregate neighbor features
    3. Concatenate with self features
    4. Apply linear transformation + nonlinearity
    
    Args:
        input_dim: Input feature dimension
        output_dim: Output embedding dimension
        aggregator: Aggregation method (mean, pool, lstm)
        num_samples: Number of neighbors to sample
        activation: Activation function
    """
    
    def __init__(self, input_dim: int, output_dim: int,
                 aggregator: GraphSAGEAggregator = GraphSAGEAggregator.MEAN,
                 num_samples: int = 10,
                 activation: str = "relu",
                 dropout: float = 0.0):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator = aggregator
        self.num_samples = num_samples
        self.dropout = dropout
        
        # Initialize weights (He initialization)
        scale = np.sqrt(2.0 / input_dim)
        
        # Weight for self features
        self.W_self = np.random.randn(input_dim, output_dim) * scale
        
        # Weight for aggregated neighbor features
        self.W_neigh = np.random.randn(input_dim, output_dim) * scale
        
        # Bias
        self.b = np.zeros(output_dim)
        
        # For pool aggregator
        if aggregator == GraphSAGEAggregator.POOL:
            self.W_pool = np.random.randn(input_dim, input_dim) * scale
            self.b_pool = np.zeros(input_dim)
            
        # Activation function
        self.activation = activation
        
        # Cache for backward pass
        self.cache = {}
        
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == "relu":
            return np.maximum(0, x)
        elif self.activation == "tanh":
            return np.tanh(x)
        elif self.activation == "sigmoid":
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        else:
            return x
            
    def _aggregate_mean(self, neighbor_features: List[np.ndarray]) -> np.ndarray:
        """Mean aggregation."""
        if not neighbor_features:
            return np.zeros(self.input_dim)
        return np.mean(neighbor_features, axis=0)
    
    def _aggregate_pool(self, neighbor_features: List[np.ndarray]) -> np.ndarray:
        """Max pooling aggregation."""
        if not neighbor_features:
            return np.zeros(self.input_dim)
            
        # Apply element-wise max-pooling MLP
        transformed = []
        for feat in neighbor_features:
            h = feat @ self.W_pool + self.b_pool
            h = np.maximum(0, h)  # ReLU
            transformed.append(h)
            
        return np.max(transformed, axis=0)
    
    def aggregate(self, neighbor_features: List[np.ndarray]) -> np.ndarray:
        """Aggregate neighbor features based on aggregator type."""
        if self.aggregator == GraphSAGEAggregator.MEAN:
            return self._aggregate_mean(neighbor_features)
        elif self.aggregator == GraphSAGEAggregator.POOL:
            return self._aggregate_pool(neighbor_features)
        else:
            return self._aggregate_mean(neighbor_features)
    
    def forward(self, graph: BipartiteGraph, node_ids: List[str],
                training: bool = True) -> Dict[str, np.ndarray]:
        """
        Forward pass for a batch of nodes.
        
        Args:
            graph: The bipartite graph
            node_ids: List of node IDs to embed
            training: Whether in training mode
            
        Returns:
            Dictionary mapping node_id -> embedding
        """
        embeddings = {}
        
        for node_id in node_ids:
            # Get self features
            self_feat = graph.get_node_features(node_id)
            
            # Sample and get neighbor features
            sampled_neighbors = graph.sample_neighbors(
                node_id, self.num_samples, weighted=True
            )
            
            neighbor_feats = [
                graph.get_node_features(n_id) 
                for n_id in sampled_neighbors
            ]
            
            # Aggregate
            agg_feat = self.aggregate(neighbor_feats)
            
            # Apply dropout during training
            if training and self.dropout > 0:
                mask = np.random.binomial(1, 1 - self.dropout, self_feat.shape)
                self_feat = self_feat * mask / (1 - self.dropout)
                mask = np.random.binomial(1, 1 - self.dropout, agg_feat.shape)
                agg_feat = agg_feat * mask / (1 - self.dropout)
            
            # Linear transformation
            h_self = self_feat @ self.W_self
            h_neigh = agg_feat @ self.W_neigh
            
            # Combine and apply activation
            h = h_self + h_neigh + self.b
            h = self._activate(h)
            
            # Normalize (L2 normalization for stable training)
            h = h / (np.linalg.norm(h) + 1e-8)
            
            embeddings[node_id] = h
            
        # Cache for backward pass
        if training:
            self.cache["embeddings"] = embeddings
            
        return embeddings
    
    def get_params(self) -> Dict[str, np.ndarray]:
        """Get layer parameters."""
        params = {
            "W_self": self.W_self,
            "W_neigh": self.W_neigh,
            "b": self.b
        }
        if self.aggregator == GraphSAGEAggregator.POOL:
            params["W_pool"] = self.W_pool
            params["b_pool"] = self.b_pool
        return params
    
    def set_params(self, params: Dict[str, np.ndarray]) -> None:
        """Set layer parameters."""
        self.W_self = params["W_self"]
        self.W_neigh = params["W_neigh"]
        self.b = params["b"]
        if "W_pool" in params:
            self.W_pool = params["W_pool"]
            self.b_pool = params["b_pool"]


# ============================================================
# GNN RECOMMENDER
# ============================================================

class GNNRecommender:
    """
    Graph Neural Network Recommender System.
    
    Implements multi-layer GraphSAGE for generating node embeddings,
    then uses these embeddings for similarity-based recommendations.
    
    Architecture:
    1. Multiple GraphSAGE layers for embedding generation
    2. Two-Tower architecture for user-item scoring
    3. Custom ranking loss with low-rank positives
    
    Example:
        >>> recommender = GNNRecommender(feature_dim=64, embedding_dim=128)
        >>> recommender.add_graphsage_layer(64, 128)
        >>> recommender.add_graphsage_layer(128, 128)
        >>> embeddings = recommender.generate_embeddings(graph)
        >>> recommendations = recommender.recommend("user_1", k=10)
    """
    
    def __init__(self, feature_dim: int, embedding_dim: int,
                 num_layers: int = 2, dropout: float = 0.1):
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        
        # GraphSAGE layers
        self.layers: List[GraphSAGELayer] = []
        
        # Build default layers
        dims = [feature_dim] + [embedding_dim] * num_layers
        for i in range(num_layers):
            layer = GraphSAGELayer(
                input_dim=dims[i],
                output_dim=dims[i + 1],
                aggregator=GraphSAGEAggregator.MEAN,
                num_samples=10,
                dropout=dropout
            )
            self.layers.append(layer)
            
        # Cached embeddings
        self.user_embeddings: Dict[str, np.ndarray] = {}
        self.item_embeddings: Dict[str, np.ndarray] = {}
        
        # Training state
        self.graph: Optional[BipartiteGraph] = None
        
    def add_graphsage_layer(self, input_dim: int, output_dim: int,
                            aggregator: GraphSAGEAggregator = GraphSAGEAggregator.MEAN,
                            num_samples: int = 10) -> None:
        """Add a GraphSAGE layer to the model."""
        layer = GraphSAGELayer(
            input_dim=input_dim,
            output_dim=output_dim,
            aggregator=aggregator,
            num_samples=num_samples,
            dropout=self.dropout
        )
        self.layers.append(layer)
        
    def generate_embeddings(self, graph: BipartiteGraph,
                            training: bool = False) -> Tuple[Dict, Dict]:
        """
        Generate embeddings for all nodes using GraphSAGE layers.
        
        Args:
            graph: The user-item bipartite graph
            training: Whether in training mode
            
        Returns:
            (user_embeddings, item_embeddings) dictionaries
        """
        self.graph = graph
        
        all_nodes = list(graph.nodes.keys())
        
        # Forward through all layers
        current_embeddings = {
            n_id: graph.get_node_features(n_id) 
            for n_id in all_nodes
        }
        
        for layer in self.layers:
            # Update node features in graph temporarily
            for n_id, emb in current_embeddings.items():
                graph.nodes[n_id].features = emb
                
            current_embeddings = layer.forward(graph, all_nodes, training)
            
        # Separate user and item embeddings
        self.user_embeddings = {
            n_id: emb for n_id, emb in current_embeddings.items()
            if graph.nodes[n_id].node_type == NodeType.USER
        }
        self.item_embeddings = {
            n_id: emb for n_id, emb in current_embeddings.items()
            if graph.nodes[n_id].node_type == NodeType.ITEM
        }
        
        return self.user_embeddings, self.item_embeddings
    
    def score(self, user_id: str, item_id: str) -> float:
        """
        Compute affinity score between user and item.
        
        Uses dot product similarity between embeddings.
        """
        if user_id not in self.user_embeddings:
            raise ValueError(f"User not found: {user_id}")
        if item_id not in self.item_embeddings:
            raise ValueError(f"Item not found: {item_id}")
            
        user_emb = self.user_embeddings[user_id]
        item_emb = self.item_embeddings[item_id]
        
        return float(np.dot(user_emb, item_emb))
    
    def recommend(self, user_id: str, k: int = 10,
                  exclude_seen: bool = True) -> List[Tuple[str, float]]:
        """
        Generate top-k recommendations for a user.
        
        Args:
            user_id: User ID to recommend for
            k: Number of recommendations
            exclude_seen: Whether to exclude items user has interacted with
            
        Returns:
            List of (item_id, score) tuples sorted by score descending
        """
        if user_id not in self.user_embeddings:
            raise ValueError(f"User not found: {user_id}")
            
        user_emb = self.user_embeddings[user_id]
        
        # Get seen items to exclude
        seen_items = set()
        if exclude_seen and self.graph is not None:
            neighbors = self.graph.get_neighbors(user_id, direction="outgoing")
            seen_items = {n_id for n_id, _ in neighbors}
            
        # Score all items
        scores = []
        for item_id, item_emb in self.item_embeddings.items():
            if item_id in seen_items:
                continue
            score = float(np.dot(user_emb, item_emb))
            scores.append((item_id, score))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:k]
    
    def recommend_batch(self, user_ids: List[str], k: int = 10) -> Dict[str, List]:
        """Generate recommendations for multiple users."""
        return {
            user_id: self.recommend(user_id, k)
            for user_id in user_ids
        }


# ============================================================
# TWO-TOWER ARCHITECTURE
# ============================================================

class TwoTowerRanker:
    """
    Two-Tower Architecture for Recommendation Ranking.
    
    Implements the two-tower model used by Uber Eats and YouTube:
    - User Tower: Encodes user features + context
    - Item Tower: Encodes item features
    - Scoring: Dot product of tower outputs
    
    Features:
    - Optional layer sharing between towers
    - Support for categorical and continuous features
    - Temperature-scaled softmax for training
    
    Args:
        user_feature_dim: Dimension of user features
        item_feature_dim: Dimension of item features
        embedding_dim: Final embedding dimension
        hidden_dims: Hidden layer dimensions
        share_layers: Whether to share bottom layers
    """
    
    def __init__(self, user_feature_dim: int, item_feature_dim: int,
                 embedding_dim: int = 128, hidden_dims: List[int] = None,
                 share_layers: bool = False, dropout: float = 0.3):
        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [256, 128]
        self.share_layers = share_layers
        self.dropout = dropout
        
        # Initialize user tower
        self.user_layers = self._build_tower(user_feature_dim)
        
        # Initialize item tower (shared or separate)
        if share_layers:
            # Project both to same dim first
            self.user_projection = self._init_layer(user_feature_dim, self.hidden_dims[0])
            self.item_projection = self._init_layer(item_feature_dim, self.hidden_dims[0])
            self.shared_layers = self._build_tower(self.hidden_dims[0], skip_first=True)
            self.item_layers = None
        else:
            self.item_layers = self._build_tower(item_feature_dim)
            
        # Temperature for softmax
        self.temperature = 0.07
        
    def _init_layer(self, in_dim: int, out_dim: int) -> Dict[str, np.ndarray]:
        """Initialize a single layer."""
        scale = np.sqrt(2.0 / in_dim)
        return {
            "W": np.random.randn(in_dim, out_dim) * scale,
            "b": np.zeros(out_dim)
        }
        
    def _build_tower(self, input_dim: int, 
                     skip_first: bool = False) -> List[Dict[str, np.ndarray]]:
        """Build tower layers."""
        layers = []
        
        dims = [input_dim] + self.hidden_dims + [self.embedding_dim]
        start_idx = 1 if skip_first else 0
        
        for i in range(start_idx, len(dims) - 1):
            layers.append(self._init_layer(dims[i], dims[i + 1]))
            
        return layers
    
    def _forward_tower(self, x: np.ndarray, 
                       layers: List[Dict], 
                       training: bool = True) -> np.ndarray:
        """Forward pass through a tower."""
        h = x
        
        for i, layer in enumerate(layers):
            h = h @ layer["W"] + layer["b"]
            
            # ReLU for all but last layer
            if i < len(layers) - 1:
                h = np.maximum(0, h)
                
                # Dropout during training
                if training and self.dropout > 0:
                    mask = np.random.binomial(1, 1 - self.dropout, h.shape)
                    h = h * mask / (1 - self.dropout)
                    
        # L2 normalize output
        h = h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-8)
        
        return h
    
    def encode_user(self, user_features: np.ndarray, 
                    training: bool = False) -> np.ndarray:
        """
        Encode user features through user tower.
        
        Args:
            user_features: (batch_size, user_feature_dim)
            training: Whether in training mode
            
        Returns:
            User embeddings (batch_size, embedding_dim)
        """
        if self.share_layers:
            h = user_features @ self.user_projection["W"] + self.user_projection["b"]
            h = np.maximum(0, h)
            return self._forward_tower(h, self.shared_layers, training)
        else:
            return self._forward_tower(user_features, self.user_layers, training)
    
    def encode_item(self, item_features: np.ndarray,
                    training: bool = False) -> np.ndarray:
        """
        Encode item features through item tower.
        
        Args:
            item_features: (batch_size, item_feature_dim)
            training: Whether in training mode
            
        Returns:
            Item embeddings (batch_size, embedding_dim)
        """
        if self.share_layers:
            h = item_features @ self.item_projection["W"] + self.item_projection["b"]
            h = np.maximum(0, h)
            return self._forward_tower(h, self.shared_layers, training)
        else:
            return self._forward_tower(item_features, self.item_layers, training)
    
    def score(self, user_features: np.ndarray, 
              item_features: np.ndarray) -> np.ndarray:
        """
        Compute scores for user-item pairs.
        
        Args:
            user_features: (batch_size, user_feature_dim)
            item_features: (batch_size, item_feature_dim)
            
        Returns:
            Scores (batch_size,)
        """
        user_emb = self.encode_user(user_features, training=False)
        item_emb = self.encode_item(item_features, training=False)
        
        # Dot product
        scores = np.sum(user_emb * item_emb, axis=-1)
        
        return scores
    
    def score_all_items(self, user_features: np.ndarray,
                        all_item_embeddings: np.ndarray) -> np.ndarray:
        """
        Score a user against all items (for inference).
        
        Args:
            user_features: (user_feature_dim,) single user
            all_item_embeddings: (num_items, embedding_dim)
            
        Returns:
            Scores (num_items,)
        """
        user_emb = self.encode_user(user_features.reshape(1, -1), training=False)
        scores = all_item_embeddings @ user_emb.T
        return scores.flatten()


# ============================================================
# RANKING LOSS FUNCTIONS
# ============================================================

class RankingLoss:
    """
    Ranking loss functions for recommendation training.
    
    Implements:
    - BPR (Bayesian Personalized Ranking)
    - Hinge Loss with low-rank positives (Uber Eats style)
    - InfoNCE / Contrastive Loss
    """
    
    @staticmethod
    def bpr_loss(positive_scores: np.ndarray, 
                 negative_scores: np.ndarray) -> float:
        """
        Bayesian Personalized Ranking loss.
        
        Maximizes the difference between positive and negative scores.
        
        Args:
            positive_scores: Scores for positive (interacted) items
            negative_scores: Scores for negative (sampled) items
            
        Returns:
            BPR loss value
        """
        diff = positive_scores - negative_scores
        loss = -np.mean(np.log(1 / (1 + np.exp(-diff)) + 1e-8))
        return float(loss)
    
    @staticmethod
    def hinge_loss_with_low_rank_positives(
        scores: np.ndarray,
        positive_mask: np.ndarray,
        rank_weights: np.ndarray,
        margin: float = 1.0
    ) -> float:
        """
        Hinge loss with low-rank positive handling.
        
        This loss penalizes the model not just for ranking negatives
        above positives, but also for ranking lower-engagement positives
        above higher-engagement positives.
        
        As used by Uber Eats to distinguish between:
        - Restaurants ordered "sometimes" vs "frequently"
        
        Args:
            scores: Model scores for all items
            positive_mask: Binary mask for positive items
            rank_weights: Weights indicating engagement level (higher = more engaged)
            margin: Hinge margin
            
        Returns:
            Loss value
        """
        positive_scores = scores[positive_mask > 0]
        positive_weights = rank_weights[positive_mask > 0]
        negative_scores = scores[positive_mask == 0]
        
        total_loss = 0.0
        count = 0
        
        # Loss for positives vs negatives
        for pos_score, pos_weight in zip(positive_scores, positive_weights):
            for neg_score in negative_scores:
                hinge = np.maximum(0, margin - (pos_score - neg_score))
                total_loss += pos_weight * hinge
                count += 1
                
        # Loss for low-rank vs high-rank positives
        for i, (score_i, weight_i) in enumerate(zip(positive_scores, positive_weights)):
            for j, (score_j, weight_j) in enumerate(zip(positive_scores, positive_weights)):
                if weight_i > weight_j:  # i should rank higher than j
                    hinge = np.maximum(0, margin - (score_i - score_j))
                    total_loss += (weight_i - weight_j) * hinge
                    count += 1
                    
        return total_loss / max(count, 1)
    
    @staticmethod
    def infonce_loss(query_emb: np.ndarray,
                     positive_emb: np.ndarray,
                     negative_embs: np.ndarray,
                     temperature: float = 0.07) -> float:
        """
        InfoNCE contrastive loss.
        
        Used for training embedding models where we want
        the query to be close to positives and far from negatives.
        
        Args:
            query_emb: Query embedding (e.g., user)
            positive_emb: Positive item embedding
            negative_embs: Negative item embeddings (num_negatives, dim)
            temperature: Temperature for softmax
            
        Returns:
            InfoNCE loss value
        """
        # Compute similarities
        pos_sim = np.dot(query_emb, positive_emb) / temperature
        neg_sims = negative_embs @ query_emb / temperature
        
        # Log-sum-exp for numerical stability
        all_sims = np.concatenate([[pos_sim], neg_sims])
        max_sim = np.max(all_sims)
        
        log_sum_exp = max_sim + np.log(np.sum(np.exp(all_sims - max_sim)))
        
        loss = -pos_sim + log_sum_exp
        
        return float(loss)


# ============================================================
# COLD START HANDLER
# ============================================================

class ColdStartHandler:
    """
    Handle cold start problems for new users and items.
    
    Strategies:
    - New Users: Use demographic/context features for initial recommendations
    - New Items: Use content features (metadata, embeddings) for similarity
    - Popularity Fallback: Recommend trending items
    """
    
    def __init__(self, popularity_window_days: int = 7):
        self.popularity_window_days = popularity_window_days
        
        # Store item popularity scores
        self.item_popularity: Dict[str, float] = {}
        
        # Content embeddings for new items
        self.content_embeddings: Dict[str, np.ndarray] = {}
        
        # Demographic preferences
        self.demographic_preferences: Dict[str, Dict[str, float]] = {}
        
    def update_popularity(self, item_counts: Dict[str, int]) -> None:
        """
        Update item popularity scores.
        
        Args:
            item_counts: Dictionary of item_id -> interaction count
        """
        total = sum(item_counts.values()) + 1
        self.item_popularity = {
            item_id: count / total
            for item_id, count in item_counts.items()
        }
        
    def set_content_embedding(self, item_id: str, embedding: np.ndarray) -> None:
        """Set content-based embedding for an item."""
        self.content_embeddings[item_id] = embedding
        
    def set_demographic_preferences(self, 
                                    demographic_key: str,
                                    item_preferences: Dict[str, float]) -> None:
        """
        Set item preferences for a demographic group.
        
        Args:
            demographic_key: e.g., "age_18_24_usa"
            item_preferences: item_id -> preference score
        """
        self.demographic_preferences[demographic_key] = item_preferences
        
    def recommend_for_new_user(self, 
                               demographic_info: Optional[Dict] = None,
                               k: int = 10) -> List[Tuple[str, float]]:
        """
        Generate recommendations for a new user.
        
        Args:
            demographic_info: Optional demographic features
            k: Number of recommendations
            
        Returns:
            List of (item_id, score) tuples
        """
        scores = defaultdict(float)
        
        # Try demographic-based recommendations
        if demographic_info:
            # Build demographic key
            age_group = demographic_info.get("age_group", "unknown")
            country = demographic_info.get("country", "unknown")
            demo_key = f"{age_group}_{country}"
            
            if demo_key in self.demographic_preferences:
                for item_id, pref in self.demographic_preferences[demo_key].items():
                    scores[item_id] += pref * 0.6  # Weight demographic signal
                    
        # Add popularity scores
        for item_id, pop in self.item_popularity.items():
            scores[item_id] += pop * 0.4  # Weight popularity
            
        # Sort and return top-k
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_items[:k]
    
    def find_similar_items(self, new_item_embedding: np.ndarray,
                           k: int = 10) -> List[Tuple[str, float]]:
        """
        Find similar items for a new item using content embeddings.
        
        Args:
            new_item_embedding: Content embedding for new item
            k: Number of similar items
            
        Returns:
            List of (item_id, similarity) tuples
        """
        if not self.content_embeddings:
            return []
            
        similarities = []
        
        for item_id, emb in self.content_embeddings.items():
            sim = np.dot(new_item_embedding, emb) / (
                np.linalg.norm(new_item_embedding) * np.linalg.norm(emb) + 1e-8
            )
            similarities.append((item_id, float(sim)))
            
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]


# ============================================================
# EVALUATION METRICS
# ============================================================

class RecommenderMetrics:
    """
    Evaluation metrics for recommendation systems.
    
    Implements:
    - Recall@k: Fraction of relevant items in top-k
    - NDCG@k: Normalized Discounted Cumulative Gain
    - MRR: Mean Reciprocal Rank
    - Hit Rate: Whether at least one relevant item in top-k
    """
    
    @staticmethod
    def recall_at_k(recommended: List[str], 
                    relevant: Set[str], 
                    k: int) -> float:
        """
        Compute Recall@k.
        
        Args:
            recommended: Ordered list of recommended item IDs
            relevant: Set of relevant (ground truth) item IDs
            k: Cutoff
            
        Returns:
            Recall score
        """
        if not relevant:
            return 0.0
            
        top_k = set(recommended[:k])
        hits = len(top_k & relevant)
        
        return hits / len(relevant)
    
    @staticmethod
    def ndcg_at_k(recommended: List[str],
                  relevant: Set[str],
                  k: int) -> float:
        """
        Compute NDCG@k.
        
        Args:
            recommended: Ordered list of recommended item IDs
            relevant: Set of relevant item IDs
            k: Cutoff
            
        Returns:
            NDCG score
        """
        if not relevant:
            return 0.0
            
        # DCG
        dcg = 0.0
        for i, item_id in enumerate(recommended[:k]):
            if item_id in relevant:
                dcg += 1.0 / np.log2(i + 2)  # +2 because i is 0-indexed
                
        # Ideal DCG
        ideal_hits = min(len(relevant), k)
        idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_hits))
        
        return dcg / idcg if idcg > 0 else 0.0
    
    @staticmethod
    def mrr(recommended: List[str], relevant: Set[str]) -> float:
        """
        Compute Mean Reciprocal Rank.
        
        Args:
            recommended: Ordered list of recommended item IDs
            relevant: Set of relevant item IDs
            
        Returns:
            MRR score
        """
        for i, item_id in enumerate(recommended):
            if item_id in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def hit_rate_at_k(recommended: List[str],
                      relevant: Set[str],
                      k: int) -> float:
        """
        Compute Hit Rate@k (binary: 1 if any hit, 0 otherwise).
        
        Args:
            recommended: Ordered list of recommended item IDs
            relevant: Set of relevant item IDs
            k: Cutoff
            
        Returns:
            1.0 if hit, 0.0 otherwise
        """
        top_k = set(recommended[:k])
        return 1.0 if len(top_k & relevant) > 0 else 0.0
    
    @staticmethod
    def evaluate_recommendations(
        recommendations: Dict[str, List[Tuple[str, float]]],
        ground_truth: Dict[str, Set[str]],
        k: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate recommendations for multiple users.
        
        Args:
            recommendations: user_id -> [(item_id, score), ...]
            ground_truth: user_id -> set of relevant item_ids
            k: Evaluation cutoff
            
        Returns:
            Dictionary of metric_name -> average value
        """
        metrics = {
            "recall@k": [],
            "ndcg@k": [],
            "mrr": [],
            "hit_rate@k": []
        }
        
        for user_id, recs in recommendations.items():
            if user_id not in ground_truth:
                continue
                
            relevant = ground_truth[user_id]
            rec_items = [item_id for item_id, _ in recs]
            
            metrics["recall@k"].append(
                RecommenderMetrics.recall_at_k(rec_items, relevant, k)
            )
            metrics["ndcg@k"].append(
                RecommenderMetrics.ndcg_at_k(rec_items, relevant, k)
            )
            metrics["mrr"].append(
                RecommenderMetrics.mrr(rec_items, relevant)
            )
            metrics["hit_rate@k"].append(
                RecommenderMetrics.hit_rate_at_k(rec_items, relevant, k)
            )
            
        return {
            name: np.mean(values) if values else 0.0
            for name, values in metrics.items()
        }


# ============================================================
# EXAMPLE USAGE
# ============================================================

def example_usage():
    """Demonstrate GNN Recommender usage."""
    
    # Create a sample bipartite graph
    graph = BipartiteGraph()
    
    # Add users with features
    for i in range(100):
        user_features = np.random.randn(64)
        graph.add_node(f"user_{i}", NodeType.USER, user_features)
        
    # Add items with features
    for i in range(500):
        item_features = np.random.randn(64)
        graph.add_node(f"item_{i}", NodeType.ITEM, item_features)
        
    # Add edges (interactions)
    for i in range(100):
        # Each user interacts with ~10 random items
        items = np.random.choice(500, size=10, replace=False)
        for item_idx in items:
            weight = np.random.uniform(0.1, 1.0)  # Interaction strength
            graph.add_edge(f"user_{i}", f"item_{item_idx}", weight=weight)
            
    # Create and train recommender
    recommender = GNNRecommender(
        feature_dim=64,
        embedding_dim=128,
        num_layers=2
    )
    
    # Generate embeddings
    user_embs, item_embs = recommender.generate_embeddings(graph)
    
    print(f"Generated {len(user_embs)} user embeddings")
    print(f"Generated {len(item_embs)} item embeddings")
    
    # Get recommendations for a user
    recs = recommender.recommend("user_0", k=10)
    print(f"\nTop 10 recommendations for user_0:")
    for item_id, score in recs:
        print(f"  {item_id}: {score:.4f}")
        
    return recommender


if __name__ == "__main__":
    example_usage()
