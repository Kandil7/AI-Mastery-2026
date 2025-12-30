"""
Vector Database Module

This module implements vector storage and similarity search capabilities,
including HNSW (Hierarchical Navigable Small World) and basic vector operations.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional, Union
from collections import defaultdict
import heapq
import math
from dataclasses import dataclass
import pickle
import os


@dataclass
class VectorItem:
    """Represents an item with vector embedding and metadata."""
    id: str
    vector: np.ndarray
    metadata: Dict[str, Any]
    created_at: float


class VectorIndex:
    """Base class for vector indexing."""
    
    def __init__(self, dim: int, metric: str = 'cosine'):
        """
        Initialize vector index.
        
        Args:
            dim: Dimension of vectors
            metric: Distance metric ('cosine', 'euclidean', 'dot')
        """
        self.dim = dim
        self.metric = metric
        self.items: Dict[str, VectorItem] = {}
    
    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any] = None):
        """Add a vector to the index."""
        if metadata is None:
            metadata = {}
        
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension {vector.shape[0]} does not match index dimension {self.dim}")
        
        # Normalize vector for cosine similarity
        if self.metric == 'cosine':
            vector = vector / (np.linalg.norm(vector) + 1e-12)
        
        self.items[id] = VectorItem(
            id=id,
            vector=vector,
            metadata=metadata,
            created_at=time.time()
        )
    
    def remove(self, id: str):
        """Remove a vector from the index."""
        if id in self.items:
            del self.items[id]
    
    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate similarity between two vectors."""
        if self.metric == 'cosine':
            return np.dot(v1, v2)  # Already normalized
        elif self.metric == 'euclidean':
            return -np.linalg.norm(v1 - v2)  # Negative for similarity (higher is more similar)
        elif self.metric == 'dot':
            return np.dot(v1, v2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for k nearest neighbors."""
        raise NotImplementedError("Subclasses must implement search method")
    
    def save(self, path: str):
        """Save the index to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        """Load the index from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


class LinearVectorIndex(VectorIndex):
    """Simple linear search index for small datasets."""
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Linear search for k nearest neighbors."""
        if self.metric == 'cosine':
            query = query / (np.linalg.norm(query) + 1e-12)
        
        # Calculate similarities with all vectors
        similarities = []
        for item_id, item in self.items.items():
            sim = self.similarity(query, item.vector)
            similarities.append((item_id, sim, item.metadata))
        
        # Sort by similarity (descending) and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:k]


class HNSWNode:
    """Node in HNSW graph."""
    
    def __init__(self, id: str, vector: np.ndarray, level: int):
        self.id = id
        self.vector = vector
        self.level = level
        self.connections: Dict[int, List[Tuple[str, float]]] = defaultdict(list)  # level -> [(neighbor_id, distance)]


class HNSWIndex(VectorIndex):
    """
    Hierarchical Navigable Small World (HNSW) index implementation.
    This provides efficient approximate nearest neighbor search.
    """
    
    def __init__(self, dim: int, metric: str = 'cosine', max_level: int = 16, 
                 ef_construction: int = 200, M: int = 16):
        """
        Initialize HNSW index.
        
        Args:
            dim: Dimension of vectors
            metric: Distance metric
            max_level: Maximum level in the hierarchy
            ef_construction: Size of the dynamic list for construction
            M: Number of connections
        """
        super().__init__(dim, metric)
        self.max_level = max_level
        self.ef_construction = ef_construction
        self.M = M
        self.M0 = M * 2
        self.level_mult = 1 / np.log(M)
        
        self.nodes: Dict[str, HNSWNode] = {}
        self.entry_point: Optional[str] = None
        self.current_max_level = -1
        
        # Random state for level generation
        self.random_state = np.random.RandomState(42)
    
    def _random_level(self) -> int:
        """Generate a random level for a new node."""
        return int(-np.log(self.random_state.random()) * self.level_mult)
    
    def _distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate distance between two vectors."""
        if self.metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            return 1 - np.dot(v1, v2)
        elif self.metric == 'euclidean':
            return np.linalg.norm(v1 - v2)
        elif self.metric == 'dot':
            # Negative dot product as distance
            return -np.dot(v1, v2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")
    
    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any] = None):
        """Add a vector to the HNSW index."""
        if metadata is None:
            metadata = {}
        
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimension {vector.shape[0]} does not match index dimension {self.dim}")
        
        # Normalize vector for cosine similarity
        if self.metric == 'cosine':
            vector = vector / (np.linalg.norm(vector) + 1e-12)
        
        # Generate random level for the new node
        level = self._random_level()
        level = min(level, self.max_level)
        
        # Create new node
        new_node = HNSWNode(id, vector, level)
        self.nodes[id] = new_node
        
        # Add to base index
        super().add(id, vector, metadata)
        
        # Update entry point if this is the first node or has higher level
        if self.entry_point is None or level > self.current_max_level:
            self.entry_point = id
            self.current_max_level = level
        
        # Insert the new node into the graph structure
        self._insert_node(new_node)
    
    def _insert_node(self, new_node: HNSWNode):
        """Insert a new node into the HNSW graph."""
        if len(self.nodes) == 1:
            # First node, nothing to connect to
            return
        
        # Find nearest neighbors at each level
        ep_id = self.entry_point
        ep_vector = self.nodes[ep_id].vector
        
        # Start from the highest level and go down
        for level in range(self.current_max_level, -1, -1):
            # Search for nearest neighbors at this level
            ep_id = self._search_layer(new_node.vector, ep_id, level, 1)[0][0]
            
            # Connect the new node to neighbors at this level
            if level <= new_node.level:
                # Get nearest neighbors to connect to
                neighbors = self._select_neighbors(new_node.vector, self.nodes[ep_id].vector, level)
                
                # Connect new node to neighbors
                for neighbor_id, _ in neighbors:
                    neighbor_node = self.nodes[neighbor_id]
                    dist = self._distance(new_node.vector, neighbor_node.vector)
                    
                    # Add connection from new node to neighbor
                    new_node.connections[level].append((neighbor_id, dist))
                    
                    # Add connection from neighbor to new node
                    neighbor_node.connections[level].append((new_node.id, dist))
                
                # Limit number of connections
                if len(new_node.connections[level]) > self.M:
                    # Prune connections to maintain graph quality
                    self._prune_connections(new_node, level)
    
    def _search_layer(self, query: np.ndarray, ep_id: str, level: int, k: int) -> List[Tuple[str, float]]:
        """Search for nearest neighbors in a specific level."""
        visited = set()
        candidate_set = [(self._distance(query, self.nodes[ep_id].vector), ep_id)]
        heapq.heapify(candidate_set)
        
        visited.add(ep_id)
        results = [(self._distance(query, self.nodes[ep_id].vector), ep_id)]
        
        while candidate_set:
            dist, current_id = heapq.heappop(candidate_set)
            
            # Check if we can terminate early
            if dist > results[0][0] and len(results) >= k:
                break
            
            current_node = self.nodes[current_id]
            
            # Check neighbors at this level
            for neighbor_id, _ in current_node.connections[level]:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    neighbor_dist = self._distance(query, self.nodes[neighbor_id].vector)
                    
                    if len(results) < k:
                        heapq.heappush(results, (-neighbor_dist, neighbor_id))
                    elif neighbor_dist < -results[0][0]:
                        heapq.heapreplace(results, (-neighbor_dist, neighbor_id))
                    
                    heapq.heappush(candidate_set, (neighbor_dist, neighbor_id))
        
        # Convert to positive distances and return top k
        results = [(-dist, node_id) for dist, node_id in results]
        results.sort()  # Sort by distance (ascending)
        return results[:k]
    
    def _select_neighbors(self, query: np.ndarray, ep_vector: np.ndarray, level: int) -> List[Tuple[str, float]]:
        """Select neighbors to connect to."""
        # For simplicity, return the closest M connections
        # In a full implementation, this would use more sophisticated selection
        neighbors = []
        for node_id, node in self.nodes.items():
            if node.level >= level:
                dist = self._distance(query, node.vector)
                neighbors.append((node_id, dist))
        
        # Sort by distance and return top M
        neighbors.sort(key=lambda x: x[1])
        return neighbors[:self.M]
    
    def _prune_connections(self, node: HNSWNode, level: int):
        """Prune connections to maintain graph quality."""
        # Simple pruning: keep only the M closest connections
        connections = node.connections[level]
        if len(connections) <= self.M:
            return
        
        # Calculate distances to all connections and keep closest M
        connections_with_dist = []
        for neighbor_id, _ in connections:
            neighbor_node = self.nodes[neighbor_id]
            dist = self._distance(node.vector, neighbor_node.vector)
            connections_with_dist.append((neighbor_id, dist))
        
        # Sort by distance and keep top M
        connections_with_dist.sort(key=lambda x: x[1])
        node.connections[level] = connections_with_dist[:self.M]
    
    def search(self, query: np.ndarray, k: int = 10) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for k nearest neighbors using HNSW."""
        if not self.nodes:
            return []
        
        if self.metric == 'cosine':
            query = query / (np.linalg.norm(query) + 1e-12)
        
        # Start from entry point
        ep_id = self.entry_point
        ep_vector = self.nodes[ep_id].vector
        
        # Search from top level down
        for level in range(self.current_max_level, -1, -1):
            # Search in this level
            ep_id = self._search_layer(query, ep_id, level, 1)[0][0]
        
        # Now perform search at the base level to get k nearest
        results = self._search_layer(query, ep_id, 0, k)
        
        # Convert to the expected format: (id, distance, metadata)
        final_results = []
        for dist, node_id in results:
            item = self.items[node_id]
            # Convert distance back to similarity (higher is better)
            if self.metric == 'cosine':
                similarity = 1 - dist  # Convert cosine distance back to similarity
            else:
                similarity = -dist  # Negative distance for similarity
            final_results.append((item.id, similarity, item.metadata))
        
        return final_results


class VectorDB:
    """Vector database with multiple indexes and advanced features."""
    
    def __init__(self):
        self.indexes: Dict[str, VectorIndex] = {}
        self.default_index: Optional[str] = None
    
    def create_index(self, name: str, dim: int, metric: str = 'cosine', 
                     index_type: str = 'hnsw', **kwargs) -> VectorIndex:
        """Create a new vector index."""
        if index_type == 'hnsw':
            index = HNSWIndex(dim, metric, **kwargs)
        elif index_type == 'linear':
            index = LinearVectorIndex(dim, metric)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        self.indexes[name] = index
        
        # Set as default if this is the first index
        if self.default_index is None:
            self.default_index = name
        
        return index
    
    def get_index(self, name: str = None) -> VectorIndex:
        """Get an index by name or return default."""
        if name is None:
            name = self.default_index
        
        if name not in self.indexes:
            raise ValueError(f"Index {name} does not exist")
        
        return self.indexes[name]
    
    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any] = None, 
            index_name: str = None):
        """Add a vector to the specified index."""
        index = self.get_index(index_name)
        index.add(id, vector, metadata)
    
    def search(self, query: np.ndarray, k: int = 10, index_name: str = None) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for nearest neighbors in the specified index."""
        index = self.get_index(index_name)
        return index.search(query, k)
    
    def remove(self, id: str, index_name: str = None):
        """Remove a vector from the specified index."""
        index = self.get_index(index_name)
        index.remove(id)
    
    def save(self, path: str):
        """Save the entire database to disk."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load(cls, path: str):
        """Load the database from disk."""
        with open(path, 'rb') as f:
            return pickle.load(f)


# Additional utility functions for vector operations

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors."""
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-12)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-12)
    return np.dot(v1_norm, v2_norm)


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate Euclidean distance between two vectors."""
    return np.linalg.norm(v1 - v2)


def dot_product(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate dot product of two vectors."""
    return np.dot(v1, v2)


def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def batch_similarity(vectors1: np.ndarray, vectors2: np.ndarray, metric: str = 'cosine') -> np.ndarray:
    """
    Calculate similarity between batches of vectors.
    
    Args:
        vectors1: Array of shape (n, dim)
        vectors2: Array of shape (m, dim)
        metric: Similarity metric ('cosine', 'euclidean', 'dot')
        
    Returns:
        Similarity matrix of shape (n, m)
    """
    if metric == 'cosine':
        # Normalize vectors
        v1_norm = vectors1 / (np.linalg.norm(vectors1, axis=1, keepdims=True) + 1e-12)
        v2_norm = vectors2 / (np.linalg.norm(vectors2, axis=1, keepdims=True) + 1e-12)
        return np.dot(v1_norm, v2_norm.T)
    elif metric == 'dot':
        return np.dot(vectors1, vectors2.T)
    elif metric == 'euclidean':
        # Calculate pairwise distances
        diff = vectors1[:, np.newaxis, :] - vectors2[np.newaxis, :, :]
        return -np.linalg.norm(diff, axis=2)  # Negative for similarity
    else:
        raise ValueError(f"Unknown metric: {metric}")


# For timing operations
import time


def benchmark_index(index: VectorIndex, queries: List[np.ndarray], k: int = 10, num_iterations: int = 10) -> Dict[str, float]:
    """Benchmark index performance."""
    times = []
    
    for _ in range(num_iterations):
        start_time = time.time()
        for query in queries:
            index.search(query, k)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        "avg_time_per_query": np.mean(times) / len(queries),
        "std_time_per_query": np.std(times) / len(queries),
        "total_time": np.mean(times),
        "qps": len(queries) / np.mean(times)
    }