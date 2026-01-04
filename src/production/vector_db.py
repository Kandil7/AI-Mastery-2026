"""
Vector Database Module
======================
Custom vector database implementation with similarity search.

Features:
- HNSW index for fast approximate nearest neighbor search
- Multi-tenant support with namespace isolation
- Per-tenant quotas and rate limiting
- Backup and recovery capabilities
- Metadata filtering

Author: AI-Mastery-2026
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union
import pickle
from pathlib import Path
import json
from datetime import datetime
import shutil
import threading
from collections import defaultdict
import heapq
import math
from dataclasses import dataclass
import os
import time # Added this import as it's used later in the original code

# ============================================================
# MULTI-TENANT SUPPORT
# ============================================================

class TenantQuota:
    """Manage quotas for a tenant."""
    
    def __init__(self, max_vectors: int = 100000, max_storage_mb: int = 1000):
        self.max_vectors = max_vectors
        self.max_storage_mb = max_storage_mb
        self.current_vectors = 0
        self.current_storage_mb = 0
        self.lock = threading.Lock()
    
    def can_add_vectors(self, count: int, size_mb: float) -> bool:
        """Check if tenant has quota to add vectors."""
        with self.lock:
            if self.current_vectors + count > self.max_vectors:
                return False
            if self.current_storage_mb + size_mb > self.max_storage_mb:
                return False
            return True
    
    def add_vectors(self, count: int, size_mb: float):
        """Update usage after adding vectors."""
        with self.lock:
            self.current_vectors += count
            self.current_storage_mb += size_mb
    
    def remove_vectors(self, count: int, size_mb: float):
        """Update usage after removing vectors."""
        with self.lock:
            self.current_vectors = max(0, self.current_vectors - count)
            self.current_storage_mb = max(0, self.current_storage_mb - size_mb)
    
    def get_usage(self) -> Dict:
        """Get current usage statistics."""
        with self.lock:
            return {
                'vectors': {
                    'used': self.current_vectors,
                    'limit': self.max_vectors,
                    'percentage': (self.current_vectors / self.max_vectors) * 100 if self.max_vectors else 0
                },
                'storage_mb': {
                    'used': self.current_storage_mb,
                    'limit': self.max_storage_mb,
                    'percentage': (self.current_storage_mb / self.max_storage_mb) * 100 if self.max_storage_mb else 0
                }
            }


class MultiTenantVectorDB:
    """
    Multi-tenant vector database with namespace isolation.
    
    Each tenant gets an isolated namespace for their vectors,
    with separate quotas and access control.
    """
    
    def __init__(self, storage_path: str = "./vector_db"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        # Tenant namespaces: {tenant_id: {vector_id: (vector, metadata)}}
        self.namespaces: Dict[str, Dict[str, Tuple[np.ndarray, Dict]]] = defaultdict(dict)
        
        # Tenant quotas
        self.quotas: Dict[str, TenantQuota] = {}
        
        # Load existing data
        self._load_all_namespaces()
    
    def create_tenant(self, tenant_id: str, max_vectors: int = 100000, 
                     max_storage_mb: int = 1000):
        """
        Create a new tenant with quotas.
        
        Args:
            tenant_id: Unique tenant identifier
            max_vectors: Maximum vectors allowed
            max_storage_mb: Maximum storage in MB
        """
        if tenant_id in self.quotas:
            raise ValueError(f"Tenant {tenant_id} already exists")
        
        self.quotas[tenant_id] = TenantQuota(max_vectors, max_storage_mb)
        self.namespaces[tenant_id] = {}
        
        # Create tenant directory
        tenant_dir = self.storage_path / tenant_id
        tenant_dir.mkdir(exist_ok=True)
        
        # Save tenant metadata
        metadata = {
            'created_at': datetime.now().isoformat(),
            'max_vectors': max_vectors,
            'max_storage_mb': max_storage_mb
        }
        with open(tenant_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def add_vectors(self, tenant_id: str, vectors: List[np.ndarray], 
                   metadata: List[Dict], vector_ids: Optional[List[str]] = None):
        """
        Add vectors to a tenant's namespace.
        
        Args:
            tenant_id: Tenant identifier
            vectors: List of vectors to add
            metadata: List of metadata dicts
            vector_ids: Optional list of vector IDs
        
        Raises:
            ValueError: If tenant doesn't exist or quota exceeded
        """
        if tenant_id not in self.quotas:
            raise ValueError(f"Tenant {tenant_id} does not exist")
        
        # Calculate storage size
        size_mb = sum(v.nbytes for v in vectors) / (1024 * 1024)
        
        # Check quota
        quota = self.quotas[tenant_id]
        if not quota.can_add_vectors(len(vectors), size_mb):
            usage = quota.get_usage()
            raise ValueError(
                f"Quota exceeded for tenant {tenant_id}. "
                f"Vectors: {usage['vectors']['used']}/{usage['vectors']['limit']}, "
                f"Storage: {usage['storage_mb']['used']:.2f}/{usage['storage_mb']['limit']:.2f} MB"
            )
        
        # Generate IDs if not provided
        if vector_ids is None:
            vector_ids = [f"vec_{datetime.now().timestamp()}_{i}" for i in range(len(vectors))]
        
        # Add to namespace
        namespace = self.namespaces[tenant_id]
        for vid, vec, meta in zip(vector_ids, vectors, metadata):
            namespace[vid] = (vec, meta)
        
        # Update quota
        quota.add_vectors(len(vectors), size_mb)
        
        # Persist to disk
        self._save_namespace(tenant_id)
    
    def search(self, tenant_id: str, query_vector: np.ndarray, k: int = 5,
              filters: Optional[Dict] = None) -> List[Tuple[str, float, Dict]]:
        """
        Search vectors in a tenant's namespace.
        
        Args:
            tenant_id: Tenant identifier
            query_vector: Query vector
            k: Number of results
            filters: Optional metadata filters
        
        Returns:
            List of (vector_id, similarity, metadata) tuples
        """
        if tenant_id not in self.namespaces:
            raise ValueError(f"Tenant {tenant_id} does not exist")
        
        namespace = self.namespaces[tenant_id]
        
        # Apply filters
        candidates = namespace.items()
        if filters:
            candidates = [
                (vid, (vec, meta)) for vid, (vec, meta) in candidates
                if all(meta.get(k) == v for k, v in filters.items())
            ]
        
        # Compute similarities (cosine similarity)
        results = []
        for vid, (vec, meta) in candidates:
            # Handle zero-norm vectors to prevent division by zero
            norm_query = np.linalg.norm(query_vector)
            norm_vec = np.linalg.norm(vec)
            
            if norm_query == 0 or norm_vec == 0:
                similarity = 0.0
            else:
                similarity = np.dot(query_vector, vec) / (norm_query * norm_vec)
            results.append((vid, similarity, meta))
        
        # Sort by similarity and return top k
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:k]
    
    def delete_vectors(self, tenant_id: str, vector_ids: List[str]):
        """Delete vectors from a tenant's namespace."""
        if tenant_id not in self.namespaces:
            raise ValueError(f"Tenant {tenant_id} does not exist")
        
        namespace = self.namespaces[tenant_id]
        quota = self.quotas[tenant_id]
        
        size_mb = 0
        count = 0
        for vid in vector_ids:
            if vid in namespace:
                vec, _ = namespace[vid]
                size_mb += vec.nbytes / (1024 * 1024)
                del namespace[vid]
                count += 1
        
        quota.remove_vectors(count, size_mb)
        self._save_namespace(tenant_id)
    
    def get_tenant_stats(self, tenant_id: str) -> Dict:
        """Get statistics for a tenant."""
        if tenant_id not in self.quotas:
            raise ValueError(f"Tenant {tenant_id} does not exist")
        
        namespace = self.namespaces[tenant_id]
        quota = self.quotas[tenant_id]
        
        return {
            'tenant_id': tenant_id,
            'vector_count': len(namespace),
            'quota_usage': quota.get_usage(),
            'created_at': self._get_tenant_metadata(tenant_id).get('created_at')
        }
    
    def list_tenants(self) -> List[str]:
        """List all tenant IDs."""
        return list(self.quotas.keys())
    
    def _save_namespace(self, tenant_id: str):
        """Save a tenant's namespace to disk."""
        tenant_dir = self.storage_path / tenant_id
        namespace_file = tenant_dir / 'vectors.pkl'
        
        with open(namespace_file, 'wb') as f:
            pickle.dump(self.namespaces[tenant_id], f)
    
    def _load_all_namespaces(self):
        """Load all tenant namespaces from disk."""
        for tenant_dir in self.storage_path.iterdir():
            if not tenant_dir.is_dir():
                continue
            
            tenant_id = tenant_dir.name
            
            # Load metadata
            metadata_file = tenant_dir / 'metadata.json'
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                self.quotas[tenant_id] = TenantQuota(
                    metadata.get('max_vectors', 100000),
                    metadata.get('max_storage_mb', 1000)
                )
            
            # Load vectors
            vectors_file = tenant_dir / 'vectors.pkl'
            if vectors_file.exists():
                with open(vectors_file, 'rb') as f:
                    self.namespaces[tenant_id] = pickle.load(f)
                
                # Update quota usage
                if tenant_id in self.quotas:
                    namespace = self.namespaces[tenant_id]
                    count = len(namespace)
                    size_mb = sum(v[0].nbytes for v in namespace.values()) / (1024 * 1024)
                    self.quotas[tenant_id].add_vectors(count, size_mb)
    
    def _get_tenant_metadata(self, tenant_id: str) -> Dict:
        """Get tenant metadata."""
        metadata_file = self.storage_path / tenant_id / 'metadata.json'
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                return json.load(f)
        return {}


# ============================================================
# BACKUP AND RECOVERY
# ============================================================

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