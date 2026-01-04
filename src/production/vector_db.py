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


# ============================================================
# ACL-BASED FILTERING (WEAVIATE/PINECONE PATTERN)
# ============================================================

class ACLPermission:
    """Access control permission levels."""
    READ = "read"
    WRITE = "write"
    ADMIN = "admin"
    NONE = "none"


@dataclass
class ACLEntry:
    """Access control list entry."""
    user_id: str
    permission: str
    granted_at: float = None
    expires_at: Optional[float] = None
    
    def __post_init__(self):
        if self.granted_at is None:
            self.granted_at = time.time()
    
    def is_valid(self) -> bool:
        """Check if ACL entry is still valid."""
        if self.expires_at is None:
            return True
        return time.time() < self.expires_at


class ACLFilter:
    """
    Object-level access control for vector search (Weaviate pattern).
    
    Prevents semantic data leakage by attaching ACLs as metadata
    and filtering search results by user permissions.
    
    Key Features:
    - Attach ACLs during indexing
    - Filter results before returning to user
    - Support for read/write/admin permissions
    - Time-based permission expiry
    
    Reference: Weaviate RBAC, Pinecone Enterprise
    """
    
    ACL_METADATA_KEY = "_acl"
    
    def __init__(self, default_permission: str = ACLPermission.NONE):
        """
        Initialize ACL filter.
        
        Args:
            default_permission: Permission for users not in ACL
        """
        self.default_permission = default_permission
        self._access_log: List[Dict[str, Any]] = []
    
    @staticmethod
    def create_acl_metadata(
        owner_id: str,
        read_users: Optional[List[str]] = None,
        write_users: Optional[List[str]] = None,
        public: bool = False
    ) -> Dict[str, Any]:
        """
        Create ACL metadata to attach to a vector.
        
        Args:
            owner_id: ID of the document owner
            read_users: Users with read access
            write_users: Users with write access
            public: If True, everyone can read
            
        Returns:
            ACL metadata dict
        """
        acl = {
            "owner": owner_id,
            "public": public,
            "read": read_users or [],
            "write": write_users or [],
            "created_at": time.time()
        }
        return {ACLFilter.ACL_METADATA_KEY: acl}
    
    def check_permission(
        self,
        user_id: str,
        metadata: Dict[str, Any],
        required_permission: str = ACLPermission.READ
    ) -> bool:
        """
        Check if user has required permission.
        
        Args:
            user_id: User requesting access
            metadata: Vector metadata containing ACL
            required_permission: Permission level needed
            
        Returns:
            True if user has permission
        """
        acl = metadata.get(self.ACL_METADATA_KEY)
        
        if not acl:
            # No ACL defined - use default
            return self.default_permission != ACLPermission.NONE
        
        # Owner has all permissions
        if acl.get("owner") == user_id:
            return True
        
        # Public documents allow read
        if acl.get("public") and required_permission == ACLPermission.READ:
            return True
        
        # Check explicit permissions
        if required_permission == ACLPermission.READ:
            return user_id in acl.get("read", []) or user_id in acl.get("write", [])
        elif required_permission == ACLPermission.WRITE:
            return user_id in acl.get("write", [])
        elif required_permission == ACLPermission.ADMIN:
            return acl.get("owner") == user_id
        
        return False
    
    def filter_search_results(
        self,
        user_id: str,
        results: List[Tuple[str, float, Dict[str, Any]]],
        permission: str = ACLPermission.READ
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Filter search results by user permissions.
        
        This is the key security function that prevents users
        from seeing vectors they don't have access to.
        
        Args:
            user_id: User requesting results
            results: Raw search results
            permission: Required permission level
            
        Returns:
            Filtered results the user can access
        """
        filtered = []
        
        for vector_id, score, metadata in results:
            if self.check_permission(user_id, metadata, permission):
                filtered.append((vector_id, score, metadata))
                self._log_access(user_id, vector_id, "granted", permission)
            else:
                self._log_access(user_id, vector_id, "denied", permission)
        
        return filtered
    
    def _log_access(
        self,
        user_id: str,
        vector_id: str,
        decision: str,
        permission: str
    ) -> None:
        """Log access attempt for audit."""
        self._access_log.append({
            "timestamp": time.time(),
            "user_id": user_id,
            "vector_id": vector_id,
            "decision": decision,
            "permission_checked": permission
        })
        
        # Keep only last 10000 entries
        if len(self._access_log) > 10000:
            self._access_log = self._access_log[-10000:]
    
    def get_access_log(
        self,
        user_id: Optional[str] = None,
        decision: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Get access log with optional filters.
        
        Args:
            user_id: Filter by user
            decision: Filter by decision (granted/denied)
            limit: Max entries to return
            
        Returns:
            Matching log entries
        """
        entries = self._access_log
        
        if user_id:
            entries = [e for e in entries if e["user_id"] == user_id]
        if decision:
            entries = [e for e in entries if e["decision"] == decision]
        
        return entries[-limit:]
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get security statistics."""
        if not self._access_log:
            return {"total_checks": 0}
        
        total = len(self._access_log)
        granted = sum(1 for e in self._access_log if e["decision"] == "granted")
        denied = total - granted
        
        return {
            "total_checks": total,
            "granted": granted,
            "denied": denied,
            "denial_rate": denied / total if total > 0 else 0,
            "unique_users": len(set(e["user_id"] for e in self._access_log)),
            "unique_vectors": len(set(e["vector_id"] for e in self._access_log))
        }


# ============================================================
# EMBEDDING DRIFT DETECTION (ARIZE/FIDDLER PATTERN)
# ============================================================

class EmbeddingDriftDetector:
    """
    Detect drift in embedding space (Arize/Fiddler pattern).
    
    Monitors changes in embedding distributions over time to detect:
    - Concept drift (semantics of queries changing)
    - Data drift (input distribution changing)
    - Model staleness (embeddings becoming less relevant)
    
    Key Features:
    - Track embedding centroid over time
    - Compute distribution distances (cosine, euclidean)
    - Statistical tests for drift significance
    - UMAP visualization support
    
    Reference: Arize AI, Fiddler ML Observability
    """
    
    def __init__(
        self,
        embedding_dim: int,
        window_size: int = 1000,
        drift_threshold: float = 0.1
    ):
        """
        Initialize drift detector.
        
        Args:
            embedding_dim: Dimension of embeddings
            window_size: Number of embeddings to track per window
            drift_threshold: Threshold for drift alerts
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.drift_threshold = drift_threshold
        
        # Reference distribution (baseline)
        self._reference_embeddings: List[np.ndarray] = []
        self._reference_centroid: Optional[np.ndarray] = None
        self._reference_std: Optional[np.ndarray] = None
        
        # Current distribution
        self._current_embeddings: List[np.ndarray] = []
        self._current_centroid: Optional[np.ndarray] = None
        
        # Drift history
        self._drift_history: List[Dict[str, Any]] = []
        self._alerts: List[Dict[str, Any]] = []
        
        self._lock = threading.Lock()
    
    def set_reference(self, embeddings: List[np.ndarray]) -> Dict[str, Any]:
        """
        Set the reference/baseline distribution.
        
        Args:
            embeddings: List of reference embeddings
            
        Returns:
            Reference statistics
        """
        with self._lock:
            self._reference_embeddings = embeddings[-self.window_size:]
            arr = np.array(self._reference_embeddings)
            self._reference_centroid = np.mean(arr, axis=0)
            self._reference_std = np.std(arr, axis=0)
        
        return {
            "reference_size": len(self._reference_embeddings),
            "centroid_norm": float(np.linalg.norm(self._reference_centroid)),
            "avg_std": float(np.mean(self._reference_std))
        }
    
    def add_embedding(self, embedding: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Add a new embedding and check for drift.
        
        Args:
            embedding: New embedding vector
            
        Returns:
            Drift alert if threshold exceeded, else None
        """
        if embedding.shape[0] != self.embedding_dim:
            raise ValueError(f"Expected dim {self.embedding_dim}, got {embedding.shape[0]}")
        
        with self._lock:
            self._current_embeddings.append(embedding)
            
            # Keep only window_size embeddings
            if len(self._current_embeddings) > self.window_size:
                self._current_embeddings = self._current_embeddings[-self.window_size:]
            
            # Update current centroid
            arr = np.array(self._current_embeddings)
            self._current_centroid = np.mean(arr, axis=0)
        
        # Check for drift if we have enough samples
        if len(self._current_embeddings) >= self.window_size // 2:
            return self._check_drift()
        
        return None
    
    def _check_drift(self) -> Optional[Dict[str, Any]]:
        """Check for drift between reference and current distributions."""
        if self._reference_centroid is None or self._current_centroid is None:
            return None
        
        # Compute drift metrics
        drift_metrics = self.compute_drift_metrics()
        
        # Record in history
        self._drift_history.append({
            "timestamp": time.time(),
            **drift_metrics
        })
        
        # Keep only last 1000 readings
        if len(self._drift_history) > 1000:
            self._drift_history = self._drift_history[-1000:]
        
        # Check if threshold exceeded
        if drift_metrics["cosine_distance"] > self.drift_threshold:
            alert = {
                "type": "embedding_drift",
                "timestamp": time.time(),
                "severity": "high" if drift_metrics["cosine_distance"] > self.drift_threshold * 2 else "medium",
                "metrics": drift_metrics,
                "threshold": self.drift_threshold
            }
            self._alerts.append(alert)
            return alert
        
        return None
    
    def compute_drift_metrics(self) -> Dict[str, float]:
        """
        Compute drift metrics between reference and current.
        
        Returns:
            Dictionary of drift metrics
        """
        if self._reference_centroid is None or self._current_centroid is None:
            return {"error": "Distributions not set"}
        
        # Cosine distance between centroids
        cosine_sim = np.dot(self._reference_centroid, self._current_centroid) / (
            np.linalg.norm(self._reference_centroid) * 
            np.linalg.norm(self._current_centroid) + 1e-12
        )
        cosine_distance = 1 - cosine_sim
        
        # Euclidean distance between centroids
        euclidean_distance = float(np.linalg.norm(
            self._reference_centroid - self._current_centroid
        ))
        
        # Compute variance change
        current_std = np.std(np.array(self._current_embeddings), axis=0)
        std_change = float(np.mean(np.abs(current_std - self._reference_std))) \
            if self._reference_std is not None else 0.0
        
        return {
            "cosine_distance": float(cosine_distance),
            "cosine_similarity": float(cosine_sim),
            "euclidean_distance": euclidean_distance,
            "std_change": std_change,
            "reference_size": len(self._reference_embeddings),
            "current_size": len(self._current_embeddings)
        }
    
    def get_drift_trend(self, window: int = 100) -> Dict[str, Any]:
        """
        Get drift trend over recent history.
        
        Args:
            window: Number of recent readings to analyze
            
        Returns:
            Trend analysis
        """
        if len(self._drift_history) < 2:
            return {"trend": "insufficient_data"}
        
        recent = self._drift_history[-window:]
        distances = [r["cosine_distance"] for r in recent]
        
        # Simple linear regression for trend
        x = np.arange(len(distances))
        slope = np.polyfit(x, distances, 1)[0] if len(distances) > 1 else 0
        
        return {
            "window_size": len(recent),
            "avg_drift": float(np.mean(distances)),
            "max_drift": float(np.max(distances)),
            "min_drift": float(np.min(distances)),
            "trend_slope": float(slope),
            "trend_direction": "increasing" if slope > 0.001 else (
                "decreasing" if slope < -0.001 else "stable"
            ),
            "above_threshold_count": sum(1 for d in distances if d > self.drift_threshold)
        }
    
    def get_embeddings_for_umap(
        self,
        include_reference: bool = True,
        include_current: bool = True,
        sample_size: Optional[int] = 500
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Get embeddings formatted for UMAP visualization.
        
        Args:
            include_reference: Include reference embeddings
            include_current: Include current embeddings
            sample_size: Max embeddings per distribution
            
        Returns:
            (embeddings_array, labels_list) for UMAP
        """
        embeddings = []
        labels = []
        
        if include_reference and self._reference_embeddings:
            ref = self._reference_embeddings
            if sample_size and len(ref) > sample_size:
                indices = np.random.choice(len(ref), sample_size, replace=False)
                ref = [ref[i] for i in indices]
            embeddings.extend(ref)
            labels.extend(["reference"] * len(ref))
        
        if include_current and self._current_embeddings:
            curr = self._current_embeddings
            if sample_size and len(curr) > sample_size:
                indices = np.random.choice(len(curr), sample_size, replace=False)
                curr = [curr[i] for i in indices]
            embeddings.extend(curr)
            labels.extend(["current"] * len(curr))
        
        if not embeddings:
            return np.array([]), []
        
        return np.array(embeddings), labels
    
    def get_alerts(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent drift alerts."""
        return self._alerts[-limit:]
    
    def clear_alerts(self) -> int:
        """Clear alerts and return count."""
        count = len(self._alerts)
        self._alerts = []
        return count
    
    def reset_current(self) -> None:
        """Reset current distribution (e.g., after retraining)."""
        with self._lock:
            self._current_embeddings = []
            self._current_centroid = None


class SecureVectorDB(VectorDB):
    """
    Vector database with ACL filtering and drift detection.
    
    Extends VectorDB with:
    - Object-level access control
    - Embedding drift monitoring
    - Security audit logging
    """
    
    def __init__(self, embedding_dim: int = 384):
        super().__init__()
        self.acl_filter = ACLFilter()
        self.drift_detector = EmbeddingDriftDetector(embedding_dim)
    
    def add_with_acl(
        self,
        id: str,
        vector: np.ndarray,
        metadata: Dict[str, Any] = None,
        owner_id: str = None,
        read_users: List[str] = None,
        write_users: List[str] = None,
        public: bool = False,
        index_name: str = None
    ) -> None:
        """Add vector with ACL metadata."""
        if metadata is None:
            metadata = {}
        
        # Add ACL to metadata
        if owner_id:
            acl_meta = ACLFilter.create_acl_metadata(
                owner_id, read_users, write_users, public
            )
            metadata.update(acl_meta)
        
        self.add(id, vector, metadata, index_name)
        
        # Track for drift detection
        self.drift_detector.add_embedding(vector)
    
    def search_with_acl(
        self,
        user_id: str,
        query: np.ndarray,
        k: int = 10,
        index_name: str = None
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search with ACL filtering."""
        # Get raw results (may contain vectors user can't access)
        raw_results = self.search(query, k * 2, index_name)  # Get extra for filtering
        
        # Filter by user permissions
        filtered = self.acl_filter.filter_search_results(user_id, raw_results)
        
        # Track query embedding for drift
        self.drift_detector.add_embedding(query)
        
        return filtered[:k]
    
    def get_security_stats(self) -> Dict[str, Any]:
        """Get combined security and drift statistics."""
        return {
            "acl": self.acl_filter.get_security_stats(),
            "drift": self.drift_detector.compute_drift_metrics(),
            "drift_trend": self.drift_detector.get_drift_trend(),
            "drift_alerts": len(self.drift_detector.get_alerts())
        }