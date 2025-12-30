"""
Vector Database Module
======================
Approximate Nearest Neighbor (ANN) search implementations.

Includes:
- Brute force (exact) search
- HNSW (Hierarchical Navigable Small World) - from scratch
- LSH (Locality-Sensitive Hashing)
- Product Quantization basics

Used in:
- Semantic search
- RAG (Retrieval-Augmented Generation)
- Recommendation systems

Author: AI-Mastery-2026
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
from heapq import heappush, heappop, nlargest, nsmallest
import random
import logging

logger = logging.getLogger(__name__)


# ============================================================
# DISTANCE FUNCTIONS
# ============================================================

def cosine_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine distance = 1 - cosine_similarity."""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 1.0
    return 1.0 - (dot / norm)


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Euclidean (L2) distance."""
    return float(np.sqrt(np.sum((v1 - v2) ** 2)))


def inner_product_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    """Negative inner product (for similarity ranking)."""
    return -float(np.dot(v1, v2))


DISTANCE_FUNCTIONS = {
    'cosine': cosine_distance,
    'euclidean': euclidean_distance,
    'l2': euclidean_distance,
    'ip': inner_product_distance,
    'inner_product': inner_product_distance,
}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class SearchResult:
    """Result from a vector search."""
    id: Any
    score: float
    vector: Optional[np.ndarray] = None
    metadata: Optional[Dict[str, Any]] = None


# ============================================================
# BRUTE FORCE (EXACT) SEARCH
# ============================================================

class BruteForceIndex:
    """
    Exact nearest neighbor search using brute force.
    
    Complexity: O(n × d) per query
    Space: O(n × d)
    
    Use for:
    - Small datasets (< 100K vectors)
    - Baseline comparison
    - Exact results required
    
    Example:
        >>> index = BruteForceIndex(metric='cosine')
        >>> index.add(vectors, ids=ids)
        >>> results = index.search(query_vector, k=10)
    """
    
    def __init__(self, metric: str = 'cosine'):
        """
        Args:
            metric: Distance metric ('cosine', 'euclidean', 'ip')
        """
        self.metric = metric
        self.distance_fn = DISTANCE_FUNCTIONS[metric]
        self.vectors: Optional[np.ndarray] = None
        self.ids: Optional[List[Any]] = None
        self.metadata: Optional[List[Dict]] = None
    
    def add(self, vectors: np.ndarray, 
            ids: Optional[List[Any]] = None,
            metadata: Optional[List[Dict]] = None):
        """
        Add vectors to the index.
        
        Args:
            vectors: Array of shape (n, d)
            ids: Optional list of IDs
            metadata: Optional list of metadata dicts
        """
        vectors = np.asarray(vectors)
        
        if self.vectors is None:
            self.vectors = vectors
            self.ids = ids if ids else list(range(len(vectors)))
            self.metadata = metadata if metadata else [{}] * len(vectors)
        else:
            self.vectors = np.vstack([self.vectors, vectors])
            new_ids = ids if ids else list(range(len(self.ids), len(self.ids) + len(vectors)))
            self.ids.extend(new_ids)
            new_meta = metadata if metadata else [{}] * len(vectors)
            self.metadata.extend(new_meta)
    
    def search(self, query: np.ndarray, k: int = 10,
               filter_fn: Optional[Callable] = None) -> List[SearchResult]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector
            k: Number of results
            filter_fn: Optional function to filter results
        
        Returns:
            List of SearchResult sorted by distance
        """
        if self.vectors is None:
            return []
        
        query = np.asarray(query)
        
        # Compute all distances
        distances = [self.distance_fn(query, v) for v in self.vectors]
        
        # Get top k
        indices = np.argsort(distances)[:k * 2]  # Get more to allow for filtering
        
        results = []
        for idx in indices:
            if filter_fn and not filter_fn(self.metadata[idx]):
                continue
            
            results.append(SearchResult(
                id=self.ids[idx],
                score=distances[idx],
                vector=self.vectors[idx],
                metadata=self.metadata[idx]
            ))
            
            if len(results) >= k:
                break
        
        return results
    
    def __len__(self) -> int:
        return len(self.vectors) if self.vectors is not None else 0


# ============================================================
# HNSW (Hierarchical Navigable Small World)
# ============================================================

class HNSWNode:
    """Node in HNSW graph."""
    
    def __init__(self, id: Any, vector: np.ndarray, level: int):
        self.id = id
        self.vector = vector
        self.level = level
        self.neighbors: Dict[int, List['HNSWNode']] = {l: [] for l in range(level + 1)}


class HNSW:
    """
    Hierarchical Navigable Small World Graph.
    
    State-of-the-art ANN algorithm used by:
    - Pinecone, Qdrant, Milvus, Weaviate
    - Facebook FAISS
    
    Complexity:
    - Build: O(n log n)
    - Search: O(log n)
    
    Key parameters:
    - M: Max connections per node
    - ef_construction: Beam width during build
    - ef_search: Beam width during search
    
    Algorithm:
    1. Multi-layer graph with exponentially decreasing density
    2. Start search from top layer, greedily navigate down
    3. At bottom layer, do beam search for precision
    
    Example:
        >>> hnsw = HNSW(dim=384, M=16, ef_construction=200)
        >>> hnsw.add_items(vectors, ids)
        >>> results = hnsw.search(query, k=10, ef=100)
    """
    
    def __init__(self, dim: int, 
                 M: int = 16,
                 ef_construction: int = 200,
                 metric: str = 'cosine',
                 ml: float = 1.0 / np.log(16)):
        """
        Args:
            dim: Vector dimensionality
            M: Max edges per node (default 16)
            ef_construction: Beam width for construction
            metric: Distance metric
            ml: Level multiplier (controls layer distribution)
        """
        self.dim = dim
        self.M = M
        self.M0 = 2 * M  # Max edges at level 0
        self.ef_construction = ef_construction
        self.ml = ml
        self.metric = metric
        self.distance_fn = DISTANCE_FUNCTIONS[metric]
        
        self.nodes: Dict[Any, HNSWNode] = {}
        self.entry_point: Optional[HNSWNode] = None
        self.max_level = 0
    
    def _random_level(self) -> int:
        """Generate random level for new node."""
        level = 0
        while random.random() < self.ml and level < 32:
            level += 1
        return level
    
    def _distance(self, node1: HNSWNode, node2: HNSWNode) -> float:
        """Compute distance between two nodes."""
        return self.distance_fn(node1.vector, node2.vector)
    
    def _search_layer(self, query: np.ndarray, 
                      entry: HNSWNode,
                      ef: int,
                      level: int) -> List[Tuple[float, HNSWNode]]:
        """
        Search single layer using greedy beam search.
        
        Args:
            query: Query vector
            entry: Entry point node
            ef: Beam width
            level: Layer level
        
        Returns:
            List of (distance, node) tuples
        """
        visited = {entry.id}
        candidates = [(self.distance_fn(query, entry.vector), entry)]
        results = [(candidates[0][0], entry)]
        
        while candidates:
            # Get closest candidate
            dist, current = heappop(candidates)
            
            # Get furthest result
            furthest_dist = max(r[0] for r in results) if results else float('inf')
            
            if dist > furthest_dist:
                break
            
            # Explore neighbors
            for neighbor in current.neighbors.get(level, []):
                if neighbor.id in visited:
                    continue
                
                visited.add(neighbor.id)
                neighbor_dist = self.distance_fn(query, neighbor.vector)
                
                if neighbor_dist < furthest_dist or len(results) < ef:
                    heappush(candidates, (neighbor_dist, neighbor))
                    results.append((neighbor_dist, neighbor))
                    
                    # Keep only ef best
                    if len(results) > ef:
                        results = nsmallest(ef, results, key=lambda x: x[0])
        
        return results
    
    def _select_neighbors(self, candidates: List[Tuple[float, HNSWNode]],
                          M: int) -> List[HNSWNode]:
        """Select M best neighbors from candidates."""
        sorted_candidates = sorted(candidates, key=lambda x: x[0])
        return [node for _, node in sorted_candidates[:M]]
    
    def add(self, vector: np.ndarray, id: Any):
        """
        Add a single vector to the index.
        
        Args:
            vector: Vector to add
            id: ID for the vector
        """
        vector = np.asarray(vector)
        level = self._random_level()
        node = HNSWNode(id, vector, level)
        
        if self.entry_point is None:
            # First node
            self.entry_point = node
            self.max_level = level
            self.nodes[id] = node
            return
        
        entry = self.entry_point
        
        # Descend from top to level+1, finding closest entry
        for l in range(self.max_level, level, -1):
            results = self._search_layer(vector, entry, ef=1, level=l)
            if results:
                entry = min(results, key=lambda x: x[0])[1]
        
        # Insert at each level from level down to 0
        for l in range(min(level, self.max_level), -1, -1):
            # Find ef_construction nearest neighbors
            results = self._search_layer(vector, entry, 
                                         ef=self.ef_construction, level=l)
            
            # Select M best as neighbors
            M = self.M if l > 0 else self.M0
            neighbors = self._select_neighbors(results, M)
            
            # Add bidirectional edges
            node.neighbors[l] = neighbors
            for neighbor in neighbors:
                neighbor.neighbors[l].append(node)
                
                # Prune if too many edges
                if len(neighbor.neighbors[l]) > M:
                    # Keep M closest
                    neighbor.neighbors[l] = self._select_neighbors(
                        [(self._distance(node, n), n) for n in neighbor.neighbors[l]], M
                    )
            
            if results:
                entry = min(results, key=lambda x: x[0])[1]
        
        # Update entry point if new node has higher level
        if level > self.max_level:
            self.entry_point = node
            self.max_level = level
        
        self.nodes[id] = node
    
    def add_items(self, vectors: np.ndarray, ids: Optional[List[Any]] = None):
        """
        Add multiple vectors to the index.
        
        Args:
            vectors: Array of vectors (n, d)
            ids: Optional list of IDs
        """
        vectors = np.asarray(vectors)
        if ids is None:
            ids = list(range(len(vectors)))
        
        for i, (vector, id_) in enumerate(zip(vectors, ids)):
            self.add(vector, id_)
            
            if (i + 1) % 1000 == 0:
                logger.info(f"Indexed {i + 1}/{len(vectors)} vectors")
    
    def search(self, query: np.ndarray, k: int = 10,
               ef: Optional[int] = None) -> List[SearchResult]:
        """
        Search for k nearest neighbors.
        
        Args:
            query: Query vector
            k: Number of results
            ef: Search beam width (default: k)
        
        Returns:
            List of SearchResult sorted by distance
        """
        if self.entry_point is None:
            return []
        
        query = np.asarray(query)
        ef = ef or max(k, 10)
        
        entry = self.entry_point
        
        # Descend from top to level 1
        for l in range(self.max_level, 0, -1):
            results = self._search_layer(query, entry, ef=1, level=l)
            if results:
                entry = min(results, key=lambda x: x[0])[1]
        
        # Search at level 0 with full ef
        results = self._search_layer(query, entry, ef=ef, level=0)
        
        # Return top k
        results = sorted(results, key=lambda x: x[0])[:k]
        
        return [
            SearchResult(id=node.id, score=dist, vector=node.vector)
            for dist, node in results
        ]
    
    def __len__(self) -> int:
        return len(self.nodes)


# ============================================================
# LSH (Locality-Sensitive Hashing)
# ============================================================

class LSH:
    """
    Locality-Sensitive Hashing for approximate nearest neighbors.
    
    Uses random hyperplane projections to hash similar vectors
    to the same buckets with high probability.
    
    Complexity:
    - Build: O(n × d × num_tables)
    - Search: O(num_tables × bucket_size)
    
    Trade-offs:
    - More tables = higher recall, more memory
    - More bits = higher precision, sparser buckets
    
    Example:
        >>> lsh = LSH(dim=384, num_tables=10, num_bits=8)
        >>> lsh.add_items(vectors, ids)
        >>> results = lsh.search(query, k=10)
    """
    
    def __init__(self, dim: int, num_tables: int = 10, num_bits: int = 8):
        """
        Args:
            dim: Vector dimensionality
            num_tables: Number of hash tables
            num_bits: Bits per hash (determines bucket granularity)
        """
        self.dim = dim
        self.num_tables = num_tables
        self.num_bits = num_bits
        
        # Random projection matrices for each table
        self.projections = [
            np.random.randn(num_bits, dim) 
            for _ in range(num_tables)
        ]
        
        # Hash tables: table_idx -> bucket_hash -> list of (id, vector)
        self.tables: List[Dict[str, List[Tuple[Any, np.ndarray]]]] = [
            {} for _ in range(num_tables)
        ]
    
    def _hash(self, vector: np.ndarray, table_idx: int) -> str:
        """Compute LSH hash for a vector."""
        projection = self.projections[table_idx]
        bits = (projection @ vector > 0).astype(int)
        return ''.join(map(str, bits))
    
    def add(self, vector: np.ndarray, id: Any):
        """Add a single vector."""
        vector = np.asarray(vector)
        
        for i, table in enumerate(self.tables):
            hash_key = self._hash(vector, i)
            if hash_key not in table:
                table[hash_key] = []
            table[hash_key].append((id, vector))
    
    def add_items(self, vectors: np.ndarray, ids: Optional[List[Any]] = None):
        """Add multiple vectors."""
        vectors = np.asarray(vectors)
        if ids is None:
            ids = list(range(len(vectors)))
        
        for vector, id_ in zip(vectors, ids):
            self.add(vector, id_)
    
    def search(self, query: np.ndarray, k: int = 10) -> List[SearchResult]:
        """Search for k nearest neighbors."""
        query = np.asarray(query)
        
        # Collect candidates from all tables
        candidates = {}  # id -> vector
        
        for i, table in enumerate(self.tables):
            hash_key = self._hash(query, i)
            
            if hash_key in table:
                for id_, vector in table[hash_key]:
                    if id_ not in candidates:
                        candidates[id_] = vector
        
        if not candidates:
            return []
        
        # Compute exact distances for candidates
        results = []
        for id_, vector in candidates.items():
            dist = cosine_distance(query, vector)
            results.append(SearchResult(id=id_, score=dist, vector=vector))
        
        # Return top k
        results.sort(key=lambda x: x.score)
        return results[:k]
    
    def __len__(self) -> int:
        # Count unique vectors across all tables (approximation)
        all_ids = set()
        for table in self.tables:
            for bucket in table.values():
                for id_, _ in bucket:
                    all_ids.add(id_)
        return len(all_ids)


# ============================================================
# VECTOR INDEX FACTORY
# ============================================================

class VectorIndex:
    """
    Factory for creating vector indexes.
    
    Unified interface for different index types.
    
    Example:
        >>> index = VectorIndex.create('hnsw', dim=384)
        >>> index.add_items(vectors, ids)
        >>> results = index.search(query, k=10)
    """
    
    @staticmethod
    def create(index_type: str, dim: int, **kwargs) -> Any:
        """
        Create a vector index.
        
        Args:
            index_type: 'brute_force', 'hnsw', 'lsh'
            dim: Vector dimensionality
            **kwargs: Index-specific parameters
        
        Returns:
            Vector index instance
        """
        if index_type == 'brute_force':
            return BruteForceIndex(metric=kwargs.get('metric', 'cosine'))
        elif index_type == 'hnsw':
            return HNSW(
                dim=dim,
                M=kwargs.get('M', 16),
                ef_construction=kwargs.get('ef_construction', 200),
                metric=kwargs.get('metric', 'cosine')
            )
        elif index_type == 'lsh':
            return LSH(
                dim=dim,
                num_tables=kwargs.get('num_tables', 10),
                num_bits=kwargs.get('num_bits', 8)
            )
        else:
            raise ValueError(f"Unknown index type: {index_type}")


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Distance functions
    'cosine_distance', 'euclidean_distance', 'inner_product_distance',
    # Data structures
    'SearchResult',
    # Index implementations
    'BruteForceIndex', 'HNSW', 'LSH',
    # Factory
    'VectorIndex',
]
