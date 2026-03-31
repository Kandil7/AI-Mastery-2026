"""
HNSW Index Module
=================

Hierarchical Navigable Small World (HNSW) index implementation.
Provides efficient approximate nearest neighbor search.

Classes:
    HNSWNode: Node in HNSW graph
    HNSWIndex: HNSW implementation for fast ANN search

Author: AI-Mastery-2026
"""

import heapq
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import numpy as np

from .core import VectorIndex


class HNSWNode:
    """Node in HNSW graph."""

    def __init__(self, id: str, vector: np.ndarray, level: int):
        self.id = id
        self.vector = vector
        self.level = level
        self.connections: Dict[int, List[Tuple[str, float]]] = defaultdict(
            list
        )  # level -> [(neighbor_id, distance)]


class HNSWIndex(VectorIndex):
    """
    Hierarchical Navigable Small World (HNSW) index implementation.
    This provides efficient approximate nearest neighbor search.
    """

    def __init__(
        self,
        dim: int,
        metric: str = "cosine",
        max_level: int = 16,
        ef_construction: int = 200,
        M: int = 16,
    ):
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
        if self.metric == "cosine":
            # Cosine distance = 1 - cosine similarity
            return 1 - np.dot(v1, v2)
        elif self.metric == "euclidean":
            return np.linalg.norm(v1 - v2)
        elif self.metric == "dot":
            # Negative dot product as distance
            return -np.dot(v1, v2)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def add(self, id: str, vector: np.ndarray, metadata: Dict[str, Any] = None):
        """Add a vector to the HNSW index."""
        if metadata is None:
            metadata = {}

        if vector.shape[0] != self.dim:
            raise ValueError(
                f"Vector dimension {vector.shape[0]} does not match index dimension {self.dim}"
            )

        # Normalize vector for cosine similarity
        if self.metric == "cosine":
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
                neighbors = self._select_neighbors(
                    new_node.vector, self.nodes[ep_id].vector, level
                )

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

    def _search_layer(
        self, query: np.ndarray, ep_id: str, level: int, k: int
    ) -> List[Tuple[str, float]]:
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
                    neighbor_dist = self._distance(
                        query, self.nodes[neighbor_id].vector
                    )

                    if len(results) < k:
                        heapq.heappush(results, (-neighbor_dist, neighbor_id))
                    elif neighbor_dist < -results[0][0]:
                        heapq.heappreplace(results, (-neighbor_dist, neighbor_id))

                    heapq.heappush(candidate_set, (neighbor_dist, neighbor_id))

        # Convert to positive distances and return top k
        results = [(-dist, node_id) for dist, node_id in results]
        results.sort()  # Sort by distance (ascending)
        return results[:k]

    def _select_neighbors(
        self, query: np.ndarray, ep_vector: np.ndarray, level: int
    ) -> List[Tuple[str, float]]:
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
        return neighbors[: self.M]

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
        node.connections[level] = connections_with_dist[: self.M]

    def search(
        self, query: np.ndarray, k: int = 10
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for k nearest neighbors using HNSW."""
        if not self.nodes:
            return []

        if self.metric == "cosine":
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
            if self.metric == "cosine":
                similarity = 1 - dist  # Convert cosine distance back to similarity
            else:
                similarity = -dist  # Negative distance for similarity
            final_results.append((item.id, similarity, item.metadata))

        return final_results
