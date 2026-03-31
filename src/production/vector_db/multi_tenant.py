"""
Multi-Tenant Support Module
===========================

Multi-tenant vector database with namespace isolation.
Each tenant gets an isolated namespace for their vectors,
with separate quotas and access control.

Classes:
    TenantQuota: Manage quotas for a tenant
    MultiTenantVectorDB: Multi-tenant vector database

Author: AI-Mastery-2026
"""

import json
import pickle
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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
                "vectors": {
                    "used": self.current_vectors,
                    "limit": self.max_vectors,
                    "percentage": (
                        (self.current_vectors / self.max_vectors) * 100
                        if self.max_vectors
                        else 0
                    ),
                },
                "storage_mb": {
                    "used": self.current_storage_mb,
                    "limit": self.max_storage_mb,
                    "percentage": (
                        (self.current_storage_mb / self.max_storage_mb) * 100
                        if self.max_storage_mb
                        else 0
                    ),
                },
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
        self.namespaces: Dict[str, Dict[str, Tuple[np.ndarray, Dict]]] = defaultdict(
            dict
        )

        # Tenant quotas
        self.quotas: Dict[str, TenantQuota] = {}

        # Load existing data
        self._load_all_namespaces()

    def create_tenant(
        self, tenant_id: str, max_vectors: int = 100000, max_storage_mb: int = 1000
    ):
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
            "created_at": self._get_current_timestamp(),
            "max_vectors": max_vectors,
            "max_storage_mb": max_storage_mb,
        }
        with open(tenant_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

    def add_vectors(
        self,
        tenant_id: str,
        vectors: List[np.ndarray],
        metadata: List[Dict],
        vector_ids: Optional[List[str]] = None,
    ):
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
            vector_ids = [
                f"vec_{self._get_current_timestamp()}_{i}" for i in range(len(vectors))
            ]

        # Add to namespace
        namespace = self.namespaces[tenant_id]
        for vid, vec, meta in zip(vector_ids, vectors, metadata):
            namespace[vid] = (vec, meta)

        # Update quota
        quota.add_vectors(len(vectors), size_mb)

        # Persist to disk
        self._save_namespace(tenant_id)

    def search(
        self,
        tenant_id: str,
        query_vector: np.ndarray,
        k: int = 5,
        filters: Optional[Dict] = None,
    ) -> List[Tuple[str, float, Dict]]:
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
                (vid, (vec, meta))
                for vid, (vec, meta) in candidates
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
            "tenant_id": tenant_id,
            "vector_count": len(namespace),
            "quota_usage": quota.get_usage(),
            "created_at": self._get_tenant_metadata(tenant_id).get("created_at"),
        }

    def list_tenants(self) -> List[str]:
        """List all tenant IDs."""
        return list(self.quotas.keys())

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime

        return datetime.now().isoformat()

    def _save_namespace(self, tenant_id: str):
        """Save a tenant's namespace to disk."""
        tenant_dir = self.storage_path / tenant_id
        namespace_file = tenant_dir / "vectors.pkl"

        with open(namespace_file, "wb") as f:
            pickle.dump(self.namespaces[tenant_id], f)

    def _load_all_namespaces(self):
        """Load all tenant namespaces from disk."""
        for tenant_dir in self.storage_path.iterdir():
            if not tenant_dir.is_dir():
                continue

            tenant_id = tenant_dir.name

            # Load metadata
            metadata_file = tenant_dir / "metadata.json"
            if metadata_file.exists():
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)

                self.quotas[tenant_id] = TenantQuota(
                    metadata.get("max_vectors", 100000),
                    metadata.get("max_storage_mb", 1000),
                )

            # Load vectors
            vectors_file = tenant_dir / "vectors.pkl"
            if vectors_file.exists():
                with open(vectors_file, "rb") as f:
                    self.namespaces[tenant_id] = pickle.load(f)

                # Update quota usage
                if tenant_id in self.quotas:
                    namespace = self.namespaces[tenant_id]
                    count = len(namespace)
                    size_mb = sum(v[0].nbytes for v in namespace.values()) / (
                        1024 * 1024
                    )
                    self.quotas[tenant_id].add_vectors(count, size_mb)

    def _get_tenant_metadata(self, tenant_id: str) -> Dict:
        """Get tenant metadata."""
        metadata_file = self.storage_path / tenant_id / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                return json.load(f)
        return {}
