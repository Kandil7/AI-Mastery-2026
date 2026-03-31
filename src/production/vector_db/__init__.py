"""
Vector Database Submodule
==========================

Modular vector database implementation with:
- Core indexing (LinearVectorIndex, HNSWIndex)
- Multi-tenant support
- ACL-based access control
- Embedding drift detection
- Security features

Usage:
    from src.production.vector_db import (
        VectorDB,
        SecureVectorDB,
        HNSWIndex,
        LinearVectorIndex,
        MultiTenantVectorDB,
        ACLFilter,
        EmbeddingDriftDetector,
    )

    # Basic usage
    db = VectorDB()
    db.create_index("main", dim=384, index_type="hnsw")
    db.add("vec1", embedding, {"source": "doc1"})
    results = db.search(query_embedding, k=5)

    # Secure usage with ACL
    secure_db = SecureVectorDB(embedding_dim=384)
    secure_db.add_with_acl(
        "vec1", embedding,
        owner_id="user1",
        read_users=["user2", "user3"]
    )
    results = secure_db.search_with_acl("user2", query_embedding, k=5)

Author: AI-Mastery-2026
Version: 2.0.0
"""

# Core indexing
from .core import LinearVectorIndex, VectorIndex, VectorItem

# HNSW implementation
from .hnsw import HNSWIndex

# Multi-tenant support
from .multi_tenant import MultiTenantVectorDB, TenantQuota

# ACL-based filtering
from .acl import ACLEntry, ACLFilter, ACLPermission

# Drift detection
from .drift import EmbeddingDriftDetector

# Combined classes
from .secure_db import SecureVectorDB, VectorDB

# Utility functions
from .utils import (
    benchmark_index,
    batch_similarity,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    normalize_vector,
)

__all__ = [
    # Core
    "VectorIndex",
    "VectorItem",
    "LinearVectorIndex",
    # HNSW
    "HNSWIndex",
    # Multi-tenant
    "MultiTenantVectorDB",
    "TenantQuota",
    # ACL
    "ACLPermission",
    "ACLEntry",
    "ACLFilter",
    # Drift
    "EmbeddingDriftDetector",
    # Combined
    "VectorDB",
    "SecureVectorDB",
    # Utils
    "cosine_similarity",
    "euclidean_distance",
    "dot_product",
    "normalize_vector",
    "batch_similarity",
    "benchmark_index",
]

__version__ = "2.0.0"
