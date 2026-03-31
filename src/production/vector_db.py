"""
Vector Database Module
=======================
Custom vector database implementation with similarity search.

NOTE: This module is now a compatibility wrapper.
The implementation has been moved to the `vector_db` submodule.

New import style:
    from src.production.vector_db import (
        VectorDB,
        SecureVectorDB,
        HNSWIndex,
        MultiTenantVectorDB,
        ACLFilter,
        EmbeddingDriftDetector,
    )

This module maintains backward compatibility by re-exporting
from the submodule.

Features:
- HNSW index for fast approximate nearest neighbor search
- Multi-tenant support with namespace isolation
- Per-tenant quotas and rate limiting
- Backup and recovery capabilities
- Metadata filtering
- ACL-based access control (Weaviate/Pinecone pattern)
- Embedding drift detection (Arize/Fiddler pattern)

Author: AI-Mastery-2026
Version: 2.0.0 (modular)
"""

# Re-export all classes from submodule for backward compatibility
from src.production.vector_db import (
    ACLEntry,
    ACLFilter,
    ACLPermission,
    EmbeddingDriftDetector,
    HNSWIndex,
    LinearVectorIndex,
    MultiTenantVectorDB,
    SecureVectorDB,
    TenantQuota,
    VectorDB,
    VectorIndex,
    VectorItem,
    benchmark_index,
    batch_similarity,
    cosine_similarity,
    dot_product,
    euclidean_distance,
    normalize_vector,
)

# For backward compatibility, also expose at module level
__all__ = [
    "VectorIndex",
    "VectorItem",
    "LinearVectorIndex",
    "HNSWIndex",
    "VectorDB",
    "SecureVectorDB",
    "MultiTenantVectorDB",
    "TenantQuota",
    "ACLPermission",
    "ACLEntry",
    "ACLFilter",
    "EmbeddingDriftDetector",
    "cosine_similarity",
    "euclidean_distance",
    "dot_product",
    "normalize_vector",
    "batch_similarity",
    "benchmark_index",
]

__version__ = "2.0.0"
