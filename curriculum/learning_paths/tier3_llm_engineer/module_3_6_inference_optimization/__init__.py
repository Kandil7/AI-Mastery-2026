"""
Module 3.6: Inference Optimization

Production-ready inference optimization:
- Flash Attention: Memory-efficient attention
- KV Cache: Key-value cache management
- Speculative Decoding: Draft models and verification
- Batching: Continuous batching and scheduling
"""

from .flash_attention import (
    FlashAttention,
    MemoryEfficientAttention,
    AttentionConfig,
)
from .kv_cache import (
    KVCache,
    PagedKVCache,
    KVCacheManager,
    CacheConfig,
)
from .speculative_decoding import (
    SpeculativeDecoder,
    DraftModel,
    VerificationStrategy,
    SpeculativeConfig,
)
from .batching import (
    ContinuousBatcher,
    RequestQueue,
    PriorityScheduler,
    BatchingConfig,
)

__all__ = [
    # Flash Attention
    "FlashAttention",
    "MemoryEfficientAttention",
    "AttentionConfig",
    # KV Cache
    "KVCache",
    "PagedKVCache",
    "KVCacheManager",
    "CacheConfig",
    # Speculative Decoding
    "SpeculativeDecoder",
    "DraftModel",
    "VerificationStrategy",
    "SpeculativeConfig",
    # Batching
    "ContinuousBatcher",
    "RequestQueue",
    "PriorityScheduler",
    "BatchingConfig",
]

__version__ = "1.0.0"
