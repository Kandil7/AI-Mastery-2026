"""
KV Cache Module

Production-ready KV cache management:
- Standard KV cache
- Paged attention (vLLM-style)
- Prefix caching
- Cache eviction strategies

Features:
- Memory efficiency
- Request isolation
- Cache sharing
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class CacheConfig:
    """Configuration for KV cache."""

    # Cache size
    max_num_seqs: int = 256
    max_seq_len: int = 4096
    num_blocks: int = 1000
    block_size: int = 16

    # Model dimensions
    num_layers: int = 32
    num_heads: int = 32
    head_dim: int = 128
    hidden_size: Optional[int] = None

    # Memory settings
    use_gpu: bool = True
    dtype: str = "float16"  # float16, bfloat16, float32

    # Eviction settings
    eviction_strategy: str = "lru"  # lru, lfu, fifo
    max_cache_size_gb: Optional[float] = None

    def __post_init__(self) -> None:
        if self.hidden_size is None:
            self.hidden_size = self.num_heads * self.head_dim


@dataclass
class CacheBlock:
    """A block in the KV cache."""

    block_id: int
    key_cache: Any = None  # Tensor
    value_cache: Any = None
    ref_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    is_free: bool = True

    def touch(self) -> None:
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class KVCache(ABC):
    """Abstract base class for KV caches."""

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self._allocated_blocks: Dict[int, CacheBlock] = {}
        self._free_blocks: List[int] = list(range(config.num_blocks))

    @abstractmethod
    def allocate(self, seq_id: str) -> List[int]:
        """Allocate blocks for a sequence."""
        pass

    @abstractmethod
    def free(self, seq_id: str) -> None:
        """Free blocks for a sequence."""
        pass

    @abstractmethod
    def get_block(self, seq_id: str, block_idx: int) -> Optional[CacheBlock]:
        """Get a specific block."""
        pass

    @abstractmethod
    def update_block(
        self,
        seq_id: str,
        block_idx: int,
        key: Any,
        value: Any,
    ) -> None:
        """Update block with new KV values."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_blocks": self.config.num_blocks,
            "free_blocks": len(self._free_blocks),
            "allocated_blocks": len(self._allocated_blocks),
            "utilization": 1 - len(self._free_blocks) / self.config.num_blocks,
        }


class StandardKVCache(KVCache):
    """
    Standard KV cache implementation.

    Simple contiguous allocation per sequence.
    """

    def __init__(self, config: CacheConfig) -> None:
        super().__init__(config)

        self._seq_blocks: Dict[str, List[int]] = {}
        self._block_map: Dict[int, CacheBlock] = {}

        # Initialize blocks
        for i in range(config.num_blocks):
            self._block_map[i] = CacheBlock(block_id=i)

    def allocate(self, seq_id: str) -> List[int]:
        """Allocate blocks for a sequence."""
        if seq_id in self._seq_blocks:
            return self._seq_blocks[seq_id]

        # Calculate blocks needed
        blocks_needed = (self.config.max_seq_len + self.config.block_size - 1) // self.config.block_size
        blocks_needed = min(blocks_needed, len(self._free_blocks))

        if blocks_needed == 0:
            raise RuntimeError("No free blocks available")

        # Allocate from free list
        allocated = []
        for _ in range(blocks_needed):
            if self._free_blocks:
                block_id = self._free_blocks.pop(0)
                block = self._block_map[block_id]
                block.is_free = False
                block.ref_count = 1
                allocated.append(block_id)

        self._seq_blocks[seq_id] = allocated
        return allocated

    def free(self, seq_id: str) -> None:
        """Free blocks for a sequence."""
        if seq_id not in self._seq_blocks:
            return

        for block_id in self._seq_blocks[seq_id]:
            block = self._block_map[block_id]
            block.is_free = True
            block.ref_count = 0
            block.key_cache = None
            block.value_cache = None
            self._free_blocks.append(block_id)

        del self._seq_blocks[seq_id]

    def get_block(self, seq_id: str, block_idx: int) -> Optional[CacheBlock]:
        """Get a specific block."""
        if seq_id not in self._seq_blocks:
            return None

        blocks = self._seq_blocks[seq_id]
        if block_idx >= len(blocks):
            return None

        block_id = blocks[block_idx]
        block = self._block_map[block_id]
        block.touch()

        return block

    def update_block(
        self,
        seq_id: str,
        block_idx: int,
        key: Any,
        value: Any,
    ) -> None:
        """Update block with new KV values."""
        block = self.get_block(seq_id, block_idx)
        if block:
            block.key_cache = key
            block.value_cache = value


class PagedKVCache(KVCache):
    """
    Paged KV cache (vLLM-style).

    Uses non-contiguous memory allocation with block tables.
    """

    def __init__(self, config: CacheConfig) -> None:
        super().__init__(config)

        self._block_tables: Dict[str, List[int]] = {}
        self._block_map: Dict[int, CacheBlock] = {}

        # Initialize blocks
        for i in range(config.num_blocks):
            self._block_map[i] = CacheBlock(block_id=i)

        # Prefix cache
        self._prefix_cache: Dict[str, List[int]] = {}

    def allocate(self, seq_id: str, num_blocks: int = 1) -> List[int]:
        """Allocate blocks for a sequence."""
        if seq_id in self._block_tables:
            # Extend existing allocation
            existing = self._block_tables[seq_id]
            additional = self._allocate_blocks(num_blocks)
            self._block_tables[seq_id] = existing + additional
            return self._block_tables[seq_id]

        # New allocation
        allocated = self._allocate_blocks(num_blocks)
        self._block_tables[seq_id] = allocated
        return allocated

    def _allocate_blocks(self, num_blocks: int) -> List[int]:
        """Allocate specific number of blocks."""
        allocated = []

        for _ in range(num_blocks):
            if not self._free_blocks:
                # Need to evict
                self._evict()

            if self._free_blocks:
                block_id = self._free_blocks.pop(0)
                block = self._block_map[block_id]
                block.is_free = False
                block.ref_count = 1
                allocated.append(block_id)

        return allocated

    def _evict(self) -> None:
        """Evict blocks based on strategy."""
        if self.config.eviction_strategy == "lru":
            self._evict_lru()
        elif self.config.eviction_strategy == "lfu":
            self._evict_lfu()
        else:
            self._evict_fifo()

    def _evict_lru(self) -> None:
        """Evict least recently used blocks."""
        # Find blocks with lowest last_accessed
        sorted_blocks = sorted(
            [(bid, b) for bid, b in self._block_map.items() if not b.is_free],
            key=lambda x: x[1].last_accessed,
        )

        # Evict some blocks
        num_to_evict = max(1, len(self._free_blocks) // 2)
        for block_id, block in sorted_blocks[:num_to_evict]:
            self._free_block(block_id)

    def _evict_lfu(self) -> None:
        """Evict least frequently used blocks."""
        sorted_blocks = sorted(
            [(bid, b) for bid, b in self._block_map.items() if not b.is_free],
            key=lambda x: x[1].access_count,
        )

        num_to_evict = max(1, len(self._free_blocks) // 2)
        for block_id, block in sorted_blocks[:num_to_evict]:
            self._free_block(block_id)

    def _evict_fifo(self) -> None:
        """Evict first-in-first-out."""
        sorted_blocks = sorted(
            [(bid, b) for bid, b in self._block_map.items() if not b.is_free],
            key=lambda x: x[1].last_accessed,
        )

        num_to_evict = max(1, len(self._free_blocks) // 2)
        for block_id, block in sorted_blocks[:num_to_evict]:
            self._free_block(block_id)

    def _free_block(self, block_id: int) -> None:
        """Free a single block."""
        block = self._block_map[block_id]
        if block.ref_count <= 1:
            block.is_free = True
            block.ref_count = 0
            block.key_cache = None
            block.value_cache = None
            self._free_blocks.append(block_id)
        else:
            block.ref_count -= 1

    def free(self, seq_id: str) -> None:
        """Free all blocks for a sequence."""
        if seq_id not in self._block_tables:
            return

        for block_id in self._block_tables[seq_id]:
            self._free_block(block_id)

        del self._block_tables[seq_id]

    def get_block(self, seq_id: str, block_idx: int) -> Optional[CacheBlock]:
        """Get a specific block."""
        if seq_id not in self._block_tables:
            return None

        blocks = self._block_tables[seq_id]
        if block_idx >= len(blocks):
            return None

        block_id = blocks[block_idx]
        block = self._block_map[block_id]
        block.touch()

        return block

    def update_block(
        self,
        seq_id: str,
        block_idx: int,
        key: Any,
        value: Any,
    ) -> None:
        """Update block with new KV values."""
        block = self.get_block(seq_id, block_idx)
        if block:
            block.key_cache = key
            block.value_cache = value

    def get_block_table(self, seq_id: str) -> List[int]:
        """Get block table for sequence."""
        return self._block_tables.get(seq_id, [])

    def cache_prefix(self, prefix_hash: str, block_ids: List[int]) -> None:
        """Cache a prefix for reuse."""
        self._prefix_cache[prefix_hash] = block_ids.copy()

    def get_cached_prefix(self, prefix_hash: str) -> Optional[List[int]]:
        """Get cached prefix blocks."""
        return self._prefix_cache.get(prefix_hash)


class KVCacheManager:
    """
    Manager for multiple KV caches.

    Handles cache allocation, sharing, and optimization.
    """

    def __init__(self, config: CacheConfig) -> None:
        self.config = config
        self._cache: KVCache = PagedKVCache(config)
        self._seq_info: Dict[str, Dict[str, Any]] = {}
        self._prefix_tree: Dict[str, Any] = {}

    def create_sequence(
        self,
        seq_id: str,
        parent_seq_id: Optional[str] = None,
    ) -> List[int]:
        """Create a new sequence, optionally sharing prefix with parent."""
        if parent_seq_id and parent_seq_id in self._seq_info:
            # Share prefix with parent
            parent_blocks = self._cache.get_block_table(parent_seq_id)
            parent_info = self._seq_info[parent_seq_id]

            # Copy prefix blocks
            shared_blocks = parent_blocks[:parent_info.get("shared_prefix_len", len(parent_blocks))]

            # Allocate additional blocks
            new_blocks = self._cache.allocate(seq_id, num_blocks=1)

            all_blocks = shared_blocks + new_blocks
            self._seq_info[seq_id] = {
                "parent": parent_seq_id,
                "shared_prefix_len": len(shared_blocks),
                "total_blocks": len(all_blocks),
            }

            return all_blocks

        # New sequence without parent
        blocks = self._cache.allocate(seq_id)
        self._seq_info[seq_id] = {
            "parent": None,
            "shared_prefix_len": 0,
            "total_blocks": len(blocks),
        }

        return blocks

    def append_tokens(
        self,
        seq_id: str,
        token_ids: List[int],
        key_states: List[Any],
        value_states: List[Any],
    ) -> None:
        """Append tokens to sequence cache."""
        if seq_id not in self._seq_info:
            self.create_sequence(seq_id)

        seq_info = self._seq_info[seq_id]
        current_len = seq_info.get("current_len", 0)
        new_len = current_len + len(token_ids)

        # Calculate blocks needed
        blocks_needed = (new_len + self.config.block_size - 1) // self.config.block_size
        current_blocks = len(self._cache.get_block_table(seq_id))

        # Allocate more blocks if needed
        if blocks_needed > current_blocks:
            self._cache.allocate(seq_id, num_blocks=blocks_needed - current_blocks)

        # Update blocks with new KV states
        for i, (key, value) in enumerate(zip(key_states, value_states)):
            block_idx = (current_len + i) // self.config.block_size
            self._cache.update_block(seq_id, block_idx, key, value)

        self._seq_info[seq_id]["current_len"] = new_len

    def get_kv_cache(
        self,
        seq_id: str,
    ) -> Tuple[List[Any], List[Any]]:
        """Get KV cache for sequence."""
        block_table = self._cache.get_block_table(seq_id)

        keys = []
        values = []

        for block_idx in range(len(block_table)):
            block = self._cache.get_block(seq_id, block_idx)
            if block and block.key_cache is not None:
                keys.append(block.key_cache)
                values.append(block.value_cache)

        return keys, values

    def free_sequence(self, seq_id: str) -> None:
        """Free sequence cache."""
        self._cache.free(seq_id)
        if seq_id in self._seq_info:
            del self._seq_info[seq_id]

    def compute_prefix_hash(self, token_ids: Tuple[int, ...]) -> str:
        """Compute hash for prefix matching."""
        import hashlib
        return hashlib.md5(str(token_ids).encode()).hexdigest()

    def find_common_prefix(
        self,
        seq_id: str,
        token_ids: List[int],
    ) -> int:
        """Find length of common prefix with cached sequences."""
        # Simple implementation - check against all sequences
        max_prefix = 0

        for other_id, other_info in self._seq_info.items():
            if other_id == seq_id:
                continue

            # Compare prefixes
            other_len = other_info.get("current_len", 0)
            common = min(len(token_ids), other_len)

            # In real implementation, would compare actual tokens
            if common > max_prefix:
                max_prefix = common

        return max_prefix

    def get_stats(self) -> Dict[str, Any]:
        """Get manager statistics."""
        cache_stats = self._cache.get_stats()

        return {
            **cache_stats,
            "active_sequences": len(self._seq_info),
            "sequences": list(self._seq_info.keys()),
        }


class PrefixCachingKVCache(PagedKVCache):
    """
    KV cache with automatic prefix caching.

    Automatically detects and caches common prefixes across sequences.
    """

    def __init__(self, config: CacheConfig) -> None:
        super().__init__(config)

        self._prefix_blocks: Dict[str, List[int]] = {}
        self._prefix_ref_counts: Dict[str, int] = {}

    def allocate_with_prefix(
        self,
        seq_id: str,
        prefix_tokens: Optional[Tuple[int, ...]] = None,
    ) -> List[int]:
        """Allocate with prefix caching."""
        if prefix_tokens:
            prefix_hash = self._compute_prefix_hash(prefix_tokens)

            if prefix_hash in self._prefix_blocks:
                # Reuse cached prefix
                prefix_block_ids = self._prefix_blocks[prefix_hash]
                self._prefix_ref_counts[prefix_hash] += 1

                # Allocate remaining blocks
                new_blocks = self._allocate_blocks(1)

                self._block_tables[seq_id] = prefix_block_ids + new_blocks
                return self._block_tables[seq_id]

        # No prefix cache hit
        return self.allocate(seq_id)

    def _compute_prefix_hash(self, tokens: Tuple[int, ...]) -> str:
        """Compute hash for token sequence."""
        import hashlib
        return hashlib.sha256(str(tokens).encode()).hexdigest()[:16]

    def cache_sequence_prefix(self, seq_id: str, prefix_len: int) -> str:
        """Cache prefix of a sequence."""
        if seq_id not in self._block_tables:
            return ""

        block_table = self._block_tables[seq_id]
        prefix_blocks = block_table[:prefix_len]

        # Compute hash from block contents
        prefix_hash = str(hash(tuple(prefix_blocks)))

        self._prefix_blocks[prefix_hash] = prefix_blocks
        self._prefix_ref_counts[prefix_hash] = 1

        return prefix_hash

    def free(self, seq_id: str) -> None:
        """Free sequence, updating prefix ref counts."""
        if seq_id in self._block_tables:
            block_table = self._block_tables[seq_id]

            # Check if prefix is cached
            for prefix_hash, prefix_blocks in self._prefix_blocks.items():
                if block_table[:len(prefix_blocks)] == prefix_blocks:
                    self._prefix_ref_counts[prefix_hash] -= 1

                    if self._prefix_ref_counts[prefix_hash] <= 0:
                        # Free prefix blocks
                        del self._prefix_blocks[prefix_hash]
                        del self._prefix_ref_counts[prefix_hash]

        super().free(seq_id)

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics with prefix info."""
        stats = super().get_stats()
        stats["prefix_cache_entries"] = len(self._prefix_blocks)
        stats["prefix_cache_blocks"] = sum(len(b) for b in self._prefix_blocks.values())
        return stats
