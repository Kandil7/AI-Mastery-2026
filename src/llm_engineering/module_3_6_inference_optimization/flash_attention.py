"""
Flash Attention Module

Production-ready memory-efficient attention implementations:
- Flash Attention 2
- Memory-efficient attention
- Block-sparse attention

Features:
- Reduced memory footprint
- Faster computation
- GPU optimization
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""

    # Model dimensions
    hidden_size: int = 4096
    num_heads: int = 32
    head_dim: Optional[int] = None
    num_kv_heads: Optional[int] = None  # For GQA

    # Attention settings
    dropout: float = 0.0
    softmax_scale: Optional[float] = None
    causal: bool = True  # Causal masking

    # Flash attention settings
    use_flash: bool = True
    block_size: int = 64
    tile_size: int = 128

    # Memory settings
    use_fp16: bool = True
    use_bf16: bool = False

    def __post_init__(self) -> None:
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_heads
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.softmax_scale is None:
            self.softmax_scale = 1.0 / math.sqrt(self.head_dim)


class FlashAttention:
    """
    Flash Attention 2 implementation.

    Reduces memory complexity from O(N²) to O(N) by
    computing attention in blocks.
    """

    def __init__(self, config: AttentionConfig) -> None:
        self.config = config
        self._flash_attn = None

        self._try_import_flash_attn()

    def _try_import_flash_attn(self) -> None:
        """Try to import flash-attn library."""
        try:
            from flash_attn import flash_attn_func
            self._flash_attn = flash_attn_func
            logger.info("Flash Attention imported successfully")
        except ImportError:
            logger.warning("flash-attn not installed. Using fallback implementation.")

    def forward(
        self,
        query: Any,  # Tensor
        key: Any,
        value: Any,
        attention_mask: Optional[Any] = None,
    ) -> Any:
        """
        Forward pass with flash attention.

        Args:
            query: Query tensor [batch, seq_len, num_heads, head_dim]
            key: Key tensor
            value: Value tensor
            attention_mask: Optional attention mask

        Returns:
            Output tensor [batch, seq_len, num_heads, head_dim]
        """
        if self._flash_attn:
            return self._forward_flash(query, key, value, attention_mask)
        else:
            return self._forward_fallback(query, key, value, attention_mask)

    def _forward_flash(
        self,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Optional[Any],
    ) -> Any:
        """Forward using flash-attn library."""
        # flash-attn expects [batch, seq_len, num_heads, head_dim]
        output = self._flash_attn(
            q=query,
            k=key,
            v=value,
            dropout_p=self.config.dropout,
            softmax_scale=self.config.softmax_scale,
            causal=self.config.causal,
        )
        return output

    def _forward_fallback(
        self,
        query: Any,
        key: Any,
        value: Any,
        attention_mask: Optional[Any],
    ) -> Any:
        """Fallback implementation using standard attention."""
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch required for fallback attention")

        # Get dimensions
        batch_size, seq_len, num_heads, head_dim = query.shape

        # Reshape for attention computation
        # [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, seq_len, head_dim]
        q = query.transpose(1, 2)
        k = key.transpose(1, 2)
        v = value.transpose(1, 2)

        # Compute attention scores
        # [batch, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.config.softmax_scale

        # Apply causal mask
        if self.config.causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=query.device),
                diagonal=1,
            ).bool()
            scores = scores.masked_fill(mask, float("-inf"))

        # Apply attention mask if provided
        if attention_mask is not None:
            scores = scores + attention_mask

        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply dropout
        if self.config.dropout > 0:
            attn_weights = torch.dropout(attn_weights, self.config.dropout, train=True)

        # Apply attention to values
        # [batch, num_heads, seq_len, head_dim]
        output = torch.matmul(attn_weights, v)

        # Reshape back
        output = output.transpose(1, 2)

        return output

    def forward_with_cache(
        self,
        query: Any,
        key: Any,
        value: Any,
        kv_cache: Optional[Tuple[Any, Any]] = None,
    ) -> Tuple[Any, Tuple[Any, Any]]:
        """
        Forward pass with KV cache support.

        Args:
            query: Query tensor
            key: Key tensor (current token)
            value: Value tensor (current token)
            kv_cache: Optional (key_cache, value_cache)

        Returns:
            Tuple of (output, updated_kv_cache)
        """
        if kv_cache is not None:
            key_cache, value_cache = kv_cache
            key = torch.cat([key_cache, key], dim=1)
            value = torch.cat([value_cache, value], dim=1)

        output = self.forward(query, key, value)

        new_kv_cache = (key, value)
        return output, new_kv_cache


class MemoryEfficientAttention:
    """
    Memory-efficient attention using block-wise computation.

    Implements the approach from "Self-Attention Does Not Need O(n²) Memory"
    """

    def __init__(
        self,
        config: AttentionConfig,
        block_size: int = 64,
    ) -> None:
        self.config = config
        self.block_size = block_size

    def forward(
        self,
        query: Any,
        key: Any,
        value: Any,
    ) -> Any:
        """
        Memory-efficient forward pass.

        Computes attention in blocks to reduce memory usage.
        """
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch required")

        batch_size, seq_len, num_heads, head_dim = query.shape

        # Initialize output and statistics
        output = torch.zeros_like(query)
        row_max = torch.full(
            (batch_size, num_heads, seq_len, 1),
            float("-inf"),
            device=query.device,
        )
        row_sum = torch.zeros(
            (batch_size, num_heads, seq_len, 1),
            device=query.device,
        )

        # Process in blocks
        num_blocks = (seq_len + self.block_size - 1) // self.block_size

        for j in range(num_blocks):
            j_start = j * self.block_size
            j_end = min((j + 1) * self.block_size, seq_len)

            # Get block of keys and values
            k_block = key[:, j_start:j_end, :, :]
            v_block = value[:, j_start:j_end, :, :]

            for i in range(num_blocks):
                i_start = i * self.block_size
                i_end = min((i + 1) * self.block_size, seq_len)

                # Get block of queries
                q_block = query[:, i_start:i_end, :, :]

                # Compute attention scores for this block
                scores = torch.einsum("bnhd,bmhd->bnhm", q_block, k_block)
                scores *= self.config.softmax_scale

                # Apply causal mask within block
                if self.config.causal:
                    mask = torch.triu(
                        torch.ones(i_end - i_start, j_end - j_start),
                        diagonal=j_start - i_start + 1,
                    ).to(query.device).bool()
                    scores = scores.masked_fill(mask[:, :, None, None], float("-inf"))

                # Compute block max and sum
                block_max = scores.max(dim=-1, keepdim=True).values
                block_weights = torch.exp(scores - block_max)
                block_sum = block_weights.sum(dim=-1, keepdim=True)

                # Update running statistics
                new_max = torch.maximum(row_max[:, :, i_start:i_end], block_max)
                exp_diff = torch.exp(row_max[:, :, i_start:i_end] - new_max)
                block_weights_scaled = torch.exp(block_max - new_max) * block_weights

                # Update output
                output_block = torch.einsum("bnhm,bmhd->bnhd", block_weights, v_block)
                output[:, i_start:i_end] = (
                    output[:, i_start:i_end] * exp_diff +
                    block_weights_scaled * output_block
                )

                # Update running sum
                row_sum[:, :, i_start:i_end] = (
                    row_sum[:, :, i_start:i_end] * exp_diff + block_sum * block_weights_scaled
                )

                # Update running max
                row_max[:, :, i_start:i_end] = new_max

        # Normalize output
        output = output / row_sum

        return output


class BlockSparseAttention:
    """
    Block-sparse attention for long sequences.

    Uses predefined sparsity patterns to reduce computation.
    """

    def __init__(
        self,
        config: AttentionConfig,
        block_size: int = 64,
        num_local_blocks: int = 4,
        num_global_blocks: int = 2,
    ) -> None:
        self.config = config
        self.block_size = block_size
        self.num_local_blocks = num_local_blocks
        self.num_global_blocks = num_global_blocks

    def forward(
        self,
        query: Any,
        key: Any,
        value: Any,
    ) -> Any:
        """Forward pass with block-sparse attention."""
        try:
            import torch
        except ImportError:
            raise RuntimeError("PyTorch required")

        batch_size, seq_len, num_heads, head_dim = query.shape
        num_blocks = seq_len // self.block_size

        # Create sparse attention mask
        attention_mask = self._create_sparse_mask(num_blocks)

        # Reshape into blocks
        q_blocks = query.view(batch_size, num_blocks, self.block_size, num_heads, head_dim)
        k_blocks = key.view(batch_size, num_blocks, self.block_size, num_heads, head_dim)
        v_blocks = value.view(batch_size, num_blocks, self.block_size, num_heads, head_dim)

        # Compute attention with sparsity
        output_blocks = []

        for i in range(num_blocks):
            # Get blocks to attend to
            attend_to = self._get_attention_targets(i, num_blocks)

            q_block = q_blocks[:, i]
            k_attend = torch.cat([k_blocks[:, j] for j in attend_to], dim=1)
            v_attend = torch.cat([v_blocks[:, j] for j in attend_to], dim=1)

            # Compute attention
            scores = torch.einsum("bnhd,bmhd->bnhm", q_block, k_attend)
            scores *= self.config.softmax_scale

            attn_weights = torch.softmax(scores, dim=-1)
            output = torch.einsum("bnhm,bmhd->bnhd", attn_weights, v_attend)

            output_blocks.append(output)

        # Concatenate and reshape
        output = torch.stack(output_blocks, dim=1)
        output = output.reshape(batch_size, seq_len, num_heads, head_dim)

        return output

    def _create_sparse_mask(self, num_blocks: int) -> Any:
        """Create sparse attention mask."""
        import torch

        mask = torch.zeros(num_blocks, num_blocks)

        for i in range(num_blocks):
            # Local attention
            for j in range(max(0, i - self.num_local_blocks), min(num_blocks, i + self.num_local_blocks + 1)):
                mask[i, j] = 1

            # Global attention (first and last blocks)
            for j in range(self.num_global_blocks):
                mask[i, j] = 1
                mask[i, num_blocks - 1 - j] = 1

        return mask.bool()

    def _get_attention_targets(self, block_idx: int, num_blocks: int) -> List[int]:
        """Get list of blocks to attend to."""
        targets = []

        # Local blocks
        for j in range(max(0, block_idx - self.num_local_blocks),
                       min(num_blocks, block_idx + self.num_local_blocks + 1)):
            targets.append(j)

        # Global blocks
        for j in range(self.num_global_blocks):
            if j not in targets:
                targets.append(j)
            if num_blocks - 1 - j not in targets:
                targets.append(num_blocks - 1 - j)

        return sorted(targets)


class AttentionRegistry:
    """
    Registry for attention implementations.

    Provides unified interface to different attention mechanisms.
    """

    _implementations: Dict[str, type] = {
        "flash": FlashAttention,
        "memory_efficient": MemoryEfficientAttention,
        "block_sparse": BlockSparseAttention,
    }

    @classmethod
    def register(cls, name: str, impl_class: type) -> None:
        """Register attention implementation."""
        cls._implementations[name] = impl_class

    @classmethod
    def create(
        cls,
        name: str,
        config: AttentionConfig,
        **kwargs: Any,
    ) -> Any:
        """Create attention implementation."""
        impl_class = cls._implementations.get(name)
        if not impl_class:
            raise ValueError(f"Unknown attention implementation: {name}")

        return impl_class(config, **kwargs)

    @classmethod
    def list_implementations(cls) -> List[str]:
        """List available implementations."""
        return list(cls._implementations.keys())


def select_attention_type(
    seq_len: int,
    available_memory_gb: float,
    has_flash_attn: bool = True,
) -> str:
    """
    Select appropriate attention type based on constraints.

    Args:
        seq_len: Sequence length
        available_memory_gb: Available GPU memory in GB
        has_flash_attn: Whether flash-attn is installed

    Returns:
        Recommended attention type
    """
    # Estimate memory requirements
    # Standard attention: O(n²) memory
    # Flash attention: O(n) memory

    if has_flash_attn and seq_len > 512:
        return "flash"

    if seq_len > 4096:
        return "block_sparse"

    if available_memory_gb < 8 and seq_len > 1024:
        return "memory_efficient"

    return "flash" if has_flash_attn else "memory_efficient"
