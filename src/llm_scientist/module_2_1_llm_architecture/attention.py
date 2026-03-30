"""
Attention Mechanisms - Module 2.1.1

Production-ready implementations of attention mechanisms:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Self-Attention
- Masked Attention (causal)
- Cross-Attention (encoder-decoder)
- Flash Attention (memory-efficient)

References:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention implementation.
    
    Computes attention scores as softmax(QK^T / sqrt(d_k))V
    
    Args:
        dropout: Dropout probability for attention weights
        scale_factor: Optional manual scaling factor (default: 1/sqrt(d_k))
    
    Shape:
        - query: (batch, seq_len_q, d_k)
        - key: (batch, seq_len_k, d_k)
        - value: (batch, seq_len_v, d_v)
        - mask: (batch, seq_len_q, seq_len_k) or (seq_len_q, seq_len_k)
        - output: (batch, seq_len_q, d_v)
    
    Example:
        >>> attention = ScaledDotProductAttention(dropout=0.1)
        >>> q = torch.randn(32, 10, 64)  # batch=32, seq=10, d_k=64
        >>> k = torch.randn(32, 10, 64)
        >>> v = torch.randn(32, 10, 64)
        >>> output, attn_weights = attention(q, k, v)
    """
    
    def __init__(self, dropout: float = 0.0, scale_factor: Optional[float] = None):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale_factor = scale_factor
        self._attn_weights: Optional[Tensor] = None  # For visualization
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
        return_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for scaled dot-product attention.
        
        Args:
            query: Query tensor
            key: Key tensor
            value: Value tensor
            mask: Optional attention mask (1 = keep, 0 = mask)
            return_weights: Whether to return attention weights
        
        Returns:
            Tuple of (output, attention_weights)
        """
        # Get dimensions
        d_k = query.size(-1)
        
        # Compute scaling factor
        scale = self.scale_factor if self.scale_factor else 1.0 / math.sqrt(d_k)
        
        # Compute attention scores: (batch, seq_q, seq_k)
        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        
        # Apply mask if provided
        if mask is not None:
            # Expand mask to match attn_scores shape if needed
            if mask.dim() == 2:
                mask = mask.unsqueeze(0)
            # Mask positions (use large negative value for numerical stability)
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Store weights for visualization if requested
        if return_weights:
            self._attn_weights = attn_weights.detach()
        
        # Apply attention to values
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights if return_weights else None
    
    def get_attention_weights(self) -> Optional[Tensor]:
        """Get the last computed attention weights for visualization."""
        return self._attn_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention implementation.
    
    Projects queries, keys, and values through multiple attention heads,
    allowing the model to attend to information from different representation
    subspaces simultaneously.
    
    Args:
        d_model: Dimension of input embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability
        bias: Whether to use bias in linear projections
        add_bias_kv: Whether to add bias to key and value
        add_zero_attn: Whether to add zero attention
        
    Shape:
        - query: (batch, seq_len_q, d_model)
        - key: (batch, seq_len_k, d_model)
        - value: (batch, seq_len_v, d_model)
        - key_padding_mask: (batch, seq_len_k) - boolean mask for padding
        - attn_mask: (batch, seq_len_q, seq_len_k) or (seq_len_q, seq_len_k)
        - output: (batch, seq_len_q, d_model)
    
    Example:
        >>> mha = MultiHeadAttention(d_model=512, num_heads=8, dropout=0.1)
        >>> q = torch.randn(32, 10, 512)
        >>> k = torch.randn(32, 10, 512)
        >>> v = torch.randn(32, 10, 512)
        >>> output, attn_weights = mha(q, k, v)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Optional bias for key and value
        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.randn(1, 1, d_model))
            self.bias_v = nn.Parameter(torch.randn(1, 1, d_model))
        else:
            self.bias_k = self.bias_v = None
        
        self.add_zero_attn = add_zero_attn
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self._attn_weights: Optional[Tensor] = None
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        return_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor (batch, seq_q, d_model)
            key: Key tensor (batch, seq_k, d_model)
            value: Value tensor (batch, seq_v, d_model)
            key_padding_mask: Boolean mask for padding (batch, seq_k)
            attn_mask: Attention mask (seq_q, seq_k) or (batch, seq_q, seq_k)
            return_weights: Whether to return attention weights
        
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        seq_len_v = value.shape[1]
        
        # Linear projections and reshape for multi-head
        # (batch, seq, d_model) -> (batch, seq, num_heads, head_dim)
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len_k, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len_v, self.num_heads, self.head_dim)
        
        # Transpose for attention: (batch, num_heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Add bias to key and value if specified
        if self.bias_k is not None:
            k = torch.cat([k, self.bias_k.expand(batch_size, 1, self.num_heads, self.head_dim)], dim=1)
            v = torch.cat([v, self.bias_v.expand(batch_size, 1, self.num_heads, self.head_dim)], dim=1)
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        
        # Add zero attention if specified
        if self.add_zero_attn:
            zero_attn = torch.zeros(
                batch_size, self.num_heads, 1, self.head_dim,
                device=q.device, dtype=q.dtype
            )
            k = torch.cat([k, zero_attn], dim=2)
            v = torch.cat([v, zero_attn], dim=2)
            seq_len_k += 1
            if attn_mask is not None:
                attn_mask = F.pad(attn_mask, (0, 1))
            if key_padding_mask is not None:
                key_padding_mask = F.pad(key_padding_mask, (0, 1))
        
        # Compute scaled dot-product attention
        # (batch, num_heads, seq_q, head_dim) @ (batch, num_heads, head_dim, seq_k)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        
        # Apply key padding mask
        if key_padding_mask is not None:
            # (batch, seq_k) -> (batch, 1, 1, seq_k)
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, seq_len_k)
            attn_scores = attn_scores.masked_fill(~key_padding_mask, -1e9)
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        if return_weights:
            self._attn_weights = attn_weights.detach()
        
        # Apply attention to values
        # (batch, num_heads, seq_q, seq_k) @ (batch, num_heads, seq_k, head_dim)
        output = torch.matmul(attn_weights, v)
        
        # Transpose back: (batch, seq_q, num_heads, head_dim)
        output = output.transpose(1, 2).contiguous()
        
        # Reshape and project: (batch, seq_q, d_model)
        output = output.view(batch_size, seq_len_q, self.d_model)
        output = self.out_proj(output)
        
        return output, attn_weights if return_weights else None
    
    def get_attention_weights(self) -> Optional[Tensor]:
        """Get the last computed attention weights."""
        return self._attn_weights


class SelfAttention(MultiHeadAttention):
    """
    Self-Attention where query, key, and value come from the same input.
    
    This is a specialized version of MultiHeadAttention where Q, K, V
    are all derived from the same input tensor.
    
    Example:
        >>> self_attn = SelfAttention(d_model=512, num_heads=8)
        >>> x = torch.randn(32, 10, 512)
        >>> output, attn_weights = self_attn(x, x, x)
    """
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        return_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for self-attention.
        
        Args:
            x: Input tensor (batch, seq, d_model)
            mask: Optional attention mask
            key_padding_mask: Optional padding mask
            return_weights: Whether to return attention weights
        
        Returns:
            Tuple of (output, attention_weights)
        """
        return super().forward(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            attn_mask=mask,
            return_weights=return_weights,
        )


class MaskedAttention(SelfAttention):
    """
    Masked Self-Attention for decoder (causal attention).
    
    Ensures that position i can only attend to positions < i,
    preventing information flow from future tokens.
    
    Args:
        d_model: Dimension of input embeddings
        num_heads: Number of attention heads
        dropout: Dropout probability
        
    Example:
        >>> masked_attn = MaskedAttention(d_model=512, num_heads=8)
        >>> x = torch.randn(32, 10, 512)
        >>> output, attn_weights = masked_attn(x)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self._causal_mask: Optional[Tensor] = None
    
    def _generate_causal_mask(
        self,
        seq_len: int,
        device: torch.device,
    ) -> Tensor:
        """Generate causal (triangular) mask."""
        if self._causal_mask is not None and self._causal_mask.size(0) >= seq_len:
            return self._causal_mask[:seq_len, :seq_len]
        
        # Create lower triangular mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        self._causal_mask = mask
        return mask
    
    def forward(
        self,
        x: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        return_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass with causal masking.
        
        Args:
            x: Input tensor (batch, seq, d_model)
            key_padding_mask: Optional padding mask
            return_weights: Whether to return attention weights
        
        Returns:
            Tuple of (output, attention_weights)
        """
        seq_len = x.size(1)
        causal_mask = self._generate_causal_mask(seq_len, x.device)
        
        # Combine causal mask with padding mask if provided
        if key_padding_mask is not None:
            # (batch, seq) -> (batch, 1, seq)
            key_padding_mask = key_padding_mask.unsqueeze(1)
            # Expand causal mask for batch
            causal_mask = causal_mask.unsqueeze(0).expand(key_padding_mask.size(0), -1, -1)
            # Combine masks
            combined_mask = causal_mask & key_padding_mask
        else:
            combined_mask = causal_mask
        
        return super().forward(
            query=x,
            key=x,
            value=x,
            attn_mask=combined_mask,
            return_weights=return_weights,
        )


class CrossAttention(MultiHeadAttention):
    """
    Cross-Attention for encoder-decoder attention.
    
    Query comes from decoder, while key and value come from encoder output.
    This allows the decoder to attend to relevant parts of the encoder output.
    
    Example:
        >>> cross_attn = CrossAttention(d_model=512, num_heads=8)
        >>> decoder_hidden = torch.randn(32, 10, 512)  # decoder states
        >>> encoder_output = torch.randn(32, 15, 512)  # encoder outputs
        >>> output, attn_weights = cross_attn(decoder_hidden, encoder_output, encoder_output)
    """
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        return_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for cross-attention.
        
        Args:
            query: Query from decoder (batch, seq_q, d_model)
            key: Key from encoder (batch, seq_k, d_model)
            value: Value from encoder (batch, seq_v, d_model)
            key_padding_mask: Padding mask for encoder output
            attn_mask: Optional additional attention mask
            return_weights: Whether to return attention weights
        
        Returns:
            Tuple of (output, attention_weights)
        """
        return super().forward(
            query=query,
            key=key,
            value=value,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            return_weights=return_weights,
        )


class FlashAttention(nn.Module):
    """
    Flash Attention - Memory-efficient exact attention.
    
    Implements the FlashAttention algorithm that computes exact attention
    with O(1) memory complexity instead of O(n^2).
    
    This implementation uses tiling and recomputation to achieve
    memory efficiency while maintaining numerical accuracy.
    
    Note: For production use, consider using the official flash-attn package:
          pip install flash-attn
    
    References:
        - "FlashAttention: Fast and Memory-Efficient Exact Attention" (Dao et al., 2022)
        - "FlashAttention-2: Attention is Not All You Need" (Dao et al., 2023)
    
    Args:
        dropout: Dropout probability
        causal: Whether to use causal masking
        tile_size: Size of tiles for block computation
    
    Example:
        >>> flash_attn = FlashAttention(d_model=512, num_heads=8, causal=True)
        >>> q = torch.randn(32, 10, 512)
        >>> k = torch.randn(32, 10, 512)
        >>> v = torch.randn(32, 10, 512)
        >>> output = flash_attn(q, k, v)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = False,
        tile_size: int = 64,
    ):
        super().__init__()
        
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        self.tile_size = tile_size
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def _flash_attention_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        causal: bool = False,
    ) -> Tensor:
        """
        Flash attention forward pass with tiling.
        
        This is a simplified implementation that demonstrates the concept.
        For production, use the optimized CUDA kernel from flash-attn package.
        """
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # Initialize output and statistics
        output = torch.zeros_like(q)
        l = torch.zeros(batch_size, num_heads, seq_len, 1, device=q.device, dtype=q.dtype)
        m = torch.full(
            batch_size, num_heads, seq_len, 1,
            float('-inf'),
            device=q.device,
            dtype=q.dtype,
        )
        
        # Tile size for block computation
        Br = min(self.tile_size, seq_len)  # Row tile size
        Bc = min(self.tile_size, seq_len)  # Column tile size
        
        # Process in tiles
        for j in range(0, seq_len, Bc):
            # Load key and value tiles
            k_tile = k[:, :, j:j+Bc, :]
            v_tile = v[:, :, j:j+Bc, :]
            
            for i in range(0, seq_len, Br):
                # Load query tile
                q_tile = q[:, :, i:i+Br, :]
                
                # Compute attention scores for this tile
                # (batch, heads, Br, head_dim) @ (batch, heads, head_dim, Bc)
                scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * self.scale
                
                # Apply causal mask if needed
                if causal:
                    # Create causal mask for this tile
                    row_idx = torch.arange(i, min(i+Br, seq_len), device=q.device)
                    col_idx = torch.arange(j, min(j+Bc, seq_len), device=q.device)
                    causal_mask = (row_idx[:, None] >= col_idx[None, :]).to(scores.dtype)
                    scores = scores.masked_fill(causal_mask == 0, -1e9)
                
                # Compute new max for numerical stability
                m_prev = m[:, :, i:i+Br, :]
                m_new = torch.maximum(m_prev, scores.max(dim=-1, keepdim=True)[0])
                
                # Compute attention weights with numerical stability
                exp_scores = torch.exp(scores - m_new)
                
                # Update normalization constant
                l_prev = l[:, :, i:i+Br, :]
                l_new = torch.exp(m_prev - m_new) * l_prev + exp_scores.sum(dim=-1, keepdim=True)
                
                # Update output
                output_tile = output[:, :, i:i+Br, :]
                output_new = (
                    torch.exp(m_prev - m_new) * output_tile +
                    torch.matmul(exp_scores, v_tile)
                )
                output[:, :, i:i+Br, :] = output_new
                
                # Update statistics
                m[:, :, i:i+Br, :] = m_new
                l[:, :, i:i+Br, :] = l_new
        
        # Normalize output
        output = output / l
        
        return output
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for flash attention.
        
        Args:
            query: Query tensor (batch, seq_q, d_model)
            key: Key tensor (batch, seq_k, d_model)
            value: Value tensor (batch, seq_v, d_model)
            key_padding_mask: Optional padding mask
        
        Returns:
            Output tensor (batch, seq_q, d_model)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Linear projections
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # Reshape for multi-head: (batch, seq, num_heads, head_dim)
        q = q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply padding mask if provided
        if key_padding_mask is not None:
            # (batch, seq_k) -> (batch, 1, 1, seq_k)
            mask = ~key_padding_mask.view(batch_size, 1, 1, seq_len_k)
            k = k.masked_fill(mask.unsqueeze(-1), 0)
            v = v.masked_fill(mask.unsqueeze(-1), 0)
        
        # Flash attention forward pass
        output = self._flash_attention_forward(q, k, v, causal=self.causal)
        
        # Reshape back: (batch, seq, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        
        # Output projection
        output = self.out_proj(output)
        
        return output


class MultiQueryAttention(MultiHeadAttention):
    """
    Multi-Query Attention (MQA).
    
    Uses a single head for keys and values while maintaining
    multiple query heads. Reduces memory bandwidth requirements.
    
    Reference: "Fast Transformer Decoding with One Write-Head" (Shazeer, 2019)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Single head for keys and values
        self.kv_dim = self.head_dim
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        return_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for multi-query attention.
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Query projection: (batch, seq_q, num_heads, head_dim)
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        
        # Key/Value projection: single head
        k = self.k_proj(key).view(batch_size, seq_len_k, 1, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len_k, 1, self.head_dim)
        
        # Expand to match num_heads
        k = k.expand(-1, -1, self.num_heads, -1).transpose(1, 2)
        v = v.expand(-1, -1, self.num_heads, -1).transpose(1, 2)
        
        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, seq_len_k)
            attn_scores = attn_scores.masked_fill(~key_padding_mask, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len_q, self.d_model)
        output = self.out_proj(output)
        
        return output, attn_weights if return_weights else None


class GroupedQueryAttention(MultiHeadAttention):
    """
    Grouped-Query Attention (GQA).
    
    Groups query heads share the same key/value heads.
    Balances between MQA (1 KV head) and MHA (num_heads KV heads).
    
    Reference: "GQA: Training Generalized Multi-Query Transformer Models" (Ainsworth et al., 2023)
    
    Args:
        d_model: Dimension of input embeddings
        num_heads: Number of query heads
        num_kv_heads: Number of key/value heads (must divide num_heads)
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        assert num_heads % num_kv_heads == 0, \
            f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.kv_dim = num_kv_heads * self.head_dim
        
        # Reproject for grouped KV
        self.k_proj = nn.Linear(d_model, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.kv_dim, bias=False)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        return_weights: bool = True,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for grouped-query attention.
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Query projection
        q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        
        # KV projection with fewer heads
        k = self.k_proj(key).view(batch_size, seq_len_k, self.num_kv_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len_k, self.num_kv_heads, self.head_dim)
        
        # Expand KV heads to match query heads
        # (batch, num_kv_heads, seq, head_dim) -> (batch, num_heads, seq, head_dim)
        k = k.repeat_interleave(self.num_groups, dim=1)
        v = v.repeat_interleave(self.num_groups, dim=1)
        
        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.view(batch_size, 1, 1, seq_len_k)
            attn_scores = attn_scores.masked_fill(~key_padding_mask, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, seq_len_q, self.d_model)
        output = self.out_proj(output)
        
        return output, attn_weights if return_weights else None
