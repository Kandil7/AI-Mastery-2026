"""
Attention Mechanisms Implementation

This module implements various attention mechanisms used in LLMs,
including scaled dot-product attention, multi-head attention, and variants.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


def scaled_dot_product_attention(
    query: np.ndarray, 
    key: np.ndarray, 
    value: np.ndarray, 
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention mechanism.
    
    Args:
        query: Query tensor of shape (..., seq_len_q, d_k)
        key: Key tensor of shape (..., seq_len_k, d_k)
        value: Value tensor of shape (..., seq_len_v, d_v)
        mask: Optional mask to prevent attention to certain positions
        
    Returns:
        Tuple of (output, attention_weights)
    """
    d_k = query.shape[-1]
    
    # Compute attention scores: (..., seq_len_q, seq_len_k)
    scores = np.matmul(query, np.transpose(key, tuple(range(len(key.shape)-2)) + (-1, -2))) / math.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        # Ensure mask has the right shape
        mask = np.broadcast_to(mask, scores.shape)
        scores = np.where(mask == 0, -1e9, scores)  # Very negative value for masked positions
    
    # Apply softmax to get attention weights
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))  # For numerical stability
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    # Compute output: (..., seq_len_q, d_v)
    output = np.matmul(attention_weights, value)
    
    return output, attention_weights


class DotProductAttention(nn.Module):
    """
    PyTorch implementation of dot-product attention.
    """
    
    def __init__(self, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of dot-product attention.
        
        Args:
            query: Query tensor of shape (..., seq_len_q, d_k)
            key: Key tensor of shape (..., seq_len_k, d_k)
            value: Value tensor of shape (..., seq_len_v, d_v)
            mask: Optional mask to prevent attention to certain positions
            
        Returns:
            Tuple of (output, attention_weights)
        """
        d_k = query.size(-1)
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Compute output
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism implementation.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers for Q, K, V projections
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        
        # Output linear layer
        self.linear_out = nn.Linear(d_model, d_model)
        
        # Attention mechanism
        self.attention = DotProductAttention(dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-head attention.
        
        Args:
            query: Query tensor of shape (batch_size, seq_len_q, d_model)
            key: Key tensor of shape (batch_size, seq_len_k, d_model)
            value: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask to prevent attention to certain positions
            
        Returns:
            Output tensor of shape (batch_size, seq_len_q, d_model)
        """
        batch_size = query.size(0)
        
        # Linear projections
        Q = self.linear_q(query)  # (batch_size, seq_len_q, d_model)
        K = self.linear_k(key)    # (batch_size, seq_len_k, d_model)
        V = self.linear_v(value)  # (batch_size, seq_len_v, d_model)
        
        # Split into heads: (batch_size, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Apply attention to each head
        attention_output, attention_weights = self.attention(Q, K, V, mask)
        
        # Combine heads: (batch_size, seq_len_q, d_model)
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # Final linear projection
        output = self.linear_out(attention_output)
        output = self.dropout(output)
        
        return output


class SelfAttention(nn.Module):
    """
    Self-attention mechanism where Q, K, V are all derived from the same input.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask to prevent attention to certain positions
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.multi_head_attention(x, x, x, mask)


class CausalSelfAttention(nn.Module):
    """
    Causal (masked) self-attention for autoregressive models like GPT.
    Prevents tokens from attending to future positions.
    """
    
    def __init__(self, d_model: int, num_heads: int, block_size: int, dropout: float = 0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through causal self-attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Create causal mask for current sequence length
        causal_mask = self.mask[:, :, :seq_len, :seq_len].expand(batch_size, -1, -1, -1)
        
        return self.multi_head_attention(x, x, x, causal_mask)


class CrossAttention(nn.Module):
    """
    Cross-attention mechanism for encoder-decoder architectures.
    Queries come from decoder, keys and values from encoder.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, dropout)
    
    def forward(
        self, 
        query: torch.Tensor,  # From decoder
        key: torch.Tensor,    # From encoder
        value: torch.Tensor,  # From encoder
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through cross-attention.
        
        Args:
            query: Query tensor from decoder (batch_size, seq_len_q, d_model)
            key: Key tensor from encoder (batch_size, seq_len_k, d_model)
            value: Value tensor from encoder (batch_size, seq_len_v, d_model)
            mask: Optional mask to prevent attention to certain positions
            
        Returns:
            Output tensor of shape (batch_size, seq_len_q, d_model)
        """
        return self.multi_head_attention(query, key, value, mask)


class RelativePositionAttention(nn.Module):
    """
    Relative position attention that considers relative positions between tokens.
    """
    
    def __init__(self, d_model: int, num_heads: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_len = max_len
        
        # Linear layers for Q, K, V projections
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
        # Relative position embeddings
        self.relative_k = nn.Parameter(torch.randn(2 * max_len - 1, self.d_k))
        self.relative_v = nn.Parameter(torch.randn(2 * max_len - 1, self.d_k))
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through relative position attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask to prevent attention to certain positions
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.linear_q(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.linear_k(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.linear_v(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute relative position indices
        rel_pos = torch.arange(seq_len).unsqueeze(0) - torch.arange(seq_len).unsqueeze(1)
        rel_pos = rel_pos + self.max_len - 1  # Shift to positive indices
        
        # Get relative position embeddings
        rel_k = self.relative_k[rel_pos]  # (seq_len, seq_len, d_k)
        rel_v = self.relative_v[rel_pos]  # (seq_len, seq_len, d_k)
        
        # Compute attention scores with relative positions
        # Standard attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position attention for keys
        rel_scores = torch.einsum('bhld,lrd->bhlr', Q, rel_k)
        scores = scores + rel_scores
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Compute output with standard values
        output = torch.matmul(attention_weights, V)
        
        # Add relative position attention for values
        rel_output = torch.einsum('bhlr,lrd->bhld', attention_weights, rel_v)
        output = output + rel_output
        
        # Combine heads and apply final linear projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.linear_out(output)
        output = self.dropout(output)
        
        return output


class FlashAttention(nn.Module):
    """
    Flash Attention implementation - a more efficient attention mechanism.
    This is a simplified version; the actual implementation is more complex.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear layers
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Flash Attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask to prevent attention to certain positions
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.linear_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.linear_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.linear_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Flash attention computation (simplified)
        # In practice, this would use optimized kernels
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        # Combine heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.linear_out(output)
        output = self.dropout(output)
        
        return output


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention - a variant that reduces memory usage.
    """
    
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        if num_heads % num_kv_heads != 0:
            raise ValueError(f"num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.d_k = d_model // num_heads
        self.num_head_groups = num_heads // num_kv_heads
        
        # Linear layers
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.linear_v = nn.Linear(d_model, num_kv_heads * self.d_k)
        self.linear_out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through Grouped Query Attention.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask to prevent attention to certain positions
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.linear_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.linear_k(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        V = self.linear_v(x).view(batch_size, seq_len, self.num_kv_heads, self.d_k).transpose(1, 2)
        
        # Repeat K and V heads to match Q heads
        K = K.repeat_interleave(self.num_head_groups, dim=1)
        V = V.repeat_interleave(self.num_head_groups, dim=1)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        # Combine heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.linear_out(output)
        output = self.dropout(output)
        
        return output


class AttentionWithRoPE(nn.Module):
    """
    Attention with Rotary Position Embedding (RoPE).
    """
    
    def __init__(self, d_model: int, num_heads: int, max_len: int = 1024, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.max_len = max_len
        
        # Linear layers
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        
        # RoPE parameters
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.d_k, 2, dtype=torch.float) / self.d_k))
        self.register_buffer('inv_freq', inv_freq)
        
        self.dropout = nn.Dropout(dropout)
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""
        x1, x2 = x[..., : self.d_k // 2], x[..., self.d_k // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """Apply Rotary Position Embedding to input."""
        # x: (batch_size, seq_len, num_heads, d_k)
        # positions: (batch_size, seq_len)
        
        batch_size, seq_len = positions.shape
        
        # Get sinusoidal positions - use the position indices directly
        pos_float = positions.float()  # (batch_size, seq_len)
        
        # Create frequency tensor: (seq_len, d_k/2)
        freqs = torch.einsum("bs, d -> bsd", pos_float, self.inv_freq.to(pos_float.device))
        # freqs shape: (batch_size, seq_len, d_k/2)
        
        emb = torch.cat((freqs, freqs), dim=-1)  # (batch_size, seq_len, d_k)
        
        # Expand for heads: (batch_size, seq_len, 1, d_k)
        cos_vals = emb.cos().unsqueeze(2)
        sin_vals = emb.sin().unsqueeze(2)
        
        # Apply rotation
        x_rotated = (x * cos_vals) + (self.rotate_half(x) * sin_vals)
        return x_rotated

    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through attention with RoPE.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask to prevent attention to certain positions
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Create position indices
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Linear projections
        Q = self.linear_q(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        K = self.linear_k(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        V = self.linear_v(x).view(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Apply RoPE to Q and K
        Q = self.apply_rope(Q, positions)
        K = self.apply_rope(K, positions)
        
        # Transpose for attention: (batch_size, num_heads, seq_len, d_k)
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        # Combine heads
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.linear_out(output)
        output = self.dropout(output)
        
        return output


def create_attention_mechanism(attention_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create different attention mechanisms.
    
    Args:
        attention_type: Type of attention mechanism
        **kwargs: Arguments for the attention mechanism
        
    Returns:
        Attention mechanism module
    """
    if attention_type == 'multihead':
        return MultiHeadAttention(
            d_model=kwargs.get('d_model', 512),
            num_heads=kwargs.get('num_heads', 8),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif attention_type == 'causal':
        return CausalSelfAttention(
            d_model=kwargs.get('d_model', 512),
            num_heads=kwargs.get('num_heads', 8),
            block_size=kwargs.get('block_size', 1024),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif attention_type == 'relative':
        return RelativePositionAttention(
            d_model=kwargs.get('d_model', 512),
            num_heads=kwargs.get('num_heads', 8),
            max_len=kwargs.get('max_len', 1024),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif attention_type == 'flash':
        return FlashAttention(
            d_model=kwargs.get('d_model', 512),
            num_heads=kwargs.get('num_heads', 8),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif attention_type == 'grouped_query':
        return GroupedQueryAttention(
            d_model=kwargs.get('d_model', 512),
            num_heads=kwargs.get('num_heads', 8),
            num_kv_heads=kwargs.get('num_kv_heads', 2),
            dropout=kwargs.get('dropout', 0.1)
        )
    elif attention_type == 'rope':
        return AttentionWithRoPE(
            d_model=kwargs.get('d_model', 512),
            num_heads=kwargs.get('num_heads', 8),
            max_len=kwargs.get('max_len', 1024),
            dropout=kwargs.get('dropout', 0.1)
        )
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test the attention mechanisms
    batch_size, seq_len, d_model = 2, 10, 512
    num_heads = 8
    
    # Create random input
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test MultiHeadAttention
    multihead_attn = MultiHeadAttention(d_model, num_heads)
    output = multihead_attn(x, x, x)
    print(f"MultiHeadAttention output shape: {output.shape}")
    
    # Test CausalSelfAttention
    causal_attn = CausalSelfAttention(d_model, num_heads, seq_len)
    output = causal_attn(x)
    print(f"CausalSelfAttention output shape: {output.shape}")
    
    # Test RelativePositionAttention
    rel_pos_attn = RelativePositionAttention(d_model, num_heads)
    output = rel_pos_attn(x)
    print(f"RelativePositionAttention output shape: {output.shape}")