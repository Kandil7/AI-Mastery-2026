"""
Advanced Attention Mechanisms Implementation for Large Language Models.
Focuses on efficiency, customization, and deep mathematical understanding.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class AttentionConfig:
    """Configuration for Attention parameters"""
    hidden_size: int
    num_attention_heads: int
    head_dim: int
    max_position_embeddings: int = 4096
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    use_flash_attention: bool = False
    rotary_embedding_base: int = 10000
    rotary_embedding_fraction: float = 1.0  # Fraction of dimension to apply rotary embedding
    attention_type: str = "scaled_dot_product"  # "scaled_dot_product", "linear", "performer"
    kv_cache_enabled: bool = True

class RotaryPositionEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.dim = int(config.head_dim * config.rotary_embedding_fraction)
        self.base = config.rotary_embedding_base
        self.max_seq_len = config.max_position_embeddings
        
        # Determine inverse frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Pre-compute cosine and sine cache
        self._set_cos_sin_cache(seq_len=self.max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        """Pre-compute cosine and sine values"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        # Reshape for broadcasting: [1, 1, seq_len, head_dim]
        cos = emb.cos()[None, None, :, :]
        sin = emb.sin()[None, None, :, :]
        
        if hasattr(self, "cos_cached"):
             self.cos_cached = cos
             self.sin_cached = sin
        else:
             self.register_buffer("cos_cached", cos, persistent=False)
             self.register_buffer("sin_cached", sin, persistent=False)
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dimension"""
        x1, x2 = x[..., :self.dim // 2], x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, 
                           position_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply Rotary Position Embeddings to Q and K vectors.
        
        Args:
            q: Query vectors [batch_size, num_heads, seq_len, head_dim]
            k: Key vectors [batch_size, num_heads, seq_len, head_dim]
            position_ids: Position indices (optional)
        
        Returns:
            Tuple of rotated Q and K
        """
        seq_len = q.shape[2]
        
        # Ensure cache is on correct device
        if self.cos_cached.device != q.device:
             self.cos_cached = self.cos_cached.to(q.device)
             self.sin_cached = self.sin_cached.to(q.device)

        if position_ids is not None:
            # Select specific positions
            # Assuming position_ids shape is [batch, seq_len] -> unsqueeze for broadcasting
            # We need to slice strictly or use gather.
            # Simplified approach:
            cos = self.cos_cached[:, :, position_ids, :]
            sin = self.sin_cached[:, :, position_ids, :]
            # Warning: this simple indexing might need adjustment based on exact position_ids shape
        else:
            if seq_len > self.cos_cached.shape[2]:
                self._set_cos_sin_cache(seq_len)
                self.cos_cached = self.cos_cached.to(q.device)
                self.sin_cached = self.sin_cached.to(q.device)
            
            cos = self.cos_cached[:, :, :seq_len, :]
            sin = self.sin_cached[:, :, :seq_len, :]
        
        # Apply One-sided RoPE or Full? Generally applied to part of dim.
        # Here assuming full dim is rotated if fraction=1.0
        
        # Apply embeddings
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        
        return q_embed, k_embed

class AttentionMechanism(nn.Module):
    """Base Attention Mechanism"""
    
    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        
        # Validation
        assert config.hidden_size % config.num_attention_heads == 0, \
            "Hidden size must be divisible by number of attention heads"
        
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        
        # Projection Layers
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Initialize RoPE
        self.rotary_emb = RotaryPositionEmbedding(config)
        
        # Weight Initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier Uniform"""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> torch.Tensor:
        """Reshape tensor for multi-head attention"""
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for attention mechanism.
        
        Args:
            hidden_states: Input hidden states [batch_size, seq_len, hidden_size]
            attention_mask: Attention mask (optional)
            position_ids: Position indices (optional)
            past_key_value: Cached Key/Value states from previous step (optional)
            output_attentions: Whether to return attention weights
        
        Returns:
            Tuple of (output, attention_weights, past_key_value)
        """
        bsz, q_len, _ = hidden_states.size()
        
        # Project Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape to [batch_size, num_heads, seq_len, head_dim]
        query_states = self._shape(query_states, q_len, bsz)
        key_states = self._shape(key_states, q_len, bsz)
        value_states = self._shape(value_states, q_len, bsz)
        
        # Apply Rotary Position Embeddings
        if position_ids is not None:
            query_states, key_states = self.rotary_emb.apply_rotary_pos_emb(
                query_states, key_states, position_ids
            )
        else:
             # If no position_ids but we are generating seq, assume 0..n
             # For simpler usage let's pass None and let RoPE handle generic seq_len logic
             # (See RoPE implementation detail above)
             query_states, key_states = self.rotary_emb.apply_rotary_pos_emb(
                 query_states, key_states, None
             )
        
        # Handle KV Cache (Append new keys/values to past ones)
        if past_key_value is not None and self.config.kv_cache_enabled:
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)
        
        # Compute Attention
        attn_output, attn_weights = self._compute_attention(
            query_states, key_states, value_states, attention_mask
        )
        
        # Reshape Output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # Update Cache
        past_key_value = (key_states, value_states) if self.config.kv_cache_enabled else None
        
        return attn_output, attn_weights, past_key_value
    
    def _compute_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Dispatch to appropriate attention computation method"""
        
        if self.config.attention_type == "scaled_dot_product":
            return self._scaled_dot_product_attention(q, k, v, attention_mask)
        elif self.config.attention_type == "linear":
            return self._linear_attention(q, k, v, attention_mask)
        elif self.config.attention_type == "performer":
            return self._performer_attention(q, k, v, attention_mask)
        else:
            raise ValueError(f"Unsupported attention type: {self.config.attention_type}")
    
    def _scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Compute Scaled Dot-Product Attention"""
        
        # Scores: [batch_size, num_heads, q_len, k_len]
        attn_weights = torch.matmul(q, k.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        # Apply Mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        
        # Dropout
        attn_weights = F.dropout(attn_weights, p=self.config.attention_dropout, training=self.training)
        
        # Weighted Sum
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output, attn_weights
    
    def _linear_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute Linear Attention.
        Useful for very long sequences where O(n^2) is prohibitive.
        Based on kernel trick (feature map).
        """
        # Feature Map (ELU + 1)
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Denominator
        kv = torch.einsum("bhld,bhle->bhde", k, v)
        
        # Numerator
        z = 1.0 / (torch.einsum("bhld,bhl->bhd", q, k.sum(dim=2)) + 1e-6)
        
        # Output
        attn_output = torch.einsum("bhld,bhde,bhd->bhle", q, kv, z)
        
        return attn_output, None
    
    def _performer_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Performer Attention (FAVOR+).
        Approximates Softmax attention with O(n) complexity using random Fourier features.
        """
        # Random Fourier Features
        m = 64  # Number of random features
        projection = torch.randn(self.head_dim, m).to(q.device) * math.sqrt(2 / m)
        
        # Project Q and K
        q_proj = torch.einsum("bhld, dm -> bhlm", q, projection)
        k_proj = torch.einsum("bhld, dm -> bhlm", k, projection)
        
        # Approximate Softmax components
        q_feat = F.softmax(q_proj, dim=-1)
        k_feat = F.softmax(k_proj, dim=-1)
        
        # Compute Output
        kv = torch.einsum("bhld,bhle->bhde", k_feat, v)
        z = torch.einsum("bhld,bhl->bhd", q_feat, k_feat.sum(dim=2))
        
        attn_output = torch.einsum("bhld,bhde,bhd->bhle", q_feat, kv, 1.0 / (z + 1e-6))
        
        return attn_output, None

class MultiQueryAttention(AttentionMechanism):
    """
    Multi-Query Attention (MQA).
    Uses multiple heads for Query, but shares a single Key and Value head across all queries.
    Significantly reduces memory bandwidth and cache usage.
    """
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        
        # In MQA, k_proj and v_proj output only 1 head dim
        self.k_proj = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.head_dim, bias=False)
        
        self._init_weights()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for MQA"""
        
        bsz, q_len, _ = hidden_states.size()
        
        # Project Q (Multiple Heads), K (1 Head), V (1 Head)
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape Q: [batch, num_heads, seq_len, head_dim]
        query_states = self._shape(query_states, q_len, bsz)
        
        # Reshape K, V: [batch, 1, seq_len, head_dim]
        key_states = key_states.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, 1, self.head_dim).transpose(1, 2)
        
        # Expand K, V to match Num Heads (Broadcasting)
        key_states_expanded = key_states.expand(-1, self.num_heads, -1, -1)
        value_states_expanded = value_states.expand(-1, self.num_heads, -1, -1)
        
        # Apply RoPE
        if position_ids is not None:
             # Apply RoPE to Query (all heads) and Key (expanded heads)
            query_states, key_states_expanded = self.rotary_emb.apply_rotary_pos_emb(
                query_states, key_states_expanded, position_ids
            )
        else:
            query_states, key_states_expanded = self.rotary_emb.apply_rotary_pos_emb(
                query_states, key_states_expanded, None
            )
        
        # Handle Cache
        # Note: We should cache the COMPRESSED (1-head) K/V for efficiency, 
        # but for simplicity here we cache what we use. Real MQA implementations optimize this.
        if past_key_value is not None and self.config.kv_cache_enabled:
             # Assumes past_key_value is also expanded. 
            key_states_expanded = torch.cat([past_key_value[0], key_states_expanded], dim=2)
            value_states_expanded = torch.cat([past_key_value[1], value_states_expanded], dim=2)
        
        # Compute Attention
        attn_output, attn_weights = self._compute_attention(
            query_states, key_states_expanded, value_states_expanded, attention_mask
        )
        
        # Reshape Output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # Update Cache
        past_key_value = (key_states_expanded, value_states_expanded) if self.config.kv_cache_enabled else None
        
        return attn_output, attn_weights, past_key_value

class GroupedQueryAttention(AttentionMechanism):
    """
    Grouped-Query Attention (GQA).
    Groups attention heads into 'g' groups. Each group shares a single K/V head.
    Interpolates between MHA (g=num_heads) and MQA (g=1).
    """
    
    def __init__(self, config: AttentionConfig, num_key_value_heads: int):
        """
        Args:
            config: Attention config
            num_key_value_heads: Number of KV heads. Must divide num_attention_heads.
        """
        super().__init__(config)
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // num_key_value_heads
        
        # Output dim for K/V proj matches num_kv_heads * head_dim
        self.k_proj = nn.Linear(config.hidden_size, self.head_dim * num_key_value_heads, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.head_dim * num_key_value_heads, bias=False)
        
        self._init_weights()
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for GQA"""
        
        bsz, q_len, _ = hidden_states.size()
        
        # Project
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Reshape Q: [batch, num_heads, seq_len, head_dim]
        query_states = self._shape(query_states, q_len, bsz)
        
        # Reshape K, V: [batch, num_kv_heads, seq_len, head_dim]
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE (Before repeating for correct rotational semantics if needed)
        # Note: RoPE is element-wise rotation on head_dim. Safe to apply before repeat.
        if position_ids is not None:
             # We need to temporarily treat Q as having matching heads or just rotate independently?
             # RoPE method expects [batch, heads, seq, dim]. It works for any 'heads' dim.
            query_states, _ = self.rotary_emb.apply_rotary_pos_emb(query_states, query_states, position_ids)
            key_states, _ = self.rotary_emb.apply_rotary_pos_emb(key_states, key_states, position_ids)
        else:
             query_states, _ = self.rotary_emb.apply_rotary_pos_emb(query_states, query_states, None)
             key_states, _ = self.rotary_emb.apply_rotary_pos_emb(key_states, key_states, None)
        
        # Repeat K/V for each group to match Q heads
        # [batch, kv_heads, seq, dim] -> [batch, kv_heads, groups, seq, dim] -> [batch, num_heads, seq, dim]
        key_states_expanded = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
        value_states_expanded = value_states.repeat_interleave(self.num_key_value_groups, dim=1)
        
        # Handle Cache
        if past_key_value is not None and self.config.kv_cache_enabled:
            key_states_expanded = torch.cat([past_key_value[0], key_states_expanded], dim=2)
            value_states_expanded = torch.cat([past_key_value[1], value_states_expanded], dim=2)
        
        # Compute Attention
        attn_output, attn_weights = self._compute_attention(
            query_states, key_states_expanded, value_states_expanded, attention_mask
        )
        
        # Reshape Output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        # Update Cache
        past_key_value = (key_states_expanded, value_states_expanded) if self.config.kv_cache_enabled else None
        
        return attn_output, attn_weights, past_key_value

class FlashAttentionWrapper(nn.Module):
    """
    Wrapper for FlashAttention to maximize GPU efficiency.
    Requires CUDA 11.4+ and `flash_attn` library.
    """
    
    def __init__(self, attention_mechanism: AttentionMechanism):
        super().__init__()
        self.attention = attention_mechanism
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass using FlashAttention"""
        
        try:
            from flash_attn.flash_attn_interface import flash_attn_func
            
            # Note: This is an illustrative wrapper. 
            # In a real scenario, we need to extract Q, K, V properly from the underlying attention mechanism
            # or refactor the attention mechanism to support a 'get_qkv' method.
            # Using standard implementation fallback if complex setup needed, but attempting basic flow:
            
            bsz, q_len, _ = hidden_states.size()
            
            q = self.attention.q_proj(hidden_states)
            k = self.attention.k_proj(hidden_states)
            v = self.attention.v_proj(hidden_states)
            
            q = self.attention._shape(q, q_len, bsz)
            k = self.attention._shape(k, q_len, bsz)
            v = self.attention._shape(v, q_len, bsz)
            
            # Apply RoPE
            if position_ids is not None:
                q, k = self.attention.rotary_emb.apply_rotary_pos_emb(q, k, position_ids)
            else:
                 q, k = self.attention.rotary_emb.apply_rotary_pos_emb(q, k, None)

            # Flash Attention expects [batch, seq_len, num_heads, head_dim]
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Run FlashAttention
            attn_output = flash_attn_func(
                q, k, v,
                dropout_p=self.attention.config.attention_dropout if self.training else 0.0,
                softmax_scale=1.0 / math.sqrt(self.attention.head_dim),
                causal=True if attention_mask is not None else False
            )
            
            # Reshape Output
            attn_output = attn_output.reshape(bsz, q_len, self.attention.hidden_size)
            attn_output = self.attention.o_proj(attn_output)
            
            return attn_output, None, past_key_value
            
        except ImportError:
            logger.warning("FlashAttention not available/installed, falling back to standard implementation")
            return self.attention(
                hidden_states, attention_mask, position_ids, past_key_value, output_attentions
            )
        except Exception as e:
            logger.warning(f"FlashAttention failed ({e}), falling back to standard implementation")
            return self.attention(
                hidden_states, attention_mask, position_ids, past_key_value, output_attentions
            )

def create_attention_layer(config: AttentionConfig, attention_type: str = "standard") -> nn.Module:
    """
    Factory to create the appropriate attention layer.
    
    Args:
        config: Attention configuration.
        attention_type: Type of attention ("standard", "multi_query", "grouped_query", "flash").
    
    Returns:
        Configured attention module.
    """
    if attention_type == "standard":
        attention = AttentionMechanism(config)
    elif attention_type == "multi_query":
        attention = MultiQueryAttention(config)
    elif attention_type == "grouped_query":
        # Default: Split heads into 4 groups if possible, else 1
        num_key_value_heads = max(1, config.num_attention_heads // 4)
        attention = GroupedQueryAttention(config, num_key_value_heads)
    else:
        # Fallback due to invalid type, user might mean standard
        logger.warning(f"Unknown attention type '{attention_type}', defaulting to standard.")
        attention = AttentionMechanism(config)
    
    # Wrap with FlashAttention if requested and not explicitly 'flash' type (which isn't a separate class here)
    if config.use_flash_attention:
        try:
             # Just verify import possible
            import flash_attn
            return FlashAttentionWrapper(attention)
        except ImportError:
            logger.warning("FlashAttention requested but not installed. Using standard implementation.")
    
    return attention

# ============================================================
# STANDALONE ATTENTION FUNCTIONS
# ============================================================

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scaled Dot-Product Attention mechanism.

    Computes attention weights and outputs using:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Args:
        Q: Query tensor of shape (batch_size, seq_len, d_k)
        K: Key tensor of shape (batch_size, seq_len, d_k)
        V: Value tensor of shape (batch_size, seq_len, d_v)
        mask: Optional mask tensor to prevent attention to certain positions

    Returns:
        Tuple of (attention_output, attention_weights)
    """
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores.masked_fill_(mask == 0, -1e9)

    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


# ============================================================
# TRANSFORMER COMPONENTS
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.

    Concatenates multiple attention heads to capture different aspects of the input.
    Each head learns to attend to different parts of the sequence.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
    """
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of Multi-Head Attention.

        Args:
            Q: Query tensor
            K: Key tensor
            V: Value tensor
            mask: Optional attention mask

        Returns:
            Multi-head attention output
        """
        batch_size = Q.size(0)

        # Linear projections
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        attn_output, _ = scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and apply final linear transformation
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        output = self.W_o(attn_output)

        return output


class FeedForwardNetwork(nn.Module):
    """
    Feed-Forward Network component of Transformer.

    Consists of two linear transformations with a ReLU activation in between.
    Applied to each position separately and identically.

    Args:
        d_model: Model dimension
        d_ff: Hidden layer dimension
        activation: Activation function to use
    """
    def __init__(self, d_model: int, d_ff: int, activation: str = 'relu'):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Feed-Forward Network.

        Args:
            x: Input tensor

        Returns:
            FFN output
        """
        return self.linear2(self.activation(self.linear1(x)))


class LayerNorm(nn.Module):
    """
    Layer Normalization.

    Normalizes across the feature dimension to stabilize training.
    Unlike BatchNorm, it normalizes per-sample rather than per-batch.

    Args:
        d_model: Model dimension
        eps: Epsilon for numerical stability
    """
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Layer Normalization.

        Args:
            x: Input tensor

        Returns:
            Normalized tensor
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / (std + self.eps)
        return self.gamma * normalized + self.beta


class TransformerBlock(nn.Module):
    """
    Single Transformer Block (Encoder).

    Combines Multi-Head Attention, Feed-Forward Network, and residual connections
    with Layer Normalization.

    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Hidden dimension for FFN
        dropout: Dropout rate
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = FeedForwardNetwork(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Transformer Block.

        Args:
            x: Input tensor

        Returns:
            Transformer block output
        """
        # Multi-head attention with residual connection
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))

        return x


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Attention mechanisms
    'scaled_dot_product_attention', 'MultiHeadAttention',
    # Transformer components
    'TransformerBlock', 'FeedForwardNetwork', 'LayerNorm',
    # Original attention classes
    'AttentionConfig', 'RotaryPositionEmbedding', 'AttentionMechanism',
    'MultiQueryAttention', 'GroupedQueryAttention', 'FlashAttentionWrapper',
    'create_attention_layer'
]

if __name__ == "__main__":
    # Test Driver
    logging.basicConfig(level=logging.INFO)

    config = AttentionConfig(
        hidden_size=256,
        num_attention_heads=8,
        head_dim=32,
        attention_dropout=0.1,
        use_flash_attention=False
    )

    print("Testing Standard Attention...")
    attn = create_attention_layer(config, "standard")
    x = torch.randn(2, 10, 256)
    out, _, _ = attn(x)
    print(f"Output: {out.shape}")

    print("\nTesting Grouped Query Attention...")
    attn_gqa = create_attention_layer(config, "grouped_query")
    out_gqa, _, _ = attn_gqa(x)
    print(f"Output: {out_gqa.shape}")

    # Test the new components
    print("\nTesting Scaled Dot-Product Attention...")
    Q = torch.randn(2, 10, 64)
    K = torch.randn(2, 15, 64)
    V = torch.randn(2, 15, 64)
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    print(f"SDPA Output: {output.shape}, Attention Weights: {attention_weights.shape}")

    print("\nTesting MultiHeadAttention...")
    mha = MultiHeadAttention(d_model=512, num_heads=8)
    x = torch.randn(4, 20, 512)
    output = mha(x, x, x)
    print(f"MHA Output: {output.shape}")

    print("\nTesting TransformerBlock...")
    block = TransformerBlock(d_model=256, num_heads=8, d_ff=512)
    x = torch.randn(2, 15, 256)
    output = block(x)
    print(f"TransformerBlock Output: {output.shape}")
