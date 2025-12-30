"""
Attention Mechanisms
====================
Self-attention and Transformer components from scratch.

The Transformer architecture (Vaswani et al., 2017) revolutionized NLP
with its self-attention mechanism, enabling parallel processing and
capturing long-range dependencies.

Key Equation:
    Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V

Components:
- Scaled Dot-Product Attention
- Multi-Head Attention
- Positional Encodings (Sinusoidal, RoPE)
- TransformerBlock

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Optional, Tuple

# Try relative import
try:
    from ..core.math_operations import softmax
except ImportError:
    def softmax(x, axis=-1):
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# ============================================================
# POSITIONAL ENCODINGS
# ============================================================

def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Sinusoidal Positional Encoding (original Transformer).
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Properties:
    - Deterministic (no learned parameters)
    - Generalizes to longer sequences than seen in training
    - Relative positions can be expressed as linear functions
    
    Args:
        seq_len: Maximum sequence length
        d_model: Model dimension
    
    Returns:
        Positional encoding matrix (seq_len, d_model)
    """
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return pe


def rotary_positional_embedding(x: np.ndarray, seq_len: int, 
                                 d_model: int, base: int = 10000) -> np.ndarray:
    """
    Rotary Position Embedding (RoPE).
    
    Used in Llama, GPT-Neo-X, and modern LLMs.
    
    Advantages over sinusoidal:
    - Relative position is encoded in dot product
    - Better generalization to longer sequences
    - Can be applied at each layer
    
    Formula:
        RoPE(x, pos) = x * cos(mθ) + rotate(x) * sin(mθ)
    
    Where rotate(x) swaps pairs and negates alternates.
    
    Args:
        x: Input tensor (batch, seq_len, d_model)
        seq_len: Sequence length
        d_model: Model dimension
        base: Base for frequency calculation
    
    Returns:
        Position-encoded tensor
    """
    # Generate rotation angles
    inv_freq = 1.0 / (base ** (np.arange(0, d_model, 2) / d_model))
    positions = np.arange(seq_len)
    angles = np.outer(positions, inv_freq)
    
    # Create rotation matrix components
    cos = np.cos(angles)
    sin = np.sin(angles)
    
    # Apply rotation
    x1, x2 = x[..., 0::2], x[..., 1::2]
    x_rotated = np.stack([-x2, x1], axis=-1).reshape(x.shape)
    
    cos_expanded = np.repeat(cos, 2, axis=-1)
    sin_expanded = np.repeat(sin, 2, axis=-1)
    
    return x * cos_expanded + x_rotated * sin_expanded


# ============================================================
# ATTENTION MECHANISMS
# ============================================================

def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray, 
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
    dropout_rate: float = 0.0,
    training: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention.
    
    The core attention mechanism from "Attention Is All You Need".
    
    Formula:
        Attention(Q, K, V) = softmax(QKᵀ / √d_k) × V
    
    Intuition:
        - Q (Query): "What am I looking for?"
        - K (Key): "What do I contain?"
        - V (Value): "What can I provide?"
        - Attention scores = compatibility between Q and K
        - Output = weighted sum of V based on scores
    
    Args:
        Q: Queries (batch, seq_q, d_k)
        K: Keys (batch, seq_k, d_k)
        V: Values (batch, seq_k, d_v)
        mask: Optional attention mask (broadcastable)
        dropout_rate: Dropout probability on attention weights
        training: Whether in training mode
    
    Returns:
        Tuple of (output, attention_weights)
    
    Example:
        >>> Q = np.random.randn(2, 10, 64)  # batch=2, seq=10, d_k=64
        >>> K = np.random.randn(2, 20, 64)
        >>> V = np.random.randn(2, 20, 64)
        >>> output, weights = scaled_dot_product_attention(Q, K, V)
        >>> print(output.shape)  # (2, 10, 64)
    """
    d_k = K.shape[-1]
    
    # Compute attention scores: Q @ K^T
    # (batch, seq_q, d_k) @ (batch, d_k, seq_k) -> (batch, seq_q, seq_k)
    scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(d_k)
    
    # Apply mask (for causal attention or padding)
    if mask is not None:
        scores = np.where(mask, scores, -1e9)
    
    # Softmax over keys
    attention_weights = softmax(scores, axis=-1)
    
    # Apply dropout during training
    if training and dropout_rate > 0:
        dropout_mask = np.random.binomial(1, 1 - dropout_rate, attention_weights.shape)
        attention_weights = attention_weights * dropout_mask / (1 - dropout_rate)
    
    # Weighted sum of values
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights


class MultiHeadAttention:
    """
    Multi-Head Attention.
    
    Allows the model to attend to information from different
    representation subspaces at different positions.
    
    Formula:
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O
        where head_i = Attention(Q @ W_Q^i, K @ W_K^i, V @ W_V^i)
    
    Parameters:
        - d_model: 512 (typical)
        - num_heads: 8
        - d_k = d_v = d_model / num_heads = 64
    
    Why multiple heads?
        - Different heads can focus on different aspects
        - Head 1: syntax (subject-verb)
        - Head 2: coreference (pronouns)
        - Head 3: semantic similarity
    
    Example:
        >>> mha = MultiHeadAttention(d_model=512, num_heads=8)
        >>> output = mha.forward(x, x, x)  # Self-attention
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.0):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout_rate: Dropout probability
        """
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout_rate = dropout_rate
        
        # Initialize projection weights
        scale = np.sqrt(2.0 / d_model)
        self.W_Q = np.random.randn(d_model, d_model) * scale
        self.W_K = np.random.randn(d_model, d_model) * scale
        self.W_V = np.random.randn(d_model, d_model) * scale
        self.W_O = np.random.randn(d_model, d_model) * scale
        
        # For backward pass
        self.Q = None
        self.K = None
        self.V = None
        self.attention_weights = None
    
    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split last dimension into (num_heads, d_k).
        
        (batch, seq, d_model) -> (batch, num_heads, seq, d_k)
        """
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)
    
    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Combine heads back to original shape.
        
        (batch, num_heads, seq, d_k) -> (batch, seq, d_model)
        """
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch_size, seq_len, self.d_model)
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                mask: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """
        Multi-head attention forward pass.
        
        Args:
            Q: Queries (batch, seq_q, d_model)
            K: Keys (batch, seq_k, d_model)
            V: Values (batch, seq_k, d_model)
            mask: Optional attention mask
            training: Training mode
        
        Returns:
            Output tensor (batch, seq_q, d_model)
        """
        # Linear projections
        Q_proj = Q @ self.W_Q
        K_proj = K @ self.W_K
        V_proj = V @ self.W_V
        
        # Split into heads
        Q_heads = self.split_heads(Q_proj)
        K_heads = self.split_heads(K_proj)
        V_heads = self.split_heads(V_proj)
        
        # Apply attention for each head
        # Reshape for batch processing
        batch_size = Q.shape[0]
        Q_flat = Q_heads.reshape(-1, Q_heads.shape[2], self.d_k)
        K_flat = K_heads.reshape(-1, K_heads.shape[2], self.d_k)
        V_flat = V_heads.reshape(-1, V_heads.shape[2], self.d_k)
        
        attention_output, self.attention_weights = scaled_dot_product_attention(
            Q_flat, K_flat, V_flat,
            mask=mask, dropout_rate=self.dropout_rate, training=training
        )
        
        # Reshape back
        attention_output = attention_output.reshape(
            batch_size, self.num_heads, -1, self.d_k
        )
        
        # Combine heads
        concat_output = self.combine_heads(attention_output)
        
        # Final linear projection
        output = concat_output @ self.W_O
        
        return output
    
    def __call__(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                 mask: Optional[np.ndarray] = None,
                 training: bool = True) -> np.ndarray:
        return self.forward(Q, K, V, mask, training)


# ============================================================
# FEED-FORWARD NETWORK
# ============================================================

class FeedForwardNetwork:
    """
    Position-wise Feed-Forward Network.
    
    Applied to each position independently and identically.
    
    Formula:
        FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
    
    Or with GELU (GPT, BERT):
        FFN(x) = GELU(xW_1 + b_1)W_2 + b_2
    
    Typical dimensions:
        - d_model: 512
        - d_ff: 2048 (4× expansion)
    
    Why 4×?
        - Larger capacity for non-linear transformations
        - Empirically found to work well
    """
    
    def __init__(self, d_model: int, d_ff: int, activation: str = 'relu',
                 dropout_rate: float = 0.0):
        """
        Args:
            d_model: Model dimension
            d_ff: Feed-forward dimension (usually 4 * d_model)
            activation: 'relu' or 'gelu'
            dropout_rate: Dropout probability
        """
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.dropout_rate = dropout_rate
        
        # Initialize weights
        scale = np.sqrt(2.0 / d_model)
        self.W1 = np.random.randn(d_model, d_ff) * scale
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * scale
        self.b2 = np.zeros(d_model)
    
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'gelu':
            # GELU approximation
            return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
        else:
            return x
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass.
        
        Args:
            x: Input (batch, seq, d_model)
            training: Training mode
        
        Returns:
            Output (batch, seq, d_model)
        """
        # First linear + activation
        hidden = self._activate(x @ self.W1 + self.b1)
        
        # Dropout
        if training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, hidden.shape)
            hidden = hidden * mask / (1 - self.dropout_rate)
        
        # Second linear
        output = hidden @ self.W2 + self.b2
        
        return output
    
    def __call__(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        return self.forward(x, training)


# ============================================================
# LAYER NORMALIZATION
# ============================================================

class LayerNorm:
    """
    Layer Normalization.
    
    Normalizes across the feature dimension (d_model).
    
    Formula:
        y = γ × (x - μ) / σ + β
    
    Where μ, σ are computed over the last dimension.
    
    Key difference from BatchNorm:
        - LayerNorm: normalize per sample, across features
        - BatchNorm: normalize per feature, across batch
    
    Why LayerNorm for Transformers?
        - Batch-independent (works with variable sequence length)
        - More stable for RNNs and attention
    """
    
    def __init__(self, d_model: int, epsilon: float = 1e-6):
        """
        Args:
            d_model: Model dimension
            epsilon: Numerical stability
        """
        self.d_model = d_model
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Apply layer normalization.
        
        Args:
            x: Input (batch, seq, d_model)
        
        Returns:
            Normalized output
        """
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        x_normalized = (x - mean) / np.sqrt(var + self.epsilon)
        
        return self.gamma * x_normalized + self.beta
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)


# ============================================================
# TRANSFORMER BLOCK
# ============================================================

class TransformerBlock:
    """
    Single Transformer Encoder Block.
    
    Architecture:
        x → LayerNorm → MultiHeadAttention → Add → LayerNorm → FFN → Add → output
    
    Uses Pre-LayerNorm (like GPT-2, modern models) for stability.
    
    Components:
        1. Multi-Head Self-Attention
        2. Add & Norm (residual connection)
        3. Feed-Forward Network
        4. Add & Norm (residual connection)
    
    Example:
        >>> block = TransformerBlock(d_model=512, num_heads=8)
        >>> output = block.forward(x, mask=causal_mask)
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 dropout_rate: float = 0.1, activation: str = 'gelu',
                 pre_norm: bool = True):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Feed-forward dimension
            dropout_rate: Dropout probability
            activation: FFN activation ('relu', 'gelu')
            pre_norm: Use pre-layer normalization
        """
        self.d_model = d_model
        self.pre_norm = pre_norm
        self.dropout_rate = dropout_rate
        
        # Layers
        self.attention = MultiHeadAttention(d_model, num_heads, dropout_rate)
        self.ffn = FeedForwardNetwork(d_model, d_ff, activation, dropout_rate)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
    
    def forward(self, x: np.ndarray, 
                mask: Optional[np.ndarray] = None,
                training: bool = True) -> np.ndarray:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input (batch, seq, d_model)
            mask: Attention mask
            training: Training mode
        
        Returns:
            Output (batch, seq, d_model)
        """
        if self.pre_norm:
            # Pre-LayerNorm architecture
            # 1. Self-attention with residual
            norm_x = self.norm1(x)
            attn_output = self.attention(norm_x, norm_x, norm_x, mask, training)
            x = x + self._dropout(attn_output, training)
            
            # 2. FFN with residual
            norm_x = self.norm2(x)
            ffn_output = self.ffn(norm_x, training)
            x = x + self._dropout(ffn_output, training)
        else:
            # Post-LayerNorm architecture (original Transformer)
            attn_output = self.attention(x, x, x, mask, training)
            x = self.norm1(x + self._dropout(attn_output, training))
            
            ffn_output = self.ffn(x, training)
            x = self.norm2(x + self._dropout(ffn_output, training))
        
        return x
    
    def _dropout(self, x: np.ndarray, training: bool) -> np.ndarray:
        """Apply dropout during training."""
        if training and self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, x.shape)
            return x * mask / (1 - self.dropout_rate)
        return x
    
    def __call__(self, x: np.ndarray, 
                 mask: Optional[np.ndarray] = None,
                 training: bool = True) -> np.ndarray:
        return self.forward(x, mask, training)


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Create causal (autoregressive) attention mask.
    
    Prevents attending to future positions.
    
    Mask:
        [[1, 0, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 0],
         [1, 1, 1, 1]]
    
    Args:
        seq_len: Sequence length
    
    Returns:
        Causal mask (seq_len, seq_len)
    """
    return np.tril(np.ones((seq_len, seq_len), dtype=bool))


def create_padding_mask(lengths: np.ndarray, max_len: int) -> np.ndarray:
    """
    Create padding mask from sequence lengths.
    
    Args:
        lengths: Array of actual sequence lengths
        max_len: Maximum sequence length
    
    Returns:
        Padding mask (batch, max_len)
    """
    batch_size = len(lengths)
    mask = np.zeros((batch_size, max_len), dtype=bool)
    
    for i, length in enumerate(lengths):
        mask[i, :length] = True
    
    return mask


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Positional Encodings
    'sinusoidal_positional_encoding', 'rotary_positional_embedding',
    # Attention
    'scaled_dot_product_attention', 'MultiHeadAttention',
    # Components
    'FeedForwardNetwork', 'LayerNorm', 'TransformerBlock',
    # Utilities
    'create_causal_mask', 'create_padding_mask',
]
