"""
Positional Encoding and Attention Mechanism
=============================================

This module provides implementations of key components in modern NLP models,
especially Transformers.

Mathematical Foundation:
------------------------

1. Positional Encoding:

   Transformers don't have recurrence or convolution, so they need to
   inject position information explicitly.

   Two versions:

   a) Sinusoidal:
      PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
      PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

      where:
      - pos: position (0 to sequence_length-1)
      - i: dimension index (0 to d_model-1)

   b) Learned:
      Learn position embeddings during training

   Properties:
   - Each dimension corresponds to a different frequency
   - Can represent any position up to max sequence length
   - Can generalize to longer sequences

2. Scaled Dot-Product Attention:

   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

   Where:
   - Q: Query matrix (batch, heads, seq_len, d_k)
   - K: Key matrix (batch, heads, seq_len, d_k)
   - V: Value matrix (batch, heads, seq_len, d_v)
   - d_k: Dimension of keys

   The scaling factor sqrt(d_k) prevents vanishing gradients
   when d_k is large.

3. Multi-Head Attention:

   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

   where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)

   Each head can attend to different positions and learn
   different types of relationships.

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Tuple


class PositionalEncoding:
    """
    Positional Encoding for Transformer Models

    Adds position information to input embeddings using sine and cosine
    functions of different frequencies.

    Properties:
    - Allows model to attend to relative positions
    - Can extrapolate to longer sequences than seen in training
    - No learned parameters (fixed function)

    Formula:
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model: Dimension of the model (embedding size)
            max_len: Maximum sequence length to support
        """
        self.d_model = d_model
        self.max_len = max_len

        # Precompute positional encodings
        self.pe = self._create_positional_encoding(max_len)

    def _create_positional_encoding(self, max_len: int) -> np.ndarray:
        """
        Create positional encoding matrix

        Shape: (max_len, d_model)
        """
        # Create position indices
        position = np.arange(max_len).reshape(-1, 1)  # (max_len, 1)

        # Create dimension indices
        dimension = np.arange(self.d_model).reshape(1, -1)  # (1, d_model)

        # Compute the divisor for each dimension
        # 10000^(2i/d_model) where i is dimension index
        # This creates a geometric progression of frequencies
        div_term = np.exp(
            np.arange(0, self.d_model, 2) * (-np.log(10000.0) / self.d_model)
        )

        # Initialize encoding matrix
        pe = np.zeros((max_len, self.d_model))

        # Apply sin to even dimensions
        pe[:, 0::2] = np.sin(position * div_term)

        # Apply cos to odd dimensions
        pe[:, 1::2] = np.cos(position * div_term)

        return pe

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Add positional encoding to input embeddings

        Args:
            x: Input embeddings, shape (batch, seq_len, d_model)

        Returns:
            Input with positional encoding added
        """
        # Get sequence length
        seq_len = x.shape[1]

        # Add positional encoding
        return x + self.pe[:seq_len, :]

    def get_encoding(self, position: int) -> np.ndarray:
        """Get encoding for a specific position"""
        return self.pe[position]


class ScaledDotProductAttention:
    """
    Scaled Dot-Product Attention

    The core attention mechanism used in Transformers.

    Formula:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

    Mathematical intuition:
    - QK^T computes similarity between queries and keys
    - Dividing by sqrt(d_k) prevents very small gradients
    - Softmax normalizes to probability distribution
    - V is weighted by attention weights

    Complexity:
    - O(n^2 * d) where n is sequence length, d is dimension
    - This is the main computational bottleneck in Transformers
    """

    def __init__(self, dropout: float = 0.1):
        self.dropout = dropout

    def forward(
        self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass

        Args:
            Q: Query, shape (batch, seq_len, d_k)
            K: Key, shape (batch, seq_len, d_k)
            V: Value, shape (batch, seq_len, d_v)
            mask: Optional attention mask

        Returns:
            Tuple of (attention output, attention weights)
        """
        d_k = Q.shape[-1]

        # Compute QK^T / sqrt(d_k)
        scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)

        # Apply mask if provided
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)

        # Softmax
        attention_weights = self._softmax(scores)

        # Apply dropout (simplified)
        # if self.dropout > 0:
        #     attention_weights = ...

        # Apply to values
        output = np.matmul(attention_weights, V)

        return output, attention_weights

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax"""
        x_stable = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x_stable)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class MultiHeadAttention:
    """
    Multi-Head Attention

    Allows the model to attend to information from different
    representation subspaces at different positions.

    Architecture:
        1. Linear projections of Q, K, V for each head
        2. Apply attention in parallel for each head
        3. Concatenate outputs
        4. Final linear projection

    Advantages:
        - Model can focus on different types of information
        - More expressive than single-head attention
        - Each head can learn different patterns
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections
        # Q, K, V projections (each creates num_heads different projections)
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)

        # Output projection
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)

        # Scaled dot-product attention
        self.attention = ScaledDotProductAttention(dropout)

        # Cache for backward pass
        self.cache = None

    def forward(
        self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass

        Args:
            Q: Query, shape (batch, seq_len, d_model)
            K: Key, shape (batch, seq_len, d_model)
            V: Value, shape (batch, seq_len, d_model)
            mask: Optional mask

        Returns:
            Tuple of (output, attention weights)
        """
        batch_size = Q.shape[0]

        # Linear projections and reshape for multiple heads
        # (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        Q = self._linear_and_split(Q, self.W_q, batch_size)
        K = self._linear_and_split(K, self.W_k, batch_size)
        V = self._linear_and_split(V, self.W_v, batch_size)

        # Apply scaled dot-product attention for each head
        head_outputs = []
        head_weights = []

        for h in range(self.num_heads):
            q_h = Q[:, h, :, :]
            k_h = K[:, h, :, :]
            v_h = V[:, h, :, :]

            out_h, attn_h = self.attention.forward(q_h, k_h, v_h, mask)
            head_outputs.append(out_h)
            head_weights.append(attn_h)

        # Concatenate heads: (batch, num_heads, seq_len, d_k) -> (batch, seq_len, d_model)
        concat_output = np.concatenate(head_outputs, axis=-1)

        # Final linear projection
        output = np.matmul(concat_output, self.W_o.T)

        # Concatenate attention weights
        attention_weights = np.stack(head_weights, axis=1)

        # Cache for backward
        self.cache = (Q, K, V, attention_weights)

        return output, attention_weights

    def _linear_and_split(
        self, x: np.ndarray, W: np.ndarray, batch_size: int
    ) -> np.ndarray:
        """Linear projection and split into heads"""
        # Linear
        x_proj = np.matmul(x, W.T)

        # Reshape: (batch, seq_len, d_model) -> (batch, num_heads, seq_len, d_k)
        seq_len = x.shape[1]

        # First reshape to (batch, seq_len, num_heads, d_k)
        x_reshaped = x_proj.reshape(batch_size, seq_len, self.num_heads, self.d_k)

        # Then transpose to (batch, num_heads, seq_len, d_k)
        return x_reshaped.transpose(0, 2, 1, 3)


class SelfAttention:
    """
    Self-Attention Layer

    A complete self-attention layer with:
    - Multi-head attention
    - Residual connection and layer normalization
    - Feed-forward network

    This is the core building block of Transformer encoders and decoders.
    """

    def __init__(
        self, d_model: int, num_heads: int, d_ff: int = 2048, dropout: float = 0.1
    ):
        self.d_model = d_model

        # Multi-head attention
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # Layer normalization (simplified)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        # Feed-forward network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Forward pass with residual connections and layer normalization

        Architecture:
            x -> MultiHeadAttention -> Add & Norm -> FFN -> Add & Norm
        """
        # Multi-head attention with residual
        attn_output, _ = self.attention.forward(x, x, x, mask)
        x = self.norm1.forward(x + attn_output)

        # Feed-forward with residual
        ffn_output = self.ffn.forward(x)
        x = self.norm2.forward(x + ffn_output)

        return x


class LayerNorm:
    """
    Layer Normalization

    Normalizes across features (not batch).

    Formula:
        y = (x - μ) / σ * γ + β

    where:
        μ = (1/n) Σ x_i (mean across features)
        σ = sqrt((1/n) Σ (x_i - μ)^2) (std across features)
        γ: learned scale parameter
        β: learned shift parameter

    Used in:
        - Transformers (after each sub-layer)
        - RNNs (sometimes)
        - GANs (often)
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        self.d_model = d_model
        self.eps = eps

        # Learnable parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Args:
            x: Input, shape (batch, seq_len, d_model)

        Returns:
            Normalized output
        """
        # Compute mean and std across last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True) + self.eps

        # Normalize
        x_norm = (x - mean) / std

        # Scale and shift
        return self.gamma * x_norm + self.beta


class FeedForwardNetwork:
    """
    Position-wise Feed-Forward Network

    Two linear transformations with ReLU activation.

    FFN(x) = max(0, xW_1 + b_1)W_2 + b_2

    Properties:
        - Same weights applied to each position
        - Increases model capacity
        - Typically d_ff = 4 * d_model
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        self.d_model = d_model
        self.d_ff = d_ff

        # First linear layer (expand)
        self.W_1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b_1 = np.zeros(d_ff)

        # Second linear layer (compress)
        self.W_2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b_2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass

        Args:
            x: Input, shape (batch, seq_len, d_model)

        Returns:
            Output, shape (batch, seq_len, d_model)
        """
        # First linear + ReLU
        hidden = np.maximum(0, np.matmul(x, self.W_1.T) + self.b_1)

        # Second linear
        output = np.matmul(hidden, self.W_2.T) + self.b_2

        return output


def test_attention():
    """Test attention mechanisms"""
    print("=" * 60)
    print("Testing Attention Mechanisms")
    print("=" * 60)

    # Test parameters
    batch_size = 2
    seq_len = 4
    d_model = 8
    num_heads = 2

    # Create inputs
    np.random.seed(42)
    x = np.random.randn(batch_size, seq_len, d_model)

    print(f"\nInput shape: {x.shape}")

    # Test positional encoding
    print("\n--- Testing Positional Encoding ---")
    pos_encoding = PositionalEncoding(d_model=d_model, max_len=100)
    x_with_pos = pos_encoding.forward(x)
    print(f"Output shape: {x_with_pos.shape}")
    print(f"Position 0 encoding: {pos_encoding.get_encoding(0)[:4]}")

    # Test scaled dot-product attention
    print("\n--- Testing Scaled Dot-Product Attention ---")
    attention = ScaledDotProductAttention()
    Q = np.random.randn(batch_size, seq_len, 4)
    K = np.random.randn(batch_size, seq_len, 4)
    V = np.random.randn(batch_size, seq_len, 4)

    output, weights = attention.forward(Q, K, V)
    print(f"Attention output shape: {output.shape}")
    print(f"Attention weights shape: {weights.shape}")

    # Test multi-head attention
    print("\n--- Testing Multi-Head Attention ---")
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output, weights = mha.forward(x, x, x)
    print(f"MHA output shape: {output.shape}")
    print(f"MHA weights shape: {weights.shape}")

    # Test self-attention layer
    print("\n--- Testing Self-Attention Layer ---")
    self_attn = SelfAttention(d_model=d_model, num_heads=num_heads, d_ff=16)
    output = self_attn.forward(x)
    print(f"Self-attention output shape: {output.shape}")

    print("\n" + "=" * 60)
    print("All attention tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_attention()
