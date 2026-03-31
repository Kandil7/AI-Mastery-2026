"""
Transformer Architecture - Module 2.1.2

Complete implementation of Transformer models:
- Positional Encodings (Sinusoidal, Learned, RoPE)
- Layer Normalization variants
- Transformer Encoder/Decoder layers
- Full Transformer model

References:
- "Attention Is All You Need" (Vaswani et al., 2017)
- "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
"""

import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .attention import MultiHeadAttention, MaskedAttention, CrossAttention


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding.
    
    Uses fixed sine and cosine functions of different frequencies
    to encode position information into embeddings.
    
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    Args:
        d_model: Dimension of embeddings
        max_len: Maximum sequence length
        dropout: Dropout probability
        
    Example:
        >>> pos_enc = PositionalEncoding(d_model=512, max_len=5000)
        >>> x = torch.randn(32, 10, 512)  # batch=32, seq=10
        >>> x = pos_enc(x)
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute div_term = 10000^(-2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sin/cos to even/odd positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension
        pe = pe.unsqueeze(0)
        
        # Register as buffer (not a parameter, but part of state_dict)
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
        
        Returns:
            Tensor with positional encoding added
        """
        # x shape: (batch, seq_len, d_model)
        # pe shape: (1, max_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned Positional Encoding.
    
    Position embeddings are learned during training rather than
    using fixed sinusoidal functions. Common in BERT and GPT models.
    
    Args:
        d_model: Dimension of embeddings
        max_len: Maximum sequence length
        dropout: Dropout probability
        
    Example:
        >>> learned_pos = LearnedPositionalEncoding(d_model=512, max_len=512)
        >>> x = torch.randn(32, 10, 512)
        >>> x = learned_pos(x)
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.max_len = max_len
        
        # Initialize with small values
        nn.init.normal_(self.position_embeddings.weight, std=0.02)
    
    def forward(self, x: Tensor, position_ids: Optional[Tensor] = None) -> Tensor:
        """
        Add learned positional encoding to input.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            position_ids: Optional position IDs (batch, seq_len)
        
        Returns:
            Tensor with positional encoding added
        """
        seq_len = x.size(1)
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device)
            position_ids = position_ids.unsqueeze(0).expand(x.size(0), -1)
        
        position_embeddings = self.position_embeddings(position_ids)
        return self.dropout(x + position_embeddings)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    
    Encodes absolute positional information by rotating the query
    and key vectors. Provides relative position encoding naturally.
    
    RoPE applies rotation matrices to query and key vectors:
    Q' = R(Q, m) and K' = R(K, n)
    where R is a rotation matrix based on position.
    
    Args:
        dim: Dimension of the embedding (must be even)
        max_seq_len: Maximum sequence length
        base: Base for frequency computation (default: 10000)
        
    Example:
        >>> rope = RotaryPositionalEmbedding(dim=64)
        >>> q = torch.randn(32, 8, 10, 64)  # batch, heads, seq, dim
        >>> k = torch.randn(32, 8, 10, 64)
        >>> q_rot, k_rot = rope(q, k)
    """
    
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
    ):
        super().__init__()
        assert dim % 2 == 0, "RoPE dimension must be even"
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Compute frequencies: theta_i = base^(-2i/dim)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for cos/sin values
        self._cos_cache: Optional[Tensor] = None
        self._sin_cache: Optional[Tensor] = None
    
    def _get_rotary_embeddings(
        self,
        seq_len: int,
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        """Get or compute rotary embeddings (cos and sin)."""
        if self._cos_cache is not None and self._cos_cache.size(0) >= seq_len:
            return self._cos_cache[:seq_len], self._sin_cache[:seq_len]
        
        # Compute positions
        t = torch.arange(seq_len, device=device, dtype=torch.float)
        
        # Compute theta: (seq_len, dim/2)
        theta = torch.einsum('i,j->ij', t, self.inv_freq)
        
        # Compute cos and sin
        cos = torch.cos(theta)
        sin = torch.sin(theta)
        
        # Cache for future use
        self._cos_cache = cos
        self._sin_cache = sin
        
        return cos, sin
    
    def _rotate_half(self, x: Tensor) -> Tensor:
        """Rotate half of the hidden dims."""
        # Split into two halves and rotate
        x1, x2 = x[..., : self.dim // 2], x[..., self.dim // 2:]
        return torch.cat((-x2, x1), dim=-1)
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        position_ids: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Apply rotary embeddings to query and key.
        
        Args:
            query: Query tensor (batch, heads, seq, dim)
            key: Key tensor (batch, heads, seq, dim)
            position_ids: Optional position IDs (batch, seq)
        
        Returns:
            Tuple of rotated (query, key)
        """
        seq_len = query.size(2)
        device = query.device
        
        # Get rotary embeddings
        cos, sin = self._get_rotary_embeddings(seq_len, device)
        
        # Handle position_ids for arbitrary positions
        if position_ids is not None:
            cos = cos[position_ids]
            sin = sin[position_ids]
        
        # Add dimension for heads: (seq, 1, dim)
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        
        # Apply rotation
        query_rotated = (query * cos) + (self._rotate_half(query) * sin)
        key_rotated = (key * cos) + (self._rotate_half(key) * sin)
        
        return query_rotated, key_rotated


class LayerNorm(nn.Module):
    """
    Layer Normalization.
    
    Normalizes activations across the feature dimension.
    
    Args:
        normalized_shape: Shape of input to normalize
        eps: Epsilon for numerical stability
        elementwise_affine: Whether to learn scale and bias
        
    Example:
        >>> ln = LayerNorm(512)
        >>> x = torch.randn(32, 10, 512)
        >>> x = ln(x)
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(normalized_shape))
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply layer normalization."""
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(variance + self.eps)
        
        if self.elementwise_affine:
            x = x * self.weight + self.bias
        
        return x


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    Simplified layer normalization that omits mean centering.
    Used in many modern LLMs (LLaMA, PaLM, etc.)
    
    Args:
        normalized_shape: Shape of input to normalize
        eps: Epsilon for numerical stability
        
    Example:
        >>> rms_norm = RMSNorm(512)
        >>> x = torch.randn(32, 10, 512)
        >>> x = rms_norm(x)
    """
    
    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
    
    def forward(self, x: Tensor) -> Tensor:
        """Apply RMS normalization."""
        # Compute RMS
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        
        # Normalize and scale
        x = x / rms * self.weight
        
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer.
    
    Consists of:
    1. Multi-Head Self-Attention + LayerNorm + Residual
    2. Feed-Forward Network + LayerNorm + Residual
    
    Args:
        d_model: Dimension of input embeddings
        num_heads: Number of attention heads
        dim_ff: Dimension of feed-forward network
        dropout: Dropout probability
        norm_layer: Normalization layer class
        activation: Activation function
        
    Example:
        >>> encoder_layer = TransformerEncoderLayer(d_model=512, num_heads=8)
        >>> x = torch.randn(32, 10, 512)
        >>> output = encoder_layer(x)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        norm_layer: nn.Module = LayerNorm,
        activation: nn.Module = nn.ReLU(),
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            activation,
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        
        self.norm1 = norm_layer(d_model)
        self.norm2 = norm_layer(d_model)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for encoder layer.
        
        Args:
            x: Input tensor (batch, seq, d_model)
            mask: Optional attention mask
            key_padding_mask: Optional padding mask
        
        Returns:
            Output tensor (batch, seq, d_model)
        """
        # Self-attention with pre-norm
        attn_input = self.norm1(x)
        attn_output, _ = self.attention(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=key_padding_mask,
            attn_mask=mask,
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward with pre-norm
        ff_input = self.norm2(x)
        ff_output = self.feed_forward(ff_input)
        x = x + ff_output
        
        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer.
    
    Consists of:
    1. Masked Multi-Head Self-Attention + LayerNorm + Residual
    2. Cross-Attention + LayerNorm + Residual
    3. Feed-Forward Network + LayerNorm + Residual
    
    Args:
        d_model: Dimension of input embeddings
        num_heads: Number of attention heads
        dim_ff: Dimension of feed-forward network
        dropout: Dropout probability
        norm_layer: Normalization layer class
        
    Example:
        >>> decoder_layer = TransformerDecoderLayer(d_model=512, num_heads=8)
        >>> x = torch.randn(32, 10, 512)
        >>> memory = torch.randn(32, 15, 512)
        >>> output = decoder_layer(x, memory)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        norm_layer: nn.Module = LayerNorm,
    ):
        super().__init__()
        
        # Masked self-attention
        self.self_attention = MaskedAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Cross-attention
        self.cross_attention = CrossAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )
        
        self.norm1 = norm_layer(d_model)
        self.norm2 = norm_layer(d_model)
        self.norm3 = norm_layer(d_model)
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for decoder layer.
        
        Args:
            x: Decoder input (batch, seq_q, d_model)
            memory: Encoder output (batch, seq_k, d_model)
            memory_mask: Optional mask for cross-attention
            memory_key_padding_mask: Optional padding mask for memory
        
        Returns:
            Output tensor (batch, seq_q, d_model)
        """
        # Masked self-attention
        attn_input = self.norm1(x)
        attn_output, _ = self.self_attention(attn_input)
        x = x + self.dropout(attn_output)
        
        # Cross-attention
        cross_input = self.norm2(x)
        cross_output, _ = self.cross_attention(
            cross_input,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            attn_mask=memory_mask,
        )
        x = x + self.dropout(cross_output)
        
        # Feed-forward
        ff_input = self.norm3(x)
        ff_output = self.feed_forward(ff_input)
        x = x + ff_output
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder (stack of encoder layers).
    
    Args:
        d_model: Dimension of embeddings
        num_heads: Number of attention heads
        num_layers: Number of encoder layers
        dim_ff: Dimension of feed-forward network
        dropout: Dropout probability
        norm_layer: Normalization layer class
        
    Example:
        >>> encoder = TransformerEncoder(d_model=512, num_heads=8, num_layers=6)
        >>> x = torch.randn(32, 10, 512)
        >>> output = encoder(x)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        norm_layer: nn.Module = LayerNorm,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_ff=dim_ff,
                dropout=dropout,
                norm_layer=norm_layer,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = norm_layer(d_model)
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor (batch, seq, d_model)
            mask: Optional attention mask
            key_padding_mask: Optional padding mask
        
        Returns:
            Encoded output (batch, seq, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask=mask, key_padding_mask=key_padding_mask)
        
        return self.norm(x)


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder (stack of decoder layers).
    
    Args:
        d_model: Dimension of embeddings
        num_heads: Number of attention heads
        num_layers: Number of decoder layers
        dim_ff: Dimension of feed-forward network
        dropout: Dropout probability
        norm_layer: Normalization layer class
        
    Example:
        >>> decoder = TransformerDecoder(d_model=512, num_heads=8, num_layers=6)
        >>> x = torch.randn(32, 10, 512)
        >>> memory = torch.randn(32, 15, 512)
        >>> output = decoder(x, memory)
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_ff: int = 2048,
        dropout: float = 0.1,
        norm_layer: nn.Module = LayerNorm,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_ff=dim_ff,
                dropout=dropout,
                norm_layer=norm_layer,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = norm_layer(d_model)
    
    def forward(
        self,
        x: Tensor,
        memory: Tensor,
        memory_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass through decoder.
        
        Args:
            x: Decoder input (batch, seq_q, d_model)
            memory: Encoder output (batch, seq_k, d_model)
            memory_mask: Optional mask for cross-attention
            memory_key_padding_mask: Optional padding mask
        
        Returns:
            Decoded output (batch, seq_q, d_model)
        """
        for layer in self.layers:
            x = layer(
                x,
                memory,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        
        return self.norm(x)


class Transformer(nn.Module):
    """
    Complete Transformer Model (Encoder-Decoder).
    
    Implements the full transformer architecture with:
    - Input/Output embeddings
    - Positional encoding
    - Encoder stack
    - Decoder stack
    - Output projection
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Dimension of embeddings
        num_heads: Number of attention heads
        num_encoder_layers: Number of encoder layers
        num_decoder_layers: Number of decoder layers
        dim_ff: Dimension of feed-forward network
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        pad_idx: Padding token index
        tie_embeddings: Whether to tie input/output embeddings
        
    Example:
        >>> transformer = Transformer(
        ...     vocab_size=30522,
        ...     d_model=512,
        ...     num_heads=8,
        ...     num_encoder_layers=6,
        ...     num_decoder_layers=6,
        ... )
        >>> src = torch.randint(0, 30522, (32, 10))
        >>> tgt = torch.randint(0, 30522, (32, 15))
        >>> output = transformer(src, tgt)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_ff: int = 2048,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pad_idx: int = 0,
        tie_embeddings: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Encoder-Decoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        )
        
        self.decoder = TransformerDecoder(
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_decoder_layers,
            dim_ff=dim_ff,
            dropout=dropout,
        )
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Tie embeddings if specified
        if tie_embeddings:
            self.output_projection.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def _generate_square_subsequent_mask(
        self,
        sz: int,
        device: torch.device,
    ) -> Tensor:
        """Generate causal mask for decoder self-attention."""
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def _generate_padding_mask(
        self,
        tokens: Tensor,
    ) -> Tensor:
        """Generate padding mask from token IDs."""
        return tokens != self.pad_idx
    
    def encode(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Encode source sequence.
        
        Args:
            src: Source tokens (batch, seq_len)
            src_mask: Optional source mask
        
        Returns:
            Encoder output (batch, seq_len, d_model)
        """
        # Embed and add positional encoding
        x = self.token_embedding(src) * math.sqrt(self.d_model)
        x = self.position_encoding(x)
        
        # Generate padding mask if not provided
        if src_mask is None:
            src_mask = self._generate_padding_mask(src)
        
        # Encode
        return self.encoder(x, key_padding_mask=src_mask)
    
    def decode(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Decode target sequence.
        
        Args:
            tgt: Target tokens (batch, seq_len)
            memory: Encoder output (batch, src_len, d_model)
            tgt_mask: Optional target mask
            memory_mask: Optional memory mask
        
        Returns:
            Decoder output (batch, seq_len, d_model)
        """
        # Embed and add positional encoding
        x = self.token_embedding(tgt) * math.sqrt(self.d_model)
        x = self.position_encoding(x)
        
        # Generate causal mask if not provided
        if tgt_mask is None:
            tgt_mask = self._generate_square_subsequent_mask(
                tgt.size(1),
                tgt.device,
            )
        
        # Decode
        return self.decoder(x, memory, memory_mask=memory_mask)
    
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for transformer.
        
        Args:
            src: Source tokens (batch, src_len)
            tgt: Target tokens (batch, tgt_len)
            src_mask: Optional source mask
            tgt_mask: Optional target mask
            memory_mask: Optional memory mask
        
        Returns:
            Logits (batch, tgt_len, vocab_size)
        """
        # Encode
        memory = self.encode(src, src_mask=src_mask)
        
        # Decode
        output = self.decode(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        return logits
    
    def generate(
        self,
        src: Tensor,
        max_len: int = 100,
        bos_idx: int = 1,
        eos_idx: int = 2,
    ) -> Tensor:
        """
        Autoregressive generation.
        
        Args:
            src: Source tokens (batch, src_len)
            max_len: Maximum generation length
            bos_idx: Beginning of sequence token
            eos_idx: End of sequence token
        
        Returns:
            Generated tokens (batch, gen_len)
        """
        batch_size = src.size(0)
        device = src.device
        
        # Start with BOS token
        generated = torch.full(
            (batch_size, 1),
            bos_idx,
            dtype=torch.long,
            device=device,
        )
        
        # Encode source once
        memory = self.encode(src)
        
        # Generate tokens
        for _ in range(max_len - 1):
            # Create causal mask
            tgt_mask = self._generate_square_subsequent_mask(
                generated.size(1),
                device,
            )
            
            # Decode
            output = self.decode(generated, memory, tgt_mask=tgt_mask)
            
            # Get next token logits
            next_token_logits = self.output_projection(output[:, -1, :])
            
            # Greedy decoding
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Append to generated
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if (next_token == eos_idx).all():
                break
        
        return generated


class TransformerLM(nn.Module):
    """
    Transformer Language Model (Decoder-only).
    
    GPT-style decoder-only transformer for language modeling.
    
    Args:
        vocab_size: Size of vocabulary
        d_model: Dimension of embeddings
        num_heads: Number of attention heads
        num_layers: Number of decoder layers
        dim_ff: Dimension of feed-forward network
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        pad_idx: Padding token index
        tie_embeddings: Whether to tie input/output embeddings
        
    Example:
        >>> lm = TransformerLM(vocab_size=50257, d_model=768, num_heads=12, num_layers=12)
        >>> input_ids = torch.randint(0, 50257, (32, 128))
        >>> logits = lm(input_ids)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dim_ff: int = 3072,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 0,
        tie_embeddings: bool = True,
        norm_layer: nn.Module = LayerNorm,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.position_encoding = LearnedPositionalEncoding(d_model, max_seq_len, dropout)
        
        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                dim_ff=dim_ff,
                dropout=dropout,
                norm_layer=norm_layer,
            )
            for _ in range(num_layers)
        ])
        
        self.norm = norm_layer(d_model)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings
        if tie_embeddings:
            self.output_projection.weight = self.token_embedding.weight
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=0.02)
    
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for language model.
        
        Args:
            input_ids: Input tokens (batch, seq_len)
            attention_mask: Optional attention mask
            labels: Optional labels for loss computation
        
        Returns:
            Tuple of (logits, loss)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Embeddings
        x = self.token_embedding(input_ids)
        x = self.position_encoding(x)
        
        # Create causal mask
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device),
            diagonal=1,
        ).bool()
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            # (batch, seq) -> (batch, 1, 1, seq)
            attention_mask = ~attention_mask.view(batch_size, 1, 1, seq_len)
            causal_mask = causal_mask | attention_mask
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, memory=None)
        
        # Normalize
        x = self.norm(x)
        
        # Project to vocabulary
        logits = self.output_projection(x)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift for next-token prediction
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=self.pad_idx,
            )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """
        Autoregressive text generation.
        
        Args:
            input_ids: Input tokens (batch, seq_len)
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            eos_token_id: End of sequence token
        
        Returns:
            Generated tokens (batch, seq_len + new_tokens)
        """
        from .sampling import TopKSampler, TopPSampler, TemperatureSampler
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        generated = input_ids
        
        for _ in range(max_new_tokens):
            # Prepare inputs (use last max_seq_len tokens)
            input_tensor = generated[:, -self.position_encoding.max_len:]
            
            # Forward pass
            logits, _ = self(input_tensor)
            
            # Get logits for last position
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                next_token_logits = TopKSampler.filter_top_k(
                    next_token_logits,
                    top_k=top_k,
                )
            
            # Apply top-p filtering
            if top_p < 1.0:
                next_token_logits = TopPSampler.filter_top_p(
                    next_token_logits,
                    top_p=top_p,
                )
            
            # Sample
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append
            generated = torch.cat([generated, next_token], dim=1)
            
            # Check for EOS
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
        
        return generated
