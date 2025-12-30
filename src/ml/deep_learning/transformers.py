"""
Transformer Implementation

This module implements the Transformer architecture from scratch using NumPy,
including multi-head attention, feed-forward networks, and positional encoding.
"""

import numpy as np
from typing import Tuple, Optional
import matplotlib.pyplot as plt


def scaled_dot_product_attention(
    Q: np.ndarray, 
    K: np.ndarray, 
    V: np.ndarray, 
    mask: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Scaled dot-product attention mechanism.
    
    Args:
        Q: Query matrix of shape (batch_size, seq_len, d_k)
        K: Key matrix of shape (batch_size, seq_len, d_k)
        V: Value matrix of shape (batch_size, seq_len, d_v)
        mask: Optional mask to prevent attention to certain positions
        
    Returns:
        Tuple of (attention_output, attention_weights)
    """
    d_k = Q.shape[-1]
    
    # Compute attention scores: (batch_size, seq_len, seq_len)
    scores = np.matmul(Q, np.transpose(K, (0, 2, 1))) / np.sqrt(d_k)
    
    # Apply mask if provided
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)  # Very negative value for masked positions
    
    # Apply softmax to get attention weights
    attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))  # For numerical stability
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    
    # Compute output: (batch_size, seq_len, d_v)
    output = np.matmul(attention_weights, V)
    
    return output, attention_weights


class MultiHeadAttention:
    """
    Multi-Head Attention mechanism implementation.
    """
    
    def __init__(self, d_model: int, num_heads: int):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.d_v = d_model // num_heads
        
        # Initialize weight matrices
        self.W_q = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_k = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_v = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
        self.W_o = np.random.randn(d_model, d_model) * np.sqrt(2.0 / d_model)
    
    def split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Split the last dimension into (num_heads, depth).
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Reshaped tensor of shape (batch_size, num_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Reshape to (batch_size, seq_len, num_heads, d_k)
        x = x.reshape(batch_size, seq_len, self.num_heads, self.d_k)
        
        # Transpose to (batch_size, num_heads, seq_len, d_k)
        return np.transpose(x, (0, 2, 1, 3))
    
    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        """
        Combine heads back to original shape.
        
        Args:
            x: Input tensor of shape (batch_size, num_heads, seq_len, d_k)
            
        Returns:
            Reshaped tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size, num_heads, seq_len, d_k = x.shape
        
        # Transpose to (batch_size, seq_len, num_heads, d_k)
        x = np.transpose(x, (0, 2, 1, 3))
        
        # Reshape to (batch_size, seq_len, d_model)
        return x.reshape(batch_size, seq_len, self.num_heads * d_k)
    
    def forward(
        self, 
        Q: np.ndarray, 
        K: np.ndarray, 
        V: np.ndarray, 
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass through multi-head attention.
        
        Args:
            Q: Query tensor of shape (batch_size, seq_len_q, d_model)
            K: Key tensor of shape (batch_size, seq_len_k, d_model)
            V: Value tensor of shape (batch_size, seq_len_v, d_model)
            mask: Optional mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = Q.shape[0]
        
        # Linear projections
        Q_proj = np.dot(Q, self.W_q)  # (batch_size, seq_len_q, d_model)
        K_proj = np.dot(K, self.W_k)  # (batch_size, seq_len_k, d_model)
        V_proj = np.dot(V, self.W_v)  # (batch_size, seq_len_v, d_model)
        
        # Split into heads
        Q_split = self.split_heads(Q_proj)  # (batch_size, num_heads, seq_len_q, d_k)
        K_split = self.split_heads(K_proj)  # (batch_size, num_heads, seq_len_k, d_k)
        V_split = self.split_heads(V_proj)  # (batch_size, num_heads, seq_len_v, d_k)
        
        # Apply scaled dot-product attention to each head
        attention_output = np.zeros((batch_size, self.num_heads, Q_split.shape[2], self.d_v))
        attention_weights = np.zeros((batch_size, self.num_heads, Q_split.shape[2], K_split.shape[2]))
        
        for head in range(self.num_heads):
            head_Q = Q_split[:, head, :, :]  # (batch_size, seq_len_q, d_k)
            head_K = K_split[:, head, :, :]  # (batch_size, seq_len_k, d_k)
            head_V = V_split[:, head, :, :]  # (batch_size, seq_len_v, d_k)
            
            head_output, head_weights = scaled_dot_product_attention(
                head_Q, head_K, head_V, mask
            )
            
            attention_output[:, head, :, :] = head_output
            attention_weights[:, head, :, :] = head_weights
        
        # Combine heads
        combined_output = self.combine_heads(attention_output)  # (batch_size, seq_len_q, d_model)
        
        # Final linear projection
        output = np.dot(combined_output, self.W_o)  # (batch_size, seq_len_q, d_model)
        
        return output, attention_weights


class PositionWiseFeedForward:
    """
    Position-wise Feed-Forward Network implementation.
    """
    
    def __init__(self, d_model: int, d_ff: int):
        """
        Initialize Position-wise Feed-Forward Network.
        
        Args:
            d_model: Model dimension
            d_ff: Hidden layer dimension
        """
        self.d_model = d_model
        self.d_ff = d_ff
        
        # Initialize weight matrices
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2.0 / d_model)
        self.b1 = np.zeros(d_ff)
        self.W2 = np.random.randn(d_ff, d_model) * np.sqrt(2.0 / d_ff)
        self.b2 = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the feed-forward network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # First linear transformation and activation
        hidden = np.dot(x, self.W1) + self.b1  # (batch_size, seq_len, d_ff)
        hidden = np.maximum(0, hidden)  # ReLU activation
        
        # Second linear transformation
        output = np.dot(hidden, self.W2) + self.b2  # (batch_size, seq_len, d_model)
        
        return output


class LayerNormalization:
    """
    Layer Normalization implementation.
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        """
        Initialize Layer Normalization.
        
        Args:
            d_model: Model dimension
            eps: Small value for numerical stability
        """
        self.d_model = d_model
        self.eps = eps
        
        # Initialize scale and shift parameters
        self.gamma = np.ones(d_model)
        self.beta = np.zeros(d_model)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through layer normalization.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            
        Returns:
            Normalized tensor of same shape
        """
        # Calculate mean and variance along the last dimension
        mean = np.mean(x, axis=-1, keepdims=True)  # (batch_size, seq_len, 1)
        variance = np.var(x, axis=-1, keepdims=True)  # (batch_size, seq_len, 1)
        
        # Normalize
        normalized = (x - mean) / np.sqrt(variance + self.eps)  # (batch_size, seq_len, d_model)
        
        # Scale and shift
        output = self.gamma * normalized + self.beta
        
        return output


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """
    Generate positional encoding.
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        
    Returns:
        Positional encoding matrix of shape (seq_len, d_model)
    """
    # Create position and model indices
    pos = np.arange(seq_len)[:, np.newaxis]  # (seq_len, 1)
    i = np.arange(d_model)[np.newaxis, :]  # (1, d_model)
    
    # Calculate angle rates
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / d_model)
    
    # Calculate angles
    angles = pos * angle_rates  # (seq_len, d_model)
    
    # Apply sin to even indices and cos to odd indices
    angles[:, 0::2] = np.sin(angles[:, 0::2])  # Apply sin to even indices
    angles[:, 1::2] = np.cos(angles[:, 1::2])  # Apply cos to odd indices
    
    return angles


class EncoderLayer:
    """
    Transformer Encoder Layer implementation.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        """
        Initialize Encoder Layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden layer dimension in feed-forward network
            dropout_rate: Dropout rate
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        # Multi-head self-attention
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        
        # Layer normalizations
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        
        # Feed-forward network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        
        # Dropout
        self.dropout1 = dropout_rate
        self.dropout2 = dropout_rate
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass through the encoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask for attention
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Multi-head self-attention
        attn_output, _ = self.multi_head_attention(x, x, x, mask)
        
        # Apply dropout
        if self.dropout1 > 0:
            # Simple dropout implementation (randomly zero out some elements)
            dropout_mask = np.random.binomial(1, 1 - self.dropout1, attn_output.shape) / (1 - self.dropout1)
            attn_output = attn_output * dropout_mask
        
        # Add & Norm
        x = self.norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        
        # Apply dropout
        if self.dropout2 > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout2, ff_output.shape) / (1 - self.dropout2)
            ff_output = ff_output * dropout_mask
        
        # Add & Norm
        output = self.norm2(x + ff_output)
        
        return output


class DecoderLayer:
    """
    Transformer Decoder Layer implementation.
    """
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout_rate: float = 0.1):
        """
        Initialize Decoder Layer.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden layer dimension in feed-forward network
            dropout_rate: Dropout rate
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout_rate
        
        # Masked multi-head self-attention
        self.masked_multi_head_attention = MultiHeadAttention(d_model, num_heads)
        
        # Multi-head attention (for encoder-decoder attention)
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads)
        
        # Layer normalizations
        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)
        self.norm3 = LayerNormalization(d_model)
        
        # Feed-forward network
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        
        # Dropout
        self.dropout1 = dropout_rate
        self.dropout2 = dropout_rate
        self.dropout3 = dropout_rate
    
    def forward(
        self, 
        x: np.ndarray, 
        encoder_output: np.ndarray, 
        look_ahead_mask: Optional[np.ndarray] = None,
        padding_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass through the decoder layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, seq_len, d_model)
            look_ahead_mask: Mask to prevent attention to future positions
            padding_mask: Mask to prevent attention to padding tokens
            
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        # Masked multi-head self-attention
        attn1_output, _ = self.masked_multi_head_attention(x, x, x, look_ahead_mask)
        
        # Apply dropout
        if self.dropout1 > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout1, attn1_output.shape) / (1 - self.dropout1)
            attn1_output = attn1_output * dropout_mask
        
        # Add & Norm
        x = self.norm1(x + attn1_output)
        
        # Multi-head attention (encoder-decoder attention)
        attn2_output, _ = self.multi_head_attention(x, encoder_output, encoder_output, padding_mask)
        
        # Apply dropout
        if self.dropout2 > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout2, attn2_output.shape) / (1 - self.dropout2)
            attn2_output = attn2_output * dropout_mask
        
        # Add & Norm
        x = self.norm2(x + attn2_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        
        # Apply dropout
        if self.dropout3 > 0:
            dropout_mask = np.random.binomial(1, 1 - self.dropout3, ff_output.shape) / (1 - self.dropout3)
            ff_output = ff_output * dropout_mask
        
        # Add & Norm
        output = self.norm3(x + ff_output)
        
        return output


class Transformer:
    """
    Transformer model implementation.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        input_vocab_size: int,
        target_vocab_size: int,
        max_seq_len: int,
        dropout_rate: float = 0.1
    ):
        """
        Initialize Transformer model.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder/decoder layers
            d_ff: Hidden layer dimension in feed-forward networks
            input_vocab_size: Size of input vocabulary
            target_vocab_size: Size of target vocabulary
            max_seq_len: Maximum sequence length
            dropout_rate: Dropout rate
        """
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout_rate
        
        # Input and target embeddings
        self.input_embedding = np.random.randn(input_vocab_size, d_model) * np.sqrt(1.0 / d_model)
        self.target_embedding = np.random.randn(target_vocab_size, d_model) * np.sqrt(1.0 / d_model)
        
        # Positional encodings
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        
        # Encoder layers
        self.encoder_layers = [
            EncoderLayer(d_model, num_heads, d_ff, dropout_rate) 
            for _ in range(num_layers)
        ]
        
        # Decoder layers
        self.decoder_layers = [
            DecoderLayer(d_model, num_heads, d_ff, dropout_rate) 
            for _ in range(num_layers)
        ]
        
        # Final linear layer
        self.final_linear = np.random.randn(d_model, target_vocab_size) * np.sqrt(2.0 / d_model)
        
        # Layer normalization before final linear
        self.norm = LayerNormalization(d_model)
    
    def encode(self, input_seq: np.ndarray) -> np.ndarray:
        """
        Encode input sequence.
        
        Args:
            input_seq: Input sequence of shape (batch_size, seq_len)
            
        Returns:
            Encoded sequence of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = input_seq.shape
        
        # Embedding
        x = self.input_embedding[input_seq]  # (batch_size, seq_len, d_model)
        
        # Scale embeddings
        x *= np.sqrt(self.d_model)
        
        # Add positional encoding
        x += self.pos_encoding[:seq_len, :]
        
        # Pass through encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        return x
    
    def decode(
        self, 
        target_seq: np.ndarray, 
        encoder_output: np.ndarray,
        look_ahead_mask: Optional[np.ndarray] = None,
        padding_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Decode target sequence.
        
        Args:
            target_seq: Target sequence of shape (batch_size, seq_len)
            encoder_output: Encoder output of shape (batch_size, seq_len, d_model)
            look_ahead_mask: Mask to prevent attention to future positions
            padding_mask: Mask to prevent attention to padding tokens
            
        Returns:
            Decoded sequence of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len = target_seq.shape
        
        # Embedding
        x = self.target_embedding[target_seq]  # (batch_size, seq_len, d_model)
        
        # Scale embeddings
        x *= np.sqrt(self.d_model)
        
        # Add positional encoding
        x += self.pos_encoding[:seq_len, :]
        
        # Pass through decoder layers
        for layer in self.decoder_layers:
            x = layer(x, encoder_output, look_ahead_mask, padding_mask)
        
        return x
    
    def forward(
        self, 
        input_seq: np.ndarray, 
        target_seq: np.ndarray,
        look_ahead_mask: Optional[np.ndarray] = None,
        padding_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Forward pass through the transformer.
        
        Args:
            input_seq: Input sequence of shape (batch_size, seq_len)
            target_seq: Target sequence of shape (batch_size, seq_len)
            look_ahead_mask: Mask to prevent attention to future positions
            padding_mask: Mask to prevent attention to padding tokens
            
        Returns:
            Output logits of shape (batch_size, seq_len, target_vocab_size)
        """
        # Encode input
        encoder_output = self.encode(input_seq)
        
        # Decode target
        decoder_output = self.decode(target_seq, encoder_output, look_ahead_mask, padding_mask)
        
        # Apply layer norm
        decoder_output = self.norm(decoder_output)
        
        # Final linear transformation
        output = np.dot(decoder_output, self.final_linear)
        
        return output
    
    def predict(self, input_seq: np.ndarray) -> np.ndarray:
        """
        Make predictions on input sequence.
        
        Args:
            input_seq: Input sequence of shape (batch_size, seq_len)
            
        Returns:
            Predictions of shape (batch_size, seq_len, target_vocab_size)
        """
        # For prediction, we need to generate the target sequence step by step
        # This is a simplified implementation - in practice, you'd use techniques like beam search
        batch_size, seq_len = input_seq.shape
        
        # Encode input
        encoder_output = self.encode(input_seq)
        
        # Start with a start token (e.g., 0) for the target sequence
        target_seq = np.zeros((batch_size, 1), dtype=int)
        
        outputs = []
        
        # Generate sequence step by step
        for i in range(seq_len):
            # Decode current target sequence
            decoder_output = self.decode(target_seq, encoder_output)
            
            # Apply layer norm
            decoder_output = self.norm(decoder_output)
            
            # Final linear transformation
            logits = np.dot(decoder_output, self.final_linear)
            
            # Get the last token's logits (for next prediction)
            next_token_logits = logits[:, -1, :]  # (batch_size, target_vocab_size)
            
            # Sample next token (using argmax for simplicity)
            next_token = np.argmax(next_token_logits, axis=-1)  # (batch_size,)
            
            # Append to target sequence
            target_seq = np.concatenate([target_seq, next_token[:, np.newaxis]], axis=1)
            
            # Store output
            outputs.append(logits)
        
        # Stack outputs along the sequence dimension
        outputs = np.concatenate(outputs, axis=1)
        
        return outputs


def create_transformer_for_classification(
    d_model: int,
    num_heads: int,
    num_layers: int,
    d_ff: int,
    vocab_size: int,
    num_classes: int,
    max_seq_len: int
) -> Transformer:
    """
    Create a Transformer model for classification tasks.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of encoder/decoder layers
        d_ff: Hidden layer dimension in feed-forward networks
        vocab_size: Size of vocabulary
        num_classes: Number of output classes
        max_seq_len: Maximum sequence length
        
    Returns:
        Transformer model for classification
    """
    # For classification, we can use just the encoder part
    # and take the representation of the first token (or apply pooling)
    return Transformer(
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        input_vocab_size=vocab_size,
        target_vocab_size=num_classes,  # For classification, target vocab = num classes
        max_seq_len=max_seq_len
    )


def cross_entropy_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate cross-entropy loss.
    
    Args:
        y_true: True labels (one-hot encoded or class indices)
        y_pred: Predicted logits
        
    Returns:
        Cross-entropy loss
    """
    # Apply softmax to predictions
    exp_pred = np.exp(y_pred - np.max(y_pred, axis=-1, keepdims=True))  # For numerical stability
    softmax_pred = exp_pred / np.sum(exp_pred, axis=-1, keepdims=True)
    
    # If y_true is class indices, convert to one-hot
    if y_true.ndim == 2:  # Class indices
        one_hot = np.eye(y_pred.shape[-1])[y_true]
    else:  # One-hot encoded
        one_hot = y_true
    
    # Clip predictions to prevent log(0)
    softmax_pred = np.clip(softmax_pred, 1e-15, 1 - 1e-15)
    
    # Calculate cross-entropy
    loss = -np.sum(one_hot * np.log(softmax_pred)) / y_true.shape[0]
    
    return loss


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy score.
    
    Args:
        y_true: True labels
        y_pred: Predicted logits or probabilities
        
    Returns:
        Accuracy score
    """
    # If y_pred contains logits, convert to class predictions
    if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
        y_pred = np.argmax(y_pred, axis=-1)
    
    # If y_true is one-hot encoded, convert to class indices
    if y_true.ndim > 1 and y_true.shape[-1] > 1:
        y_true = np.argmax(y_true, axis=-1)
    
    return np.mean(y_true.flatten() == y_pred.flatten())