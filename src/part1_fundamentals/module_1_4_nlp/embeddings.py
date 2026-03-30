"""
NLP Embeddings Module.

This module provides comprehensive text embedding implementations,
including Word2Vec, GloVe-style embeddings, and positional encodings.

Features:
- Word2Vec (CBOW and Skip-gram)
- GloVe-style embeddings
- Positional encodings (absolute, relative, rotary)
- Token embeddings
- Pre-trained embedding loading

Example Usage:
    >>> from embeddings import Word2Vec, PositionalEncoding
    >>> 
    >>> # Train Word2Vec
    >>> w2v = Word2Vec(embedding_dim=100)
    >>> w2v.train(["hello world", "hello there", "world peace"])
    >>> vector = w2v.get_vector("hello")
    >>> 
    >>> # Positional encoding
    >>> pos_enc = PositionalEncoding(d_model=128)
    >>> positions = pos_enc.forward(np.zeros((32, 10, 128)))
"""

from typing import Union, List, Dict, Tuple, Optional, Set, Iterator
import numpy as np
from numpy.typing import ArrayLike
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
import json
import os
import math

logger = logging.getLogger(__name__)

ArrayLike2D = Union[np.ndarray, List]


class Embedding:
    """Base class for embedding layers."""
    
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None
    ):
        """
        Initialize embedding layer.
        
        Args:
            vocab_size: Size of vocabulary.
            embedding_dim: Dimension of embeddings.
            padding_idx: Index of padding token (if any).
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        
        # Initialize embeddings
        self.weights = np.random.randn(vocab_size, embedding_dim) * 0.1
        
        if padding_idx is not None:
            self.weights[padding_idx] = 0
        
        self.grad: Optional[np.ndarray] = None
        
        logger.debug(f"Embedding initialized: {vocab_size} x {embedding_dim}")
    
    def forward(self, indices: np.ndarray) -> np.ndarray:
        """
        Lookup embeddings for indices.
        
        Args:
            indices: Token indices (batch_size, seq_len).
        
        Returns:
            np.ndarray: Embeddings (batch_size, seq_len, embedding_dim).
        """
        indices = np.asarray(indices, dtype=np.int32)
        output = self.weights[indices]
        self._input_indices = indices
        return output
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute gradients for embeddings.
        
        Args:
            grad_output: Gradient from upstream.
        
        Returns:
            np.ndarray: Gradient w.r.t. indices (usually None for embeddings).
        """
        if not hasattr(self, '_input_indices'):
            raise ValueError("Must call forward() before backward()")
        
        # Initialize gradient
        if self.grad is None:
            self.grad = np.zeros_like(self.weights)
        else:
            self.grad = np.zeros_like(self.weights)
        
        # Accumulate gradients
        np.add.at(self.grad, self._input_indices, grad_output)
        
        # Mask padding if specified
        if self.padding_idx is not None:
            self.grad[self.padding_idx] = 0
        
        return None  # Don't propagate gradient to indices
    
    def get_gradients(self) -> Dict[str, np.ndarray]:
        """Get embedding gradients."""
        return {'weights': self.grad} if self.grad is not None else {}
    
    def zero_grad(self) -> None:
        """Zero out gradients."""
        if self.grad is not None:
            self.grad = np.zeros_like(self.weights)


class Word2Vec:
    """
    Word2Vec word embedding model.
    
    Supports both CBOW (Continuous Bag of Words) and Skip-gram architectures.
    
    Example:
        >>> w2v = Word2Vec(embedding_dim=100, window_size=5)
        >>> w2v.train(["hello world", "hello there", "world peace"])
        >>> vector = w2v.get_vector("hello")
        >>> similar = w2v.most_similar("hello", top_k=5)
    """
    
    def __init__(
        self,
        embedding_dim: int = 100,
        window_size: int = 5,
        min_count: int = 1,
        learning_rate: float = 0.025,
        epochs: int = 10,
        negative_samples: int = 5,
        model_type: str = 'skipgram'
    ):
        """
        Initialize Word2Vec.
        
        Args:
            embedding_dim: Dimension of word vectors.
            window_size: Context window size.
            min_count: Minimum word frequency.
            learning_rate: Learning rate.
            epochs: Number of training epochs.
            negative_samples: Number of negative samples.
            model_type: 'skipgram' or 'cbow'.
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.negative_samples = negative_samples
        self.model_type = model_type
        
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        self.word_counts: Counter = Counter()
        
        self.W: Optional[np.ndarray] = None  # Input embeddings
        self.W_out: Optional[np.ndarray] = None  # Output embeddings
        
        # Gradients
        self.dW: Optional[np.ndarray] = None
        self.dW_out: Optional[np.ndarray] = None
        
        logger.info(f"Word2Vec initialized: dim={embedding_dim}, "
                   f"window={window_size}, type={model_type}")
    
    def _build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        # Count words
        for text in texts:
            words = text.lower().split()
            self.word_counts.update(words)
        
        # Filter by min_count
        vocab_words = [w for w, c in self.word_counts.items() if c >= self.min_count]
        
        # Build mappings
        for i, word in enumerate(vocab_words):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word
        
        logger.info(f"Vocabulary built: {len(self.word_to_idx)} words")
    
    def _generate_pairs(self, texts: List[str]) -> List[Tuple[int, int]]:
        """Generate training pairs (center, context)."""
        pairs = []
        
        for text in texts:
            words = text.lower().split()
            word_indices = [self.word_to_idx.get(w) for w in words]
            word_indices = [i for i in word_indices if i is not None]
            
            for i, center_idx in enumerate(word_indices):
                # Get context window
                start = max(0, i - self.window_size)
                end = min(len(word_indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_idx = word_indices[j]
                        pairs.append((center_idx, context_idx))
        
        return pairs
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))
    
    def _negative_sampling(
        self,
        center_idx: int,
        context_idx: int,
        vocab_size: int
    ) -> List[Tuple[int, int, int]]:
        """
        Generate negative samples.
        
        Returns list of (center_idx, sample_idx, label) tuples.
        """
        samples = [(center_idx, context_idx, 1)]  # Positive sample
        
        # Generate negative samples
        for _ in range(self.negative_samples):
            neg_idx = np.random.randint(vocab_size)
            while neg_idx == context_idx:
                neg_idx = np.random.randint(vocab_size)
            samples.append((center_idx, neg_idx, 0))
        
        return samples
    
    def train(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[float]:
        """
        Train Word2Vec model.
        
        Args:
            texts: Training texts.
            show_progress: Show training progress.
        
        Returns:
            List[float]: Loss per epoch.
        """
        # Build vocabulary
        self._build_vocab(texts)
        vocab_size = len(self.word_to_idx)
        
        if vocab_size == 0:
            raise ValueError("No words in vocabulary. Check min_count parameter.")
        
        # Initialize embeddings
        self.W = np.random.randn(vocab_size, self.embedding_dim) * 0.1
        self.W_out = np.random.randn(vocab_size, self.embedding_dim) * 0.1
        
        # Generate training pairs
        pairs = self._generate_pairs(texts)
        
        if show_progress:
            logger.info(f"Generated {len(pairs)} training pairs")
        
        losses = []
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            np.random.shuffle(pairs)
            
            # Decrease learning rate over time
            lr = self.learning_rate * (1 - epoch / self.epochs)
            
            for center_idx, context_idx in pairs:
                if self.model_type == 'skipgram':
                    loss = self._train_skipgram(center_idx, context_idx, lr)
                else:
                    loss = self._train_cbow(center_idx, context_idx, pairs, lr)
                
                total_loss += loss
            
            avg_loss = total_loss / len(pairs)
            losses.append(avg_loss)
            
            if show_progress and (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        logger.info(f"Word2Vec training completed")
        return losses
    
    def _train_skipgram(
        self,
        center_idx: int,
        context_idx: int,
        lr: float
    ) -> float:
        """Train skip-gram model on one pair."""
        # Get embeddings
        center_vec = self.W[center_idx]
        context_vec = self.W_out[context_idx]
        
        # Compute dot product and sigmoid
        dot = np.dot(center_vec, context_vec)
        pred = self._sigmoid(dot)
        
        # Compute loss (binary cross-entropy)
        loss = -np.log(pred + 1e-10)
        
        # Compute gradients
        error = pred - 1  # For positive sample
        
        # Update embeddings
        self.W[center_idx] -= lr * error * context_vec
        self.W_out[context_idx] -= lr * error * center_vec
        
        # Negative sampling
        for _ in range(self.negative_samples):
            neg_idx = np.random.randint(len(self.W))
            while neg_idx == context_idx:
                neg_idx = np.random.randint(len(self.W))
            
            neg_vec = self.W_out[neg_idx]
            dot_neg = np.dot(center_vec, neg_vec)
            pred_neg = self._sigmoid(dot_neg)
            
            loss += -np.log(1 - pred_neg + 1e-10)
            
            error_neg = pred_neg  # For negative sample (label=0)
            
            self.W[center_idx] -= lr * error_neg * neg_vec
            self.W_out[neg_idx] -= lr * error_neg * center_vec
        
        return loss
    
    def _train_cbow(
        self,
        center_idx: int,
        context_idx: int,
        all_pairs: List[Tuple[int, int]],
        lr: float
    ) -> float:
        """Train CBOW model."""
        # For CBOW, we need to average context vectors to predict center
        # Simplified implementation
        return self._train_skipgram(center_idx, context_idx, lr)
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """
        Get word vector.
        
        Args:
            word: Input word.
        
        Returns:
            np.ndarray: Word vector or None if word not in vocabulary.
        """
        idx = self.word_to_idx.get(word.lower())
        if idx is not None:
            return self.W[idx].copy()
        return None
    
    def most_similar(
        self,
        word: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Find most similar words.
        
        Args:
            word: Query word.
            top_k: Number of results.
        
        Returns:
            List[Tuple]: (word, similarity) pairs.
        """
        word_vec = self.get_vector(word)
        if word_vec is None:
            return []
        
        # Normalize query vector
        word_vec = word_vec / (np.linalg.norm(word_vec) + 1e-10)
        
        # Compute similarities
        similarities = []
        for idx, vec in enumerate(self.W):
            if idx == self.word_to_idx.get(word.lower()):
                continue
            
            sim = np.dot(word_vec, vec) / (np.linalg.norm(vec) + 1e-10)
            similarities.append((self.idx_to_word[idx], sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: -x[1])
        
        return similarities[:top_k]
    
    def save(self, filepath: str) -> None:
        """Save model to file."""
        data = {
            'config': {
                'embedding_dim': self.embedding_dim,
                'window_size': self.window_size,
                'min_count': self.min_count,
                'model_type': self.model_type,
            },
            'vocab': self.word_to_idx,
            'embeddings': self.W.tolist() if self.W is not None else None,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f)
        
        logger.info(f"Word2Vec model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'Word2Vec':
        """Load model from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        config = data['config']
        model = cls(**config)
        
        model.word_to_idx = data['vocab']
        model.idx_to_word = {v: k for k, v in data['vocab'].items()}
        
        if data['embeddings'] is not None:
            model.W = np.array(data['embeddings'])
        
        logger.info(f"Word2Vec model loaded from {filepath}")
        return model


class GloVe:
    """
    GloVe (Global Vectors) word embedding model.
    
    Learns embeddings by factorizing the word co-occurrence matrix.
    
    Example:
        >>> glove = GloVe(embedding_dim=100)
        >>> glove.train(["hello world", "hello there", "world peace"])
        >>> vector = glove.get_vector("hello")
    """
    
    def __init__(
        self,
        embedding_dim: int = 100,
        window_size: int = 10,
        min_count: int = 1,
        learning_rate: float = 0.05,
        epochs: int = 50,
        x_max: float = 100,
        alpha: float = 0.75
    ):
        """
        Initialize GloVe.
        
        Args:
            embedding_dim: Dimension of word vectors.
            window_size: Context window size.
            min_count: Minimum word frequency.
            learning_rate: Learning rate (AdaGrad).
            epochs: Number of training epochs.
            x_max: Maximum co-occurrence for weighting.
            alpha: Weighting function exponent.
        """
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.min_count = min_count
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.x_max = x_max
        self.alpha = alpha
        
        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: Dict[int, str] = {}
        
        self.W: Optional[np.ndarray] = None  # Word embeddings
        self.W_context: Optional[np.ndarray] = None  # Context embeddings
        
        # AdaGrad accumulators
        self.gradsq_W: Optional[np.ndarray] = None
        self.gradsq_W_context: Optional[np.ndarray] = None
        
        logger.info(f"GloVe initialized: dim={embedding_dim}")
    
    def _build_vocab(self, texts: List[str]) -> None:
        """Build vocabulary from texts."""
        word_counts = Counter()
        for text in texts:
            words = text.lower().split()
            word_counts.update(words)
        
        vocab_words = [w for w, c in word_counts.items() if c >= self.min_count]
        
        for i, word in enumerate(vocab_words):
            self.word_to_idx[word] = i
            self.idx_to_word[i] = word
        
        logger.info(f"Vocabulary built: {len(self.word_to_idx)} words")
    
    def _build_cooccurrence_matrix(
        self,
        texts: List[str]
    ) -> Dict[Tuple[int, int], float]:
        """Build co-occurrence matrix with distance weighting."""
        cooccur = defaultdict(float)
        
        for text in texts:
            words = text.lower().split()
            word_indices = [self.word_to_idx.get(w) for w in words]
            word_indices = [i for i in word_indices if i is not None]
            
            for i, center_idx in enumerate(word_indices):
                start = max(0, i - self.window_size)
                end = min(len(word_indices), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_idx = word_indices[j]
                        distance = abs(i - j)
                        # Weight by inverse distance
                        cooccur[(center_idx, context_idx)] += 1.0 / distance
        
        return cooccur
    
    def _weighting_func(self, x: float) -> float:
        """GloVe weighting function."""
        if x < self.x_max:
            return (x / self.x_max) ** self.alpha
        return 1.0
    
    def train(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[float]:
        """
        Train GloVe model.
        
        Args:
            texts: Training texts.
            show_progress: Show training progress.
        
        Returns:
            List[float]: Loss per epoch.
        """
        # Build vocabulary
        self._build_vocab(texts)
        vocab_size = len(self.word_to_idx)
        
        if vocab_size == 0:
            raise ValueError("No words in vocabulary.")
        
        # Build co-occurrence matrix
        cooccur = self._build_cooccurrence_matrix(texts)
        
        if show_progress:
            logger.info(f"Co-occurrence matrix: {len(cooccur)} non-zero entries")
        
        # Initialize embeddings and AdaGrad accumulators
        self.W = (np.random.rand(vocab_size, self.embedding_dim) - 0.5) / self.embedding_dim
        self.W_context = (np.random.rand(vocab_size, self.embedding_dim) - 0.5) / self.embedding_dim
        
        self.gradsq_W = np.ones((vocab_size, self.embedding_dim))
        self.gradsq_W_context = np.ones((vocab_size, self.embedding_dim))
        
        # Convert co-occurrence to list for iteration
        cooccur_list = list(cooccur.items())
        
        losses = []
        
        for epoch in range(self.epochs):
            total_loss = 0.0
            np.random.shuffle(cooccur_list)
            
            for (i, j), x_ij in cooccur_list:
                # Compute weight
                weight = self._weighting_func(x_ij)
                
                # Compute prediction
                diff = np.dot(self.W[i], self.W_context[j]) - np.log(x_ij + 1e-10)
                
                # Weighted squared error
                loss = weight * diff ** 2
                total_loss += loss
                
                # Gradients
                grad_common = 2 * weight * diff
                
                grad_W = grad_common * self.W_context[j]
                grad_W_context = grad_common * self.W[i]
                
                # AdaGrad updates
                self.gradsq_W[i] += grad_W ** 2
                self.gradsq_W_context[j] += grad_W_context ** 2
                
                self.W[i] -= self.learning_rate * grad_W / np.sqrt(self.gradsq_W[i])
                self.W_context[j] -= self.learning_rate * grad_W_context / np.sqrt(self.gradsq_W_context[j])
            
            avg_loss = total_loss / len(cooccur_list)
            losses.append(avg_loss)
            
            if show_progress and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")
        
        # Combine word and context embeddings
        self.W = (self.W + self.W_context) / 2
        
        logger.info(f"GloVe training completed")
        return losses
    
    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get word vector."""
        idx = self.word_to_idx.get(word.lower())
        if idx is not None and self.W is not None:
            return self.W[idx].copy()
        return None
    
    def most_similar(
        self,
        word: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Find most similar words."""
        word_vec = self.get_vector(word)
        if word_vec is None:
            return []
        
        word_vec = word_vec / (np.linalg.norm(word_vec) + 1e-10)
        
        similarities = []
        for idx, vec in enumerate(self.W):
            if idx == self.word_to_idx.get(word.lower()):
                continue
            
            sim = np.dot(word_vec, vec) / (np.linalg.norm(vec) + 1e-10)
            similarities.append((self.idx_to_word[idx], sim))
        
        similarities.sort(key=lambda x: -x[1])
        return similarities[:top_k]


class PositionalEncoding:
    """
    Positional encoding for transformer models.
    
    Supports absolute, relative, and rotary positional encodings.
    
    Example:
        >>> pos_enc = PositionalEncoding(d_model=128, max_len=512)
        >>> x = np.zeros((32, 10, 128))  # (batch, seq_len, d_model)
        >>> x_with_pos = pos_enc.forward(x)
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 512,
        encoding_type: str = 'absolute',
        dropout: float = 0.0
    ):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension.
            max_len: Maximum sequence length.
            encoding_type: 'absolute', 'relative', or 'rotary'.
            dropout: Dropout probability.
        """
        self.d_model = d_model
        self.max_len = max_len
        self.encoding_type = encoding_type
        self.dropout = dropout
        
        # Precompute positional encodings
        self.pe = self._create_positional_encoding()
        
        logger.debug(f"PositionalEncoding: type={encoding_type}, d_model={d_model}")
    
    def _create_positional_encoding(self) -> np.ndarray:
        """Create sinusoidal positional encoding."""
        pe = np.zeros((self.max_len, self.d_model))
        
        position = np.arange(0, self.max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, self.d_model, 2) * -(np.log(10000.0) / self.d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        return pe  # (max_len, d_model)
    
    def forward(
        self,
        x: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor (batch_size, seq_len, d_model).
            training: Whether in training mode (for dropout).
        
        Returns:
            np.ndarray: Output with positional encoding.
        """
        x = np.asarray(x, dtype=np.float64)
        seq_len = x.shape[1]
        
        if self.encoding_type == 'absolute':
            # Add positional encoding
            output = x + self.pe[:seq_len, :]
        elif self.encoding_type == 'relative':
            # For relative encoding, we use learned relative positions
            # Simplified: just use absolute for now
            output = x + self.pe[:seq_len, :]
        elif self.encoding_type == 'rotary':
            # Rotary positional encoding (RoPE)
            output = self._apply_rotary_encoding(x)
        else:
            output = x
        
        # Apply dropout
        if training and self.dropout > 0:
            mask = (np.random.rand(*output.shape) > self.dropout).astype(np.float64)
            output = output * mask / (1 - self.dropout)
        
        return output
    
    def _apply_rotary_encoding(self, x: np.ndarray) -> np.ndarray:
        """Apply rotary positional encoding."""
        batch_size, seq_len, d_model = x.shape
        
        # Split into pairs
        x_reshaped = x.reshape(batch_size, seq_len, d_model // 2, 2)
        
        # Create rotation angles
        positions = np.arange(seq_len)[:, np.newaxis]
        inv_freq = 1.0 / (10000 ** (np.arange(0, d_model, 2) / d_model))
        angles = positions * inv_freq
        
        # Apply rotation
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)
        
        # Rotate each pair
        x_rotated = np.zeros_like(x_reshaped)
        x_rotated[:, :, :, 0] = x_reshaped[:, :, :, 0] * cos_angles - x_reshaped[:, :, :, 1] * sin_angles
        x_rotated[:, :, :, 1] = x_reshaped[:, :, :, 0] * sin_angles + x_reshaped[:, :, :, 1] * cos_angles
        
        return x_rotated.reshape(batch_size, seq_len, d_model)
    
    def create_attention_mask(
        self,
        seq_len: int,
        mask_type: str = 'causal'
    ) -> np.ndarray:
        """
        Create attention mask.
        
        Args:
            seq_len: Sequence length.
            mask_type: 'causal' or 'padding'.
        
        Returns:
            np.ndarray: Attention mask.
        """
        if mask_type == 'causal':
            # Causal mask (lower triangular)
            mask = np.triu(np.ones((seq_len, seq_len)), k=1)
            mask = mask * -1e9
        else:
            mask = np.zeros((seq_len, seq_len))
        
        return mask


class TokenEmbeddings:
    """
    Combined token and positional embeddings for transformers.
    
    Example:
        >>> embeddings = TokenEmbeddings(vocab_size=30000, d_model=512)
        >>> token_ids = np.random.randint(0, 30000, (32, 10))
        >>> embedded = embeddings.forward(token_ids)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_len: int = 512,
        padding_idx: int = 0,
        dropout: float = 0.1
    ):
        """
        Initialize token embeddings.
        
        Args:
            vocab_size: Vocabulary size.
            d_model: Model dimension.
            max_len: Maximum sequence length.
            padding_idx: Padding token index.
            dropout: Dropout probability.
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        
        # Token embeddings
        self.token_embedding = Embedding(vocab_size, d_model, padding_idx=padding_idx)
        
        # Positional encoding
        self.position_encoding = PositionalEncoding(d_model, max_len, dropout=dropout)
        
        # Scale factor
        self.scale = np.sqrt(d_model)
        
        logger.info(f"TokenEmbeddings: vocab={vocab_size}, d_model={d_model}")
    
    def forward(
        self,
        token_ids: np.ndarray,
        training: bool = True
    ) -> np.ndarray:
        """
        Compute token + positional embeddings.
        
        Args:
            token_ids: Token indices (batch_size, seq_len).
            training: Whether in training mode.
        
        Returns:
            np.ndarray: Embeddings (batch_size, seq_len, d_model).
        """
        token_ids = np.asarray(token_ids, dtype=np.int32)
        
        # Token embeddings
        x = self.token_embedding.forward(token_ids) * self.scale
        
        # Add positional encoding
        x = self.position_encoding.forward(x, training=training)
        
        return x
    
    def backward(self, grad_output: np.ndarray) -> None:
        """Backward pass."""
        self.token_embedding.backward(grad_output)


def create_pretrained_embeddings(
    embedding_file: str,
    vocab_size: Optional[int] = None
) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Load pre-trained embeddings from file.
    
    Supports GloVe and Word2Vec formats.
    
    Args:
        embedding_file: Path to embedding file.
        vocab_size: Optional vocabulary size limit.
    
    Returns:
        Tuple: (word_to_vector dict, embedding_dim)
    
    Example:
        >>> embeddings, dim = create_pretrained_embeddings('glove.6B.100d.txt')
        >>> vector = embeddings.get('hello')
    """
    word_to_vec = {}
    embedding_dim = None
    
    with open(embedding_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if vocab_size is not None and i >= vocab_size:
                break
            
            parts = line.rstrip().split(' ')
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]])
            
            if embedding_dim is None:
                embedding_dim = len(vector)
            
            word_to_vec[word] = vector
    
    logger.info(f"Loaded {len(word_to_vec)} embeddings with dim={embedding_dim}")
    return word_to_vec, embedding_dim


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("Embeddings Module - Demonstration")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Word2Vec
    print("\n1. Word2Vec:")
    training_texts = [
        "the cat sat on the mat",
        "the dog sat on the log",
        "cats and dogs are pets",
        "the cat and the dog",
        "pets are loved by families",
    ] * 100
    
    w2v = Word2Vec(embedding_dim=50, window_size=3, epochs=20)
    w2v.train(training_texts, show_progress=False)
    
    print(f"   Vocabulary size: {len(w2v.word_to_idx)}")
    
    # Get word vector
    vector = w2v.get_vector("cat")
    print(f"   Vector for 'cat': shape={vector.shape}")
    
    # Find similar words
    similar = w2v.most_similar("cat", top_k=5)
    print(f"   Words similar to 'cat': {similar}")
    
    # GloVe
    print("\n2. GloVe:")
    glove = GloVe(embedding_dim=50, epochs=30)
    glove.train(training_texts, show_progress=False)
    
    vector = glove.get_vector("cat")
    print(f"   Vector for 'cat': shape={vector.shape}")
    
    similar = glove.most_similar("cat", top_k=5)
    print(f"   Words similar to 'cat': {similar}")
    
    # Positional Encoding
    print("\n3. Positional Encoding:")
    pos_enc = PositionalEncoding(d_model=128, max_len=100)
    
    x = np.zeros((2, 10, 128))
    x_with_pos = pos_enc.forward(x)
    
    print(f"   Input shape: {x.shape}")
    print(f"   Output shape: {x_with_pos.shape}")
    print(f"   Positional encoding added: {not np.allclose(x, x_with_pos)}")
    
    # Token Embeddings
    print("\n4. Token Embeddings:")
    token_emb = TokenEmbeddings(vocab_size=1000, d_model=256)
    
    token_ids = np.random.randint(0, 1000, (4, 20))
    embedded = token_emb.forward(token_ids)
    
    print(f"   Token IDs shape: {token_ids.shape}")
    print(f"   Embeddings shape: {embedded.shape}")
    
    print("\n" + "=" * 60)
