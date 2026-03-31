"""
Model Configuration Classes
===========================

Configuration classes for various model architectures and training setups.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class ModelType(str, Enum):
    """Supported model types."""
    TRANSFORMER = "transformer"
    BERT = "bert"
    GPT2 = "gpt2"
    LSTM = "lstm"
    CNN = "cnn"
    RESNET = "resnet"
    CUSTOM = "custom"


class OptimizerType(str, Enum):
    """Supported optimizer types."""
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    RMSPROP = "rmsprop"
    ADAGRAD = "adagrad"


@dataclass
class ModelConfig:
    """
    Base model configuration.

    Attributes:
        model_type: Type of model
        hidden_dim: Hidden layer dimension
        num_layers: Number of layers
        dropout: Dropout rate
        activation: Activation function
    """
    model_type: ModelType = ModelType.CUSTOM
    hidden_dim: int = 768
    num_layers: int = 12
    dropout: float = 0.1
    activation: str = "gelu"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_type": self.model_type.value,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "activation": self.activation,
        }


@dataclass
class TransformerConfig(ModelConfig):
    """
    Transformer-specific configuration.

    Attributes:
        num_heads: Number of attention heads
        ff_dim: Feed-forward dimension
        max_seq_len: Maximum sequence length
        vocab_size: Vocabulary size
        positional_encoding: Type of positional encoding
    """
    model_type: ModelType = ModelType.TRANSFORMER
    num_heads: int = 12
    ff_dim: int = 3072
    max_seq_len: int = 512
    vocab_size: int = 30522
    positional_encoding: str = "sinusoidal"  # sinusoidal, learned, rope

    def __post_init__(self):
        """Validate configuration."""
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})"
            )

    @property
    def head_dim(self) -> int:
        """Dimension per attention head."""
        return self.hidden_dim // self.num_heads


@dataclass
class TrainingConfig:
    """
    Training configuration.

    Attributes:
        learning_rate: Initial learning rate
        weight_decay: Weight decay for regularization
        batch_size: Training batch size
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps
        max_grad_norm: Maximum gradient norm for clipping
        optimizer: Optimizer type
        scheduler: Learning rate scheduler type
        early_stopping_patience: Patience for early stopping
        save_every: Save checkpoint every N epochs
        device: Device for training
    """
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 10
    warmup_steps: int = 1000
    max_grad_norm: float = 1.0
    optimizer: OptimizerType = OptimizerType.ADAM
    scheduler: str = "cosine"  # linear, cosine, constant
    early_stopping_patience: int = 5
    save_every: int = 1
    device: str = "cpu"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "warmup_steps": self.warmup_steps,
            "max_grad_norm": self.max_grad_norm,
            "optimizer": self.optimizer.value,
            "scheduler": self.scheduler,
            "early_stopping_patience": self.early_stopping_patience,
            "save_every": self.save_every,
            "device": self.device,
        }


@dataclass
class LLMConfig:
    """
    Large Language Model configuration.

    Attributes:
        model_name: Pre-trained model name or path
        tokenizer_name: Tokenizer name or path
        max_length: Maximum input length
        truncation: Truncation strategy
        padding: Padding strategy
        use_cache: Use KV cache for inference
    """
    model_name: str = "bert-base-uncased"
    tokenizer_name: Optional[str] = None
    max_length: int = 512
    truncation: str = "longest_first"
    padding: str = "max_length"
    use_cache: bool = True

    def __post_init__(self):
        """Set defaults."""
        if self.tokenizer_name is None:
            self.tokenizer_name = self.model_name


@dataclass
class RAGConfig:
    """
    Retrieval-Augmented Generation configuration.

    Attributes:
        chunk_size: Size of document chunks
        chunk_overlap: Overlap between chunks
        top_k: Number of retrieved documents
        embedding_model: Embedding model name
        vector_store_type: Type of vector store
        rerank: Enable reranking
        rerank_model: Reranking model name
    """
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_store_type: str = "faiss"  # faiss, hnsw, qdrant
    rerank: bool = True
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "embedding_model": self.embedding_model,
            "vector_store_type": self.vector_store_type,
            "rerank": self.rerank,
            "rerank_model": self.rerank_model,
        }
