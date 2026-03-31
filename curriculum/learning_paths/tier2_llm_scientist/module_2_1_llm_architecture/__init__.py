"""
Module 2.1: LLM Architecture

Core building blocks of Large Language Models:
- Attention mechanisms (Multi-head, Self, Masked, Cross, Flash)
- Transformer architectures (Encoder, Decoder, Encoder-Decoder)
- Tokenization (BPE, WordPiece, SentencePiece)
- Sampling strategies (Greedy, Beam, Temperature, Top-p, Top-k)
"""

from .attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    SelfAttention,
    MaskedAttention,
    CrossAttention,
    FlashAttention,
)
from .transformer import (
    PositionalEncoding,
    LearnedPositionalEncoding,
    RotaryPositionalEmbedding,
    TransformerEncoderLayer,
    TransformerDecoderLayer,
    TransformerEncoder,
    TransformerDecoder,
    Transformer,
)
from .tokenization import (
    BPETokenizer,
    WordPieceTokenizer,
    SentencePieceTokenizer,
    TokenProcessor,
)
from .sampling import (
    GreedySampler,
    BeamSearchSampler,
    TemperatureSampler,
    TopPSampler,
    TopKSampler,
    ContrastiveSampler,
)

__all__ = [
    # Attention
    "ScaledDotProductAttention",
    "MultiHeadAttention",
    "SelfAttention",
    "MaskedAttention",
    "CrossAttention",
    "FlashAttention",
    # Transformer
    "PositionalEncoding",
    "LearnedPositionalEncoding",
    "RotaryPositionalEmbedding",
    "TransformerEncoderLayer",
    "TransformerDecoderLayer",
    "TransformerEncoder",
    "TransformerDecoder",
    "Transformer",
    # Tokenization
    "BPETokenizer",
    "WordPieceTokenizer",
    "SentencePieceTokenizer",
    "TokenProcessor",
    # Sampling
    "GreedySampler",
    "BeamSearchSampler",
    "TemperatureSampler",
    "TopPSampler",
    "TopKSampler",
    "ContrastiveSampler",
]
