# Deep Learning Module
"""
Deep learning components implemented from scratch and with PyTorch.
Includes neural networks, CNNs, RNNs, LSTMs, and transformers.
"""

from .neural_networks import NeuralNetwork, MultiClassClassifier
from .cnn import Conv2D, MaxPool2D, Flatten, CNN
from .rnn import RNNCell, LSTMCell, GRUCell, RNN
from .transformers import (
    MultiHeadAttention,
    PositionWiseFeedForward,
    LayerNormalization,
    EncoderLayer,
    DecoderLayer,
    Transformer,
)
from .diffusion import (
    DiffusionConfig,
    NoiseSchedule,
    DDPM,
    LatentDiffusionModel,
    ClassifierFreeGuidance,
    create_diffusion_model,
    UNet,
    SinusoidalPositionEmbeddings,
    ResBlock,
    AttentionBlock,
)

__all__ = [
    # Neural Networks
    "NeuralNetwork",
    "MultiClassClassifier",
    # CNN
    "Conv2D",
    "MaxPool2D",
    "Flatten",
    "CNN",
    # RNN
    "RNNCell",
    "LSTMCell",
    "GRUCell",
    "RNN",
    # Transformers
    "MultiHeadAttention",
    "PositionWiseFeedForward",
    "LayerNormalization",
    "EncoderLayer",
    "DecoderLayer",
    "Transformer",
    # Diffusion Models
    "DiffusionConfig",
    "NoiseSchedule",
    "DDPM",
    "LatentDiffusionModel",
    "ClassifierFreeGuidance",
    "create_diffusion_model",
    "UNet",
    "SinusoidalPositionEmbeddings",
    "ResBlock",
    "AttentionBlock",
]
