"""
Deep Learning Implementation Module
====================================
This module provides implementations of deep learning algorithms from scratch.

Classes:
    - ConvLayer: Convolutional layer for CNNs
    - MaxPoolLayer: Max pooling layer
    - FlattenLayer: Flatten layer for converting 2D to 1D
    - FullyConnectedLayer: Dense layer
    - ReLU, Sigmoid, Softmax: Activation functions
    - CrossEntropyLoss: Cross-entropy loss for classification
    - RNNCell: Single RNN cell
    - LSTMCell: Single LSTM cell
    - SimpleCNN: Complete CNN model
    - SimpleRNN: Complete RNN model
    - SimpleLSTM: Complete LSTM model

Author: AI-Mastery-2026
"""

from .cnn import (
    ConvLayer,
    MaxPoolLayer,
    FlattenLayer,
    FullyConnectedLayer,
    ReLU,
    Sigmoid,
    Softmax,
    CrossEntropyLoss,
    SimpleCNN,
)
from .rnn import RNNCell, SimpleRNN
from .lstm import LSTMCell, SimpleLSTM

__all__ = [
    # CNN components
    "ConvLayer",
    "MaxPoolLayer",
    "FlattenLayer",
    "FullyConnectedLayer",
    "ReLU",
    "Sigmoid",
    "Softmax",
    "CrossEntropyLoss",
    "SimpleCNN",
    # RNN components
    "RNNCell",
    "SimpleRNN",
    # LSTM components
    "LSTMCell",
    "SimpleLSTM",
]
