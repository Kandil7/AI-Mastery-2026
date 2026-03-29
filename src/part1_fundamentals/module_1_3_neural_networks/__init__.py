"""
Module 1.3: Neural Networks from Scratch.

This module provides complete neural network implementations from scratch:
- Activation functions (ReLU, Sigmoid, Tanh, Softmax, GELU, Swish, etc.)
- Loss functions (MSE, Cross-Entropy, BCE, Huber, etc.)
- Layers (Dense, Conv2D, MaxPool2D, Dropout, BatchNorm)
- Optimizers (SGD, Adam, RMSprop, Adagrad, AdamW)
- Complete MLP implementation

Example Usage:
    >>> from module_1_3_neural_networks import MLP, Trainer
    >>> from module_1_3_neural_networks import Dense, ReLU, Dropout, BatchNormalization
    >>> from module_1_3_neural_networks import MSELoss, CrossEntropyLoss
    >>> from module_1_3_neural_networks import SGD, Adam, RMSprop
    >>> 
    >>> # Create MLP
    >>> model = MLP(
    ...     input_size=784,
    ...     hidden_sizes=[256, 128],
    ...     output_size=10,
    ...     activation='relu',
    ...     dropout=0.5
    ... )
    >>> 
    >>> # Train
    >>> trainer = Trainer(model, loss='cross_entropy', optimizer='adam')
    >>> history = trainer.fit(X_train, y_train, epochs=10)
    >>> 
    >>> # Manual layer usage
    >>> dense = Dense(input_size=10, output_size=5)
    >>> x = np.random.randn(32, 10)
    >>> output = dense.forward(x)
"""

from .activations import (
    ActivationFunction,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU,
    ELU,
    Swish,
    SiLU,
    GELU,
    Mish,
    Softmax,
    Identity,
    BinaryStep,
    get_activation,
)

from .losses import (
    LossFunction,
    MSELoss,
    MSELossStable,
    MAELoss,
    HuberLoss,
    BinaryCrossEntropyLoss,
    CrossEntropyLoss,
    CategoricalCrossEntropyLoss,
    SparseCategoricalCrossEntropyLoss,
    HingeLoss,
    KLDivergenceLoss,
    get_loss,
)

from .layers import (
    Layer,
    Dense,
    Conv2D,
    MaxPool2D,
    Dropout,
    BatchNormalization,
    Flatten,
    Reshape,
    Sequential,
)

from .optimizers import (
    Optimizer,
    SGD,
    Momentum,
    Adam,
    AdamW,
    RMSprop,
    Adagrad,
    LearningRateScheduler,
    GradientClipper,
    get_optimizer,
)

from .mlp import (
    MLP,
    Trainer,
    TrainingConfig,
    TrainingHistory,
    create_mlp_classifier,
    create_mlp_regressor,
)

__all__ = [
    # Activations
    'ActivationFunction',
    'Sigmoid',
    'Tanh',
    'ReLU',
    'LeakyReLU',
    'ELU',
    'Swish',
    'SiLU',
    'GELU',
    'Mish',
    'Softmax',
    'Identity',
    'BinaryStep',
    'get_activation',
    
    # Losses
    'LossFunction',
    'MSELoss',
    'MSELossStable',
    'MAELoss',
    'HuberLoss',
    'BinaryCrossEntropyLoss',
    'CrossEntropyLoss',
    'CategoricalCrossEntropyLoss',
    'SparseCategoricalCrossEntropyLoss',
    'HingeLoss',
    'KLDivergenceLoss',
    'get_loss',
    
    # Layers
    'Layer',
    'Dense',
    'Conv2D',
    'MaxPool2D',
    'Dropout',
    'BatchNormalization',
    'Flatten',
    'Reshape',
    'Sequential',
    
    # Optimizers
    'Optimizer',
    'SGD',
    'Momentum',
    'Adam',
    'AdamW',
    'RMSprop',
    'Adagrad',
    'LearningRateScheduler',
    'GradientClipper',
    'get_optimizer',
    
    # MLP
    'MLP',
    'Trainer',
    'TrainingConfig',
    'TrainingHistory',
    'create_mlp_classifier',
    'create_mlp_regressor',
]

__version__ = '1.0.0'
__author__ = 'AI Mastery 2026'
