"""
Machine Learning Module
=======================

Classical and Deep Learning algorithms implemented from scratch.
"""

from .classical import (
    LinearRegression,
    RidgeRegression,
    LassoRegression,
    LogisticRegression,
    MultinomialLogisticRegression,
    Node,
    DecisionTree,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForest,
    RandomForestRegressor,
    AdaBoostClassifier,
    GradientBoostingRegressor,
    SVM,
    SVR,
)

from .neural_networks_scratch import (
    Layer,
    Dense,
    Activation,
    Dropout,
    BatchNormalization,
    MSELoss,
    CrossEntropyLoss,
    BinaryCrossEntropyLoss,
    NeuralNetwork,
    LSTM,
    Conv2D,
    MaxPool2D,
    Flatten,
)
from .deep_learning import (
    NeuralNetwork as TorchNeuralNetwork,
    MultiClassClassifier,
    Conv2D as TorchConv2D,
    MaxPool2D as TorchMaxPool2D,
    Flatten as TorchFlatten,
    CNN,
    RNNCell,
    LSTMCell,
    GRUCell,
    RNN,
    MultiHeadAttention,
    PositionWiseFeedForward,
    LayerNormalization,
    EncoderLayer,
    DecoderLayer,
    Transformer,
)

from .vision import ResidualBlock, ResNet18

from .gnn_recommender import (
    NodeType,
    Node,
    Edge,
    BipartiteGraph,
    GraphSAGEAggregator,
    GraphSAGELayer,
    GNNRecommender,
    TwoTowerRanker,
    RankingLoss,
    ColdStartHandler,
    RecommenderMetrics,
)

__all__ = [
    # Classical ML
    "LinearRegression",
    "RidgeRegression",
    "LassoRegression",
    "LogisticRegression",
    "MultinomialLogisticRegression",
    "Node",
    "DecisionTree",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForest",
    "RandomForestRegressor",
    "AdaBoostClassifier",
    "GradientBoostingRegressor",
    "SVM",
    "SVR",
    # Deep Learning
    "NeuralNetwork",
    "MultiClassClassifier",
    "Conv2D",
    "MaxPool2D",
    "Flatten",
    "CNN",
    "RNNCell",
    "LSTMCell",
    "GRUCell",
    "RNN",
    "MultiHeadAttention",
    "PositionWiseFeedForward",
    "LayerNormalization",
    "EncoderLayer",
    "DecoderLayer",
    "Transformer",
    # Vision
    "ResidualBlock",
    "ResNet18",
    # GNN Recommender
    "NodeType",
    "Node",
    "Edge",
    "BipartiteGraph",
    "GraphSAGEAggregator",
    "GraphSAGELayer",
    "GNNRecommender",
    "TwoTowerRanker",
    "RankingLoss",
    "ColdStartHandler",
    "RecommenderMetrics",
]
