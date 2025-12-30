# Guide: Module `src/ml`

The `src/ml` module builds upon the mathematical foundations in `src/core` to implement classical and modern machine learning algorithms from scratch. This module is central to the "White-Box" learning path, translating theory into practical code.

## 1. `classical.py`

This file contains from-scratch implementations of a wide array of classical (non-deep-learning) machine learning algorithms. The classes are designed to mimic the familiar `scikit-learn` API, with `fit()`, `predict()`, and `score()` methods.

### Key Algorithms Implemented

*   **`LinearRegressionScratch`**:
    *   Implements ordinary least squares regression.
    *   Supports two fitting methods:
        1.  `'closed_form'`: Uses the Normal Equation (`w = (XᵀX)⁻¹Xᵀy`), which is fast for small numbers of features.
        2.  `'gradient_descent'`: Uses the optimization framework to iteratively find the best weights.
    *   Supports `l1` (Lasso) and `l2` (Ridge) regularization.

*   **`LogisticRegressionScratch`**:
    *   Implements logistic regression for classification.
    *   Handles binary classification using the sigmoid function and binary cross-entropy loss.
    *   Supports multi-class classification via two strategies:
        1.  `'ovr'` (One-vs-Rest): Trains a binary classifier for each class against all others.
        2.  `'softmax'`: Uses the softmax function and cross-entropy loss for true multi-class classification.

*   **`KNNScratch` (K-Nearest Neighbors)**:
    *   A non-parametric, "lazy learning" algorithm.
    *   Makes predictions based on the majority class of the `k` closest training examples.
    *   Supports multiple distance metrics (`euclidean`, `manhattan`, `cosine`).
    *   Can use `uniform` or `distance`-based weighting for predictions.

*   **`DecisionTreeScratch`**:
    *   A from-scratch implementation of a decision tree classifier.
    *   Recursively finds the best feature and threshold to split the data based on **Information Gain**.
    *   Supports two impurity measures for calculating gain: `'gini'` impurity and `'entropy'`.
    *   Includes pruning parameters like `max_depth` and `min_samples_split` to prevent overfitting.

*   **`RandomForestScratch`**:
    *   An ensemble model that builds multiple decision trees to improve robustness and reduce overfitting.
    *   Uses two key techniques:
        1.  **Bagging (Bootstrap Aggregating):** Each tree is trained on a random sample of the data (with replacement).
        2.  **Random Feature Selection:** At each split in a tree, only a random subset of features is considered.
    *   Final predictions are made by a majority vote of all trees.

*   **`GaussianNBScratch` (Gaussian Naive Bayes)**:
    *   A probabilistic classifier based on Bayes' Theorem.
    *   It "naively" assumes that all features are independent.
    *   Models the distribution of features for each class as a Gaussian (normal) distribution.

## 2. `deep_learning.py`

This file provides the building blocks for creating and training neural networks from scratch. It demonstrates the core concepts of forward and backward propagation.

### Key Components

*   **`Layer` Abstract Base Class**:
    *   Defines the interface for all network layers, requiring `forward()` and `backward()` methods.

*   **Layer Implementations**:
    *   `Dense`: A standard fully-connected layer (`y = Wx + b`). It handles weight initialization (`Xavier`, `He`) and computes gradients for weights and biases during the backward pass.
    *   `Activation`: A layer to apply non-linear activation functions like `relu`, `sigmoid`, `tanh`, and `softmax`.
    *   `Dropout`: A regularization technique that randomly zeroes out a fraction of neurons during training to prevent co-adaptation and overfitting.
    *   `BatchNormalization`: A layer that normalizes the activations within a batch to stabilize and accelerate training. It maintains running statistics for use during inference.

*   **Loss Functions**:
    *   `Loss` Abstract Base Class: Defines the interface for loss functions.
    *   `MSELoss`: Mean Squared Error, for regression tasks.
    *   `CrossEntropyLoss` and `BinaryCrossEntropyLoss`: For multi-class and binary classification, respectively. They are implemented to work efficiently with `softmax` and `sigmoid` outputs.

*   **`NeuralNetwork` Class**:
    *   A sequential model that stacks the layers together.
    *   `add(layer)`: Appends a layer to the network.
    *   `compile(loss, learning_rate)`: Configures the model with a loss function and learning rate.
    *   `fit(X, y, ...)`: A complete training loop that handles:
        *   Mini-batching and data shuffling.
        *   Executing the forward pass for each batch.
        *   Calculating the loss.
        *   Executing the backward pass to update weights across all layers.
        *   Optional validation and progress printing.
    *   `predict(X)` and `evaluate(X, y)`: For making predictions and evaluating model performance.

*   **Convolutional Primitives**:
    *   `conv2d_single` and `max_pool2d`: Simplified, educational implementations of 2D convolution and max pooling to demonstrate the core mechanics of Convolutional Neural Networks (CNNs).

---

The `src/ml` module provides a powerful hands-on toolkit for understanding how fundamental machine learning algorithms work internally, bridging the gap between theory and implementation.
