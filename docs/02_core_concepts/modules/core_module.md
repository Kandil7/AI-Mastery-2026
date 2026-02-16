# Guide: Module `src/core`

The `src/core` module is the mathematical heart of the `AI-Mastery-2026` project. It contains from-scratch implementations of the fundamental linear algebra and optimization algorithms that power most machine learning models. Understanding this module is the first step in the "White-Box Approach."

## 1. `math_operations.py`

This file focuses on vector and matrix operations, decomposition techniques, and key numerical utilities. All implementations are done using `NumPy`.

### Key Features

*   **Vector Operations:**
    *   `dot_product`, `magnitude`, `normalize`
    *   Distance metrics: `euclidean_distance`, `manhattan_distance`
    *   `cosine_similarity`: A critical function for measuring the angle between vectors, widely used in NLP and recommendation systems.

*   **Matrix Operations:**
    *   `matrix_multiply`: A from-scratch implementation of matrix multiplication to demonstrate the underlying `O(n³)` complexity.
    *   `transpose`, `identity_matrix`, `trace`, `frobenius_norm`.

*   **Matrix Decomposition:**
    *   `power_iteration`: An algorithm to find the dominant eigenvector and eigenvalue of a matrix.
    *   `qr_decomposition`: Decomposes a matrix `A` into an orthogonal matrix `Q` and an upper triangular matrix `R`.
    *   `eigendecomposition`: Implemented using the QR algorithm to find all eigenvalues and eigenvectors of a symmetric matrix.
    *   `svd_simple`: A simplified wrapper around `np.linalg.svd` that is used to explain low-rank approximation.
    *   `low_rank_approximation`: Demonstrates how to use SVD for dimensionality reduction or noise reduction.

*   **Dimensionality Reduction:**
    *   **`PCA` class:** A full, from-scratch implementation of Principal Component Analysis.
        *   `fit()`: Computes the principal components by performing eigendecomposition on the covariance matrix.
        *   `transform()`: Projects data onto the principal components.
        *   `inverse_transform()`: Reconstructs the data from its reduced representation.

*   **Numerical Utilities & Activation Functions:**
    *   `softmax`: A numerically stable implementation, crucial for multi-class classification and attention mechanisms.
    *   `sigmoid`, `relu`, `tanh`, `gelu`: Common activation functions used in neural networks.

## 2. `optimization.py`

This file provides a framework for gradient-based optimization, which is the core process for training most machine learning models.

### Key Features

*   **`Optimizer` Base Class:**
    *   An abstract base class that defines the interface for all optimizers (`step()` method).

*   **Gradient Descent Optimizers:**
    *   `SGD`: Implements Stochastic Gradient Descent with optional **momentum** and **Nesterov acceleration**.
    *   `Adam`: The workhorse of deep learning, combining momentum and adaptive learning rates. Implemented with bias correction.
    *   `AdamW`: An improved version of Adam with decoupled weight decay, often leading to better generalization.
    *   `RMSprop` and `AdaGrad`: Other adaptive learning rate optimizers, particularly useful for specific data types (sparse data, RNNs).

*   **Learning Rate Schedulers:**
    *   `LRScheduler` Base Class: Defines the interface for schedulers.
    *   `StepLR`: Decays the learning rate by a fixed factor at set intervals.
    *   `ExponentialLR`: Applies a continuous exponential decay to the learning rate.
    *   `CosineAnnealingLR`: A popular and effective scheduler that varies the learning rate according to a cosine curve.
    *   `WarmupScheduler`: A scheduler that gradually increases the learning rate for a set number of initial steps before decaying it, helping to stabilize training in large models like Transformers.

*   **Regularization Techniques:**
    *   Functions for applying `l1_regularization` (Lasso, for sparsity), `l2_regularization` (Ridge/Weight Decay, for preventing overfitting), and `elastic_net_regularization`. These functions return both the penalty and the gradient, ready to be integrated into a loss function.

*   **Training and Verification:**
    *   `gradient_descent_train`: A generic training loop that demonstrates how to use the optimizers, schedulers, and regularization techniques to train a model. It handles batching, shuffling, and progress reporting.
    *   `numerical_gradient` and `gradient_check`: Essential utility functions for debugging backpropagation. They allow you to verify that your analytical (hand-derived) gradients are correct by comparing them to a numerical approximation.

---

This module provides all the building blocks needed to construct and train machine learning models from the ground up. The clear, commented, from-scratch code makes it an excellent resource for learning.
