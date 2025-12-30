# AI-Mastery-2026: Complete Documentation

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Foundations](#mathematical-foundations)
3. [Classical Machine Learning](#classical-machine-learning)
4. [Deep Learning](#deep-learning)
5. [LLM Engineering](#llm-engineering)
6. [Production Systems](#production-systems)
7. [API Reference](#api-reference)
8. [Best Practices](#best-practices)

## Introduction

The AI-Mastery-2026 toolkit is built on the "White-Box Approach" philosophy, emphasizing understanding of mathematical foundations before using abstractions. This documentation provides comprehensive coverage of all components, from basic mathematical operations to production-ready systems.

### Philosophy
- **Math First**: Derive equations, understand foundations
- **Code Second**: Implement from scratch with NumPy
- **Libraries Third**: Use sklearn/PyTorch knowing what happens underneath
- **Production Always**: Every concept includes deployment considerations

## Mathematical Foundations

### Core Mathematical Operations

#### Vector Operations
- `dot_product(v1, v2)`: Computes the dot product of two vectors
  - Mathematical Definition: a · b = Σ(aᵢ × bᵢ) = ||a|| ||b|| cos(θ)
  - Used in: Similarity calculations, projections

- `magnitude(v)`: Computes the L2 norm (Euclidean magnitude) of a vector
  - Mathematical Definition: ||v|| = √(Σvᵢ²)
  - Used in: Normalization, distance calculations

- `normalize(v)`: Normalizes a vector to unit length
  - Mathematical Definition: v̂ = v / ||v||
  - Used in: Creating unit vectors for direction-only calculations

- `cosine_similarity(v1, v2)`: Computes cosine similarity between two vectors
  - Mathematical Definition: cos(θ) = (a · b) / (||a|| × ||b||)
  - Range: [-1, 1] where 1 is same direction, 0 is orthogonal, -1 is opposite direction
  - Used in: Semantic similarity, recommendation systems

#### Matrix Operations
- `matrix_multiply(A, B)`: Matrix multiplication from scratch
  - Mathematical Definition: C[i,j] = Σₖ A[i,k] × B[k,j]
  - Complexity: O(n³) for square matrices
  - Used in: Linear transformations, neural network layers

- `transpose(A)`: Matrix transpose operation
  - Mathematical Definition: B[i,j] = A[j,i]
  - Used in: Matrix operations, solving linear systems

- `identity_matrix(n)`: Creates an n×n identity matrix
  - Mathematical Definition: I[i,i] = 1, I[i,j] = 0 for i ≠ j
  - Used in: Matrix operations, as neutral element for multiplication

#### Matrix Decomposition
- `power_iteration(A, num_iterations, tolerance)`: Finds dominant eigenvalue/eigenvector
  - Algorithm: Repeatedly multiply by matrix and normalize
  - Used in: PageRank algorithm, principal component analysis

- `qr_decomposition(A)`: QR decomposition using Gram-Schmidt
  - Decomposes A = QR where Q is orthogonal and R is upper triangular
  - Used in: Solving linear systems, least squares problems

- `eigendecomposition(A, num_iterations)`: Finds eigenvalues and eigenvectors
  - Finds A = VΛV⁻¹ where V contains eigenvectors and Λ contains eigenvalues
  - Used in: Principal Component Analysis, dimensionality reduction

#### Principal Component Analysis (PCA)
- `class PCA`: Implements PCA from scratch
  - Mathematical Foundation:
    1. Center data: X_c = X - μ
    2. Covariance: Σ = (1/n) X_cᵀ X_c
    3. Eigendecomposition: Σ = VΛVᵀ
    4. Project: X_new = X_c × V[:, :k]
  - Used in: Dimensionality reduction, feature extraction, visualization

### Probability and Statistics

#### Probability Distributions
- `class Gaussian`: Gaussian (Normal) Distribution
  - Mathematical Definition: f(x) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))
  - Properties: Mean μ, Variance σ², Maximum entropy for given mean and variance
  - Applications: Modeling errors and noise, prior distributions in Bayesian ML

- `class Bernoulli`: Bernoulli Distribution
  - Mathematical Definition: P(X = 1) = p, P(X = 0) = 1 - p
  - Applications: Binary classification, coin flips, dropout in neural networks

- `class Categorical`: Categorical Distribution (generalized Bernoulli)
  - Mathematical Definition: P(X = k) = p_k for k = 1, ..., K
  - Applications: Multi-class classification, language models

#### Information Theory
- `entropy(p, base)`: Shannon Entropy
  - Mathematical Definition: H(X) = -Σ p(x) × log(p(x))
  - Interpretation: Measures uncertainty/randomness, maximum for uniform distribution
  - Used in: Decision trees (information gain)

- `cross_entropy(p, q)`: Cross Entropy
  - Mathematical Definition: H(p, q) = -Σ p(x) × log(q(x))
  - Used as loss function in classification

- `kl_divergence(p, q)`: Kullback-Leibler Divergence
  - Mathematical Definition: D_KL(P || Q) = Σ p(x) × log(p(x) / q(x))
  - Properties: D_KL(P || Q) ≥ 0, D_KL(P || Q) = 0 iff P = Q, not symmetric
  - Applications: VAE loss function, distribution comparison

## Classical Machine Learning

### Linear Regression
- `class LinearRegressionScratch`: Linear Regression from scratch
  - Model: ŷ = Xw + b
  - Loss: MSE = (1/n) Σ(yᵢ - ŷᵢ)²
  - Methods: Closed-form (Normal Equation) and Gradient Descent
  - Mathematical Derivation: ∂L/∂w = (2/n) Xᵀ(Xw - y)

### Logistic Regression
- `class LogisticRegressionScratch`: Logistic Regression from scratch
  - Model: P(y=1|x) = σ(xᵀw + b) = 1 / (1 + e^{-(xᵀw + b)})
  - Loss (Binary Cross-Entropy): L = -(1/n) Σ[yᵢ log(p̂ᵢ) + (1-yᵢ) log(1-p̂ᵢ)]
  - Gradient: ∂L/∂w = (1/n) Xᵀ(σ(Xw) - y)

### K-Nearest Neighbors
- `class KNNScratch`: K-Nearest Neighbors from scratch
  - Algorithm: Store all training data, for new point find k closest neighbors, predict by majority vote
  - Time Complexity: O(n) per prediction (naive implementation)
  - Suffers from "curse of dimensionality" in high dimensions

### Decision Trees
- `class DecisionTreeScratch`: Decision Tree Classifier from scratch
  - Splitting Criteria: Gini Impurity or Entropy
  - Gini Impurity: G = 1 - Σpᵢ²
  - Entropy: H = -Σpᵢ log(pᵢ)
  - Information Gain: IG = H(parent) - Σ(nⱼ/n)H(childⱼ)

### Random Forest
- `class RandomForestScratch`: Random Forest Classifier from scratch
  - Ensemble method combining: Bootstrap aggregating (Bagging) and Random feature selection at each split
  - Reduces overfitting compared to individual decision trees

## Deep Learning

### Neural Network Components
- `class Dense`: Fully Connected (Dense) Layer
  - Forward: y = Wx + b
  - Backward: ∂L/∂W = ∂L/∂y × xᵀ, ∂L/∂b = ∂L/∂y, ∂L/∂x = Wᵀ × ∂L/∂y

- `class Activation`: Activation Layer
  - Applies element-wise activation function (ReLU, Sigmoid, Tanh, Softmax, etc.)
  - ReLU: max(0, x), Sigmoid: 1 / (1 + e^{-x}), Tanh: (e^x - e^{-x}) / (e^x + e^{-x})

- `class Dropout`: Dropout regularization layer
  - Randomly zeros out units during training to prevent overfitting
  - During inference, all units are active (scaled by keep probability)

- `class BatchNormalization`: Batch Normalization layer
  - Normalizes activations to have zero mean and unit variance
  - Reduces internal covariate shift and allows higher learning rates

### Loss Functions
- `class MSELoss`: Mean Squared Error Loss
  - L = (1/n) Σ(ŷ - y)²
  - ∂L/∂ŷ = (2/n)(ŷ - y)

- `class CrossEntropyLoss`: Cross-Entropy Loss (for classification)
  - Binary: L = -[y log(p) + (1-y) log(1-p)]
  - Multi-class: L = -Σ yᵢ log(pᵢ)
  - With softmax output, gradient simplifies to: ∂L/∂z = p - y

### Neural Network (Sequential Model)
- `class NeuralNetwork`: Sequential Neural Network model
  - Stacks layers in sequence, handles forward/backward propagation
  - Implements training loop with mini-batch processing

## LLM Engineering

### Attention Mechanisms
- `scaled_dot_product_attention(Q, K, V, mask)`: Scaled Dot-Product Attention mechanism
  - Computes attention weights and outputs using: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

- `class MultiHeadAttention`: Multi-Head Attention mechanism
  - Concatenates multiple attention heads to capture different aspects of the input
  - Each head learns to attend to different parts of the sequence

- `class TransformerBlock`: Single Transformer Block (Encoder)
  - Combines Multi-Head Attention, Feed-Forward Network, and residual connections with Layer Normalization

### RAG Pipeline
- `class RAGPipeline`: Complete RAG pipeline
  - Pipeline: Query → Embed → Retrieve → Rerank → Context → LLM → Response
  - Components: TextChunker, EmbeddingModel, Retriever, Reranker, ContextAssembler

### Fine-Tuning Techniques
- `class LoRALayer`: LoRA (Low-Rank Adaptation) layer
  - Original: y = Wx, With LoRA: y = Wx + (B @ A)x × (α/r)
  - Parameter-efficient fine-tuning method

## Production Systems

### Caching
- `class LRUCache`: Least Recently Used (LRU) Cache
  - Thread-safe implementation with eviction policy
  - Uses OrderedDict for O(1) access and ordering

- `class EmbeddingCache`: Specialized cache for embeddings with content hashing
  - Features: Content-based hashing for deduplication, Batch get/set operations

### Monitoring
- `class DriftDetector`: Unified drift detection for multiple features
  - Monitors data drift and concept drift in production
  - Methods: KS test, PSI (Population Stability Index)

- `class PerformanceMonitor`: Monitor model performance over time
  - Tracks prediction latencies, classification metrics, regression metrics, error rates

### Deployment
- `class ModelSerializer`: Unified model serialization across formats
  - Supports pickle, joblib, ONNX, PyTorch, and TensorFlow
  - Includes metadata tracking for model provenance

- `class ModelVersionManager`: Manages multiple model versions for safe deployments
  - Supports blue-green deployments, canary releases, quick rollbacks

## API Reference

### Core Mathematical Operations
```python
from src.core.math_operations import (
    dot_product, magnitude, normalize, cosine_similarity,
    euclidean_distance, manhattan_distance,
    matrix_multiply, transpose, identity_matrix, trace,
    power_iteration, gram_schmidt, qr_decomposition,
    covariance_matrix, PCA,
    softmax, sigmoid, relu
)
```

### Classical ML Algorithms
```python
from src.ml.classical import (
    LinearRegressionScratch, LogisticRegressionScratch,
    KNNScratch, DecisionTreeScratch, RandomForestScratch,
    GaussianNBScratch
)
```

### Deep Learning Components
```python
from src.ml.deep_learning import (
    Dense, Activation, Dropout, BatchNormalization,
    MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss,
    NeuralNetwork
)
```

### LLM Components
```python
from src.llm.attention import (
    scaled_dot_product_attention, MultiHeadAttention,
    TransformerBlock, FeedForwardNetwork, LayerNorm
)

from src.llm.rag import (
    Document, RetrievalResult, TextChunker,
    EmbeddingModel, Retriever, Reranker,
    ContextAssembler, RAGPipeline
)

from src.llm.fine_tuning import (
    LoRALayer, LinearWithLoRA, AdapterLayer,
    quantize_nf4, dequantize_nf4
)
```

### Production Components
```python
from src.production.caching import (
    LRUCache, RedisCache, EmbeddingCache, PredictionCache
)

from src.production.monitoring import (
    DriftDetector, PerformanceMonitor, AlertManager,
    ks_test, psi, chi_square_test
)

from src.production.deployment import (
    ModelSerializer, ModelVersionManager, HealthChecker,
    GracefulShutdown
)
```

## Best Practices

### Mathematical Foundations
- Always understand the mathematical underpinnings before implementing
- Use numerical stability techniques (e.g., log-sum-exp trick)
- Implement gradient checking to verify backpropagation implementations

### Model Development
- Follow the bias-variance tradeoff principle
- Use cross-validation for model selection
- Implement proper train/validation/test splits

### Production Considerations
- Monitor model performance and data drift in production
- Implement proper error handling and logging
- Use version control for models and experiments
- Design for graceful degradation

### Performance Optimization
- Use vectorized operations where possible
- Consider memory efficiency for large datasets
- Profile code to identify bottlenecks
- Use appropriate data structures for the task