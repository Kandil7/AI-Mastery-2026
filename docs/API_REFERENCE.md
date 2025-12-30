# AI Engineer Toolkit: API Reference

This document provides a reference for the key modules and classes in the toolkit.

## Table of Contents
- [src.core](#src-core): Mathematical Foundations
- [src.ml](#src-ml): Machine Learning Algorithms
- [src.llm](#src-llm): Large Language Model Engineering
- [src.production](#src-production): Production Engineering

---

## src.core

Foundational mathematics and optimization algorithms implemented from scratch.

### `math_operations`
*Linear algebra, matrix operations, and activation functions.*
- `dot(v1, v2)`: Compute dot product.
- `cosine_similarity(v1, v2)`: Compute cosine similarity.
- `PCA`: Principal Component Analysis implementation.
  - `fit(X)`: Compute principal components.
  - `transform(X)`: Project data onto components.
- Activations: `sigmoid`, `relu`, `softmax`, `tanh`.

### `probability`
*Probability distributions and information theory.*
- Distributions: `Gaussian`, `Bernoulli`, `Categorical`, `Uniform`.
  - Methods: `sample(n)`, `pdf(x)`, `log_pdf(x)`.
- Information Theory: `entropy`, `kl_divergence`, `cross_entropy`.
- Sampling: `rejection_sampling`, `metropolis_hastings` (MCMC).

### `optimization`
*Gradient-based optimization algorithms.*
- `SGD`: Stochastic Gradient Descent (with Momentum/Nesterov).
- `Adam`: Adaptive Moment Estimation.
- `RMSprop`: Root Mean Square Propagation.
- `LRScheduler`: Base class for learning rate schedulers (`StepLR`, `CosineAnnealingLR`).

---

## src.ml

Classical and deep learning algorithms implemented using pure NumPy.

### `classical`
*Traditional machine learning algorithms.*
- `LinearRegressionScratch`: Closed-form and gradient descent solvers.
- `LogisticRegression`: Binary and multiclass classification.
- `KMeans`: Clustering algorithm.
- `DecisionTree`: CART implementation with Gini/Entropy impurity.

### `deep_learning`
*Neural network primitives and backpropagation engine.*
- `Dense`: Fully connected layer.
- `Dropout`: Regularization layer.
- `Conv2D`: 2D Convolutional layer (basic implementation).
- `Sequential`: Container for stacking layers.
- `Loss`: `MSE`, `CrossEntropyLoss`.

---

## src.llm

Tools for building LLM applications, RAG systems, and agents.

### `attention`
*Transformer components.*
- `MultiHeadAttention`: Scaled dot-product attention with multiple heads.
- `TransformerBlock`: Full encoder block (Attention + FeedForward + Norm).
- `PositionalEncoding`: Sinusoidal and RoPE embeddings.

### `rag`
*Retrieval Augmented Generation pipeline.*
- `RAGPipeline`: Main coordinator (Retrieve -> Context -> Generate).
- `VectorIndex`: Abstract base class for vector storage.
- `Document`: Data class for text content and metadata.

### `fine_tuning`
*Parameter-Efficient Fine-Tuning (PEFT).*
- `LoRAConfig`: Configuration for Low-Rank Adaptation.
- `LoRALayer`: Wrapper for linear layers to add trainable rank-decomposition matrices.

### `agents`
*Agentic workflows.*
- `ReActAgent`: Implements Reason+Act pattern.
- `Tool`: Wrapper for Python functions to be used by agents.

---

## src.production

Utilities for deploying, monitoring, and scaling AI systems.

### `caching`
*Performance optimization.*
- `LRUCache`: Thread-safe Least Recently Used cache.
- `RedisCache`: Interface for Redis (requires `redis-py`).
- Decorators: `@cached`, `@async_cached`.

### `api`
*Model serving.*
- `ModelServer`: Wrapper to expose models via FastAPI.
- `PredictionRequest`: Pydantic model for input validation.

### `monitoring`
*System observability.*
- `DriftDetector`: Statistical tests for data drift (KS-test).
- `PerformanceMonitor`: Latency and throughput tracking.

### `deployment`
*Ops scripts and utilities.*
- `load_config`: YAML configuration loader.
- `HealthCheck`: System status verification.
