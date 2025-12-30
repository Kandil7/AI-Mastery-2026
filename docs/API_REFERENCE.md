# AI-Mastery-2026: API Reference Documentation

## Core Modules

### Core Mathematical Operations (`src.core.math_operations`)

#### Vector Operations
- `dot_product(v1: Vector, v2: Vector) -> float`
  - Compute dot product of two vectors
  - Mathematical Definition: a · b = Σ(aᵢ × bᵢ) = ||a|| ||b|| cos(θ)

- `magnitude(v: Vector) -> float`
  - Compute L2 norm (Euclidean magnitude) of a vector
  - Mathematical Definition: ||v|| = √(Σvᵢ²)

- `normalize(v: Vector) -> np.ndarray`
  - Normalize vector to unit length
  - Mathematical Definition: v̂ = v / ||v||

- `cosine_similarity(v1: Vector, v2: Vector) -> float`
  - Compute cosine similarity between two vectors
  - Mathematical Definition: cos(θ) = (a · b) / (||a|| × ||b||)
  - Range: [-1, 1]

- `euclidean_distance(v1: Vector, v2: Vector) -> float`
  - Compute Euclidean distance between two vectors
  - Mathematical Definition: d(a, b) = √(Σ(aᵢ - bᵢ)²)

- `manhattan_distance(v1: Vector, v2: Vector) -> float`
  - Compute Manhattan (L1) distance between two vectors
  - Mathematical Definition: d(a, b) = Σ|aᵢ - bᵢ|

#### Matrix Operations
- `matrix_multiply(A: Matrix, B: Matrix) -> np.ndarray`
  - Matrix multiplication from scratch (no np.matmul)
  - Mathematical Definition: C[i,j] = Σₖ A[i,k] × B[k,j]

- `transpose(A: Matrix) -> np.ndarray`
  - Matrix transpose
  - Mathematical Definition: B[i,j] = A[j,i]

- `identity_matrix(n: int) -> np.ndarray`
  - Create n×n identity matrix

- `trace(A: Matrix) -> float`
  - Compute trace of a square matrix (sum of diagonal elements)
  - Mathematical Definition: tr(A) = Σ A[i,i]

- `frobenius_norm(A: Matrix) -> float`
  - Compute Frobenius norm of a matrix
  - Mathematical Definition: ||A||_F = √(Σᵢⱼ A[i,j]²)

#### Matrix Decomposition
- `power_iteration(A: Matrix, num_iterations: int = 100, tolerance: float = 1e-10) -> Tuple[float, np.ndarray]`
  - Power iteration method to find dominant eigenvalue/eigenvector

- `gram_schmidt(vectors: Matrix) -> np.ndarray`
  - Gram-Schmidt orthogonalization
  - Produces orthonormal basis from input vectors

- `qr_decomposition(A: Matrix) -> Tuple[np.ndarray, np.ndarray]`
  - QR decomposition using Gram-Schmidt
  - Decomposes A = QR where Q is orthogonal and R is upper triangular

- `eigendecomposition(A: Matrix, num_iterations: int = 100) -> Tuple[np.ndarray, np.ndarray]`
  - Eigendecomposition using QR algorithm
  - Finds eigenvalues and eigenvectors: A = VΛV⁻¹

- `svd_simple(A: Matrix, k: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]`
  - Simplified SVD using eigendecomposition
  - Decomposes A = UΣVᵀ

- `low_rank_approximation(A: Matrix, k: int) -> np.ndarray`
  - Low-rank matrix approximation using truncated SVD

#### Principal Component Analysis (PCA)
- `class PCA(n_components: int)`
  - Principal Component Analysis from scratch
  - Mathematical Foundation: Center data → Covariance → Eigendecomposition → Project

  - `fit(X: Matrix) -> PCA`
    - Fit PCA model to data

  - `transform(X: Matrix) -> np.ndarray`
    - Apply dimensionality reduction

  - `fit_transform(X: Matrix) -> np.ndarray`
    - Fit and transform in one step

  - `inverse_transform(X_transformed: Matrix) -> np.ndarray`
    - Reconstruct data from reduced dimensions

#### Activation Functions
- `softmax(x: Vector, axis: int = -1) -> np.ndarray`
  - Softmax function (numerically stable)
  - Mathematical Definition: softmax(xᵢ) = exp(xᵢ) / Σⱼ exp(xⱼ)

- `log_softmax(x: Vector, axis: int = -1) -> np.ndarray`
  - Log-softmax (numerically stable)

- `sigmoid(x: Vector) -> np.ndarray`
  - Sigmoid activation function
  - Mathematical Definition: σ(x) = 1 / (1 + exp(-x))

- `relu(x: Vector) -> np.ndarray`
  - ReLU activation function
  - Mathematical Definition: ReLU(x) = max(0, x)

- `tanh(x: Vector) -> np.ndarray`
  - Hyperbolic tangent activation

- `gelu(x: Vector) -> np.ndarray`
  - Gaussian Error Linear Unit
  - Used in BERT, GPT and modern Transformers

### Core Optimization (`src.core.optimization`)

#### Optimizers
- `class Optimizer(ABC)`
  - Abstract base class for optimizers
  - All optimizers implement the update rule: θ_new = θ_old - step

- `class SGD(learning_rate: float = 0.01, momentum: float = 0.0, nesterov: bool = False)`
  - Stochastic Gradient Descent with optional momentum
  - Update Rule (with momentum): v = β × v + (1 - β) × ∇L, θ = θ - α × v

- `class Adam(learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8)`
  - Adam (Adaptive Moment Estimation) optimizer
  - Combines momentum with adaptive learning rates per parameter

- `class AdamW(learning_rate: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8, weight_decay: float = 0.01)`
  - Adam with decoupled weight decay (AdamW)

- `class RMSprop(learning_rate: float = 0.01, decay: float = 0.99, epsilon: float = 1e-8)`
  - RMSprop optimizer
  - Divides learning rate by running average of gradient magnitudes

- `class AdaGrad(learning_rate: float = 0.01, epsilon: float = 1e-8)`
  - Adaptive Gradient Algorithm
  - Adapts learning rate based on historical gradients

#### Learning Rate Schedulers
- `class LRScheduler(ABC)`
  - Base class for learning rate schedulers

- `class StepLR(optimizer: Optimizer, step_size: int, gamma: float = 0.1)`
  - Decay learning rate by gamma every step_size epochs

- `class ExponentialLR(optimizer: Optimizer, gamma: float = 0.95)`
  - Exponential decay

- `class CosineAnnealingLR(optimizer: Optimizer, T_max: int, eta_min: float = 0.0)`
  - Cosine annealing schedule
  - Smoothly decreases LR following cosine curve

- `class WarmupScheduler(optimizer: Optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.0)`
  - Linear warmup followed by decay

#### Regularization
- `l1_regularization(weights: np.ndarray, lambda_: float) -> Tuple[float, np.ndarray]`
  - L1 (Lasso) regularization
  - Penalty: λ × Σ|wᵢ|, Gradient: λ × sign(w)
  - Effect: Encourages sparsity (many weights become exactly 0)

- `l2_regularization(weights: np.ndarray, lambda_: float) -> Tuple[float, np.ndarray]`
  - L2 (Ridge) regularization
  - Penalty: λ × Σwᵢ², Gradient: 2λ × w
  - Effect: Shrinks weights towards zero (but rarely exactly 0)

- `elastic_net_regularization(weights: np.ndarray, lambda_: float, l1_ratio: float = 0.5) -> Tuple[float, np.ndarray]`
  - Elastic Net regularization (L1 + L2)
  - Combines benefits of L1 (sparsity) and L2 (stability)

#### Training Utilities
- `gradient_descent_train(X: np.ndarray, y: np.ndarray, loss_fn: Callable, initial_params: np.ndarray, optimizer: Optimizer, epochs: int = 100, batch_size: Optional[int] = None, regularization: Optional[Callable] = None, reg_lambda: float = 0.01, verbose: bool = True) -> Tuple[np.ndarray, List[float]]`
  - Generic gradient descent training loop

- `numerical_gradient(f: LossFunc, x: np.ndarray, epsilon: float = 1e-5) -> np.ndarray`
  - Compute numerical gradient for gradient checking
  - Uses central difference: (f(x+ε) - f(x-ε)) / 2ε

- `gradient_check(analytical_grad: np.ndarray, numerical_grad: np.ndarray, threshold: float = 1e-5) -> bool`
  - Verify analytical gradient against numerical gradient

### Probability and Statistics (`src.core.probability`)

#### Distributions
- `class Distribution`
  - Base class for probability distributions

- `class Gaussian(mean: float = 0.0, std: float = 1.0)`
  - Gaussian (Normal) Distribution
  - Mathematical Definition: f(x) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))

- `class MultivariateGaussian(mean: Array, cov: Array)`
  - Multivariate Gaussian Distribution
  - Mathematical Definition: f(x) = (2π)^(-k/2) |Σ|^(-1/2) × exp(-0.5 × (x-μ)ᵀ Σ⁻¹ (x-μ))

- `class Bernoulli(p: float = 0.5)`
  - Bernoulli Distribution
  - Mathematical Definition: P(X = 1) = p, P(X = 0) = 1 - p

- `class Categorical(probs: Array)`
  - Categorical Distribution (generalized Bernoulli)
  - Mathematical Definition: P(X = k) = p_k for k = 1, ..., K

- `class Uniform(low: float = 0.0, high: float = 1.0)`
  - Uniform Distribution
  - Mathematical Definition: f(x) = 1/(b-a) for a ≤ x ≤ b

- `class Exponential(rate: float = 1.0)`
  - Exponential Distribution
  - Mathematical Definition: f(x) = λ × exp(-λx) for x ≥ 0

#### Sampling Algorithms
- `rejection_sampling(target_pdf: Callable, proposal_sampler: Callable, proposal_pdf: Callable, M: float, n_samples: int) -> np.ndarray`
  - Rejection Sampling Algorithm

- `importance_sampling(target_pdf: Callable, proposal_sampler: Callable, proposal_pdf: Callable, f: Callable, n_samples: int) -> Tuple[float, float]`
  - Importance Sampling for Expectation Estimation

- `metropolis_hastings(target_log_pdf: Callable, proposal_sampler: Callable, initial: float, n_samples: int, burn_in: int = 1000) -> np.ndarray`
  - Metropolis-Hastings MCMC Sampler

#### Information Theory
- `entropy(p: Array, base: float = 2) -> float`
  - Shannon Entropy
  - Mathematical Definition: H(X) = -Σ p(x) × log(p(x))

- `cross_entropy(p: Array, q: Array) -> float`
  - Cross Entropy
  - Mathematical Definition: H(p, q) = -Σ p(x) × log(q(x))

- `kl_divergence(p: Array, q: Array) -> float`
  - Kullback-Leibler Divergence
  - Mathematical Definition: D_KL(P || Q) = Σ p(x) × log(p(x) / q(x))

- `js_divergence(p: Array, q: Array) -> float`
  - Jensen-Shannon Divergence
  - Mathematical Definition: D_JS(P || Q) = 0.5 × D_KL(P || M) + 0.5 × D_KL(Q || M)

- `mutual_information(joint_probs: Array) -> float`
  - Mutual Information
  - Mathematical Definition: I(X; Y) = Σₓ Σᵧ p(x,y) × log(p(x,y) / (p(x) × p(y)))

#### Bayesian Utilities
- `bayes_theorem(prior: float, likelihood: float, evidence: float) -> float`
  - Bayes' Theorem
  - Mathematical Definition: P(A|B) = P(B|A) × P(A) / P(B)

- `posterior_update(prior: Array, likelihood: Array) -> np.ndarray`
  - Bayesian posterior update for discrete distributions

- `beta_binomial_posterior(alpha_prior: float, beta_prior: float, successes: int, failures: int) -> Tuple[float, float]`
  - Beta-Binomial Conjugate Update

- `gaussian_posterior(prior_mean: float, prior_var: float, data_mean: float, data_var: float, n: int) -> Tuple[float, float]`
  - Gaussian-Gaussian Conjugate Update

#### Utility Functions
- `log_sum_exp(x: Array) -> float`
  - Numerically stable log-sum-exp
  - Computes log(Σ exp(xᵢ)) without overflow

- `normalize_log_probs(log_probs: Array) -> np.ndarray`
  - Convert log probabilities to normalized probabilities

## Classical Machine Learning (`src.ml.classical`)

#### Base Classes
- `class BaseEstimator(ABC)`
  - Abstract base class for all estimators

- `class BaseClassifier(BaseEstimator)`
  - Base class for classifiers

- `class BaseRegressor(BaseEstimator)`
  - Base class for regressors

#### Linear Regression
- `class LinearRegressionScratch(method: str = 'closed_form', learning_rate: float = 0.01, n_iterations: int = 1000, regularization: Optional[str] = None, reg_lambda: float = 0.01)`
  - Linear Regression from scratch
  - Model: ŷ = Xw + b
  - Loss: MSE = (1/n) Σ(yᵢ - ŷᵢ)²

  - `fit(X: np.ndarray, y: np.ndarray) -> LinearRegressionScratch`
    - Fit linear regression model

  - `predict(X: np.ndarray) -> np.ndarray`
    - Predict target values

#### Logistic Regression
- `class LogisticRegressionScratch(learning_rate: float = 0.01, n_iterations: int = 1000, regularization: Optional[str] = 'l2', reg_lambda: float = 0.01, multiclass: str = 'ovr')`
  - Logistic Regression from scratch
  - Model: P(y=1|x) = σ(xᵀw + b) = 1 / (1 + e^{-(xᵀw + b)})

  - `fit(X: np.ndarray, y: np.ndarray) -> LogisticRegressionScratch`
    - Fit logistic regression model

  - `predict(X: np.ndarray) -> np.ndarray`
    - Predict class labels

  - `predict_proba(X: np.ndarray) -> np.ndarray`
    - Predict class probabilities

#### K-Nearest Neighbors
- `class KNNScratch(k: int = 5, metric: str = 'euclidean', weights: str = 'uniform')`
  - K-Nearest Neighbors from scratch

  - `fit(X: np.ndarray, y: np.ndarray) -> KNNScratch`
    - Store training data (lazy learning)

  - `predict(X: np.ndarray) -> np.ndarray`
    - Predict class labels for X

#### Decision Trees
- `class DecisionTreeScratch(max_depth: int = 10, min_samples_split: int = 2, criterion: str = 'gini')`
  - Decision Tree Classifier from scratch

  - `fit(X: np.ndarray, y: np.ndarray) -> DecisionTreeScratch`
    - Build decision tree

  - `predict(X: np.ndarray) -> np.ndarray`
    - Predict class labels

#### Random Forest
- `class RandomForestScratch(n_estimators: int = 100, max_depth: int = 10, min_samples_split: int = 2, max_features: Union[str, int] = 'sqrt', bootstrap: bool = True)`
  - Random Forest Classifier from scratch

  - `fit(X: np.ndarray, y: np.ndarray) -> RandomForestScratch`
    - Train random forest

  - `predict(X: np.ndarray) -> np.ndarray`
    - Predict by majority vote

#### Naive Bayes
- `class GaussianNBScratch`
  - Gaussian Naive Bayes from scratch
  - Assumption: Features are independent and normally distributed

  - `fit(X: np.ndarray, y: np.ndarray) -> GaussianNBScratch`
    - Compute class priors and Gaussian parameters

  - `predict(X: np.ndarray) -> np.ndarray`
    - Predict class labels

  - `predict_log_proba(X: np.ndarray) -> np.ndarray`
    - Predict log probabilities

## Deep Learning (`src.ml.deep_learning`)

#### Layer Abstractions
- `class Layer(ABC)`
  - Abstract base class for neural network layers

- `class Dense(input_size: int, output_size: int, weight_init: str = 'xavier')`
  - Fully Connected (Dense) Layer
  - Forward: y = Wx + b
  - Backward: ∂L/∂W = ∂L/∂y × xᵀ, ∂L/∂b = ∂L/∂y, ∂L/∂x = Wᵀ × ∂L/∂y

  - `forward(input_data: np.ndarray, training: bool = True) -> np.ndarray`
    - Forward pass

  - `backward(output_gradient: np.ndarray, learning_rate: float) -> np.ndarray`
    - Backward pass

- `class Activation(activation: str = 'relu', alpha: float = 0.01)`
  - Activation Layer
  - Applies element-wise activation function

  - `forward(input_data: np.ndarray, training: bool = True) -> np.ndarray`
    - Forward pass

  - `backward(output_gradient: np.ndarray, learning_rate: float) -> np.ndarray`
    - Backward pass

- `class Dropout(rate: float = 0.5)`
  - Dropout regularization layer
  - Randomly zeros out units during training to prevent overfitting

  - `forward(input_data: np.ndarray, training: bool = True) -> np.ndarray`
    - Forward pass

  - `backward(output_gradient: np.ndarray, learning_rate: float) -> np.ndarray`
    - Backward pass

- `class BatchNormalization(n_features: int, momentum: float = 0.9, epsilon: float = 1e-5)`
  - Batch Normalization layer
  - Normalizes activations to have zero mean and unit variance

  - `forward(input_data: np.ndarray, training: bool = True) -> np.ndarray`
    - Forward pass

  - `backward(output_gradient: np.ndarray, learning_rate: float) -> np.ndarray`
    - Backward pass

#### Loss Functions
- `class Loss(ABC)`
  - Abstract base class for loss functions

- `class MSELoss`
  - Mean Squared Error Loss
  - L = (1/n) Σ(ŷ - y)²
  - ∂L/∂ŷ = (2/n)(ŷ - y)

  - `forward(y_pred: np.ndarray, y_true: np.ndarray) -> float`
    - Compute loss

  - `backward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray`
    - Compute gradient of loss w.r.t. predictions

- `class CrossEntropyLoss(from_logits: bool = False)`
  - Cross-Entropy Loss (for classification)
  - With softmax output, gradient simplifies to: ∂L/∂z = p - y

  - `forward(y_pred: np.ndarray, y_true: np.ndarray) -> float`
    - Compute loss

  - `backward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray`
    - Compute gradient of loss w.r.t. predictions

- `class BinaryCrossEntropyLoss`
  - Binary Cross-Entropy Loss
  - L = -[y log(σ(z)) + (1-y) log(1-σ(z))]
  - ∂L/∂z = σ(z) - y

  - `forward(y_pred: np.ndarray, y_true: np.ndarray) -> float`
    - Compute loss

  - `backward(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray`
    - Compute gradient of loss w.r.t. predictions

#### Neural Network Model
- `class NeuralNetwork`
  - Sequential Neural Network model
  - Stacks layers in sequence, handles forward/backward propagation

  - `add(layer: Layer)`
    - Add a layer to the model

  - `compile(loss: Loss, learning_rate: float = 0.001)`
    - Configure the model for training

  - `fit(X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32, validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None, verbose: bool = True) -> dict`
    - Train the model

  - `predict(X: np.ndarray) -> np.ndarray`
    - Make predictions

  - `predict_proba(X: np.ndarray) -> np.ndarray`
    - Get prediction probabilities

  - `evaluate(X: np.ndarray, y: np.ndarray) -> Tuple[float, float]`
    - Evaluate model on test data

  - `summary()`
    - Print model summary

## LLM Engineering (`src.llm`)

### Attention Mechanisms (`src.llm.attention`)

- `scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]`
  - Scaled Dot-Product Attention mechanism
  - Computes attention weights and outputs using: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V

- `class MultiHeadAttention(d_model: int, num_heads: int)`
  - Multi-Head Attention mechanism
  - Concatenates multiple attention heads to capture different aspects of the input

  - `forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor`
    - Forward pass of Multi-Head Attention

- `class FeedForwardNetwork(d_model: int, d_ff: int, activation: str = 'relu')`
  - Feed-Forward Network component of Transformer
  - Consists of two linear transformations with a ReLU activation in between

  - `forward(x: torch.Tensor) -> torch.Tensor`
    - Forward pass of Feed-Forward Network

- `class LayerNorm(d_model: int, eps: float = 1e-6)`
  - Layer Normalization
  - Normalizes across the feature dimension to stabilize training

  - `forward(x: torch.Tensor) -> torch.Tensor`
    - Forward pass of Layer Normalization

- `class TransformerBlock(d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1)`
  - Single Transformer Block (Encoder)
  - Combines Multi-Head Attention, Feed-Forward Network, and residual connections with Layer Normalization

  - `forward(x: torch.Tensor) -> torch.Tensor`
    - Forward pass of Transformer Block

### RAG Pipeline (`src.llm.rag`)

- `class Document(content: str, id: str = "", metadata: Dict[str, Any] = field(default_factory=dict), embedding: Optional[np.ndarray] = None)`
  - Document with content and metadata

- `class RetrievalResult(document: Document, score: float, rank: int)`
  - Result from retrieval

- `class TextChunker(chunk_size: int = 512, overlap: int = 50, strategy: str = 'fixed')`
  - Text chunking for RAG pipelines

  - `chunk(text: str, metadata: Dict = None) -> List[Document]`
    - Split text into chunks

- `class EmbeddingModel(model_fn: Optional[Callable] = None, dim: int = 384)`
  - Embedding model interface

  - `embed(texts: List[str]) -> np.ndarray`
    - Embed list of texts

  - `embed_query(query: str) -> np.ndarray`
    - Embed a single query

  - `embed_documents(documents: List[Document]) -> List[Document]`
    - Embed documents and attach embeddings

- `class Retriever(embedding_model: EmbeddingModel, strategy: str = 'dense')`
  - Document retriever with multiple strategies

  - `add_documents(documents: List[Document])`
    - Add documents to the retriever

  - `retrieve(query: str, k: int = 5) -> List[RetrievalResult]`
    - Retrieve top-k documents

- `class Reranker(rerank_fn: Optional[Callable] = None)`
  - Cross-encoder reranker

  - `rerank(query: str, results: List[RetrievalResult], top_k: int = 3) -> List[RetrievalResult]`
    - Rerank results

- `class ContextAssembler(max_tokens: int = 2048, strategy: str = 'stuffing')`
  - Assemble retrieved documents into LLM context

  - `assemble(query: str, results: List[RetrievalResult]) -> str`
    - Assemble context from retrieval results

- `class RAGPipeline(embedding_model: Optional[EmbeddingModel] = None, retriever: Optional[Retriever] = None, reranker: Optional[Reranker] = None, context_assembler: Optional[ContextAssembler] = None, llm_fn: Optional[Callable] = None)`
  - Complete RAG pipeline

  - `add_documents(documents: List[Document])`
    - Add documents to the pipeline

  - `query(query: str, k: int = 5, return_sources: bool = True) -> Dict[str, Any]`
    - Query the RAG pipeline

### Fine-Tuning Techniques (`src.llm.fine_tuning`)

- `class LoRALayer(in_features: int, out_features: int, r: int = 8, alpha: float = 16.0, dropout: float = 0.0)`
  - LoRA adaptation layer
  - Original: y = Wx, With LoRA: y = Wx + (B @ A)x × (α/r)

  - `forward(x: np.ndarray, training: bool = True) -> np.ndarray`
    - Forward pass for LoRA delta: ΔWx = BAx * scaling

  - `merge_weights(base_weights: np.ndarray) -> np.ndarray`
    - Merge LoRA weights into base weights for inference

- `class LinearWithLoRA(in_features: int, out_features: int, r: int = 8)`
  - Linear layer equipped with LoRA
  - Simulates a frozen base layer + trainable LoRA adapter

  - `forward(x: np.ndarray, training: bool = True) -> np.ndarray`
    - Forward pass

  - `merge_lora()`
    - Permanent merge for deployment

- `class AdapterLayer(d_model: int, bottleneck: int = 64)`
  - Bottleneck Adapter module for PEFT
  - Usually inserted after FFN or Attention blocks

  - `forward(x: np.ndarray) -> np.ndarray`
    - Forward pass

- `quantize_nf4(weights: np.ndarray) -> Tuple[np.ndarray, float]`
  - Simulate 4-bit NormalFloat quantization (NF4)

- `dequantize_nf4(quantized: np.ndarray, scale: float) -> np.ndarray`
  - Dequantize 4-bit weights back to float

## Production Systems (`src.production`)

### Caching (`src.production.caching`)

- `class LRUCache(max_size: int = 1000, default_ttl: Optional[int] = None)`
  - Least Recently Used (LRU) Cache
  - Thread-safe implementation with eviction policy

  - `get(key: str) -> Optional[T]`
    - Get value from cache

  - `set(key: str, value: T, ttl: Optional[int] = None) -> bool`
    - Set value in cache with optional TTL (seconds)

  - `delete(key: str) -> bool`
    - Delete key from cache

  - `exists(key: str) -> bool`
    - Check if key exists in cache

- `class RedisCache(url: Optional[str] = None, prefix: str = "cache:", default_ttl: int = 3600, serializer: str = "json")`
  - Redis-backed distributed cache

  - `get(key: str) -> Optional[T]`
    - Get value from Redis

  - `set(key: str, value: T, ttl: Optional[int] = None) -> bool`
    - Set value in Redis

  - `delete(key: str) -> bool`
    - Delete key from Redis

- `class EmbeddingCache(cache_backend: Optional[CacheInterface] = None, hash_algorithm: str = "sha256")`
  - Specialized cache for embeddings with content hashing

  - `get(text: str) -> Optional[List[float]]`
    - Get embedding for text

  - `set(text: str, embedding: List[float], ttl: Optional[int] = None) -> bool`
    - Set embedding for text

  - `get_or_compute(texts: List[str], embed_fn: Callable[[List[str]], List[List[float]]], ttl: Optional[int] = None) -> List[List[float]]`
    - Get embeddings from cache or compute and cache

### Monitoring (`src.production.monitoring`)

- `class DriftDetector(method: str = 'ks', threshold: float = 0.05, feature_names: Optional[List[str]] = None)`
  - Unified drift detection for multiple features
  - Monitors data drift and concept drift in production

  - `set_reference(data: np.ndarray)`
    - Set reference (training) data distribution

  - `detect_drift(current_data: np.ndarray) -> List[DriftResult]`
    - Detect drift in current data compared to reference

- `class PerformanceMonitor(window_size: int = 1000)`
  - Monitor model performance over time

  - `record_prediction(y_true: Optional[float] = None, y_pred: Optional[float] = None, latency_ms: Optional[float] = None, error: bool = False)`
    - Record a prediction for monitoring

  - `get_metrics(task: str = 'classification') -> PerformanceMetrics`
    - Compute current performance metrics

- `class AlertManager`
  - Simple alerting system for monitoring

  - `add_handler(handler: Callable)`
    - Add an alert handler function

  - `alert(severity: str, message: str, metadata: Optional[Dict[str, Any]] = None)`
    - Send an alert

### Deployment (`src.production.deployment`)

- `class ModelSerializer`
  - Unified model serialization across formats
  - Supports pickle, joblib, ONNX, PyTorch, and TensorFlow

  - `save(model: Any, path: str, format: ModelFormat = ModelFormat.PICKLE, metadata: Optional[ModelMetadata] = None, compress: bool = True) -> str`
    - Save model to file

  - `load(path: str, format: Optional[ModelFormat] = None, model_class: Optional[type] = None) -> Any`
    - Load model from file

  - `get_metadata(path: str) -> Optional[ModelMetadata]`
    - Load model metadata if available

- `class ModelVersionManager(base_path: str)`
  - Manages multiple model versions for safe deployments
  - Supports blue-green deployments, canary releases, quick rollbacks

  - `register(version: str, model: Any, metrics: Optional[Dict[str, float]] = None, format: ModelFormat = ModelFormat.JOBLIB) -> str`
    - Register a new model version

  - `activate(version: str) -> bool`
    - Activate a model version (blue-green switch)

  - `rollback() -> bool`
    - Rollback to previous version

  - `load_active() -> Optional[Any]`
    - Load the currently active model

- `class HealthChecker`
  - Health check system for ML services
  - Supports multiple check types: Liveness, Readiness, Dependency

  - `add_check(name: str, check_fn: Callable[[], Union[bool, HealthCheckResult]], critical: bool = True)`
    - Add a health check

  - `run_all() -> Dict[str, Any]`
    - Run all health checks

- `class GracefulShutdown(timeout: int = 30)`
  - Handles graceful shutdown for ML services
  - Features: Catches SIGTERM and SIGINT, waits for in-flight requests, runs cleanup callbacks

  - `add_cleanup(callback: Callable[[], None])`
    - Add a cleanup callback to run on shutdown

  - `request_started()`
    - Track that a request has started

  - `request_finished()`
    - Track that a request has finished

## API Module (`src.production.api`)

- `class PredictionRequest(features: List[float], model_name: Optional[str] = "default")`
  - Input schema for prediction endpoint
  - Pydantic provides automatic validation, serialization, and OpenAPI documentation

- `class PredictionResponse(prediction: Union[float, int, List[float]], confidence: Optional[float] = None, model_name: str, latency_ms: float, timestamp: str)`
  - Output schema for predictions

- `class ChatRequest(messages: List[ChatMessage], temperature: float = 0.7, max_tokens: int = 512, stream: bool = False)`
  - Chat completion request

- `class ModelCache`
  - Singleton model cache for efficient inference
  - Loads models once and reuses them across requests

  - `load_model(name: str, model: Any)`
    - Load a model into cache

  - `get_model(name: str) -> Optional[Any]`
    - Get a model from cache

- `create_app(title: str = "ML Model API", version: str = "1.0.0", debug: bool = False) -> FastAPI`
  - Factory function to create FastAPI application

- `register_routes(app: FastAPI)`
  - Register all API routes

## Utility Functions

- `load_sklearn_model(path: str)`
  - Load a scikit-learn model from disk

- `load_pytorch_model(path: str, model_class)`
  - Load a PyTorch model from disk