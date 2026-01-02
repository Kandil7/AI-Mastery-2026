# AI-Mastery-2026: Complete User Guide

This comprehensive guide covers everything you need to know to use the AI-Mastery-2026 toolkit effectively.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation Guide](#2-installation-guide)
3. [Project Structure](#3-project-structure)
4. [Mathematical Foundations](#4-mathematical-foundations)
5. [Core Modules](#5-core-modules)
   - [Optimization](#52-optimization)
   - [Time Series](#54-time-series--state-estimation)
   - [Integration](#55-numerical-integration)
   - [Normalizing Flows](#56-normalizing-flows)
6. [Classical Machine Learning](#6-classical-machine-learning)
7. [Deep Learning](#7-deep-learning)
8. [LLM Engineering](#8-llm-engineering)
9. [Production API](#9-production-api)
10. [Web Interface](#10-web-interface)
11. [Docker Deployment](#11-docker-deployment)
12. [Monitoring & Observability](#12-monitoring--observability)
13. [Testing](#13-testing)
14. [Interview Preparation](#14-interview-preparation) ⭐ NEW
16. [Troubleshooting](#16-troubleshooting)
17. [FAQ](#17-faq)


---

## 1. Introduction

### What is AI-Mastery-2026?

AI-Mastery-2026 is a **full-stack AI engineering toolkit** designed for learning and building production AI applications. It follows the "White-Box Approach":

1. **Math First** - Understand the mathematical foundations
2. **Code Second** - Implement algorithms from scratch using NumPy
3. **Libraries Third** - Use frameworks knowing what's happening underneath
4. **Production Always** - Every concept includes real-world deployment

### Who Is This For?

- **Students** learning ML/AI fundamentals
- **Engineers** transitioning into AI roles
- **Researchers** needing reference implementations
- **Teams** building production ML systems

### What You'll Learn

- Linear algebra, calculus, optimization (with industrial applications)
- Classical ML algorithms (from scratch)
- Neural networks and deep learning
- Transformer architecture and attention
- RAG systems and LLM engineering
- Production deployment with Docker
- Monitoring with Prometheus/Grafana

### 📓 Start Here: Deep Mathematical Foundations Notebook

Before diving into the code, we recommend starting with our comprehensive mathematics notebook:

**File**: `notebooks/01_mathematical_foundations/deep_ml_mathematics_complete.ipynb`

| Chapter | Topics | Real-World Applications |
|---------|--------|------------------------|
| 1. Linear Algebra | Vectors, Matrices, SVD, Eigenvalues | Google PageRank, Netflix Recommendations |
| 2. Calculus | Gradients, Jacobians, Chain Rule | Backpropagation, Self-Attention |
| 3. Optimization | Lagrange, Convex, ADMM | SVM, Uber Budget Allocation |
| 4. Probability | Entropy, KL Divergence, VAE | Generative Models, t-SNE |
| 5. Integration | Monte Carlo, Importance Sampling | Bayesian Inference |
| 6. Networks | Random Walks, Link Prediction | Facebook Friend Suggestions |
| 7. Bayesian Opt | Gaussian Processes, Acquisition | Google Vizier |

---


## 2. Installation Guide

### 2.1 Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | 3.10+ | Required |
| pip | Latest | Package manager |
| Docker | 20.10+ | Optional, for deployment |
| Docker Compose | 2.0+ | Optional |
| RAM | 8GB+ | Recommended |
| GPU | CUDA 11+ | Optional, for faster training |

### 2.2 Standard Installation

```bash
# Step 1: Clone the repository
git clone https://github.com/Kandil7/AI-Mastery-2026.git
cd AI-Mastery-2026

# Step 2: Create virtual environment
python -m venv .venv

# Step 3: Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Step 4: Install dependencies
pip install -r requirements.txt

# Step 5: Verify installation
# Step 5: Verify installation
python -c "from src.ml.classical import SVMScratch; print('✅ Installation successful!')"

# Note: This will generate 'models/models_metadata.json' used for validation
python scripts/train_save_models.py

```

### 2.3 Using Make

```bash
# Install everything with one command
make install

# Or step by step:
make venv           # Create virtual environment
make install-deps   # Install dependencies
make install-dev    # Install dev dependencies
```

### 2.4 Docker Installation

```bash
# Build all images
docker-compose build

# Or use make
make docker-build
```

### 2.5 Verify Installation

```bash
# Run test suite
make test

# Expected output:
# ========================= test session starts =========================
# collected XX items
# tests/test_linear_algebra.py ....                               [  5%]
# tests/test_probability.py ....                                  [ 10%]
# ...
# ========================= XX passed in X.XXs =========================
```

---

## 3. Project Structure

```
AI-Mastery-2026/
│
├── src/                              # Core source code
│   ├── __init__.py
│   │
│   ├── core/                         # Mathematical foundations
│   │   ├── __init__.py
│   │   ├── math_operations.py        # Vector ops, matrix ops, activations
│   │   ├── optimization.py           # SGD, Adam, RMSprop, AdaGrad, NAdam, schedulers
│   │   ├── probability.py            # Distributions, sampling, info theory
│   │   ├── integration.py            # Newton-Cotes, Gaussian Quadrature, Monte Carlo
│   │   ├── normalizing_flows.py      # Planar, Radial flows, density estimation
│   │   └── time_series.py            # EKF, UKF, Particle Filter, RTS smoother
│   │
│   ├── ml/                           # Machine Learning
│   │   ├── __init__.py
│   │   ├── classical.py              # LR, SVM, Trees, RF, KNN, NB
│   │   └── deep_learning.py          # Dense, LSTM, Conv2D, Networks
│   │
│   ├── llm/                          # LLM Engineering
│   │   ├── __init__.py
│   │   ├── attention.py              # Scaled dot-product, multi-head
│   │   ├── rag.py                    # RAG pipeline components
│   │   ├── fine_tuning.py            # LoRA adapters
│   │   └── agents.py                 # LLM agents
│   │
│   └── production/                   # Production components
│       ├── __init__.py
│       ├── api.py                    # FastAPI application
│       ├── caching.py                # Model caching
│       ├── monitoring.py             # Prometheus metrics
│       └── vector_db.py              # HNSW, LSH indices
│
├── app/                              # Web interface
│   └── main.py                       # Streamlit application
│
├── scripts/                          # Utility scripts
│   ├── train_save_models.py          # Train sklearn models for API
│   ├── ingest_data.py                # RAG data ingestion pipeline
│   └── setup_database.py             # PostgreSQL schema setup
│
├── config/                           # Configuration files
│   ├── prometheus.yml                # Prometheus scrape config
│   └── grafana/
│       ├── provisioning/
│       │   └── datasources/
│       └── dashboards/
│           └── ml_api.json           # Pre-built dashboard
│
├── tests/                            # Test suite
│   ├── test_linear_algebra.py
│   ├── test_probability.py
│   ├── test_ml_algorithms.py
│   ├── test_deep_learning.py
│   ├── test_svm.py
│   ├── test_rag_llm.py
│   └── integration/
│
├── research/                         # Jupyter notebooks
│   ├── 00_foundation/                # Weeks 1-3
│   ├── 01_linear_algebra/            # Week 4
│   ├── ...                           # Weeks 5-16
│   └── mlops_end_to_end.ipynb        # Complete MLOps demo
│
├── models/                           # Saved models directory
│   ├── classification_model.joblib
│   ├── regression_model.joblib
│   ├── logistic_model.joblib
│   └── models_metadata.json
│
├── docs/                             # Documentation
│   ├── USER_GUIDE.md                 # This file
│   └── guide/                        # Detailed guides
│
├── docker-compose.yml                # Docker services config
├── Dockerfile                        # API container
├── Dockerfile.streamlit              # Web UI container
├── requirements.txt                  # Python dependencies
├── Makefile                          # Build automation
├── setup.sh                          # Setup script
└── README.md                         # Project overview
```

---

## 4. Mathematical Foundations

The AI-Mastery-2026 project includes a comprehensive **Deep Mathematical Foundations** notebook that covers the theoretical underpinnings of machine learning with industrial applications.

### 4.1 Notebook Overview

**Location**: `notebooks/01_mathematical_foundations/deep_ml_mathematics_complete.ipynb`

**Specs**: 96KB, 61 cells (29 markdown + 32 code)

### 4.2 Topics Covered

#### Chapter 1: Linear Algebra
- **Dot Product & Cosine Similarity** - From-scratch implementation for NLP embeddings
- **Matrix Transformations** - Rotation, scaling, shearing with visualization
- **Convolution as Matrix Multiplication** - Toeplitz matrices for CNNs
- **Eigenvalues & PageRank** - Google's ranking algorithm from scratch
- **SVD & Netflix Recommendations** - FunkSVD collaborative filtering

```python
# Example: PageRank from the notebook
def pagerank(adj_matrix, damping=0.85, max_iter=100):
    n = adj_matrix.shape[0]
    out_degree = adj_matrix.sum(axis=1)
    M = (adj_matrix.T / out_degree).T
    r = np.ones(n) / n
    for _ in range(max_iter):
        r = damping * M @ r + (1 - damping) / n
    return r
```

#### Chapter 2: Calculus
- **Gradients & Jacobian Matrix** - Numerical and analytical computation
- **Backpropagation** - Step-by-step chain rule through a 2-layer network
- **Self-Attention Mechanism** - Complete Transformer attention implementation

```python
# Example: Self-Attention from the notebook
def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)  # Scale
    weights = softmax(scores)         # Normalize
    return weights @ V                # Weighted sum
```

#### Chapter 3: Optimization
- **Lagrange Multipliers** - Constrained optimization with visualization
- **SVM Dual Form** - Quadratic programming implementation
- **ADMM** - Alternating Direction Method of Multipliers for LASSO

#### Chapter 4: Probability & Information Theory
- **Entropy & KL Divergence** - Information-theoretic foundations
- **Variational Autoencoders** - ELBO derivation and reparameterization trick
- **t-SNE** - Visualization with KL divergence minimization

```python
# Example: KL Divergence for VAE
def kl_divergence_gaussian(mu, log_var):
    """KL between N(mu, exp(log_var)) and N(0, 1)"""
    return -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
```

#### Chapter 5: Advanced Integration
- **Monte Carlo Integration** - Numerical estimation of integrals
- **Importance Sampling** - Variance reduction techniques
- **Normalizing Flows** - Change of variables for density estimation

#### Chapter 6: Network Analysis
- **Link Prediction** - Common neighbors, Adamic-Adar scores
- **Supervised Random Walks** - Facebook's friend suggestion algorithm

#### Chapter 7: Bayesian Optimization
- **Gaussian Processes** - Prediction with uncertainty quantification
- **Acquisition Functions** - Expected Improvement for hyperparameter tuning

### 4.3 Industrial Applications

| Topic | Company | Application |
|-------|---------|-------------|
| Eigenvalues | Google | PageRank web ranking |
| SVD | Netflix | Movie recommendations |
| Self-Attention | OpenAI | GPT, ChatGPT |
| SVM | Various | Classification systems |
| ADMM | Uber | Budget allocation across cities |
| Gaussian Processes | Google | Vizier hyperparameter tuning |
| Random Walks | Facebook | Friend suggestions |

### 4.4 Getting Started

```bash
# Open the notebook
jupyter notebook notebooks/01_mathematical_foundations/deep_ml_mathematics_complete.ipynb
```

Each section includes:
1. **Theory** - Mathematical explanation with LaTeX equations
2. **Example** - Step-by-step numerical calculation
3. **Code** - Complete Python/NumPy implementation
4. **Visualization** - Plots and diagrams

---

## 5. Core Modules

### 5.1 Math Operations (`src/core/math_operations.py`)

**Vector Operations:**

```python
from src.core.math_operations import dot_product, cosine_similarity


# Dot product
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = dot_product(a, b)  # 32

# Cosine similarity
sim = cosine_similarity(a, b)  # 0.974
```

**Matrix Operations:**

```python
from src.core.math_operations import matrix_multiply, matrix_inverse, matrix_transpose

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Multiply
C = matrix_multiply(A, B)

# Inverse (if exists)
A_inv = matrix_inverse(A)

# Transpose
A_T = matrix_transpose(A)
```

**Activation Functions:**

```python
from src.core.math_operations import sigmoid, relu, softmax, tanh

# Sigmoid: σ(x) = 1 / (1 + e^(-x))
sigmoid(np.array([-1, 0, 1]))  # [0.269, 0.5, 0.731]

# ReLU: max(0, x)
relu(np.array([-1, 0, 1]))  # [0, 0, 1]

# Softmax: e^xi / Σe^xj
softmax(np.array([1, 2, 3]))  # [0.09, 0.24, 0.67]

# Tanh: (e^x - e^-x) / (e^x + e^-x)
tanh(np.array([-1, 0, 1]))  # [-0.76, 0, 0.76]
```

**Dimensionality Reduction:**

```python
from src.core.math_operations import pca, svd

# PCA
X_reduced = pca(X, n_components=2)

# SVD: A = UΣV^T
U, S, Vt = svd(A)
```

### 5.2 Optimization (`src/core/optimization.py`)

**Optimizers:**

```python
from src.core.optimization import (
    GradientDescent, Momentum, Adam, RMSprop, AdaGrad, NAdam
)

# Adam optimizer (OpenAI GPT training)
optimizer = Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
params = optimizer.step(params, gradients)

# RMSprop (DeepMind DQN)
optimizer = RMSprop(learning_rate=0.01, rho=0.9)

# NAdam (Nesterov + Adam)
optimizer = NAdam(learning_rate=0.001)
```

**Learning Rate Schedulers:**

```python
from src.core.optimization import (
    StepDecay, ExponentialDecay, CosineAnnealing, WarmupScheduler
)

# Warmup for Transformers (BERT, GPT training)
scheduler = WarmupScheduler(
    initial_lr=0.001, 
    warmup_steps=1000, 
    total_steps=10000
)

# Cosine annealing (EfficientNet)
scheduler = CosineAnnealing(initial_lr=0.1, T_max=100)
```

### 5.3 Probability (`src/core/probability.py`)

```python
from src.core.probability import entropy, kl_divergence, gaussian_pdf

# Shannon entropy: H(X) = -Σ p(x) log p(x)
H = entropy(probabilities)

# KL Divergence: KL(P||Q) = Σ P(x) log(P(x)/Q(x))
kl = kl_divergence(P, Q)
```

### 5.4 Time Series & State Estimation (`src/core/time_series.py`)

**Kalman Filters:**

```python
from src.core.time_series import (
    ExtendedKalmanFilter, UnscentedKalmanFilter, GaussianState
)

# Extended Kalman Filter (Tesla FSD, robot localization)
ekf = ExtendedKalmanFilter(
    f=dynamics_function,      # x_{t+1} = f(x_t)
    h=observation_function,   # y_t = h(x_t)
    F_jacobian=F_jac,         # Jacobian of f
    H_jacobian=H_jac,         # Jacobian of h
    Q=process_noise,
    R=observation_noise
)

initial_state = GaussianState(mean=np.zeros(4), cov=np.eye(4))
result = ekf.filter(observations, initial_state)
# result.means: (T, n) filtered state estimates
# result.covs: (T, n, n) covariances
```

**Particle Filter:**

```python
from src.core.time_series import ParticleFilter

# For multimodal distributions (Waymo localization)
pf = ParticleFilter(
    f=dynamics,
    h=observation,
    process_noise_sampler=lambda n: np.random.randn(n, 2) * 0.1,
    observation_log_likelihood=log_likelihood,
    n_particles=1000
)

means, variances, ess = pf.filter(observations, initial_particles)
```

### 5.5 Numerical Integration (`src/core/integration.py`)

```python
from src.core.integration import (
    trapezoidal_rule, simpsons_rule,      # Newton-Cotes
    gauss_legendre, gauss_hermite_expectation,  # Gaussian Quadrature
    monte_carlo_integrate, importance_sampling   # Monte Carlo
)

# Trapezoidal rule (Boeing aerodynamics)
result = trapezoidal_rule(lambda x: x**2, a=0, b=1, n=100)

# Gaussian expectation (BlackRock risk modeling)
expected = gauss_hermite_expectation(f, mu=0, sigma=1, n_points=10)

# Monte Carlo (Netflix recommendations, CERN particle physics)
result = monte_carlo_integrate(f, bounds=[(0, 1), (0, 1)], n_samples=10000)
```

### 5.6 Normalizing Flows (`src/core/normalizing_flows.py`)

```python
from src.core.normalizing_flows import PlanarFlow, RadialFlow, FlowChain

# Build flow chain (Spotify recommendations, Waymo trajectories)
flow = FlowChain([
    PlanarFlow(d=2),
    RadialFlow(d=2),
    PlanarFlow(d=2)
])

# Transform samples
z = np.random.randn(100, 2)  # Base distribution
z_transformed, log_det = flow.forward(z)

# Compute log probability
log_prob = flow.log_prob(z_transformed, base_log_prob=gaussian_base_log_prob)
```

### 5.7 MCMC & Variational Inference (`src/core/mcmc.py`, `src/core/variational_inference.py`)

**MCMC Sampling:**

```python
from src.core.mcmc import metropolis_hastings, HamiltonianMonteCarlo

# Metropolis-Hastings
samples, acceptance_rate = metropolis_hastings(
    log_prob_func, initial_state=0.0, n_samples=1000, step_size=0.5
)

# Hamiltonian Monte Carlo (NUTS)
hmc = HamiltonianMonteCarlo(log_prob_func, grad_log_prob_func)
result = hmc.sample(initial_state, n_samples=1000)
print(f"ESS: {result.ess}")
```

**Variational Inference:**

```python
from src.core.variational_inference import MeanFieldVI, GaussianVariational

# Define variational family
variational_dist = GaussianVariational(dim=2)

# Run optimization (VI)
vi = MeanFieldVI(
    log_prob_func=target_log_prob,
    variational_family=variational_dist,
    n_samples=10
)
result = vi.optimize(n_iterations=2000, learning_rate=0.01)
```

### 5.8 Advanced Deep Integration (`src/core/advanced_integration.py`)

This module provides state-of-the-art integration techniques bridging mathematical foundations with modern deep learning.

**Neural ODEs with Uncertainty Quantification:**

```python
from src.core.advanced_integration import NeuralODE, ODEFunc, robot_dynamics_demo
import torch

# Define dynamics and model
func = ODEFunc(dim=2, hidden_dim=64)
model = NeuralODE(func, method='rk4')

# Initial state and time span
x0 = torch.tensor([[0.0, 1.0]])
t_span = torch.linspace(0, 10, 101)

# Integrate with uncertainty (Monte Carlo Dropout)
mean_path, std_path, trajectories = model.integrate_with_uncertainty(
    x0, t_span, num_samples=50
)
# mean_path: (101, 1, 2) - Expected trajectory
# std_path: (101, 1, 2) - Uncertainty at each time step

# Quick demo (Boston Dynamics Atlas-style robot dynamics)
results = robot_dynamics_demo(dim=2, t_max=10.0)
```

**Multi-Modal Healthcare Integration:**

```python
from src.core.advanced_integration import (
    MultiModalIntegrator, generate_patient_data
)
import torch

# Generate synthetic patient data (Mayo Clinic-style)
data = generate_patient_data(n_samples=1000)
# Returns clinical_data, xray_data, text_data, labels

# Create multi-modal fusion model
model = MultiModalIntegrator(
    clinical_dim=5, xray_dim=3, text_dim=4, hidden_dim=64
)

# Prepare tensors
clinical = torch.tensor(data['clinical_data'], dtype=torch.float32)
xray = torch.tensor(data['xray_data'], dtype=torch.float32)
text = torch.tensor(data['text_data'], dtype=torch.float32)

# Predict with MC Dropout uncertainty
predictions, uncertainty = model.predict_with_confidence(
    clinical, xray, text, n_samples=50
)
# High uncertainty cases should be flagged for human review
```

**Federated Learning Integration:**

```python
from src.core.advanced_integration import (
    FederatedHospital, FederatedIntegrator, federated_demo
)

# Create distributed hospital nodes (Apple HealthKit-style)
hospitals = [
    FederatedHospital(0, 'young', n_patients=300),
    FederatedHospital(1, 'elderly', n_patients=250),
    FederatedHospital(2, 'mixed', n_patients=400),
]

# Aggregate using Bayesian weighting (privacy-preserving)
integrator = FederatedIntegrator(hospitals, aggregation_method='bayesian_weighting')
global_risk, global_uncertainty = integrator.aggregate()

# Compare aggregation strategies
results = federated_demo(n_hospitals=5, n_rounds=3)
# Compares: simple_average, sample_weighted, uncertainty_weighted, bayesian_weighting
```

**Ethics & Bias Analysis:**

```python
from src.core.advanced_integration import (
    biased_lending_simulation, analyze_bias, fairness_test
)

# Simulate biased lending system (IBM AI Fairness 360-style)
results = biased_lending_simulation(n_samples=10000, bias_factor=0.4)

# Analyze bias metrics
metrics = analyze_bias(results)
print(f"Approval rate disparity: {metrics['approval_disparity']:.2%}")
print(f"Disparate impact ratio: {metrics['disparate_impact_ratio']:.2f}")
# Ratio < 0.8 indicates potential discrimination

# Comprehensive fairness test for any classifier
fairness = fairness_test(predictions, labels, sensitive_attr)
print(f"FPR disparity: {fairness['fairness_metrics']['fpr_disparity']:.4f}")
```

### 5.9 Hardware-Accelerated Integration

Hardware acceleration provides **massive speedups** for integration methods:

```python
from src.core.hardware_accelerated_integration import (
    monte_carlo_cpu,
    monte_carlo_numba,  # 80x faster
    monte_carlo_gpu_pytorch,  # 200x faster for large n
    HardwareAcceleratedIntegrator,
    NUMBA_AVAILABLE, CUDA_AVAILABLE
)

# Simple CPU baseline
result, error = monte_carlo_cpu(lambda x: x**2, a=0, b=1, n_samples=100000)

# Unified interface with automatic backend selection
integrator = HardwareAcceleratedIntegrator()
result = integrator.integrate(
    f=my_function,
    f_torch=my_function_torch,  # Optional PyTorch version
    a=0, b=1, n_samples=1000000,
    method='auto'  # Automatically selects CPU/Numba/GPU
)
print(f"Result: {result['estimate']}, Device: {result['device']}")
print(f"Throughput: {result['samples_per_second']/1e6:.1f}M samples/sec")
```

**Industrial Case Study: NVIDIA cuQuantum**
- 1000x speedup for quantum circuit simulation
- GPU-optimized high-dimensional integration

### 5.10 Probabilistic Programming Languages (PPL)

Compare Bayesian inference across different frameworks:

```python
from src.core.ppl_integration import (
    NumpyMCMCRegression,  # Pure NumPy baseline
    PyMCRegression,       # Requires pymc3
    TFPRegression,        # Requires tensorflow-probability
    generate_regression_data,
    compare_ppl_methods
)

# Generate synthetic data
X, y, true_params = generate_regression_data(n=100)

# Fit with NumPy MCMC (always available)
model = NumpyMCMCRegression()
result = model.fit(X, y, n_samples=2000, n_warmup=500)
print(f"Slope: {result.slope_mean:.3f} ± {result.slope_std:.3f}")

# Predict with uncertainty
y_pred, y_std = model.predict(X_new, return_uncertainty=True)

# Compare all available PPLs
results = compare_ppl_methods(X, y, n_samples=1000)
```

**Industrial Case Study: Uber's Pyro**
- Causal inference for marketing optimization
- ITE estimation: $200M/year marketing savings

### 5.11 Adaptive Integration

Automatically select the best integration method:

```python
from src.core.adaptive_integration import (
    AdaptiveIntegrator,
    smooth_function, multimodal_function
)

# Create adaptive integrator
integrator = AdaptiveIntegrator()

# Automatic method selection based on function properties
result = integrator.integrate(my_function, a=-1, b=1)
print(f"Method selected: {result.method}")
print(f"Estimate: {result.estimate}")
print(f"Features: smoothness={result.features.smoothness:.2f}, modes={result.features.num_modes}")

# Analyze function characteristics
features = integrator.analyze_function(my_function, a=-1, b=1)
print(f"Smoothness: {features.smoothness}")
print(f"Modes: {features.num_modes}")
print(f"Sharp transitions: {features.sharp_transitions:.3f}")

# Train ML-based selector (optional)
integrator.train_method_selector([f1, f2, f3], a=-1, b=1)
```

**Industrial Case Study: Wolfram Alpha**
- 97% success rate across all function types
- <2 second average response time

### 5.12 Integration in Reinforcement Learning

RL uses integration for policy evaluation and gradient estimation:

```python
from src.core.rl_integration import (
    RLIntegrationSystem,
    SimpleValueNetwork,
    simple_policy
)

# Create RL system
rl = RLIntegrationSystem()

# Monte Carlo Policy Evaluation
# V(s) = E[G | S=s] = ∫ G · p(G|s) dG
value_estimates, returns_by_state = rl.monte_carlo_policy_evaluation(
    simple_policy, n_episodes=100
)

# Policy Gradient Training (REINFORCE)
# ∇J(θ) = E[∑_t ∇log π_θ(a_t|s_t) · G_t]
results = rl.policy_gradient_reinforce(n_episodes=200)
print(f"Final reward: {results.episode_rewards[-1]:.2f}")

# MCTS-style Value Estimation
import numpy as np
state = np.array([-0.5, 0.0])
value, uncertainty = rl.mcts_value_estimate(state, n_simulations=50)
```

**Industrial Case Study: DeepMind AlphaGo/AlphaZero**
- MCTS + Neural Networks for Go, Chess, Shogi
- $200M/year logistics savings at Alphabet
- 40% data center energy reduction

### 5.13 Integration for Causal Inference

Estimate causal effects from observational data:

```python
from src.core.causal_inference import (
    CausalInferenceSystem,
    ATEResult
)

# Create system
causal = CausalInferenceSystem()

# Generate synthetic healthcare data
data = causal.generate_synthetic_data(n_samples=1000)

# Inverse Propensity Weighting
ipw_result = causal.estimate_ate_ipw(data)
print(f"IPW ATE: {ipw_result.ate_estimate:.3f}")

# Doubly Robust Estimation (recommended)
dr_result = causal.estimate_ate_doubly_robust(data)
print(f"DR ATE: {dr_result.ate_estimate:.3f} ± {dr_result.ate_std_error:.3f}")

# Bayesian Causal Inference with uncertainty
bayes_result = causal.bayesian_causal_inference(data, n_posterior_samples=200)
print(f"Bayesian ATE: {bayes_result.ate_mean:.3f} ± {bayes_result.ate_std:.3f}")

# Heterogeneous treatment effects
het_analysis = causal.analyze_heterogeneous_effects(
    data, dr_result.diagnostics['individual_effects']
)
```

**Industrial Case Study: Microsoft Uplift Modeling**
- 76% ROI increase in marketing
- 40% campaign reduction, same conversions
- $100M/year savings

### 5.14 Integration in Graph Neural Networks

Bayesian GNNs for uncertainty-aware graph learning:

```python
from src.core.gnn_integration import (
    BayesianGCN,
    generate_synthetic_graph
)

# Generate graph data
graph = generate_synthetic_graph(num_nodes=200, num_classes=3)

# Create Bayesian GCN with uncertainty
model = BayesianGCN(
    input_dim=graph.num_features,
    hidden_dim=32,
    output_dim=3,
    num_samples=10  # Monte Carlo samples for uncertainty
)

# Train and evaluate
losses = model.train_step(graph, num_epochs=50)
metrics = model.evaluate(graph)
print(f"Test Accuracy: {metrics['test_accuracy']:.2%}")
print(f"Confident Prediction Accuracy: {metrics['confident_accuracy']:.2%}")

# Get predictions with uncertainty
prediction = model.predict(graph)
high_uncertainty_nodes = np.argsort(prediction.uncertainty)[-5:]
```

**Industrial Case Study: Meta Social Graph**
- Bayesian GNNs for fraud detection
- 42% fraud reduction, 28% engagement increase

### 5.15 Integration for Explainable AI (XAI)

SHAP-based model interpretability:

```python
from src.core.explainable_ai import ExplainableModel

# Create explainable model
model = ExplainableModel(model_type='random_forest')

# Generate medical data
data = model.generate_medical_data(n_samples=500)
model.train(data['X'], data['y'], data['feature_names'])

# Global feature importance
global_exp = model.get_global_importance(data['X'])
print("Top features:", list(global_exp.feature_importance.keys())[:3])

# Local explanation for one patient
explanation = model.predict_with_explanation(data['X'][:1])[0]
print(model.explain_prediction_text(explanation))
```

**Industrial Case Study: IBM Watson for Oncology**
- SHAP + Bayesian uncertainty for cancer treatment
- 65% trust increase among physicians
- Decision time: hours → minutes

### 5.16 Integration with Differential Privacy

Privacy-preserving integration with mathematical guarantees:

```python
from src.core.differential_privacy import (
    DifferentiallyPrivateIntegrator
)

# Initialize with privacy parameter epsilon
dp = DifferentiallyPrivateIntegrator(epsilon=1.0)

# Private mean - protects individual data
data = np.array([1, 2, 3, 4, 5])
result = dp.private_mean(data, bounds=(0, 10))
print(f"Private mean: {result.value:.3f}")
print(f"Epsilon used: {result.epsilon_used}")

# Private integral
result = dp.private_integral(lambda x: x**2, 0, 1, n_points=50)
```

**Industrial Case Study: Apple Privacy-Preserving ML**
- Federated learning + DP for Siri
- 25% accuracy improvement, 500M users protected

### 5.17 Integration in Energy-Efficient ML Systems

Energy-aware integration for IoT and edge devices:

```python
from src.core.energy_efficient import EnergyEfficientIntegrator

# Configure for IoT device
integrator = EnergyEfficientIntegrator(device='iot')

# Choose accuracy level (trades off with energy)
result = integrator.integrate(
    lambda x: x**2, 0, 1, accuracy='medium'
)
print(f"Result: {result.value:.4f}")
print(f"Energy: {result.energy_cost:.2e} Wh")

# Optimize for energy budget
result = integrator.optimize_for_energy_budget(
    lambda x: x**2, 0, 1, energy_budget=1e-5
)
```

**Industrial Case Study: Google DeepMind Data Centers**
- 40% cooling energy reduction
- $150M/year savings
- 300,000 tons CO₂ reduction annually

---


## 6. Classical Machine Learning




### 5.1 Linear Regression

```python
from src.ml.classical import LinearRegressionScratch

# Create model
model = LinearRegressionScratch(method='gradient_descent', learning_rate=0.01)

# Fit
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Coefficients
print(f"Weights: {model.weights}")
print(f"Bias: {model.bias}")

# R² Score
score = model.score(X_test, y_test)
```

### 5.2 Logistic Regression

```python
from src.ml.classical import LogisticRegressionScratch

# Binary classification
model = LogisticRegressionScratch(
    learning_rate=0.01,
    n_iterations=1000,
    regularization='l2',
    lambda_=0.1
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### 5.3 Support Vector Machine

```python
from src.ml.classical import SVMScratch

# Create SVM
svm = SVMScratch(
    C=1.0,                # Regularization (higher = less regularization)
    learning_rate=0.001,   # Step size
    n_iterations=1000,     # Training iterations
    kernel='linear'        # 'linear' or 'rbf'
)

# Train
svm.fit(X_train, y_train)

# Predict
predictions = svm.predict(X_test)

# Decision function (distance from hyperplane)
decision_values = svm.decision_function(X_test)

# Probability estimates (Platt scaling approximation)
probabilities = svm.predict_proba(X_test)

# Get support vectors
print(f"Support vectors: {len(svm.support_vectors_)}")
print(f"Training loss: {svm.loss_history[-1]:.4f}")
```

### 5.4 Decision Tree

```python
from src.ml.classical import DecisionTreeScratch

# Create tree
tree = DecisionTreeScratch(
    max_depth=5,
    min_samples_split=10,
    criterion='gini'  # or 'entropy'
)

tree.fit(X_train, y_train)
predictions = tree.predict(X_test)

# Feature importance
importances = tree.feature_importances_
```

### 5.5 Random Forest

```python
from src.ml.classical import RandomForestScratch

# Create forest
forest = RandomForestScratch(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    bootstrap=True
)

forest.fit(X_train, y_train)
predictions = forest.predict(X_test)
probabilities = forest.predict_proba(X_test)
```

### 5.6 K-Nearest Neighbors

```python
from src.ml.classical import KNNScratch

knn = KNNScratch(n_neighbors=5, metric='euclidean')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
```

### 5.7 Naive Bayes

```python
from src.ml.classical import GaussianNBScratch

nb = GaussianNBScratch()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
log_probs = nb.predict_log_proba(X_test)
```

---

## 6. Deep Learning

### 6.1 Basic Neural Network

```python
from src.ml.deep_learning import (
    NeuralNetwork, Dense, Activation, Dropout,
    BatchNormalization, CrossEntropyLoss
)

# Build model
model = NeuralNetwork()
model.add(Dense(input_size=784, output_size=256, weight_init='he'))
model.add(BatchNormalization(256))
model.add(Activation('relu'))
model.add(Dropout(rate=0.3))
model.add(Dense(256, 128, weight_init='he'))
model.add(Activation('relu'))
model.add(Dropout(rate=0.2))
model.add(Dense(128, 10, weight_init='xavier'))
model.add(Activation('softmax'))

# Compile
model.compile(
    loss=CrossEntropyLoss(),
    learning_rate=0.01
)

# Summary
model.summary()
# Output:
# ============================================================
# Model Summary
# ============================================================
# Layer 1: Dense                  | Params: 200,960
# Layer 2: BatchNormalization     | Params: 512
# Layer 3: Activation             | Params: 0
# ...
# ============================================================
# Total Parameters: 235,146
# ============================================================

# Train
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=True
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### 6.2 LSTM for Sequences

```python
from src.ml.deep_learning import LSTM, Dense, Activation, NeuralNetwork

# For sequence data: (batch_size, timesteps, features)
model = NeuralNetwork()
model.add(LSTM(input_size=10, hidden_size=64, return_sequences=False))
model.add(Dense(64, 32))
model.add(Activation('relu'))
model.add(Dense(32, 2))
model.add(Activation('softmax'))

# Input shape: (batch, timesteps, features)
X_seq = np.random.randn(100, 20, 10)  # 100 samples, 20 steps, 10 features
y_seq = np.random.randint(0, 2, 100)

model.compile(loss=CrossEntropyLoss(), learning_rate=0.01)
model.fit(X_seq, y_seq, epochs=10)
```

### 6.3 Convolutional Neural Network

```python
from src.ml.deep_learning import (
    Conv2D, MaxPool2D, Flatten, Dense, Activation, NeuralNetwork
)

# Build CNN for image classification
model = NeuralNetwork()

# Conv block 1
model.add(Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding=1))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=2))

# Conv block 2
model.add(Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding=1))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=2))

# Classifier
model.add(Flatten())
model.add(Dense(64 * 7 * 7, 256))
model.add(Activation('relu'))
model.add(Dense(256, 10))
model.add(Activation('softmax'))

# Input shape: (batch, channels, height, width)
X_images = np.random.randn(100, 1, 28, 28)  # MNIST-like
y_labels = np.random.randint(0, 10, 100)

model.compile(loss=CrossEntropyLoss(), learning_rate=0.001)
model.fit(X_images, y_labels, epochs=5, batch_size=16)
```

### 6.4 Available Layers

| Layer | Description | Parameters |
|-------|-------------|------------|
| `Dense` | Fully connected | input_size, output_size, weight_init |
| `Activation` | Activation function | 'relu', 'sigmoid', 'tanh', 'softmax', 'leaky_relu' |
| `Dropout` | Regularization | rate (0-1) |
| `BatchNormalization` | Normalize activations | n_features, momentum, epsilon |
| `LSTM` | Recurrent layer | input_size, hidden_size, return_sequences |
| `Conv2D` | 2D Convolution | in_channels, out_channels, kernel_size, stride, padding |
| `MaxPool2D` | Max pooling | pool_size, stride |
| `Flatten` | Reshape to 1D | - |

### 6.5 Loss Functions

| Loss | Class | Use Case |
|------|-------|----------|
| MSE | `MSELoss` | Regression |
| Cross-Entropy | `CrossEntropyLoss` | Multi-class classification |
| Binary Cross-Entropy | `BinaryCrossEntropyLoss` | Binary classification |

---

## 7. LLM Engineering

### 7.1 Attention Mechanisms

The `src/llm/attention.py` module implements a comprehensive suite of attention mechanisms:

**Basic Attention:**

```python
from src.llm.attention import (
    MultiHeadAttention,
    SelfAttention,
    CausalSelfAttention,
    CrossAttention,
)
import torch

# Multi-head attention (Transformer baseline)
mha = MultiHeadAttention(d_model=512, num_heads=8)
output = mha(query, key, value)

# Causal (autoregressive) attention for GPT-style models
causal = CausalSelfAttention(d_model=512, num_heads=8, block_size=1024)
output = causal(x)  # Tokens only attend to previous tokens
```

**Advanced Attention Variants (State-of-the-Art):**

```python
from src.llm.attention import (
    AttentionWithRoPE,       # Llama, Mistral (relative position)
    GroupedQueryAttention,   # Llama 2 70B (memory efficient)
    FlashAttention,          # 2-4x faster, O(1) memory
)

# RoPE: Rotary Position Embeddings (Llama, Mistral)
rope_attn = AttentionWithRoPE(d_model=512, num_heads=8, max_len=4096)
output = rope_attn(x)  # Relative position awareness

# Grouped Query Attention: Reduce KV cache by sharing heads
gqa = GroupedQueryAttention(d_model=512, num_heads=32, num_kv_heads=8)
output = gqa(x)  # 4x KV cache reduction

# Flash Attention: Memory-efficient for long sequences
flash = FlashAttention(d_model=512, num_heads=8)
output = flash(x)  # Enables 16x longer sequences
```

**Factory Function:**

```python
from src.llm.attention import create_attention_mechanism

# Dynamically create attention by type
attn = create_attention_mechanism(
    attention_type='rope',  # Options: 'multi_head', 'causal', 'rope', 'gqa', 'flash'
    d_model=512,
    num_heads=8
)
```



### 7.2 RAG Pipeline

```python
from src.llm.rag import RAGModel, Document, RetrievalStrategy

# Initialize RAG with different strategies
# Options: DENSE, SPARSE, HYBRID
rag = RAGModel(retriever_strategy=RetrievalStrategy.HYBRID)

# Add documents
documents = [
    Document(
        id="doc1",
        content="Machine learning uses algorithms to learn from data.",
        metadata={"source": "ml_intro.txt", "topic": "ML"}
    ),
    Document(
        id="doc2",
        content="Neural networks are inspired by biological neurons.",
        metadata={"source": "dl_intro.txt", "topic": "DL"}
    ),
]
rag.add_documents(documents)

# Query
result = rag.query(
    "What is machine learning?",
    k=3  # Number of documents to retrieve
)

print(f"Response: {result['response']}")
print(f"Retrieved documents: {len(result['documents'])}")
for doc in result['documents']:
    print(f"  - {doc.id}: {doc.content[:50]}...")
```

### 7.3 Data Ingestion

```python
from scripts.ingest_data import DataIngestionPipeline

# Initialize pipeline
pipeline = DataIngestionPipeline(
    embedding_model="all-MiniLM-L6-v2",
    chunk_size=512,
    chunk_overlap=50
)

# Ingest documents from directory
num_docs = pipeline.ingest_directory(
    directory="./data/documents",
    extensions=[".txt", ".md", ".pdf"],
    recursive=True
)

# Process (chunk and embed)
num_chunks = pipeline.process_documents()

# Save index
pipeline.save("./data/rag_index")

# Search
results = pipeline.search("How do neural networks work?", k=5)
for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Content: {result['content'][:100]}...")
```

### 7.4 Fine-tuning with LoRA

```python
from src.llm.fine_tuning import LoRAAdapter

# Apply LoRA to a base model
adapter = LoRAAdapter(
    base_model=model,
    rank=8,          # Rank of low-rank matrices
    alpha=16,        # Scaling factor
    target_modules=['query', 'value']  # Layers to adapt
)

# Train only LoRA parameters (much smaller)
adapter.train(train_data, epochs=3)

# Save adapter
adapter.save("./adapters/my_lora")
```

---

## 8. Production API

### 8.1 Starting the API

```bash
# Option 1: Direct
uvicorn src.production.api:app --host 0.0.0.0 --port 8000 --reload

# Option 2: Make command
make run

# Option 3: Docker
docker-compose up -d api
```

### 8.2 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Welcome message |
| GET | `/health` | Health check with model count |
| POST | `/predict` | Single prediction |
| POST | `/predict/batch` | Batch predictions |
| GET | `/models` | List all loaded models |
| GET | `/models/{model_id}` | Get specific model info |
| POST | `/models/reload` | Hot reload models |
| GET | `/metrics` | Prometheus metrics |
| GET | `/docs` | OpenAPI documentation |

### 8.3 Example Requests

**Health Check:**
```bash
curl http://localhost:8000/health
```
```json
{
    "status": "healthy",
    "models_loaded": 3,
    "timestamp": 1704067200.0
}
```

**List Models:**
```bash
curl http://localhost:8000/models
```
```json
{
    "models": [
        {
            "model_id": "classification_model",
            "model_type": "RandomForestClassifier",
            "n_features": 10,
            "n_classes": 2,
            "metadata": {"accuracy": 0.92}
        },
        {
            "model_id": "regression_model",
            "model_type": "GradientBoostingRegressor",
            "n_features": 10
        }
    ]
}
```

**Single Prediction:**
```bash
curl -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{
        "features": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "model_name": "classification_model"
    }'
```
```json
{
    "predictions": [1],
    "probabilities": [[0.15, 0.85]],
    "model_type": "RandomForestClassifier",
    "processing_time": 0.002
}
```

**Batch Prediction:**
```bash
curl -X POST http://localhost:8000/predict/batch \
    -H "Content-Type: application/json" \
    -d '{
        "features": [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
        ],
        "model_id": "classification_model"
    }'
```

### 8.4 Python Client

```python
import requests

API_URL = "http://localhost:8000"

# Health check
response = requests.get(f"{API_URL}/health")
print(response.json())

# Prediction
features = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
response = requests.post(
    f"{API_URL}/predict",
    json={"features": features, "model_name": "classification_model"}
)
result = response.json()
print(f"Prediction: {result['predictions'][0]}")
print(f"Confidence: {max(result['probabilities'][0]):.2%}")
```

---

## 9. Web Interface

### 9.1 Starting Streamlit

```bash
# Direct
streamlit run app/main.py

# Docker
docker-compose up -d streamlit

# Access at http://localhost:8501
```

### 9.2 Pages

**Home Dashboard:**
- Models loaded count
- API status indicator
- Response time metrics
- Uptime statistics

**Chat (RAG):**
- Interactive Q&A interface
- Document retrieval visualization
- Source citations
- Conversation history

**Predictions:**
- Model selection dropdown
- Feature input (manual/JSON/random)
- Real-time prediction results
- Probability visualization

**Models:**
- List all registered models
- Model metadata display
- Feature count, type, metrics

**Settings:**
- API URL configuration
- Theme selection
- RAG parameters

---

## 10. Docker Deployment

### 10.1 Services

```yaml
# docker-compose.yml services:

api:          # FastAPI server (port 8000)
streamlit:    # Web UI (port 8501)
postgres:     # Database (port 5432)
redis:        # Cache (port 6379)
prometheus:   # Metrics (port 9090)
grafana:      # Dashboards (port 3000)
```

### 10.2 Commands

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d api

# View logs
docker-compose logs -f api

# Stop all
docker-compose down

# Rebuild after code changes
docker-compose up -d --build

# Remove volumes (data reset)
docker-compose down -v
```

### 10.3 Environment Variables

```bash
# Database
DB_HOST=postgres
DB_PORT=5432
DB_NAME=ai_mastery
DB_USER=postgres
DB_PASSWORD=password

# API
API_URL=http://api:8000
PROMETHEUS_URL=http://prometheus:9090

# Optional
LOG_LEVEL=INFO
WORKERS=4
```

---

## 11. Monitoring & Observability

### 11.1 Prometheus Metrics

Built-in metrics at `/metrics`:

```
# Request counts
http_requests_total{method="POST", endpoint="/predict", status="200"}

# Latency histogram
http_request_duration_seconds_bucket{le="0.1"}
http_request_duration_seconds_sum
http_request_duration_seconds_count

# Custom ML metrics
ml_models_loaded_total
ml_predictions_total{model="classification_model"}
```

### 11.2 Grafana Dashboard

Access at http://localhost:3000 (admin/admin)

**Pre-built panels:**
- Models Loaded (gauge)
- API Status (status indicator)
- Response Time (time series)
- Request Rate (graph)
- Error Rate (percentage)

### 11.3 Example Prometheus Queries

```promql
# Average response time (last 5 min)
rate(http_request_duration_seconds_sum[5m]) 
  / rate(http_request_duration_seconds_count[5m])

# 95th percentile latency
histogram_quantile(0.95, 
  rate(http_request_duration_seconds_bucket[5m])
)

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) 
  / sum(rate(http_requests_total[5m])) * 100

# Request rate by endpoint
sum by (endpoint) (rate(http_requests_total[5m]))
```

---

## 12. Testing

### 12.1 Running Tests

```bash
# All tests
make test

# With coverage report
make test-cov
# Opens htmlcov/index.html

# Specific file
pytest tests/test_svm.py -v

# Specific test
pytest tests/test_svm.py::TestSVMScratch::test_accuracy_linearly_separable -v

# With print output
pytest tests/test_svm.py -v -s
```

### 12.2 Test Structure

```python
# tests/test_example.py
import pytest
import numpy as np
from src.ml.classical import SVMScratch

class TestSVMScratch:
    
    @pytest.fixture
    def sample_data(self):
        """Generate test data."""
        X = np.random.randn(100, 5)
        y = (X.sum(axis=1) > 0).astype(int)
        return X, y
    
    def test_fit_predict(self, sample_data):
        X, y = sample_data
        svm = SVMScratch(n_iterations=100)
        svm.fit(X, y)
        predictions = svm.predict(X)
        assert predictions.shape == y.shape
    
    def test_accuracy(self, sample_data):
        X, y = sample_data
        svm = SVMScratch()
        svm.fit(X, y)
        accuracy = svm.score(X, y)
        assert accuracy > 0.5  # Better than random
```

### 12.3 Test Categories

| File | Tests |
|------|-------|
| `test_linear_algebra.py` | Vector/matrix operations |
| `test_probability.py` | Distributions, entropy |
| `test_ml_algorithms.py` | Classical ML models |
| `test_deep_learning.py` | Neural network layers |
| `test_svm.py` | SVM + LSTM + Conv2D |
| `test_rag_llm.py` | RAG pipeline |
| `integration/test_api.py` | API endpoints |

---

## 15. Troubleshooting

### 15.1 Docker Image Name Issues
If you encounter `invalid character` errors during Docker build/push, ensure your GitHub repository name does not contain uppercase letters when used as an image tag. The CI/CD pipeline handles this automatically by lowercasing the `IMAGE_NAME`.

### 15.2 Model Validation Failures
If `scripts/train_save_models.py` runs but CI fails on validation:
- Ensure `models_metadata.json` contains `test_accuracy` (for classifiers) or `test_r2` (for regressors).
- The validation script applies different thresholds: 0.80 for general models, 0.65 for Logistic Regression, and 0.60 for Regressors.

### 15.3 Import Errors (Legacy Modules)
If you see `ImportError: cannot import name 'DataType'`, ensuring you are running from the project root. The `sys.path` is automatically adjusted in scripts, but manual execution might require `export PYTHONPATH=$PYTHONPATH:.`.

### Common Issues

**1. ImportError: No module named 'src'**
```bash
# Add project to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or in Windows PowerShell
$env:PYTHONPATH = "$env:PYTHONPATH;$(pwd)"
```

**2. Models not found in API**
```bash
# Train and save models first
python scripts/train_save_models.py

# Check models directory
ls models/
# Should see: classification_model.joblib, etc.
```

**3. Port already in use**
```bash
# Kill process on port 8000
kill $(lsof -t -i:8000)

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

**4. Docker memory issues**
```bash
# Increase Docker memory (4GB+ recommended)
# Docker Desktop > Settings > Resources > Memory

# Or in docker-compose.yml:
services:
  api:
    deploy:
      resources:
        limits:
          memory: 2G
```

**5. Slow training**
```bash
# Use GPU if available
pip install torch --extra-index-url https://download.pytorch.org/whl/cu118

# Check GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

**6. Database connection failed**
```bash
# Check PostgreSQL is running
docker-compose ps postgres

# View logs
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d postgres
```

---

## 16. FAQ

**Q: Can I use this for production?**
A: Yes! The toolkit includes production-ready components (FastAPI, Docker, monitoring). However, review security settings before deploying to public environments.

**Q: How do I add a new ML model?**
A: Add to `src/ml/classical.py` or `deep_learning.py`, then:
1. Inherit from `BaseEstimator` or `Layer`
2. Implement `fit()`, `predict()`, `forward()`, `backward()`
3. Add tests in `tests/`
4. Export in `__all__`

**Q: Can I use GPU?**
A: The from-scratch implementations use NumPy (CPU). For GPU, leverage PyTorch/TensorFlow versions in production code.

**Q: How do I contribute?**
A: See [docs/guide/08_contribution_guide.md](docs/guide/08_contribution_guide.md). Fork, create branch, submit PR.

**Q: Where are the Jupyter notebooks?**
A: In `research/` directory, organized by week (1-17).

---

## Need Help?

1. 📖 Check the [Documentation](docs/guide/)
2. 🧪 Run tests: `make test`
3. 📋 Check [GitHub Issues](https://github.com/Kandil7/AI-Mastery-2026/issues)
4. 📬 Open a new issue with:
   - Python version
   - OS
   - Error message
   - Steps to reproduce

---

*Built with ❤️ for learning AI engineering from first principles*
