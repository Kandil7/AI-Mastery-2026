"""
Probability and Statistics Module
=================================
Probability distributions, sampling methods, and information theory
implemented from first principles following the White-Box Approach.

Mathematical Foundations:
- Probability distributions (discrete and continuous)
- Sampling algorithms
- Information theory (entropy, KL divergence)
- Bayesian probability

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Union, Optional, Tuple, List, Callable
import math

# Type aliases
Array = Union[List[float], np.ndarray]


# ============================================================
# PROBABILITY DISTRIBUTIONS
# ============================================================

class Distribution:
    """Base class for probability distributions."""
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Draw n samples from the distribution."""
        raise NotImplementedError
    
    def pdf(self, x: Array) -> np.ndarray:
        """Probability density function (for continuous) or PMF (for discrete)."""
        raise NotImplementedError
    
    def log_pdf(self, x: Array) -> np.ndarray:
        """Log probability density (for numerical stability)."""
        return np.log(self.pdf(x) + 1e-10)
    
    def cdf(self, x: Array) -> np.ndarray:
        """Cumulative distribution function."""
        raise NotImplementedError
    
    @property
    def mean(self) -> float:
        """Expected value."""
        raise NotImplementedError
    
    @property
    def variance(self) -> float:
        """Variance."""
        raise NotImplementedError


class Gaussian(Distribution):
    """
    Gaussian (Normal) Distribution.
    
    Mathematical Definition:
        f(x) = (1/√(2πσ²)) × exp(-(x-μ)²/(2σ²))
    
    Properties:
        - Mean: μ
        - Variance: σ²
        - Maximum entropy for given mean and variance
    
    Applications:
        - Modeling errors and noise
        - Prior distributions in Bayesian ML
        - Variational inference
    
    Example:
        >>> dist = Gaussian(mean=0, std=1)
        >>> samples = dist.sample(1000)
        >>> print(f"Sample mean: {np.mean(samples):.3f}")
    """
    
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """
        Args:
            mean: Mean of the distribution (μ)
            std: Standard deviation (σ)
        """
        assert std > 0, "Standard deviation must be positive"
        self._mean = mean
        self._std = std
        self._var = std ** 2
    
    def sample(self, n: int = 1) -> np.ndarray:
        """
        Sample using Box-Muller transform.
        
        Algorithm:
            1. Generate U1, U2 ~ Uniform(0,1)
            2. Z0 = √(-2 ln U1) × cos(2π U2)
            3. Z1 = √(-2 ln U1) × sin(2π U2)
        """
        u1 = np.random.uniform(0, 1, n)
        u2 = np.random.uniform(0, 1, n)
        
        # Box-Muller transform
        z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        
        # Scale and shift
        return self._mean + self._std * z
    
    def pdf(self, x: Array) -> np.ndarray:
        """Probability density function."""
        x = np.asarray(x)
        coef = 1 / (self._std * np.sqrt(2 * np.pi))
        exponent = -((x - self._mean) ** 2) / (2 * self._var)
        return coef * np.exp(exponent)
    
    def log_pdf(self, x: Array) -> np.ndarray:
        """Log probability density (numerically stable)."""
        x = np.asarray(x)
        return (
            -0.5 * np.log(2 * np.pi * self._var) 
            - ((x - self._mean) ** 2) / (2 * self._var)
        )
    
    def cdf(self, x: Array) -> np.ndarray:
        """
        CDF using error function.
        
        Φ(x) = 0.5 × (1 + erf((x - μ) / (σ√2)))
        """
        x = np.asarray(x)
        return 0.5 * (1 + np.erf((x - self._mean) / (self._std * np.sqrt(2))))
    
    @property
    def mean(self) -> float:
        return self._mean
    
    @property
    def variance(self) -> float:
        return self._var
    
    @property
    def std(self) -> float:
        return self._std


class MultivariateGaussian(Distribution):
    """
    Multivariate Gaussian Distribution.
    
    Mathematical Definition:
        f(x) = (2π)^(-k/2) |Σ|^(-1/2) × exp(-0.5 × (x-μ)ᵀ Σ⁻¹ (x-μ))
    
    Applications:
        - Word embeddings (assuming Gaussian prior)
        - Variational Autoencoders (VAE)
        - Gaussian Mixture Models
    """
    
    def __init__(self, mean: Array, cov: Array):
        """
        Args:
            mean: Mean vector of shape (k,)
            cov: Covariance matrix of shape (k, k)
        """
        self._mean = np.asarray(mean)
        self._cov = np.asarray(cov)
        self.k = len(self._mean)
        
        # Precompute for efficiency
        self._cov_inv = np.linalg.inv(self._cov)
        self._cov_det = np.linalg.det(self._cov)
        self._L = np.linalg.cholesky(self._cov)
    
    def sample(self, n: int = 1) -> np.ndarray:
        """
        Sample using Cholesky decomposition.
        
        Algorithm:
            1. L = cholesky(Σ)
            2. z ~ N(0, I)
            3. x = μ + L @ z
        """
        z = np.random.randn(n, self.k)
        return self._mean + z @ self._L.T
    
    def pdf(self, x: Array) -> np.ndarray:
        """Probability density function."""
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        diff = x - self._mean
        exponent = -0.5 * np.sum(diff @ self._cov_inv * diff, axis=1)
        coef = 1 / ((2 * np.pi) ** (self.k / 2) * np.sqrt(self._cov_det))
        
        return coef * np.exp(exponent)
    
    def log_pdf(self, x: Array) -> np.ndarray:
        """Log probability density."""
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        diff = x - self._mean
        mahalanobis = np.sum(diff @ self._cov_inv * diff, axis=1)
        
        return (
            -0.5 * self.k * np.log(2 * np.pi)
            - 0.5 * np.log(self._cov_det)
            - 0.5 * mahalanobis
        )
    
    @property
    def mean(self) -> np.ndarray:
        return self._mean
    
    @property
    def variance(self) -> np.ndarray:
        return np.diag(self._cov)


class Bernoulli(Distribution):
    """
    Bernoulli Distribution.
    
    Mathematical Definition:
        P(X = 1) = p
        P(X = 0) = 1 - p
    
    Applications:
        - Binary classification
        - Coin flips
        - Dropout in neural networks
    """
    
    def __init__(self, p: float = 0.5):
        """
        Args:
            p: Probability of success (0 < p < 1)
        """
        assert 0 <= p <= 1, "Probability must be in [0, 1]"
        self.p = p
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Sample from Bernoulli distribution."""
        return (np.random.uniform(0, 1, n) < self.p).astype(int)
    
    def pdf(self, x: Array) -> np.ndarray:
        """Probability mass function."""
        x = np.asarray(x)
        return np.where(x == 1, self.p, 1 - self.p)
    
    @property
    def mean(self) -> float:
        return self.p
    
    @property
    def variance(self) -> float:
        return self.p * (1 - self.p)


class Categorical(Distribution):
    """
    Categorical Distribution (generalized Bernoulli).
    
    Mathematical Definition:
        P(X = k) = p_k for k = 1, ..., K
    
    Applications:
        - Multi-class classification
        - Language models (next token prediction)
        - Topic models
    """
    
    def __init__(self, probs: Array):
        """
        Args:
            probs: Probability vector (must sum to 1)
        """
        self.probs = np.asarray(probs)
        assert abs(np.sum(self.probs) - 1.0) < 1e-6, "Probabilities must sum to 1"
        self.k = len(self.probs)
    
    def sample(self, n: int = 1) -> np.ndarray:
        """
        Sample using inverse CDF method.
        """
        u = np.random.uniform(0, 1, n)
        cumsum = np.cumsum(self.probs)
        samples = np.searchsorted(cumsum, u)
        return samples
    
    def pdf(self, x: Array) -> np.ndarray:
        """Probability mass function."""
        x = np.asarray(x).astype(int)
        return self.probs[x]
    
    @property
    def mean(self) -> float:
        """Expected value (index)."""
        return np.sum(np.arange(self.k) * self.probs)
    
    @property
    def variance(self) -> float:
        indices = np.arange(self.k)
        return np.sum((indices - self.mean) ** 2 * self.probs)


class Uniform(Distribution):
    """
    Uniform Distribution.
    
    Mathematical Definition:
        f(x) = 1/(b-a) for a ≤ x ≤ b
    """
    
    def __init__(self, low: float = 0.0, high: float = 1.0):
        assert high > low, "High must be greater than low"
        self.low = low
        self.high = high
        self._range = high - low
    
    def sample(self, n: int = 1) -> np.ndarray:
        return self.low + self._range * np.random.uniform(0, 1, n)
    
    def pdf(self, x: Array) -> np.ndarray:
        x = np.asarray(x)
        return np.where((x >= self.low) & (x <= self.high), 1 / self._range, 0)
    
    def cdf(self, x: Array) -> np.ndarray:
        x = np.asarray(x)
        return np.clip((x - self.low) / self._range, 0, 1)
    
    @property
    def mean(self) -> float:
        return (self.low + self.high) / 2
    
    @property
    def variance(self) -> float:
        return (self._range ** 2) / 12


class Exponential(Distribution):
    """
    Exponential Distribution.
    
    Mathematical Definition:
        f(x) = λ × exp(-λx) for x ≥ 0
    
    Applications:
        - Time between events (Poisson process)
        - Survival analysis
        - Queue wait times
    """
    
    def __init__(self, rate: float = 1.0):
        """
        Args:
            rate: Rate parameter λ (λ > 0)
        """
        assert rate > 0, "Rate must be positive"
        self.rate = rate
    
    def sample(self, n: int = 1) -> np.ndarray:
        """Sample using inverse CDF method."""
        u = np.random.uniform(0, 1, n)
        return -np.log(1 - u) / self.rate
    
    def pdf(self, x: Array) -> np.ndarray:
        x = np.asarray(x)
        return np.where(x >= 0, self.rate * np.exp(-self.rate * x), 0)
    
    def cdf(self, x: Array) -> np.ndarray:
        x = np.asarray(x)
        return np.where(x >= 0, 1 - np.exp(-self.rate * x), 0)
    
    @property
    def mean(self) -> float:
        return 1 / self.rate
    
    @property
    def variance(self) -> float:
        return 1 / (self.rate ** 2)


# ============================================================
# SAMPLING ALGORITHMS
# ============================================================

def rejection_sampling(
    target_pdf: Callable[[np.ndarray], np.ndarray],
    proposal_sampler: Callable[[int], np.ndarray],
    proposal_pdf: Callable[[np.ndarray], np.ndarray],
    M: float,
    n_samples: int
) -> np.ndarray:
    """
    Rejection Sampling Algorithm.
    
    Algorithm:
        1. Sample x from proposal distribution q(x)
        2. Sample u ~ Uniform(0, 1)
        3. Accept x if u < p(x) / (M × q(x))
    
    Args:
        target_pdf: Target distribution p(x)
        proposal_sampler: Function to sample from proposal q(x)
        proposal_pdf: Proposal distribution PDF q(x)
        M: Scaling factor (M × q(x) ≥ p(x) for all x)
        n_samples: Number of samples to generate
    
    Returns:
        Accepted samples from target distribution
    
    Example:
        >>> target = lambda x: np.exp(-x**2 / 2) / np.sqrt(2*np.pi)
        >>> proposal_sample = lambda n: np.random.uniform(-4, 4, n)
        >>> proposal_pdf = lambda x: np.ones_like(x) / 8
        >>> samples = rejection_sampling(target, proposal_sample, proposal_pdf, 4, 1000)
    """
    samples = []
    n_accepted = 0
    
    while n_accepted < n_samples:
        # Sample from proposal
        x = proposal_sampler(1)[0]
        
        # Acceptance probability
        u = np.random.uniform(0, 1)
        accept_prob = target_pdf(np.array([x]))[0] / (M * proposal_pdf(np.array([x]))[0])
        
        if u < accept_prob:
            samples.append(x)
            n_accepted += 1
    
    return np.array(samples)


def importance_sampling(
    target_pdf: Callable[[np.ndarray], np.ndarray],
    proposal_sampler: Callable[[int], np.ndarray],
    proposal_pdf: Callable[[np.ndarray], np.ndarray],
    f: Callable[[np.ndarray], np.ndarray],
    n_samples: int
) -> Tuple[float, float]:
    """
    Importance Sampling for Expectation Estimation.
    
    Used to estimate E_p[f(x)] when sampling from p(x) is difficult.
    
    Algorithm:
        E_p[f(x)] ≈ (1/n) × Σᵢ f(xᵢ) × w(xᵢ)
        where w(xᵢ) = p(xᵢ) / q(xᵢ)
    
    Args:
        target_pdf: Target distribution p(x)
        proposal_sampler: Function to sample from proposal q(x)
        proposal_pdf: Proposal distribution PDF q(x)
        f: Function whose expectation we want
        n_samples: Number of samples
    
    Returns:
        Tuple of (estimated expectation, variance estimate)
    """
    # Sample from proposal
    samples = proposal_sampler(n_samples)
    
    # Compute importance weights
    weights = target_pdf(samples) / proposal_pdf(samples)
    
    # Normalize weights (self-normalized importance sampling)
    weights = weights / np.sum(weights)
    
    # Compute weighted estimate
    values = f(samples)
    estimate = np.sum(weights * values)
    
    # Variance estimate
    variance = np.sum(weights ** 2 * (values - estimate) ** 2)
    
    return estimate, variance


def metropolis_hastings(
    target_log_pdf: Callable[[float], float],
    proposal_sampler: Callable[[float], float],
    initial: float,
    n_samples: int,
    burn_in: int = 1000
) -> np.ndarray:
    """
    Metropolis-Hastings MCMC Sampler.
    
    Used for sampling from distributions whose PDF is known up to a constant.
    
    Algorithm:
        1. Start at x_0
        2. Propose x' ~ q(x'|x_t)
        3. Accept with probability min(1, p(x')/p(x_t))
    
    Args:
        target_log_pdf: Log of target distribution (up to constant)
        proposal_sampler: Function to sample proposal given current state
        initial: Initial state
        n_samples: Number of samples
        burn_in: Number of samples to discard
    
    Returns:
        MCMC samples from target distribution
    """
    samples = []
    current = initial
    current_log_prob = target_log_pdf(current)
    
    for i in range(n_samples + burn_in):
        # Propose new state
        proposed = proposal_sampler(current)
        proposed_log_prob = target_log_pdf(proposed)
        
        # Accept/reject
        log_accept_ratio = proposed_log_prob - current_log_prob
        
        if np.log(np.random.uniform(0, 1)) < log_accept_ratio:
            current = proposed
            current_log_prob = proposed_log_prob
        
        if i >= burn_in:
            samples.append(current)
    
    return np.array(samples)


# ============================================================
# INFORMATION THEORY
# ============================================================

def entropy(p: Array, base: float = 2) -> float:
    """
    Shannon Entropy.
    
    Mathematical Definition:
        H(X) = -Σ p(x) × log(p(x))
    
    Interpretation:
        - Measures uncertainty/randomness
        - Maximum for uniform distribution
        - Used in decision trees (information gain)
    
    Args:
        p: Probability distribution
        base: Logarithm base (2 for bits, e for nats)
    
    Returns:
        Entropy value
    """
    p = np.asarray(p)
    p = p[p > 0]  # Ignore zero probabilities
    
    if base == 2:
        return -np.sum(p * np.log2(p))
    elif base == np.e:
        return -np.sum(p * np.log(p))
    else:
        return -np.sum(p * np.log(p)) / np.log(base)


def cross_entropy(p: Array, q: Array) -> float:
    """
    Cross Entropy.
    
    Mathematical Definition:
        H(p, q) = -Σ p(x) × log(q(x))
    
    Used as loss function in classification.
    
    Args:
        p: True distribution
        q: Predicted distribution
    
    Returns:
        Cross entropy value
    """
    p = np.asarray(p)
    q = np.asarray(q) + 1e-10  # Avoid log(0)
    
    return -np.sum(p * np.log(q))


def kl_divergence(p: Array, q: Array) -> float:
    """
    Kullback-Leibler Divergence.
    
    Mathematical Definition:
        D_KL(P || Q) = Σ p(x) × log(p(x) / q(x))
    
    Properties:
        - D_KL(P || Q) ≥ 0
        - D_KL(P || Q) = 0 iff P = Q
        - Not symmetric: D_KL(P || Q) ≠ D_KL(Q || P)
    
    Applications:
        - VAE loss function
        - Distribution comparison
        - Model compression
    
    Args:
        p: True distribution
        q: Approximate distribution
    
    Returns:
        KL divergence value
    """
    p = np.asarray(p)
    q = np.asarray(q) + 1e-10
    
    # Mask zero probabilities in p
    mask = p > 0
    p_masked = p[mask]
    q_masked = q[mask]
    
    return np.sum(p_masked * np.log(p_masked / q_masked))


def js_divergence(p: Array, q: Array) -> float:
    """
    Jensen-Shannon Divergence.
    
    Mathematical Definition:
        D_JS(P || Q) = 0.5 × D_KL(P || M) + 0.5 × D_KL(Q || M)
        where M = 0.5 × (P + Q)
    
    Properties:
        - Symmetric: D_JS(P || Q) = D_JS(Q || P)
        - Bounded: 0 ≤ D_JS ≤ 1 (for base-2 log)
    
    Args:
        p: First distribution
        q: Second distribution
    
    Returns:
        JS divergence value
    """
    p = np.asarray(p)
    q = np.asarray(q)
    m = 0.5 * (p + q)
    
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def mutual_information(joint_probs: Array) -> float:
    """
    Mutual Information.
    
    Mathematical Definition:
        I(X; Y) = Σₓ Σᵧ p(x,y) × log(p(x,y) / (p(x) × p(y)))
    
    Interpretation:
        - Measures dependency between variables
        - I(X; Y) = 0 iff X and Y are independent
        - Used in feature selection
    
    Args:
        joint_probs: Joint probability matrix P(X, Y)
    
    Returns:
        Mutual information value
    """
    joint = np.asarray(joint_probs)
    
    # Marginals
    p_x = np.sum(joint, axis=1)
    p_y = np.sum(joint, axis=0)
    
    # Compute MI
    mi = 0.0
    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += joint[i, j] * np.log(joint[i, j] / (p_x[i] * p_y[j]))
    
    return mi


def conditional_entropy(joint_probs: Array) -> float:
    """
    Conditional Entropy H(Y|X).
    
    Mathematical Definition:
        H(Y|X) = H(X, Y) - H(X)
    
    Args:
        joint_probs: Joint probability matrix P(X, Y)
    
    Returns:
        Conditional entropy value
    """
    joint = np.asarray(joint_probs)
    
    # Marginal of X
    p_x = np.sum(joint, axis=1)
    
    # Joint entropy
    joint_entropy = entropy(joint.flatten(), base=np.e)
    
    # Marginal entropy
    marginal_entropy = entropy(p_x, base=np.e)
    
    return joint_entropy - marginal_entropy


# ============================================================
# BAYESIAN UTILITIES
# ============================================================

def bayes_theorem(prior: float, likelihood: float, evidence: float) -> float:
    """
    Bayes' Theorem.
    
    Mathematical Definition:
        P(A|B) = P(B|A) × P(A) / P(B)
    
    Args:
        prior: P(A) - prior probability
        likelihood: P(B|A) - likelihood
        evidence: P(B) - evidence (marginal likelihood)
    
    Returns:
        Posterior probability P(A|B)
    """
    return (likelihood * prior) / evidence


def posterior_update(prior: Array, likelihood: Array) -> np.ndarray:
    """
    Bayesian posterior update for discrete distributions.
    
    Args:
        prior: Prior probabilities
        likelihood: Likelihood of observed data
    
    Returns:
        Normalized posterior distribution
    """
    prior = np.asarray(prior)
    likelihood = np.asarray(likelihood)
    
    unnormalized = prior * likelihood
    return unnormalized / np.sum(unnormalized)


def beta_binomial_posterior(
    alpha_prior: float, 
    beta_prior: float, 
    successes: int, 
    failures: int
) -> Tuple[float, float]:
    """
    Beta-Binomial Conjugate Update.
    
    Prior: Beta(α, β)
    Likelihood: Binomial(n, p)
    Posterior: Beta(α + k, β + n - k)
    
    Args:
        alpha_prior: Prior α parameter
        beta_prior: Prior β parameter
        successes: Number of successes observed
        failures: Number of failures observed
    
    Returns:
        Posterior (α, β) parameters
    """
    alpha_post = alpha_prior + successes
    beta_post = beta_prior + failures
    return alpha_post, beta_post


def gaussian_posterior(
    prior_mean: float,
    prior_var: float,
    data_mean: float,
    data_var: float,
    n: int
) -> Tuple[float, float]:
    """
    Gaussian-Gaussian Conjugate Update.
    
    Compute posterior for mean given observed data.
    
    Args:
        prior_mean: Prior mean
        prior_var: Prior variance
        data_mean: Sample mean
        data_var: Known data variance
        n: Number of observations
    
    Returns:
        Posterior (mean, variance)
    """
    precision_prior = 1 / prior_var
    precision_data = n / data_var
    
    precision_post = precision_prior + precision_data
    mean_post = (precision_prior * prior_mean + precision_data * data_mean) / precision_post
    var_post = 1 / precision_post
    
    return mean_post, var_post


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def log_sum_exp(x: Array) -> float:
    """
    Numerically stable log-sum-exp.
    
    Computes log(Σ exp(xᵢ)) without overflow.
    """
    x = np.asarray(x)
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(x - x_max)))


def normalize_log_probs(log_probs: Array) -> np.ndarray:
    """
    Convert log probabilities to normalized probabilities.
    """
    log_probs = np.asarray(log_probs)
    log_sum = log_sum_exp(log_probs)
    return np.exp(log_probs - log_sum)


# ============================================================
# EXPORTS
# ============================================================

__all__ = [
    # Distributions
    'Distribution', 'Gaussian', 'MultivariateGaussian', 
    'Bernoulli', 'Categorical', 'Uniform', 'Exponential',
    # Sampling
    'rejection_sampling', 'importance_sampling', 'metropolis_hastings',
    # Information theory
    'entropy', 'cross_entropy', 'kl_divergence', 'js_divergence',
    'mutual_information', 'conditional_entropy',
    # Bayesian
    'bayes_theorem', 'posterior_update', 'beta_binomial_posterior', 'gaussian_posterior',
    # Utilities
    'log_sum_exp', 'normalize_log_probs',
]
