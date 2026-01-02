"""
Variational Inference Module
============================

This module implements Variational Inference (VI) methods from scratch using NumPy.
VI converts Bayesian inference into an optimization problem, making it suitable
for large-scale applications where MCMC would be too slow.

Methods included:
- Evidence Lower Bound (ELBO) computation
- Mean-Field Variational Inference
- Stochastic Variational Inference (SVI)
- Reparameterization Trick
- Coordinate Ascent VI (CAVI)

Industrial Applications:
- Uber: Demand forecasting with uncertainty
- Spotify: Music recommendation with Bayesian models
- Netflix: User preference modeling
- LinkedIn: Graph-based recommendation systems
"""

import numpy as np
from typing import Callable, Tuple, Optional, List, Dict, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VIResult:
    """
    Container for Variational Inference results.
    
    Attributes:
        variational_params: Optimized variational parameters
        elbo_history: ELBO values during optimization
        converged: Whether optimization converged
        n_iterations: Number of iterations performed
        final_elbo: Final ELBO value
    """
    variational_params: Dict
    elbo_history: List[float]
    converged: bool
    n_iterations: int
    final_elbo: float


# =============================================================================
# VARIATIONAL FAMILIES
# =============================================================================

class VariationalFamily(ABC):
    """
    Abstract base class for variational families.
    
    A variational family defines the class of distributions q(z; λ)
    used to approximate the true posterior p(z|x).
    """
    
    @abstractmethod
    def sample(self, n_samples: int) -> np.ndarray:
        """Sample from the variational distribution."""
        pass
    
    @abstractmethod
    def log_prob(self, z: np.ndarray) -> np.ndarray:
        """Compute log q(z; λ) for given z."""
        pass
    
    @abstractmethod
    def entropy(self) -> float:
        """Compute entropy H[q] = -E_q[log q(z)]."""
        pass
    
    @abstractmethod
    def get_params(self) -> Dict:
        """Get variational parameters."""
        pass
    
    @abstractmethod
    def set_params(self, params: Dict) -> None:
        """Set variational parameters."""
        pass


class GaussianVariational(VariationalFamily):
    """
    Gaussian variational family with diagonal covariance.
    
    q(z; μ, σ) = N(z; μ, diag(σ²))
    
    This is the standard mean-field Gaussian approximation.
    
    Attributes:
        d: Dimensionality
        mean: Mean vector μ (d,)
        log_std: Log standard deviation log(σ) (d,)
    
    Industrial Use Case:
        Uber uses Gaussian variational approximations for their
        demand forecasting models, enabling real-time uncertainty
        estimation for surge pricing.
    """
    
    def __init__(self, d: int, mean: Optional[np.ndarray] = None, 
                 log_std: Optional[np.ndarray] = None):
        """
        Initialize Gaussian variational family.
        
        Args:
            d: Dimensionality
            mean: Initial mean (default: zeros)
            log_std: Initial log std (default: zeros)
        """
        self.d = d
        self.mean = mean if mean is not None else np.zeros(d)
        self.log_std = log_std if log_std is not None else np.zeros(d)
    
    @property
    def std(self) -> np.ndarray:
        """Standard deviation."""
        return np.exp(self.log_std)
    
    @property
    def var(self) -> np.ndarray:
        """Variance."""
        return np.exp(2 * self.log_std)
    
    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sample from q(z) using the reparameterization trick.
        
        z = μ + σ ⊙ ε, where ε ~ N(0, I)
        
        This enables gradient computation through the sampling process.
        """
        eps = np.random.randn(n_samples, self.d)
        return self.mean + self.std * eps
    
    def log_prob(self, z: np.ndarray) -> np.ndarray:
        """
        Compute log q(z; μ, σ) for given z.
        
        log q(z) = -0.5 * Σ_i [(z_i - μ_i)²/σ_i² + log(2πσ_i²)]
        """
        if z.ndim == 1:
            z = z.reshape(1, -1)
        
        log_probs = -0.5 * np.sum(
            ((z - self.mean) / self.std) ** 2 + 
            2 * self.log_std + 
            np.log(2 * np.pi),
            axis=1
        )
        return log_probs
    
    def entropy(self) -> float:
        """
        Compute entropy of the Gaussian.
        
        H[q] = 0.5 * d * (1 + log(2π)) + Σ_i log(σ_i)
        """
        return 0.5 * self.d * (1 + np.log(2 * np.pi)) + np.sum(self.log_std)
    
    def get_params(self) -> Dict:
        """Get variational parameters."""
        return {
            'mean': self.mean.copy(),
            'log_std': self.log_std.copy()
        }
    
    def set_params(self, params: Dict) -> None:
        """Set variational parameters."""
        self.mean = params['mean'].copy()
        self.log_std = params['log_std'].copy()
    
    def kl_to_standard_normal(self) -> float:
        """
        Compute KL divergence to standard normal.
        
        KL(q || N(0,I)) = 0.5 * Σ_i [μ_i² + σ_i² - 1 - log(σ_i²)]
        
        This is commonly used as a regularizer in VAEs.
        """
        return 0.5 * np.sum(
            self.mean ** 2 + self.var - 1 - 2 * self.log_std
        )


# =============================================================================
# ELBO COMPUTATION
# =============================================================================

def compute_elbo(
    log_joint: Callable[[np.ndarray], np.ndarray],
    variational: VariationalFamily,
    n_samples: int = 100
) -> Tuple[float, np.ndarray]:
    """
    Compute the Evidence Lower Bound (ELBO).
    
    The ELBO is:
        ELBO = E_q[log p(z, x)] - E_q[log q(z)]
             = E_q[log p(z, x)] + H[q]
    
    where p(z, x) is the joint distribution and q(z) is the variational
    approximation.
    
    By Jensen's inequality: log p(x) ≥ ELBO
    
    Maximizing ELBO is equivalent to minimizing KL(q || p(z|x)).
    
    Args:
        log_joint: Function computing log p(z, x)
        variational: Variational distribution q(z)
        n_samples: Number of Monte Carlo samples
    
    Returns:
        Tuple of (ELBO estimate, samples used)
    
    Example:
        >>> # For Bayesian linear regression
        >>> log_joint = lambda z: log_prior(z) + log_likelihood(X, y, z)
        >>> q = GaussianVariational(d=10)
        >>> elbo, _ = compute_elbo(log_joint, q, n_samples=100)
    
    Interview Question:
        Q: What's the relationship between ELBO and the marginal likelihood?
        A: log p(x) = ELBO + KL(q || p(z|x)). Since KL ≥ 0, ELBO is
           a lower bound on log p(x). Maximizing ELBO pushes q toward
           the true posterior.
    """
    # Sample from variational distribution
    z_samples = variational.sample(n_samples)
    
    # Compute expectations
    log_joint_values = log_joint(z_samples)
    log_q_values = variational.log_prob(z_samples)
    
    # ELBO = E_q[log p(z,x) - log q(z)]
    elbo = np.mean(log_joint_values - log_q_values)
    
    return elbo, z_samples


def compute_elbo_gradient(
    log_joint: Callable[[np.ndarray], np.ndarray],
    grad_log_joint: Callable[[np.ndarray], np.ndarray],
    variational: GaussianVariational,
    n_samples: int = 100
) -> Tuple[Dict, float]:
    """
    Compute ELBO gradient using the reparameterization trick.
    
    The reparameterization trick enables low-variance gradient estimates
    by expressing z = μ + σ ⊙ ε and differentiating through this.
    
    Args:
        log_joint: Function computing log p(z, x)
        grad_log_joint: Function computing ∇_z log p(z, x)
        variational: Gaussian variational distribution
        n_samples: Number of Monte Carlo samples
    
    Returns:
        Tuple of (gradient dict, ELBO estimate)
    
    Industrial Use Case:
        The reparameterization trick is the foundation of Variational
        Autoencoders (VAEs), used by Spotify for music representation
        learning and Netflix for user embedding.
    """
    d = variational.d
    mean = variational.mean
    log_std = variational.log_std
    std = variational.std
    
    # Sample epsilon
    eps = np.random.randn(n_samples, d)
    z = mean + std * eps
    
    # Compute log joint and its gradient
    log_joint_values = log_joint(z)
    grad_values = np.array([grad_log_joint(z_i) for z_i in z])  # (n_samples, d)
    
    # Gradient w.r.t. mean: E[∇_z log p(z,x)]
    grad_mean = np.mean(grad_values, axis=0)
    
    # Gradient w.r.t. log_std: E[∇_z log p(z,x) * ε * σ] - 1
    # The -1 comes from the entropy gradient
    grad_log_std = np.mean(grad_values * eps * std, axis=0) - 1
    
    # ELBO estimate
    log_q = variational.log_prob(z)
    elbo = np.mean(log_joint_values - log_q)
    
    return {'mean': grad_mean, 'log_std': grad_log_std}, elbo


# =============================================================================
# MEAN-FIELD VARIATIONAL INFERENCE
# =============================================================================

class MeanFieldVI:
    """
    Mean-Field Variational Inference optimizer.
    
    Uses stochastic gradient ascent to maximize the ELBO with
    a factorized (mean-field) variational approximation:
    
        q(z) = Π_i q_i(z_i)
    
    Attributes:
        variational: Variational family (e.g., GaussianVariational)
        learning_rate: Learning rate for gradient ascent
        n_samples: Number of samples for gradient estimation
    
    Industrial Use Case:
        LinkedIn uses mean-field VI for their large-scale graph
        models, processing billions of connections efficiently.
    """
    
    def __init__(
        self,
        variational: VariationalFamily,
        learning_rate: float = 0.01,
        n_samples: int = 100
    ):
        """
        Initialize Mean-Field VI optimizer.
        
        Args:
            variational: Variational distribution
            learning_rate: Learning rate
            n_samples: Samples per gradient estimate
        """
        self.variational = variational
        self.learning_rate = learning_rate
        self.n_samples = n_samples
        self.elbo_history = []
    
    def step(
        self,
        log_joint: Callable[[np.ndarray], np.ndarray],
        grad_log_joint: Callable[[np.ndarray], np.ndarray]
    ) -> float:
        """
        Perform one gradient ascent step.
        
        Args:
            log_joint: Log joint probability function
            grad_log_joint: Gradient of log joint
        
        Returns:
            Current ELBO estimate
        """
        if not isinstance(self.variational, GaussianVariational):
            raise NotImplementedError("Only GaussianVariational supported")
        
        grad, elbo = compute_elbo_gradient(
            log_joint, grad_log_joint, self.variational, self.n_samples
        )
        
        # Gradient ascent update
        self.variational.mean += self.learning_rate * grad['mean']
        self.variational.log_std += self.learning_rate * grad['log_std']
        
        self.elbo_history.append(elbo)
        return elbo
    
    def fit(
        self,
        log_joint: Callable[[np.ndarray], np.ndarray],
        grad_log_joint: Callable[[np.ndarray], np.ndarray],
        n_iterations: int = 1000,
        tol: float = 1e-6,
        patience: int = 50,
        verbose: bool = True
    ) -> VIResult:
        """
        Fit the variational distribution.
        
        Args:
            log_joint: Log joint probability function
            grad_log_joint: Gradient of log joint
            n_iterations: Maximum iterations
            tol: Convergence tolerance
            patience: Early stopping patience
            verbose: Print progress
        
        Returns:
            VIResult with optimized parameters
        
        Interview Question:
            Q: How do you know if VI has converged?
            A: Monitor ELBO plateau. Also check if variational parameters
               have stabilized. Unlike MCMC, VI gives point estimates
               so R-hat doesn't apply.
        """
        best_elbo = -np.inf
        no_improvement = 0
        converged = False
        
        for i in range(n_iterations):
            elbo = self.step(log_joint, grad_log_joint)
            
            if elbo > best_elbo + tol:
                best_elbo = elbo
                no_improvement = 0
            else:
                no_improvement += 1
            
            if no_improvement >= patience:
                converged = True
                if verbose:
                    print(f"Converged at iteration {i+1}, ELBO: {elbo:.4f}")
                break
            
            if verbose and (i + 1) % 100 == 0:
                print(f"Iteration {i+1}, ELBO: {elbo:.4f}")
        
        return VIResult(
            variational_params=self.variational.get_params(),
            elbo_history=self.elbo_history,
            converged=converged,
            n_iterations=len(self.elbo_history),
            final_elbo=self.elbo_history[-1]
        )


# =============================================================================
# STOCHASTIC VARIATIONAL INFERENCE
# =============================================================================

class StochasticVI:
    """
    Stochastic Variational Inference with mini-batching.
    
    For large datasets, SVI uses mini-batches to scale VI to
    millions of data points. Uses Adam optimizer for stability.
    
    Attributes:
        variational: Variational distribution
        learning_rate: Initial learning rate
        n_samples: MC samples per gradient
        batch_size: Mini-batch size
    
    Industrial Use Case:
        Spotify uses SVI for their Bayesian music recommendation
        models, processing billions of plays from 500M+ users.
    """
    
    def __init__(
        self,
        variational: GaussianVariational,
        learning_rate: float = 0.001,
        n_samples: int = 10,
        batch_size: int = 128,
        beta1: float = 0.9,
        beta2: float = 0.999
    ):
        """
        Initialize SVI optimizer with Adam.
        
        Args:
            variational: Gaussian variational distribution
            learning_rate: Learning rate for Adam
            n_samples: MC samples for gradient
            batch_size: Mini-batch size
            beta1: Adam beta1
            beta2: Adam beta2
        """
        self.variational = variational
        self.learning_rate = learning_rate
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.beta1 = beta1
        self.beta2 = beta2
        
        # Adam state
        self.m_mean = np.zeros(variational.d)
        self.v_mean = np.zeros(variational.d)
        self.m_log_std = np.zeros(variational.d)
        self.v_log_std = np.zeros(variational.d)
        self.t = 0
        
        self.elbo_history = []
    
    def step(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        log_likelihood_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        grad_log_likelihood_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
        n_total: int,
        prior_log_prob: Callable[[np.ndarray], float],
        grad_prior_log_prob: Callable[[np.ndarray], np.ndarray]
    ) -> float:
        """
        Perform one SVI step with a mini-batch.
        
        Args:
            X_batch: Mini-batch features (batch_size, d_x)
            y_batch: Mini-batch targets (batch_size,)
            log_likelihood_fn: Log likelihood function
            grad_log_likelihood_fn: Gradient of log likelihood
            n_total: Total dataset size (for scaling)
            prior_log_prob: Prior log probability
            grad_prior_log_prob: Gradient of prior
        
        Returns:
            ELBO estimate
        """
        self.t += 1
        eps = 1e-8
        
        # Sample from variational
        d = self.variational.d
        epsilon = np.random.randn(self.n_samples, d)
        z = self.variational.mean + self.variational.std * epsilon
        
        batch_size = len(X_batch)
        scale = n_total / batch_size
        
        # Compute gradients
        grad_mean = np.zeros(d)
        grad_log_std = np.zeros(d)
        elbo_sum = 0.0
        
        for i in range(self.n_samples):
            z_i = z[i]
            
            # Prior gradient
            prior_grad = grad_prior_log_prob(z_i)
            prior_lp = prior_log_prob(z_i)
            
            # Likelihood gradient (scaled for mini-batch)
            lik_grad = scale * grad_log_likelihood_fn(X_batch, y_batch, z_i)
            lik_lp = scale * log_likelihood_fn(X_batch, y_batch, z_i)
            
            # Combined gradient
            total_grad = prior_grad + lik_grad
            
            # Accumulate
            grad_mean += total_grad
            grad_log_std += total_grad * epsilon[i] * self.variational.std
            
            # ELBO contribution
            elbo_sum += prior_lp + lik_lp - self.variational.log_prob(z_i.reshape(1, -1))[0]
        
        grad_mean /= self.n_samples
        grad_log_std = grad_log_std / self.n_samples - 1  # Entropy gradient
        elbo = elbo_sum / self.n_samples
        
        # Adam updates
        self.m_mean = self.beta1 * self.m_mean + (1 - self.beta1) * grad_mean
        self.v_mean = self.beta2 * self.v_mean + (1 - self.beta2) * grad_mean ** 2
        
        self.m_log_std = self.beta1 * self.m_log_std + (1 - self.beta1) * grad_log_std
        self.v_log_std = self.beta2 * self.v_log_std + (1 - self.beta2) * grad_log_std ** 2
        
        # Bias correction
        m_mean_hat = self.m_mean / (1 - self.beta1 ** self.t)
        v_mean_hat = self.v_mean / (1 - self.beta2 ** self.t)
        m_log_std_hat = self.m_log_std / (1 - self.beta1 ** self.t)
        v_log_std_hat = self.v_log_std / (1 - self.beta2 ** self.t)
        
        # Update parameters
        self.variational.mean += self.learning_rate * m_mean_hat / (np.sqrt(v_mean_hat) + eps)
        self.variational.log_std += self.learning_rate * m_log_std_hat / (np.sqrt(v_log_std_hat) + eps)
        
        self.elbo_history.append(elbo)
        return elbo


# =============================================================================
# COORDINATE ASCENT VARIATIONAL INFERENCE
# =============================================================================

def coordinate_ascent_vi(
    natural_params_update: Callable[[int, Dict], Dict],
    initial_params: Dict,
    n_iterations: int = 100,
    tol: float = 1e-6,
    verbose: bool = True
) -> VIResult:
    """
    Coordinate Ascent Variational Inference (CAVI).
    
    For conjugate exponential family models, CAVI provides closed-form
    updates for each factor of the variational distribution.
    
    Algorithm:
        For each latent variable z_i:
            Update q(z_i) ∝ exp(E_{-i}[log p(z, x)])
    
    Args:
        natural_params_update: Function (i, params) -> updated params for factor i
        initial_params: Initial variational parameters
        n_iterations: Maximum iterations
        tol: Convergence tolerance
        verbose: Print progress
    
    Returns:
        VIResult with converged parameters
    
    Interview Question:
        Q: When is CAVI preferred over gradient-based VI?
        A: CAVI is exact for conjugate models (no gradient variance),
           faster per iteration, but only works with specific model
           structures. Gradient VI is more general but noisier.
    """
    params = {k: v.copy() if hasattr(v, 'copy') else v 
              for k, v in initial_params.items()}
    n_factors = len(params)
    
    elbo_history = []
    converged = False
    
    for iteration in range(n_iterations):
        params_old = {k: v.copy() if hasattr(v, 'copy') else v 
                      for k, v in params.items()}
        
        # Update each factor
        for i in range(n_factors):
            params = natural_params_update(i, params)
        
        # Check convergence
        param_diff = sum(
            np.max(np.abs(params[k] - params_old[k])) 
            for k in params if hasattr(params[k], '__sub__')
        )
        
        elbo_history.append(-param_diff)  # Proxy for ELBO
        
        if param_diff < tol:
            converged = True
            if verbose:
                print(f"CAVI converged at iteration {iteration + 1}")
            break
        
        if verbose and (iteration + 1) % 10 == 0:
            print(f"Iteration {iteration + 1}, param change: {param_diff:.6f}")
    
    return VIResult(
        variational_params=params,
        elbo_history=elbo_history,
        converged=converged,
        n_iterations=len(elbo_history),
        final_elbo=elbo_history[-1] if elbo_history else 0.0
    )


# =============================================================================
# BAYESIAN LINEAR REGRESSION WITH VI
# =============================================================================

class BayesianLinearRegressionVI:
    """
    Bayesian Linear Regression using Variational Inference.
    
    Model:
        w ~ N(0, α⁻¹ I)  [prior]
        y | X, w ~ N(Xw, β⁻¹ I)  [likelihood]
    
    Variational approximation:
        q(w) = N(w; μ_w, Σ_w)
    
    For this conjugate model, we can derive closed-form updates.
    
    Industrial Use Case:
        Bayesian linear regression with VI is used in recommendation
        systems where we need fast inference on user preferences
        with uncertainty estimates for exploration/exploitation.
    """
    
    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0
    ):
        """
        Initialize Bayesian Linear Regression.
        
        Args:
            alpha: Prior precision (inverse variance)
            beta: Likelihood precision
        """
        self.alpha = alpha
        self.beta = beta
        self.mean = None
        self.cov = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianLinearRegressionVI':
        """
        Fit the model using variational inference.
        
        For linear regression, the posterior is exactly Gaussian,
        so VI gives the exact answer (no approximation error).
        
        Args:
            X: Features (n, d)
            y: Targets (n,)
        
        Returns:
            self
        """
        n, d = X.shape
        
        # Posterior precision (inverse covariance)
        precision = self.alpha * np.eye(d) + self.beta * X.T @ X
        
        # Posterior covariance
        self.cov = np.linalg.inv(precision)
        
        # Posterior mean
        self.mean = self.beta * self.cov @ X.T @ y
        
        return self
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions with uncertainty.
        
        Args:
            X: Test features (m, d)
            return_std: Whether to return predictive std
        
        Returns:
            Mean predictions, optionally with std
        """
        mean_pred = X @ self.mean
        
        if return_std:
            # Predictive variance = observation noise + model uncertainty
            var_pred = (1.0 / self.beta) + np.sum(X @ self.cov * X, axis=1)
            std_pred = np.sqrt(var_pred)
            return mean_pred, std_pred
        
        return mean_pred
    
    def elbo(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the ELBO for this model.
        
        For the exact posterior, ELBO = log marginal likelihood.
        """
        n, d = X.shape
        
        # Log likelihood term
        predictions = X @ self.mean
        residuals = y - predictions
        ll = -0.5 * self.beta * np.sum(residuals ** 2)
        ll += 0.5 * n * np.log(self.beta / (2 * np.pi))
        
        # Prior term
        prior = -0.5 * self.alpha * np.sum(self.mean ** 2)
        prior += 0.5 * d * np.log(self.alpha / (2 * np.pi))
        
        # Entropy of variational distribution
        entropy = 0.5 * d * (1 + np.log(2 * np.pi)) + 0.5 * np.linalg.slogdet(self.cov)[1]
        
        # KL divergence (analytically for Gaussian)
        kl = 0.5 * (
            self.alpha * (np.sum(self.mean ** 2) + np.trace(self.cov)) -
            d + d * np.log(self.alpha) - np.linalg.slogdet(self.cov)[1]
        )
        
        return ll + prior + entropy - kl


# =============================================================================
# STEIN VARIATIONAL GRADIENT DESCENT (SVGD)
# =============================================================================

def svgd_kernel(X: np.ndarray, h: float = -1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute RBF kernel and gradient for SVGD.
    
    k(x, x') = exp(-||x - x'||² / h)
    
    Args:
        X: Particles (n, d)
        h: Bandwidth (auto-computed if -1)
    
    Returns:
        Kernel matrix (n, n), kernel gradient (n, n, d)
    """
    n, d = X.shape
    
    # Pairwise squared distances
    sq_dist = np.sum(X ** 2, axis=1, keepdims=True) - 2 * X @ X.T + np.sum(X ** 2, axis=1)
    
    # Auto bandwidth using median heuristic
    if h < 0:
        h = np.median(sq_dist[np.triu_indices(n, k=1)])
        h = np.sqrt(0.5 * h / np.log(n + 1))
    
    # Kernel matrix
    K = np.exp(-sq_dist / (2 * h ** 2))
    
    # Kernel gradient
    grad_K = np.zeros((n, n, d))
    for i in range(n):
        for j in range(n):
            grad_K[i, j] = K[i, j] * (X[j] - X[i]) / (h ** 2)
    
    return K, grad_K


def svgd(
    log_prob: Callable[[np.ndarray], np.ndarray],
    grad_log_prob: Callable[[np.ndarray], np.ndarray],
    initial_particles: np.ndarray,
    n_iterations: int = 1000,
    learning_rate: float = 0.1,
    bandwidth: float = -1
) -> np.ndarray:
    """
    Stein Variational Gradient Descent.
    
    SVGD is a hybrid between VI and MCMC that uses a set of particles
    to approximate the posterior. It combines the efficiency of VI
    with better multi-modal coverage.
    
    Update rule:
        x_i^{t+1} = x_i^t + ε * φ(x_i^t)
        
        φ(x) = (1/n) Σ_j [k(x_j, x) ∇_{x_j} log p(x_j) + ∇_{x_j} k(x_j, x)]
    
    The first term pushes particles toward high probability regions.
    The second term (repulsive) maintains diversity.
    
    Args:
        log_prob: Log probability function
        grad_log_prob: Gradient of log probability
        initial_particles: Starting particles (n, d)
        n_iterations: Number of iterations
        learning_rate: Step size
        bandwidth: RBF kernel bandwidth
    
    Returns:
        Final particles (n, d)
    
    Industrial Use Case:
        SVGD is used at Uber for demand prediction where the posterior
        is multi-modal due to different traffic patterns.
    
    Interview Question:
        Q: How does SVGD compare to standard VI?
        A: SVGD uses non-parametric approximation (particles) instead
           of a parametric family. Better for multi-modal posteriors
           but requires storing particles.
    """
    particles = initial_particles.copy()
    n, d = particles.shape
    
    for _ in range(n_iterations):
        # Compute gradients at all particles
        grads = np.array([grad_log_prob(p) for p in particles])  # (n, d)
        
        # Compute kernel and its gradient
        K, grad_K = svgd_kernel(particles, bandwidth)
        
        # SVGD update
        phi = (K @ grads + np.sum(grad_K, axis=0)) / n
        particles += learning_rate * phi
    
    return particles


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compare_vi_methods(
    log_joint: Callable,
    grad_log_joint: Callable,
    d: int,
    methods: List[str] = ['mfvi', 'svgd'],
    n_iterations: int = 500
) -> Dict:
    """
    Compare different VI methods on the same target.
    
    Args:
        log_joint: Log joint probability
        grad_log_joint: Gradient of log joint
        d: Dimensionality
        methods: Methods to compare
        n_iterations: Iterations per method
    
    Returns:
        Dictionary with results from each method
    """
    results = {}
    
    if 'mfvi' in methods:
        q = GaussianVariational(d)
        vi = MeanFieldVI(q, learning_rate=0.01)
        result = vi.fit(log_joint, grad_log_joint, n_iterations, verbose=False)
        results['mfvi'] = {
            'mean': q.mean,
            'std': q.std,
            'final_elbo': result.final_elbo,
            'elbo_history': result.elbo_history
        }
    
    if 'svgd' in methods:
        initial = np.random.randn(100, d)
        particles = svgd(
            lambda x: log_joint(x.reshape(1, -1))[0],
            lambda x: grad_log_joint(x),
            initial,
            n_iterations,
            learning_rate=0.1
        )
        results['svgd'] = {
            'mean': np.mean(particles, axis=0),
            'std': np.std(particles, axis=0),
            'particles': particles
        }
    
    return results
