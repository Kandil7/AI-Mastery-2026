"""
Integration Module
==================

This module implements numerical integration (quadrature) methods and 
Monte Carlo integration techniques from scratch using NumPy.

Methods included:
- Newton-Cotes methods (Trapezoidal, Simpson's Rule)
- Gaussian Quadrature (Gauss-Legendre, Gauss-Hermite)
- Bayesian Quadrature
- Monte Carlo Integration with variance reduction

Industrial Applications:
- NASA JPL: Trajectory calculations
- BlackRock: Option pricing
- Netflix: Recommendation systems
- AstraZeneca: Drug discovery
"""

import numpy as np
from typing import Callable, Tuple, List, Optional, Union
from scipy.special import roots_hermite, roots_legendre
import warnings


# =============================================================================
# NEWTON-COTES METHODS
# =============================================================================

def trapezoidal_rule(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int
) -> float:
    """
    Trapezoidal rule for numerical integration.
    
    Approximates ∫[a,b] f(x)dx using linear interpolation between
    equidistant points. Error is O(1/N²).
    
    Args:
        f: Function to integrate (vectorized)
        a: Lower integration bound
        b: Upper integration bound
        n: Number of intervals (n+1 points)
    
    Returns:
        Approximate integral value
    
    Example:
        >>> f = lambda x: x**2
        >>> trapezoidal_rule(f, 0, 1, 100)  # ≈ 0.3333
    
    Industrial Use Case:
        Boeing uses trapezoidal rule for integrating pressure distributions
        over aircraft wing surfaces to compute lift coefficients.
    """
    if n < 1:
        raise ValueError("Number of intervals must be at least 1")
    
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    
    # Trapezoidal formula: h/2 * (f0 + 2*f1 + 2*f2 + ... + 2*f_{n-1} + fn)
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return integral


def simpsons_rule(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int
) -> float:
    """
    Simpson's 1/3 rule for numerical integration.
    
    Approximates ∫[a,b] f(x)dx using quadratic interpolation.
    Error is O(1/N⁴) - much faster convergence for smooth functions.
    
    Args:
        f: Function to integrate (vectorized)
        a: Lower integration bound
        b: Upper integration bound
        n: Number of intervals (must be even)
    
    Returns:
        Approximate integral value
    
    Example:
        >>> f = lambda x: np.exp(-x**2)
        >>> simpsons_rule(f, 0, 1, 100)  # ≈ 0.7468
    
    Interview Question:
        Q: When would you prefer Simpson's over Trapezoidal?
        A: For smooth functions where O(1/N⁴) vs O(1/N²) matters.
           Trapezoidal is better for noisy data or discontinuities.
    """
    if n < 2:
        raise ValueError("Number of intervals must be at least 2")
    if n % 2 != 0:
        n += 1  # Ensure even number of intervals
    
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    
    # Simpson's formula: h/3 * (f0 + 4*f1 + 2*f2 + 4*f3 + ... + 4*f_{n-1} + fn)
    integral = y[0] + y[-1]
    integral += 4 * np.sum(y[1:-1:2])  # Odd indices
    integral += 2 * np.sum(y[2:-2:2])  # Even indices (excluding endpoints)
    integral *= h / 3
    
    return integral


def adaptive_quadrature(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    tol: float = 1e-8,
    max_depth: int = 50
) -> Tuple[float, int]:
    """
    Adaptive quadrature using recursive subdivision.
    
    Recursively subdivides intervals where the error estimate
    (difference between Trapezoidal and Simpson's) exceeds tolerance.
    
    Args:
        f: Function to integrate
        a: Lower bound
        b: Upper bound
        tol: Error tolerance
        max_depth: Maximum recursion depth
    
    Returns:
        Tuple of (integral value, number of function evaluations)
    
    Industrial Use Case:
        NOAA uses adaptive quadrature for atmospheric integration
        where precision requirements vary across the domain.
    """
    def _adaptive_helper(a, b, fa, fb, fm, whole, tol, depth, f_evals):
        mid = (a + b) / 2
        left_mid = (a + mid) / 2
        right_mid = (mid + b) / 2
        
        flm = f(np.array([left_mid]))[0]
        frm = f(np.array([right_mid]))[0]
        f_evals[0] += 2
        
        # Simpson's rule on left and right halves
        left = (mid - a) / 6 * (fa + 4 * flm + fm)
        right = (b - mid) / 6 * (fm + 4 * frm + fb)
        delta = left + right - whole
        
        if depth <= 0 or abs(delta) <= 15 * tol:
            return left + right + delta / 15
        
        return (_adaptive_helper(a, mid, fa, fm, flm, left, tol/2, depth-1, f_evals) +
                _adaptive_helper(mid, b, fm, fb, frm, right, tol/2, depth-1, f_evals))
    
    fa = f(np.array([a]))[0]
    fb = f(np.array([b]))[0]
    fm = f(np.array([(a + b) / 2]))[0]
    f_evals = [3]
    
    whole = (b - a) / 6 * (fa + 4 * fm + fb)
    result = _adaptive_helper(a, b, fa, fb, fm, whole, tol, max_depth, f_evals)
    
    return result, f_evals[0]


# =============================================================================
# GAUSSIAN QUADRATURE
# =============================================================================

def gauss_legendre(
    f: Callable[[np.ndarray], np.ndarray],
    a: float,
    b: float,
    n: int
) -> float:
    """
    Gauss-Legendre quadrature for integration over [a, b].
    
    Exactly integrates polynomials up to degree 2n-1.
    Uses Legendre polynomial roots as nodes.
    
    Args:
        f: Function to integrate (vectorized)
        a: Lower integration bound
        b: Upper integration bound
        n: Number of quadrature points
    
    Returns:
        Approximate integral value
    
    Example:
        >>> f = lambda x: x**5  # Polynomial of degree 5
        >>> gauss_legendre(f, -1, 1, 3)  # Exact with n=3 (2*3-1=5)
    
    Interview Question:
        Q: Why is Gaussian quadrature more efficient than Newton-Cotes?
        A: It optimizes both nodes AND weights, achieving 2N-1 polynomial
           exactness with N points vs N-1 for Newton-Cotes.
    """
    # Get Legendre polynomial roots and weights for [-1, 1]
    nodes, weights = roots_legendre(n)
    
    # Transform from [-1, 1] to [a, b]
    transformed_nodes = 0.5 * (b - a) * nodes + 0.5 * (b + a)
    transformed_weights = 0.5 * (b - a) * weights
    
    # Evaluate and sum
    return np.sum(transformed_weights * f(transformed_nodes))


def gauss_hermite_expectation(
    f: Callable[[np.ndarray], np.ndarray],
    mu: float = 0.0,
    sigma: float = 1.0,
    n: int = 10
) -> float:
    """
    Compute E[f(X)] where X ~ N(μ, σ²) using Gauss-Hermite quadrature.
    
    Gauss-Hermite is designed for integrals of form ∫f(x)exp(-x²)dx,
    making it perfect for Gaussian expectations.
    
    Args:
        f: Function whose expectation we want
        mu: Mean of Gaussian distribution
        sigma: Standard deviation of Gaussian
        n: Number of quadrature points
    
    Returns:
        E[f(X)] ≈ (1/√π) Σ wₙ f(√2 σ xₙ + μ)
    
    Example:
        >>> f = lambda x: x**2
        >>> gauss_hermite_expectation(f, mu=0, sigma=1, n=5)  # E[X²] = 1
    
    Industrial Use Case:
        BlackRock uses Gauss-Hermite for computing expected payoffs
        in option pricing under Gaussian market models.
    """
    # Get Hermite polynomial roots and weights
    nodes, weights = roots_hermite(n)
    
    # Transform for general Gaussian N(μ, σ²)
    transformed_nodes = np.sqrt(2) * sigma * nodes + mu
    transformed_weights = weights / np.sqrt(np.pi)
    
    return np.sum(transformed_weights * f(transformed_nodes))


def gauss_hermite_multivariate(
    f: Callable[[np.ndarray], float],
    mu: np.ndarray,
    cov: np.ndarray,
    n_per_dim: int = 5
) -> float:
    """
    Multivariate Gaussian expectation using tensor product of Gauss-Hermite.
    
    Computes E[f(X)] where X ~ N(μ, Σ) using sparse grid or full tensor.
    
    Args:
        f: Function of d-dimensional input
        mu: Mean vector (d,)
        cov: Covariance matrix (d, d)
        n_per_dim: Quadrature points per dimension
    
    Returns:
        Approximate expectation
    
    Warning:
        Full tensor product scales as O(n^d) - use sparse grids for d > 3.
    """
    d = len(mu)
    nodes_1d, weights_1d = roots_hermite(n_per_dim)
    
    # Cholesky decomposition for transformation
    L = np.linalg.cholesky(cov)
    
    # Create tensor product grid (expensive for high d)
    if d > 3:
        warnings.warn(f"Full tensor product with d={d} may be slow. Consider sparse grids.")
    
    # Generate all combinations of 1D nodes
    grids = np.meshgrid(*[nodes_1d for _ in range(d)], indexing='ij')
    nodes = np.stack([g.ravel() for g in grids], axis=1)  # (n^d, d)
    
    # Generate weight products
    weight_grids = np.meshgrid(*[weights_1d for _ in range(d)], indexing='ij')
    weights = np.prod(np.stack([w.ravel() for w in weight_grids], axis=1), axis=1)
    
    # Transform nodes: z -> √2 * L @ z + μ
    transformed_nodes = np.sqrt(2) * (nodes @ L.T) + mu
    
    # Normalize weights
    normalized_weights = weights / (np.pi ** (d / 2))
    
    # Evaluate function at all points
    values = np.array([f(x) for x in transformed_nodes])
    
    return np.sum(normalized_weights * values)


# =============================================================================
# MONTE CARLO INTEGRATION
# =============================================================================

def monte_carlo_integrate(
    f: Callable[[np.ndarray], np.ndarray],
    sampler: Callable[[int], np.ndarray],
    n_samples: int,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Monte Carlo integration using random sampling.
    
    Computes E[f(X)] where X is sampled from a distribution.
    Error decreases as O(1/√N), independent of dimensionality.
    
    Args:
        f: Function to integrate (vectorized)
        sampler: Function that generates n samples from target distribution
        n_samples: Number of Monte Carlo samples
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (estimate, standard_error)
    
    Example:
        >>> f = lambda x: x**2
        >>> sampler = lambda n: np.random.randn(n)  # Standard normal
        >>> estimate, se = monte_carlo_integrate(f, sampler, 10000)
        >>> # estimate ≈ 1.0 (E[X²] for X~N(0,1))
    
    Industrial Use Case:
        Netflix uses Monte Carlo for estimating expected watch time
        across 200M+ users in high-dimensional preference space.
    """
    if seed is not None:
        np.random.seed(seed)
    
    samples = sampler(n_samples)
    values = f(samples)
    
    estimate = np.mean(values)
    standard_error = np.std(values) / np.sqrt(n_samples)
    
    return estimate, standard_error


def importance_sampling(
    f: Callable[[np.ndarray], np.ndarray],
    p: Callable[[np.ndarray], np.ndarray],
    q: Callable[[np.ndarray], np.ndarray],
    q_sampler: Callable[[int], np.ndarray],
    n_samples: int,
    seed: Optional[int] = None
) -> Tuple[float, float, float]:
    """
    Importance sampling for variance reduction.
    
    Computes E_p[f(X)] by sampling from proposal q(x):
    E_p[f(X)] = E_q[f(X) * p(X)/q(X)]
    
    Args:
        f: Function to integrate
        p: Target density (need not be normalized)
        q: Proposal density (must cover support of p)
        q_sampler: Sampler from proposal distribution
        n_samples: Number of samples
        seed: Random seed
    
    Returns:
        Tuple of (estimate, standard_error, effective_sample_size)
    
    Interview Question:
        Q: How do you choose a good proposal distribution?
        A: q should be close to |f(x)|p(x) for minimum variance.
           For rare events, shift q toward the rare region.
    
    Industrial Use Case:
        JPMorgan uses importance sampling for rare event estimation
        in Value-at-Risk calculations.
    """
    if seed is not None:
        np.random.seed(seed)
    
    samples = q_sampler(n_samples)
    
    # Importance weights: w = p(x) / q(x)
    p_vals = p(samples)
    q_vals = q(samples)
    
    # Avoid division by zero
    q_vals = np.maximum(q_vals, 1e-10)
    weights = p_vals / q_vals
    
    # Normalized importance weights (self-normalized estimator)
    normalized_weights = weights / np.sum(weights)
    
    # Weighted estimate
    f_vals = f(samples)
    estimate = np.sum(normalized_weights * f_vals)
    
    # Effective sample size
    ess = 1.0 / np.sum(normalized_weights ** 2)
    
    # Standard error (approximate)
    weighted_variance = np.sum(normalized_weights * (f_vals - estimate) ** 2)
    standard_error = np.sqrt(weighted_variance / ess)
    
    return estimate, standard_error, ess


def stratified_sampling(
    f: Callable[[np.ndarray], np.ndarray],
    bounds: List[Tuple[float, float]],
    n_strata_per_dim: int,
    samples_per_stratum: int,
    seed: Optional[int] = None
) -> Tuple[float, float]:
    """
    Stratified sampling for variance reduction in multi-dimensional integration.
    
    Partitions the domain into strata and samples proportionally from each.
    Reduces variance compared to simple random sampling.
    
    Args:
        f: Function to integrate over uniform distribution
        bounds: List of (low, high) tuples for each dimension
        n_strata_per_dim: Number of strata per dimension
        samples_per_stratum: Samples to draw per stratum
        seed: Random seed
    
    Returns:
        Tuple of (estimate, standard_error)
    
    Industrial Use Case:
        Pixar uses stratified sampling in path tracing to reduce
        noise in global illumination rendering.
    """
    if seed is not None:
        np.random.seed(seed)
    
    d = len(bounds)
    n_strata = n_strata_per_dim ** d
    
    # Volume of each stratum
    total_volume = np.prod([b[1] - b[0] for b in bounds])
    stratum_volume = total_volume / n_strata
    
    # Generate stratum indices
    strata_indices = np.arange(n_strata)
    
    estimates = []
    for idx in strata_indices:
        # Convert flat index to multi-dimensional stratum coordinates
        coords = []
        temp = idx
        for dim in range(d):
            coords.append(temp % n_strata_per_dim)
            temp //= n_strata_per_dim
        
        # Sample within this stratum
        samples = np.zeros((samples_per_stratum, d))
        for dim in range(d):
            low, high = bounds[dim]
            stratum_width = (high - low) / n_strata_per_dim
            stratum_low = low + coords[dim] * stratum_width
            stratum_high = stratum_low + stratum_width
            samples[:, dim] = np.random.uniform(stratum_low, stratum_high, samples_per_stratum)
        
        # Evaluate function
        values = np.array([f(s) for s in samples])
        estimates.append(np.mean(values))
    
    # Final estimate is average across strata scaled by volume
    estimate = np.mean(estimates) * total_volume
    standard_error = np.std(estimates) * total_volume / np.sqrt(n_strata)
    
    return estimate, standard_error


# =============================================================================
# BAYESIAN QUADRATURE
# =============================================================================

class BayesianQuadrature:
    """
    Bayesian Quadrature using Gaussian Process regression.
    
    Treats the integrand as a sample from a GP and computes
    a posterior distribution over the integral value.
    
    Attributes:
        kernel: Covariance kernel function
        noise_var: Observation noise variance
        X: Observed input locations
        y: Observed function values
    
    Industrial Use Case:
        AstraZeneca uses Bayesian Quadrature in drug discovery 
        to minimize expensive lab experiments while maintaining
        confidence in efficacy estimates.
    """
    
    def __init__(
        self,
        kernel: Callable[[np.ndarray, np.ndarray], np.ndarray],
        noise_var: float = 1e-6
    ):
        """
        Initialize Bayesian Quadrature.
        
        Args:
            kernel: Kernel function k(x, x') -> covariance
            noise_var: Noise variance for observations
        """
        self.kernel = kernel
        self.noise_var = noise_var
        self.X = None
        self.y = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BayesianQuadrature':
        """
        Fit the GP model to observations.
        
        Args:
            X: Input locations (n, d)
            y: Function values (n,)
        
        Returns:
            self
        """
        self.X = np.atleast_2d(X)
        self.y = np.atleast_1d(y)
        return self
    
    def add_observation(self, x: np.ndarray, y: float) -> 'BayesianQuadrature':
        """
        Add a single observation.
        
        Args:
            x: New input location
            y: Function value at x
        """
        x = np.atleast_2d(x)
        if self.X is None:
            self.X = x
            self.y = np.array([y])
        else:
            self.X = np.vstack([self.X, x])
            self.y = np.append(self.y, y)
        return self
    
    def integral_posterior(
        self,
        p_sampler: Callable[[int], np.ndarray],
        n_mc_samples: int = 1000
    ) -> Tuple[float, float]:
        """
        Compute posterior mean and variance of the integral.
        
        The integral is Z = ∫f(x)p(x)dx = E_p[f(x)]
        
        Args:
            p_sampler: Sampler from the target distribution p(x)
            n_mc_samples: MC samples for kernel expectations
        
        Returns:
            Tuple of (posterior_mean, posterior_variance)
        
        Interview Question:
            Q: What's the computational complexity of Bayesian Quadrature?
            A: O(N³) for N observations due to GP matrix inversion.
               Use sparse GPs or inducing points for scaling.
        """
        if self.X is None:
            raise ValueError("No observations. Call fit() or add_observation() first.")
        
        n = len(self.X)
        
        # Compute kernel matrix K(X, X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(self.X[i], self.X[j])
        K += self.noise_var * np.eye(n)
        
        # Compute z vector: z_i = E_p[k(X^*, x_i)]
        x_samples = p_sampler(n_mc_samples)
        z = np.zeros(n)
        for i in range(n):
            k_vals = np.array([self.kernel(x, self.X[i]) for x in x_samples])
            z[i] = np.mean(k_vals)
        
        # Compute kernel expectation: E_p,p'[k(X, X')]
        k_pp = 0.0
        for i in range(n_mc_samples):
            for j in range(n_mc_samples):
                k_pp += self.kernel(x_samples[i], x_samples[j])
        k_pp /= (n_mc_samples * n_mc_samples)
        
        # Posterior mean: z^T (K + σ²I)^{-1} y
        K_inv = np.linalg.inv(K)
        posterior_mean = z @ K_inv @ self.y
        
        # Posterior variance: E[k(X,X')] - z^T (K + σ²I)^{-1} z
        posterior_var = k_pp - z @ K_inv @ z
        posterior_var = max(0, posterior_var)  # Numerical stability
        
        return posterior_mean, np.sqrt(posterior_var)
    
    def suggest_next_point(
        self,
        candidates: np.ndarray,
        p: Callable[[np.ndarray], float]
    ) -> int:
        """
        Suggest next evaluation point for active learning.
        
        Selects the point that maximizes expected information gain
        (reduction in integral variance).
        
        Args:
            candidates: Array of candidate points (m, d)
            p: Target density function
        
        Returns:
            Index of best candidate
        """
        if self.X is None:
            # No observations yet - pick point with highest density
            densities = np.array([p(x) for x in candidates])
            return np.argmax(densities)
        
        # Compute posterior variance at each candidate
        n = len(self.X)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self.kernel(self.X[i], self.X[j])
        K += self.noise_var * np.eye(n)
        K_inv = np.linalg.inv(K)
        
        # Score = posterior variance * density (simplified acquisition)
        scores = np.zeros(len(candidates))
        for i, x in enumerate(candidates):
            k_x = np.array([self.kernel(x, xi) for xi in self.X])
            var = self.kernel(x, x) - k_x @ K_inv @ k_x
            scores[i] = var * p(x)
        
        return np.argmax(scores)


def rbf_kernel(length_scale: float = 1.0, variance: float = 1.0) -> Callable:
    """
    Create an RBF (Gaussian) kernel function.
    
    k(x, x') = σ² exp(-||x - x'||² / (2ℓ²))
    
    Args:
        length_scale: Kernel length scale ℓ
        variance: Kernel variance σ²
    
    Returns:
        Kernel function
    """
    def kernel(x1: np.ndarray, x2: np.ndarray) -> float:
        x1 = np.atleast_1d(x1)
        x2 = np.atleast_1d(x2)
        sq_dist = np.sum((x1 - x2) ** 2)
        return variance * np.exp(-sq_dist / (2 * length_scale ** 2))
    return kernel


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def compare_integration_methods(
    f: Callable,
    a: float,
    b: float,
    true_value: float,
    n_points: List[int]
) -> dict:
    """
    Compare convergence of different integration methods.
    
    Args:
        f: Function to integrate
        a, b: Integration bounds
        true_value: Known true integral value
        n_points: List of node counts to test
    
    Returns:
        Dictionary with errors for each method
    """
    results = {
        'n_points': n_points,
        'trapezoidal_errors': [],
        'simpsons_errors': [],
        'gauss_legendre_errors': []
    }
    
    for n in n_points:
        trap_result = trapezoidal_rule(f, a, b, n)
        simp_result = simpsons_rule(f, a, b, n)
        gl_result = gauss_legendre(f, a, b, min(n, 50))  # GL limited
        
        results['trapezoidal_errors'].append(abs(trap_result - true_value))
        results['simpsons_errors'].append(abs(simp_result - true_value))
        results['gauss_legendre_errors'].append(abs(gl_result - true_value))
    
    return results


def monte_carlo_convergence(
    f: Callable,
    sampler: Callable,
    true_value: float,
    sample_sizes: List[int],
    n_trials: int = 10
) -> dict:
    """
    Analyze Monte Carlo convergence rate.
    
    Args:
        f: Function to integrate
        sampler: Sample generator
        true_value: Known true value
        sample_sizes: List of sample sizes
        n_trials: Number of trials per size
    
    Returns:
        Dictionary with mean errors and std errors
    """
    results = {
        'sample_sizes': sample_sizes,
        'mean_errors': [],
        'std_errors': []
    }
    
    for n in sample_sizes:
        errors = []
        for trial in range(n_trials):
            estimate, _ = monte_carlo_integrate(f, sampler, n)
            errors.append(abs(estimate - true_value))
        
        results['mean_errors'].append(np.mean(errors))
        results['std_errors'].append(np.std(errors))
    
    return results
