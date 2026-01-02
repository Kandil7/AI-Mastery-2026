"""
Markov Chain Monte Carlo (MCMC) Methods
========================================

This module implements MCMC sampling algorithms from scratch using NumPy.
These methods are essential for Bayesian inference when direct sampling
from the posterior distribution is intractable.

Methods included:
- Metropolis-Hastings Algorithm
- Hamiltonian Monte Carlo (HMC)
- No-U-Turn Sampler (NUTS) - simplified version
- MCMC Diagnostics (R-hat, ESS, acceptance rate)

Industrial Applications:
- Airbnb: Dynamic pricing optimization
- Uber: Demand forecasting with uncertainty
- Goldman Sachs: Risk modeling in high dimensions
- Pfizer: Drug efficacy estimation from clinical trials
"""

import numpy as np
from typing import Callable, Tuple, Optional, List, Dict, Union
from dataclasses import dataclass
import warnings


# =============================================================================
# DATA CLASSES FOR MCMC RESULTS
# =============================================================================

@dataclass
class MCMCResult:
    """
    Container for MCMC sampling results.
    
    Attributes:
        samples: Array of posterior samples (n_samples, d)
        acceptance_rate: Fraction of accepted proposals
        log_probs: Log probability at each sample
        diagnostics: Dictionary with additional metrics
    """
    samples: np.ndarray
    acceptance_rate: float
    log_probs: np.ndarray
    diagnostics: Dict


# =============================================================================
# METROPOLIS-HASTINGS ALGORITHM
# =============================================================================

def metropolis_hastings(
    log_prob: Callable[[np.ndarray], float],
    initial_state: np.ndarray,
    n_samples: int,
    proposal_std: float = 1.0,
    n_burnin: int = 1000,
    seed: Optional[int] = None
) -> MCMCResult:
    """
    Metropolis-Hastings MCMC sampler with Gaussian proposal.
    
    The algorithm generates samples from a target distribution p(x)
    by constructing a Markov chain that converges to p(x) as its
    stationary distribution.
    
    Algorithm:
        1. Start at x^(0)
        2. For t = 1, 2, ..., n_samples:
           a. Propose x' ~ N(x^(t-1), σ²I)
           b. Compute α = min(1, p(x')/p(x^(t-1)))
           c. Accept x' with probability α, else stay at x^(t-1)
    
    Args:
        log_prob: Function computing log p(x) (up to constant)
        initial_state: Starting point (d,)
        n_samples: Number of samples to generate
        proposal_std: Standard deviation of Gaussian proposal
        n_burnin: Number of burn-in samples to discard
        seed: Random seed for reproducibility
    
    Returns:
        MCMCResult with samples and diagnostics
    
    Example:
        >>> # Sample from a 2D Gaussian
        >>> log_prob = lambda x: -0.5 * np.sum(x**2)
        >>> result = metropolis_hastings(log_prob, np.zeros(2), 10000)
        >>> print(f"Mean: {result.samples.mean(axis=0)}")
    
    Industrial Use Case:
        Airbnb uses Metropolis-Hastings variants for pricing models
        where the posterior over price elasticity is complex and
        multi-modal due to regional variations.
    
    Interview Question:
        Q: Why do we need burn-in in MCMC?
        A: The chain starts from an arbitrary point and needs time
           to converge to the stationary distribution. Burn-in
           discards samples before convergence.
    """
    if seed is not None:
        np.random.seed(seed)
    
    d = len(initial_state)
    total_samples = n_samples + n_burnin
    
    # Storage
    samples = np.zeros((total_samples, d))
    log_probs = np.zeros(total_samples)
    accepted = 0
    
    # Initialize
    current_state = initial_state.copy()
    current_log_prob = log_prob(current_state)
    
    for t in range(total_samples):
        # Propose new state
        proposal = current_state + proposal_std * np.random.randn(d)
        proposal_log_prob = log_prob(proposal)
        
        # Compute acceptance probability (log scale for numerical stability)
        log_alpha = proposal_log_prob - current_log_prob
        
        # Accept or reject
        if np.log(np.random.rand()) < log_alpha:
            current_state = proposal
            current_log_prob = proposal_log_prob
            if t >= n_burnin:
                accepted += 1
        
        samples[t] = current_state
        log_probs[t] = current_log_prob
    
    # Remove burn-in
    samples = samples[n_burnin:]
    log_probs = log_probs[n_burnin:]
    acceptance_rate = accepted / n_samples
    
    # Compute diagnostics
    diagnostics = {
        'n_burnin': n_burnin,
        'proposal_std': proposal_std,
        'ess': effective_sample_size(samples),
        'mean': np.mean(samples, axis=0),
        'std': np.std(samples, axis=0)
    }
    
    return MCMCResult(
        samples=samples,
        acceptance_rate=acceptance_rate,
        log_probs=log_probs,
        diagnostics=diagnostics
    )


# =============================================================================
# HAMILTONIAN MONTE CARLO
# =============================================================================

class HamiltonianMonteCarlo:
    """
    Hamiltonian Monte Carlo (HMC) sampler.
    
    HMC uses Hamiltonian dynamics to propose samples, leading to
    much higher acceptance rates and lower autocorrelation than
    random-walk Metropolis-Hastings, especially in high dimensions.
    
    The algorithm treats the target distribution as the potential energy
    U(q) = -log p(q), and introduces auxiliary momentum variables p.
    The joint distribution is:
        H(q, p) = U(q) + K(p) where K(p) = 0.5 * p^T M^{-1} p
    
    Attributes:
        log_prob: Log probability function
        grad_log_prob: Gradient of log probability
        step_size: Leapfrog step size ε
        n_leapfrog: Number of leapfrog steps L
        mass_matrix: Diagonal mass matrix (for preconditioning)
    
    Industrial Use Case:
        Stan (a probabilistic programming language) uses HMC as its
        core inference engine. Companies like Facebook, Google, and
        pharmaceutical giants use Stan for complex Bayesian models.
    """
    
    def __init__(
        self,
        log_prob: Callable[[np.ndarray], float],
        grad_log_prob: Callable[[np.ndarray], np.ndarray],
        step_size: float = 0.1,
        n_leapfrog: int = 10,
        mass_matrix: Optional[np.ndarray] = None
    ):
        """
        Initialize HMC sampler.
        
        Args:
            log_prob: Function computing log p(x)
            grad_log_prob: Function computing ∇ log p(x)
            step_size: Leapfrog integrator step size
            n_leapfrog: Number of leapfrog steps per proposal
            mass_matrix: Diagonal of mass matrix (default: identity)
        """
        self.log_prob = log_prob
        self.grad_log_prob = grad_log_prob
        self.step_size = step_size
        self.n_leapfrog = n_leapfrog
        self.mass_matrix = mass_matrix
    
    def _leapfrog(
        self,
        q: np.ndarray,
        p: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Leapfrog integrator for Hamiltonian dynamics.
        
        The leapfrog method is a symplectic integrator that preserves
        volume in phase space, which is crucial for detailed balance.
        
        Steps:
            1. Half step for momentum: p = p + (ε/2) ∇ log p(q)
            2. Full step for position: q = q + ε M^{-1} p
            3. Half step for momentum: p = p + (ε/2) ∇ log p(q)
        
        Args:
            q: Position (current sample)
            p: Momentum (auxiliary variable)
        
        Returns:
            New (q, p) after L leapfrog steps
        """
        q = q.copy()
        p = p.copy()
        
        # Mass matrix inverse (diagonal)
        if self.mass_matrix is not None:
            M_inv = 1.0 / self.mass_matrix
        else:
            M_inv = np.ones(len(q))
        
        # Half step for momentum
        p = p + 0.5 * self.step_size * self.grad_log_prob(q)
        
        # Full steps
        for _ in range(self.n_leapfrog - 1):
            q = q + self.step_size * M_inv * p
            p = p + self.step_size * self.grad_log_prob(q)
        
        # Last full step for position
        q = q + self.step_size * M_inv * p
        
        # Half step for momentum
        p = p + 0.5 * self.step_size * self.grad_log_prob(q)
        
        # Negate momentum for reversibility
        p = -p
        
        return q, p
    
    def _hamiltonian(self, q: np.ndarray, p: np.ndarray) -> float:
        """
        Compute Hamiltonian H(q, p) = U(q) + K(p).
        
        U(q) = -log p(q) is the potential energy
        K(p) = 0.5 * p^T M^{-1} p is the kinetic energy
        """
        if self.mass_matrix is not None:
            M_inv = 1.0 / self.mass_matrix
        else:
            M_inv = np.ones(len(q))
        
        potential = -self.log_prob(q)
        kinetic = 0.5 * np.sum(M_inv * p ** 2)
        
        return potential + kinetic
    
    def sample(
        self,
        initial_state: np.ndarray,
        n_samples: int,
        n_burnin: int = 1000,
        seed: Optional[int] = None,
        adapt_step_size: bool = True,
        target_accept: float = 0.65
    ) -> MCMCResult:
        """
        Generate samples using HMC.
        
        Args:
            initial_state: Starting point (d,)
            n_samples: Number of samples to generate
            n_burnin: Number of burn-in samples
            seed: Random seed
            adapt_step_size: Whether to adapt step size during burn-in
            target_accept: Target acceptance rate for adaptation
        
        Returns:
            MCMCResult with samples and diagnostics
        
        Interview Question:
            Q: What is the optimal acceptance rate for HMC?
            A: Around 65-80%. Too high means step size is too small
               (inefficient). Too low means proposals are rejected
               (wasted computation).
        """
        if seed is not None:
            np.random.seed(seed)
        
        d = len(initial_state)
        total_samples = n_samples + n_burnin
        
        # Mass matrix
        if self.mass_matrix is not None:
            M = self.mass_matrix
        else:
            M = np.ones(d)
        
        # Storage
        samples = np.zeros((total_samples, d))
        log_probs = np.zeros(total_samples)
        accepted = 0
        step_size = self.step_size
        
        # Dual averaging parameters for step size adaptation
        mu = np.log(10 * step_size)
        log_step_size_bar = 0.0
        H_bar = 0.0
        gamma = 0.05
        t0 = 10
        kappa = 0.75
        
        current_q = initial_state.copy()
        
        for t in range(total_samples):
            # Sample momentum
            p = np.random.randn(d) * np.sqrt(M)
            
            current_p = p.copy()
            current_H = self._hamiltonian(current_q, current_p)
            
            # Leapfrog integration
            proposed_q, proposed_p = self._leapfrog(current_q, p)
            proposed_H = self._hamiltonian(proposed_q, proposed_p)
            
            # Metropolis acceptance step
            log_alpha = current_H - proposed_H
            alpha = min(1.0, np.exp(log_alpha))
            
            if np.random.rand() < alpha:
                current_q = proposed_q
                if t >= n_burnin:
                    accepted += 1
            
            samples[t] = current_q
            log_probs[t] = self.log_prob(current_q)
            
            # Step size adaptation during burn-in
            if adapt_step_size and t < n_burnin:
                w = 1.0 / (t + t0)
                H_bar = (1 - w) * H_bar + w * (target_accept - alpha)
                log_step_size = mu - np.sqrt(t + 1) / gamma * H_bar
                step_size = np.exp(log_step_size)
                
                w2 = (t + 1) ** (-kappa)
                log_step_size_bar = w2 * log_step_size + (1 - w2) * log_step_size_bar
                
                self.step_size = step_size
        
        # Use adapted step size
        if adapt_step_size:
            self.step_size = np.exp(log_step_size_bar)
        
        # Remove burn-in
        samples = samples[n_burnin:]
        log_probs = log_probs[n_burnin:]
        acceptance_rate = accepted / n_samples
        
        diagnostics = {
            'n_burnin': n_burnin,
            'final_step_size': self.step_size,
            'n_leapfrog': self.n_leapfrog,
            'ess': effective_sample_size(samples),
            'mean': np.mean(samples, axis=0),
            'std': np.std(samples, axis=0)
        }
        
        return MCMCResult(
            samples=samples,
            acceptance_rate=acceptance_rate,
            log_probs=log_probs,
            diagnostics=diagnostics
        )


# =============================================================================
# NO-U-TURN SAMPLER (NUTS) - SIMPLIFIED
# =============================================================================

def nuts_sampler(
    log_prob: Callable[[np.ndarray], float],
    grad_log_prob: Callable[[np.ndarray], np.ndarray],
    initial_state: np.ndarray,
    n_samples: int,
    n_burnin: int = 1000,
    step_size: float = 0.1,
    max_tree_depth: int = 10,
    seed: Optional[int] = None
) -> MCMCResult:
    """
    No-U-Turn Sampler (NUTS) - simplified implementation.
    
    NUTS automatically tunes the number of leapfrog steps by building
    a binary tree of states and stopping when the trajectory starts
    to turn back on itself (the "U-turn" criterion).
    
    This is a simplified version that demonstrates the core concepts.
    For production use, consider PyMC, Stan, or TensorFlow Probability.
    
    Args:
        log_prob: Log probability function
        grad_log_prob: Gradient of log probability
        initial_state: Starting point
        n_samples: Number of samples
        n_burnin: Burn-in period
        step_size: Leapfrog step size
        max_tree_depth: Maximum binary tree depth
        seed: Random seed
    
    Returns:
        MCMCResult with samples and diagnostics
    
    Industrial Use Case:
        NUTS is the default algorithm in Stan, used by:
        - Facebook for experimentation analysis
        - Spotify for music recommendation models
        - Pharmaceutical companies for clinical trial analysis
    
    Interview Question:
        Q: Why is NUTS preferred over standard HMC?
        A: HMC requires manual tuning of L (leapfrog steps). Too few
           gives random-walk behavior, too many wastes computation.
           NUTS automatically finds optimal L per iteration.
    """
    if seed is not None:
        np.random.seed(seed)
    
    d = len(initial_state)
    total_samples = n_samples + n_burnin
    
    samples = np.zeros((total_samples, d))
    log_probs = np.zeros(total_samples)
    accepted = 0
    
    current_q = initial_state.copy()
    
    def leapfrog(q, p, eps):
        """Single leapfrog step."""
        p = p + 0.5 * eps * grad_log_prob(q)
        q = q + eps * p
        p = p + 0.5 * eps * grad_log_prob(q)
        return q, p
    
    def hamiltonian(q, p):
        return -log_prob(q) + 0.5 * np.sum(p ** 2)
    
    def u_turn(q_minus, q_plus, p_minus, p_plus):
        """Check if trajectory is making a U-turn."""
        dq = q_plus - q_minus
        return (np.dot(dq, p_minus) < 0) or (np.dot(dq, p_plus) < 0)
    
    for t in range(total_samples):
        # Sample momentum
        p0 = np.random.randn(d)
        H0 = hamiltonian(current_q, p0)
        
        # Slice variable
        log_u = np.log(np.random.rand()) + log_prob(current_q) - 0.5 * np.sum(p0 ** 2)
        
        # Initialize tree
        q_minus = current_q.copy()
        q_plus = current_q.copy()
        p_minus = p0.copy()
        p_plus = p0.copy()
        
        j = 0  # Tree depth
        n = 1  # Number of acceptable states
        s = 1  # Continue building?
        
        candidate_q = current_q.copy()
        
        while s == 1 and j < max_tree_depth:
            # Choose direction
            direction = 2 * (np.random.rand() < 0.5) - 1
            
            # Build tree in that direction
            if direction == -1:
                q_minus, p_minus, _, _, q_prime, n_prime, s_prime = \
                    _build_tree_simple(q_minus, p_minus, log_u, direction, 
                                       j, step_size, log_prob, grad_log_prob)
            else:
                _, _, q_plus, p_plus, q_prime, n_prime, s_prime = \
                    _build_tree_simple(q_plus, p_plus, log_u, direction,
                                       j, step_size, log_prob, grad_log_prob)
            
            # Metropolis-Hastings step
            if s_prime == 1 and np.random.rand() < min(1, n_prime / n):
                candidate_q = q_prime
            
            # Update counters
            n = n + n_prime
            s = s_prime * (1 - int(u_turn(q_minus, q_plus, p_minus, p_plus)))
            j += 1
        
        # Accept candidate
        if not np.allclose(candidate_q, current_q):
            accepted += 1 if t >= n_burnin else 0
        
        current_q = candidate_q
        samples[t] = current_q
        log_probs[t] = log_prob(current_q)
    
    # Remove burn-in
    samples = samples[n_burnin:]
    log_probs = log_probs[n_burnin:]
    acceptance_rate = accepted / n_samples if n_samples > 0 else 0.0
    
    diagnostics = {
        'n_burnin': n_burnin,
        'step_size': step_size,
        'max_tree_depth': max_tree_depth,
        'ess': effective_sample_size(samples),
        'mean': np.mean(samples, axis=0),
        'std': np.std(samples, axis=0)
    }
    
    return MCMCResult(
        samples=samples,
        acceptance_rate=acceptance_rate,
        log_probs=log_probs,
        diagnostics=diagnostics
    )


def _build_tree_simple(
    q: np.ndarray,
    p: np.ndarray,
    log_u: float,
    direction: int,
    depth: int,
    step_size: float,
    log_prob: Callable,
    grad_log_prob: Callable
) -> Tuple:
    """
    Recursively build the binary tree for NUTS (simplified).
    
    Returns:
        q_minus, p_minus, q_plus, p_plus, q_prime, n_prime, s_prime
    """
    eps = direction * step_size
    
    if depth == 0:
        # Base case: single leapfrog step
        p = p + 0.5 * eps * grad_log_prob(q)
        q = q + eps * p
        p = p + 0.5 * eps * grad_log_prob(q)
        
        # Check if state is acceptable
        log_joint = log_prob(q) - 0.5 * np.sum(p ** 2)
        n_prime = int(log_u <= log_joint)
        s_prime = int(log_u < log_joint + 1000)  # Numerical threshold
        
        return q, p, q, p, q, n_prime, s_prime
    else:
        # Recursion
        q_minus, p_minus, q_plus, p_plus, q_prime, n_prime, s_prime = \
            _build_tree_simple(q, p, log_u, direction, depth - 1,
                               step_size, log_prob, grad_log_prob)
        
        if s_prime == 1:
            if direction == -1:
                q_minus, p_minus, _, _, q_double, n_double, s_double = \
                    _build_tree_simple(q_minus, p_minus, log_u, direction,
                                       depth - 1, step_size, log_prob, grad_log_prob)
            else:
                _, _, q_plus, p_plus, q_double, n_double, s_double = \
                    _build_tree_simple(q_plus, p_plus, log_u, direction,
                                       depth - 1, step_size, log_prob, grad_log_prob)
            
            if np.random.rand() < n_double / max(n_prime + n_double, 1):
                q_prime = q_double
            
            dq = q_plus - q_minus
            s_prime = s_double * int(np.dot(dq, p_minus) >= 0) * int(np.dot(dq, p_plus) >= 0)
            n_prime = n_prime + n_double
        
        return q_minus, p_minus, q_plus, p_plus, q_prime, n_prime, s_prime


# =============================================================================
# MCMC DIAGNOSTICS
# =============================================================================

def effective_sample_size(samples: np.ndarray) -> np.ndarray:
    """
    Compute effective sample size (ESS) for MCMC samples.
    
    ESS estimates the number of independent samples equivalent to
    the correlated MCMC samples. Higher is better.
    
    ESS = N / (1 + 2 * Σ_k ρ_k)
    
    where ρ_k is the autocorrelation at lag k.
    
    Args:
        samples: MCMC samples (n_samples, d)
    
    Returns:
        ESS for each dimension (d,)
    
    Interview Question:
        Q: What's a good ESS?
        A: Rule of thumb: ESS > 100 per parameter for reliable estimates.
           ESS > 400 for accurate tail probability estimation.
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    
    n, d = samples.shape
    ess = np.zeros(d)
    
    for dim in range(d):
        x = samples[:, dim]
        x = x - np.mean(x)
        
        # Compute autocorrelation using FFT
        n_padded = 2 ** int(np.ceil(np.log2(2 * n - 1)))
        fft_x = np.fft.fft(x, n_padded)
        acf = np.fft.ifft(fft_x * np.conj(fft_x))[:n].real
        acf = acf / acf[0]
        
        # Sum autocorrelations until they become negative
        tau = 1.0
        for k in range(1, n):
            if acf[k] < 0.05:  # Cutoff for numerical stability
                break
            tau += 2 * acf[k]
        
        ess[dim] = n / tau
    
    return ess


def gelman_rubin_diagnostic(
    chains: List[np.ndarray]
) -> np.ndarray:
    """
    Compute Gelman-Rubin R-hat diagnostic for multiple chains.
    
    R-hat compares within-chain and between-chain variance.
    Values close to 1.0 indicate convergence (typically R-hat < 1.05).
    
    Args:
        chains: List of MCMC chains, each (n_samples, d)
    
    Returns:
        R-hat for each dimension (d,)
    
    Interview Question:
        Q: Why run multiple chains?
        A: Multiple chains from different starting points help
           diagnose convergence. If chains haven't mixed, the
           posterior estimates may be unreliable.
    """
    n_chains = len(chains)
    n_samples = chains[0].shape[0]
    d = chains[0].shape[1] if chains[0].ndim > 1 else 1
    
    # Stack chains
    all_chains = np.array(chains)  # (n_chains, n_samples, d)
    if all_chains.ndim == 2:
        all_chains = all_chains[:, :, np.newaxis]
    
    # Within-chain variance
    W = np.mean(np.var(all_chains, axis=1, ddof=1), axis=0)
    
    # Between-chain variance
    chain_means = np.mean(all_chains, axis=1)
    B = n_samples * np.var(chain_means, axis=0, ddof=1)
    
    # Pooled variance estimate
    var_plus = (n_samples - 1) / n_samples * W + B / n_samples
    
    # R-hat
    r_hat = np.sqrt(var_plus / W)
    
    return r_hat


def mcmc_diagnostics(
    result: MCMCResult,
    param_names: Optional[List[str]] = None
) -> Dict:
    """
    Compute comprehensive MCMC diagnostics.
    
    Args:
        result: MCMCResult from a sampler
        param_names: Optional names for parameters
    
    Returns:
        Dictionary with diagnostic information
    
    Industrial Use Case:
        At Airbnb, MCMC diagnostics are automatically computed
        and alerts are triggered if ESS < 100 or acceptance
        rate falls outside [0.15, 0.85].
    """
    samples = result.samples
    n_samples, d = samples.shape
    
    if param_names is None:
        param_names = [f'param_{i}' for i in range(d)]
    
    ess = effective_sample_size(samples)
    
    diagnostics = {
        'n_samples': n_samples,
        'acceptance_rate': result.acceptance_rate,
        'ess': dict(zip(param_names, ess)),
        'ess_per_sample': dict(zip(param_names, ess / n_samples)),
        'summary': {}
    }
    
    for i, name in enumerate(param_names):
        col = samples[:, i]
        diagnostics['summary'][name] = {
            'mean': np.mean(col),
            'std': np.std(col),
            'median': np.median(col),
            'q_2.5': np.percentile(col, 2.5),
            'q_97.5': np.percentile(col, 97.5),
            'ess': ess[i]
        }
    
    # Warnings
    warnings_list = []
    if result.acceptance_rate < 0.1:
        warnings_list.append("Low acceptance rate (<10%). Consider reducing step size.")
    if result.acceptance_rate > 0.9:
        warnings_list.append("High acceptance rate (>90%). Consider increasing step size.")
    if np.any(ess < 100):
        warnings_list.append("Low ESS detected (<100). Consider running longer chains.")
    
    diagnostics['warnings'] = warnings_list
    
    return diagnostics


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def trace_plot_data(samples: np.ndarray) -> Dict:
    """
    Prepare data for trace plots.
    
    Args:
        samples: MCMC samples (n_samples, d)
    
    Returns:
        Dictionary with iteration indices and samples
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    
    return {
        'iterations': np.arange(len(samples)),
        'samples': samples
    }


def autocorrelation(samples: np.ndarray, max_lag: int = 100) -> np.ndarray:
    """
    Compute autocorrelation function for MCMC samples.
    
    Args:
        samples: MCMC samples (n_samples,) or (n_samples, d)
        max_lag: Maximum lag to compute
    
    Returns:
        Autocorrelation values (max_lag,) or (max_lag, d)
    """
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)
    
    n, d = samples.shape
    max_lag = min(max_lag, n - 1)
    
    acf = np.zeros((max_lag, d))
    
    for dim in range(d):
        x = samples[:, dim]
        x = x - np.mean(x)
        var = np.var(x)
        
        for lag in range(max_lag):
            if var > 0:
                acf[lag, dim] = np.mean(x[lag:] * x[:n-lag]) / var
            else:
                acf[lag, dim] = 1.0 if lag == 0 else 0.0
    
    return acf.squeeze()


def thinning(samples: np.ndarray, thin: int) -> np.ndarray:
    """
    Thin MCMC samples to reduce autocorrelation.
    
    Args:
        samples: MCMC samples (n_samples, d)
        thin: Keep every thin-th sample
    
    Returns:
        Thinned samples
    """
    return samples[::thin]


# =============================================================================
# EXAMPLE: BAYESIAN LOGISTIC REGRESSION WITH HMC
# =============================================================================

def bayesian_logistic_regression_hmc(
    X: np.ndarray,
    y: np.ndarray,
    n_samples: int = 5000,
    prior_std: float = 10.0,
    seed: Optional[int] = None
) -> MCMCResult:
    """
    Bayesian logistic regression using HMC.
    
    Model:
        β ~ N(0, prior_std² I)
        y_i ~ Bernoulli(sigmoid(X_i @ β))
    
    Args:
        X: Features (n, d)
        y: Binary labels (n,)
        n_samples: Number of posterior samples
        prior_std: Prior standard deviation
        seed: Random seed
    
    Returns:
        MCMCResult with posterior samples
    
    Industrial Use Case:
        This is the foundation for uncertainty-aware classification
        used in medical diagnosis (Pfizer), fraud detection (PayPal),
        and recommendation systems (Netflix).
    """
    n, d = X.shape
    
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    
    def log_prob(beta):
        # Prior: N(0, prior_std² I)
        log_prior = -0.5 * np.sum(beta ** 2) / (prior_std ** 2)
        
        # Likelihood
        logits = X @ beta
        log_likelihood = np.sum(y * logits - np.log(1 + np.exp(np.clip(logits, -500, 500))))
        
        return log_prior + log_likelihood
    
    def grad_log_prob(beta):
        # Gradient of log prior
        grad_prior = -beta / (prior_std ** 2)
        
        # Gradient of log likelihood
        probs = sigmoid(X @ beta)
        grad_likelihood = X.T @ (y - probs)
        
        return grad_prior + grad_likelihood
    
    # Initial state
    initial_beta = np.zeros(d)
    
    # Run HMC
    hmc = HamiltonianMonteCarlo(
        log_prob=log_prob,
        grad_log_prob=grad_log_prob,
        step_size=0.01,
        n_leapfrog=20
    )
    
    return hmc.sample(initial_beta, n_samples, n_burnin=1000, seed=seed)
