"""
Pure Python Implementation of Probability Components ("White-Box").
Avoids scipy/numpy to understand distributions and samplers from scratch.
"""
import math
import random
from typing import List, Callable, Union

class Gaussian:
    """Univariate Gaussian Distribution"""
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std
        self.variance = std ** 2

    def pdf(self, x: float) -> float:
        """Probability Density Function"""
        coeff = 1.0 / (self.std * math.sqrt(2 * math.pi))
        exponent = -0.5 * ((x - self.mean) / self.std) ** 2
        return coeff * math.exp(exponent)

    def cdf(self, x: float) -> float:
        """Cumulative Distribution Function (Error function approximation)"""
        return 0.5 * (1 + math.erf((x - self.mean) / (self.std * math.sqrt(2))))

    def sample(self) -> float:
        """Box-Muller Transform for sampling"""
        u1 = random.random()
        u2 = random.random()
        z0 = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return self.mean + z0 * self.std

def kullback_leibler_divergence(p: 'Gaussian', q: 'Gaussian') -> float:
    """
    Analytical KL Divergence between two Gaussians: KL(P || Q)
    Formula: log(s2/s1) + (s1^2 + (m1-m2)^2)/(2s2^2) - 1/2
    """
    term1 = math.log(q.std / p.std)
    term2 = (p.variance + (p.mean - q.mean)**2) / (2 * q.variance)
    return term1 + term2 - 0.5

def metropolis_hastings(
    log_prob_func: Callable[[float], float], 
    proposal_width: float = 0.5, 
    n_samples: int = 1000,
    start: float = 0.0
) -> List[float]:
    """
    MCMC Sampling using Metropolis-Hastings.
    Args:
        log_prob_func: Function returning log probability of a state
    """
    samples = []
    current = start
    current_log_prob = log_prob_func(current)
    
    for _ in range(n_samples):
        # 1. Propose new state (Gaussian random walk)
        proposal = current + (random.random() - 0.5) * 2 * proposal_width
        
        # 2. Accept/Reject
        proposal_log_prob = log_prob_func(proposal)
        
        # Acceptance ratio (log space): p(new)/p(old) -> log(p_new) - log(p_old)
        log_acceptance_ratio = proposal_log_prob - current_log_prob
        
        if math.log(random.random()) < log_acceptance_ratio:
            current = proposal
            current_log_prob = proposal_log_prob
            
        samples.append(current)
        
    return samples
