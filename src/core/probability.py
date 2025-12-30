"""
Probability Module

This module implements fundamental probability concepts, distributions,
and statistical methods from scratch using NumPy.
"""

import numpy as np
from typing import Union, List, Tuple, Callable
from scipy.special import gamma
import math


def uniform_pdf(x: Union[float, np.ndarray], a: float = 0, b: float = 1) -> Union[float, np.ndarray]:
    """
    Probability density function for uniform distribution.
    
    Args:
        x: Value(s) at which to evaluate the PDF
        a: Lower bound
        b: Upper bound
        
    Returns:
        PDF value(s)
    """
    x = np.asarray(x)
    pdf = np.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)
    return pdf


def uniform_cdf(x: Union[float, np.ndarray], a: float = 0, b: float = 1) -> Union[float, np.ndarray]:
    """
    Cumulative distribution function for uniform distribution.
    
    Args:
        x: Value(s) at which to evaluate the CDF
        a: Lower bound
        b: Upper bound
        
    Returns:
        CDF value(s)
    """
    x = np.asarray(x)
    cdf = np.where(x < a, 0.0, np.where(x > b, 1.0, (x - a) / (b - a)))
    return cdf


def gaussian_pdf(x: Union[float, np.ndarray], mu: float = 0, sigma: float = 1) -> Union[float, np.ndarray]:
    """
    Probability density function for normal (Gaussian) distribution.
    
    Args:
        x: Value(s) at which to evaluate the PDF
        mu: Mean
        sigma: Standard deviation
        
    Returns:
        PDF value(s)
    """
    x = np.asarray(x)
    coefficient = 1.0 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coefficient * np.exp(exponent)


def gaussian_cdf(x: Union[float, np.ndarray], mu: float = 0, sigma: float = 1) -> Union[float, np.ndarray]:
    """
    Cumulative distribution function for normal (Gaussian) distribution.
    Using approximation of error function.
    
    Args:
        x: Value(s) at which to evaluate the CDF
        mu: Mean
        sigma: Standard deviation
        
    Returns:
        CDF value(s)
    """
    x = np.asarray(x)
    # Using the error function approximation: CDF = 0.5 * (1 + erf((x - mu) / (sigma * sqrt(2))))
    return 0.5 * (1 + erf_approx((x - mu) / (sigma * np.sqrt(2))))


def erf_approx(x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Approximation of the error function using a rational approximation.
    
    Args:
        x: Value(s) at which to evaluate erf
        
    Returns:
        Approximate erf value(s)
    """
    x = np.asarray(x)
    # Save sign for later
    sign = np.sign(x)
    x = np.abs(x)
    
    # Constants for rational approximation
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * np.exp(-x * x)
    
    return sign * y


def beta_pdf(x: Union[float, np.ndarray], alpha: float, beta: float) -> Union[float, np.ndarray]:
    """
    Probability density function for beta distribution.
    
    Args:
        x: Value(s) at which to evaluate the PDF (between 0 and 1)
        alpha: First shape parameter
        beta: Second shape parameter
        
    Returns:
        PDF value(s)
    """
    x = np.asarray(x)
    # Beta function: B(alpha, beta) = Gamma(alpha) * Gamma(beta) / Gamma(alpha + beta)
    beta_func = (gamma(alpha) * gamma(beta)) / gamma(alpha + beta)
    pdf = (x ** (alpha - 1) * (1 - x) ** (beta - 1)) / beta_func
    pdf = np.where((x < 0) | (x > 1), 0.0, pdf)
    return pdf


def gamma_pdf(x: Union[float, np.ndarray], k: float, theta: float) -> Union[float, np.ndarray]:
    """
    Probability density function for gamma distribution.
    
    Args:
        x: Value(s) at which to evaluate the PDF (positive values)
        k: Shape parameter
        theta: Scale parameter
        
    Returns:
        PDF value(s)
    """
    x = np.asarray(x)
    # Gamma PDF: (x^(k-1) * e^(-x/theta)) / (Gamma(k) * theta^k)
    numerator = (x ** (k - 1)) * np.exp(-x / theta)
    denominator = gamma(k) * (theta ** k)
    pdf = numerator / denominator
    pdf = np.where(x <= 0, 0.0, pdf)
    return pdf


def sample_gaussian(mean: float = 0, std: float = 1, size: int = 1) -> Union[float, np.ndarray]:
    """
    Sample from a Gaussian distribution using Box-Muller transform.
    
    Args:
        mean: Mean of the distribution
        std: Standard deviation of the distribution
        size: Number of samples to generate
        
    Returns:
        Sample(s) from the Gaussian distribution
    """
    if size == 1:
        # Box-Muller transform for single sample
        u1 = np.random.uniform(0, 1)
        u2 = np.random.uniform(0, 1)
        z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        return mean + std * z0
    else:
        # Generate multiple samples
        samples = np.zeros(size)
        for i in range(0, size, 2):
            u1 = np.random.uniform(0, 1)
            u2 = np.random.uniform(0, 1)
            z0 = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            z1 = np.sqrt(-2 * np.log(u1)) * np.sin(2 * np.pi * u2)
            
            samples[i] = mean + std * z0
            if i + 1 < size:
                samples[i + 1] = mean + std * z1
        return samples


def sample_uniform(low: float = 0, high: float = 1, size: int = 1) -> Union[float, np.ndarray]:
    """
    Sample from a uniform distribution.
    
    Args:
        low: Lower bound
        high: Upper bound
        size: Number of samples to generate
        
    Returns:
        Sample(s) from the uniform distribution
    """
    return np.random.uniform(low, high, size)


def sample_exponential(lambd: float, size: int = 1) -> Union[float, np.ndarray]:
    """
    Sample from an exponential distribution using inverse transform sampling.
    
    Args:
        lambd: Rate parameter (lambda)
        size: Number of samples to generate
        
    Returns:
        Sample(s) from the exponential distribution
    """
    u = np.random.uniform(0, 1, size)
    # Inverse CDF: F^(-1)(u) = -ln(1-u) / lambda
    # Since 1-u is also uniform on [0,1], we can use -ln(u) / lambda
    return -np.log(u) / lambd


def bayes_rule(prior: float, likelihood: float, evidence: float) -> float:
    """
    Apply Bayes' rule: P(A|B) = P(B|A) * P(A) / P(B)
    
    Args:
        prior: P(A) - Prior probability
        likelihood: P(B|A) - Likelihood
        evidence: P(B) - Evidence (marginal probability)
        
    Returns:
        Posterior probability P(A|B)
    """
    return (likelihood * prior) / evidence


def bayesian_update(
    prior_probs: np.ndarray, 
    likelihoods: np.ndarray
) -> np.ndarray:
    """
    Perform Bayesian update for multiple hypotheses.
    
    Args:
        prior_probs: Prior probabilities for each hypothesis
        likelihoods: Likelihood of evidence given each hypothesis
        
    Returns:
        Posterior probabilities for each hypothesis
    """
    # Calculate unnormalized posteriors
    unnormalized = prior_probs * likelihoods
    
    # Normalize to get proper probabilities
    posterior_probs = unnormalized / np.sum(unnormalized)
    
    return posterior_probs


def entropy(probabilities: np.ndarray) -> float:
    """
    Calculate Shannon entropy of a probability distribution.
    
    Args:
        probabilities: Array of probabilities that sum to 1
        
    Returns:
        Entropy value
    """
    # Avoid log(0) by using where condition
    probs = np.asarray(probabilities)
    entropy_val = -np.sum(np.where(probs > 0, probs * np.log2(probs), 0))
    return entropy_val


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate cross-entropy between two distributions.
    
    Args:
        p: True distribution
        q: Predicted/approximate distribution
        
    Returns:
        Cross-entropy value
    """
    p = np.asarray(p)
    q = np.asarray(q)
    # Avoid log(0) by using where condition
    cross_ent = -np.sum(np.where((p > 0) & (q > 0), p * np.log2(q), 0))
    return cross_ent


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """
    Calculate Kullback-Leibler divergence between two distributions.
    
    Args:
        p: True distribution
        q: Approximate distribution
        
    Returns:
        KL divergence value (non-symmetric)
    """
    p = np.asarray(p)
    q = np.asarray(q)
    # Avoid division by zero and log(0) by using where condition
    kl_div = np.sum(np.where((p > 0) & (q > 0), p * np.log2(p / q), 0))
    return kl_div


def mutual_information(joint_prob: np.ndarray, x_marginal: np.ndarray, y_marginal: np.ndarray) -> float:
    """
    Calculate mutual information between two variables.
    
    Args:
        joint_prob: Joint probability distribution P(X,Y)
        x_marginal: Marginal distribution P(X)
        y_marginal: Marginal distribution P(Y)
        
    Returns:
        Mutual information value
    """
    # Calculate P(X,Y) * log(P(X,Y) / (P(X) * P(Y)))
    mi = 0.0
    for i in range(len(x_marginal)):
        for j in range(len(y_marginal)):
            joint = joint_prob[i, j]
            px = x_marginal[i]
            py = y_marginal[j]
            
            if joint > 0 and px > 0 and py > 0:
                mi += joint * np.log2(joint / (px * py))
    
    return mi


def sample_categorical(probabilities: np.ndarray, size: int = 1) -> Union[int, np.ndarray]:
    """
    Sample from a categorical distribution.
    
    Args:
        probabilities: Array of probabilities that sum to 1
        size: Number of samples to generate
        
    Returns:
        Sample(s) from the categorical distribution
    """
    # Normalize probabilities to ensure they sum to 1
    probs = np.asarray(probabilities)
    probs = probs / np.sum(probs)
    
    # Create cumulative distribution
    cumsum = np.cumsum(probs)
    
    # Generate uniform random numbers and find corresponding categories
    samples = np.random.uniform(0, 1, size)
    categories = np.searchsorted(cumsum, samples)
    
    if size == 1:
        return int(categories[0])
    else:
        return categories.astype(int)


def maximum_likelihood_estimation(
    data: np.ndarray, 
    distribution: str, 
    initial_params: List[float]
) -> List[float]:
    """
    Estimate parameters using Maximum Likelihood Estimation.
    
    Args:
        data: Observed data
        distribution: Type of distribution ('gaussian', 'exponential', etc.)
        initial_params: Initial parameter guesses
        
    Returns:
        Estimated parameters
    """
    data = np.asarray(data)
    
    if distribution == 'gaussian':
        # For Gaussian, MLE has closed form: mean and variance
        mean = np.mean(data)
        variance = np.var(data)
        return [mean, np.sqrt(variance)]
    elif distribution == 'exponential':
        # For exponential, MLE of lambda = 1 / mean
        mean = np.mean(data)
        return [1.0 / mean]
    else:
        raise ValueError(f"MLE for {distribution} distribution not implemented")