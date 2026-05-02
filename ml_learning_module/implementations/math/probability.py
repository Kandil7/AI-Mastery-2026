"""
Probability and Statistics Module for Machine Learning

This module provides:
- Probability distributions (Normal, Uniform, Binomial, etc.)
- Statistical operations
- Bayesian inference basics

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Tuple, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class DistributionType(Enum):
    """Enumeration of supported distributions."""

    NORMAL = "normal"
    UNIFORM = "uniform"
    BERNOULLI = "bernoulli"
    BINOMIAL = "binomial"
    POISSON = "poisson"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    BETA = "beta"


class Distribution:
    """
    Base class for probability distributions.

    Provides methods for:
    - Sampling (random generation)
    - Computing PDF/PMF
    - Computing statistics (mean, variance)
    """

    def __init__(self, **params):
        """Initialize distribution with parameters."""
        self.params = params

    def sample(self, n: int = 1) -> np.ndarray:
        """Generate random samples from distribution."""
        raise NotImplementedError("Subclass must implement sample()")

    def pdf(self, x: float) -> float:
        """Probability density/mass function."""
        raise NotImplementedError("Subclass must implement pdf()")

    def mean(self) -> float:
        """Compute expected value."""
        raise NotImplementedError("Subclass must implement mean()")

    def variance(self) -> float:
        """Compute variance."""
        raise NotImplementedError("Subclass must implement variance()")


class NormalDistribution(Distribution):
    """Normal (Gaussian) distribution."""

    def __init__(self, mean: float = 0.0, std: float = 1.0):
        """
        Initialize Normal distribution.

        Args:
            mean: Mean (μ) of the distribution.
            std: Standard deviation (σ) of the distribution.

        Example:
            >>> dist = NormalDistribution(mean=0, std=1)
            >>> samples = dist.sample(1000)
            >>> np.mean(samples)  # Should be close to 0
        """
        super().__init__(mean=mean, std=std)
        self.mean_val = mean
        self.std_val = std
        self.var_val = std**2

    def sample(self, n: int = 1) -> np.ndarray:
        """Generate random samples."""
        return np.random.normal(self.mean_val, self.std_val, n)

    def pdf(self, x: float) -> float:
        """Compute probability density."""
        return (1 / (self.std_val * np.sqrt(2 * np.pi))) * np.exp(
            -0.5 * ((x - self.mean_val) / self.std_val) ** 2
        )

    def cdf(self, x: float) -> float:
        """Compute cumulative distribution function."""
        return 0.5 * (
            1 + np.math.erf((x - self.mean_val) / (self.std_val * np.sqrt(2)))
        )

    def mean(self) -> float:
        """Compute mean."""
        return self.mean_val

    def variance(self) -> float:
        """Compute variance."""
        return self.var_val

    def std(self) -> float:
        """Compute standard deviation."""
        return self.std_val


class UniformDistribution(Distribution):
    """Uniform distribution."""

    def __init__(self, low: float = 0.0, high: float = 1.0):
        """
        Initialize Uniform distribution.

        Args:
            low: Lower bound (a).
            high: Upper bound (b).

        Example:
            >>> dist = UniformDistribution(0, 10)
            >>> samples = dist.sample(1000)
            >>> np.mean(samples)  # Should be close to 5
        """
        super().__init__(low=low, high=high)
        self.low = low
        self.high = high

    def sample(self, n: int = 1) -> np.ndarray:
        """Generate random samples."""
        return np.random.uniform(self.low, self.high, n)

    def pdf(self, x: float) -> float:
        """Compute probability density."""
        if self.low <= x <= self.high:
            return 1 / (self.high - self.low)
        return 0.0

    def mean(self) -> float:
        """Compute mean."""
        return (self.low + self.high) / 2

    def variance(self) -> float:
        """Compute variance."""
        return (self.high - self.low) ** 2 / 12


class BernoulliDistribution(Distribution):
    """Bernoulli distribution (single trial)."""

    def __init__(self, p: float = 0.5):
        """
        Initialize Bernoulli distribution.

        Args:
            p: Probability of success.

        Example:
            >>> dist = BernoulliDistribution(p=0.7)
            >>> samples = dist.sample(1000)
            >>> np.mean(samples)  # Should be close to 0.7
        """
        super().__init__(p=p)
        self.p = p

    def sample(self, n: int = 1) -> np.ndarray:
        """Generate random samples."""
        return np.random.binomial(1, self.p, n)

    def pmf(self, x: int) -> float:
        """Compute probability mass function."""
        if x == 1:
            return self.p
        elif x == 0:
            return 1 - self.p
        return 0.0

    def mean(self) -> float:
        """Compute mean."""
        return self.p

    def variance(self) -> float:
        """Compute variance."""
        return self.p * (1 - self.p)


class BinomialDistribution(Distribution):
    """Binomial distribution (multiple trials)."""

    def __init__(self, n: int, p: float = 0.5):
        """
        Initialize Binomial distribution.

        Args:
            n: Number of trials.
            p: Probability of success per trial.

        Example:
            >>> dist = BinomialDistribution(n=10, p=0.5)
            >>> samples = dist.sample(1000)
            >>> np.mean(samples)  # Should be close to 5
        """
        super().__init__(n=n, p=p)
        self.n = n
        self.p = p

    def sample(self, n: int = 1) -> np.ndarray:
        """Generate random samples."""
        return np.random.binomial(self.n, self.p, n)

    def pmf(self, k: int) -> float:
        """Compute probability mass function P(X=k)."""
        from scipy.special import comb

        return comb(self.n, k) * (self.p**k) * ((1 - self.p) ** (self.n - k))

    def mean(self) -> float:
        """Compute mean."""
        return self.n * self.p

    def variance(self) -> float:
        """Compute variance."""
        return self.n * self.p * (1 - self.p)


class PoissonDistribution(Distribution):
    """Poisson distribution (count of rare events)."""

    def __init__(self, lambda_: float):
        """
        Initialize Poisson distribution.

        Args:
            lambda_: Average rate (λ) of events.

        Example:
            >>> dist = PoissonDistribution(lambda_=5)
            >>> samples = dist.sample(1000)
            >>> np.mean(samples)  # Should be close to 5
        """
        super().__init__(lambda_=lambda_)
        self.lam = lambda_

    def sample(self, n: int = 1) -> np.ndarray:
        """Generate random samples."""
        return np.random.poisson(self.lam, n)

    def pmf(self, k: int) -> float:
        """Compute probability mass function."""
        from math import exp

        return (self.lam**k * exp(-self.lam)) / np.math.factorial(k)

    def mean(self) -> float:
        """Compute mean."""
        return self.lam

    def variance(self) -> float:
        """Compute variance."""
        return self.lam


class ExponentialDistribution(Distribution):
    """Exponential distribution."""

    def __init__(self, scale: float = 1.0):
        """
        Initialize Exponential distribution.

        Args:
            scale: Scale parameter (1/λ).
        """
        super().__init__(scale=scale)
        self.scale = scale
        self.rate = 1 / scale

    def sample(self, n: int = 1) -> np.ndarray:
        """Generate random samples."""
        return np.random.exponential(self.scale, n)

    def mean(self) -> float:
        """Compute mean."""
        return self.scale

    def variance(self) -> float:
        """Compute variance."""
        return self.scale**2


class BetaDistribution(Distribution):
    """Beta distribution (continuous on [0, 1])."""

    def __init__(self, alpha: float, beta: float):
        """
        Initialize Beta distribution.

        Args:
            alpha: First shape parameter (α).
            beta: Second shape parameter (β).
        """
        super().__init__(alpha=alpha, beta=beta)
        self.alpha = alpha
        self.beta = beta

    def sample(self, n: int = 1) -> np.ndarray:
        """Generate random samples."""
        return np.random.beta(self.alpha, self.beta, n)

    def mean(self) -> float:
        """Compute mean."""
        return self.alpha / (self.alpha + self.beta)

    def variance(self) -> float:
        """Compute variance."""
        ab_sum = self.alpha + self.beta
        return (self.alpha * self.beta) / (ab_sum**2 * (ab_sum + 1))


# Factory function
def Distribution(distribution_type: str, **kwargs) -> Distribution:
    """
    Factory function to create distribution instances.

    Args:
        distribution_type: Type of distribution ('normal', 'uniform', etc.)
        **kwargs: Distribution-specific parameters.

    Returns:
        Distribution instance.

    Example:
        >>> dist = Distribution('normal', mean=0, std=1)
        >>> samples = dist.sample(100)
    """
    dist_type = distribution_type.lower()

    if dist_type == "normal":
        return NormalDistribution(**kwargs)
    elif dist_type == "uniform":
        return UniformDistribution(**kwargs)
    elif dist_type == "bernoulli":
        return BernoulliDistribution(**kwargs)
    elif dist_type == "binomial":
        return BinomialDistribution(**kwargs)
    elif dist_type == "poisson":
        return PoissonDistribution(**kwargs)
    elif dist_type == "exponential":
        return ExponentialDistribution(**kwargs)
    elif dist_type == "beta":
        return BetaDistribution(**kwargs)
    else:
        raise ValueError(f"Unknown distribution: {distribution_type}")


class ProbabilityOperations:
    """
    Probability operations and utilities.

    Provides methods for:
    - Computing probabilities
    - Bayes theorem
    - Statistical tests
    """

    def __init__(self):
        """Initialize ProbabilityOperations."""
        pass

    @staticmethod
    def conditional_probability(p_a_and_b: float, p_b: float) -> float:
        """
        Compute conditional probability P(A|B) = P(A∩B) / P(B).

        Args:
            p_a_and_b: P(A and B).
            p_b: P(B).

        Returns:
            P(A|B).
        """
        if p_b == 0:
            raise ValueError("P(B) cannot be zero")
        return p_a_and_b / p_b

    @staticmethod
    def bayes_theorem(p_b_given_a: float, p_a: float, p_b: float) -> float:
        """
        Compute P(A|B) using Bayes theorem.

        P(A|B) = P(B|A) × P(A) / P(B)

        Args:
            p_b_given_a: P(B|A) - likelihood.
            p_a: P(A) - prior.
            p_b: P(B) - evidence.

        Returns:
            P(A|B) - posterior.

        Example:
            >>> # P(Disease|Positive) calculation
            >>> prob = ProbabilityOperations.bayes_theorem(
            ...     p_b_given_a=0.99,  # P(Positive|Disease) = 99%
            ...     p_a=0.01,          # P(Disease) = 1%
            ...     p_b=0.059          # P(Positive) ≈ 5.9%
            ... )
        """
        if p_b == 0:
            raise ValueError("P(B) cannot be zero")
        return (p_b_given_a * p_a) / p_b

    @staticmethod
    def expected_value(values: np.ndarray, probabilities: np.ndarray) -> float:
        """
        Compute expected value E[X] = Σ x × P(X=x).

        Args:
            values: Array of possible values.
            probabilities: Corresponding probabilities.

        Returns:
            Expected value.
        """
        if len(values) != len(probabilities):
            raise ValueError("Values and probabilities must have same length")
        return np.sum(values * probabilities)

    @staticmethod
    def variance(values: np.ndarray, probabilities: np.ndarray) -> float:
        """
        Compute variance Var(X) = E[X²] - E[X]².

        Args:
            values: Array of possible values.
            probabilities: Corresponding probabilities.

        Returns:
            Variance.
        """
        mean = ProbabilityOperations.expected_value(values, probabilities)
        mean_sq = ProbabilityOperations.expected_value(values**2, probabilities)
        return mean_sq - mean**2

    @staticmethod
    def covariance(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute covariance between two variables.

        Cov(X,Y) = E[XY] - E[X]E[Y]

        Args:
            x: First variable data.
            y: Second variable data.

        Returns:
            Covariance.
        """
        if len(x) != len(y):
            raise ValueError("Arrays must have same length")

        return np.mean(x * y) - np.mean(x) * np.mean(y)

    @staticmethod
    def correlation(x: np.ndarray, y: np.ndarray) -> float:
        """
        Compute Pearson correlation coefficient.

        r = Cov(X,Y) / (σ_X × σ_Y)

        Args:
            x: First variable data.
            y: Second variable data.

        Returns:
            Correlation coefficient (-1 to 1).
        """
        cov = ProbabilityOperations.covariance(x, y)
        std_x = np.std(x)
        std_y = np.std(y)

        if std_x == 0 or std_y == 0:
            return 0.0

        return cov / (std_x * std_y)


class HypothesisTesting:
    """
    Hypothesis testing utilities.

    Provides methods for:
    - t-test
    - chi-square test
    - Confidence intervals
    """

    @staticmethod
    def t_test(
        sample1: np.ndarray, sample2: np.ndarray, equal_variance: bool = True
    ) -> dict:
        """
        Perform two-sample t-test.

        Tests whether two samples have different means.

        Args:
            sample1: First sample data.
            sample2: Second sample data.
            equal_variance: Whether to assume equal variances.

        Returns:
            Dictionary with t-statistic and p-value.

        Example:
            >>> t_stat, p_val = HypothesisTesting.t_test(data1, data2)
        """
        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)

        if equal_variance:
            # Pooled variance
            sp = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
            t_stat = (mean1 - mean2) / (sp * np.sqrt(1 / n1 + 1 / n2))
            df = n1 + n2 - 2
        else:
            # Welch's t-test
            se = np.sqrt(var1 / n1 + var2 / n2)
            t_stat = (mean1 - mean2) / se
            # Welch-Satterthwaite degrees of freedom
            df = (var1 / n1 + var2 / n2) ** 2 / (
                (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
            )

        # Two-tailed p-value
        from scipy import stats

        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

        return {"t_statistic": t_stat, "p_value": p_value, "df": df}

    @staticmethod
    def confidence_interval(
        data: np.ndarray, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for the mean.

        Args:
            data: Sample data.
            confidence: Confidence level (e.g., 0.95 for 95%).

        Returns:
            Tuple of (lower_bound, upper_bound).
        """
        mean = np.mean(data)
        se = np.std(data, ddof=1) / np.sqrt(len(data))

        from scipy import stats

        t_crit = stats.t.ppf((1 + confidence) / 2, len(data) - 1)

        return (mean - t_crit * se, mean + t_crit * se)


# Demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("Probability Module - Demonstration")
    print("=" * 60)

    # Test distributions
    print("\n--- Distribution Sampling ---")

    np.random.seed(42)

    # Normal distribution
    normal = NormalDistribution(mean=0, std=1)
    samples = normal.sample(10000)
    print(f"\nNormal(μ=0, σ=1):")
    print(f"  Sample mean: {np.mean(samples):.4f} (expected: 0)")
    print(f"  Sample std:  {np.std(samples):.4f} (expected: 1)")

    # Uniform distribution
    uniform = UniformDistribution(0, 10)
    samples = uniform.sample(10000)
    print(f"\nUniform(a=0, b=10):")
    print(f"  Sample mean: {np.mean(samples):.4f} (expected: 5)")
    print(f"  Sample std:  {np.std(samples):.4f} (expected: ~2.89)")

    # Bernoulli distribution
    bernoulli = BernoulliDistribution(p=0.7)
    samples = bernoulli.sample(10000)
    print(f"\nBernoulli(p=0.7):")
    print(f"  Sample mean: {np.mean(samples):.4f} (expected: 0.7)")

    # Binomial distribution
    binomial = BinomialDistribution(n=10, p=0.5)
    samples = binomial.sample(10000)
    print(f"\nBinomial(n=10, p=0.5):")
    print(f"  Sample mean: {np.mean(samples):.4f} (expected: 5)")

    # Poisson distribution
    poisson = PoissonDistribution(lambda_=5)
    samples = poisson.sample(10000)
    print(f"\nPoisson(λ=5):")
    print(f"  Sample mean: {np.mean(samples):.4f} (expected: 5)")

    # Bayesian example
    print("\n--- Bayesian Inference Example ---")

    # P(Disease) = 1%
    p_disease = 0.01
    # P(Positive | Disease) = 99%
    p_pos_given_disease = 0.99
    # P(Positive | No Disease) = 5%
    p_pos_given_no_disease = 0.05

    # P(Positive)
    p_positive = p_pos_given_disease * p_disease + p_pos_given_no_disease * (
        1 - p_disease
    )

    # P(Disease | Positive) - using Bayes
    p_disease_given_positive = (p_pos_given_disease * p_disease) / p_positive

    print(f"\nMedical Test Example:")
    print(f"  P(Disease) = {p_disease}")
    print(f"  P(Positive | Disease) = {p_pos_given_disease}")
    print(f"  P(Positive | No Disease) = {p_pos_given_no_disease}")
    print(f"  P(Disease | Positive) = {p_disease_given_positive:.4f}")

    # Correlation example
    print("\n--- Correlation Example ---")

    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 5, 4, 5])

    corr = ProbabilityOperations.correlation(x, y)
    cov = ProbabilityOperations.covariance(x, y)

    print(f"\nx = {x}")
    print(f"y = {y}")
    print(f"Covariance: {cov:.4f}")
    print(f"Correlation: {corr:.4f}")

    # Hypothesis testing
    print("\n--- Hypothesis Testing ---")

    group1 = np.array([85, 87, 92, 78, 88, 82])
    group2 = np.array([72, 75, 68, 74, 70, 76])

    result = HypothesisTesting.t_test(group1, group2)
    print(f"\nTwo-sample t-test:")
    print(f"  Group 1 mean: {np.mean(group1):.2f}")
    print(f"  Group 2 mean: {np.mean(group2):.2f}")
    print(f"  t-statistic: {result['t_statistic']:.4f}")
    print(f"  p-value: {result['p_value']:.4f}")

    print("\n" + "=" * 60)
