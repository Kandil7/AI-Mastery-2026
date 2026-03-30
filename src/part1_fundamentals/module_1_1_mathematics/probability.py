"""
Probability and Statistics Module for Machine Learning.

This module provides probability and statistics operations essential for ML,
including distributions, Bayes theorem, expectation, variance, covariance, and hypothesis testing.

Example Usage:
    >>> import numpy as np
    >>> from probability import ProbabilityOperations, Distribution
    >>> 
    >>> # Create a normal distribution
    >>> dist = Distribution.normal(mean=0, std=1)
    >>> samples = dist.sample(1000)
    >>> 
    >>> # Compute statistics
    >>> ops = ProbabilityOperations()
    >>> mean = ops.expectation(samples)
    >>> var = ops.variance(samples)
    >>> print(f"Sample mean: {mean:.4f}, Sample variance: {var:.4f}")
"""

from typing import Callable, Union, List, Tuple, Optional, Dict, Any
import numpy as np
from numpy.typing import ArrayLike
import logging
from dataclasses import dataclass
from enum import Enum
from scipy import stats as scipy_stats
from scipy.special import erf, erfinv, gamma, beta as beta_func

logger = logging.getLogger(__name__)

Vector = Union[np.ndarray, List[float]]
ScalarFunction = Callable[[float], float]


class DistributionType(Enum):
    """Supported probability distribution types."""
    NORMAL = "normal"
    UNIFORM = "uniform"
    BERNOULLI = "bernoulli"
    BINOMIAL = "binomial"
    POISSON = "poisson"
    EXPONENTIAL = "exponential"
    BETA = "beta"
    GAMMA = "gamma"
    CHI_SQUARED = "chi_squared"
    STUDENT_T = "student_t"


@dataclass
class Distribution:
    """
    Probability distribution representation.
    
    Attributes:
        dist_type: Type of distribution.
        params: Distribution parameters.
        pdf: Probability density/mass function.
        cdf: Cumulative distribution function.
        ppf: Percent point function (inverse CDF).
    """
    dist_type: DistributionType
    params: Dict[str, float]
    pdf: Callable[[float], float]
    cdf: Callable[[float], float]
    ppf: Callable[[float], float]
    
    def sample(self, n: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """
        Generate random samples from the distribution.
        
        Args:
            n: Number of samples.
            seed: Random seed for reproducibility.
        
        Returns:
            np.ndarray: Random samples.
        """
        if seed is not None:
            np.random.seed(seed)
        
        if self.dist_type == DistributionType.NORMAL:
            return np.random.normal(
                self.params['mean'], 
                self.params['std'], 
                n
            )
        elif self.dist_type == DistributionType.UNIFORM:
            return np.random.uniform(
                self.params['low'], 
                self.params['high'], 
                n
            )
        elif self.dist_type == DistributionType.BERNOULLI:
            return np.random.binomial(1, self.params['p'], n)
        elif self.dist_type == DistributionType.BINOMIAL:
            return np.random.binomial(
                int(self.params['n']), 
                self.params['p'], 
                n
            )
        elif self.dist_type == DistributionType.POISSON:
            return np.random.poisson(self.params['lambda_'], n)
        elif self.dist_type == DistributionType.EXPONENTIAL:
            return np.random.exponential(self.params['scale'], n)
        elif self.dist_type == DistributionType.BETA:
            return np.random.beta(self.params['alpha'], self.params['beta'], n)
        elif self.dist_type == DistributionType.GAMMA:
            return np.random.gamma(self.params['shape'], self.params['scale'], n)
        elif self.dist_type == DistributionType.CHI_SQUARED:
            return np.random.chisquare(self.params['df'], n)
        elif self.dist_type == DistributionType.STUDENT_T:
            return np.random.standard_t(self.params['df'], n)
        else:
            raise ValueError(f"Unknown distribution type: {self.dist_type}")
    
    @classmethod
    def normal(cls, mean: float = 0.0, std: float = 1.0) -> 'Distribution':
        """
        Create a normal (Gaussian) distribution.
        
        Args:
            mean: Mean (μ) of the distribution.
            std: Standard deviation (σ) of the distribution.
        
        Returns:
            Distribution: Normal distribution object.
        
        Example:
            >>> dist = Distribution.normal(mean=0, std=1)
            >>> samples = dist.sample(1000)
            >>> np.isclose(np.mean(samples), 0, atol=0.2)
            True
        """
        def pdf(x):
            return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
        
        def cdf(x):
            return 0.5 * (1 + erf((x - mean) / (std * np.sqrt(2))))
        
        def ppf(p):
            return mean + std * np.sqrt(2) * erfinv(2 * p - 1)
        
        return cls(
            dist_type=DistributionType.NORMAL,
            params={'mean': mean, 'std': std},
            pdf=pdf,
            cdf=cdf,
            ppf=ppf
        )
    
    @classmethod
    def uniform(cls, low: float = 0.0, high: float = 1.0) -> 'Distribution':
        """
        Create a uniform distribution.
        
        Args:
            low: Lower bound.
            high: Upper bound.
        
        Returns:
            Distribution: Uniform distribution object.
        """
        def pdf(x):
            return np.where((x >= low) & (x <= high), 1.0 / (high - low), 0.0)
        
        def cdf(x):
            return np.where(x < low, 0.0, np.where(x > high, 1.0, (x - low) / (high - low)))
        
        def ppf(p):
            return low + p * (high - low)
        
        return cls(
            dist_type=DistributionType.UNIFORM,
            params={'low': low, 'high': high},
            pdf=pdf,
            cdf=cdf,
            ppf=ppf
        )
    
    @classmethod
    def bernoulli(cls, p: float = 0.5) -> 'Distribution':
        """
        Create a Bernoulli distribution.
        
        Args:
            p: Probability of success.
        
        Returns:
            Distribution: Bernoulli distribution object.
        """
        def pmf(x):
            x = np.asarray(x)
            return np.where(x == 1, p, np.where(x == 0, 1 - p, 0.0))
        
        def cdf(x):
            return np.where(x < 0, 0.0, np.where(x < 1, 1 - p, 1.0))
        
        def ppf(p_val):
            return np.where(p_val < 1 - p, 0.0, 1.0)
        
        return cls(
            dist_type=DistributionType.BERNOULLI,
            params={'p': p},
            pdf=pmf,
            cdf=cdf,
            ppf=ppf
        )
    
    @classmethod
    def binomial(cls, n: int = 10, p: float = 0.5) -> 'Distribution':
        """
        Create a binomial distribution.
        
        Args:
            n: Number of trials.
            p: Probability of success.
        
        Returns:
            Distribution: Binomial distribution object.
        """
        from math import comb
        
        def pmf(x):
            x = np.asarray(x, dtype=int)
            return comb(n, x) * (p ** x) * ((1 - p) ** (n - x))
        
        def cdf(x):
            return scipy_stats.binom.cdf(x, n, p)
        
        def ppf(p_val):
            return scipy_stats.binom.ppf(p_val, n, p)
        
        return cls(
            dist_type=DistributionType.BINOMIAL,
            params={'n': n, 'p': p},
            pdf=pmf,
            cdf=cdf,
            ppf=ppf
        )
    
    @classmethod
    def poisson(cls, lambda_: float = 1.0) -> 'Distribution':
        """
        Create a Poisson distribution.
        
        Args:
            lambda_: Rate parameter (mean number of events).
        
        Returns:
            Distribution: Poisson distribution object.
        """
        def pmf(x):
            x = np.asarray(x, dtype=int)
            return (lambda_ ** x) * np.exp(-lambda_) / gamma(x + 1)
        
        def cdf(x):
            return scipy_stats.poisson.cdf(x, lambda_)
        
        def ppf(p_val):
            return scipy_stats.poisson.ppf(p_val, lambda_)
        
        return cls(
            dist_type=DistributionType.POISSON,
            params={'lambda_': lambda_},
            pdf=pmf,
            cdf=cdf,
            ppf=ppf
        )
    
    @classmethod
    def exponential(cls, scale: float = 1.0) -> 'Distribution':
        """
        Create an exponential distribution.
        
        Args:
            scale: Scale parameter (1/rate).
        
        Returns:
            Distribution: Exponential distribution object.
        """
        rate = 1.0 / scale
        
        def pdf(x):
            return np.where(x >= 0, rate * np.exp(-rate * x), 0.0)
        
        def cdf(x):
            return np.where(x >= 0, 1 - np.exp(-rate * x), 0.0)
        
        def ppf(p_val):
            return -scale * np.log(1 - p_val)
        
        return cls(
            dist_type=DistributionType.EXPONENTIAL,
            params={'scale': scale, 'rate': rate},
            pdf=pdf,
            cdf=cdf,
            ppf=ppf
        )
    
    @classmethod
    def beta(cls, alpha: float = 1.0, beta: float = 1.0) -> 'Distribution':
        """
        Create a Beta distribution.
        
        Args:
            alpha: First shape parameter.
            beta: Second shape parameter.
        
        Returns:
            Distribution: Beta distribution object.
        """
        def pdf(x):
            norm_const = 1.0 / beta_func(alpha, beta)
            return np.where((x >= 0) & (x <= 1), 
                          norm_const * (x ** (alpha - 1)) * ((1 - x) ** (beta - 1)), 
                          0.0)
        
        def cdf(x):
            return scipy_stats.beta.cdf(x, alpha, beta)
        
        def ppf(p_val):
            return scipy_stats.beta.ppf(p_val, alpha, beta)
        
        return cls(
            dist_type=DistributionType.BETA,
            params={'alpha': alpha, 'beta': beta},
            pdf=pdf,
            cdf=cdf,
            ppf=ppf
        )
    
    @classmethod
    def gamma(cls, shape: float = 1.0, scale: float = 1.0) -> 'Distribution':
        """
        Create a Gamma distribution.
        
        Args:
            shape: Shape parameter (k).
            scale: Scale parameter (θ).
        
        Returns:
            Distribution: Gamma distribution object.
        """
        def pdf(x):
            k, theta = shape, scale
            return np.where(x > 0,
                          (x ** (k - 1)) * np.exp(-x / theta) / (gamma(k) * (theta ** k)),
                          0.0)
        
        def cdf(x):
            return scipy_stats.gamma.cdf(x, shape, scale=scale)
        
        def ppf(p_val):
            return scipy_stats.gamma.ppf(p_val, shape, scale=scale)
        
        return cls(
            dist_type=DistributionType.GAMMA,
            params={'shape': shape, 'scale': scale},
            pdf=pdf,
            cdf=cdf,
            ppf=ppf
        )
    
    @classmethod
    def chi_squared(cls, df: int = 1) -> 'Distribution':
        """
        Create a Chi-squared distribution.
        
        Args:
            df: Degrees of freedom.
        
        Returns:
            Distribution: Chi-squared distribution object.
        """
        return cls.gamma(shape=df / 2, scale=2).dist_type.__class__(
            dist_type=DistributionType.CHI_SQUARED,
            params={'df': df},
            pdf=lambda x: scipy_stats.chi2.pdf(x, df),
            cdf=lambda x: scipy_stats.chi2.cdf(x, df),
            ppf=lambda p: scipy_stats.chi2.ppf(p, df)
        )
    
    @classmethod
    def student_t(cls, df: int = 1) -> 'Distribution':
        """
        Create a Student's t-distribution.
        
        Args:
            df: Degrees of freedom.
        
        Returns:
            Distribution: Student's t-distribution object.
        """
        def pdf(x):
            return scipy_stats.t.pdf(x, df)
        
        def cdf(x):
            return scipy_stats.t.cdf(x, df)
        
        def ppf(p_val):
            return scipy_stats.t.ppf(p_val, df)
        
        return cls(
            dist_type=DistributionType.STUDENT_T,
            params={'df': df},
            pdf=pdf,
            cdf=cdf,
            ppf=ppf
        )


class ProbabilityOperations:
    """
    Probability and statistics operations for machine learning.
    
    This class provides methods for:
    - Basic probability (Bayes theorem, conditional probability)
    - Expectation, variance, covariance
    - Correlation coefficients
    - Hypothesis testing
    - Maximum likelihood estimation
    - KL divergence and cross-entropy
    
    Example:
        >>> ops = ProbabilityOperations()
        >>> # Bayes theorem
        >>> p_a_given_b = ops.bayes_theorem(0.9, 0.01, 0.1)
        >>> print(f"P(A|B) = {p_a_given_b:.4f}")
    """
    
    def __init__(self, epsilon: float = 1e-10):
        """
        Initialize ProbabilityOperations.
        
        Args:
            epsilon: Small value for numerical stability. Default: 1e-10.
        """
        self.epsilon = epsilon
        logger.debug(f"ProbabilityOperations initialized with epsilon={epsilon}")
    
    def bayes_theorem(
        self,
        p_b_given_a: float,
        p_a: float,
        p_b: float
    ) -> float:
        """
        Apply Bayes' theorem to compute P(A|B).
        
        Formula: P(A|B) = P(B|A) * P(A) / P(B)
        
        Args:
            p_b_given_a: Likelihood P(B|A).
            p_a: Prior P(A).
            p_b: Marginal likelihood P(B).
        
        Returns:
            float: Posterior P(A|B).
        
        Raises:
            ValueError: If probabilities are invalid.
        
        Example:
            >>> ops = ProbabilityOperations()
            >>> # Medical test example
            >>> # P(positive|disease) = 0.99 (sensitivity)
            >>> # P(disease) = 0.001 (prevalence)
            >>> # P(positive) = 0.01 (overall positive rate)
            >>> p_disease_given_positive = ops.bayes_theorem(0.99, 0.001, 0.01)
            >>> print(f"P(disease|positive) = {p_disease_given_positive:.4f}")
            P(disease|positive) = 0.0990
        """
        if not (0 <= p_b_given_a <= 1):
            raise ValueError(f"P(B|A) must be in [0, 1], got {p_b_given_a}")
        if not (0 <= p_a <= 1):
            raise ValueError(f"P(A) must be in [0, 1], got {p_a}")
        if p_b <= self.epsilon:
            raise ValueError(f"P(B) must be positive, got {p_b}")
        
        result = (p_b_given_a * p_a) / p_b
        result = min(1.0, max(0.0, result))  # Clamp to [0, 1]
        
        logger.debug(f"Bayes theorem: P(A|B) = {result:.6f}")
        return result
    
    def bayes_with_marginal(
        self,
        p_b_given_a: float,
        p_a: float,
        p_b_given_not_a: float
    ) -> float:
        """
        Apply Bayes' theorem computing marginal P(B) from components.
        
        P(B) = P(B|A) * P(A) + P(B|¬A) * P(¬A)
        P(A|B) = P(B|A) * P(A) / P(B)
        
        Args:
            p_b_given_a: P(B|A).
            p_a: P(A).
            p_b_given_not_a: P(B|¬A).
        
        Returns:
            float: Posterior P(A|B).
        
        Example:
            >>> ops = ProbabilityOperations()
            >>> # Disease testing
            >>> # Sensitivity: P(positive|disease) = 0.99
            >>> # Prevalence: P(disease) = 0.001
            >>> # False positive rate: P(positive|no disease) = 0.01
            >>> p_disease_given_positive = ops.bayes_with_marginal(0.99, 0.001, 0.01)
            >>> print(f"P(disease|positive) = {p_disease_given_positive:.4f}")
            P(disease|positive) = 0.0902
        """
        p_not_a = 1 - p_a
        p_b = p_b_given_a * p_a + p_b_given_not_a * p_not_a
        
        return self.bayes_theorem(p_b_given_a, p_a, p_b)
    
    def conditional_probability(
        self,
        p_a_and_b: float,
        p_b: float
    ) -> float:
        """
        Compute conditional probability P(A|B).
        
        Formula: P(A|B) = P(A ∩ B) / P(B)
        
        Args:
            p_a_and_b: Joint probability P(A ∩ B).
            p_b: Marginal probability P(B).
        
        Returns:
            float: Conditional probability P(A|B).
        """
        if p_b <= self.epsilon:
            raise ValueError(f"P(B) must be positive, got {p_b}")
        
        result = p_a_and_b / p_b
        result = min(1.0, max(0.0, result))
        
        logger.debug(f"Conditional probability P(A|B) = {result:.6f}")
        return result
    
    def joint_probability_independent(
        self,
        probabilities: List[float]
    ) -> float:
        """
        Compute joint probability of independent events.
        
        For independent events: P(A ∩ B) = P(A) * P(B)
        
        Args:
            probabilities: List of individual probabilities.
        
        Returns:
            float: Joint probability.
        
        Example:
            >>> ops = ProbabilityOperations()
            >>> # Probability of 3 independent events all occurring
            >>> ops.joint_probability_independent([0.5, 0.3, 0.8])
            0.12
        """
        result = 1.0
        for p in probabilities:
            if not (0 <= p <= 1):
                raise ValueError(f"Probabilities must be in [0, 1], got {p}")
            result *= p
        
        logger.debug(f"Joint probability of {len(probabilities)} independent events: {result}")
        return result
    
    def expectation(
        self,
        values: np.ndarray,
        probabilities: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute expected value E[X].
        
        For discrete: E[X] = Σ x_i * p_i
        For uniform weights: E[X] = (1/n) * Σ x_i (sample mean)
        
        Args:
            values: Random variable values.
            probabilities: Optional probability weights. If None, uses uniform.
        
        Returns:
            float: Expected value.
        
        Example:
            >>> ops = ProbabilityOperations()
            >>> # Expected value of dice roll
            >>> values = np.array([1, 2, 3, 4, 5, 6])
            >>> probs = np.array([1/6] * 6)
            >>> ops.expectation(values, probs)
            3.5
        """
        values_arr = np.asarray(values, dtype=np.float64)
        
        if probabilities is None:
            result = np.mean(values_arr)
        else:
            probs_arr = np.asarray(probabilities, dtype=np.float64)
            if not np.isclose(np.sum(probs_arr), 1.0, atol=1e-6):
                probs_arr = probs_arr / np.sum(probs_arr)  # Normalize
            result = np.sum(values_arr * probs_arr)
        
        logger.debug(f"Expectation: E[X] = {result:.6f}")
        return float(result)
    
    def variance(
        self,
        values: np.ndarray,
        ddof: int = 0
    ) -> float:
        """
        Compute variance Var(X) = E[(X - μ)²].
        
        Args:
            values: Random variable values.
            ddof: Delta degrees of freedom. 0 for population, 1 for sample.
        
        Returns:
            float: Variance.
        
        Example:
            >>> ops = ProbabilityOperations()
            >>> values = np.array([1, 2, 3, 4, 5])
            >>> ops.variance(values, ddof=0)
            2.0
        """
        values_arr = np.asarray(values, dtype=np.float64)
        result = float(np.var(values_arr, ddof=ddof))
        logger.debug(f"Variance: Var(X) = {result:.6f}")
        return result
    
    def standard_deviation(
        self,
        values: np.ndarray,
        ddof: int = 0
    ) -> float:
        """
        Compute standard deviation σ = √Var(X).
        
        Args:
            values: Random variable values.
            ddof: Delta degrees of freedom.
        
        Returns:
            float: Standard deviation.
        """
        return np.sqrt(self.variance(values, ddof=ddof))
    
    def covariance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        ddof: int = 0
    ) -> float:
        """
        Compute covariance Cov(X, Y) = E[(X - μₓ)(Y - μᵧ)].
        
        Args:
            x: First random variable.
            y: Second random variable.
            ddof: Delta degrees of freedom.
        
        Returns:
            float: Covariance.
        
        Example:
            >>> ops = ProbabilityOperations()
            >>> x = np.array([1, 2, 3, 4, 5])
            >>> y = np.array([2, 4, 6, 8, 10])  # Perfectly correlated
            >>> ops.covariance(x, y)
            2.5
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        
        if x_arr.shape != y_arr.shape:
            raise ValueError(f"x and y must have same shape")
        
        result = float(np.cov(x_arr, y_arr, ddof=ddof)[0, 1])
        logger.debug(f"Covariance: Cov(X,Y) = {result:.6f}")
        return result
    
    def correlation(
        self,
        x: np.ndarray,
        y: np.ndarray,
        method: str = 'pearson'
    ) -> float:
        """
        Compute correlation coefficient.
        
        Methods:
        - 'pearson': Pearson correlation (linear relationship)
        - 'spearman': Spearman rank correlation (monotonic relationship)
        
        Args:
            x: First random variable.
            y: Second random variable.
            method: Correlation method. Default: 'pearson'.
        
        Returns:
            float: Correlation coefficient in [-1, 1].
        
        Example:
            >>> ops = ProbabilityOperations()
            >>> x = np.array([1, 2, 3, 4, 5])
            >>> y = np.array([2, 4, 6, 8, 10])
            >>> ops.correlation(x, y)
            1.0
        """
        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y, dtype=np.float64)
        
        if method == 'pearson':
            # Pearson correlation
            cov = self.covariance(x_arr, y_arr, ddof=1)
            std_x = self.standard_deviation(x_arr, ddof=1)
            std_y = self.standard_deviation(y_arr, ddof=1)
            
            if std_x < self.epsilon or std_y < self.epsilon:
                return 0.0
            
            result = cov / (std_x * std_y)
        
        elif method == 'spearman':
            # Spearman rank correlation
            rank_x = scipy_stats.rankdata(x_arr)
            rank_y = scipy_stats.rankdata(y_arr)
            result, _ = scipy_stats.spearmanr(rank_x, rank_y)
        
        else:
            raise ValueError(f"Method must be 'pearson' or 'spearman', got '{method}'")
        
        result = float(np.clip(result, -1.0, 1.0))
        logger.debug(f"Correlation ({method}): ρ = {result:.6f}")
        return result
    
    def covariance_matrix(self, data: np.ndarray) -> np.ndarray:
        """
        Compute covariance matrix for multivariate data.
        
        Args:
            data: Data matrix (n_samples × n_features).
        
        Returns:
            np.ndarray: Covariance matrix (n_features × n_features).
        
        Example:
            >>> ops = ProbabilityOperations()
            >>> data = np.array([[1, 2], [2, 4], [3, 6], [4, 8]])
            >>> cov_matrix = ops.covariance_matrix(data)
            >>> cov_matrix.shape
            (2, 2)
        """
        data_arr = np.asarray(data, dtype=np.float64)
        result = np.cov(data_arr, rowvar=False)
        logger.debug(f"Covariance matrix computed: shape {result.shape}")
        return result
    
    def moment(
        self,
        values: np.ndarray,
        order: int = 1,
        central: bool = False
    ) -> float:
        """
        Compute the n-th moment of a distribution.
        
        Raw moment: E[Xⁿ]
        Central moment: E[(X - μ)ⁿ]
        
        Args:
            values: Random variable values.
            order: Order of the moment.
            central: If True, compute central moment. Default: False.
        
        Returns:
            float: Moment value.
        
        Example:
            >>> ops = ProbabilityOperations()
            >>> values = np.array([1, 2, 3, 4, 5])
            >>> # First raw moment = mean
            >>> ops.moment(values, order=1)
            3.0
            >>> # Second central moment = variance
            >>> ops.moment(values, order=2, central=True)
            2.0
        """
        values_arr = np.asarray(values, dtype=np.float64)
        
        if central:
            mean = np.mean(values_arr)
            result = np.mean((values_arr - mean) ** order)
        else:
            result = np.mean(values_arr ** order)
        
        logger.debug(f"{order}-th {'central' if central else 'raw'} moment: {result:.6f}")
        return float(result)
    
    def skewness(self, values: np.ndarray) -> float:
        """
        Compute skewness (third standardized moment).
        
        Skewness measures asymmetry of the distribution.
        - Positive: Right-skewed (long right tail)
        - Negative: Left-skewed (long left tail)
        - Zero: Symmetric
        
        Args:
            values: Random variable values.
        
        Returns:
            float: Skewness value.
        """
        values_arr = np.asarray(values, dtype=np.float64)
        n = len(values_arr)
        
        if n < 3:
            return 0.0
        
        mean = np.mean(values_arr)
        std = np.std(values_arr, ddof=0)
        
        if std < self.epsilon:
            return 0.0
        
        m3 = np.mean((values_arr - mean) ** 3)
        result = m3 / (std ** 3)
        
        logger.debug(f"Skewness: {result:.6f}")
        return float(result)
    
    def kurtosis(self, values: np.ndarray, excess: bool = True) -> float:
        """
        Compute kurtosis (fourth standardized moment).
        
        Kurtosis measures "tailedness" of the distribution.
        - High kurtosis: Heavy tails, more outliers
        - Low kurtosis: Light tails, fewer outliers
        
        Args:
            values: Random variable values.
            excess: If True, return excess kurtosis (kurtosis - 3). Default: True.
        
        Returns:
            float: Kurtosis value.
        """
        values_arr = np.asarray(values, dtype=np.float64)
        n = len(values_arr)
        
        if n < 4:
            return 0.0
        
        mean = np.mean(values_arr)
        std = np.std(values_arr, ddof=0)
        
        if std < self.epsilon:
            return 0.0
        
        m4 = np.mean((values_arr - mean) ** 4)
        result = m4 / (std ** 4)
        
        if excess:
            result -= 3  # Excess kurtosis (normal distribution has 0)
        
        logger.debug(f"Kurtosis (excess={excess}): {result:.6f}")
        return float(result)
    
    def kl_divergence(
        self,
        p: np.ndarray,
        q: np.ndarray
    ) -> float:
        """
        Compute Kullback-Leibler divergence D_KL(P || Q).
        
        KL divergence measures how P differs from Q.
        D_KL(P || Q) = Σ P(i) * log(P(i) / Q(i))
        
        Properties:
        - Always non-negative
        - Zero iff P = Q
        - Not symmetric (not a true distance)
        
        Args:
            p: First probability distribution.
            q: Second probability distribution (reference).
        
        Returns:
            float: KL divergence (in nats).
        
        Raises:
            ValueError: If distributions are invalid.
        
        Example:
            >>> ops = ProbabilityOperations()
            >>> p = np.array([0.5, 0.3, 0.2])
            >>> q = np.array([0.4, 0.4, 0.2])
            >>> ops.kl_divergence(p, q)  # doctest: +ELLIPSIS
            0.0...
        """
        p_arr = np.asarray(p, dtype=np.float64)
        q_arr = np.asarray(q, dtype=np.float64)
        
        if p_arr.shape != q_arr.shape:
            raise ValueError("Distributions must have same shape")
        
        # Normalize
        p_arr = p_arr / np.sum(p_arr)
        q_arr = q_arr / np.sum(q_arr)
        
        # Check for zeros in q (would cause infinite divergence)
        if np.any(q_arr <= self.epsilon):
            logger.warning("q contains zeros, KL divergence may be infinite")
            q_arr = np.clip(q_arr, self.epsilon, 1.0)
            q_arr = q_arr / np.sum(q_arr)
        
        # Compute KL divergence
        mask = p_arr > self.epsilon
        result = np.sum(p_arr[mask] * np.log(p_arr[mask] / q_arr[mask]))
        
        logger.debug(f"KL divergence D_KL(P||Q) = {result:.6f} nats")
        return float(result)
    
    def cross_entropy(
        self,
        p: np.ndarray,
        q: np.ndarray
    ) -> float:
        """
        Compute cross-entropy H(P, Q).
        
        Cross-entropy measures the average number of bits needed to encode
        data from P using a code optimized for Q.
        
        H(P, Q) = H(P) + D_KL(P || Q) = -Σ P(i) * log(Q(i))
        
        Args:
            p: True distribution.
            q: Predicted distribution.
        
        Returns:
            float: Cross-entropy (in nats).
        
        Example:
            >>> ops = ProbabilityOperations()
            >>> p = np.array([1.0, 0.0, 0.0])  # True label
            >>> q = np.array([0.7, 0.2, 0.1])  # Prediction
            >>> ops.cross_entropy(p, q)  # doctest: +ELLIPSIS
            0.356...
        """
        p_arr = np.asarray(p, dtype=np.float64)
        q_arr = np.asarray(q, dtype=np.float64)
        
        if p_arr.shape != q_arr.shape:
            raise ValueError("Distributions must have same shape")
        
        # Clip q to avoid log(0)
        q_clipped = np.clip(q_arr, self.epsilon, 1.0)
        
        result = -np.sum(p_arr * np.log(q_clipped))
        
        logger.debug(f"Cross-entropy H(P,Q) = {result:.6f} nats")
        return float(result)
    
    def entropy(self, p: np.ndarray, base: float = np.e) -> float:
        """
        Compute Shannon entropy H(P).
        
        Entropy measures uncertainty/randomness of a distribution.
        H(P) = -Σ P(i) * log(P(i))
        
        Args:
            p: Probability distribution.
            base: Logarithm base. Default: e (nats). Use 2 for bits.
        
        Returns:
            float: Entropy value.
        
        Example:
            >>> ops = ProbabilityOperations()
            >>> # Fair coin: maximum entropy for binary
            >>> p = np.array([0.5, 0.5])
            >>> ops.entropy(p, base=2)
            1.0
        """
        p_arr = np.asarray(p, dtype=np.float64)
        p_arr = p_arr / np.sum(p_arr)  # Normalize
        
        # Only consider non-zero probabilities
        mask = p_arr > self.epsilon
        result = -np.sum(p_arr[mask] * np.log(p_arr[mask]))
        
        if base != np.e:
            result = result / np.log(base)
        
        logger.debug(f"Entropy H(P) = {result:.6f} (base {base})")
        return float(result)
    
    def maximum_likelihood_estimation(
        self,
        data: np.ndarray,
        dist_type: DistributionType = DistributionType.NORMAL
    ) -> Dict[str, float]:
        """
        Compute maximum likelihood estimates for distribution parameters.
        
        Args:
            data: Observed data samples.
            dist_type: Type of distribution to fit.
        
        Returns:
            Dict[str, float]: Estimated parameters.
        
        Example:
            >>> ops = ProbabilityOperations()
            >>> np.random.seed(42)
            >>> data = np.random.normal(5, 2, 1000)
            >>> params = ops.maximum_likelihood_estimation(data)
            >>> np.isclose(params['mean'], 5, atol=0.3)
            True
            >>> np.isclose(params['std'], 2, atol=0.3)
            True
        """
        data_arr = np.asarray(data, dtype=np.float64)
        
        if dist_type == DistributionType.NORMAL:
            return {
                'mean': float(np.mean(data_arr)),
                'std': float(np.std(data_arr, ddof=0))
            }
        
        elif dist_type == DistributionType.BERNOULLI:
            return {'p': float(np.mean(data_arr))}
        
        elif dist_type == DistributionType.POISSON:
            return {'lambda_': float(np.mean(data_arr))}
        
        elif dist_type == DistributionType.EXPONENTIAL:
            return {'scale': float(np.mean(data_arr))}
        
        elif dist_type == DistributionType.BINOMIAL:
            # For binomial, we need to know n
            # Here we estimate p assuming n is known or can be inferred
            return {'p': float(np.mean(data_arr) / np.max(data_arr))}
        
        else:
            # Use scipy's fit for other distributions
            if dist_type == DistributionType.BETA:
                params = scipy_stats.beta.fit(data_arr, floc=0, fscale=1)
                return {'alpha': params[0], 'beta': params[1]}
            elif dist_type == DistributionType.GAMMA:
                params = scipy_stats.gamma.fit(data_arr)
                return {'shape': params[0], 'scale': params[2]}
            else:
                raise ValueError(f"MLE not implemented for {dist_type}")
    
    def confidence_interval(
        self,
        data: np.ndarray,
        confidence: float = 0.95,
        method: str = 't'
    ) -> Tuple[float, float]:
        """
        Compute confidence interval for the mean.
        
        Args:
            data: Sample data.
            confidence: Confidence level (e.g., 0.95 for 95%). Default: 0.95.
            method: Method to use ('t' for t-distribution, 'z' for normal).
        
        Returns:
            Tuple[float, float]: (lower_bound, upper_bound).
        
        Example:
            >>> ops = ProbabilityOperations()
            >>> np.random.seed(42)
            >>> data = np.random.normal(10, 2, 100)
            >>> ci = ops.confidence_interval(data, confidence=0.95)
            >>> ci[0] < 10 < ci[1]  # True mean should be in CI
            True
        """
        data_arr = np.asarray(data, dtype=np.float64)
        n = len(data_arr)
        mean = np.mean(data_arr)
        std_err = np.std(data_arr, ddof=1) / np.sqrt(n)
        
        alpha = 1 - confidence
        
        if method == 't':
            # Use t-distribution (better for small samples)
            t_crit = scipy_stats.t.ppf(1 - alpha / 2, df=n - 1)
            margin = t_crit * std_err
        elif method == 'z':
            # Use normal distribution (large samples)
            z_crit = scipy_stats.norm.ppf(1 - alpha / 2)
            margin = z_crit * std_err
        else:
            raise ValueError(f"Method must be 't' or 'z', got '{method}'")
        
        lower = mean - margin
        upper = mean + margin
        
        logger.debug(f"{confidence*100:.0f}% CI: [{lower:.4f}, {upper:.4f}]")
        return (float(lower), float(upper))


class HypothesisTesting:
    """
    Statistical hypothesis testing for machine learning.
    
    This class provides methods for:
    - t-tests (one-sample, two-sample, paired)
    - z-tests
    - chi-squared tests
    - ANOVA
    - Mann-Whitney U test
    
    Example:
        >>> ht = HypothesisTesting()
        >>> # Two-sample t-test
        >>> sample1 = np.random.normal(10, 2, 100)
        >>> sample2 = np.random.normal(11, 2, 100)
        >>> t_stat, p_value = ht.two_sample_t_test(sample1, sample2)
        >>> print(f"t = {t_stat:.4f}, p = {p_value:.4f}")
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize hypothesis testing.
        
        Args:
            alpha: Significance level. Default: 0.05.
        """
        self.alpha = alpha
        logger.debug(f"HypothesisTesting initialized with alpha={alpha}")
    
    def one_sample_t_test(
        self,
        sample: np.ndarray,
        pop_mean: float = 0.0
    ) -> Tuple[float, float]:
        """
        Perform one-sample t-test.
        
        Tests if sample mean differs significantly from population mean.
        
        Null hypothesis: sample mean = population mean
        
        Args:
            sample: Sample data.
            pop_mean: Hypothesized population mean.
        
        Returns:
            Tuple[float, float]: (t-statistic, p-value).
        
        Example:
            >>> ht = HypothesisTesting()
            >>> np.random.seed(42)
            >>> sample = np.random.normal(10, 2, 50)
            >>> t_stat, p_value = ht.one_sample_t_test(sample, pop_mean=10)
            >>> p_value > 0.05  # Should not reject null
            True
        """
        sample_arr = np.asarray(sample, dtype=np.float64)
        n = len(sample_arr)
        sample_mean = np.mean(sample_arr)
        sample_std = np.std(sample_arr, ddof=1)
        
        t_stat = (sample_mean - pop_mean) / (sample_std / np.sqrt(n))
        df = n - 1
        
        # Two-tailed p-value
        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df))
        
        logger.debug(f"One-sample t-test: t={t_stat:.4f}, p={p_value:.4f}, df={df}")
        return (float(t_stat), float(p_value))
    
    def two_sample_t_test(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray,
        equal_var: bool = True
    ) -> Tuple[float, float]:
        """
        Perform two-sample t-test (independent samples).
        
        Tests if two samples have significantly different means.
        
        Null hypothesis: μ₁ = μ₂
        
        Args:
            sample1: First sample.
            sample2: Second sample.
            equal_var: If True, assume equal variances (Student's t-test).
                      If False, use Welch's t-test.
        
        Returns:
            Tuple[float, float]: (t-statistic, p-value).
        
        Example:
            >>> ht = HypothesisTesting()
            >>> np.random.seed(42)
            >>> sample1 = np.random.normal(10, 2, 50)
            >>> sample2 = np.random.normal(11, 2, 50)
            >>> t_stat, p_value = ht.two_sample_t_test(sample1, sample2)
            >>> print(f"t = {t_stat:.4f}, p = {p_value:.4f}")
        """
        s1 = np.asarray(sample1, dtype=np.float64)
        s2 = np.asarray(sample2, dtype=np.float64)
        
        n1, n2 = len(s1), len(s2)
        mean1, mean2 = np.mean(s1), np.mean(s2)
        var1, var2 = np.var(s1, ddof=1), np.var(s2, ddof=1)
        
        if equal_var:
            # Student's t-test (pooled variance)
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            se = np.sqrt(pooled_var * (1/n1 + 1/n2))
            df = n1 + n2 - 2
        else:
            # Welch's t-test
            se = np.sqrt(var1/n1 + var2/n2)
            # Welch-Satterthwaite degrees of freedom
            num = (var1/n1 + var2/n2) ** 2
            denom = (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
            df = num / denom
        
        t_stat = (mean1 - mean2) / se
        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df))
        
        logger.debug(f"Two-sample t-test: t={t_stat:.4f}, p={p_value:.4f}, df={df:.2f}")
        return (float(t_stat), float(p_value))
    
    def paired_t_test(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform paired t-test (dependent samples).
        
        Tests if the mean difference between paired observations is zero.
        
        Null hypothesis: mean difference = 0
        
        Args:
            sample1: First set of paired observations.
            sample2: Second set of paired observations.
        
        Returns:
            Tuple[float, float]: (t-statistic, p-value).
        
        Example:
            >>> ht = HypothesisTesting()
            >>> np.random.seed(42)
            >>> before = np.random.normal(10, 2, 30)
            >>> after = before + np.random.normal(0.5, 0.5, 30)  # Treatment effect
            >>> t_stat, p_value = ht.paired_t_test(before, after)
            >>> print(f"t = {t_stat:.4f}, p = {p_value:.4f}")
        """
        s1 = np.asarray(sample1, dtype=np.float64)
        s2 = np.asarray(sample2, dtype=np.float64)
        
        if s1.shape != s2.shape:
            raise ValueError("Samples must have same shape for paired test")
        
        differences = s1 - s2
        return self.one_sample_t_test(differences, pop_mean=0.0)
    
    def z_test(
        self,
        sample: np.ndarray,
        pop_mean: float = 0.0,
        pop_std: float = 1.0
    ) -> Tuple[float, float]:
        """
        Perform one-sample z-test (known population variance).
        
        Null hypothesis: sample mean = population mean
        
        Args:
            sample: Sample data.
            pop_mean: Hypothesized population mean.
            pop_std: Known population standard deviation.
        
        Returns:
            Tuple[float, float]: (z-statistic, p-value).
        
        Example:
            >>> ht = HypothesisTesting()
            >>> np.random.seed(42)
            >>> sample = np.random.normal(10, 2, 100)
            >>> z_stat, p_value = ht.z_test(sample, pop_mean=10, pop_std=2)
            >>> print(f"z = {z_stat:.4f}, p = {p_value:.4f}")
        """
        sample_arr = np.asarray(sample, dtype=np.float64)
        n = len(sample_arr)
        sample_mean = np.mean(sample_arr)
        
        z_stat = (sample_mean - pop_mean) / (pop_std / np.sqrt(n))
        p_value = 2 * (1 - scipy_stats.norm.cdf(abs(z_stat)))
        
        logger.debug(f"Z-test: z={z_stat:.4f}, p={p_value:.4f}")
        return (float(z_stat), float(p_value))
    
    def chi_squared_test(
        self,
        observed: np.ndarray,
        expected: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Perform chi-squared goodness-of-fit test.
        
        Tests if observed frequencies differ significantly from expected.
        
        Null hypothesis: observed = expected
        
        Args:
            observed: Observed frequencies.
            expected: Expected frequencies. If None, assumes uniform.
        
        Returns:
            Tuple[float, float]: (chi-squared statistic, p-value).
        
        Example:
            >>> ht = HypothesisTesting()
            >>> # Test if dice is fair
            >>> observed = np.array([10, 12, 8, 11, 9, 10])
            >>> expected = np.array([10, 10, 10, 10, 10, 10])
            >>> chi2, p_value = ht.chi_squared_test(observed, expected)
            >>> p_value > 0.05  # Should not reject null (fair dice)
            True
        """
        obs = np.asarray(observed, dtype=np.float64)
        
        if expected is None:
            exp = np.full_like(obs, np.sum(obs) / len(obs))
        else:
            exp = np.asarray(expected, dtype=np.float64)
        
        # Chi-squared statistic
        chi2 = np.sum((obs - exp) ** 2 / exp)
        df = len(obs) - 1
        
        p_value = 1 - scipy_stats.chi2.cdf(chi2, df)
        
        logger.debug(f"Chi-squared test: χ²={chi2:.4f}, p={p_value:.4f}, df={df}")
        return (float(chi2), float(p_value))
    
    def chi_squared_independence(
        self,
        contingency_table: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform chi-squared test of independence.
        
        Tests if two categorical variables are independent.
        
        Null hypothesis: variables are independent
        
        Args:
            contingency_table: Contingency table of observed frequencies.
        
        Returns:
            Tuple[float, float]: (chi-squared statistic, p-value).
        
        Example:
            >>> ht = HypothesisTesting()
            >>> # Gender vs Preference
            >>> table = np.array([[30, 20], [15, 35]])
            >>> chi2, p_value = ht.chi_squared_independence(table)
            >>> print(f"χ² = {chi2:.4f}, p = {p_value:.4f}")
        """
        table = np.asarray(contingency_table, dtype=np.float64)
        
        # Calculate expected frequencies
        row_totals = np.sum(table, axis=1, keepdims=True)
        col_totals = np.sum(table, axis=0, keepdims=True)
        total = np.sum(table)
        
        expected = np.outer(row_totals.flatten(), col_totals.flatten()) / total
        
        # Chi-squared statistic
        chi2 = np.sum((table - expected) ** 2 / expected)
        
        df = (table.shape[0] - 1) * (table.shape[1] - 1)
        p_value = 1 - scipy_stats.chi2.cdf(chi2, df)
        
        logger.debug(f"Chi-squared independence test: χ²={chi2:.4f}, p={p_value:.4f}, df={df}")
        return (float(chi2), float(p_value))
    
    def anova_one_way(self, *samples) -> Tuple[float, float]:
        """
        Perform one-way ANOVA test.
        
        Tests if three or more groups have significantly different means.
        
        Null hypothesis: all group means are equal
        
        Args:
            *samples: Variable number of sample arrays.
        
        Returns:
            Tuple[float, float]: (F-statistic, p-value).
        
        Example:
            >>> ht = HypothesisTesting()
            >>> np.random.seed(42)
            >>> group1 = np.random.normal(10, 2, 30)
            >>> group2 = np.random.normal(11, 2, 30)
            >>> group3 = np.random.normal(12, 2, 30)
            >>> f_stat, p_value = ht.anova_one_way(group1, group2, group3)
            >>> print(f"F = {f_stat:.4f}, p = {p_value:.4f}")
        """
        if len(samples) < 2:
            raise ValueError("Need at least 2 groups for ANOVA")
        
        # Use scipy's implementation
        f_stat, p_value = scipy_stats.f_oneway(*samples)
        
        logger.debug(f"One-way ANOVA: F={f_stat:.4f}, p={p_value:.4f}")
        return (float(f_stat), float(p_value))
    
    def mann_whitney_u(
        self,
        sample1: np.ndarray,
        sample2: np.ndarray
    ) -> Tuple[float, float]:
        """
        Perform Mann-Whitney U test (non-parametric).
        
        Tests if two independent samples come from the same distribution.
        Alternative to t-test when normality assumption is violated.
        
        Null hypothesis: distributions are equal
        
        Args:
            sample1: First sample.
            sample2: Second sample.
        
        Returns:
            Tuple[float, float]: (U statistic, p-value).
        
        Example:
            >>> ht = HypothesisTesting()
            >>> np.random.seed(42)
            >>> sample1 = np.random.exponential(2, 50)
            >>> sample2 = np.random.exponential(3, 50)
            >>> u_stat, p_value = ht.mann_whitney_u(sample1, sample2)
            >>> print(f"U = {u_stat:.4f}, p = {p_value:.4f}")
        """
        s1 = np.asarray(sample1, dtype=np.float64)
        s2 = np.asarray(sample2, dtype=np.float64)
        
        u_stat, p_value = scipy_stats.mannwhitneyu(s1, s2, alternative='two-sided')
        
        logger.debug(f"Mann-Whitney U test: U={u_stat:.4f}, p={p_value:.4f}")
        return (float(u_stat), float(p_value))
    
    def test_normality(
        self,
        sample: np.ndarray,
        method: str = 'shapiro'
    ) -> Tuple[float, float]:
        """
        Test if sample comes from a normal distribution.
        
        Methods:
        - 'shapiro': Shapiro-Wilk test (best for small samples)
        - 'kstest': Kolmogorov-Smirnov test
        - 'anderson': Anderson-Darling test
        
        Null hypothesis: sample is normally distributed
        
        Args:
            sample: Sample data.
            method: Test method. Default: 'shapiro'.
        
        Returns:
            Tuple[float, float]: (test statistic, p-value).
        
        Example:
            >>> ht = HypothesisTesting()
            >>> np.random.seed(42)
            >>> sample = np.random.normal(10, 2, 100)
            >>> stat, p_value = ht.test_normality(sample)
            >>> p_value > 0.05  # Should not reject normality
            True
        """
        sample_arr = np.asarray(sample, dtype=np.float64)
        
        if method == 'shapiro':
            stat, p_value = scipy_stats.shapiro(sample_arr)
        elif method == 'kstest':
            # Normalize and test against standard normal
            normalized = (sample_arr - np.mean(sample_arr)) / np.std(sample_arr, ddof=1)
            stat, p_value = scipy_stats.kstest(normalized, 'norm')
        elif method == 'anderson':
            result = scipy_stats.anderson(sample_arr, dist='norm')
            stat = result.statistic
            # Anderson-Darling doesn't give exact p-value
            p_value = 0.05 if stat < result.critical_values[1] else 0.01
        else:
            raise ValueError(f"Method must be 'shapiro', 'kstest', or 'anderson'")
        
        logger.debug(f"Normality test ({method}): stat={stat:.4f}, p={p_value:.4f}")
        return (float(stat), float(p_value))


if __name__ == "__main__":
    # Example usage and demonstrations
    print("=" * 60)
    print("Probability Module - Demonstration")
    print("=" * 60)
    
    ops = ProbabilityOperations()
    ht = HypothesisTesting()
    
    # Bayes theorem
    print("\n1. Bayes' Theorem:")
    print("   Medical test example:")
    print("   - Sensitivity P(+|disease) = 0.99")
    print("   - Prevalence P(disease) = 0.001")
    print("   - False positive P(+|no disease) = 0.01")
    p_posterior = ops.bayes_with_marginal(0.99, 0.001, 0.01)
    print(f"   P(disease|positive) = {p_posterior:.4f}")
    
    # Distributions
    print("\n2. Probability Distributions:")
    dist = Distribution.normal(mean=0, std=1)
    samples = dist.sample(10000, seed=42)
    print(f"   Normal(0, 1): mean={np.mean(samples):.4f}, std={np.std(samples):.4f}")
    
    # Expectation and variance
    print("\n3. Expectation and Variance:")
    dice = np.array([1, 2, 3, 4, 5, 6])
    print(f"   Expected dice roll: {ops.expectation(dice)}")
    print(f"   Variance of dice roll: {ops.variance(dice):.4f}")
    
    # Correlation
    print("\n4. Correlation:")
    x = np.random.randn(100)
    y = 2 * x + np.random.randn(100) * 0.5
    print(f"   Correlation(x, y): {ops.correlation(x, y):.4f}")
    
    # KL divergence
    print("\n5. KL Divergence:")
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.4, 0.4, 0.2])
    print(f"   D_KL(P||Q) = {ops.kl_divergence(p, q):.4f} nats")
    
    # Entropy
    print("\n6. Entropy:")
    fair_coin = np.array([0.5, 0.5])
    biased_coin = np.array([0.9, 0.1])
    print(f"   Fair coin entropy: {ops.entropy(fair_coin, base=2):.4f} bits")
    print(f"   Biased coin entropy: {ops.entropy(biased_coin, base=2):.4f} bits")
    
    # Hypothesis testing
    print("\n7. Hypothesis Testing:")
    np.random.seed(42)
    sample1 = np.random.normal(10, 2, 50)
    sample2 = np.random.normal(11, 2, 50)
    t_stat, p_value = ht.two_sample_t_test(sample1, sample2)
    print(f"   Two-sample t-test: t={t_stat:.4f}, p={p_value:.4f}")
    
    # Confidence interval
    print("\n8. Confidence Interval:")
    data = np.random.normal(10, 2, 100)
    ci = ops.confidence_interval(data, confidence=0.95)
    print(f"   95% CI for mean: [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # MLE
    print("\n9. Maximum Likelihood Estimation:")
    data = np.random.normal(5, 2, 1000)
    params = ops.maximum_likelihood_estimation(data)
    print(f"   Estimated: mean={params['mean']:.4f}, std={params['std']:.4f}")
    print(f"   True: mean=5.0, std=2.0")
    
    print("\n" + "=" * 60)
