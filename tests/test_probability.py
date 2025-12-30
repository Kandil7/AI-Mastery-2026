"""
Unit tests for Probability Module
=================================
Tests for distributions, sampling, and information theory functions.
"""

import pytest
import numpy as np
from src.core.probability import (
    Gaussian, MultivariateGaussian, Bernoulli, Categorical, Uniform, Exponential,
    entropy, cross_entropy, kl_divergence, js_divergence, mutual_information,
    bayes_theorem, posterior_update, beta_binomial_posterior, gaussian_posterior,
    log_sum_exp, normalize_log_probs
)


class TestGaussian:
    """Tests for Gaussian distribution."""
    
    def test_standard_normal_mean(self):
        """Standard normal should have mean 0."""
        dist = Gaussian(mean=0, std=1)
        samples = dist.sample(10000)
        assert abs(np.mean(samples)) < 0.1
    
    def test_standard_normal_std(self):
        """Standard normal should have std 1."""
        dist = Gaussian(mean=0, std=1)
        samples = dist.sample(10000)
        assert abs(np.std(samples) - 1.0) < 0.1
    
    def test_custom_mean_std(self):
        """Custom mean and std should be reflected in samples."""
        dist = Gaussian(mean=5, std=2)
        samples = dist.sample(10000)
        assert abs(np.mean(samples) - 5) < 0.2
        assert abs(np.std(samples) - 2) < 0.2
    
    def test_pdf_at_mean(self):
        """PDF should be maximum at mean."""
        dist = Gaussian(mean=0, std=1)
        assert dist.pdf(np.array([0]))[0] > dist.pdf(np.array([1]))[0]
    
    def test_cdf_properties(self):
        """CDF should be 0.5 at mean for symmetric distribution."""
        dist = Gaussian(mean=0, std=1)
        assert abs(dist.cdf(np.array([0]))[0] - 0.5) < 0.01
    
    def test_mean_property(self):
        """Mean property should match construction."""
        dist = Gaussian(mean=3, std=2)
        assert dist.mean == 3
    
    def test_variance_property(self):
        """Variance property should be std^2."""
        dist = Gaussian(mean=0, std=3)
        assert dist.variance == 9


class TestBernoulli:
    """Tests for Bernoulli distribution."""
    
    def test_sample_range(self):
        """Samples should be 0 or 1."""
        dist = Bernoulli(p=0.5)
        samples = dist.sample(100)
        assert all(s in [0, 1] for s in samples)
    
    def test_empirical_probability(self):
        """Empirical mean should approximate p."""
        dist = Bernoulli(p=0.7)
        samples = dist.sample(10000)
        assert abs(np.mean(samples) - 0.7) < 0.05
    
    def test_pdf(self):
        """PMF should be correct."""
        dist = Bernoulli(p=0.3)
        assert dist.pdf(np.array([1]))[0] == 0.3
        assert dist.pdf(np.array([0]))[0] == 0.7


class TestCategorical:
    """Tests for Categorical distribution."""
    
    def test_sample_range(self):
        """Samples should be valid indices."""
        probs = [0.2, 0.3, 0.5]
        dist = Categorical(probs)
        samples = dist.sample(100)
        assert all(0 <= s < 3 for s in samples)
    
    def test_empirical_distribution(self):
        """Empirical distribution should approximate probs."""
        probs = [0.2, 0.3, 0.5]
        dist = Categorical(probs)
        samples = dist.sample(10000)
        counts = np.bincount(samples, minlength=3) / len(samples)
        for i, p in enumerate(probs):
            assert abs(counts[i] - p) < 0.05


class TestUniform:
    """Tests for Uniform distribution."""
    
    def test_sample_range(self):
        """Samples should be in [low, high]."""
        dist = Uniform(low=2, high=5)
        samples = dist.sample(100)
        assert all(2 <= s <= 5 for s in samples)
    
    def test_mean(self):
        """Mean should be (low + high) / 2."""
        dist = Uniform(low=0, high=10)
        assert dist.mean == 5


class TestExponential:
    """Tests for Exponential distribution."""
    
    def test_positive_samples(self):
        """All samples should be positive."""
        dist = Exponential(rate=1)
        samples = dist.sample(100)
        assert all(s >= 0 for s in samples)
    
    def test_mean(self):
        """Mean should be 1/rate."""
        dist = Exponential(rate=2)
        assert dist.mean == 0.5


class TestInformationTheory:
    """Tests for information theory functions."""
    
    def test_entropy_uniform(self):
        """Uniform distribution should have max entropy."""
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert abs(entropy(p) - 2.0) < 0.01  # log2(4) = 2
    
    def test_entropy_deterministic(self):
        """Deterministic distribution should have zero entropy."""
        p = np.array([1.0, 0.0, 0.0, 0.0])
        assert abs(entropy(p)) < 0.01
    
    def test_kl_divergence_same(self):
        """KL divergence of same distribution should be 0."""
        p = np.array([0.3, 0.4, 0.3])
        assert abs(kl_divergence(p, p)) < 0.01
    
    def test_kl_divergence_positive(self):
        """KL divergence should be non-negative."""
        p = np.array([0.4, 0.4, 0.2])
        q = np.array([0.3, 0.3, 0.4])
        assert kl_divergence(p, q) >= 0
    
    def test_js_divergence_symmetric(self):
        """JS divergence should be symmetric."""
        p = np.array([0.4, 0.4, 0.2])
        q = np.array([0.3, 0.3, 0.4])
        assert abs(js_divergence(p, q) - js_divergence(q, p)) < 0.01
    
    def test_cross_entropy(self):
        """Cross entropy should be >= entropy."""
        p = np.array([0.3, 0.4, 0.3])
        q = np.array([0.2, 0.5, 0.3])
        # H(p, q) >= H(p)
        assert cross_entropy(p, q) >= entropy(p, base=np.e) - 0.01


class TestBayesian:
    """Tests for Bayesian utilities."""
    
    def test_bayes_theorem(self):
        """Basic Bayes theorem calculation."""
        # P(A|B) = P(B|A) * P(A) / P(B)
        posterior = bayes_theorem(prior=0.5, likelihood=0.8, evidence=0.4)
        assert abs(posterior - 1.0) < 0.01
    
    def test_posterior_update_normalization(self):
        """Posterior should sum to 1."""
        prior = np.array([0.3, 0.3, 0.4])
        likelihood = np.array([0.8, 0.5, 0.2])
        posterior = posterior_update(prior, likelihood)
        assert abs(np.sum(posterior) - 1.0) < 0.01
    
    def test_beta_binomial_update(self):
        """Beta-binomial conjugate update should be correct."""
        alpha, beta = beta_binomial_posterior(
            alpha_prior=1, beta_prior=1,
            successes=3, failures=2
        )
        assert alpha == 4
        assert beta == 3


class TestUtilities:
    """Tests for utility functions."""
    
    def test_log_sum_exp(self):
        """log_sum_exp should be numerically stable."""
        x = np.array([1000, 1001, 1002])
        result = log_sum_exp(x)
        # Should not overflow
        assert np.isfinite(result)
        assert result > 1000
    
    def test_normalize_log_probs(self):
        """Normalized log probs should sum to 1."""
        log_probs = np.array([-1, -2, -3])
        probs = normalize_log_probs(log_probs)
        assert abs(np.sum(probs) - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
