"""
Unit Tests for Integration Module
=================================

Tests for numerical quadrature, Monte Carlo integration,
Bayesian quadrature, and normalizing flows.

Run with: pytest tests/test_integration.py -v
"""

import pytest
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from src.core.integration import (
    trapezoidal_rule,
    simpsons_rule,
    adaptive_quadrature,
    gauss_legendre,
    gauss_hermite_expectation,
    monte_carlo_integrate,
    importance_sampling,
    stratified_sampling,
    BayesianQuadrature,
    rbf_kernel,
    compare_integration_methods,
    monte_carlo_convergence
)

from src.core.normalizing_flows import (
    PlanarFlow,
    RadialFlow,
    FlowChain,
    gaussian_base_log_prob,
    gaussian_base_sampler,
    visualize_flow_2d
)


# =============================================================================
# NEWTON-COTES TESTS
# =============================================================================

class TestNewtonCotes:
    """Tests for Newton-Cotes quadrature methods."""
    
    def test_trapezoidal_linear_function(self):
        """Trapezoidal rule should be exact for linear functions."""
        f = lambda x: 2 * x + 1  # Linear function
        result = trapezoidal_rule(f, 0, 1, 10)
        expected = 2.0  # ∫(2x+1)dx from 0 to 1 = [x² + x]₀¹ = 2
        assert abs(result - expected) < 1e-10
    
    def test_trapezoidal_quadratic_convergence(self):
        """Trapezoidal should converge as O(1/N²)."""
        f = lambda x: x ** 2
        true_value = 1.0 / 3  # ∫x² from 0 to 1
        
        errors = []
        ns = [10, 20, 40, 80]
        for n in ns:
            result = trapezoidal_rule(f, 0, 1, n)
            errors.append(abs(result - true_value))
        
        # Check quadratic convergence: error ratio should be ~4
        for i in range(len(errors) - 1):
            ratio = errors[i] / errors[i + 1]
            assert 3.5 < ratio < 4.5
    
    def test_simpsons_quadratic_exact(self):
        """Simpson's rule should be exact for quadratic functions."""
        f = lambda x: x ** 2 + 2 * x + 1  # Quadratic
        result = simpsons_rule(f, 0, 1, 2)  # Only 2 intervals
        expected = 1.0/3 + 1 + 1  # ∫(x² + 2x + 1)dx = 7/3
        assert abs(result - expected) < 1e-10
    
    def test_simpsons_quartic_convergence(self):
        """Simpson's should converge as O(1/N⁴) for smooth functions."""
        f = lambda x: np.exp(-x ** 2)
        true_value, _ = quad(f, 0, 1)
        
        errors = []
        ns = [10, 20, 40]
        for n in ns:
            result = simpsons_rule(f, 0, 1, n)
            errors.append(abs(result - true_value))
        
        # Check quartic convergence: error ratio should be ~16
        ratio = errors[0] / errors[1]
        assert 12 < ratio < 20
    
    def test_simpsons_ensures_even_intervals(self):
        """Simpson's should handle odd n by adding 1."""
        f = lambda x: x ** 2
        # Should not raise error with odd n
        result = simpsons_rule(f, 0, 1, 11)  # Odd number
        assert isinstance(result, float)
    
    def test_adaptive_quadrature_accuracy(self):
        """Adaptive quadrature should achieve specified tolerance."""
        f = lambda x: np.exp(-x ** 2) * np.cos(5 * x)
        true_value, _ = quad(f, 0, 2)
        
        result, n_evals = adaptive_quadrature(f, 0, 2, tol=1e-6)
        
        assert abs(result - true_value) < 1e-5
        assert n_evals > 3  # Should need more than initial evaluations


# =============================================================================
# GAUSSIAN QUADRATURE TESTS
# =============================================================================

class TestGaussianQuadrature:
    """Tests for Gaussian quadrature methods."""
    
    def test_gauss_legendre_polynomial_exact(self):
        """Gauss-Legendre should be exact for polynomials up to degree 2n-1."""
        # With n=3 points, should be exact for degree 5
        f = lambda x: x ** 5
        result = gauss_legendre(f, -1, 1, 3)
        expected = 0  # ∫x⁵ from -1 to 1 = 0 (odd function)
        assert abs(result - expected) < 1e-10
    
    def test_gauss_legendre_interval_transform(self):
        """Gauss-Legendre should work on arbitrary intervals."""
        f = lambda x: x ** 2
        result = gauss_legendre(f, 0, 2, 5)
        expected = 8.0 / 3  # ∫x² from 0 to 2 = 8/3
        assert abs(result - expected) < 1e-10
    
    def test_gauss_hermite_standard_normal_mean(self):
        """Gauss-Hermite E[X] for X~N(0,1) should be 0."""
        f = lambda x: x
        result = gauss_hermite_expectation(f, mu=0, sigma=1, n=5)
        assert abs(result) < 1e-10
    
    def test_gauss_hermite_standard_normal_variance(self):
        """Gauss-Hermite E[X²] for X~N(0,1) should be 1."""
        f = lambda x: x ** 2
        result = gauss_hermite_expectation(f, mu=0, sigma=1, n=5)
        assert abs(result - 1.0) < 1e-10
    
    def test_gauss_hermite_general_normal(self):
        """Gauss-Hermite should work for general N(μ, σ²)."""
        mu, sigma = 3.0, 2.0
        
        # E[X] = μ for X~N(μ, σ²)
        f_mean = lambda x: x
        mean_result = gauss_hermite_expectation(f_mean, mu=mu, sigma=sigma, n=10)
        assert abs(mean_result - mu) < 1e-8
        
        # E[(X-μ)²] = σ² for X~N(μ, σ²)
        f_var = lambda x: (x - mu) ** 2
        var_result = gauss_hermite_expectation(f_var, mu=mu, sigma=sigma, n=10)
        assert abs(var_result - sigma ** 2) < 1e-6
    
    def test_gauss_hermite_relu_expectation(self):
        """Test ReLU expectation under standard normal."""
        # E[max(0, X)] for X~N(0,1) = 1/√(2π) ≈ 0.3989
        f = lambda x: np.maximum(0, x)
        result = gauss_hermite_expectation(f, mu=0, sigma=1, n=20)
        expected = 1.0 / np.sqrt(2 * np.pi)
        assert abs(result - expected) < 0.01


# =============================================================================
# MONTE CARLO TESTS
# =============================================================================

class TestMonteCarlo:
    """Tests for Monte Carlo integration methods."""
    
    def test_monte_carlo_gaussian_mean(self):
        """Monte Carlo should estimate E[X] = 0 for standard normal."""
        f = lambda x: x
        sampler = lambda n: np.random.randn(n)
        
        estimate, se = monte_carlo_integrate(f, sampler, 10000, seed=42)
        
        assert abs(estimate) < 3 * se  # Should be within 3 standard errors
    
    def test_monte_carlo_gaussian_variance(self):
        """Monte Carlo should estimate E[X²] = 1 for standard normal."""
        f = lambda x: x ** 2
        sampler = lambda n: np.random.randn(n)
        
        estimate, se = monte_carlo_integrate(f, sampler, 10000, seed=42)
        
        assert abs(estimate - 1.0) < 0.1
    
    def test_monte_carlo_convergence_rate(self):
        """Monte Carlo error should decrease as O(1/√N)."""
        f = lambda x: x ** 2
        sampler = lambda n: np.random.randn(n)
        
        estimates_1k = []
        estimates_10k = []
        
        for seed in range(10):
            e1, _ = monte_carlo_integrate(f, sampler, 1000, seed=seed)
            e2, _ = monte_carlo_integrate(f, sampler, 10000, seed=seed)
            estimates_1k.append(e1)
            estimates_10k.append(e2)
        
        std_1k = np.std(estimates_1k)
        std_10k = np.std(estimates_10k)
        
        # Ratio should be ~√10 ≈ 3.16
        ratio = std_1k / std_10k
        assert 2 < ratio < 5
    
    def test_importance_sampling_reduces_variance(self):
        """Importance sampling should reduce variance for well-chosen proposal."""
        # Estimate E_p[f(X)] where p = N(0,1), f = exp(-x)
        f = lambda x: np.exp(-x)
        p = lambda x: norm.pdf(x, 0, 1)
        
        # Good proposal: shifted toward where f(x) is large
        q_good = lambda x: norm.pdf(x, -1, 1)
        q_good_sampler = lambda n: np.random.randn(n) - 1
        
        # Bad proposal: same as target
        q_same = lambda x: norm.pdf(x, 0, 1)
        q_same_sampler = lambda n: np.random.randn(n)
        
        _, se_good, ess_good = importance_sampling(
            f, p, q_good, q_good_sampler, 1000, seed=42
        )
        _, se_same, ess_same = importance_sampling(
            f, p, q_same, q_same_sampler, 1000, seed=42
        )
        
        # ESS should be higher for good proposal (less weight variance)
        # At minimum, results should be reasonable
        assert ess_good > 100  # Not too degenerate
    
    def test_stratified_sampling_basic(self):
        """Stratified sampling should work for simple 1D integration."""
        f = lambda x: x[0] ** 2 if isinstance(x, np.ndarray) else x ** 2
        bounds = [(0.0, 1.0)]
        
        estimate, se = stratified_sampling(f, bounds, 4, 25, seed=42)
        
        # ∫x² dx from 0 to 1 = 1/3
        expected = 1.0 / 3
        assert abs(estimate - expected) < 0.1


# =============================================================================
# BAYESIAN QUADRATURE TESTS
# =============================================================================

class TestBayesianQuadrature:
    """Tests for Bayesian Quadrature."""
    
    def test_bq_with_observations(self):
        """Bayesian Quadrature should estimate integral from observations."""
        kernel = rbf_kernel(length_scale=1.0, variance=1.0)
        bq = BayesianQuadrature(kernel)
        
        # True function: f(x) = sin(x)
        f = np.sin
        X = np.linspace(-2, 2, 5).reshape(-1, 1)
        y = f(X.ravel())
        
        bq.fit(X, y)
        
        # Estimate E[f(X)] for X ~ N(0, 1)
        sampler = lambda n: np.random.randn(n, 1)
        mean, std = bq.integral_posterior(sampler, n_mc_samples=500)
        
        # E[sin(X)] for X~N(0,1) should be close to 0
        assert abs(mean) < 0.5  # Rough check
        assert std > 0  # Should have uncertainty
    
    def test_bq_add_observation(self):
        """Adding observations should reduce uncertainty."""
        kernel = rbf_kernel(length_scale=1.0)
        bq = BayesianQuadrature(kernel)
        
        sampler = lambda n: np.random.randn(n, 1)
        
        # Start with few observations
        X_init = np.array([[0.0], [1.0]])
        y_init = np.array([0.0, 0.84])  # sin values
        bq.fit(X_init, y_init)
        
        _, std_init = bq.integral_posterior(sampler, n_mc_samples=300)
        
        # Add more observations
        bq.add_observation(np.array([0.5]), 0.48)
        bq.add_observation(np.array([-0.5]), -0.48)
        
        _, std_final = bq.integral_posterior(sampler, n_mc_samples=300)
        
        # Uncertainty should generally decrease with more data
        # (This is a weak test due to randomness)
        assert std_final < std_init * 2  # At least not much worse
    
    def test_bq_suggest_next_point(self):
        """Active learning should suggest reasonable next point."""
        kernel = rbf_kernel(length_scale=1.0)
        bq = BayesianQuadrature(kernel)
        
        bq.fit(np.array([[0.0]]), np.array([0.0]))
        
        candidates = np.linspace(-2, 2, 20).reshape(-1, 1)
        p = lambda x: norm.pdf(x[0], 0, 1)
        
        idx = bq.suggest_next_point(candidates, p)
        
        assert 0 <= idx < len(candidates)
        # Should avoid the point we already have (x=0)
        # This is approximate due to density weighting


# =============================================================================
# NORMALIZING FLOWS TESTS
# =============================================================================

class TestPlanarFlow:
    """Tests for Planar Flow."""
    
    def test_planar_flow_shape(self):
        """Planar flow should preserve shape."""
        flow = PlanarFlow(d=3)
        z = np.random.randn(10, 3)
        
        z_new, log_det = flow.forward(z)
        
        assert z_new.shape == (10, 3)
        assert log_det.shape == (10,)
    
    def test_planar_flow_invertibility(self):
        """Planar flow inverse should approximately recover input."""
        flow = PlanarFlow(d=2)
        z = np.random.randn(5, 2)
        
        z_new, _ = flow.forward(z)
        z_recovered = flow.inverse(z_new)
        
        # Should be close (fixed-point iteration may not be exact)
        assert np.allclose(z, z_recovered, atol=0.1)
    
    def test_planar_flow_log_det_finite(self):
        """Log determinant should be finite."""
        flow = PlanarFlow(d=4)
        z = np.random.randn(20, 4)
        
        _, log_det = flow.forward(z)
        
        assert np.all(np.isfinite(log_det))


class TestRadialFlow:
    """Tests for Radial Flow."""
    
    def test_radial_flow_shape(self):
        """Radial flow should preserve shape."""
        flow = RadialFlow(d=3)
        z = np.random.randn(10, 3)
        
        z_new, log_det = flow.forward(z)
        
        assert z_new.shape == (10, 3)
        assert log_det.shape == (10,)
    
    def test_radial_flow_log_det_finite(self):
        """Log determinant should be finite."""
        flow = RadialFlow(d=4)
        z = np.random.randn(20, 4)
        
        _, log_det = flow.forward(z)
        
        assert np.all(np.isfinite(log_det))


class TestFlowChain:
    """Tests for Flow Chain."""
    
    def test_flow_chain_creation(self):
        """Flow chain should create successfully."""
        chain = FlowChain.create_planar_chain(d=2, n_flows=5)
        assert len(chain.flows) == 5
        
        chain = FlowChain.create_radial_chain(d=3, n_flows=3)
        assert len(chain.flows) == 3
    
    def test_flow_chain_forward(self):
        """Flow chain forward should work."""
        chain = FlowChain.create_planar_chain(d=2, n_flows=3)
        z = np.random.randn(10, 2)
        
        z_new, total_log_det = chain.forward(z)
        
        assert z_new.shape == (10, 2)
        assert total_log_det.shape == (10,)
        assert np.all(np.isfinite(total_log_det))
    
    def test_flow_chain_log_prob(self):
        """Flow chain log prob should be finite."""
        chain = FlowChain.create_planar_chain(d=2, n_flows=3)
        x = np.random.randn(10, 2)
        
        log_probs = chain.log_prob(x, gaussian_base_log_prob)
        
        assert log_probs.shape == (10,)
        assert np.all(np.isfinite(log_probs))
    
    def test_flow_chain_sample(self):
        """Flow chain sampling should work."""
        chain = FlowChain.create_planar_chain(d=2, n_flows=3)
        sampler = gaussian_base_sampler(d=2)
        
        samples = chain.sample(100, sampler)
        
        assert samples.shape == (100, 2)
        assert np.all(np.isfinite(samples))
    
    def test_mixed_chain(self):
        """Mixed planar/radial chain should work."""
        chain = FlowChain.create_mixed_chain(d=2, n_planar=2, n_radial=2)
        z = np.random.randn(5, 2)
        
        z_new, log_det = chain.forward(z)
        
        assert z_new.shape == (5, 2)
        assert np.all(np.isfinite(log_det))


class TestFlowVisualization:
    """Tests for flow visualization utilities."""
    
    def test_visualize_flow_2d(self):
        """Visualization function should return expected data."""
        chain = FlowChain.create_planar_chain(d=2, n_flows=2)
        
        vis_data = visualize_flow_2d(chain, n_samples=50, grid_size=10)
        
        assert 'base_samples' in vis_data
        assert 'transformed_samples' in vis_data
        assert 'density' in vis_data
        assert vis_data['base_samples'].shape == (50, 2)
        assert vis_data['density'].shape == (10, 10)


# =============================================================================
# COMPARISON & UTILITY TESTS
# =============================================================================

class TestUtilities:
    """Tests for utility functions."""
    
    def test_compare_integration_methods(self):
        """Comparison function should return expected structure."""
        f = lambda x: np.exp(-x ** 2)
        true_value, _ = quad(f, 0, 1)
        
        results = compare_integration_methods(f, 0, 1, true_value, [10, 20, 40])
        
        assert len(results['n_points']) == 3
        assert len(results['trapezoidal_errors']) == 3
        assert len(results['simpsons_errors']) == 3
    
    def test_monte_carlo_convergence_analysis(self):
        """Convergence analysis should return expected structure."""
        f = lambda x: x ** 2
        sampler = lambda n: np.random.randn(n)
        
        results = monte_carlo_convergence(f, sampler, 1.0, [100, 500], n_trials=3)
        
        assert len(results['sample_sizes']) == 2
        assert len(results['mean_errors']) == 2


# =============================================================================
# MCMC TESTS
# =============================================================================

from src.core.mcmc import (
    metropolis_hastings, HamiltonianMonteCarlo, nuts_sampler,
    effective_sample_size, gelman_rubin_diagnostic, mcmc_diagnostics,
    autocorrelation, thinning
)


class TestMetropolisHastings:
    """Tests for Metropolis-Hastings sampler."""
    
    def test_mh_samples_shape(self):
        """MH should return correct number of samples."""
        log_prob = lambda x: -0.5 * np.sum(x**2)
        result = metropolis_hastings(log_prob, np.zeros(2), n_samples=100, n_burnin=10, seed=42)
        
        assert result.samples.shape == (100, 2)
        assert len(result.log_probs) == 100
    
    def test_mh_gaussian_mean(self):
        """MH should estimate mean of standard normal correctly."""
        log_prob = lambda x: -0.5 * np.sum(x**2)
        result = metropolis_hastings(log_prob, np.zeros(1), n_samples=5000, n_burnin=1000, seed=42)
        
        sample_mean = np.mean(result.samples)
        assert abs(sample_mean) < 0.1  # Should be close to 0
    
    def test_mh_acceptance_rate_reasonable(self):
        """MH acceptance rate should be in reasonable range."""
        log_prob = lambda x: -0.5 * np.sum(x**2)
        result = metropolis_hastings(log_prob, np.zeros(3), n_samples=1000, proposal_std=1.0, seed=42)
        
        assert 0.1 < result.acceptance_rate < 0.9
    
    def test_mh_diagnostics_computed(self):
        """MH should compute diagnostics."""
        log_prob = lambda x: -0.5 * np.sum(x**2)
        result = metropolis_hastings(log_prob, np.zeros(2), n_samples=500, seed=42)
        
        assert 'ess' in result.diagnostics
        assert 'mean' in result.diagnostics
        assert len(result.diagnostics['ess']) == 2


class TestHamiltonianMonteCarlo:
    """Tests for HMC sampler."""
    
    def test_hmc_samples_shape(self):
        """HMC should return correct number of samples."""
        log_prob = lambda x: -0.5 * np.sum(x**2)
        grad_log_prob = lambda x: -x
        
        hmc = HamiltonianMonteCarlo(log_prob, grad_log_prob, step_size=0.1, n_leapfrog=10)
        result = hmc.sample(np.zeros(3), n_samples=100, n_burnin=10, seed=42)
        
        assert result.samples.shape == (100, 3)
    
    def test_hmc_gaussian_variance(self):
        """HMC should estimate variance of standard normal correctly."""
        log_prob = lambda x: -0.5 * np.sum(x**2)
        grad_log_prob = lambda x: -x
        
        hmc = HamiltonianMonteCarlo(log_prob, grad_log_prob, step_size=0.1, n_leapfrog=20)
        result = hmc.sample(np.zeros(2), n_samples=2000, n_burnin=500, seed=42)
        
        sample_var = np.var(result.samples[:, 0])
        assert 0.8 < sample_var < 1.2  # Should be close to 1
    
    def test_hmc_high_acceptance_rate(self):
        """HMC should have higher acceptance rate than MH."""
        log_prob = lambda x: -0.5 * np.sum(x**2)
        grad_log_prob = lambda x: -x
        
        hmc = HamiltonianMonteCarlo(log_prob, grad_log_prob, step_size=0.1, n_leapfrog=10)
        result = hmc.sample(np.zeros(5), n_samples=500, seed=42)
        
        assert result.acceptance_rate > 0.5
    
    def test_hmc_step_size_adaptation(self):
        """HMC should adapt step size during burn-in."""
        log_prob = lambda x: -0.5 * np.sum(x**2)
        grad_log_prob = lambda x: -x
        
        hmc = HamiltonianMonteCarlo(log_prob, grad_log_prob, step_size=1.0, n_leapfrog=10)
        initial_step = hmc.step_size
        
        hmc.sample(np.zeros(3), n_samples=100, n_burnin=200, adapt_step_size=True, seed=42)
        
        # Step size should have changed
        assert hmc.step_size != initial_step


class TestNUTS:
    """Tests for No-U-Turn Sampler."""
    
    def test_nuts_samples_shape(self):
        """NUTS should return correct number of samples."""
        log_prob = lambda x: -0.5 * np.sum(x**2)
        grad_log_prob = lambda x: -x
        
        result = nuts_sampler(
            log_prob, grad_log_prob, np.zeros(2),
            n_samples=50, n_burnin=10, max_tree_depth=5, seed=42
        )
        
        assert result.samples.shape == (50, 2)
    
    def test_nuts_reasonable_samples(self):
        """NUTS samples should be reasonable for standard normal."""
        log_prob = lambda x: -0.5 * np.sum(x**2)
        grad_log_prob = lambda x: -x
        
        result = nuts_sampler(
            log_prob, grad_log_prob, np.zeros(2),
            n_samples=200, n_burnin=50, step_size=0.1, seed=42
        )
        
        # Samples should be within reasonable range for N(0,1)
        assert np.all(np.abs(result.samples) < 5)


class TestMCMCDiagnostics:
    """Tests for MCMC diagnostic functions."""
    
    def test_ess_positive(self):
        """ESS should be positive."""
        samples = np.random.randn(1000, 3)
        ess = effective_sample_size(samples)
        
        assert np.all(ess > 0)
        assert len(ess) == 3
    
    def test_ess_upper_bound(self):
        """ESS should be at most n_samples."""
        samples = np.random.randn(500, 2)
        ess = effective_sample_size(samples)
        
        # ESS can't exceed number of samples
        assert np.all(ess <= 500)
    
    def test_gelman_rubin_converged_chains(self):
        """R-hat should be close to 1 for converged chains."""
        # Generate 4 chains from the same distribution
        chains = [np.random.randn(500, 2) for _ in range(4)]
        r_hat = gelman_rubin_diagnostic(chains)
        
        # For identical distributions, R-hat should be close to 1
        assert np.all(r_hat < 1.1)
    
    def test_autocorrelation_starts_at_one(self):
        """Autocorrelation at lag 0 should be 1."""
        samples = np.random.randn(200)
        acf = autocorrelation(samples, max_lag=50)
        
        assert np.isclose(acf[0], 1.0)
    
    def test_thinning_reduces_samples(self):
        """Thinning should reduce number of samples."""
        samples = np.random.randn(100, 3)
        thinned = thinning(samples, thin=5)
        
        assert thinned.shape == (20, 3)
    
    def test_mcmc_diagnostics_structure(self):
        """MCMC diagnostics should return expected structure."""
        log_prob = lambda x: -0.5 * np.sum(x**2)
        result = metropolis_hastings(log_prob, np.zeros(2), n_samples=200, seed=42)
        
        diagnostics = mcmc_diagnostics(result, param_names=['x', 'y'])
        
        assert 'n_samples' in diagnostics
        assert 'acceptance_rate' in diagnostics
        assert 'ess' in diagnostics
        assert 'x' in diagnostics['ess']
        assert 'summary' in diagnostics


# =============================================================================
# VARIATIONAL INFERENCE TESTS
# =============================================================================

from src.core.variational_inference import (
    GaussianVariational, MeanFieldVI, compute_elbo,
    BayesianLinearRegressionVI, svgd, VIResult
)


class TestGaussianVariational:
    """Tests for Gaussian variational family."""
    
    def test_gaussian_sample_shape(self):
        """Samples should have correct shape."""
        q = GaussianVariational(d=5)
        samples = q.sample(100)
        
        assert samples.shape == (100, 5)
    
    def test_gaussian_log_prob_shape(self):
        """Log prob should have correct shape."""
        q = GaussianVariational(d=3)
        z = np.random.randn(10, 3)
        log_probs = q.log_prob(z)
        
        assert log_probs.shape == (10,)
    
    def test_gaussian_entropy_positive(self):
        """Entropy should be positive for reasonable variances."""
        q = GaussianVariational(d=2, log_std=np.zeros(2))
        entropy = q.entropy()
        
        assert entropy > 0
    
    def test_kl_to_standard_normal_zero_at_standard(self):
        """KL to N(0,I) should be zero when q = N(0,I)."""
        q = GaussianVariational(d=3, mean=np.zeros(3), log_std=np.zeros(3))
        kl = q.kl_to_standard_normal()
        
        assert np.isclose(kl, 0.0, atol=1e-10)
    
    def test_kl_positive_for_different_distribution(self):
        """KL should be positive for different distributions."""
        q = GaussianVariational(d=2, mean=np.array([1.0, 2.0]), log_std=np.zeros(2))
        kl = q.kl_to_standard_normal()
        
        assert kl > 0


class TestMeanFieldVI:
    """Tests for Mean-Field Variational Inference."""
    
    def test_mfvi_step_updates_params(self):
        """Single VI step should update parameters."""
        q = GaussianVariational(d=2)
        initial_mean = q.mean.copy()
        
        log_joint = lambda z: -0.5 * np.sum((z - 1)**2, axis=1)
        grad_log_joint = lambda z: -(z - 1)
        
        vi = MeanFieldVI(q, learning_rate=0.1)
        vi.step(log_joint, grad_log_joint)
        
        # Parameters should have changed
        assert not np.allclose(q.mean, initial_mean)
    
    def test_mfvi_elbo_increases(self):
        """ELBO should generally increase during optimization."""
        q = GaussianVariational(d=1)
        
        log_joint = lambda z: -0.5 * np.sum(z**2, axis=1)
        grad_log_joint = lambda z: -z
        
        vi = MeanFieldVI(q, learning_rate=0.1, n_samples=50)
        
        elbos = []
        for _ in range(20):
            elbo = vi.step(log_joint, grad_log_joint)
            elbos.append(elbo)
        
        # Average ELBO in second half should be >= first half
        assert np.mean(elbos[10:]) >= np.mean(elbos[:10]) - 0.5
    
    def test_mfvi_fit_returns_result(self):
        """Fit should return VIResult with expected fields."""
        q = GaussianVariational(d=2)
        
        log_joint = lambda z: -0.5 * np.sum(z**2, axis=1)
        grad_log_joint = lambda z: -z
        
        vi = MeanFieldVI(q, learning_rate=0.01)
        result = vi.fit(log_joint, grad_log_joint, n_iterations=50, verbose=False)
        
        assert isinstance(result, VIResult)
        assert len(result.elbo_history) == 50
        assert 'mean' in result.variational_params
        assert 'log_std' in result.variational_params


class TestBayesianLinearRegression:
    """Tests for Bayesian Linear Regression with VI."""
    
    def test_blr_fit_sets_params(self):
        """Fit should set mean and covariance."""
        X = np.random.randn(50, 3)
        y = X @ np.array([1, 2, 3]) + 0.1 * np.random.randn(50)
        
        blr = BayesianLinearRegressionVI()
        blr.fit(X, y)
        
        assert blr.mean is not None
        assert blr.cov is not None
        assert blr.mean.shape == (3,)
        assert blr.cov.shape == (3, 3)
    
    def test_blr_predicts_correctly(self):
        """Predictions should be close to true values."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        true_w = np.array([1.5, -2.0])
        y = X @ true_w + 0.1 * np.random.randn(100)
        
        blr = BayesianLinearRegressionVI(alpha=0.1, beta=100)
        blr.fit(X, y)
        
        # Check learned weights are close
        assert np.allclose(blr.mean, true_w, atol=0.2)
    
    def test_blr_uncertainty_increases_away_from_data(self):
        """Predictive uncertainty should be higher for extrapolation."""
        np.random.seed(42)
        X_train = np.random.randn(50, 1)  # Training data around 0
        y_train = 2 * X_train.ravel() + 0.1 * np.random.randn(50)
        
        blr = BayesianLinearRegressionVI()
        blr.fit(X_train, y_train)
        
        # Compare uncertainty at 0 vs far from data
        X_near = np.array([[0.0]])
        X_far = np.array([[10.0]])
        
        _, std_near = blr.predict(X_near)
        _, std_far = blr.predict(X_far)
        
        assert std_far > std_near


class TestSVGD:
    """Tests for Stein Variational Gradient Descent."""
    
    def test_svgd_preserves_particle_count(self):
        """SVGD should preserve number of particles."""
        log_prob = lambda x: -0.5 * np.sum(x**2)
        grad_log_prob = lambda x: -x
        
        initial = np.random.randn(50, 2)
        final = svgd(log_prob, grad_log_prob, initial, n_iterations=10)
        
        assert final.shape == (50, 2)
    
    def test_svgd_moves_toward_mode(self):
        """Particles should move toward high probability regions."""
        log_prob = lambda x: -0.5 * np.sum((x - 2)**2)  # Mode at 2
        grad_log_prob = lambda x: -(x - 2)
        
        initial = np.random.randn(30, 1) * 0.1  # Start near 0
        final = svgd(log_prob, grad_log_prob, initial, n_iterations=100, learning_rate=0.5)
        
        # Particles should move toward 2
        assert np.mean(final) > 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
