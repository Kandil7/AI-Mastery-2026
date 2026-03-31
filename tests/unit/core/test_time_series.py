"""
Unit Tests for Time Series Module
=================================

Tests for Extended Kalman Filter, Unscented Kalman Filter,
Particle Filter, and RTS Smoother.

Run with: pytest tests/test_time_series.py -v
"""

import pytest
import numpy as np

from src.core.time_series import (
    GaussianState,
    FilterResult,
    ExtendedKalmanFilter,
    UnscentedKalmanFilter,
    ParticleFilter,
    rts_smoother,
    create_linear_system,
    simulate_system,
    compare_filters
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def linear_system():
    """Create a simple 2D linear system."""
    A = np.array([[0.9, 0.1], [0.0, 0.95]])
    C = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1
    return A, C, Q, R


@pytest.fixture
def nonlinear_system():
    """Create a simple nonlinear system."""
    def f(x):
        return np.array([np.sin(x[0]) + 0.9 * x[1], 0.95 * x[1]])
    
    def h(x):
        return np.array([x[0]])
    
    def F_jac(x):
        return np.array([[np.cos(x[0]), 0.9], [0.0, 0.95]])
    
    def H_jac(x):
        return np.array([[1.0, 0.0]])
    
    Q = np.eye(2) * 0.01
    R = np.eye(1) * 0.1
    
    return f, h, F_jac, H_jac, Q, R


# =============================================================================
# GAUSSIAN STATE TESTS
# =============================================================================

class TestGaussianState:
    """Tests for GaussianState dataclass."""
    
    def test_initialization(self):
        """GaussianState should initialize correctly."""
        mean = np.array([1.0, 2.0])
        cov = np.eye(2) * 0.5
        state = GaussianState(mean, cov)
        
        assert state.dim == 2
        np.testing.assert_array_equal(state.mean, mean)
        np.testing.assert_array_equal(state.cov, cov)
    
    def test_sample(self):
        """Sampling should produce correct shape."""
        state = GaussianState(np.zeros(3), np.eye(3))
        samples = state.sample(100)
        
        assert samples.shape == (100, 3)
    
    def test_sample_mean_approximation(self):
        """Sample mean should approximate true mean."""
        mean = np.array([5.0, -3.0])
        state = GaussianState(mean, np.eye(2) * 0.01)
        samples = state.sample(1000)
        
        np.testing.assert_allclose(np.mean(samples, axis=0), mean, atol=0.1)


# =============================================================================
# EKF TESTS
# =============================================================================

class TestExtendedKalmanFilter:
    """Tests for Extended Kalman Filter."""
    
    def test_linear_system_predict(self, linear_system):
        """EKF predict should work on linear system."""
        A, C, Q, R = linear_system
        
        ekf = ExtendedKalmanFilter(
            f=lambda x: A @ x,
            h=lambda x: C @ x,
            F_jacobian=lambda x: A,
            H_jacobian=lambda x: C,
            Q=Q, R=R
        )
        
        state = GaussianState(np.array([1.0, 0.5]), np.eye(2) * 0.1)
        predicted = ekf.predict(state)
        
        assert predicted.mean.shape == (2,)
        assert predicted.cov.shape == (2, 2)
    
    def test_linear_system_update(self, linear_system):
        """EKF update should incorporate observation."""
        A, C, Q, R = linear_system
        
        ekf = ExtendedKalmanFilter(
            f=lambda x: A @ x,
            h=lambda x: C @ x,
            F_jacobian=lambda x: A,
            H_jacobian=lambda x: C,
            Q=Q, R=R
        )
        
        state = GaussianState(np.array([1.0, 0.5]), np.eye(2) * 0.5)
        y = np.array([1.2])
        
        updated, log_lik = ekf.update(state, y)
        
        # Updated covariance should be smaller (more certain)
        assert np.trace(updated.cov) < np.trace(state.cov)
        assert np.isfinite(log_lik)
    
    def test_filter_full_sequence(self, linear_system):
        """EKF should filter a full sequence."""
        A, C, Q, R = linear_system
        
        ekf = ExtendedKalmanFilter(
            f=lambda x: A @ x,
            h=lambda x: C @ x,
            F_jacobian=lambda x: A,
            H_jacobian=lambda x: C,
            Q=Q, R=R
        )
        
        # Generate synthetic observations
        T = 20
        observations = np.random.randn(T, 1)
        initial_state = GaussianState(np.zeros(2), np.eye(2))
        
        result = ekf.filter(observations, initial_state)
        
        assert result.means.shape == (T, 2)
        assert result.covs.shape == (T, 2, 2)
        assert np.isfinite(result.log_likelihood)
    
    def test_nonlinear_system(self, nonlinear_system):
        """EKF should handle nonlinear systems."""
        f, h, F_jac, H_jac, Q, R = nonlinear_system
        
        ekf = ExtendedKalmanFilter(f, h, F_jac, H_jac, Q, R)
        
        state = GaussianState(np.array([0.5, 0.3]), np.eye(2) * 0.1)
        predicted = ekf.predict(state)
        
        assert predicted.mean.shape == (2,)
        assert np.all(np.isfinite(predicted.mean))


# =============================================================================
# UKF TESTS
# =============================================================================

class TestUnscentedKalmanFilter:
    """Tests for Unscented Kalman Filter."""
    
    def test_linear_system_matches_ekf(self, linear_system):
        """UKF should match EKF for linear systems."""
        A, C, Q, R = linear_system
        
        ekf, ukf = create_linear_system(A, C, Q, R)
        
        initial_state = GaussianState(np.array([1.0, 0.5]), np.eye(2) * 0.1)
        
        ekf_pred = ekf.predict(initial_state)
        ukf_pred = ukf.predict(initial_state)
        
        np.testing.assert_allclose(ekf_pred.mean, ukf_pred.mean, atol=1e-6)
        np.testing.assert_allclose(ekf_pred.cov, ukf_pred.cov, atol=1e-6)
    
    def test_sigma_points_count(self, linear_system):
        """UKF should use 2n+1 sigma points."""
        A, C, Q, R = linear_system
        _, ukf = create_linear_system(A, C, Q, R)
        
        state = GaussianState(np.ones(2), np.eye(2))
        sigma_points, Wm, Wc = ukf._compute_sigma_points(state)
        
        # 2*2 + 1 = 5 sigma points for 2D state
        assert sigma_points.shape == (5, 2)
        assert len(Wm) == 5
        assert len(Wc) == 5
    
    def test_weights_sum_to_one(self, linear_system):
        """UKF weights should sum to approximately 1."""
        A, C, Q, R = linear_system
        _, ukf = create_linear_system(A, C, Q, R)
        
        state = GaussianState(np.ones(2), np.eye(2))
        _, Wm, _ = ukf._compute_sigma_points(state)
        
        np.testing.assert_allclose(np.sum(Wm), 1.0, atol=1e-10)
    
    def test_filter_full_sequence(self, nonlinear_system):
        """UKF should filter nonlinear sequence."""
        f, h, _, _, Q, R = nonlinear_system
        
        ukf = UnscentedKalmanFilter(f, h, Q, R)
        
        T = 20
        observations = np.random.randn(T, 1)
        initial_state = GaussianState(np.zeros(2), np.eye(2))
        
        result = ukf.filter(observations, initial_state)
        
        assert result.means.shape == (T, 2)
        assert np.all(np.isfinite(result.means))


# =============================================================================
# PARTICLE FILTER TESTS
# =============================================================================

class TestParticleFilter:
    """Tests for Particle Filter."""
    
    def test_initialization(self):
        """Particle filter should initialize correctly."""
        pf = ParticleFilter(
            f=lambda x: x,
            h=lambda x: x,
            process_noise_sampler=lambda n: np.random.randn(n, 2) * 0.1,
            observation_log_likelihood=lambda y, x: -0.5 * np.sum((y - x)**2),
            n_particles=100
        )
        
        assert pf.n_particles == 100
    
    def test_systematic_resampling(self):
        """Systematic resampling should produce valid indices."""
        pf = ParticleFilter(
            f=lambda x: x,
            h=lambda x: x,
            process_noise_sampler=lambda n: np.zeros((n, 2)),
            observation_log_likelihood=lambda y, x: 0.0,
            n_particles=100
        )
        
        weights = np.random.random(100)
        weights /= np.sum(weights)
        
        indices = pf._systematic_resample(weights)
        
        assert len(indices) == 100
        assert np.all(indices >= 0)
        assert np.all(indices < 100)
    
    def test_effective_sample_size(self):
        """ESS should be correct for uniform weights."""
        pf = ParticleFilter(
            f=lambda x: x, h=lambda x: x,
            process_noise_sampler=lambda n: np.zeros((n, 1)),
            observation_log_likelihood=lambda y, x: 0.0,
            n_particles=100
        )
        
        # Uniform weights should give ESS = N
        uniform_weights = np.ones(100) / 100
        ess = pf._effective_sample_size(uniform_weights)
        
        np.testing.assert_allclose(ess, 100.0, atol=1e-10)
    
    def test_filter_tracks_state(self):
        """Particle filter should track a simple state."""
        n_particles = 500
        
        # Simple random walk
        pf = ParticleFilter(
            f=lambda x: x,
            h=lambda x: x,
            process_noise_sampler=lambda n: np.random.randn(n, 1) * 0.1,
            observation_log_likelihood=lambda y, x: -0.5 * np.sum((y - x)**2) / 0.1,
            n_particles=n_particles
        )
        
        # Generate observations near 0
        T = 10
        observations = np.random.randn(T, 1) * 0.1
        
        initial_particles = np.random.randn(n_particles, 1)
        means, _, ess_history = pf.filter(observations, initial_particles)
        
        assert means.shape == (T, 1)
        assert len(ess_history) == T


# =============================================================================
# RTS SMOOTHER TESTS
# =============================================================================

class TestRTSSmoother:
    """Tests for RTS Smoother."""
    
    def test_smoother_reduces_variance(self, linear_system):
        """Smoothed estimates should have lower variance than filtered."""
        A, C, Q, R = linear_system
        
        ekf = ExtendedKalmanFilter(
            f=lambda x: A @ x,
            h=lambda x: C @ x,
            F_jacobian=lambda x: A,
            H_jacobian=lambda x: C,
            Q=Q, R=R
        )
        
        # Generate data
        T = 30
        true_states, observations = simulate_system(
            f=lambda x: A @ x,
            h=lambda x: C @ x,
            x0=np.zeros(2),
            Q=Q, R=R, T=T, seed=42
        )
        
        # Filter
        initial_state = GaussianState(np.zeros(2), np.eye(2))
        filter_result = ekf.filter(observations, initial_state)
        
        # Smooth
        smoothed_means, smoothed_covs = rts_smoother(
            filter_result.means, filter_result.covs,
            f=lambda x: A @ x,
            F_jacobian=lambda x: A,
            Q=Q
        )
        
        # Smoothed covariances should be smaller on average
        filter_trace = np.mean([np.trace(c) for c in filter_result.covs])
        smooth_trace = np.mean([np.trace(c) for c in smoothed_covs])
        
        assert smooth_trace <= filter_trace


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_create_linear_system(self, linear_system):
        """Should create EKF and UKF for linear system."""
        A, C, Q, R = linear_system
        ekf, ukf = create_linear_system(A, C, Q, R)
        
        assert isinstance(ekf, ExtendedKalmanFilter)
        assert isinstance(ukf, UnscentedKalmanFilter)
    
    def test_simulate_system(self, linear_system):
        """Should simulate state space system."""
        A, C, Q, R = linear_system
        
        states, obs = simulate_system(
            f=lambda x: A @ x,
            h=lambda x: C @ x,
            x0=np.zeros(2),
            Q=Q, R=R, T=20, seed=42
        )
        
        assert states.shape == (20, 2)
        assert obs.shape == (20, 1)
    
    def test_compare_filters(self, linear_system):
        """Should compare EKF and UKF results."""
        A, C, Q, R = linear_system
        
        true_states, obs = simulate_system(
            f=lambda x: A @ x,
            h=lambda x: C @ x,
            x0=np.zeros(2),
            Q=Q, R=R, T=20, seed=42
        )
        
        ekf, ukf = create_linear_system(A, C, Q, R)
        initial = GaussianState(np.zeros(2), np.eye(2))
        
        ekf_result = ekf.filter(obs, initial)
        ukf_result = ukf.filter(obs, initial)
        
        comparison = compare_filters(true_states, ekf_result, ukf_result)
        
        assert 'ekf_rmse' in comparison
        assert 'ukf_rmse' in comparison
        assert comparison['ekf_rmse'] > 0
        assert comparison['ukf_rmse'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
