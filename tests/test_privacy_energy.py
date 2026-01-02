"""
Tests for Differential Privacy and Energy-Efficient Integration modules.
"""

import pytest
import numpy as np


# ============================================================================
# Differential Privacy Tests
# ============================================================================

class TestDifferentialPrivacy:
    """Tests for differential privacy integration."""
    
    def test_privacy_budget_tracking(self):
        """Test privacy budget initialization and tracking."""
        from src.core.differential_privacy import PrivacyBudget
        
        budget = PrivacyBudget(
            total_epsilon=10.0,
            used_epsilon=0.0,
            total_delta=1e-4,
            used_delta=0.0
        )
        
        assert budget.remaining_epsilon == 10.0
        assert not budget.is_exhausted
        
        budget.consume(5.0)
        assert budget.remaining_epsilon == 5.0
        
        budget.consume(5.0)
        assert budget.is_exhausted
    
    def test_laplace_noise(self):
        """Test Laplace noise mechanism."""
        from src.core.differential_privacy import DifferentiallyPrivateIntegrator
        
        dp = DifferentiallyPrivateIntegrator(epsilon=1.0, sensitivity=1.0, seed=42)
        
        # Add noise multiple times
        noises = []
        for _ in range(100):
            _, noise = dp.add_laplace_noise(0.0)
            noises.append(noise)
        
        # Check noise distribution (mean should be ~0, scale = sensitivity/epsilon = 1)
        assert abs(np.mean(noises)) < 0.5  # Should be centered
        assert 0.5 < np.std(noises) < 2.0  # Scale ~ 1/sqrt(2) for Laplace
    
    def test_gaussian_noise(self):
        """Test Gaussian noise mechanism."""
        from src.core.differential_privacy import DifferentiallyPrivateIntegrator
        
        dp = DifferentiallyPrivateIntegrator(epsilon=1.0, delta=1e-5, seed=42)
        
        noises = []
        for _ in range(100):
            _, noise = dp.add_gaussian_noise(0.0)
            noises.append(noise)
        
        assert abs(np.mean(noises)) < 1.0
        assert np.std(noises) > 0
    
    def test_private_mean(self):
        """Test private mean estimation."""
        from src.core.differential_privacy import DifferentiallyPrivateIntegrator
        
        data = np.array([1, 2, 3, 4, 5])
        true_mean = 3.0
        
        dp = DifferentiallyPrivateIntegrator(epsilon=1.0, seed=42)
        result = dp.private_mean(data, bounds=(0, 10))
        
        # Private mean should be somewhat close to true mean
        assert abs(result.value - true_mean) < 3.0
        assert result.epsilon_used == 1.0
        assert result.method == 'laplace'
    
    def test_private_sum(self):
        """Test private sum."""
        from src.core.differential_privacy import DifferentiallyPrivateIntegrator
        
        data = np.array([1, 2, 3, 4, 5])
        true_sum = 15.0
        
        dp = DifferentiallyPrivateIntegrator(epsilon=1.0, seed=42)
        result = dp.private_sum(data, bounds=(0, 10))
        
        assert abs(result.value - true_sum) < 20.0  # Relaxed due to noise
        assert result.method == 'laplace'
    
    def test_private_integral(self):
        """Test private numerical integration."""
        from src.core.differential_privacy import DifferentiallyPrivateIntegrator
        
        dp = DifferentiallyPrivateIntegrator(epsilon=1.0, seed=42)
        
        result = dp.private_integral(lambda x: x**2, 0, 1, n_points=20)
        true_value = 1/3
        
        # With noise, should still be in reasonable range
        assert abs(result.value - true_value) < 1.0
        assert result.method == 'trapezoidal_dp'
    
    def test_dp_bayesian_quadrature(self):
        """Test DP Bayesian Quadrature."""
        from src.core.differential_privacy import DifferentiallyPrivateBayesianQuadrature
        
        dp_bq = DifferentiallyPrivateBayesianQuadrature(epsilon=1.0)
        
        nodes = np.linspace(0, 1, 10)
        weights = np.ones(10) * 0.1
        
        result = dp_bq.private_bayesian_quadrature(
            lambda x: x**2, nodes, weights
        )
        
        assert 'estimate' in result
        assert 'uncertainty' in result
        assert result['uncertainty'] > 0
    
    def test_adaptive_private_integration(self):
        """Test adaptive privacy budget allocation."""
        from src.core.differential_privacy import DifferentiallyPrivateBayesianQuadrature
        
        dp_bq = DifferentiallyPrivateBayesianQuadrature(epsilon=1.0)
        
        result = dp_bq.adaptive_private_integration(
            lambda x: np.sin(x), 0, np.pi,
            epsilon_budget=2.0,
            target_accuracy=0.1
        )
        
        assert 'estimate' in result
        assert result['total_epsilon_used'] == 2.0


# ============================================================================
# Energy Efficient Tests
# ============================================================================

class TestEnergyEfficient:
    """Tests for energy-efficient integration."""
    
    def test_device_profiles(self):
        """Test device profile definitions."""
        from src.core.energy_efficient import DEVICE_PROFILES
        
        assert 'iot' in DEVICE_PROFILES
        assert 'mobile' in DEVICE_PROFILES
        assert 'server' in DEVICE_PROFILES
        
        iot = DEVICE_PROFILES['iot']
        assert iot.compute_power_watt > 0
        assert iot.max_operations > 0
    
    def test_device_energy_estimation(self):
        """Test energy estimation for devices."""
        from src.core.energy_efficient import DEVICE_PROFILES
        
        iot = DEVICE_PROFILES['iot']
        energy = iot.estimate_energy(n_ops=1000, mem_accesses=500)
        
        assert energy > 0
        assert energy < 0.01  # Should be small for IoT
    
    def test_trapezoidal_integration(self):
        """Test trapezoidal rule."""
        from src.core.energy_efficient import EnergyEfficientIntegrator
        
        integrator = EnergyEfficientIntegrator(device='mobile')
        result = integrator.trapezoidal(lambda x: x**2, 0, 1, n=100)
        
        true_value = 1/3
        assert abs(result.value - true_value) < 0.01
        assert result.n_evaluations == 101
        assert result.energy_cost > 0
    
    def test_simpson_integration(self):
        """Test Simpson's rule."""
        from src.core.energy_efficient import EnergyEfficientIntegrator
        
        integrator = EnergyEfficientIntegrator(device='mobile')
        result = integrator.simpson(lambda x: x**2, 0, 1, n=100)
        
        true_value = 1/3
        assert abs(result.value - true_value) < 0.0001  # Simpson should be exact for x^2
    
    def test_gauss_legendre(self):
        """Test Gauss-Legendre quadrature."""
        from src.core.energy_efficient import EnergyEfficientIntegrator
        
        integrator = EnergyEfficientIntegrator(device='iot')
        result = integrator.gauss_legendre(lambda x: x**2, 0, 1, n=5)
        
        true_value = 1/3
        assert abs(result.value - true_value) < 0.0001
        assert result.n_evaluations == 5  # Very efficient
    
    def test_sparse_grid(self):
        """Test sparse grid integration."""
        from src.core.energy_efficient import EnergyEfficientIntegrator
        
        integrator = EnergyEfficientIntegrator(device='iot')
        result = integrator.sparse_grid(lambda x: x**2, 0, 1, level=3)
        
        assert result.n_evaluations > 0
        assert result.energy_cost > 0
    
    def test_adaptive_quadrature(self):
        """Test adaptive quadrature."""
        from src.core.energy_efficient import EnergyEfficientIntegrator
        
        integrator = EnergyEfficientIntegrator(device='mobile')
        result = integrator.adaptive_quadrature(
            lambda x: np.sin(x), 0, np.pi, tol=1e-4
        )
        
        true_value = 2.0
        assert abs(result.value - true_value) < 0.01
    
    def test_integrate_auto_select(self):
        """Test automatic method selection."""
        from src.core.energy_efficient import EnergyEfficientIntegrator
        
        integrator = EnergyEfficientIntegrator(device='iot')
        
        result_low = integrator.integrate(lambda x: x**2, 0, 1, accuracy='low')
        result_high = integrator.integrate(lambda x: x**2, 0, 1, accuracy='high')
        
        # High accuracy should use more evaluations
        assert result_high.n_evaluations >= result_low.n_evaluations
    
    def test_optimize_for_budget(self):
        """Test energy budget optimization."""
        from src.core.energy_efficient import EnergyEfficientIntegrator
        
        integrator = EnergyEfficientIntegrator(device='iot')
        result = integrator.optimize_for_energy_budget(
            lambda x: x**2, 0, 1,
            energy_budget=1e-5
        )
        
        assert result.energy_cost <= 1e-5
    
    def test_compare_methods(self):
        """Test method comparison."""
        from src.core.energy_efficient import EnergyEfficientIntegrator
        
        integrator = EnergyEfficientIntegrator(device='mobile')
        results = integrator.compare_methods(
            lambda x: x**2, 0, 1,
            true_value=1/3
        )
        
        assert len(results) > 0
        for name, result in results.items():
            assert result.error_estimate >= 0


# ============================================================================
# Demo Function Tests
# ============================================================================

class TestDemoFunctions:
    """Test that demo functions run without error."""
    
    def test_differential_privacy_demo(self):
        """Test DP demo runs."""
        from src.core.differential_privacy import differential_privacy_demo
        
        results = differential_privacy_demo()
        
        assert len(results) > 0
        assert all('epsilon' in r for r in results)
    
    def test_energy_efficient_demo(self):
        """Test energy efficiency demo runs."""
        from src.core.energy_efficient import energy_efficient_demo
        
        results = energy_efficient_demo()
        
        assert len(results) > 0


# ============================================================================
# Integration Tests
# ============================================================================

class TestModuleImports:
    """Test that all modules can be imported."""
    
    def test_import_differential_privacy(self):
        """Test differential privacy module import."""
        from src.core import differential_privacy
        
        assert hasattr(differential_privacy, 'DifferentiallyPrivateIntegrator')
        assert hasattr(differential_privacy, 'PrivacyBudget')
    
    def test_import_energy_efficient(self):
        """Test energy efficient module import."""
        from src.core import energy_efficient
        
        assert hasattr(energy_efficient, 'EnergyEfficientIntegrator')
        assert hasattr(energy_efficient, 'DEVICE_PROFILES')
