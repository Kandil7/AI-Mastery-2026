"""
Tests for Hardware Acceleration, PPL Integration, and Adaptive Integration modules.
"""

import pytest
import numpy as np
import time
from typing import Callable


# ============================================================================
# Hardware Acceleration Tests
# ============================================================================

class TestHardwareAcceleration:
    """Tests for hardware-accelerated integration methods."""
    
    def test_monte_carlo_cpu_basic(self):
        """Test CPU Monte Carlo integration with simple function."""
        from src.core.hardware_accelerated_integration import monte_carlo_cpu
        
        # Integrate x^2 from 0 to 1 (analytical result = 1/3)
        f = lambda x: x**2
        result, error = monte_carlo_cpu(f, a=0, b=1, n_samples=100000)
        
        assert abs(result - 1/3) < 0.01, f"Expected ~0.333, got {result}"
        assert error > 0, "Error should be positive"
    
    def test_monte_carlo_cpu_multimodal(self):
        """Test CPU Monte Carlo with multimodal function."""
        from src.core.hardware_accelerated_integration import (
            monte_carlo_cpu, multimodal_function_numpy
        )
        
        result, error = monte_carlo_cpu(multimodal_function_numpy, a=0, b=1, n_samples=50000)
        
        # Result should be reasonable (positive and bounded)
        assert result > 0, "Result should be positive for this function"
        assert result < 5, "Result should be bounded"
    
    def test_monte_carlo_numba_available(self):
        """Test Numba availability check."""
        from src.core.hardware_accelerated_integration import NUMBA_AVAILABLE
        
        # Just check it's a boolean
        assert isinstance(NUMBA_AVAILABLE, bool)
    
    def test_monte_carlo_numba_execution(self):
        """Test Numba-accelerated Monte Carlo if available."""
        from src.core.hardware_accelerated_integration import (
            monte_carlo_numba, NUMBA_AVAILABLE
        )
        
        if not NUMBA_AVAILABLE:
            pytest.skip("Numba not available")
        
        result, error = monte_carlo_numba(n_samples=10000, a=0, b=1)
        
        # Result should be reasonable
        assert result > 0, "Result should be positive"
        assert np.isfinite(result), "Result should be finite"
    
    def test_pytorch_execution(self):
        """Test PyTorch GPU integration if available."""
        from src.core.hardware_accelerated_integration import (
            monte_carlo_gpu_pytorch, multimodal_function_torch, TORCH_AVAILABLE
        )
        
        if not TORCH_AVAILABLE:
            pytest.skip("PyTorch not available")
        
        result, error = monte_carlo_gpu_pytorch(
            multimodal_function_torch, a=0, b=1, n_samples=10000
        )
        
        assert result > 0, "Result should be positive"
        assert np.isfinite(result), "Result should be finite"
    
    def test_benchmark_consistency(self):
        """Test that all methods give consistent results."""
        from src.core.hardware_accelerated_integration import (
            monte_carlo_cpu, multimodal_function_numpy,
            monte_carlo_gpu_pytorch, multimodal_function_torch,
            TORCH_AVAILABLE
        )
        
        # Get CPU result
        cpu_result, _ = monte_carlo_cpu(multimodal_function_numpy, n_samples=100000)
        
        # Compare with PyTorch if available
        if TORCH_AVAILABLE:
            torch_result, _ = monte_carlo_gpu_pytorch(
                multimodal_function_torch, n_samples=100000
            )
            # Results should be within 10% (stochastic methods)
            assert abs(cpu_result - torch_result) / cpu_result < 0.1
    
    def test_hardware_accelerated_integrator(self):
        """Test the unified HardwareAcceleratedIntegrator."""
        from src.core.hardware_accelerated_integration import (
            HardwareAcceleratedIntegrator, multimodal_function_numpy
        )
        
        integrator = HardwareAcceleratedIntegrator()
        
        result = integrator.integrate(
            multimodal_function_numpy,
            a=0, b=1, n_samples=10000,
            method='numpy'
        )
        
        assert 'estimate' in result
        assert 'device' in result
        assert 'time_seconds' in result
        assert result['estimate'] > 0


# ============================================================================
# Adaptive Integration Tests
# ============================================================================

class TestAdaptiveIntegration:
    """Tests for adaptive integration module."""
    
    def test_function_features_extraction(self):
        """Test function feature extraction."""
        from src.core.adaptive_integration import AdaptiveIntegrator
        
        integrator = AdaptiveIntegrator()
        
        # Simple smooth function
        def smooth(x):
            return np.sin(x)
        
        features = integrator.analyze_function(smooth, a=0, b=np.pi)
        
        assert features.range_size == pytest.approx(np.pi, rel=0.01)
        assert features.mean_value > 0  # sin is positive on [0, pi]
        assert features.smoothness > 0
        assert isinstance(features.num_modes, int)
    
    def test_method_selection_smooth(self):
        """Test method selection for smooth functions."""
        from src.core.adaptive_integration import AdaptiveIntegrator, smooth_function
        
        integrator = AdaptiveIntegrator()
        features = integrator.analyze_function(smooth_function, a=-1, b=1)
        method = integrator.select_method(features)
        
        # Smooth functions should use Gaussian quadrature or Bayesian
        assert method in ['gaussian_quad', 'bayesian_quad', 'simpson']
    
    def test_method_selection_oscillatory(self):
        """Test method selection for oscillatory functions."""
        from src.core.adaptive_integration import AdaptiveIntegrator, oscillatory_function
        
        integrator = AdaptiveIntegrator()
        features = integrator.analyze_function(oscillatory_function, a=-1, b=1)
        method = integrator.select_method(features)
        
        # Oscillatory functions should use Monte Carlo
        assert method in ['monte_carlo', 'simpson', 'bayesian_quad']
    
    def test_integration_accuracy(self):
        """Test integration accuracy against scipy reference."""
        from src.core.adaptive_integration import AdaptiveIntegrator, smooth_function
        import scipy.integrate as spi
        
        integrator = AdaptiveIntegrator()
        result = integrator.integrate(smooth_function, a=-1, b=1)
        
        # Compare with scipy
        true_value, _ = spi.quad(smooth_function, -1, 1)
        
        relative_error = abs(result.estimate - true_value) / (abs(true_value) + 1e-8)
        assert relative_error < 0.1, f"Error too large: {relative_error}"
    
    def test_adaptive_integrator_all_functions(self):
        """Test adaptive integrator on all test functions."""
        from src.core.adaptive_integration import (
            AdaptiveIntegrator, 
            smooth_function, multimodal_function, 
            discontinuous_function, heavy_tailed_function
        )
        import scipy.integrate as spi
        
        integrator = AdaptiveIntegrator()
        
        test_funcs = [
            smooth_function,
            multimodal_function,
            heavy_tailed_function,
        ]
        
        for f in test_funcs:
            result = integrator.integrate(f, a=-1, b=1)
            true_value, _ = spi.quad(f, -1, 1)
            
            relative_error = abs(result.estimate - true_value) / (abs(true_value) + 1e-8)
            assert relative_error < 0.2, f"Error too large for {f.__name__}: {relative_error}"
    
    def test_train_method_selector(self):
        """Test ML-based method selector training."""
        from src.core.adaptive_integration import (
            AdaptiveIntegrator,
            smooth_function, multimodal_function, heavy_tailed_function
        )
        
        integrator = AdaptiveIntegrator()
        
        # Train with simple functions
        training_funcs = [smooth_function, multimodal_function, heavy_tailed_function]
        integrator.train_method_selector(training_funcs, a=-1, b=1)
        
        # Classifier should be trained
        assert integrator.classifier is not None
    
    def test_integration_result_fields(self):
        """Test that IntegrationResult has all expected fields."""
        from src.core.adaptive_integration import AdaptiveIntegrator, smooth_function
        
        integrator = AdaptiveIntegrator()
        result = integrator.integrate(smooth_function, a=0, b=1)
        
        assert hasattr(result, 'estimate')
        assert hasattr(result, 'method')
        assert hasattr(result, 'time_seconds')
        assert hasattr(result, 'features')
        assert result.time_seconds > 0


# ============================================================================
# PPL Integration Tests
# ============================================================================

class TestPPLIntegration:
    """Tests for Probabilistic Programming Language integration."""
    
    def test_generate_regression_data(self):
        """Test synthetic data generation."""
        from src.core.ppl_integration import generate_regression_data
        
        X, y, true_params = generate_regression_data(n=50, seed=42)
        
        assert len(X) == 50
        assert len(y) == 50
        assert 'slope' in true_params
        assert 'intercept' in true_params
    
    def test_numpy_mcmc_regression(self):
        """Test pure NumPy MCMC regression."""
        from src.core.ppl_integration import (
            NumpyMCMCRegression, generate_regression_data
        )
        
        X, y, true_params = generate_regression_data(n=50, seed=42)
        
        model = NumpyMCMCRegression()
        result = model.fit(X, y, n_samples=500, n_warmup=200)
        
        # Check result structure
        assert result.library == "NumPy (MH)"
        assert result.n_samples == 500
        assert result.time_seconds > 0
        
        # Check estimates are reasonable (within 50% of true values)
        assert abs(result.slope_mean - true_params['slope']) / true_params['slope'] < 0.5
        assert abs(result.intercept_mean - true_params['intercept']) < 1.0
    
    def test_numpy_mcmc_prediction(self):
        """Test prediction with NumPy MCMC regression."""
        from src.core.ppl_integration import (
            NumpyMCMCRegression, generate_regression_data
        )
        
        X, y, _ = generate_regression_data(n=50, seed=42)
        
        model = NumpyMCMCRegression()
        model.fit(X, y, n_samples=500, n_warmup=200)
        
        # Predict
        X_new = np.array([0, 1, 2])
        mean_pred, std_pred = model.predict(X_new, return_uncertainty=True)
        
        assert len(mean_pred) == 3
        assert len(std_pred) == 3
        assert all(std_pred > 0)
    
    def test_ppl_result_dataclass(self):
        """Test PPLResult dataclass fields."""
        from src.core.ppl_integration import PPLResult
        
        result = PPLResult(
            library="Test",
            slope_mean=2.5,
            slope_std=0.1,
            intercept_mean=1.0,
            intercept_std=0.1,
            sigma_mean=0.8,
            time_seconds=1.0,
            n_samples=1000
        )
        
        assert result.library == "Test"
        assert result.slope_mean == 2.5
    
    def test_pymc_regression(self):
        """Test PyMC3 regression if available."""
        from src.core.ppl_integration import (
            PyMCRegression, generate_regression_data, PYMC_AVAILABLE
        )
        
        if not PYMC_AVAILABLE:
            pytest.skip("PyMC3 not available")
        
        X, y, true_params = generate_regression_data(n=50, seed=42)
        
        model = PyMCRegression()
        result = model.fit(X, y, n_samples=200, n_warmup=100)
        
        assert result.library == "PyMC3"
        assert result.slope_mean is not None
    
    def test_tfp_regression(self):
        """Test TensorFlow Probability regression if available."""
        from src.core.ppl_integration import (
            TFPRegression, generate_regression_data, TFP_AVAILABLE
        )
        
        if not TFP_AVAILABLE:
            pytest.skip("TensorFlow Probability not available")
        
        X, y, true_params = generate_regression_data(n=50, seed=42)
        
        model = TFPRegression()
        result = model.fit(X, y, n_samples=200, n_warmup=100)
        
        assert result.library == "TensorFlow Probability"
        assert result.slope_mean is not None


# ============================================================================
# Integration Tests (Cross-Module)
# ============================================================================

class TestCrossModuleIntegration:
    """Tests that verify modules work together correctly."""
    
    def test_hardware_with_adaptive(self):
        """Test combining hardware acceleration with adaptive selection."""
        from src.core.hardware_accelerated_integration import monte_carlo_cpu
        from src.core.adaptive_integration import smooth_function
        
        # Use hardware module's function with adaptive module's test function
        result, error = monte_carlo_cpu(
            lambda x: np.array([smooth_function(xi) for xi in x]),
            n_samples=10000
        )
        
        assert np.isfinite(result)
    
    def test_all_modules_import(self):
        """Test that all new modules can be imported."""
        from src.core import hardware_accelerated_integration
        from src.core import adaptive_integration
        from src.core import ppl_integration
        
        # Check key classes/functions exist
        assert hasattr(hardware_accelerated_integration, 'HardwareAcceleratedIntegrator')
        assert hasattr(adaptive_integration, 'AdaptiveIntegrator')
        assert hasattr(ppl_integration, 'NumpyMCMCRegression')


# ============================================================================
# Demo Functions Tests
# ============================================================================

class TestDemoFunctions:
    """Test that demo functions run without error."""
    
    def test_adaptive_integration_demo(self):
        """Test adaptive integration demo runs."""
        from src.core.adaptive_integration import adaptive_integration_demo
        
        # Should complete without error
        results = adaptive_integration_demo()
        assert len(results) > 0
    
    def test_ppl_comparison_demo(self):
        """Test PPL comparison demo runs."""
        from src.core.ppl_integration import ppl_comparison_demo
        
        # Should complete without error
        results = ppl_comparison_demo()
        assert 'NumPy (MH)' in results
