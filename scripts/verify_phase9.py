"""Quick verification of Phase 9 advanced integration extensions."""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("PHASE 9: ADVANCED INTEGRATION EXTENSIONS VERIFICATION")
print("=" * 60)

# Test 1: Hardware Acceleration Module
print("\n[1] Hardware Accelerated Integration...")
from src.core.hardware_accelerated_integration import (
    monte_carlo_cpu, 
    HardwareAcceleratedIntegrator,
    NUMBA_AVAILABLE,
    TORCH_AVAILABLE
)

result, error = monte_carlo_cpu(lambda x: x**2, a=0, b=1, n_samples=10000)
expected = 1/3
rel_error = abs(result - expected) / expected
print(f"    monte_carlo_cpu: result={result:.6f}, expected={expected:.4f}, error={rel_error:.2%}")
print(f"    Numba available: {NUMBA_AVAILABLE}")
print(f"    PyTorch available: {TORCH_AVAILABLE}")
assert rel_error < 0.05, "Error too large"
print("    ✓ Hardware acceleration module OK")

# Test 2: Adaptive Integration Module
print("\n[2] Adaptive Integration...")
from src.core.adaptive_integration import (
    AdaptiveIntegrator,
    smooth_function,
    multimodal_function
)
import scipy.integrate as spi

integrator = AdaptiveIntegrator()
result = integrator.integrate(smooth_function, a=-1, b=1)
true_val, _ = spi.quad(smooth_function, -1, 1)
rel_error = abs(result.estimate - true_val) / (abs(true_val) + 1e-8)

print(f"    Function: smooth_function")
print(f"    Method selected: {result.method}")
print(f"    Result: {result.estimate:.6f}, True: {true_val:.6f}")
print(f"    Error: {rel_error:.2%}")
print(f"    Features: smoothness={result.features.smoothness:.2f}, modes={result.features.num_modes}")
assert rel_error < 0.10, "Error too large"
print("    ✓ Adaptive integration module OK")

# Test 3: PPL Integration Module
print("\n[3] PPL Integration...")
from src.core.ppl_integration import (
    NumpyMCMCRegression,
    generate_regression_data,
    PYMC_AVAILABLE,
    TFP_AVAILABLE
)
import numpy as np

X, y, true_params = generate_regression_data(n=50, seed=42)
model = NumpyMCMCRegression()
ppl_result = model.fit(X, y, n_samples=500, n_warmup=200)

print(f"    True slope: {true_params['slope']:.3f}")
print(f"    Estimated slope: {ppl_result.slope_mean:.3f} ± {ppl_result.slope_std:.3f}")
print(f"    Time: {ppl_result.time_seconds:.2f}s")
print(f"    PyMC available: {PYMC_AVAILABLE}")
print(f"    TFP available: {TFP_AVAILABLE}")

slope_error = abs(ppl_result.slope_mean - true_params['slope']) / true_params['slope']
assert slope_error < 0.3, f"Slope error too large: {slope_error:.2%}"
print("    ✓ PPL integration module OK")

# Test 4: Prediction with uncertainty
print("\n[4] Prediction with Uncertainty...")
X_new = np.array([0, 1, 2])
y_pred, y_std = model.predict(X_new, return_uncertainty=True)
print(f"    Predictions: {y_pred}")
print(f"    Uncertainties: {y_std}")
assert len(y_pred) == 3
assert all(y_std > 0)
print("    ✓ Prediction with uncertainty OK")

# Summary
print("\n" + "=" * 60)
print("ALL PHASE 9 MODULES VERIFIED ✓")
print("=" * 60)
print("\nModules created:")
print("  - src/core/hardware_accelerated_integration.py")
print("  - src/core/adaptive_integration.py")
print("  - src/core/ppl_integration.py")
print("  - tests/test_hardware_ppl_adaptive.py")
print("\nDocumentation updated:")
print("  - docs/USER_GUIDE.md (sections 5.9-5.11)")
print("  - docs/interview_prep.md (Hardware/PPL/Adaptive)")
