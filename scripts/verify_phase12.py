"""Quick verification of Phase 12: Differential Privacy and Energy Efficiency."""
import sys
sys.path.insert(0, '.')
import numpy as np

print("=" * 60)
print("PHASE 12: PRIVACY & ENERGY EFFICIENCY VERIFICATION")
print("=" * 60)

# Test 1: Differential Privacy Module
print("\n[1] Differential Privacy Module...")
from src.core.differential_privacy import (
    DifferentiallyPrivateIntegrator,
    DifferentiallyPrivateBayesianQuadrature,
    PrivacyBudget
)

# Test privacy budget
budget = PrivacyBudget(total_epsilon=10.0, used_epsilon=0.0, 
                        total_delta=1e-4, used_delta=0.0)
assert budget.remaining_epsilon == 10.0
budget.consume(5.0)
assert budget.remaining_epsilon == 5.0
print("    ✓ Privacy budget tracking works")

# Test private mean
dp = DifferentiallyPrivateIntegrator(epsilon=1.0, seed=42)
data = np.array([1, 2, 3, 4, 5])
result = dp.private_mean(data, bounds=(0, 10))
assert result.epsilon_used == 1.0
print(f"    ✓ Private mean: {result.value:.3f} (true: 3.0)")

# Test DP Bayesian Quadrature
dp_bq = DifferentiallyPrivateBayesianQuadrature(epsilon=1.0)
nodes = np.linspace(0, 1, 10)
weights = np.ones(10) * 0.1
bq_result = dp_bq.private_bayesian_quadrature(lambda x: x**2, nodes, weights)
assert 'estimate' in bq_result
assert bq_result['uncertainty'] > 0
print(f"    ✓ DP Bayesian Quadrature: {bq_result['estimate']:.3f}")
print("    ✓ Differential privacy module OK")

# Test 2: Energy Efficient Module
print("\n[2] Energy Efficient Module...")
from src.core.energy_efficient import (
    EnergyEfficientIntegrator,
    DEVICE_PROFILES
)

# Test device profiles
assert 'iot' in DEVICE_PROFILES
assert 'mobile' in DEVICE_PROFILES
iot = DEVICE_PROFILES['iot']
assert iot.compute_power_watt == 0.1
print("    ✓ Device profiles loaded")

# Test integrator
integrator = EnergyEfficientIntegrator(device='iot')
result = integrator.trapezoidal(lambda x: x**2, 0, 1, n=50)
assert abs(result.value - 1/3) < 0.01
assert result.energy_cost > 0
print(f"    ✓ Trapezoidal: {result.value:.4f}, energy={result.energy_cost:.2e} Wh")

result = integrator.gauss_legendre(lambda x: x**2, 0, 1, n=5)
assert abs(result.value - 1/3) < 0.0001
print(f"    ✓ Gauss-Legendre: {result.value:.4f} (only {result.n_evaluations} evals)")

# Test auto-select
result_low = integrator.integrate(lambda x: x**2, 0, 1, accuracy='low')
result_high = integrator.integrate(lambda x: x**2, 0, 1, accuracy='high')
assert result_high.n_evaluations >= result_low.n_evaluations
print("    ✓ Auto-select works")

# Test energy budget optimization
result = integrator.optimize_for_energy_budget(lambda x: x**2, 0, 1, energy_budget=1e-5)
assert result.energy_cost <= 1e-5
print(f"    ✓ Budget optimization: {result.method}")
print("    ✓ Energy efficient module OK")

# Summary
print("\n" + "=" * 60)
print("ALL PHASE 12 MODULES VERIFIED ✓")
print("=" * 60)
print("\nModules created:")
print("  - src/core/differential_privacy.py (~400 lines)")
print("  - src/core/energy_efficient.py (~420 lines)")
print("  - tests/test_privacy_energy.py (~270 lines)")
print("\nNotebook updated:")
print("  - Chapter 18: Differential Privacy (3 cells)")
print("  - Chapter 19: Energy Efficiency (3 cells)")
print("  - Interview Questions (1 cell)")
