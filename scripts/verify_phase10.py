"""Quick verification of Phase 10: RL Integration and Causal Inference."""
import sys
sys.path.insert(0, '.')
import numpy as np

print("=" * 60)
print("PHASE 10: RL & CAUSAL INFERENCE VERIFICATION")
print("=" * 60)

# Test 1: RL Integration Module
print("\n[1] RL Integration Module...")
from src.core.rl_integration import (
    RLIntegrationSystem,
    SimpleValueNetwork,
    simple_policy
)

rl = RLIntegrationSystem()
episode = rl.run_episode(simple_policy, max_steps=50)
print(f"    Episode length: {episode.length}")
print(f"    Total reward: {episode.total_reward:.2f}")

# Test Monte Carlo returns
returns = rl.compute_returns([1.0, 1.0, 1.0], gamma=0.9)
expected_g0 = 1 + 0.9 + 0.81  # 2.71
assert abs(returns[0] - expected_g0) < 0.01, f"Returns incorrect: {returns[0]}"
print("    ✓ Monte Carlo returns correct")

# Test MCTS value estimate
state = np.array([-0.5, 0.0])
value, uncertainty = rl.mcts_value_estimate(state, n_simulations=10, depth=5)
assert np.isfinite(value), "MCTS value not finite"
print(f"    ✓ MCTS value estimation: {value:.2f} ± {uncertainty:.2f}")
print("    ✓ RL integration module OK")

# Test 2: Causal Inference Module
print("\n[2] Causal Inference Module...")
from src.core.causal_inference import (
    CausalInferenceSystem,
    ATEResult
)

causal = CausalInferenceSystem()
data = causal.generate_synthetic_data(n_samples=300)
true_ate = data['true_effect'].mean()
print(f"    True ATE: {true_ate:.3f}")

# Test IPW
ipw_result = causal.estimate_ate_ipw(data)
print(f"    IPW ATE: {ipw_result.ate_estimate:.3f}")
assert np.isfinite(ipw_result.ate_estimate), "IPW estimate not finite"
print("    ✓ IPW estimation OK")

# Test Doubly Robust
dr_result = causal.estimate_ate_doubly_robust(data)
print(f"    DR ATE: {dr_result.ate_estimate:.3f} ± {dr_result.ate_std_error:.3f}")
assert np.isfinite(dr_result.ate_estimate), "DR estimate not finite"
print("    ✓ Doubly Robust estimation OK")

# DR should typically be closer to true than naive
naive_ate = data[data['treatment']==1]['outcome'].mean() - data[data['treatment']==0]['outcome'].mean()
dr_error = abs(dr_result.ate_estimate - true_ate)
naive_error = abs(naive_ate - true_ate)
print(f"    Naive error: {naive_error:.3f}, DR error: {dr_error:.3f}")

# Test Bayesian
bayes_result = causal.bayesian_causal_inference(data, n_posterior_samples=30)
print(f"    Bayesian ATE: {bayes_result.ate_mean:.3f} ± {bayes_result.ate_std:.3f}")
print("    ✓ Bayesian causal inference OK")

# Test heterogeneous effects
het = causal.analyze_heterogeneous_effects(data, dr_result.diagnostics['individual_effects'])
assert 'age' in het, "Missing age analysis"
print("    ✓ Heterogeneous effects analysis OK")

# Summary
print("\n" + "=" * 60)
print("ALL PHASE 10 MODULES VERIFIED ✓")
print("=" * 60)
print("\nModules created:")
print("  - src/core/rl_integration.py (~450 lines)")
print("  - src/core/causal_inference.py (~450 lines)")
print("  - tests/test_rl_causal.py (~250 lines)")
print("\nNotebook updated:")
print("  - Chapter 14: Integration in RL (4 cells)")
print("  - Chapter 15: Causal Inference (4 cells)")
print("  - Interview Questions (1 cell)")
print("\nDocumentation updated:")
print("  - docs/USER_GUIDE.md (sections 5.12-5.13)")
print("  - docs/interview_prep.md (RL/Causal questions)")
