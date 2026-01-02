
import sys
import os
import traceback

sys.path.insert(0, '.')

def debug_module(name, import_func):
    print(f"\nTesting {name}...")
    try:
        import_func()
        print(f"[OK] {name} SUCCESS")
    except Exception:
        print(f"[FAIL] {name} FAILED")
        traceback.print_exc()

def test_mcmc():
    from src.core.mcmc import metropolis_hastings
    print("  Imported metropolis_hastings")
    samples = metropolis_hastings(
        log_prob=lambda x: -0.5 * x**2,
        initial_state=0.0,
        n_samples=10
    )
    print("  Execution complete")

def test_vi():
    from src.core.variational_inference import MeanFieldVI
    print("  Imported MeanFieldVI")
    vi = MeanFieldVI(dim=2)
    print("  Execution complete")

def test_ppl():
    from src.core.ppl_integration import BayesianRegression
    print("  Imported BayesianRegression")
    model = BayesianRegression('custom')
    print("  Execution complete")

if __name__ == "__main__":
    debug_module("MCMC", test_mcmc)
    debug_module("Variational Inference", test_vi)
    debug_module("PPL Integration", test_ppl)
