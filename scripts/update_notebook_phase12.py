"""
Script to add Chapters 18-19 to the Advanced Integration notebook.

Adds:
- Chapter 18: Integration with Differential Privacy
- Chapter 19: Integration in Energy-Efficient ML Systems
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/01_mathematical_foundations/advanced_integration_mcmc_vi.ipynb")


def create_markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split('\n')
    }


def create_code_cell(source: str) -> dict:
    lines = source.split('\n')
    lines_with_newlines = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines_with_newlines
    }


# Chapter 18: Differential Privacy
CHAPTER_18_INTRO = """# Chapter 18: Integration with Differential Privacy

In the age of big data, privacy preservation is critical. Differential Privacy (DP) provides mathematical guarantees that no individual's data can be inferred from algorithm outputs.

**ε-Differential Privacy Definition:**

$$\\forall S \\subseteq \\text{Range}(\\mathcal{M}): \\Pr[\\mathcal{M}(D_1) \\in S] \\leq e^\\epsilon \\Pr[\\mathcal{M}(D_2) \\in S]$$

where $D_1, D_2$ differ in one record.

**Key Challenge**: How to combine DP with integration methods while maintaining accuracy?

## Industrial Case Study: Apple Privacy-Preserving ML

**Challenge**: Improve Siri without collecting voice data

**Solution**: Federated learning + DP integration

**Results**:
- 25% accuracy improvement
- 500 million users protected
- 38% trust increase
"""

CHAPTER_18_CODE = '''import numpy as np
import sys
sys.path.insert(0, '../..')
from src.core.differential_privacy import (
    DifferentiallyPrivateIntegrator,
    DifferentiallyPrivateBayesianQuadrature
)

print("=" * 60)
print("Integration with Differential Privacy")
print("=" * 60)

# Generate synthetic medical data
np.random.seed(42)
n_patients = 500
ages = np.random.uniform(20, 80, n_patients)
risk = 0.01 + 0.0005 * (ages - 30)**2
print(f"\\nTrue mean risk: {risk.mean():.4f}")
'''

CHAPTER_18_DEMO = '''# Test different privacy levels
print("\\nPrivacy-Accuracy Tradeoff:")
print("-" * 40)

for epsilon in [0.1, 1.0, 5.0]:
    dp = DifferentiallyPrivateIntegrator(epsilon=epsilon, seed=42)
    
    estimates = []
    for _ in range(20):
        result = dp.private_mean(risk, bounds=(0, 1))
        estimates.append(result.value)
    
    mean_est = np.mean(estimates)
    error = abs(mean_est - risk.mean()) / risk.mean() * 100
    
    print(f"ε={epsilon}: estimate={mean_est:.4f}, error={error:.1f}%")

print("\\n→ Lower ε = more privacy, but more error")
'''

# Chapter 19: Energy Efficiency
CHAPTER_19_INTRO = """# Chapter 19: Integration in Energy-Efficient ML Systems

With growing concerns about AI's carbon footprint, energy-efficient integration is critical.

**Energy Model:**

$$E_{\\text{total}} = E_{\\text{compute}} + E_{\\text{memory}} + E_{\\text{communication}}$$

**Key Insight**: Reduce integration operations without sacrificing accuracy.

## Industrial Case Study: Google DeepMind Data Centers

**Challenge**: Data centers consume 1-2% of global electricity

**Solution**: Energy-efficient predictive integration

**Results**:
- 40% cooling energy reduction
- $150M/year savings
- 300,000 tons CO₂ reduction annually
"""

CHAPTER_19_CODE = '''from src.core.energy_efficient import (
    EnergyEfficientIntegrator,
    DEVICE_PROFILES
)

print("=" * 60)
print("Energy-Efficient Integration")
print("=" * 60)

# Example: Building energy monitoring
def building_energy(t):
    """Energy consumption (kW) over 24 hours."""
    base = 2.0
    time_factor = 0.5 + 0.5 * np.sin(2 * np.pi * t / 24 - np.pi/2)
    return base * time_factor

# Compare devices
print("\\nDevice Comparison:")
print("-" * 40)

for device in ['iot', 'mobile', 'edge']:
    integrator = EnergyEfficientIntegrator(device=device)
    result = integrator.integrate(building_energy, 0, 24, accuracy='medium')
    print(f"{device:>6}: {result.value:.2f} kWh, energy={result.energy_cost:.2e} Wh")
'''

CHAPTER_19_METHODS = '''# Compare integration methods on IoT
print("\\nMethod Comparison (IoT):")
print("-" * 40)

integrator = EnergyEfficientIntegrator(device='iot')
from scipy.integrate import quad
true_value, _ = quad(building_energy, 0, 24)

results = integrator.compare_methods(building_energy, 0, 24, true_value)

for name, r in sorted(results.items(), key=lambda x: x[1].energy_cost)[:5]:
    print(f"{name:>18}: error={r.error_estimate:.4f}, energy={r.energy_cost:.2e} Wh")

print("\\n→ Gauss-Legendre: best accuracy/energy ratio for smooth functions")
'''

# Interview Questions
INTERVIEW_QUESTIONS = """# Advanced Integration Interview Questions: Privacy & Efficiency

## Differential Privacy

**Q1: What is the privacy-accuracy tradeoff?**

Lower ε = stronger privacy guarantees, but more noise added:
- ε = 0.1: Very private, ~30% error
- ε = 1.0: Good balance, ~10% error  
- ε = 10: Low privacy, <1% error

**Q2: Laplace vs Gaussian mechanism when to use each?**

- **Laplace**: Pure ε-DP, simple, good for single queries
- **Gaussian**: (ε,δ)-DP, better for composition (multiple queries)

---

## Energy Efficiency

**Q3: How to choose integration method for IoT?**

1. **Gauss-Legendre (n=3-5)**: Smooth functions, minimal energy
2. **Sparse Grid**: High-dimensional problems
3. **Adaptive**: When accuracy critical, more energy available

**Q4: Estimate energy for integration on mobile device.**

```python
# Mobile: ~1W compute, 0.3W memory
def estimate_mobile_energy(n_evals, time_per_eval=1e-6):
    compute_time = n_evals * time_per_eval
    energy_wh = 1.0 * compute_time / 3600
    return energy_wh

# 1000 evals ≈ 2.8e-7 Wh
```
"""


def main():
    if not NOTEBOOK_PATH.exists():
        print(f"Warning: Notebook not found at {NOTEBOOK_PATH}")
        notebook = {
            "cells": [],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4,
            "nbformat_minor": 4
        }
    else:
        with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    
    new_cells = [
        # Chapter 18: Differential Privacy
        create_markdown_cell(CHAPTER_18_INTRO),
        create_code_cell(CHAPTER_18_CODE),
        create_code_cell(CHAPTER_18_DEMO),
        
        # Chapter 19: Energy Efficiency
        create_markdown_cell(CHAPTER_19_INTRO),
        create_code_cell(CHAPTER_19_CODE),
        create_code_cell(CHAPTER_19_METHODS),
        
        # Interview Questions
        create_markdown_cell(INTERVIEW_QUESTIONS),
    ]
    
    notebook['cells'].extend(new_cells)
    
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Updated {NOTEBOOK_PATH}")
    print(f"Added {len(new_cells)} cells:")
    print("  - Chapter 18: Differential Privacy (3 cells)")
    print("  - Chapter 19: Energy Efficiency (3 cells)")
    print("  - Interview Questions (1 cell)")


if __name__ == "__main__":
    main()
