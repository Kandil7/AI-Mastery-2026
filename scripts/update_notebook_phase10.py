"""
Script to add Chapters 14-15 to the Advanced Integration notebook.

Adds:
- Chapter 14: Integration in Reinforcement Learning
- Chapter 15: Integration for Causal Inference
"""

import json
from pathlib import Path

NOTEBOOK_PATH = Path("notebooks/01_mathematical_foundations/advanced_integration_mcmc_vi.ipynb")


def create_markdown_cell(source: str) -> dict:
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.split('\n')
    }


def create_code_cell(source: str) -> dict:
    """Create a code cell."""
    lines = source.split('\n')
    lines_with_newlines = [line + '\n' for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines_with_newlines
    }


# Chapter 14: Integration in Reinforcement Learning
CHAPTER_14_INTRO = """# Chapter 14: Integration in Reinforcement Learning

In Reinforcement Learning (RL), agents learn to make optimal decisions through environment interaction. **Integration** plays a crucial role, especially when dealing with uncertainty in dynamics and rewards.

## The RL Objective as Integration

The RL objective is to learn a policy π(a|s) that maximizes expected cumulative reward:

$$J(\\pi) = \\mathbb{E}_{\\tau \\sim p_\\pi(\\tau)}\\left[\\sum_{t=0}^T \\gamma^t r(s_t, a_t)\\right] = \\int p_\\pi(\\tau) R(\\tau) d\\tau$$

where:
- τ = (s₀, a₀, s₁, a₁, ..., s_T) is a trajectory
- p_π(τ) is the trajectory distribution under policy π
- γ is the discount factor

**The challenge**: Environment dynamics p(s_{t+1}|s_t, a_t) may be unknown or complex, making integration over all possible trajectories difficult.

## Industrial Case Study: DeepMind's AlphaGo/AlphaZero

**Challenge**: Go has ~10¹⁷⁰ possible states (exhaustive search impossible)

**Solution**: Combine Monte Carlo Tree Search (MCTS) + Neural Networks:

$$Q(s,a) = \\frac{1}{N(s,a)}\\sum_{i=1}^{N(s,a)} G_i(s,a) + c \\cdot P(s,a) \\cdot \\frac{\\sqrt{\\sum_b N(s,b)}}{1 + N(s,a)}$$

**Results**:
- Defeated world champion Lee Sedol (2016)
- Superhuman in Go, Chess, and Shogi with same algorithm
- Applied to logistics: **$200M/year savings** at Alphabet
- **40% reduction** in data center energy consumption
"""

CHAPTER_14_CODE = '''import numpy as np
import sys
sys.path.insert(0, '../..')
from src.core.rl_integration import (
    RLIntegrationSystem,
    SimpleValueNetwork,
    simple_policy,
    Episode
)

print("=" * 60)
print("Integration in Reinforcement Learning")
print("=" * 60)

# Create RL system
rl = RLIntegrationSystem()

# 1. Monte Carlo Policy Evaluation
print("\\n1. Monte Carlo Policy Evaluation")
print("-" * 40)
print("V(s) = E[G | S=s] = ∫ G · p(G|s) dG")
print("\\nThis is Monte Carlo integration of future rewards...")

value_estimates, returns_by_state = rl.monte_carlo_policy_evaluation(
    simple_policy, n_episodes=50
)

print(f"\\nEvaluated {len(returns_by_state)} unique states")
print(f"Average value estimate: {value_estimates[-1]:.2f}")
'''

CHAPTER_14_POLICY_GRADIENT = '''# 2. Policy Gradient (REINFORCE)
print("\\n2. Policy Gradient Training (REINFORCE)")
print("-" * 40)
print("∇J(θ) = E[∑_t ∇log π_θ(a_t|s_t) · G_t]")
print("\\nThis integrates over trajectories to estimate gradients...")

results = rl.policy_gradient_reinforce(n_episodes=50)

print(f"\\nTraining completed in {results.training_time:.2f}s")
print(f"Initial reward: {results.episode_rewards[0]:.2f}")
print(f"Final reward: {results.episode_rewards[-1]:.2f}")
print(f"Improvement: {results.episode_rewards[-1] - results.episode_rewards[0]:.2f}")
'''

CHAPTER_14_MCTS = '''# 3. MCTS Value Estimation
print("\\n3. Monte Carlo Tree Search (MCTS) Value Estimation")
print("-" * 40)

test_states = [
    np.array([-0.5, 0.0]),   # Start position
    np.array([-0.2, 0.02]),  # Near goal
    np.array([-0.9, -0.05])  # Far from goal
]

for state in test_states:
    value, uncertainty = rl.mcts_value_estimate(state, n_simulations=30, depth=10)
    print(f"State ({state[0]:.2f}, {state[1]:.2f}): "
          f"Value = {value:.2f} ± {uncertainty:.2f}")
'''

# Chapter 15: Causal Inference
CHAPTER_15_INTRO = """# Chapter 15: Integration for Causal Inference

Causal Inference aims to estimate **causal effects** rather than mere correlations. Integration is fundamental:

$$\\text{ATE} = \\mathbb{E}[Y(1) - Y(0)] = \\int \\mathbb{E}[Y(1) - Y(0) | X = x] \\, p(x) \\, dx$$

where:
- Y(1), Y(0) are **potential outcomes** with/without treatment
- X are observed covariates
- This is an integral over the covariate distribution

## Why Naive Estimation Fails

In observational data, treatment assignment often depends on covariates (**confounding**):
- Sicker patients more likely to receive treatment
- Wealthier customers more likely to respond to ads

Naive comparison (treated vs. control means) conflates:
- True treatment effect
- Selection bias

## Industrial Case Study: Microsoft Uplift Modeling

**Challenge**: Which customers will buy **BECAUSE** of marketing email?

**Solution**: Causal inference to estimate individual "uplift":

$$\\text{Uplift}(x) = P(Y=1|T=1,X=x) - P(Y=1|T=0,X=x)$$

**Results**:
- **76% ROI increase** in marketing campaigns
- **40% reduction** in campaign volume (same conversions)
- **$100M/year savings** in marketing costs
"""

CHAPTER_15_CODE = '''from src.core.causal_inference import (
    CausalInferenceSystem,
    ATEResult,
    CATEResult
)

print("=" * 60)
print("Integration for Causal Inference")
print("=" * 60)

# Create system
causal = CausalInferenceSystem()

# Generate observational data with confounding
print("\\nGenerating synthetic healthcare data...")
data = causal.generate_synthetic_data(n_samples=500)

true_ate = data['true_effect'].mean()
naive_ate = data[data['treatment']==1]['outcome'].mean() - data[data['treatment']==0]['outcome'].mean()

print(f"True ATE: {true_ate:.3f}")
print(f"Naive ATE (biased): {naive_ate:.3f}")
print(f"Confounding bias: {naive_ate - true_ate:.3f}")
'''

CHAPTER_15_METHODS = '''# Compare estimation methods
print("\\n" + "=" * 60)
print("Causal Estimation Methods")
print("=" * 60)

# 1. Inverse Propensity Weighting
print("\\n1. Inverse Propensity Weighting (IPW)")
ipw_result = causal.estimate_ate_ipw(data)
print(f"   ATE: {ipw_result.ate_estimate:.3f} ± {ipw_result.ate_std_error:.3f}")

# 2. Doubly Robust Estimation  
print("\\n2. Doubly Robust Estimation")
dr_result = causal.estimate_ate_doubly_robust(data)
print(f"   ATE: {dr_result.ate_estimate:.3f} ± {dr_result.ate_std_error:.3f}")

# 3. Bayesian Causal Inference
print("\\n3. Bayesian Causal Inference")
bayes_result = causal.bayesian_causal_inference(data, n_posterior_samples=100)
print(f"   ATE: {bayes_result.ate_mean:.3f} ± {bayes_result.ate_std:.3f}")

# Summary comparison
print("\\n" + "-" * 60)
print(f"{'Method':<25} {'Estimate':<12} {'Error vs True':<15}")
print("-" * 60)

methods = [
    ('Naive', naive_ate),
    ('IPW', ipw_result.ate_estimate),
    ('Doubly Robust', dr_result.ate_estimate),
    ('Bayesian', bayes_result.ate_mean),
    ('True', true_ate)
]

for name, est in methods:
    if name == 'True':
        print(f"{name:<25} {est:<12.3f}")
    else:
        error = abs(est - true_ate) / true_ate
        print(f"{name:<25} {est:<12.3f} {error:<15.1%}")
'''

CHAPTER_15_HET = '''# Heterogeneous Treatment Effects
print("\\n" + "=" * 60)
print("Heterogeneous Treatment Effects by Age Group")
print("=" * 60)

het_analysis = causal.analyze_heterogeneous_effects(
    data, 
    dr_result.diagnostics['individual_effects']
)

print("\\nTreatment effects vary by patient characteristics:")
print(het_analysis['age'])

print("\\nKey insight: Older patients may benefit more from treatment!")
print("This enables personalized medicine and targeted interventions.")
'''

# Interview Questions
INTERVIEW_QUESTIONS = """# Advanced Integration Interview Questions: RL & Causal Inference

## Reinforcement Learning

**Q1: Explain how Monte Carlo integration is used in REINFORCE.**

**A**: REINFORCE estimates the policy gradient using Monte Carlo sampling:

$$\\nabla J(\\theta) = \\mathbb{E}\\left[\\sum_t \\nabla \\log \\pi_\\theta(a_t|s_t) \\cdot G_t\\right]$$

We can't evaluate this expectation analytically, so we:
1. Sample trajectories τ from the current policy
2. Compute returns G_t for each timestep
3. Average the gradient estimates

This is Monte Carlo integration over the trajectory distribution.

---

**Q2: What is the exploration-exploitation tradeoff in MCTS?**

**A**: MCTS balances via the UCB formula:

$$Q(s,a) + c \\cdot P(s,a) \\cdot \\frac{\\sqrt{N(s)}}{1 + N(s,a)}$$

- **Exploitation**: First term Q(s,a) favors high-value actions
- **Exploration**: Second term grows for rarely-visited actions
- **c** controls the balance (higher = more exploration)

---

## Causal Inference

**Q3: Why is Doubly Robust estimation preferred?**

**A**: Doubly Robust (DR) is consistent if EITHER:
1. The propensity model is correct, OR
2. The outcome model is correct

This "double protection" makes it more robust to misspecification:

$$\\hat{\\tau}_{DR} = \\frac{1}{n}\\sum_i \\left[\\hat{\\mu}_1(X_i) - \\hat{\\mu}_0(X_i) + \\frac{T_i(Y_i - \\hat{\\mu}_1(X_i))}{e(X_i)} - \\frac{(1-T_i)(Y_i - \\hat{\\mu}_0(X_i))}{1-e(X_i)}\\right]$$

---

**Q4: Implement a simple propensity score trimming function.**

```python
def trim_propensity_scores(ps, min_val=0.05, max_val=0.95):
    '''
    Trim extreme propensity scores to reduce variance.
    
    Extreme scores (near 0 or 1) create high-variance weights
    in IPW estimation.
    '''
    return np.clip(ps, min_val, max_val)
```

---

**Q5: What is the fundamental problem of causal inference?**

**A**: We can never observe both Y(1) AND Y(0) for the same individual.

This is a **missing data problem**: for each person, we observe either:
- Y(1) if treated, but Y(0) is counterfactual (unobserved)
- Y(0) if control, but Y(1) is counterfactual (unobserved)

We use statistical assumptions (ignorability, overlap) to estimate causal effects despite never observing individual treatment effects.
"""


def main():
    """Main function to update the notebook."""
    if not NOTEBOOK_PATH.exists():
        print(f"Error: Notebook not found at {NOTEBOOK_PATH}")
        notebook = {
            "cells": [],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
    else:
        with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
    
    new_cells = [
        # Chapter 14: RL Integration
        create_markdown_cell(CHAPTER_14_INTRO),
        create_code_cell(CHAPTER_14_CODE),
        create_code_cell(CHAPTER_14_POLICY_GRADIENT),
        create_code_cell(CHAPTER_14_MCTS),
        
        # Chapter 15: Causal Inference
        create_markdown_cell(CHAPTER_15_INTRO),
        create_code_cell(CHAPTER_15_CODE),
        create_code_cell(CHAPTER_15_METHODS),
        create_code_cell(CHAPTER_15_HET),
        
        # Interview Questions
        create_markdown_cell(INTERVIEW_QUESTIONS),
    ]
    
    notebook['cells'].extend(new_cells)
    
    with open(NOTEBOOK_PATH, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Successfully updated {NOTEBOOK_PATH}")
    print(f"Added {len(new_cells)} new cells:")
    print("  - Chapter 14: Integration in RL (4 cells)")
    print("  - Chapter 15: Causal Inference (4 cells)")
    print("  - Interview Questions (1 cell)")


if __name__ == "__main__":
    main()
