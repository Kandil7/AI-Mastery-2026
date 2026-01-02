"""
Script to update the advanced_integration_mcmc_vi.ipynb notebook
with enhanced visualizations and interactive demos.
"""

import json
import sys
from pathlib import Path


def create_enhanced_chapters():
    """Create new notebook cells for enhanced chapters."""
    
    cells = []
    
    # Chapter 3 Enhancement: Neural ODEs with Robot Dynamics
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3.1 Neural ODEs: Robot Dynamics Visualization\n",
            "\n",
            "Let's visualize the uncertainty propagation in a robot dynamics simulation.\n",
            "This mirrors the approach used by **Boston Dynamics** for Atlas robot control."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "from src.core.advanced_integration import robot_dynamics_demo\n",
            "\n",
            "# Run the demo\n",
            "results = robot_dynamics_demo(dim=2, t_max=10.0, n_steps=101)\n",
            "\n",
            "# Extract data\n",
            "mean_path = results['mean_path'][:, 0, :]  # Shape: (101, 2)\n",
            "std_path = results['std_path'][:, 0, :]\n",
            "t = results['t_span']\n",
            "\n",
            "# Plot\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# Position over time with uncertainty\n",
            "ax = axes[0]\n",
            "ax.plot(t, mean_path[:, 0], 'b-', lw=2, label='Position (mean)')\n",
            "ax.fill_between(t, \n",
            "                mean_path[:, 0] - 2*std_path[:, 0],\n",
            "                mean_path[:, 0] + 2*std_path[:, 0],\n",
            "                alpha=0.3, color='blue', label='95% CI')\n",
            "ax.set_xlabel('Time')\n",
            "ax.set_ylabel('Position')\n",
            "ax.set_title('Robot Joint Position with Uncertainty')\n",
            "ax.legend()\n",
            "ax.grid(True, alpha=0.3)\n",
            "\n",
            "# Phase space plot\n",
            "ax = axes[1]\n",
            "for i in range(min(20, results['trajectories'].shape[0])):\n",
            "    traj = results['trajectories'][i, :, 0, :]\n",
            "    ax.plot(traj[:, 0], traj[:, 1], 'k-', alpha=0.1)\n",
            "ax.plot(mean_path[:, 0], mean_path[:, 1], 'b-', lw=2, label='Mean trajectory')\n",
            "ax.scatter([mean_path[0, 0]], [mean_path[0, 1]], c='green', s=100, zorder=10, label='Start')\n",
            "ax.scatter([mean_path[-1, 0]], [mean_path[-1, 1]], c='red', s=100, zorder=10, label='End')\n",
            "ax.set_xlabel('Position')\n",
            "ax.set_ylabel('Velocity')\n",
            "ax.set_title('Phase Space Trajectory')\n",
            "ax.legend()\n",
            "ax.grid(True, alpha=0.3)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(f\"Final position uncertainty: {std_path[-1, 0]:.4f}\")\n",
            "print(f\"Uncertainty growth rate: {std_path[-1, 0] / std_path[0, 0]:.2f}x\")"
        ]
    })
    
    # Chapter 4 Enhancement: Multi-Modal Healthcare
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4.1 Multi-Modal Healthcare Integration Demo\n",
            "\n",
            "This demo shows how to fuse clinical data, imaging, and text records.\n",
            "Inspired by **Mayo Clinic's** AI diagnostic system."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from src.core.advanced_integration import MultiModalIntegrator, generate_patient_data\n",
            "\n",
            "# Generate synthetic data\n",
            "data = generate_patient_data(n_samples=500)\n",
            "\n",
            "# Create model\n",
            "model = MultiModalIntegrator(\n",
            "    clinical_dim=5, xray_dim=3, text_dim=4, hidden_dim=64\n",
            ")\n",
            "\n",
            "# Prepare tensors\n",
            "clinical = torch.tensor(data['clinical_data'], dtype=torch.float32)\n",
            "xray = torch.tensor(data['xray_data'], dtype=torch.float32)\n",
            "text = torch.tensor(data['text_data'], dtype=torch.float32)\n",
            "\n",
            "# Get predictions with uncertainty\n",
            "predictions, uncertainty = model.predict_with_confidence(\n",
            "    clinical, xray, text, n_samples=30\n",
            ")\n",
            "\n",
            "# Visualization\n",
            "fig, axes = plt.subplots(1, 3, figsize=(15, 4))\n",
            "\n",
            "# 1. Prediction distribution\n",
            "ax = axes[0]\n",
            "ax.hist(predictions[data['labels'] == 0], bins=30, alpha=0.7, label='Healthy', density=True)\n",
            "ax.hist(predictions[data['labels'] == 1], bins=30, alpha=0.7, label='Disease', density=True)\n",
            "ax.set_xlabel('Predicted Probability')\n",
            "ax.set_ylabel('Density')\n",
            "ax.set_title('Prediction Distribution by Class')\n",
            "ax.legend()\n",
            "\n",
            "# 2. Uncertainty vs correctness\n",
            "ax = axes[1]\n",
            "correct = (predictions > 0.5).astype(int) == data['labels']\n",
            "ax.hist(uncertainty[correct], bins=30, alpha=0.7, label='Correct', density=True)\n",
            "ax.hist(uncertainty[~correct], bins=30, alpha=0.7, label='Incorrect', density=True)\n",
            "ax.set_xlabel('Uncertainty')\n",
            "ax.set_ylabel('Density')\n",
            "ax.set_title('Uncertainty Distribution')\n",
            "ax.legend()\n",
            "\n",
            "# 3. High uncertainty cases\n",
            "ax = axes[2]\n",
            "high_unc_idx = np.argsort(uncertainty)[-20:]\n",
            "ax.scatter(predictions[high_unc_idx], uncertainty[high_unc_idx], \n",
            "           c=data['labels'][high_unc_idx], cmap='coolwarm', s=50)\n",
            "ax.set_xlabel('Prediction')\n",
            "ax.set_ylabel('Uncertainty')\n",
            "ax.set_title('High Uncertainty Cases (need human review)')\n",
            "ax.axhline(y=np.percentile(uncertainty, 90), color='k', linestyle='--', alpha=0.5)\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# Summary stats\n",
            "accuracy = np.mean(correct)\n",
            "mean_unc_correct = np.mean(uncertainty[correct])\n",
            "mean_unc_incorrect = np.mean(uncertainty[~correct])\n",
            "print(f\"Accuracy: {accuracy:.2%}\")\n",
            "print(f\"Mean uncertainty (correct): {mean_unc_correct:.4f}\")\n",
            "print(f\"Mean uncertainty (incorrect): {mean_unc_incorrect:.4f}\")"
        ]
    })
    
    # Chapter 5 Enhancement: Federated Learning
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 5.1 Federated Learning: Hospital Network Simulation\n",
            "\n",
            "Simulating a federated healthcare analytics system with 5 hospitals.\n",
            "This mirrors **Apple HealthKit's** privacy-preserving approach."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "from src.core.advanced_integration import federated_demo, FederatedHospital\n",
            "\n",
            "# Run federated demo\n",
            "results = federated_demo(n_hospitals=5, n_rounds=3)\n",
            "\n",
            "# Plot aggregation method comparison\n",
            "fig, axes = plt.subplots(1, 2, figsize=(14, 5))\n",
            "\n",
            "# 1. Aggregation methods comparison\n",
            "ax = axes[0]\n",
            "methods = list(results['results'].keys())\n",
            "colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))\n",
            "\n",
            "for i, method in enumerate(methods):\n",
            "    history = results['results'][method]['history']\n",
            "    ax.plot(range(1, len(history)+1), history, 'o-', \n",
            "            color=colors[i], lw=2, markersize=8, label=method.replace('_', ' '))\n",
            "\n",
            "ax.axhline(y=results['true_risk'], color='k', linestyle='--', lw=2, label='True global risk')\n",
            "ax.set_xlabel('Aggregation Round')\n",
            "ax.set_ylabel('Estimated Global Risk')\n",
            "ax.set_title('Comparison of Federated Aggregation Strategies')\n",
            "ax.legend(loc='best')\n",
            "ax.grid(True, alpha=0.3)\n",
            "\n",
            "# 2. Hospital age distributions\n",
            "ax = axes[1]\n",
            "hospitals = [FederatedHospital(i, ['young', 'elderly', 'mixed', 'young', 'elderly'][i], 200)\n",
            "             for i in range(5)]\n",
            "\n",
            "for i, h in enumerate(hospitals):\n",
            "    ax.hist(h.data.age, bins=20, alpha=0.5, label=f\"Hospital {i} ({h.data_dist})\")\n",
            "\n",
            "ax.set_xlabel('Patient Age')\n",
            "ax.set_ylabel('Count')\n",
            "ax.set_title('Age Distribution Across Hospitals (Non-IID)')\n",
            "ax.legend(loc='best')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# Summary\n",
            "print(\"\\n=== Aggregation Method Errors ===\")\n",
            "for method in methods:\n",
            "    final = results['results'][method]['final_risk']\n",
            "    error = abs(final - results['true_risk'])\n",
            "    print(f\"{method:25s}: {final:.4f} (error: {error:.4f})\")"
        ]
    })
    
    # Chapter 6 Enhancement: Ethics & Bias
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 6.1 Ethics: Bias Detection in Lending Decisions\n",
            "\n",
            "Analyzing algorithmic bias in a simulated lending system.\n",
            "This follows **IBM AI Fairness 360** methodology."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "from src.core.advanced_integration import biased_lending_simulation, analyze_bias\n",
            "\n",
            "# Run simulation with moderate bias\n",
            "results = biased_lending_simulation(n_samples=10000, bias_factor=0.4)\n",
            "metrics = analyze_bias(results)\n",
            "\n",
            "# Visualization\n",
            "fig, axes = plt.subplots(2, 2, figsize=(12, 10))\n",
            "\n",
            "# 1. True worth distribution\n",
            "ax = axes[0, 0]\n",
            "ax.hist(results['true_worth'][results['sensitive_attr'] == 0], \n",
            "        bins=50, alpha=0.6, label='Group 0', density=True)\n",
            "ax.hist(results['true_worth'][results['sensitive_attr'] == 1], \n",
            "        bins=50, alpha=0.6, label='Group 1', density=True)\n",
            "ax.set_xlabel('True Creditworthiness')\n",
            "ax.set_ylabel('Density')\n",
            "ax.set_title('True Creditworthiness by Group')\n",
            "ax.legend()\n",
            "\n",
            "# 2. Perceived worth (after bias)\n",
            "ax = axes[0, 1]\n",
            "ax.hist(results['perceived_worth'][results['sensitive_attr'] == 0], \n",
            "        bins=50, alpha=0.6, label='Group 0', density=True)\n",
            "ax.hist(results['perceived_worth'][results['sensitive_attr'] == 1], \n",
            "        bins=50, alpha=0.6, label='Group 1', density=True)\n",
            "ax.axvline(x=0.6, color='r', linestyle='--', label='Approval threshold')\n",
            "ax.set_xlabel('Perceived Creditworthiness')\n",
            "ax.set_ylabel('Density')\n",
            "ax.set_title('Perceived Worth (Biased)')\n",
            "ax.legend()\n",
            "\n",
            "# 3. Approval rates comparison\n",
            "ax = axes[1, 0]\n",
            "groups = ['Group 0', 'Group 1']\n",
            "rates = [metrics['approval_rate_group0'], metrics['approval_rate_group1']]\n",
            "colors = ['steelblue', 'coral']\n",
            "bars = ax.bar(groups, rates, color=colors)\n",
            "ax.axhline(y=0.8 * rates[0], color='k', linestyle='--', alpha=0.5, label='80% rule threshold')\n",
            "ax.set_ylabel('Approval Rate')\n",
            "ax.set_title(f'Approval Rates (Disparity: {metrics[\"approval_disparity\"]:.1%})')\n",
            "ax.legend()\n",
            "for bar, rate in zip(bars, rates):\n",
            "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, \n",
            "            f'{rate:.1%}', ha='center', fontsize=12)\n",
            "\n",
            "# 4. Fairness metrics summary\n",
            "ax = axes[1, 1]\n",
            "metric_names = ['Disparate Impact\\nRatio', 'True Worth\\nDifference', 'Underestimation\\nDifference']\n",
            "metric_values = [\n",
            "    metrics['disparate_impact_ratio'],\n",
            "    abs(metrics['true_worth_group0'] - metrics['true_worth_group1']),\n",
            "    abs(metrics['underestimation_group1'] - metrics['underestimation_group0'])\n",
            "]\n",
            "colors = ['red' if metric_values[0] < 0.8 else 'green', 'steelblue', 'coral']\n",
            "ax.barh(metric_names, metric_values, color=colors)\n",
            "ax.axvline(x=0.8, color='k', linestyle='--', alpha=0.5, label='Fair threshold')\n",
            "ax.set_xlabel('Value')\n",
            "ax.set_title('Fairness Metrics')\n",
            "\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "# Summary\n",
            "print(\"\\n=== Bias Analysis Summary ===\")\n",
            "print(f\"Approval rate Group 0: {metrics['approval_rate_group0']:.2%}\")\n",
            "print(f\"Approval rate Group 1: {metrics['approval_rate_group1']:.2%}\")\n",
            "print(f\"Disparate Impact Ratio: {metrics['disparate_impact_ratio']:.3f}\")\n",
            "print(f\"\\nLegal Status: {'⚠️ POTENTIAL DISCRIMINATION' if metrics['disparate_impact_ratio'] < 0.8 else '✅ Within acceptable range'}\")"
        ]
    })
    
    return cells


def update_notebook():
    """Update the notebook with enhanced chapters."""
    notebook_path = Path(r"k:\learning\technical\ai-ml\AI-Mastery-2026\notebooks\01_mathematical_foundations\advanced_integration_mcmc_vi.ipynb")
    
    # Read existing notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    # Get new cells
    new_cells = create_enhanced_chapters()
    
    # Find insertion point (after Chapter 3, 4, 5, 6 headers)
    # For simplicity, we'll append to the end
    notebook['cells'].extend(new_cells)
    
    # Write back
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    print(f"Successfully updated {notebook_path}")
    print(f"Added {len(new_cells)} new cells")


if __name__ == "__main__":
    update_notebook()
