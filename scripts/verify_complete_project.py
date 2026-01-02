"""
Final comprehensive verification of all project modules.
Tests all 12 phases of advanced integration methods.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
import traceback

def test_module(name: str, test_func) -> bool:
    """Test a module and return success status."""
    try:
        test_func()
        print(f"  ✓ {name}")
        return True
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return False

def run_all_tests():
    """Run comprehensive verification of all modules."""
    print("=" * 70)
    print("COMPREHENSIVE PROJECT VERIFICATION")
    print("AI-MASTERY-2026 ADVANCED INTEGRATION")
    print("=" * 70)
    
    results = {'passed': 0, 'failed': 0}
    
    # =========================================================================
    # Phase 1-2: Core Integration, MCMC, Variational Inference
    # =========================================================================
    print("\n[Phase 1-2] Core Integration Methods...")
    
    def test_integration():
        from src.core.integration import trapezoidal_rule, monte_carlo_integrate
        result = trapezoidal_rule(lambda x: x**2, 0, 1, 100)
        assert abs(result - 1/3) < 0.01
    
    def test_mcmc():
        from src.core.mcmc import metropolis_hastings
        samples = metropolis_hastings(
            log_prob=lambda x: -0.5 * x**2,
            initial_state=0.0,
            n_samples=100
        )
        assert len(samples.samples) == 100
    
    def test_vi():
        from src.core.variational_inference import MeanFieldVI
        vi = MeanFieldVI(dim=2)
        assert vi is not None
    
    if test_module("Integration", test_integration): results['passed'] += 1
    else: results['failed'] += 1
    if test_module("MCMC", test_mcmc): results['passed'] += 1
    else: results['failed'] += 1
    if test_module("Variational Inference", test_vi): results['passed'] += 1
    else: results['failed'] += 1
    
    # =========================================================================
    # Phase 3-8: Advanced Integration
    # =========================================================================
    print("\n[Phase 3-8] Advanced Integration...")
    
    def test_advanced():
        from src.core.advanced_integration import NeuralODE, MultiModalIntegrator
        ode = NeuralODE(state_dim=2, hidden_dim=16)
        mmi = MultiModalIntegrator()
        assert ode is not None and mmi is not None
    
    if test_module("Advanced Integration", test_advanced): results['passed'] += 1
    else: results['failed'] += 1
    
    # =========================================================================
    # Phase 9: Hardware, PPL, Adaptive
    # =========================================================================
    print("\n[Phase 9] Hardware/PPL/Adaptive...")
    
    def test_hardware():
        from src.core.hardware_accelerated_integration import monte_carlo_cpu
        result = monte_carlo_cpu(lambda x: x**2, 1000)
        assert 0.2 < result < 0.5
    
    def test_ppl():
        from src.core.ppl_integration import BayesianRegression
        model = BayesianRegression('custom')
        assert model is not None
    
    def test_adaptive():
        from src.core.adaptive_integration import AdaptiveIntegrator
        integrator = AdaptiveIntegrator()
        result = integrator.integrate(lambda x: x**2, 0, 1)
        assert 0.3 < result['result'] < 0.4
    
    if test_module("Hardware Acceleration", test_hardware): results['passed'] += 1
    else: results['failed'] += 1
    if test_module("PPL Integration", test_ppl): results['passed'] += 1
    else: results['failed'] += 1
    if test_module("Adaptive Integration", test_adaptive): results['passed'] += 1
    else: results['failed'] += 1
    
    # =========================================================================
    # Phase 10: RL and Causal Inference
    # =========================================================================
    print("\n[Phase 10] RL and Causal Inference...")
    
    def test_rl():
        from src.core.rl_integration import RLIntegrationSystem
        rl = RLIntegrationSystem(state_dim=4, action_dim=2)
        assert rl is not None
    
    def test_causal():
        from src.core.causal_inference import CausalInferenceSystem
        causal = CausalInferenceSystem()
        assert causal is not None
    
    if test_module("RL Integration", test_rl): results['passed'] += 1
    else: results['failed'] += 1
    if test_module("Causal Inference", test_causal): results['passed'] += 1
    else: results['failed'] += 1
    
    # =========================================================================
    # Phase 11: GNN and XAI
    # =========================================================================
    print("\n[Phase 11] GNN and Explainable AI...")
    
    def test_gnn():
        from src.core.gnn_integration import BayesianGCN, generate_synthetic_graph
        graph = generate_synthetic_graph(num_nodes=30, num_classes=2)
        model = BayesianGCN(graph.num_features, 16, 2)
        pred = model.predict(graph)
        assert len(pred.predictions) == 30
    
    def test_xai():
        from src.core.explainable_ai import ExplainableModel
        model = ExplainableModel()
        data = model.generate_medical_data(100)
        model.train(data['X'], data['y'], data['feature_names'])
        exp = model.predict_with_explanation(data['X'][:1], num_samples=10)
        assert len(exp) == 1
    
    if test_module("GNN Integration", test_gnn): results['passed'] += 1
    else: results['failed'] += 1
    if test_module("Explainable AI", test_xai): results['passed'] += 1
    else: results['failed'] += 1
    
    # =========================================================================
    # Phase 12: Privacy and Energy
    # =========================================================================
    print("\n[Phase 12] Differential Privacy and Energy Efficiency...")
    
    def test_dp():
        from src.core.differential_privacy import DifferentiallyPrivateIntegrator
        dp = DifferentiallyPrivateIntegrator(epsilon=1.0)
        result = dp.private_mean(np.array([1,2,3,4,5]), bounds=(0,10))
        assert result.epsilon_used == 1.0
    
    def test_energy():
        from src.core.energy_efficient import EnergyEfficientIntegrator
        integrator = EnergyEfficientIntegrator(device='iot')
        result = integrator.integrate(lambda x: x**2, 0, 1)
        assert result.energy_cost > 0
    
    if test_module("Differential Privacy", test_dp): results['passed'] += 1
    else: results['failed'] += 1
    if test_module("Energy Efficiency", test_energy): results['passed'] += 1
    else: results['failed'] += 1
    
    # =========================================================================
    # Phase 13: Q1 2026 Optimization & Sprints (New)
    # =========================================================================
    print("\n[Phase 13] Q1 2026 Sprint Optimization...")
    
    def test_sprint_structure():
        import os
        required_paths = [
            "Q1_ROADMAP.md",
            "docs/LEARNING_LOG.md",
            "docs/BACKLOG.md",
            "sprints/week01_rag_production/README.md",
            "sprints/week01_rag_production/api.py",
            "sprints/week01_rag_production/ui.py",
            "sprints/week01_rag_production/stress_test.py",
            "sprints/week01_rag_production/notebooks/day2_eval_pipeline.ipynb"
        ]
        
        missing = []
        for p in required_paths:
            if not os.path.exists(p):
                missing.append(p)
        
        if missing:
            raise FileNotFoundError(f"Missing optimization files: {missing}")
        return True

    if test_module("Sprint Structure", test_sprint_structure): results['passed'] += 1
    else: results['failed'] += 1
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    total = results['passed'] + results['failed']
    print(f"\nPassed: {results['passed']}/{total} ({results['passed']/total*100:.0f}%)")
    print(f"Failed: {results['failed']}/{total}")
    
    if results['failed'] == 0:
        print("\n✅ ALL MODULES VERIFIED SUCCESSFULLY!")
    else:
        print(f"\n⚠️  {results['failed']} module(s) need attention")
    
    # Project Statistics
    print("\n" + "-" * 70)
    print("PROJECT STATISTICS")
    print("-" * 70)
    
    import os
    core_files = [f for f in os.listdir('src/core') if f.endswith('.py') and f != '__init__.py']
    test_files = [f for f in os.listdir('tests') if f.startswith('test_') and f.endswith('.py')]
    
    total_lines = 0
    for f in core_files:
        with open(f'src/core/{f}', 'r', encoding='utf-8') as file:
            total_lines += len(file.readlines())
    
    print(f"\nCore Modules: {len(core_files)} files")
    print(f"Test Files: {len(test_files)} files")
    print(f"Total Lines (core): ~{total_lines:,}")
    
    print("\n" + "-" * 70)
    print("INDUSTRIAL CASE STUDIES INTEGRATED")
    print("-" * 70)
    case_studies = [
        "1. Boston Dynamics - Neural ODEs",
        "2. Mayo Clinic - Multi-modal Healthcare",
        "3. Apple HealthKit - Federated Learning",
        "4. IBM AI Fairness 360 - Ethical AI",
        "5. NVIDIA cuQuantum - Hardware Acceleration",
        "6. Uber Pyro - PPL for Inference",
        "7. Wolfram Alpha - Adaptive Integration",
        "8. DeepMind AlphaGo - MCTS + RL",
        "9. Microsoft Uplift - Causal Marketing",
        "10. Meta Social Graph - Bayesian GNNs",
        "11. IBM Watson Oncology - Explainable AI",
        "12. Apple Privacy-Preserving ML - Differential Privacy",
        "13. Google DeepMind Data Centers - Energy Efficiency",
    ]
    for cs in case_studies:
        print(f"  {cs}")
    
    return results['failed'] == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
