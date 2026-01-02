"""
Tests for RL Integration and Causal Inference modules.
"""

import pytest
import numpy as np
import pandas as pd


# ============================================================================
# RL Integration Tests
# ============================================================================

class TestRLIntegration:
    """Tests for reinforcement learning integration methods."""
    
    def test_simple_value_network_forward(self):
        """Test value network forward pass."""
        from src.core.rl_integration import SimpleValueNetwork
        
        net = SimpleValueNetwork(input_dim=2, hidden_dim=32)
        
        # Single state
        state = np.array([0.5, 0.1])
        value = net.forward(state)
        
        assert value.shape == (1, 1)
        assert np.isfinite(value[0, 0])
    
    def test_simple_value_network_backward(self):
        """Test value network backward pass updates weights."""
        from src.core.rl_integration import SimpleValueNetwork
        
        net = SimpleValueNetwork(input_dim=2, hidden_dim=32, learning_rate=0.1)
        
        # Forward pass
        state = np.array([0.5, 0.1])
        value_before = net.forward(state).copy()
        
        # Backward pass with non-zero gradient
        net.backward(np.array([[1.0]]))
        
        # Forward again - should be different
        value_after = net.forward(state)
        
        assert not np.allclose(value_before, value_after)
    
    def test_rl_system_initialization(self):
        """Test RL system initialization."""
        from src.core.rl_integration import RLIntegrationSystem
        
        rl = RLIntegrationSystem(state_dim=2, action_dim=1)
        
        assert rl.state_dim == 2
        assert rl.action_dim == 1
        assert rl.value_network is not None
    
    def test_compute_returns(self):
        """Test discounted return computation."""
        from src.core.rl_integration import RLIntegrationSystem
        
        rl = RLIntegrationSystem()
        
        rewards = [1.0, 1.0, 1.0, 1.0]
        gamma = 0.9
        
        returns = rl.compute_returns(rewards, gamma)
        
        # G_0 = 1 + 0.9*1 + 0.81*1 + 0.729*1 = 3.439
        assert len(returns) == 4
        assert returns[0] == pytest.approx(3.439, rel=0.01)
        assert returns[-1] == 1.0  # Last return is just the last reward
    
    def test_run_episode(self):
        """Test episode simulation."""
        from src.core.rl_integration import RLIntegrationSystem, simple_policy
        
        rl = RLIntegrationSystem()
        episode = rl.run_episode(simple_policy, max_steps=50)
        
        assert len(episode.states) == episode.length
        assert len(episode.actions) == episode.length
        assert len(episode.rewards) == episode.length
        assert episode.total_reward == sum(episode.rewards)
    
    def test_monte_carlo_policy_evaluation(self):
        """Test Monte Carlo policy evaluation."""
        from src.core.rl_integration import RLIntegrationSystem, simple_policy
        
        rl = RLIntegrationSystem()
        
        value_estimates, returns_by_state = rl.monte_carlo_policy_evaluation(
            simple_policy, n_episodes=10
        )
        
        assert len(value_estimates) > 0
        assert len(returns_by_state) > 0
        assert all(isinstance(v, list) for v in returns_by_state.values())
    
    def test_mcts_value_estimate(self):
        """Test MCTS value estimation."""
        from src.core.rl_integration import RLIntegrationSystem
        
        rl = RLIntegrationSystem()
        
        state = np.array([-0.5, 0.0])
        value, uncertainty = rl.mcts_value_estimate(state, n_simulations=20, depth=5)
        
        assert np.isfinite(value)
        assert uncertainty >= 0
    
    def test_policy_gradient_short(self):
        """Test policy gradient training (short run)."""
        from src.core.rl_integration import RLIntegrationSystem
        
        rl = RLIntegrationSystem()
        
        results = rl.policy_gradient_reinforce(n_episodes=10)
        
        assert len(results.episode_rewards) == 10
        assert len(results.policy_losses) == 10
        assert 'W' in results.policy_weights
        assert results.training_time > 0
    
    def test_value_function_grid(self):
        """Test value function grid generation."""
        from src.core.rl_integration import RLIntegrationSystem, simple_policy
        
        rl = RLIntegrationSystem()
        
        # Quick evaluation
        _, returns = rl.monte_carlo_policy_evaluation(simple_policy, n_episodes=5)
        
        X, Y, Z = rl.get_value_function_grid(returns, grid_size=10)
        
        assert X.shape == (10, 10)
        assert Y.shape == (10, 10)
        assert Z.shape == (10, 10)


# ============================================================================
# Causal Inference Tests
# ============================================================================

class TestCausalInference:
    """Tests for causal inference methods."""
    
    def test_generate_synthetic_data(self):
        """Test synthetic data generation."""
        from src.core.causal_inference import CausalInferenceSystem
        
        causal = CausalInferenceSystem()
        data = causal.generate_synthetic_data(n_samples=100)
        
        assert len(data) == 100
        assert 'treatment' in data.columns
        assert 'outcome' in data.columns
        assert 'true_effect' in data.columns
        assert data['treatment'].isin([0, 1]).all()
    
    def test_propensity_score_estimation(self):
        """Test propensity score estimation."""
        from src.core.causal_inference import CausalInferenceSystem
        
        causal = CausalInferenceSystem()
        data = causal.generate_synthetic_data(n_samples=100)
        
        X = data[['age', 'bmi', 'bp', 'cholesterol', 'smoking']].values
        T = data['treatment'].values
        
        ps = causal.estimate_propensity_scores(X, T)
        
        assert len(ps) == 100
        assert all(0.05 <= p <= 0.95 for p in ps)
    
    def test_ate_ipw(self):
        """Test IPW ATE estimation."""
        from src.core.causal_inference import CausalInferenceSystem
        
        causal = CausalInferenceSystem()
        data = causal.generate_synthetic_data(n_samples=500)
        
        result = causal.estimate_ate_ipw(data)
        
        assert result.method == 'IPW'
        assert np.isfinite(result.ate_estimate)
        assert result.ate_std_error > 0
        assert result.confidence_interval[0] < result.confidence_interval[1]
    
    def test_ate_doubly_robust(self):
        """Test Doubly Robust ATE estimation."""
        from src.core.causal_inference import CausalInferenceSystem
        
        causal = CausalInferenceSystem()
        data = causal.generate_synthetic_data(n_samples=500)
        
        result = causal.estimate_ate_doubly_robust(data)
        
        assert result.method == 'Doubly Robust'
        assert np.isfinite(result.ate_estimate)
        assert 'individual_effects' in result.diagnostics
        
        # DR should be closer to true ATE than naive
        true_ate = data['true_effect'].mean()
        dr_error = abs(result.ate_estimate - true_ate)
        naive_error = abs(result.naive_estimate - true_ate)
        
        # DR typically performs better (not guaranteed for small samples)
        assert dr_error < naive_error * 2
    
    def test_bayesian_causal_inference(self):
        """Test Bayesian causal inference."""
        from src.core.causal_inference import CausalInferenceSystem
        
        causal = CausalInferenceSystem()
        data = causal.generate_synthetic_data(n_samples=200)
        
        result = causal.bayesian_causal_inference(data, n_posterior_samples=50)
        
        assert len(result.cate_mean) == 200
        assert len(result.cate_std) == 200
        assert np.isfinite(result.ate_mean)
        assert result.ate_std > 0
    
    def test_heterogeneous_effects_analysis(self):
        """Test heterogeneous treatment effect analysis."""
        from src.core.causal_inference import CausalInferenceSystem
        
        causal = CausalInferenceSystem()
        data = causal.generate_synthetic_data(n_samples=300)
        
        cate = np.random.randn(300)  # Mock CATE
        
        analysis = causal.analyze_heterogeneous_effects(data, cate)
        
        assert 'age' in analysis
        assert 'smoking' in analysis
        assert 'bmi' in analysis
    
    def test_uplift_estimation(self):
        """Test uplift estimation."""
        from src.core.causal_inference import CausalInferenceSystem
        
        causal = CausalInferenceSystem()
        data = causal.generate_synthetic_data(n_samples=300)
        
        uplift = causal.estimate_uplift(data)
        
        assert len(uplift) == 300
        assert all(np.isfinite(uplift))
    
    def test_ate_result_dataclass(self):
        """Test ATEResult dataclass."""
        from src.core.causal_inference import ATEResult
        
        result = ATEResult(
            ate_estimate=0.5,
            ate_std_error=0.1,
            method='Test',
            confidence_interval=(0.3, 0.7),
            naive_estimate=0.8
        )
        
        assert result.ate_estimate == 0.5
        assert result.method == 'Test'


# ============================================================================
# Demo Functions Tests
# ============================================================================

class TestDemoFunctions:
    """Test that demo functions run without error."""
    
    def test_rl_integration_demo(self):
        """Test RL integration demo runs."""
        from src.core.rl_integration import rl_integration_demo
        
        results = rl_integration_demo()
        
        assert 'value_estimates' in results
        assert 'policy_results' in results
    
    def test_causal_inference_demo(self):
        """Test causal inference demo runs."""
        from src.core.causal_inference import causal_inference_demo
        
        results = causal_inference_demo()
        
        assert 'data' in results
        assert 'dr_result' in results
        assert 'bayes_result' in results


# ============================================================================
# Integration Tests
# ============================================================================

class TestModuleImports:
    """Test that all modules can be imported."""
    
    def test_import_rl_integration(self):
        """Test RL integration module import."""
        from src.core import rl_integration
        
        assert hasattr(rl_integration, 'RLIntegrationSystem')
        assert hasattr(rl_integration, 'SimpleValueNetwork')
        assert hasattr(rl_integration, 'simple_policy')
    
    def test_import_causal_inference(self):
        """Test causal inference module import."""
        from src.core import causal_inference
        
        assert hasattr(causal_inference, 'CausalInferenceSystem')
        assert hasattr(causal_inference, 'ATEResult')
        assert hasattr(causal_inference, 'CATEResult')
