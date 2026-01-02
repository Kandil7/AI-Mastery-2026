"""
Integration in Reinforcement Learning

This module implements integration methods for reinforcement learning,
including Monte Carlo policy evaluation, policy gradients, and MCTS concepts.

Industrial Case Study: DeepMind's AlphaGo/AlphaZero
- Challenge: Go has ~10^170 possible states (exhaustive search impossible)
- Solution: Monte Carlo Tree Search + Neural Networks + Bayesian integration
- Result: Defeated world champion Lee Sedol (2016), superhuman in Go/Chess/Shogi
- Impact: Logistics optimization saving $200M/year at Alphabet
"""

import numpy as np
from typing import Callable, Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False
        warnings.warn("Gymnasium/gym not available. RL simulations disabled.")


@dataclass
class Episode:
    """Represents a single RL episode."""
    states: List[np.ndarray]
    actions: List[float]
    rewards: List[float]
    total_reward: float
    length: int


@dataclass
class PolicyGradientResult:
    """Results from policy gradient training."""
    episode_rewards: List[float]
    policy_losses: List[float]
    policy_weights: Dict[str, np.ndarray]
    training_time: float


class SimpleValueNetwork:
    """
    Simple neural network for value function approximation.
    
    Architecture: input -> hidden (tanh) -> output
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, learning_rate: float = 0.001):
        """Initialize the value network."""
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        
        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(1)
        
        # Cache for backward pass
        self._cache = {}
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the network."""
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        self._cache['x'] = x
        self._cache['z1'] = x @ self.W1 + self.b1
        self._cache['a1'] = np.tanh(self._cache['z1'])
        self._cache['z2'] = self._cache['a1'] @ self.W2 + self.b2
        
        return self._cache['z2']
    
    def backward(self, grad_output: np.ndarray) -> None:
        """Backward pass to update weights."""
        if isinstance(grad_output, (int, float)):
            grad_output = np.array([[grad_output]])
        elif grad_output.ndim == 1:
            grad_output = grad_output.reshape(-1, 1)
        
        # Gradients for layer 2
        grad_W2 = self._cache['a1'].T @ grad_output
        grad_b2 = np.sum(grad_output, axis=0)
        
        # Gradients for layer 1
        grad_a1 = grad_output @ self.W2.T
        grad_z1 = grad_a1 * (1 - np.tanh(self._cache['z1'])**2)  # tanh derivative
        grad_W1 = self._cache['x'].T @ grad_z1
        grad_b1 = np.sum(grad_z1, axis=0)
        
        # Update weights
        self.W2 -= self.learning_rate * grad_W2
        self.b2 -= self.learning_rate * grad_b2
        self.W1 -= self.learning_rate * grad_W1
        self.b1 -= self.learning_rate * grad_b1
    
    def predict(self, x: np.ndarray) -> float:
        """Get scalar value prediction."""
        return self.forward(x).flatten()[0]


class RLIntegrationSystem:
    """
    Reinforcement Learning system with integration methods.
    
    Demonstrates how integration is used in RL for:
    1. Monte Carlo policy evaluation (averaging returns)
    2. Policy gradient estimation (integrating over trajectories)
    3. Value function approximation
    
    The key RL objective involves integration over trajectories:
    
    J(π) = E[∑ γ^t r(s_t, a_t)] = ∫ p_π(τ) R(τ) dτ
    
    where τ is a trajectory and p_π(τ) is its probability under policy π.
    
    Example:
        >>> rl = RLIntegrationSystem(state_dim=2, action_dim=1)
        >>> values, returns = rl.monte_carlo_policy_evaluation(my_policy, n_episodes=100)
    """
    
    def __init__(self, state_dim: int = 2, action_dim: int = 1, 
                 env_name: str = 'MountainCarContinuous-v0'):
        """
        Initialize the RL system.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            env_name: Name of the gym environment (if available)
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.env_name = env_name
        self.env = None
        
        if GYM_AVAILABLE:
            try:
                self.env = gym.make(env_name)
                self.state_dim = self.env.observation_space.shape[0]
                if hasattr(self.env.action_space, 'shape'):
                    self.action_dim = self.env.action_space.shape[0]
            except Exception as e:
                warnings.warn(f"Could not create environment {env_name}: {e}")
        
        # Value function approximator
        self.value_network = SimpleValueNetwork(self.state_dim)
    
    def _simulate_step(self, state: np.ndarray, action: float) -> Tuple[np.ndarray, float, bool]:
        """
        Simulate a step in a simple environment (fallback if gym unavailable).
        
        Simulates a simplified mountain car dynamics.
        """
        position, velocity = state
        
        # Physics simulation
        force = np.clip(action, -1, 1)
        gravity = 0.0025
        
        velocity += force * 0.001 - np.cos(3 * position) * gravity
        velocity = np.clip(velocity, -0.07, 0.07)
        position += velocity
        position = np.clip(position, -1.2, 0.6)
        
        # Terminal conditions
        done = position >= 0.45
        
        # Reward
        reward = -1.0 + (100.0 if done else 0.0)
        
        return np.array([position, velocity]), reward, done
    
    def run_episode(self, policy: Callable[[np.ndarray], float], 
                    max_steps: int = 200) -> Episode:
        """
        Run a single episode using the given policy.
        
        Args:
            policy: Function mapping state to action
            max_steps: Maximum steps per episode
            
        Returns:
            Episode dataclass with trajectory data
        """
        states, actions, rewards = [], [], []
        
        # Initialize
        if self.env is not None:
            state, _ = self.env.reset()
        else:
            state = np.array([-0.5 + np.random.uniform(-0.1, 0.1), 0.0])
        
        done = False
        step = 0
        
        while not done and step < max_steps:
            states.append(state.copy())
            action = policy(state)
            actions.append(action)
            
            # Step environment
            if self.env is not None:
                action_input = [action] if self.action_dim == 1 else action
                next_state, reward, terminated, truncated, _ = self.env.step(action_input)
                done = terminated or truncated
            else:
                next_state, reward, done = self._simulate_step(state, action)
            
            rewards.append(reward)
            state = next_state
            step += 1
        
        return Episode(
            states=states,
            actions=actions,
            rewards=rewards,
            total_reward=sum(rewards),
            length=len(rewards)
        )
    
    def compute_returns(self, rewards: List[float], gamma: float = 0.99) -> List[float]:
        """
        Compute discounted returns G_t = ∑_{k=0}^∞ γ^k r_{t+k}.
        
        This is Monte Carlo integration of future rewards.
        
        Args:
            rewards: List of rewards from an episode
            gamma: Discount factor
            
        Returns:
            List of returns for each timestep
        """
        returns = []
        G = 0
        
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        
        return returns
    
    def monte_carlo_policy_evaluation(self, policy: Callable[[np.ndarray], float],
                                       n_episodes: int = 500,
                                       gamma: float = 0.99) -> Tuple[List[float], Dict]:
        """
        Estimate value function using Monte Carlo integration.
        
        Monte Carlo policy evaluation estimates V(s) by averaging returns:
        
        V(s) ≈ (1/N) ∑_{i=1}^N G_i(s)
        
        This is Monte Carlo integration of the value integral:
        V(s) = E[G | S_0 = s] = ∫ G · p(G|s) dG
        
        Args:
            policy: Policy function π(s) -> a
            n_episodes: Number of episodes for estimation
            gamma: Discount factor
            
        Returns:
            Tuple of (value_estimates, state_returns_dict)
        """
        returns_by_state = {}
        value_estimates = []
        episode_rewards = []
        
        for episode_idx in range(n_episodes):
            # Run episode
            episode = self.run_episode(policy)
            episode_rewards.append(episode.total_reward)
            
            # Compute returns
            returns = self.compute_returns(episode.rewards, gamma)
            
            # Update state-return estimates (tabular)
            for state, G in zip(episode.states, returns):
                # Discretize state for tabular tracking
                state_key = tuple(np.round(state, 2))
                if state_key not in returns_by_state:
                    returns_by_state[state_key] = []
                returns_by_state[state_key].append(G)
            
            # Track average value estimate
            if episode_idx % 10 == 0:
                avg_value = np.mean([np.mean(Gs) for Gs in returns_by_state.values()])
                value_estimates.append(avg_value)
        
        return value_estimates, returns_by_state
    
    def policy_gradient_reinforce(self, n_episodes: int = 300,
                                   gamma: float = 0.99,
                                   learning_rate: float = 0.01) -> PolicyGradientResult:
        """
        Train policy using REINFORCE algorithm (Monte Carlo policy gradient).
        
        REINFORCE uses the policy gradient theorem:
        
        ∇J(θ) = E[∑_t ∇log π_θ(a_t|s_t) · G_t]
        
        This is estimated via Monte Carlo integration over trajectories.
        
        Args:
            n_episodes: Number of training episodes
            gamma: Discount factor
            learning_rate: Learning rate for policy updates
            
        Returns:
            PolicyGradientResult with training history
        """
        import time
        start_time = time.perf_counter()
        
        # Initialize policy (simple linear policy)
        policy_W = np.random.randn(self.state_dim, self.action_dim) * 0.1
        policy_b = np.zeros(self.action_dim)
        
        def policy(state: np.ndarray) -> float:
            """Gaussian policy: mean from linear model, fixed std."""
            mean = state @ policy_W + policy_b
            return np.tanh(mean[0])  # Bound action to [-1, 1]
        
        episode_rewards = []
        policy_losses = []
        
        for episode_idx in range(n_episodes):
            # Run episode
            episode = self.run_episode(policy)
            episode_rewards.append(episode.total_reward)
            
            # Compute returns
            returns = self.compute_returns(episode.rewards, gamma)
            
            # Normalize returns (variance reduction)
            returns_arr = np.array(returns)
            returns_norm = (returns_arr - np.mean(returns_arr)) / (np.std(returns_arr) + 1e-8)
            
            # Compute policy gradient and update
            episode_loss = 0
            for t, (state, action, G) in enumerate(zip(episode.states, episode.actions, returns_norm)):
                # Baseline: use value function
                baseline = self.value_network.predict(state)
                advantage = G - baseline
                
                # Policy gradient: ∇log π(a|s) · advantage
                # For linear policy, ∇log π ∝ state
                grad = advantage * state
                
                # Update policy weights
                policy_W += learning_rate * grad.reshape(-1, 1)
                
                episode_loss += np.abs(advantage)
                
                # Update value function
                value_pred = self.value_network.forward(state)
                value_grad = value_pred - G
                self.value_network.backward(value_grad)
            
            policy_losses.append(episode_loss)
            
            # Progress
            if (episode_idx + 1) % 50 == 0:
                avg_reward = np.mean(episode_rewards[-50:])
                print(f"Episode {episode_idx + 1}: Avg Reward = {avg_reward:.2f}")
        
        elapsed = time.perf_counter() - start_time
        
        return PolicyGradientResult(
            episode_rewards=episode_rewards,
            policy_losses=policy_losses,
            policy_weights={'W': policy_W, 'b': policy_b},
            training_time=elapsed
        )
    
    def mcts_value_estimate(self, state: np.ndarray, 
                            n_simulations: int = 100,
                            depth: int = 10,
                            gamma: float = 0.99) -> Tuple[float, float]:
        """
        Estimate state value using Monte Carlo Tree Search concept.
        
        MCTS uses Monte Carlo integration to estimate Q-values:
        
        Q(s,a) = (1/N) ∑_{i=1}^N G_i(s,a) + c·P(s,a)·√(∑N(s,b))/(1+N(s,a))
        
        This is the approach used in AlphaGo/AlphaZero.
        
        Args:
            state: State to evaluate
            n_simulations: Number of Monte Carlo simulations
            depth: Maximum simulation depth
            gamma: Discount factor
            
        Returns:
            Tuple of (estimated_value, uncertainty)
        """
        values = []
        
        for _ in range(n_simulations):
            # Random rollout from state
            current_state = state.copy()
            total_return = 0
            discount = 1.0
            
            for step in range(depth):
                # Random action
                action = np.random.uniform(-1, 1)
                
                # Simulate step
                if self.env is not None:
                    # Reset env to current state (approximation)
                    next_state, reward, done = self._simulate_step(current_state, action)
                else:
                    next_state, reward, done = self._simulate_step(current_state, action)
                
                total_return += discount * reward
                discount *= gamma
                
                if done:
                    break
                    
                current_state = next_state
            
            values.append(total_return)
        
        return np.mean(values), np.std(values)
    
    def get_value_function_grid(self, returns_by_state: Dict,
                                 position_range: Tuple[float, float] = (-1.2, 0.6),
                                 velocity_range: Tuple[float, float] = (-0.07, 0.07),
                                 grid_size: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a grid of value function estimates for visualization.
        
        Args:
            returns_by_state: Dictionary from MC policy evaluation
            position_range: Range for position axis
            velocity_range: Range for velocity axis
            grid_size: Number of points per dimension
            
        Returns:
            Tuple of (X_grid, Y_grid, Value_grid)
        """
        positions = np.linspace(*position_range, grid_size)
        velocities = np.linspace(*velocity_range, grid_size)
        X, Y = np.meshgrid(positions, velocities)
        Z = np.zeros_like(X)
        
        for i in range(grid_size):
            for j in range(grid_size):
                state_key = (round(X[i, j], 2), round(Y[i, j], 2))
                if state_key in returns_by_state:
                    Z[i, j] = np.mean(returns_by_state[state_key])
                else:
                    # Interpolate using value network
                    state = np.array([X[i, j], Y[i, j]])
                    Z[i, j] = self.value_network.predict(state)
        
        return X, Y, Z


def simple_policy(state: np.ndarray) -> float:
    """
    Simple heuristic policy for Mountain Car.
    
    Push right when low position or positive velocity.
    """
    position, velocity = state
    if position < -0.4:
        return 0.5  # Push right
    elif velocity < 0:
        return 0.3  # Light push right
    else:
        return -0.3  # Push left (build momentum)


def rl_integration_demo():
    """
    Demonstrate RL integration capabilities.
    
    Industrial Case Study: DeepMind AlphaGo/AlphaZero
    - Used MCTS with neural networks for state evaluation
    - Monte Carlo integration for Q-value estimation
    - Defeated world champion, achieved superhuman performance
    - Applied to logistics: $200M/year savings at Alphabet
    """
    print("=" * 60)
    print("Integration in Reinforcement Learning")
    print("=" * 60)
    print("\nIndustrial Case Study: DeepMind AlphaGo/AlphaZero")
    print("- Challenge: Go has ~10^170 possible states")
    print("- Solution: MCTS + Neural Networks + Monte Carlo integration")
    print("- Result: Superhuman performance in Go, Chess, Shogi")
    print("- Impact: $200M/year logistics savings at Alphabet\n")
    
    # Create RL system
    rl = RLIntegrationSystem()
    
    # Monte Carlo Policy Evaluation
    print("=" * 60)
    print("1. Monte Carlo Policy Evaluation")
    print("=" * 60)
    
    value_estimates, returns_by_state = rl.monte_carlo_policy_evaluation(
        simple_policy, n_episodes=100
    )
    
    print(f"Evaluated {len(returns_by_state)} unique states")
    print(f"Final average value estimate: {value_estimates[-1]:.2f}")
    
    # Policy Gradient Training
    print("\n" + "=" * 60)
    print("2. Policy Gradient Training (REINFORCE)")
    print("=" * 60)
    
    results = rl.policy_gradient_reinforce(n_episodes=100)
    
    print(f"\nTraining completed in {results.training_time:.2f}s")
    print(f"Final episode reward: {results.episode_rewards[-1]:.2f}")
    print(f"Average last 10 rewards: {np.mean(results.episode_rewards[-10:]):.2f}")
    
    # MCTS Value Estimation
    print("\n" + "=" * 60)
    print("3. MCTS Value Estimation")
    print("=" * 60)
    
    test_state = np.array([-0.5, 0.0])
    value, uncertainty = rl.mcts_value_estimate(test_state, n_simulations=50)
    
    print(f"State: position={test_state[0]:.2f}, velocity={test_state[1]:.2f}")
    print(f"Estimated value: {value:.2f} ± {uncertainty:.2f}")
    
    return {
        'value_estimates': value_estimates,
        'returns_by_state': returns_by_state,
        'policy_results': results
    }


# Module exports
__all__ = [
    'RLIntegrationSystem',
    'SimpleValueNetwork',
    'Episode',
    'PolicyGradientResult',
    'simple_policy',
    'rl_integration_demo',
    'GYM_AVAILABLE',
]
