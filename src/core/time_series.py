"""
Time Series & State Estimation Module
======================================

This module implements state estimation algorithms for time series
and dynamical systems from scratch using NumPy.

Methods included:
- Extended Kalman Filter (EKF)
- Unscented Kalman Filter (UKF)
- Particle Filter (Sequential Monte Carlo)
- Kalman Smoother (RTS smoother)

Industrial Applications:
- Tesla FSD: Vehicle trajectory prediction
- Boston Dynamics: Robot state estimation
- Garmin: GPS position tracking
- Pfizer: Clinical trial patient modeling
"""

import numpy as np
from typing import Callable, Tuple, List, Optional
from dataclasses import dataclass


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class GaussianState:
    """
    Gaussian state representation with mean and covariance.
    
    Attributes:
        mean: State mean vector (n,)
        cov: State covariance matrix (n, n)
    """
    mean: np.ndarray
    cov: np.ndarray
    
    @property
    def dim(self) -> int:
        return len(self.mean)
    
    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Sample from the Gaussian distribution."""
        return np.random.multivariate_normal(self.mean, self.cov, n_samples)


@dataclass 
class FilterResult:
    """
    Result container for filtering algorithms.
    
    Attributes:
        means: Filtered state means (T, n)
        covs: Filtered state covariances (T, n, n)
        log_likelihood: Log marginal likelihood of observations
    """
    means: np.ndarray
    covs: np.ndarray
    log_likelihood: float = 0.0


# =============================================================================
# EXTENDED KALMAN FILTER
# =============================================================================

class ExtendedKalmanFilter:
    """
    Extended Kalman Filter for nonlinear state estimation.
    
    Linearizes the system around the current estimate using Jacobians.
    
    State space model:
        x_{t+1} = f(x_t) + w_t,  w_t ~ N(0, Q)
        y_t = h(x_t) + v_t,      v_t ~ N(0, R)
    
    Industrial Use Case:
        Tesla Autopilot uses EKF variants for sensor fusion, combining
        lidar, camera, and radar measurements to estimate vehicle pose
        and nearby object positions.
    
    Interview Question:
        Q: When does EKF fail compared to UKF?
        A: EKF uses first-order Taylor expansion, which fails for highly
           nonlinear systems. UKF captures higher-order moments via sigma
           points without requiring Jacobian computation.
    """
    
    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        F_jacobian: Callable[[np.ndarray], np.ndarray],
        H_jacobian: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray
    ):
        """
        Initialize Extended Kalman Filter.
        
        Args:
            f: State transition function x_{t+1} = f(x_t)
            h: Observation function y_t = h(x_t)
            F_jacobian: Jacobian of f with respect to x
            H_jacobian: Jacobian of h with respect to x
            Q: Process noise covariance
            R: Observation noise covariance
        """
        self.f = f
        self.h = h
        self.F_jacobian = F_jacobian
        self.H_jacobian = H_jacobian
        self.Q = Q
        self.R = R
    
    def predict(self, state: GaussianState) -> GaussianState:
        """
        Prediction step: propagate state through dynamics.
        
        x_{t+1|t} = f(x_{t|t})
        P_{t+1|t} = F P_{t|t} F^T + Q
        
        Args:
            state: Current state estimate
        
        Returns:
            Predicted state
        """
        # Propagate mean
        mean_pred = self.f(state.mean)
        
        # Compute Jacobian at current state
        F = self.F_jacobian(state.mean)
        
        # Propagate covariance
        cov_pred = F @ state.cov @ F.T + self.Q
        
        return GaussianState(mean_pred, cov_pred)
    
    def update(self, state: GaussianState, y: np.ndarray) -> Tuple[GaussianState, float]:
        """
        Update step: incorporate observation.
        
        K = P H^T (H P H^T + R)^{-1}
        x_{t|t} = x_{t|t-1} + K (y - h(x_{t|t-1}))
        P_{t|t} = (I - K H) P_{t|t-1}
        
        Args:
            state: Predicted state
            y: Observation
        
        Returns:
            Tuple of (updated state, log likelihood of observation)
        """
        # Compute Jacobian at predicted state
        H = self.H_jacobian(state.mean)
        
        # Innovation
        y_pred = self.h(state.mean)
        innovation = y - y_pred
        
        # Innovation covariance
        S = H @ state.cov @ H.T + self.R
        
        # Kalman gain
        K = state.cov @ H.T @ np.linalg.inv(S)
        
        # Update mean and covariance
        mean_upd = state.mean + K @ innovation
        cov_upd = (np.eye(state.dim) - K @ H) @ state.cov
        
        # Log likelihood
        log_lik = -0.5 * (
            len(y) * np.log(2 * np.pi) +
            np.log(np.linalg.det(S)) +
            innovation.T @ np.linalg.inv(S) @ innovation
        )
        
        return GaussianState(mean_upd, cov_upd), log_lik
    
    def filter(
        self,
        observations: np.ndarray,
        initial_state: GaussianState
    ) -> FilterResult:
        """
        Run EKF on a sequence of observations.
        
        Args:
            observations: Observations (T, m)
            initial_state: Initial state estimate
        
        Returns:
            FilterResult with filtered states
        """
        T = len(observations)
        n = initial_state.dim
        
        means = np.zeros((T, n))
        covs = np.zeros((T, n, n))
        total_log_lik = 0.0
        
        state = initial_state
        
        for t, y in enumerate(observations):
            # Predict
            state = self.predict(state)
            
            # Update
            state, log_lik = self.update(state, y)
            
            means[t] = state.mean
            covs[t] = state.cov
            total_log_lik += log_lik
        
        return FilterResult(means, covs, total_log_lik)


# =============================================================================
# UNSCENTED KALMAN FILTER
# =============================================================================

class UnscentedKalmanFilter:
    """
    Unscented Kalman Filter for nonlinear state estimation.
    
    Uses sigma points to capture mean and covariance through nonlinear
    transformations without requiring Jacobians.
    
    Industrial Use Case:
        Boston Dynamics' Spot robot uses UKF for state estimation,
        fusing IMU, joint encoders, and camera data for stable locomotion.
    
    Interview Question:
        Q: Why 2n+1 sigma points in UKF?
        A: One point at the mean, plus two points along each of the n
           principal axes (positive and negative). This captures the
           mean and covariance exactly for linear transforms.
    """
    
    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        Q: np.ndarray,
        R: np.ndarray,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0
    ):
        """
        Initialize Unscented Kalman Filter.
        
        Args:
            f: State transition function
            h: Observation function
            Q: Process noise covariance
            R: Observation noise covariance
            alpha: Spread of sigma points (small positive, e.g., 1e-3)
            beta: Prior knowledge about distribution (2 for Gaussian)
            kappa: Secondary scaling parameter (typically 0)
        """
        self.f = f
        self.h = h
        self.Q = Q
        self.R = R
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
    
    def _compute_sigma_points(self, state: GaussianState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute sigma points and weights.
        
        Returns:
            Tuple of (sigma_points, weights_mean, weights_cov)
        """
        n = state.dim
        lam = self.alpha**2 * (n + self.kappa) - n
        
        # Compute sigma points
        sqrt_matrix = np.linalg.cholesky((n + lam) * state.cov)
        
        sigma_points = np.zeros((2*n + 1, n))
        sigma_points[0] = state.mean
        
        for i in range(n):
            sigma_points[i + 1] = state.mean + sqrt_matrix[i]
            sigma_points[n + i + 1] = state.mean - sqrt_matrix[i]
        
        # Weights for mean
        Wm = np.full(2*n + 1, 1 / (2 * (n + lam)))
        Wm[0] = lam / (n + lam)
        
        # Weights for covariance
        Wc = Wm.copy()
        Wc[0] += (1 - self.alpha**2 + self.beta)
        
        return sigma_points, Wm, Wc
    
    def predict(self, state: GaussianState) -> GaussianState:
        """
        Prediction step using sigma points.
        
        Args:
            state: Current state estimate
        
        Returns:
            Predicted state
        """
        sigma_points, Wm, Wc = self._compute_sigma_points(state)
        
        # Propagate sigma points through dynamics
        sigma_points_pred = np.array([self.f(sp) for sp in sigma_points])
        
        # Compute predicted mean
        mean_pred = np.sum(Wm[:, None] * sigma_points_pred, axis=0)
        
        # Compute predicted covariance
        cov_pred = np.zeros((state.dim, state.dim))
        for i, sp in enumerate(sigma_points_pred):
            diff = sp - mean_pred
            cov_pred += Wc[i] * np.outer(diff, diff)
        cov_pred += self.Q
        
        return GaussianState(mean_pred, cov_pred)
    
    def update(self, state: GaussianState, y: np.ndarray) -> Tuple[GaussianState, float]:
        """
        Update step using sigma points.
        
        Args:
            state: Predicted state
            y: Observation
        
        Returns:
            Tuple of (updated state, log likelihood)
        """
        sigma_points, Wm, Wc = self._compute_sigma_points(state)
        
        # Propagate sigma points through observation function
        sigma_obs = np.array([self.h(sp) for sp in sigma_points])
        
        # Compute predicted observation mean
        y_pred = np.sum(Wm[:, None] * sigma_obs, axis=0)
        
        # Compute innovation covariance
        S = np.zeros((len(y), len(y)))
        for i, so in enumerate(sigma_obs):
            diff = so - y_pred
            S += Wc[i] * np.outer(diff, diff)
        S += self.R
        
        # Compute cross covariance
        Pxy = np.zeros((state.dim, len(y)))
        for i, (sp, so) in enumerate(zip(sigma_points, sigma_obs)):
            Pxy += Wc[i] * np.outer(sp - state.mean, so - y_pred)
        
        # Kalman gain
        K = Pxy @ np.linalg.inv(S)
        
        # Update
        innovation = y - y_pred
        mean_upd = state.mean + K @ innovation
        cov_upd = state.cov - K @ S @ K.T
        
        # Log likelihood
        log_lik = -0.5 * (
            len(y) * np.log(2 * np.pi) +
            np.log(np.linalg.det(S)) +
            innovation.T @ np.linalg.inv(S) @ innovation
        )
        
        return GaussianState(mean_upd, cov_upd), log_lik
    
    def filter(
        self,
        observations: np.ndarray,
        initial_state: GaussianState
    ) -> FilterResult:
        """
        Run UKF on a sequence of observations.
        
        Args:
            observations: Observations (T, m)
            initial_state: Initial state estimate
        
        Returns:
            FilterResult with filtered states
        """
        T = len(observations)
        n = initial_state.dim
        
        means = np.zeros((T, n))
        covs = np.zeros((T, n, n))
        total_log_lik = 0.0
        
        state = initial_state
        
        for t, y in enumerate(observations):
            state = self.predict(state)
            state, log_lik = self.update(state, y)
            
            means[t] = state.mean
            covs[t] = state.cov
            total_log_lik += log_lik
        
        return FilterResult(means, covs, total_log_lik)


# =============================================================================
# PARTICLE FILTER
# =============================================================================

class ParticleFilter:
    """
    Particle Filter (Sequential Monte Carlo) for state estimation.
    
    Represents the posterior with a weighted set of samples (particles).
    Handles multimodal distributions and non-Gaussian noise.
    
    Industrial Use Case:
        Waymo's self-driving cars use particle filters for localization,
        handling the multimodal nature of position estimates in GPS-denied
        environments like tunnels or urban canyons.
    
    Interview Question:
        Q: When do particle filters struggle?
        A: High-dimensional state spaces (particle degeneracy), and when
           the proposal distribution is far from the optimal proposal.
           Typically limited to <20 state dimensions.
    """
    
    def __init__(
        self,
        f: Callable[[np.ndarray], np.ndarray],
        h: Callable[[np.ndarray], np.ndarray],
        process_noise_sampler: Callable[[int], np.ndarray],
        observation_log_likelihood: Callable[[np.ndarray, np.ndarray], float],
        n_particles: int = 1000
    ):
        """
        Initialize Particle Filter.
        
        Args:
            f: State transition function x_{t+1} = f(x_t)
            h: Observation function (for visualization only)
            process_noise_sampler: Function to sample process noise
            observation_log_likelihood: log p(y|x) function
            n_particles: Number of particles
        """
        self.f = f
        self.h = h
        self.process_noise_sampler = process_noise_sampler
        self.observation_log_likelihood = observation_log_likelihood
        self.n_particles = n_particles
    
    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        """
        Systematic resampling to avoid particle degeneracy.
        
        More efficient than multinomial resampling with lower variance.
        
        Args:
            weights: Normalized weights summing to 1
        
        Returns:
            Indices of resampled particles
        """
        n = len(weights)
        positions = (np.arange(n) + np.random.random()) / n
        
        cumsum = np.cumsum(weights)
        indices = np.zeros(n, dtype=int)
        
        i, j = 0, 0
        while i < n:
            if positions[i] < cumsum[j]:
                indices[i] = j
                i += 1
            else:
                j += 1
        
        return indices
    
    def _effective_sample_size(self, weights: np.ndarray) -> float:
        """
        Compute effective sample size.
        
        ESS = 1 / Σ w_i² 
        
        Low ESS indicates particle degeneracy.
        """
        return 1.0 / np.sum(weights**2)
    
    def filter(
        self,
        observations: np.ndarray,
        initial_particles: np.ndarray,
        resample_threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, List[float]]:
        """
        Run particle filter on observations.
        
        Args:
            observations: Observations (T, m)
            initial_particles: Initial particles (n_particles, n)
            resample_threshold: Resample when ESS < threshold * n_particles
        
        Returns:
            Tuple of (particle_means, particle_vars, effective_sample_sizes)
        """
        T = len(observations)
        n = initial_particles.shape[1]
        
        particles = initial_particles.copy()
        weights = np.ones(self.n_particles) / self.n_particles
        
        means = np.zeros((T, n))
        variances = np.zeros((T, n))
        ess_history = []
        
        for t, y in enumerate(observations):
            # Propagate particles through dynamics
            noise = self.process_noise_sampler(self.n_particles)
            particles = np.array([self.f(p) for p in particles]) + noise
            
            # Update weights based on observation likelihood
            log_weights = np.array([
                self.observation_log_likelihood(y, p) for p in particles
            ])
            
            # Normalize weights using log-sum-exp trick for stability
            max_log_weight = np.max(log_weights)
            weights = np.exp(log_weights - max_log_weight)
            weights /= np.sum(weights)
            
            # Compute effective sample size
            ess = self._effective_sample_size(weights)
            ess_history.append(ess)
            
            # Resample if ESS is too low
            if ess < resample_threshold * self.n_particles:
                indices = self._systematic_resample(weights)
                particles = particles[indices]
                weights = np.ones(self.n_particles) / self.n_particles
            
            # Compute weighted mean and variance
            means[t] = np.sum(weights[:, None] * particles, axis=0)
            variances[t] = np.sum(
                weights[:, None] * (particles - means[t])**2, axis=0
            )
        
        return means, variances, ess_history


# =============================================================================
# RAUCH-TUNG-STRIEBEL SMOOTHER
# =============================================================================

def rts_smoother(
    filter_means: np.ndarray,
    filter_covs: np.ndarray,
    f: Callable[[np.ndarray], np.ndarray],
    F_jacobian: Callable[[np.ndarray], np.ndarray],
    Q: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rauch-Tung-Striebel (RTS) smoother for improved state estimates.
    
    Runs a backward pass after filtering to incorporate future observations.
    
    x_{t|T} = x_{t|t} + G_t (x_{t+1|T} - x_{t+1|t})
    P_{t|T} = P_{t|t} + G_t (P_{t+1|T} - P_{t+1|t}) G_t^T
    
    where G_t = P_{t|t} F_t^T P_{t+1|t}^{-1}
    
    Industrial Use Case:
        Garmin GPS receivers use RTS smoothing for track analysis after
        a workout, providing smoother and more accurate path reconstruction.
    
    Args:
        filter_means: Forward filter means (T, n)
        filter_covs: Forward filter covariances (T, n, n)
        f: State transition function
        F_jacobian: Jacobian of f
        Q: Process noise covariance
    
    Returns:
        Tuple of (smoothed_means, smoothed_covs)
    """
    T, n = filter_means.shape
    
    smoothed_means = filter_means.copy()
    smoothed_covs = filter_covs.copy()
    
    for t in range(T - 2, -1, -1):
        # Predicted state at t+1 from state at t
        F = F_jacobian(filter_means[t])
        pred_mean = f(filter_means[t])
        pred_cov = F @ filter_covs[t] @ F.T + Q
        
        # Smoother gain
        G = filter_covs[t] @ F.T @ np.linalg.inv(pred_cov)
        
        # Smoothed estimates
        smoothed_means[t] = filter_means[t] + G @ (smoothed_means[t+1] - pred_mean)
        smoothed_covs[t] = filter_covs[t] + G @ (smoothed_covs[t+1] - pred_cov) @ G.T
    
    return smoothed_means, smoothed_covs


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_linear_system(
    A: np.ndarray,
    C: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray
) -> Tuple[ExtendedKalmanFilter, UnscentedKalmanFilter]:
    """
    Create EKF and UKF for a linear system for comparison.
    
    State space:
        x_{t+1} = A x_t + w_t
        y_t = C x_t + v_t
    
    Args:
        A: State transition matrix
        C: Observation matrix
        Q: Process noise covariance
        R: Observation noise covariance
    
    Returns:
        Tuple of (EKF, UKF)
    """
    f = lambda x: A @ x
    h = lambda x: C @ x
    F_jac = lambda x: A
    H_jac = lambda x: C
    
    ekf = ExtendedKalmanFilter(f, h, F_jac, H_jac, Q, R)
    ukf = UnscentedKalmanFilter(f, h, Q, R)
    
    return ekf, ukf


def simulate_system(
    f: Callable[[np.ndarray], np.ndarray],
    h: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    T: int,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate a state space system.
    
    Args:
        f: State transition
        h: Observation function
        x0: Initial state
        Q: Process noise covariance
        R: Observation noise covariance
        T: Number of time steps
        seed: Random seed
    
    Returns:
        Tuple of (true_states, observations)
    """
    if seed is not None:
        np.random.seed(seed)
    
    n = len(x0)
    m = len(R)
    
    states = np.zeros((T, n))
    observations = np.zeros((T, m))
    
    x = x0.copy()
    
    for t in range(T):
        # Process noise
        w = np.random.multivariate_normal(np.zeros(n), Q)
        x = f(x) + w
        states[t] = x
        
        # Observation noise
        v = np.random.multivariate_normal(np.zeros(m), R)
        observations[t] = h(x) + v
    
    return states, observations


def compare_filters(
    true_states: np.ndarray,
    ekf_result: FilterResult,
    ukf_result: FilterResult
) -> dict:
    """
    Compare EKF and UKF performance.
    
    Args:
        true_states: Ground truth states
        ekf_result: EKF filter result
        ukf_result: UKF filter result
    
    Returns:
        Dictionary with RMSE for each filter
    """
    ekf_rmse = np.sqrt(np.mean((ekf_result.means - true_states)**2))
    ukf_rmse = np.sqrt(np.mean((ukf_result.means - true_states)**2))
    
    return {
        'ekf_rmse': ekf_rmse,
        'ukf_rmse': ukf_rmse,
        'ekf_log_likelihood': ekf_result.log_likelihood,
        'ukf_log_likelihood': ukf_result.log_likelihood
    }
