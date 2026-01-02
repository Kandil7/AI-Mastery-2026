# Core mathematical foundations module

# Integration methods
from src.core.integration import (
    trapezoidal_rule, simpsons_rule, adaptive_quadrature,
    gauss_legendre, gauss_hermite_expectation,
    monte_carlo_integrate, importance_sampling, stratified_sampling,
    BayesianQuadrature, rbf_kernel
)

# Normalizing flows
from src.core.normalizing_flows import (
    PlanarFlow, RadialFlow, FlowChain,
    gaussian_base_log_prob, gaussian_base_sampler
)

# Time series & state estimation
from src.core.time_series import (
    GaussianState, FilterResult,
    ExtendedKalmanFilter, UnscentedKalmanFilter,
    ParticleFilter, rts_smoother,
    create_linear_system, simulate_system, compare_filters
)

# Optimization (from optimization.py)
from src.core.optimization import (
    Optimizer, GradientDescent, Momentum, Adam,
    RMSprop, AdaGrad, NAdam,
    LearningRateScheduler, StepDecay, ExponentialDecay,
    CosineAnnealing, WarmupScheduler,
    minimize, lagrange_multipliers, newton_raphson
)