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

# MCMC Methods
from src.core.mcmc import (
    MCMCResult,
    metropolis_hastings, HamiltonianMonteCarlo, nuts_sampler,
    effective_sample_size, gelman_rubin_diagnostic, mcmc_diagnostics,
    autocorrelation, thinning, trace_plot_data,
    bayesian_logistic_regression_hmc
)

# Advanced Integration
from src.core.advanced_integration import (
    NeuralODE,
    ODEFunc,
    MultiModalIntegrator,
    FederatedIntegrator,
    biased_lending_simulation
)

# Variational Inference
from src.core.variational_inference import (
    VIResult, GaussianVariational,
    compute_elbo, compute_elbo_gradient,
    MeanFieldVI, StochasticVI, coordinate_ascent_vi,
    BayesianLinearRegressionVI, svgd
)
