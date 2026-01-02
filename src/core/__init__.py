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

# Phase 9: Hardware Acceleration, PPL, Adaptive Integration
from src.core.hardware_accelerated_integration import (
    HardwareAcceleratedIntegrator,
    monte_carlo_cpu,
    NUMBA_AVAILABLE, TORCH_AVAILABLE
)
from src.core.ppl_integration import (
    BayesianRegressionBase, NumpyMCMCRegression, PPLResult,
    PYMC_AVAILABLE, TFP_AVAILABLE
)
from src.core.adaptive_integration import (
    AdaptiveIntegrator, FunctionFeatures
)

# Phase 10: RL Integration and Causal Inference
from src.core.rl_integration import (
    RLIntegrationSystem, Episode, PolicyGradientResult
)
from src.core.causal_inference import (
    CausalInferenceSystem, ATEResult, CATEResult
)

# Phase 11: GNN Integration and Explainable AI
from src.core.gnn_integration import (
    BayesianGCN, BayesianGCNLayer, GraphData, generate_synthetic_graph
)
from src.core.explainable_ai import (
    ExplainableModel, TreeSHAP, FeatureExplanation, GlobalExplanation
)

# Phase 12: Differential Privacy and Energy Efficiency
from src.core.differential_privacy import (
    DifferentiallyPrivateIntegrator, DifferentiallyPrivateBayesianQuadrature,
    PrivacyBudget, PrivateEstimate
)
from src.core.energy_efficient import (
    EnergyEfficientIntegrator, DeviceProfile, DEVICE_PROFILES
)
