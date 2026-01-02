# Core mathematical foundations module

# Integration methods (NEW)
from src.core.integration import (
    trapezoidal_rule, simpsons_rule, adaptive_quadrature,
    gauss_legendre, gauss_hermite_expectation,
    monte_carlo_integrate, importance_sampling, stratified_sampling,
    BayesianQuadrature, rbf_kernel
)

# Normalizing flows (NEW)
from src.core.normalizing_flows import (
    PlanarFlow, RadialFlow, FlowChain,
    gaussian_base_log_prob, gaussian_base_sampler
)