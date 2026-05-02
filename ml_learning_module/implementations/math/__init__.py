"""
ML Learning Module - Package Initialization

This package contains complete implementations of ML algorithms
with full explanations and visualizations.

Author: AI-Mastery-2026
"""

# Import all modules
from .vectors import VectorOperations
from .matrices import MatrixOperations
from .calculus import CalculusOperations, Optimizer
from .probability import Distribution, ProbabilityOperations, HypothesisTesting

__version__ = "1.0.0"
__author__ = "AI-Mastery-2026"

__all__ = [
    # Vector operations
    "VectorOperations",
    # Matrix operations
    "MatrixOperations",
    # Calculus and optimization
    "CalculusOperations",
    "Optimizer",
    # Probability
    "Distribution",
    "ProbabilityOperations",
    "HypothesisTesting",
]

print("✅ ML Learning Module loaded successfully!")
print("   Available: VectorOperations, MatrixOperations, CalculusOperations,")
print("              Optimizer, ProbabilityOperations, Distribution")
