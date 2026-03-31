"""
Module 2.8: New Trends

Cutting-edge LLM techniques:
- Model merging (SLERP, DARE, TIES)
- Multimodal models
- Interpretability
- Test-time compute
"""

from .model_merging import (
    ModelMerger,
    SLERPMerger,
    DAREMerger,
    TIESMerger,
    TaskArithmetic,
    ModelSoups,
)
from .multimodal import (
    CLIPModel,
    LLaVAModel,
    VisionLanguageConfig,
    MultimodalProcessor,
)
from .interpretability import (
    SparseAutoencoder,
    FeatureVisualizer,
    ActivationPatcher,
    InterpretabilityAnalyzer,
)
from .test_time_compute import (
    ChainOfThoughtGenerator,
    SelfConsistency,
    MajorityVoting,
    VerificationModule,
    TestTimeCompute,
)

__all__ = [
    # Model Merging
    "ModelMerger",
    "SLERPMerger",
    "DAREMerger",
    "TIESMerger",
    "TaskArithmetic",
    "ModelSoups",
    # Multimodal
    "CLIPModel",
    "LLaVAModel",
    "VisionLanguageConfig",
    "MultimodalProcessor",
    # Interpretability
    "SparseAutoencoder",
    "FeatureVisualizer",
    "ActivationPatcher",
    "InterpretabilityAnalyzer",
    # Test-Time Compute
    "ChainOfThoughtGenerator",
    "SelfConsistency",
    "MajorityVoting",
    "VerificationModule",
    "TestTimeCompute",
]
