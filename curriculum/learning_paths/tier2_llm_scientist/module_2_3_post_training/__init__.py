"""
Module 2.3: Post-Training Datasets

Dataset preparation for post-training:
- Data formats (ShareGPT, ChatML, Alpaca)
- Synthetic data generation
- Data enhancement (CoT, BSM, self-reflection)
- Quality filtering
"""

from .formats import (
    ShareGPTFormat,
    ChatMLFormat,
    AlpacaFormat,
    ConversationTemplate,
    FormatConverter,
)
from .synthetic_data import (
    InstructionGenerator,
    SeedTaskGenerator,
    SelfInstruct,
    SyntheticDataPipeline,
)
from .enhancement import (
    ChainOfThoughtGenerator,
    BranchSolveMerge,
    SelfReflection,
    SelfCorrection,
)
from .quality_filtering import (
    RewardModelFilter,
    PerplexityFilter,
    DiversityScorer,
    QualityFilterPipeline,
)

__all__ = [
    # Formats
    "ShareGPTFormat",
    "ChatMLFormat",
    "AlpacaFormat",
    "ConversationTemplate",
    "FormatConverter",
    # Synthetic Data
    "InstructionGenerator",
    "SeedTaskGenerator",
    "SelfInstruct",
    "SyntheticDataPipeline",
    # Enhancement
    "ChainOfThoughtGenerator",
    "BranchSolveMerge",
    "SelfReflection",
    "SelfCorrection",
    # Quality Filtering
    "RewardModelFilter",
    "PerplexityFilter",
    "DiversityScorer",
    "QualityFilterPipeline",
]
