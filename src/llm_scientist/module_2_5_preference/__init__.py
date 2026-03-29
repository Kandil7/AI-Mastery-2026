"""
Module 2.5: Preference Alignment

Preference alignment components:
- Rejection sampling
- Direct Preference Optimization (DPO)
- RLHF with PPO
- Reward modeling
"""

from .rejection_sampling import (
    RejectionSampler,
    MultiResponseGenerator,
    ResponseScorer,
    PreferencePairCreator,
)
from .dpo import (
    DPOConfig,
    DPOLoss,
    DPOTrainer,
    DPOPipeline,
)
from .rlhf import (
    PPOConfig,
    PPOTrainer,
    RewardModel,
    ValueModel,
    RLHFPipeline,
)
from .reward_modeling import (
    RewardModelConfig,
    RewardModelTrainer,
    BradleyTerryModel,
    PairwiseDataset,
)

__all__ = [
    # Rejection Sampling
    "RejectionSampler",
    "MultiResponseGenerator",
    "ResponseScorer",
    "PreferencePairCreator",
    # DPO
    "DPOConfig",
    "DPOLoss",
    "DPOTrainer",
    "DPOPipeline",
    # RLHF
    "PPOConfig",
    "PPOTrainer",
    "RewardModel",
    "ValueModel",
    "RLHFPipeline",
    # Reward Modeling
    "RewardModelConfig",
    "RewardModelTrainer",
    "BradleyTerryModel",
    "PairwiseDataset",
]
