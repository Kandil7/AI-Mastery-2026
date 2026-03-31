"""
LLM Scientist - Production-Ready LLM Implementation Suite

A comprehensive collection of production-grade implementations covering:
- LLM Architecture (Attention, Transformers, Tokenization, Sampling)
- Pre-Training (Data Prep, Distributed Training, Optimization, Monitoring)
- Post-Training Datasets (Formats, Synthetic Data, Enhancement, Quality Filtering)
- Supervised Fine-Tuning (SFT, LoRA, QLoRA, Distributed)
- Preference Alignment (Rejection Sampling, DPO, RLHF, Reward Modeling)
- Evaluation (Benchmarks, Human Eval, Model-Based Eval, Feedback Analysis)
- Quantization (Base Quant, GGUF, GPTQ, AWQ, EXL2)
- New Trends (Model Merging, Multimodal, Interpretability, Test-Time Compute)

Author: AI-Mastery-2026
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI-Mastery-2026"

from .module_2_1_llm_architecture import attention, transformer, tokenization, sampling
from .module_2_2_pretraining import data_prep, distributed_training, optimization, monitoring
from .module_2_3_post_training import formats, synthetic_data, enhancement, quality_filtering
from .module_2_4_sft import sft, lora, qlora, distributed
from .module_2_5_preference import rejection_sampling, dpo, rlhf, reward_modeling
from .module_2_6_evaluation import benchmarks, human_eval, model_based_eval, feedback_analysis
from .module_2_7_quantization import base_quant, gguf, gptq, awq, exl2
from .module_2_8_new_trends import model_merging, multimodal, interpretability, test_time_compute

__all__ = [
    # Module 2.1: LLM Architecture
    "attention",
    "transformer",
    "tokenization",
    "sampling",
    # Module 2.2: Pre-Training
    "data_prep",
    "distributed_training",
    "optimization",
    "monitoring",
    # Module 2.3: Post-Training Datasets
    "formats",
    "synthetic_data",
    "enhancement",
    "quality_filtering",
    # Module 2.4: Supervised Fine-Tuning
    "sft",
    "lora",
    "qlora",
    "distributed",
    # Module 2.5: Preference Alignment
    "rejection_sampling",
    "dpo",
    "rlhf",
    "reward_modeling",
    # Module 2.6: Evaluation
    "benchmarks",
    "human_eval",
    "model_based_eval",
    "feedback_analysis",
    # Module 2.7: Quantization
    "base_quant",
    "gguf",
    "gptq",
    "awq",
    "exl2",
    # Module 2.8: New Trends
    "model_merging",
    "multimodal",
    "interpretability",
    "test_time_compute",
]
