"""
Module 2.2: Pre-Training Models

Large-scale pre-training components:
- Data preparation (collection, cleaning, deduplication, filtering)
- Distributed training (data, pipeline, tensor parallelism, FSDP, DeepSpeed)
- Optimization (mixed precision, gradient checkpointing, LR scheduling)
- Monitoring (loss tracking, GPU utilization, memory profiling)
"""

from .data_prep import (
    DataCollector,
    DataCleaner,
    Deduplicator,
    DataFilter,
    QualityScorer,
    PreTrainingDataset,
)
from .distributed_training import (
    DataParallelTrainer,
    PipelineParallelTrainer,
    TensorParallelTrainer,
    FSDPTrainer,
    DeepSpeedTrainer,
)
from .optimization import (
    MixedPrecisionTrainer,
    GradientCheckpointing,
    GradientClipper,
    LearningRateScheduler,
    OptimizerFactory,
)
from .monitoring import (
    LossTracker,
    GradientMonitor,
    GPUMonitor,
    MemoryProfiler,
    WandBLogger,
)

__all__ = [
    # Data Prep
    "DataCollector",
    "DataCleaner",
    "Deduplicator",
    "DataFilter",
    "QualityScorer",
    "PreTrainingDataset",
    # Distributed Training
    "DataParallelTrainer",
    "PipelineParallelTrainer",
    "TensorParallelTrainer",
    "FSDPTrainer",
    "DeepSpeedTrainer",
    # Optimization
    "MixedPrecisionTrainer",
    "GradientCheckpointing",
    "GradientClipper",
    "LearningRateScheduler",
    "OptimizerFactory",
    # Monitoring
    "LossTracker",
    "GradientMonitor",
    "GPUMonitor",
    "MemoryProfiler",
    "WandBLogger",
]
