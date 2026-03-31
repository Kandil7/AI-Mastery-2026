"""
Distributed Training - Module 2.2.2

Production-ready distributed training implementations:
- Data Parallelism (DDP)
- Pipeline Parallelism
- Tensor Parallelism
- FSDP (Fully Sharded Data Parallel)
- DeepSpeed ZeRO integration

References:
- "PyTorch Distributed" (Li et al.)
- "DeepSpeed: Extreme-scale Model Training" (Rasley et al., 2020)
- "Megatron-LM: Training Multi-Billion Parameter Language Models" (Shoeybi et al., 2019)
"""

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training."""
    # World size and rank
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Parallelism settings
    data_parallel: bool = True
    tensor_parallel: bool = False
    pipeline_parallel: bool = False
    fsdp: bool = False
    
    # FSDP settings
    fsdp_sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    fsdp_cpu_offload: bool = False
    fsdp_mixed_precision: bool = True
    
    # DeepSpeed settings
    deepspeed_enabled: bool = False
    deepspeed_stage: int = 2  # 1, 2, 3
    
    # Communication settings
    gradient_accumulation_steps: int = 1
    find_unused_parameters: bool = False
    
    # Checkpoint settings
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 1000


def setup_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
) -> Tuple[int, int, int]:
    """
    Setup distributed training environment.
    
    Args:
        backend: Distributed backend ('nccl' for GPU, 'gloo' for CPU)
        init_method: Initialization method (URL or env)
    
    Returns:
        Tuple of (world_size, rank, local_rank)
    """
    # Initialize process group
    if not dist.is_initialized():
        if init_method is None:
            init_method = "env://"
        
        dist.init_process_group(backend=backend, init_method=init_method)
    
    # Get distributed info
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    logger.info(f"Distributed initialized: world_size={world_size}, rank={rank}, local_rank={local_rank}")
    
    return world_size, rank, local_rank


def cleanup_distributed() -> None:
    """Cleanup distributed training environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training cleaned up")


class BaseDistributedTrainer(ABC):
    """Abstract base class for distributed trainers."""
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
    ):
        self.model = model
        self.config = config
        self.is_master = config.rank == 0
    
    @abstractmethod
    def prepare_model(self) -> nn.Module:
        """Prepare model for distributed training."""
        pass
    
    @abstractmethod
    def prepare_optimizer(
        self,
        optimizer_cls: type,
        **optimizer_kwargs,
    ) -> Optimizer:
        """Prepare optimizer for distributed training."""
        pass
    
    @abstractmethod
    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with gradient handling."""
        pass
    
    @abstractmethod
    def step(self, optimizer: Optimizer) -> None:
        """Optimizer step."""
        pass
    
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: str,
    ) -> None:
        """Save checkpoint."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        if self.config.fsdp:
            # FSDP requires special handling
            save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(self.model, StateDictType.FULL_STATE_DICT, save_policy):
                checkpoint['model_state_dict'] = self.model.state_dict()
        
        if self.is_master:
            torch.save(checkpoint, path)
            logger.info(f"Checkpoint saved to {path}")
    
    def load_checkpoint(
        self,
        path: str,
        map_location: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Load checkpoint."""
        if map_location is None:
            map_location = f"cuda:{self.config.local_rank}" if torch.cuda.is_available() else "cpu"
        
        checkpoint = torch.load(path, map_location=map_location)
        logger.info(f"Checkpoint loaded from {path}")
        return checkpoint


class DataParallelTrainer(BaseDistributedTrainer):
    """
    Data Parallel Training using PyTorch DDP.
    
    Standard data parallelism where each GPU has a full model copy
    and processes a different data shard.
    
    Args:
        model: Model to train
        config: Distributed configuration
        
    Example:
        >>> config = DistributedConfig()
        >>> trainer = DataParallelTrainer(model, config)
        >>> model = trainer.prepare_model()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
    ):
        super().__init__(model, config)
        self._ddp_model: Optional[DDP] = None
    
    def prepare_model(self) -> nn.Module:
        """Prepare model with DDP."""
        if torch.cuda.is_available():
            self.model = self.model.to(self.config.local_rank)
        
        self._ddp_model = DDP(
            self.model,
            device_ids=[self.config.local_rank] if torch.cuda.is_available() else None,
            output_device=self.config.local_rank if torch.cuda.is_available() else None,
            find_unused_parameters=self.config.find_unused_parameters,
        )
        
        logger.info("Model wrapped with DDP")
        return self._ddp_model
    
    def prepare_optimizer(
        self,
        optimizer_cls: type,
        **optimizer_kwargs,
    ) -> Optimizer:
        """Prepare optimizer."""
        return optimizer_cls(self.model.parameters(), **optimizer_kwargs)
    
    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass."""
        loss.backward()
    
    def step(self, optimizer: Optimizer) -> None:
        """Optimizer step."""
        optimizer.step()
    
    @property
    def model(self) -> nn.Module:
        """Get underlying model."""
        return self._ddp_model.module if self._ddp_model else self._model
    
    @model.setter
    def model(self, value: nn.Module) -> None:
        self._model = value


class PipelineParallelTrainer(BaseDistributedTrainer):
    """
    Pipeline Parallel Training.
    
    Splits model layers across GPUs and uses pipeline scheduling
    to overlap computation.
    
    Args:
        model: Model to train (must be sequential)
        config: Distributed configuration
        num_stages: Number of pipeline stages
        chunks: Number of micro-batches
        
    Reference:
        "GPipe: Efficient Training of Giant Neural Networks" (Huang et al., 2019)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        num_stages: int = 4,
        chunks: int = 4,
    ):
        super().__init__(model, config)
        self.num_stages = num_stages
        self.chunks = chunks
        self._pipeline_model: Optional[nn.Module] = None
    
    def _split_model(self) -> List[nn.Sequential]:
        """Split model into pipeline stages."""
        # Get all layers
        layers = list(self.model.children())
        
        if len(layers) < self.num_stages:
            logger.warning(f"Model has fewer layers ({len(layers)}) than stages ({self.num_stages})")
            self.num_stages = len(layers)
        
        # Split layers evenly
        layers_per_stage = len(layers) // self.num_stages
        stages = []
        
        for i in range(self.num_stages):
            start = i * layers_per_stage
            end = start + layers_per_stage if i < self.num_stages - 1 else len(layers)
            stage_layers = layers[start:end]
            stages.append(nn.Sequential(*stage_layers))
        
        return stages
    
    def prepare_model(self) -> nn.Module:
        """Prepare model for pipeline parallelism."""
        try:
            from torch.distributed.pipeline.sync import Pipe
            
            stages = self._split_model()
            
            # Assign stages to devices
            devices = list(range(min(self.num_stages, torch.cuda.device_count())))
            
            # Wrap with Pipe
            self._pipeline_model = Pipe(
                nn.Sequential(*stages),
                chunks=self.chunks,
            )
            
            logger.info(f"Model wrapped for pipeline parallelism with {self.num_stages} stages")
            return self._pipeline_model
        
        except ImportError:
            logger.warning("Pipeline parallelism requires torch >= 1.8. Falling back to DDP.")
            return DataParallelTrainer(self.model, self.config).prepare_model()
    
    def prepare_optimizer(
        self,
        optimizer_cls: type,
        **optimizer_kwargs,
    ) -> Optimizer:
        """Prepare optimizer."""
        return optimizer_cls(self.model.parameters(), **optimizer_kwargs)
    
    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass."""
        loss.backward()
    
    def step(self, optimizer: Optimizer) -> None:
        """Optimizer step."""
        optimizer.step()


class TensorParallelTrainer(BaseDistributedTrainer):
    """
    Tensor Parallel Training.
    
    Splits individual tensor operations (like matrix multiplications)
    across GPUs. Used in Megatron-LM style training.
    
    Args:
        model: Model to train
        config: Distributed configuration
        tp_size: Tensor parallel size
        
    Reference:
        "Megatron-LM: Training Multi-Billion Parameter Language Models" (Shoeybi et al., 2019)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        tp_size: int = 2,
    ):
        super().__init__(model, config)
        self.tp_size = tp_size
        self._tp_group: Optional[dist.ProcessGroup] = None
    
    def _create_tp_group(self) -> None:
        """Create tensor parallel process group."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        num_tp_groups = world_size // self.tp_size
        
        for i in range(num_tp_groups):
            start_rank = i * self.tp_size
            ranks = list(range(start_rank, start_rank + self.tp_size))
            
            if rank in ranks:
                self._tp_group = dist.new_group(ranks=ranks)
                break
    
    def _apply_tensor_parallel(self, module: nn.Module) -> nn.Module:
        """Apply tensor parallelism to model layers."""
        # This is a simplified implementation
        # Full implementation would modify linear layers to split weights
        
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                # Split output dimension for tensor parallel
                if child.out_features % self.tp_size == 0:
                    # Create column parallel linear
                    new_out_features = child.out_features // self.tp_size
                    new_linear = nn.Linear(
                        child.in_features,
                        new_out_features,
                        bias=child.bias is not None,
                    )
                    
                    # Copy weights (split across output dim)
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    start = rank * new_out_features
                    end = start + new_out_features
                    
                    with torch.no_grad():
                        new_linear.weight.copy_(child.weight[start:end, :])
                        if child.bias is not None:
                            new_linear.bias.copy_(child.bias[start:end])
                    
                    setattr(module, name, new_linear)
            
            elif len(list(child.children())) > 0:
                self._apply_tensor_parallel(child)
        
        return module
    
    def prepare_model(self) -> nn.Module:
        """Prepare model for tensor parallelism."""
        if dist.is_initialized():
            self._create_tp_group()
        
        self.model = self._apply_tensor_parallel(self.model)
        
        if torch.cuda.is_available():
            self.model = self.model.to(self.config.local_rank)
        
        logger.info(f"Model prepared for tensor parallelism (tp_size={self.tp_size})")
        return self.model
    
    def prepare_optimizer(
        self,
        optimizer_cls: type,
        **optimizer_kwargs,
    ) -> Optimizer:
        """Prepare optimizer."""
        return optimizer_cls(self.model.parameters(), **optimizer_kwargs)
    
    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with all-reduce."""
        loss.backward()
        
        # All-reduce gradients across tensor parallel group
        if self._tp_group is not None:
            for param in self.model.parameters():
                if param.grad is not None:
                    dist.all_reduce(param.grad, group=self._tp_group)
    
    def step(self, optimizer: Optimizer) -> None:
        """Optimizer step."""
        optimizer.step()


class FSDPTrainer(BaseDistributedTrainer):
    """
    Fully Sharded Data Parallel (FSDP) Training.
    
    Shards model parameters, gradients, and optimizer states across GPUs.
    More memory efficient than DDP for large models.
    
    Args:
        model: Model to train
        config: Distributed configuration
        
    Reference:
        "PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel" (Zhao et al., 2023)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
    ):
        super().__init__(model, config)
        self._fsdp_model: Optional[FSDP] = None
    
    def _get_sharding_strategy(self) -> ShardingStrategy:
        """Get FSDP sharding strategy from config."""
        strategies = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
            "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
        }
        return strategies.get(self.config.fsdp_sharding_strategy, ShardingStrategy.FULL_SHARD)
    
    def _get_auto_wrap_policy(self) -> Callable:
        """Get auto wrap policy for FSDP."""
        # Try to use transformer auto wrap policy
        try:
            from transformers import PreTrainedModel
            
            if isinstance(self.model, PreTrainedModel):
                return transformer_auto_wrap_policy
        except ImportError:
            pass
        
        # Default: wrap each layer
        return lambda_auto_wrap_policy
    
    def _get_mixed_precision_config(self) -> Optional[MixedPrecision]:
        """Get mixed precision config for FSDP."""
        if not self.config.fsdp_mixed_precision:
            return None
        
        return MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )
    
    def prepare_model(self) -> nn.Module:
        """Prepare model with FSDP."""
        if not dist.is_initialized():
            logger.warning("FSDP requires distributed training. Falling back to regular model.")
            return self.model
        
        self._fsdp_model = FSDP(
            self.model,
            sharding_strategy=self._get_sharding_strategy(),
            cpu_offload=self.config.fsdp_cpu_offload,
            auto_wrap_policy=self._get_auto_wrap_policy(),
            mixed_precision=self._get_mixed_precision_config(),
            device_id=self.config.local_rank if torch.cuda.is_available() else None,
        )
        
        logger.info("Model wrapped with FSDP")
        return self._fsdp_model
    
    def prepare_optimizer(
        self,
        optimizer_cls: type,
        **optimizer_kwargs,
    ) -> Optimizer:
        """Prepare optimizer."""
        return optimizer_cls(self.model.parameters(), **optimizer_kwargs)
    
    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass."""
        self._fsdp_model.backward(loss)
    
    def step(self, optimizer: Optimizer) -> None:
        """Optimizer step."""
        optimizer.step()
    
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: str,
    ) -> None:
        """Save FSDP checkpoint."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(
            self._fsdp_model,
            StateDictType.FULL_STATE_DICT,
            save_policy,
        ):
            state_dict = self._fsdp_model.state_dict()
            checkpoint['model_state_dict'] = state_dict
        
        if self.is_master:
            import os
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(checkpoint, path)
            logger.info(f"FSDP checkpoint saved to {path}")


class DeepSpeedTrainer(BaseDistributedTrainer):
    """
    DeepSpeed ZeRO Training.
    
    Uses DeepSpeed's ZeRO optimizer for memory-efficient training.
    Supports ZeRO stages 1-3.
    
    Args:
        model: Model to train
        config: Distributed configuration
        deepspeed_config: DeepSpeed configuration dict
        
    Reference:
        "DeepSpeed: Extreme-scale Model Training" (Rasley et al., 2020)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        deepspeed_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(model, config)
        self.deepspeed_config = deepspeed_config or self._get_default_config()
        self._deepspeed_model: Optional[Any] = None
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default DeepSpeed configuration."""
        return {
            "train_batch_size": 32,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": 1e-4,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                }
            },
            "fp16": {
                "enabled": True,
                "loss_scale": 0,
                "initial_scale_power": 16,
            },
            "zero_optimization": {
                "stage": self.config.deepspeed_stage,
                "offload_optimizer": {
                    "device": "cpu" if self.config.fsdp_cpu_offload else "none",
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
            },
        }
    
    def prepare_model(self) -> nn.Module:
        """Prepare model with DeepSpeed."""
        try:
            import deepspeed
            
            self._deepspeed_model, _, _, _ = deepspeed.initialize(
                model=self.model,
                config=self.deepspeed_config,
            )
            
            logger.info(f"Model initialized with DeepSpeed ZeRO stage {self.config.deepspeed_stage}")
            return self._deepspeed_model
        
        except ImportError:
            logger.warning("DeepSpeed not installed. Falling back to DDP.")
            return DataParallelTrainer(self.model, self.config).prepare_model()
    
    def prepare_optimizer(
        self,
        optimizer_cls: type,
        **optimizer_kwargs,
    ) -> Optimizer:
        """Prepare optimizer (handled by DeepSpeed)."""
        # DeepSpeed handles optimizer internally
        return self._deepspeed_model.optimizer if self._deepspeed_model else None
    
    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass with DeepSpeed."""
        if self._deepspeed_model:
            self._deepspeed_model.backward(loss)
        else:
            loss.backward()
    
    def step(self, optimizer: Optimizer) -> None:
        """Optimizer step with DeepSpeed."""
        if self._deepspeed_model:
            self._deepspeed_model.step()
        elif optimizer:
            optimizer.step()
    
    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: str,
    ) -> None:
        """Save DeepSpeed checkpoint."""
        if self._deepspeed_model:
            self._deepspeed_model.save_checkpoint(path, client_state=checkpoint)
            logger.info(f"DeepSpeed checkpoint saved to {path}")
    
    def load_checkpoint(
        self,
        path: str,
    ) -> Tuple[Dict[str, Any], Any]:
        """Load DeepSpeed checkpoint."""
        if self._deepspeed_model:
            _, client_state = self._deepspeed_model.load_checkpoint(path)
            return client_state or {}, self._deepspeed_model
        return {}, None


class HybridParallelTrainer(BaseDistributedTrainer):
    """
    Hybrid Parallel Training (3D Parallelism).
    
    Combines data, tensor, and pipeline parallelism for
    training very large models.
    
    Args:
        model: Model to train
        config: Distributed configuration
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        
    Reference:
        "Efficient Large-Scale Language Model Training on GPU Clusters" (Narayanan et al., 2021)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedConfig,
        tp_size: int = 1,
        pp_size: int = 1,
    ):
        super().__init__(model, config)
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.dp_size = 1
        
        self._tp_group: Optional[dist.ProcessGroup] = None
        self._pp_group: Optional[dist.ProcessGroup] = None
        self._dp_group: Optional[dist.ProcessGroup] = None
    
    def _create_parallel_groups(self) -> None:
        """Create tensor, pipeline, and data parallel groups."""
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        
        self.dp_size = world_size // (self.tp_size * self.pp_size)
        
        # Create groups
        for i in range(self.pp_size):
            for j in range(self.tp_size):
                for k in range(self.dp_size):
                    # Compute ranks for each group
                    base = i * self.tp_size * self.dp_size + j * self.dp_size + k
                    
                    # TP group
                    tp_ranks = [base + m * self.dp_size for m in range(self.tp_size)]
                    if rank in tp_ranks:
                        self._tp_group = dist.new_group(ranks=tp_ranks)
                    
                    # PP group
                    pp_ranks = [i * self.tp_size * self.dp_size + m * self.dp_size + k 
                               for m in range(self.tp_size)]
                    if rank in pp_ranks:
                        self._pp_group = dist.new_group(ranks=pp_ranks)
                    
                    # DP group
                    dp_ranks = [i * self.tp_size * self.dp_size + j * self.dp_size + m 
                               for m in range(self.dp_size)]
                    if rank in dp_ranks:
                        self._dp_group = dist.new_group(ranks=dp_ranks)
    
    def prepare_model(self) -> nn.Module:
        """Prepare model for hybrid parallelism."""
        if dist.is_initialized():
            self._create_parallel_groups()
        
        # Apply tensor parallelism
        if self.tp_size > 1:
            tp_trainer = TensorParallelTrainer(self.model, self.config, self.tp_size)
            self.model = tp_trainer.prepare_model()
        
        # Apply pipeline parallelism
        if self.pp_size > 1:
            pp_trainer = PipelineParallelTrainer(self.model, self.config, self.pp_size)
            self.model = pp_trainer.prepare_model()
        
        # Wrap with DDP for data parallelism
        if self.dp_size > 1 and torch.cuda.is_available():
            self.model = DDP(
                self.model,
                device_ids=[self.config.local_rank],
                process_group=self._dp_group,
            )
        
        logger.info(f"Hybrid parallelism: TP={self.tp_size}, PP={self.pp_size}, DP={self.dp_size}")
        return self.model
    
    def prepare_optimizer(
        self,
        optimizer_cls: type,
        **optimizer_kwargs,
    ) -> Optimizer:
        """Prepare optimizer."""
        return optimizer_cls(self.model.parameters(), **optimizer_kwargs)
    
    def backward(self, loss: torch.Tensor) -> None:
        """Backward pass."""
        loss.backward()
    
    def step(self, optimizer: Optimizer) -> None:
        """Optimizer step."""
        optimizer.step()


def get_distributed_sampler(
    dataset: torch.utils.data.Dataset,
    config: DistributedConfig,
    shuffle: bool = True,
    seed: int = 42,
) -> DistributedSampler:
    """
    Get distributed sampler for dataset.
    
    Args:
        dataset: Dataset to sample from
        config: Distributed configuration
        shuffle: Whether to shuffle
        seed: Random seed
    
    Returns:
        DistributedSampler
    """
    return DistributedSampler(
        dataset,
        num_replicas=config.world_size,
        rank=config.rank,
        shuffle=shuffle,
        seed=seed,
        drop_last=True,
    )


def create_dataloader(
    dataset: torch.utils.data.Dataset,
    config: DistributedConfig,
    batch_size: int,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create distributed dataloader.
    
    Args:
        dataset: Dataset
        config: Distributed configuration
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster transfer
        shuffle: Whether to shuffle
    
    Returns:
        DataLoader
    """
    sampler = get_distributed_sampler(dataset, config, shuffle=shuffle)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
