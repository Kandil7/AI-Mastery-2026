"""
Distributed Fine-Tuning - Module 2.4.4

Production-ready distributed SFT implementations:
- FSDP for SFT
- DeepSpeed for SFT
- Distributed data loading

References:
- "PyTorch FSDP" (Zhao et al., 2023)
- "DeepSpeed" (Rasley et al., 2020)
"""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

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
from torch.utils.data import DataLoader, DistributedSampler

logger = logging.getLogger(__name__)


@dataclass
class DistributedSFTConfig:
    """Configuration for distributed SFT."""
    # Distributed settings
    backend: str = "nccl"
    gradient_checkpointing: bool = True
    
    # FSDP settings
    use_fsdp: bool = True
    fsdp_sharding_strategy: str = "FULL_SHARD"
    fsdp_cpu_offload: bool = False
    fsdp_mixed_precision: bool = True
    
    # DeepSpeed settings
    use_deepspeed: bool = False
    deepspeed_stage: int = 3
    deepspeed_config: Optional[Dict] = None
    
    # Training settings
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.03
    
    # Data settings
    max_seq_length: int = 2048
    num_workers: int = 4
    
    # Output settings
    output_dir: str = "./distributed_sft_output"
    save_steps: int = 500
    logging_steps: int = 10


def setup_distributed_environment() -> Tuple[int, int, int]:
    """
    Setup distributed training environment.
    
    Returns:
        Tuple of (world_size, rank, local_rank)
    """
    # Initialize process group
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set device
    torch.cuda.set_device(local_rank)
    
    logger.info(f"Distributed setup: world_size={world_size}, rank={rank}, local_rank={local_rank}")
    
    return world_size, rank, local_rank


def cleanup_distributed() -> None:
    """Cleanup distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()


class FSDPSFTTrainer:
    """
    FSDP-based SFT Trainer.
    
    Uses Fully Sharded Data Parallel for memory-efficient training.
    
    Args:
        model: Model to train
        config: Distributed SFT configuration
        tokenizer: Tokenizer
        train_dataset: Training dataset
        
    Example:
        >>> trainer = FSDPSFTTrainer(model, config, tokenizer, dataset)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedSFTConfig,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup distributed
        self.world_size, self.rank, self.local_rank = setup_distributed_environment()
        
        # Prepare model with FSDP
        self.model = self._prepare_fsdp_model(model)
        
        # Device
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        # Optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.use_amp = config.fsdp_mixed_precision
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _prepare_fsdp_model(self, model: nn.Module) -> FSDP:
        """Prepare model with FSDP."""
        # Auto wrap policy
        auto_wrap_policy = self._get_auto_wrap_policy(model)
        
        # Mixed precision
        mixed_precision = None
        if self.config.fsdp_mixed_precision:
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        
        # Sharding strategy
        sharding_strategy = {
            "FULL_SHARD": ShardingStrategy.FULL_SHARD,
            "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
            "NO_SHARD": ShardingStrategy.NO_SHARD,
        }.get(self.config.fsdp_sharding_strategy, ShardingStrategy.FULL_SHARD)
        
        # Create FSDP model
        fsdp_model = FSDP(
            model,
            sharding_strategy=sharding_strategy,
            cpu_offload=self.config.fsdp_cpu_offload,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision,
            device_id=self.local_rank,
            sync_module_states=True,
            use_orig_params=True,
        )
        
        logger.info(f"Model wrapped with FSDP (strategy={self.config.fsdp_sharding_strategy})")
        
        return fsdp_model
    
    def _get_auto_wrap_policy(
        self,
        model: nn.Module,
    ) -> Callable:
        """Get auto wrap policy for FSDP."""
        # Try transformer auto wrap policy
        try:
            from transformers import PreTrainedModel
            
            if isinstance(model, PreTrainedModel):
                return transformer_auto_wrap_policy
        except ImportError:
            pass
        
        # Fallback: lambda policy
        def lambda_policy(module, recurse, nonwrapped_numel):
            return nonwrapped_numel >= 1e8
        
        return lambda_policy
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        # Use FSDP's get_model_params for proper parameter handling
        params = self.model.parameters()
        
        return torch.optim.AdamW(
            params,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        
        num_training_steps = (
            len(self.train_dataset) //
            (self.config.batch_size * self.gradient_accumulation_steps * self.world_size) *
            self.config.num_epochs
        )
        
        num_warmup_steps = int(num_training_steps * self.config.warmup_ratio)
        
        scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - num_warmup_steps,
        )
        
        if num_warmup_steps > 0:
            warmup_scheduler = LinearLR(
                self.optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=num_warmup_steps,
            )
            scheduler = SequentialLR(
                self.optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[num_warmup_steps],
            )
        
        return scheduler
    
    def _get_dataloader(
        self,
        dataset: Any,
        shuffle: bool = False,
    ) -> DataLoader:
        """Create distributed dataloader."""
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            drop_last=True,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            collate_fn=self._collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
    
    def _collate_fn(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch."""
        return {
            'input_ids': torch.tensor([f['input_ids'] for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f['attention_mask'] for f in features], dtype=torch.long),
            'labels': torch.tensor([f['labels'] for f in features], dtype=torch.long),
        }
    
    def train(self) -> Dict[str, Any]:
        """Run FSDP training."""
        from tqdm import tqdm
        
        logger.info("Starting FSDP SFT training...")
        
        train_dataloader = self._get_dataloader(self.train_dataset, shuffle=True)
        
        self.model.train()
        
        progress_bar = tqdm(
            total=len(train_dataloader) * self.config.num_epochs,
            disable=self.rank != 0,
        )
        
        training_metrics = {'loss': [], 'eval_loss': []}
        accumulation_step = 0
        
        for epoch in range(self.config.num_epochs):
            train_dataloader.sampler.set_epoch(epoch)
            
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass with mixed precision
                if self.use_amp:
                    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.model(
                            input_ids=batch['input_ids'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['labels'],
                        )
                        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                else:
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                    )
                    loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                loss = loss / self.config.gradient_accumulation_steps
                
                # Backward pass
                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulation_step += 1
                
                # Optimizer step
                if accumulation_step % self.config.gradient_accumulation_steps == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    if self.rank == 0:
                        training_metrics['loss'].append(loss.item() * self.config.gradient_accumulation_steps)
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0 and self.rank == 0:
                    self.save_checkpoint(f'checkpoint-{self.global_step}')
        
        progress_bar.close()
        
        # Save final checkpoint
        if self.rank == 0:
            self.save_checkpoint('final')
        
        logger.info("FSDP SFT training complete!")
        
        return training_metrics
    
    def save_checkpoint(self, name: str) -> None:
        """Save FSDP checkpoint."""
        output_path = self.output_dir / name
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Use FSDP's full state dict
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            save_policy,
        ):
            state_dict = self.model.state_dict()
            
            if self.rank == 0:
                torch.save(state_dict, output_path / 'model.pt')
                
                # Save config
                self.tokenizer.save_pretrained(output_path)
                
                # Save training state
                training_state = {
                    'global_step': self.global_step,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                }
                if self.scaler:
                    training_state['scaler_state_dict'] = self.scaler.state_dict()
                
                torch.save(training_state, output_path / 'training_state.pt')
        
        logger.info(f"Checkpoint saved to {output_path}")


class DeepSpeedSFTTrainer:
    """
    DeepSpeed-based SFT Trainer.
    
    Uses DeepSpeed ZeRO for memory-efficient training.
    
    Args:
        model: Model to train
        config: Distributed SFT configuration
        tokenizer: Tokenizer
        train_dataset: Training dataset
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: DistributedSFTConfig,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        # Setup distributed
        self.world_size, self.rank, self.local_rank = setup_distributed_environment()
        
        # Initialize DeepSpeed
        self._init_deepspeed(model)
        
        # Device
        self.device = torch.device(f'cuda:{self.local_rank}')
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _init_deepspeed(self, model: nn.Module) -> None:
        """Initialize DeepSpeed."""
        try:
            import deepspeed
        except ImportError:
            raise ImportError("DeepSpeed not installed. Install with: pip install deepspeed")
        
        # Default DeepSpeed config
        ds_config = self.config.deepspeed_config or {
            "train_batch_size": self.config.batch_size * self.world_size,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.config.learning_rate,
                    "betas": [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": 0.01,
                }
            },
            "fp16": {
                "enabled": False,
            },
            "bf16": {
                "enabled": True,
            },
            "zero_optimization": {
                "stage": self.config.deepspeed_stage,
                "offload_optimizer": {
                    "device": "cpu",
                    "pin_memory": True,
                },
                "allgather_partitions": True,
                "allgather_bucket_size": 2e8,
                "overlap_comm": True,
                "reduce_scatter": True,
                "reduce_bucket_size": 2e8,
                "contiguous_gradients": True,
            },
            "gradient_clipping": 1.0,
        }
        
        # Initialize DeepSpeed
        self.model, self.optimizer, _, self.scheduler = deepspeed.initialize(
            model=model,
            config=ds_config,
        )
        
        logger.info(f"DeepSpeed initialized (ZeRO stage {self.config.deepspeed_stage})")
    
    def _get_dataloader(
        self,
        dataset: Any,
        shuffle: bool = False,
    ) -> DataLoader:
        """Create distributed dataloader."""
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=shuffle,
            drop_last=True,
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            collate_fn=self._collate_fn,
            num_workers=self.config.num_workers,
            pin_memory=True,
        )
    
    def _collate_fn(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch."""
        return {
            'input_ids': torch.tensor([f['input_ids'] for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f['attention_mask'] for f in features], dtype=torch.long),
            'labels': torch.tensor([f['labels'] for f in features], dtype=torch.long),
        }
    
    def train(self) -> Dict[str, Any]:
        """Run DeepSpeed training."""
        from tqdm import tqdm
        
        logger.info("Starting DeepSpeed SFT training...")
        
        train_dataloader = self._get_dataloader(self.train_dataset, shuffle=True)
        
        self.model.train()
        
        progress_bar = tqdm(
            total=len(train_dataloader) * self.config.num_epochs,
            disable=self.rank != 0,
        )
        
        training_metrics = {'loss': [], 'eval_loss': []}
        
        for epoch in range(self.config.num_epochs):
            train_dataloader.sampler.set_epoch(epoch)
            
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                )
                
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                
                # DeepSpeed backward
                self.model.backward(loss)
                
                # DeepSpeed step
                self.model.step()
                
                self.global_step += 1
                progress_bar.update(1)
                
                if self.rank == 0:
                    training_metrics['loss'].append(loss.item())
                
                # Save checkpoint
                if self.global_step % self.config.save_steps == 0:
                    self.save_checkpoint(f'checkpoint-{self.global_step}')
        
        progress_bar.close()
        
        # Save final checkpoint
        if self.rank == 0:
            self.save_checkpoint('final')
        
        logger.info("DeepSpeed SFT training complete!")
        
        return training_metrics
    
    def save_checkpoint(self, name: str) -> None:
        """Save DeepSpeed checkpoint."""
        output_path = self.output_dir / name
        
        # DeepSpeed save
        self.model.save_checkpoint(
            str(output_path),
            client_state={
                'global_step': self.global_step,
            },
        )
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(output_path))
        
        logger.info(f"DeepSpeed checkpoint saved to {output_path}")


def create_distributed_sft_dataloader(
    dataset: Any,
    batch_size: int,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
) -> DataLoader:
    """
    Create distributed dataloader for SFT.
    
    Args:
        dataset: Dataset
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Whether to shuffle
        pin_memory: Pin memory
    
    Returns:
        Distributed DataLoader
    """
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=lambda x: {
            'input_ids': torch.tensor([f['input_ids'] for f in x], dtype=torch.long),
            'attention_mask': torch.tensor([f['attention_mask'] for f in x], dtype=torch.long),
            'labels': torch.tensor([f['labels'] for f in x], dtype=torch.long),
        },
    )
