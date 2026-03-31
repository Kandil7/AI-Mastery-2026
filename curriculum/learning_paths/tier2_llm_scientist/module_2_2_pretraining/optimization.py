"""
Optimization Techniques - Module 2.2.3

Production-ready optimization implementations:
- Mixed Precision Training (AMP)
- Gradient Checkpointing
- Gradient Clipping
- Learning Rate Scheduling
- Optimizer Factory

References:
- "Mixed Precision Training" (Micikevicius et al., 2017)
- "Training Deep Nets with Sublinear Memory Cost" (Chen et al., 2016)
"""

import math
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LambdaLR,
    LinearLR,
    OneCycleLR,
    SequentialLR,
)

logger = logging.getLogger(__name__)


class SchedulerType(Enum):
    """Learning rate scheduler types."""
    LINEAR = "linear"
    COSINE = "cosine"
    COSINE_WITH_RESTARTS = "cosine_with_restarts"
    POLYNOMIAL = "polynomial"
    CONSTANT = "constant"
    CONSTANT_WITH_WARMUP = "constant_with_warmup"
    ONE_CYCLE = "one_cycle"
    CUSTOM = "custom"


@dataclass
class OptimizerConfig:
    """Configuration for optimizer."""
    optimizer_type: str = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    momentum: float = 0.9
    nesterov: bool = True
    
    # AdamW specific
    correct_bias: bool = True
    
    # SGD specific
    dampening: float = 0.0
    
    # Adafactor specific
    scale_parameter: bool = True
    relative_step: bool = True
    warmup_init: bool = False


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler."""
    scheduler_type: SchedulerType = SchedulerType.COSINE
    num_warmup_steps: int = 1000
    num_training_steps: int = 100000
    min_lr_ratio: float = 0.1
    num_cycles: int = 1  # For cosine with restarts
    power: float = 1.0  # For polynomial
    last_epoch: int = -1
    
    # One cycle specific
    pct_start: float = 0.3
    anneal_strategy: str = "cos"
    cycle_momentum: bool = True
    base_momentum: float = 0.85
    max_momentum: float = 0.95
    div_factor: float = 25.0
    final_div_factor: float = 10000.0


class MixedPrecisionTrainer:
    """
    Mixed Precision Training with Automatic Mixed Precision (AMP).
    
    Uses FP16 for most operations while maintaining FP32 master weights
    for numerical stability.
    
    Args:
        enabled: Whether to enable mixed precision
        dtype: Precision dtype (torch.float16 or torch.bfloat16)
        loss_scale: Dynamic loss scaling
        growth_factor: Loss scale growth factor
        backoff_factor: Loss scale backoff factor
        
    Example:
        >>> amp_trainer = MixedPrecisionTrainer(enabled=True)
        >>> with amp_trainer.autocast():
        ...     output = model(input)
        >>> loss = criterion(output, target)
        >>> amp_trainer.scale_loss(loss).backward()
    """
    
    def __init__(
        self,
        enabled: bool = True,
        dtype: torch.dtype = torch.float16,
        init_scale: float = 2.0 ** 16,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ):
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = dtype
        
        if self.enabled:
            self.scaler = GradScaler(
                init_scale=init_scale,
                growth_factor=growth_factor,
                backoff_factor=backoff_factor,
                growth_interval=growth_interval,
            )
        else:
            self.scaler = None
    
    def autocast(self, enabled: bool = True):
        """Context manager for autocast."""
        if self.enabled and enabled:
            return autocast(dtype=self.dtype)
        else:
            # Return a no-op context manager
            return torch.cuda.amp.autocast(enabled=False)
    
    def scale_loss(self, loss: Tensor) -> Tensor:
        """Scale loss for backward pass."""
        if self.scaler is not None:
            return self.scaler.scale(loss)
        return loss
    
    def step(
        self,
        optimizer: Optimizer,
        clip_grad_norm: Optional[float] = None,
    ) -> None:
        """
        Optimizer step with gradient unscaling.
        
        Args:
            optimizer: Optimizer to step
            clip_grad_norm: Optional gradient norm clipping
        """
        if self.scaler is not None:
            if clip_grad_norm is not None:
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]['params'],
                    clip_grad_norm,
                )
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            if clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    optimizer.param_groups[0]['params'],
                    clip_grad_norm,
                )
            optimizer.step()
    
    def get_scale(self) -> float:
        """Get current loss scale."""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict."""
        if self.scaler is not None:
            return self.scaler.state_dict()
        return {}
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict."""
        if self.scaler is not None and state_dict:
            self.scaler.load_state_dict(state_dict)


class GradientCheckpointing:
    """
    Gradient Checkpointing for memory efficiency.
    
    Trades computation for memory by recomputing activations during
    backward pass instead of storing them.
    
    Args:
        use_reentrant: Whether to use reentrant checkpointing
        
    Example:
        >>> checkpoint = GradientCheckpointing()
        >>> output = checkpoint(model, input)
    """
    
    def __init__(self, use_reentrant: bool = True):
        self.use_reentrant = use_reentrant
    
    def __call__(
        self,
        module: nn.Module,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Apply gradient checkpointing to module forward.
        
        Args:
            module: Module to checkpoint
            *args: Positional arguments for forward
            **kwargs: Keyword arguments for forward
        
        Returns:
            Module output
        """
        return torch.utils.checkpoint.checkpoint(
            module,
            *args,
            use_reentrant=self.use_reentrant,
            **kwargs,
        )
    
    def apply_to_model(
        self,
        model: nn.Module,
        layer_names: Optional[List[str]] = None,
    ) -> nn.Module:
        """
        Apply gradient checkpointing to specific layers.
        
        Args:
            model: Model to modify
            layer_names: Names of layers to checkpoint
        
        Returns:
            Modified model
        """
        if layer_names is None:
            # Default: checkpoint all transformer blocks
            layer_names = []
            for name, module in model.named_modules():
                if 'layer' in name.lower() or 'block' in name.lower():
                    if len(list(module.children())) > 0:
                        layer_names.append(name)
        
        for name in layer_names:
            try:
                layer = dict(model.named_modules())[name]
                
                # Wrap forward method
                original_forward = layer.forward
                
                def make_checkpointed_forward(module, orig_forward):
                    def checkpointed_forward(*args, **kwargs):
                        return torch.utils.checkpoint.checkpoint(
                            orig_forward,
                            *args,
                            use_reentrant=self.use_reentrant,
                            **kwargs,
                        )
                    return checkpointed_forward
                
                layer.forward = make_checkpointed_forward(layer, original_forward)
                logger.info(f"Applied gradient checkpointing to {name}")
            
            except KeyError:
                logger.warning(f"Layer {name} not found in model")
        
        return model


class GradientClipper:
    """
    Gradient Clipping utilities.
    
    Prevents exploding gradients by clipping gradient norms or values.
    
    Args:
        max_norm: Maximum gradient norm
        norm_type: Type of norm (2 for L2, 'inf' for max)
        error_if_nonfinite: Whether to error on NaN/Inf gradients
        
    Example:
        >>> clipper = GradientClipper(max_norm=1.0)
        >>> clipper.clip(model)
    """
    
    def __init__(
        self,
        max_norm: float = 1.0,
        norm_type: Union[float, str] = 2.0,
        error_if_nonfinite: bool = False,
    ):
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.error_if_nonfinite = error_if_nonfinite
    
    def clip(
        self,
        model: nn.Module,
        parameters: Optional[torch.nn.ParameterList] = None,
    ) -> Tensor:
        """
        Clip gradients.
        
        Args:
            model: Model whose gradients to clip
            parameters: Optional specific parameters to clip
        
        Returns:
            Total norm of gradients
        """
        if parameters is None:
            parameters = [p for p in model.parameters() if p.grad is not None]
        else:
            parameters = [p for p in parameters if p.grad is not None]
        
        return torch.nn.utils.clip_grad_norm_(
            parameters,
            self.max_norm,
            norm_type=self.norm_type,
            error_if_nonfinite=self.error_if_nonfinite,
        )
    
    def clip_value(
        self,
        model: nn.Module,
        clip_value: float,
    ) -> None:
        """
        Clip gradient values.
        
        Args:
            model: Model whose gradients to clip
            clip_value: Maximum absolute gradient value
        """
        torch.nn.utils.clip_grad_value_(
            model.parameters(),
            clip_value,
        )
    
    @staticmethod
    def get_grad_stats(
        model: nn.Module,
    ) -> Dict[str, float]:
        """
        Get gradient statistics.
        
        Args:
            model: Model to analyze
        
        Returns:
            Dictionary of gradient statistics
        """
        grads = [p.grad.flatten() for p in model.parameters() if p.grad is not None]
        
        if not grads:
            return {
                'norm': 0.0,
                'max': 0.0,
                'min': 0.0,
                'mean': 0.0,
                'std': 0.0,
            }
        
        all_grads = torch.cat(grads)
        
        return {
            'norm': all_grads.norm().item(),
            'max': all_grads.max().item(),
            'min': all_grads.min().item(),
            'mean': all_grads.mean().item(),
            'std': all_grads.std().item(),
        }


class LearningRateScheduler:
    """
    Learning Rate Scheduler with warmup and decay.
    
    Implements various LR schedules commonly used in LLM training.
    
    Args:
        scheduler_type: Type of scheduler
        optimizer: Optimizer to schedule
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        
    Example:
        >>> scheduler = LearningRateScheduler(
        ...     scheduler_type=SchedulerType.COSINE,
        ...     optimizer=optimizer,
        ...     num_warmup_steps=1000,
        ...     num_training_steps=100000,
        ... )
        >>> for step in range(num_training_steps):
        ...     scheduler.step()
    """
    
    def __init__(
        self,
        scheduler_type: SchedulerType,
        optimizer: Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        min_lr_ratio: float = 0.1,
        num_cycles: int = 1,
        power: float = 1.0,
        last_epoch: int = -1,
    ):
        self.scheduler_type = scheduler_type
        self.optimizer = optimizer
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.min_lr_ratio = min_lr_ratio
        self.num_cycles = num_cycles
        self.power = power
        self.last_epoch = last_epoch
        
        self._scheduler = self._create_scheduler()
    
    def _lr_lambda(self, current_step: int) -> float:
        """Compute learning rate multiplier."""
        if current_step < self.num_warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, self.num_warmup_steps))
        
        # Progress through decay phase
        progress = float(current_step - self.num_warmup_steps) / float(
            max(1, self.num_training_steps - self.num_warmup_steps)
        )
        
        if self.scheduler_type == SchedulerType.LINEAR:
            return max(self.min_lr_ratio, 1.0 - (1.0 - self.min_lr_ratio) * progress)
        
        elif self.scheduler_type == SchedulerType.COSINE:
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * 0.5 * (
                1.0 + math.cos(math.pi * progress)
            )
        
        elif self.scheduler_type == SchedulerType.POLYNOMIAL:
            return self.min_lr_ratio + (1.0 - self.min_lr_ratio) * (1.0 - progress) ** self.power
        
        elif self.scheduler_type == SchedulerType.CONSTANT:
            return 1.0
        
        return 1.0
    
    def _create_scheduler(self) -> LambdaLR:
        """Create the PyTorch scheduler."""
        return LambdaLR(
            self.optimizer,
            lr_lambda=self._lr_lambda,
            last_epoch=self.last_epoch,
        )
    
    def step(self) -> None:
        """Step the scheduler."""
        self._scheduler.step()
    
    def get_lr(self) -> List[float]:
        """Get current learning rates."""
        return [group['lr'] for group in self.optimizer.param_groups]
    
    def get_last_lr(self) -> List[float]:
        """Get last computed learning rates."""
        return self._scheduler.get_last_lr()
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict."""
        return self._scheduler.state_dict()
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict."""
        self._scheduler.load_state_dict(state_dict)
    
    @classmethod
    def create_scheduler(
        cls,
        config: SchedulerConfig,
        optimizer: Optimizer,
    ) -> 'LearningRateScheduler':
        """
        Create scheduler from config.
        
        Args:
            config: Scheduler configuration
            optimizer: Optimizer to schedule
        
        Returns:
            LearningRateScheduler
        """
        return cls(
            scheduler_type=config.scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=config.num_warmup_steps,
            num_training_steps=config.num_training_steps,
            min_lr_ratio=config.min_lr_ratio,
            num_cycles=config.num_cycles,
            power=config.power,
            last_epoch=config.last_epoch,
        )


class OptimizerFactory:
    """
    Factory for creating optimizers.
    
    Supports multiple optimizer types with configurable parameters.
    
    Example:
        >>> config = OptimizerConfig(optimizer_type='adamw', learning_rate=1e-4)
        >>> optimizer = OptimizerFactory.create(config, model.parameters())
    """
    
    _optimizers = {
        'adam': torch.optim.Adam,
        'adamw': torch.optim.AdamW,
        'sgd': torch.optim.SGD,
        'adagrad': torch.optim.Adagrad,
        'rmsprop': torch.optim.RMSprop,
        'adadelta': torch.optim.Adadelta,
    }
    
    @classmethod
    def create(
        cls,
        config: OptimizerConfig,
        params: Union[torch.nn.ParameterList, List[Dict[str, Any]]],
    ) -> Optimizer:
        """
        Create optimizer from config.
        
        Args:
            config: Optimizer configuration
            params: Model parameters
        
        Returns:
            Optimizer instance
        """
        optimizer_type = config.optimizer_type.lower()
        
        if optimizer_type not in cls._optimizers:
            raise ValueError(f"Unknown optimizer: {optimizer_type}. Available: {list(cls._optimizers.keys())}")
        
        optimizer_cls = cls._optimizers[optimizer_type]
        
        # Build kwargs based on optimizer type
        kwargs = cls._build_kwargs(config, optimizer_type)
        
        return optimizer_cls(params, **kwargs)
    
    @classmethod
    def _build_kwargs(
        cls,
        config: OptimizerConfig,
        optimizer_type: str,
    ) -> Dict[str, Any]:
        """Build optimizer kwargs from config."""
        kwargs = {}
        
        if optimizer_type in ['adam', 'adamw']:
            kwargs['lr'] = config.learning_rate
            kwargs['betas'] = config.betas
            kwargs['eps'] = config.eps
            kwargs['weight_decay'] = config.weight_decay
            
            if optimizer_type == 'adamw':
                kwargs['correct_bias'] = config.correct_bias
        
        elif optimizer_type == 'sgd':
            kwargs['lr'] = config.learning_rate
            kwargs['momentum'] = config.momentum
            kwargs['weight_decay'] = config.weight_decay
            kwargs['nesterov'] = config.nesterov
            kwargs['dampening'] = config.dampening
        
        elif optimizer_type == 'rmsprop':
            kwargs['lr'] = config.learning_rate
            kwargs['momentum'] = config.momentum
            kwargs['eps'] = config.eps
            kwargs['weight_decay'] = config.weight_decay
        
        elif optimizer_type == 'adagrad':
            kwargs['lr'] = config.learning_rate
            kwargs['lr_decay'] = 0.0
            kwargs['weight_decay'] = config.weight_decay
            kwargs['eps'] = config.eps
        
        elif optimizer_type == 'adadelta':
            kwargs['lr'] = config.learning_rate
            kwargs['rho'] = config.betas[0]
            kwargs['eps'] = config.eps
            kwargs['weight_decay'] = config.weight_decay
        
        return kwargs
    
    @classmethod
    def create_with_weight_decay(
        cls,
        config: OptimizerConfig,
        model: nn.Module,
        no_decay_params: Optional[List[str]] = None,
    ) -> Optimizer:
        """
        Create optimizer with proper weight decay handling.
        
        Args:
            config: Optimizer configuration
            model: Model to optimize
            no_decay_params: Parameter names to exclude from weight decay
        
        Returns:
            Optimizer with parameter groups
        """
        no_decay = no_decay_params or ['bias', 'LayerNorm.weight', 'layer_norm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': config.weight_decay,
            },
            {
                'params': [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': 0.0,
            },
        ]
        
        return cls.create(config, optimizer_grouped_parameters)
    
    @classmethod
    def register_optimizer(cls, name: str, optimizer_cls: type) -> None:
        """Register a custom optimizer."""
        cls._optimizers[name.lower()] = optimizer_cls


class AdafactorOptimizer(torch.optim.Optimizer):
    """
    Adafactor Optimizer.
    
    Memory-efficient optimizer that adapts learning rate based on
    parameter scale. Useful for very large models.
    
    Reference:
        "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost" (Shazeer & Stern, 2018)
    """
    
    def __init__(
        self,
        params,
        lr: Optional[float] = None,
        eps: Tuple[float, float] = (1e-30, 1e-3),
        clip_threshold: float = 1.0,
        decay_rate: float = -0.8,
        beta1: Optional[float] = None,
        weight_decay: float = 0.0,
        scale_parameter: bool = True,
        relative_step: bool = True,
        warmup_init: bool = False,
    ):
        defaults = {
            'lr': lr,
            'eps': eps,
            'clip_threshold': clip_threshold,
            'decay_rate': decay_rate,
            'beta1': beta1,
            'weight_decay': weight_decay,
            'scale_parameter': scale_parameter,
            'relative_step': relative_step,
            'warmup_init': warmup_init,
        }
        super().__init__(params, defaults)
    
    def _get_lr(self, param_group: Dict, param_shape: Tuple) -> float:
        """Get learning rate for parameter."""
        if param_group['relative_step']:
            min_step = 1e-6 * param_group['step'] if param_group['warmup_init'] else 1e-2
            relative_step = min(min_step, param_group['step'] ** -0.5)
            
            if param_group['scale_parameter']:
                lr = relative_step * self._get_scale(param_shape)
            else:
                lr = relative_step
        else:
            lr = param_group['lr'] if param_group['lr'] is not None else 1e-3
        
        return lr
    
    def _get_scale(self, param_shape: Tuple) -> float:
        """Get scale factor for learning rate."""
        return math.sqrt(max(param_shape))
    
    def _get_rms(self, grads: Tensor) -> Tensor:
        """Get RMS of gradients."""
        return grads.pow(2).mean().sqrt()
    
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Optimizer step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adafactor does not support sparse gradients.')
                
                state = self.state[p]
                shape = grad.shape
                
                # Initialize state
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(grad)
                    state['exp_avg_sq_row'] = None
                    state['exp_avg_sq_col'] = None
                    
                    if len(shape) >= 2:
                        state['exp_avg_sq_row'] = torch.zeros(shape[:-1], device=grad.device, dtype=grad.dtype)
                        state['exp_avg_sq_col'] = torch.zeros(shape[-1:], device=grad.device, dtype=grad.dtype)
                
                state['step'] += 1
                
                # Get learning rate
                lr = self._get_lr(group, shape)
                
                # Apply weight decay
                if group['weight_decay'] != 0:
                    grad = grad.add(p.data, alpha=group['weight_decay'])
                
                # Update exponential moving average
                beta2 = 1.0 - (state['step'] ** group['decay_rate'])
                
                if len(shape) >= 2:
                    # Factorized update for large tensors
                    exp_avg_sq_row = state['exp_avg_sq_row']
                    exp_avg_sq_col = state['exp_avg_sq_col']
                    
                    exp_avg_sq_row.mul_(beta2).add_(grad.pow(2).mean(dim=-1), alpha=1.0 - beta2)
                    exp_avg_sq_col.mul_(beta2).add_(grad.pow(2).mean(dim=tuple(range(grad.dim() - 1))), alpha=1.0 - beta2)
                    
                    # Combine row and column statistics
                    avg_sq = torch.matmul(
                        exp_avg_sq_row.unsqueeze(-1),
                        exp_avg_sq_col.unsqueeze(0),
                    )
                else:
                    # Standard update for small tensors
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2).add_(grad.pow(2), alpha=1.0 - beta2)
                    avg_sq = exp_avg_sq
                
                # Apply clipping
                rms = self._get_rms(grad)
                clip = max(group['clip_threshold'], rms)
                grad = grad / max(1.0, rms / clip)
                
                # Update parameters
                if group['beta1'] is None:
                    # No momentum
                    p.data.addcdiv_(grad, avg_sq.sqrt() + group['eps'][1], value=-lr)
                else:
                    # With momentum
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(grad)
                    
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(grad, alpha=1.0 - group['beta1'])
                    
                    p.data.addcdiv_(exp_avg, avg_sq.sqrt() + group['eps'][1], value=-lr)
        
        return loss


class LionOptimizer(torch.optim.Optimizer):
    """
    Lion Optimizer.
    
    Memory-efficient optimizer that uses sign-based updates.
    
    Reference:
        "Symbolic Discovery of Optimization Algorithms" (Chen et al., 2023)
    """
    
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ):
        defaults = {
            'lr': lr,
            'betas': betas,
            'weight_decay': weight_decay,
        }
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure: Optional[Callable] = None) -> Optional[float]:
        """Optimizer step."""
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize state
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                
                # Update momentum
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Compute update (sign of momentum)
                update = torch.sign(exp_avg)
                
                # Apply weight decay
                if group['weight_decay'] > 0:
                    update.add_(p, alpha=group['weight_decay'])
                
                # Update parameters
                p.add_(update, alpha=-group['lr'])
        
        return loss


def get_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    no_decay_params: Optional[List[str]] = None,
) -> Optimizer:
    """
    Convenience function to create optimizer.
    
    Args:
        model: Model to optimize
        optimizer_type: Type of optimizer
        learning_rate: Learning rate
        weight_decay: Weight decay
        betas: Adam betas
        eps: Adam epsilon
        no_decay_params: Parameters to exclude from weight decay
    
    Returns:
        Optimizer
    """
    config = OptimizerConfig(
        optimizer_type=optimizer_type,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
    )
    
    if no_decay_params is not None:
        return OptimizerFactory.create_with_weight_decay(config, model, no_decay_params)
    
    return OptimizerFactory.create(config, model.parameters())


def get_scheduler(
    optimizer: Optimizer,
    scheduler_type: str = "cosine",
    num_warmup_steps: int = 1000,
    num_training_steps: int = 100000,
    min_lr_ratio: float = 0.1,
) -> LearningRateScheduler:
    """
    Convenience function to create scheduler.
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_type: Type of scheduler
        num_warmup_steps: Number of warmup steps
        num_training_steps: Total training steps
        min_lr_ratio: Minimum LR ratio
    
    Returns:
        LearningRateScheduler
    """
    return LearningRateScheduler(
        scheduler_type=SchedulerType(scheduler_type),
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        min_lr_ratio=min_lr_ratio,
    )
