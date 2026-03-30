"""
Training Monitoring - Module 2.2.4

Production-ready monitoring implementations:
- Loss tracking and visualization
- Gradient monitoring
- GPU utilization tracking
- Memory profiling
- Weights & Biases integration

References:
- "Weights & Biases: Experiment Tracking" (wandb.ai)
- "PyTorch Profiler" (pytorch.org)
"""

import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training metrics container."""
    step: int = 0
    epoch: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    gradient_norm: float = 0.0
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0
    gpu_memory_used: float = 0.0
    gpu_utilization: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'step': self.step,
            'epoch': self.epoch,
            'loss': self.loss,
            'learning_rate': self.learning_rate,
            'gradient_norm': self.gradient_norm,
            'samples_per_second': self.samples_per_second,
            'tokens_per_second': self.tokens_per_second,
            'gpu_memory_used': self.gpu_memory_used,
            'gpu_utilization': self.gpu_utilization,
        }


class LossTracker:
    """
    Loss Tracker for monitoring training progress.
    
    Tracks loss values with smoothing and provides statistics.
    
    Args:
        window_size: Size of sliding window for smoothing
        log_interval: Interval for logging
        
    Example:
        >>> tracker = LossTracker(window_size=100)
        >>> tracker.update(loss=2.5, step=1)
        >>> print(tracker.get_average())
    """
    
    def __init__(
        self,
        window_size: int = 100,
        log_interval: int = 10,
    ):
        self.window_size = window_size
        self.log_interval = log_interval
        
        self._loss_history: deque = deque(maxlen=window_size)
        self._step_history: deque = deque(maxlen=window_size)
        self._total_loss = 0.0
        self._total_steps = 0
        
        # Per-split tracking
        self._split_losses: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
    
    def update(
        self,
        loss: float,
        step: int,
        split: str = 'train',
    ) -> None:
        """
        Update loss tracker.
        
        Args:
            loss: Current loss value
            step: Current step
            split: Data split (train, val, test)
        """
        self._loss_history.append(loss)
        self._step_history.append(step)
        self._split_losses[split].append(loss)
        
        self._total_loss += loss
        self._total_steps += 1
    
    def get_average(self, split: str = 'train') -> float:
        """Get average loss over window."""
        if split in self._split_losses and self._split_losses[split]:
            return sum(self._split_losses[split]) / len(self._split_losses[split])
        return 0.0
    
    def get_global_average(self) -> float:
        """Get global average loss."""
        if self._total_steps == 0:
            return 0.0
        return self._total_loss / self._total_steps
    
    def get_std(self, split: str = 'train') -> float:
        """Get standard deviation of loss."""
        losses = list(self._split_losses.get(split, []))
        if len(losses) < 2:
            return 0.0
        
        mean = sum(losses) / len(losses)
        variance = sum((x - mean) ** 2 for x in losses) / len(losses)
        return variance ** 0.5
    
    def get_trend(self, window: int = 10) -> str:
        """
        Get loss trend (increasing, decreasing, stable).
        
        Args:
            window: Window size for trend calculation
        
        Returns:
            Trend string
        """
        losses = list(self._loss_history)
        if len(losses) < window * 2:
            return 'unknown'
        
        recent = sum(losses[-window:]) / window
        older = sum(losses[-window*2:-window]) / window
        
        change = (older - recent) / max(older, 1e-8)
        
        if change > 0.05:
            return 'decreasing'
        elif change < -0.05:
            return 'increasing'
        return 'stable'
    
    def reset(self) -> None:
        """Reset tracker."""
        self._loss_history.clear()
        self._step_history.clear()
        self._split_losses.clear()
        self._total_loss = 0.0
        self._total_steps = 0
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict."""
        return {
            'loss_history': list(self._loss_history),
            'step_history': list(self._step_history),
            'total_loss': self._total_loss,
            'total_steps': self._total_steps,
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load state dict."""
        self._loss_history = deque(state_dict.get('loss_history', []), maxlen=self.window_size)
        self._step_history = deque(state_dict.get('step_history', []), maxlen=self.window_size)
        self._total_loss = state_dict.get('total_loss', 0.0)
        self._total_steps = state_dict.get('total_steps', 0)


class GradientMonitor:
    """
    Gradient Monitor for tracking gradient health.
    
    Monitors gradient norms, vanishing/exploding gradients,
    and gradient distributions.
    
    Args:
        model: Model to monitor
        log_interval: Interval for logging
        
    Example:
        >>> monitor = GradientMonitor(model)
        >>> stats = monitor.get_stats()
        >>> monitor.check_health()
    """
    
    def __init__(
        self,
        model: nn.Module,
        log_interval: int = 100,
    ):
        self.model = model
        self.log_interval = log_interval
        
        self._grad_history: deque = deque(maxlen=100)
        self._param_stats: Dict[str, List[float]] = defaultdict(list)
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get gradient statistics.
        
        Returns:
            Dictionary of gradient statistics
        """
        stats = {
            'total_norm': 0.0,
            'max_norm': 0.0,
            'min_norm': float('inf'),
            'mean_norm': 0.0,
            'num_zeros': 0,
            'num_nans': 0,
        }
        
        norms = []
        zero_count = 0
        nan_count = 0
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad = param.grad
                
                # Check for NaN/Inf
                if torch.isnan(grad).any():
                    nan_count += 1
                    logger.warning(f"NaN gradients in {name}")
                
                if torch.isinf(grad).any():
                    nan_count += 1
                    logger.warning(f"Inf gradients in {name}")
                
                # Compute norm
                norm = grad.norm().item()
                norms.append(norm)
                
                # Check for zero gradients
                if norm < 1e-10:
                    zero_count += 1
                
                # Track per-parameter stats
                self._param_stats[name].append(norm)
        
        if norms:
            stats['total_norm'] = sum(n**2 for n in norms) ** 0.5
            stats['max_norm'] = max(norms)
            stats['min_norm'] = min(norms)
            stats['mean_norm'] = sum(norms) / len(norms)
        
        stats['num_zeros'] = zero_count
        stats['num_nans'] = nan_count
        
        # Store history
        self._grad_history.append(stats['total_norm'])
        
        return stats
    
    def check_health(self) -> Dict[str, bool]:
        """
        Check gradient health.
        
        Returns:
            Dictionary of health checks
        """
        stats = self.get_stats()
        
        return {
            'has_gradients': stats['total_norm'] > 0,
            'no_nans': stats['num_nans'] == 0,
            'no_all_zeros': stats['num_zeros'] < len(list(self.model.parameters())),
            'not_exploding': stats['max_norm'] < 1000,
            'not_vanishing': stats['mean_norm'] > 1e-7,
        }
    
    def get_gradient_histogram(
        self,
        num_bins: int = 100,
    ) -> Tuple[List[float], List[float]]:
        """
        Get histogram of gradient values.
        
        Args:
            num_bins: Number of bins
        
        Returns:
            Tuple of (bin_edges, counts)
        """
        all_grads = []
        
        for param in self.model.parameters():
            if param.grad is not None:
                all_grads.append(param.grad.flatten())
        
        if not all_grads:
            return [], []
        
        all_grads = torch.cat(all_grads).cpu().numpy()
        counts, bin_edges = torch.histogram(torch.from_numpy(all_grads), bins=num_bins)
        
        return bin_edges.tolist(), counts.tolist()
    
    def log_stats(self, step: int) -> None:
        """Log gradient statistics."""
        if step % self.log_interval == 0:
            stats = self.get_stats()
            health = self.check_health()
            
            logger.info(f"Step {step} - Gradient Stats:")
            logger.info(f"  Total norm: {stats['total_norm']:.6f}")
            logger.info(f"  Max norm: {stats['max_norm']:.6f}")
            logger.info(f"  Mean norm: {stats['mean_norm']:.6f}")
            logger.info(f"  Zero gradients: {stats['num_zeros']}")
            logger.info(f"  NaN/Inf: {stats['num_nans']}")
            logger.info(f"  Health: {health}")


class GPUMonitor:
    """
    GPU Monitor for tracking GPU utilization.
    
    Monitors GPU memory, utilization, and temperature.
    
    Args:
        gpu_ids: GPU IDs to monitor
        log_interval: Interval for logging
        
    Example:
        >>> monitor = GPUMonitor(gpu_ids=[0, 1])
        >>> stats = monitor.get_stats()
    """
    
    def __init__(
        self,
        gpu_ids: Optional[List[int]] = None,
        log_interval: int = 10,
    ):
        self.gpu_ids = gpu_ids or [0]
        self.log_interval = log_interval
        
        self._history: deque = deque(maxlen=100)
        self._nvml_initialized = False
        
        # Try to initialize NVML
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml_initialized = True
            self._nvml = pynvml
        except ImportError:
            logger.warning("pynvml not installed. Install with: pip install nvidia-ml-py3")
        except Exception:
            logger.warning("Failed to initialize NVML")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get GPU statistics.
        
        Returns:
            Dictionary of GPU stats
        """
        stats = {
            'gpus': [],
            'total_memory_used': 0,
            'avg_utilization': 0,
        }
        
        if self._nvml_initialized:
            for gpu_id in self.gpu_ids:
                try:
                    handle = self._nvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    
                    # Memory info
                    mem_info = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                    memory_used = mem_info.used / (1024 ** 3)  # GB
                    memory_total = mem_info.total / (1024 ** 3)  # GB
                    
                    # Utilization
                    util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                    mem_util = util.memory
                    
                    # Temperature
                    temp = self._nvml.nvmlDeviceGetTemperature(handle, self._nvml.NVML_TEMPERATURE_GPU)
                    
                    gpu_stats = {
                        'id': gpu_id,
                        'memory_used_gb': memory_used,
                        'memory_total_gb': memory_total,
                        'memory_utilization': mem_util,
                        'gpu_utilization': gpu_util,
                        'temperature': temp,
                    }
                    
                    stats['gpus'].append(gpu_stats)
                    stats['total_memory_used'] += memory_used
                    stats['avg_utilization'] += gpu_util
                
                except Exception as e:
                    logger.warning(f"Failed to get stats for GPU {gpu_id}: {e}")
            
            if stats['gpus']:
                stats['avg_utilization'] /= len(stats['gpus'])
        
        else:
            # Fallback to PyTorch
            for gpu_id in self.gpu_ids:
                if torch.cuda.is_available():
                    memory_used = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
                    memory_reserved = torch.cuda.memory_reserved(gpu_id) / (1024 ** 3)
                    
                    stats['gpus'].append({
                        'id': gpu_id,
                        'memory_used_gb': memory_used,
                        'memory_reserved_gb': memory_reserved,
                    })
                    stats['total_memory_used'] += memory_used
        
        self._history.append(stats)
        return stats
    
    def get_peak_memory(self) -> float:
        """Get peak memory usage across all GPUs."""
        peak = 0.0
        
        for gpu_id in self.gpu_ids:
            if torch.cuda.is_available():
                peak = max(peak, torch.cuda.max_memory_allocated(gpu_id) / (1024 ** 3))
        
        return peak
    
    def reset_peak_memory(self) -> None:
        """Reset peak memory tracking."""
        for gpu_id in self.gpu_ids:
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(gpu_id)
    
    def log_stats(self, step: int) -> None:
        """Log GPU statistics."""
        if step % self.log_interval == 0:
            stats = self.get_stats()
            
            logger.info(f"Step {step} - GPU Stats:")
            for gpu in stats['gpus']:
                logger.info(f"  GPU {gpu['id']}: {gpu.get('memory_used_gb', 0):.2f}GB used, "
                           f"{gpu.get('gpu_utilization', 0):.1f}% util")
    
    def cleanup(self) -> None:
        """Cleanup resources."""
        if self._nvml_initialized:
            self._nvml.nvmlShutdown()


class MemoryProfiler:
    """
    Memory Profiler for detailed memory analysis.
    
    Profiles memory allocation, identifies memory leaks,
    and tracks memory by component.
    
    Args:
        enabled: Whether to enable profiling
        profile_interval: Interval for profiling
        
    Example:
        >>> profiler = MemoryProfiler(enabled=True)
        >>> with profiler.profile("forward"):
        ...     output = model(input)
    """
    
    def __init__(
        self,
        enabled: bool = True,
        profile_interval: int = 100,
    ):
        self.enabled = enabled and torch.cuda.is_available()
        self.profile_interval = profile_interval
        
        self._snapshots: Dict[str, List[Dict]] = defaultdict(list)
        self._timings: Dict[str, List[float]] = defaultdict(list)
        
        # PyTorch profiler
        self._profiler: Optional[torch.profiler.profile] = None
    
    def snapshot(self, label: str = 'default') -> Dict[str, float]:
        """
        Take memory snapshot.
        
        Args:
            label: Label for snapshot
        
        Returns:
            Memory snapshot dict
        """
        if not self.enabled:
            return {}
        
        snapshot = {
            'allocated': torch.cuda.memory_allocated() / (1024 ** 2),  # MB
            'reserved': torch.cuda.memory_reserved() / (1024 ** 2),  # MB
            'max_allocated': torch.cuda.max_memory_allocated() / (1024 ** 2),
            'time': time.time(),
        }
        
        self._snapshots[label].append(snapshot)
        return snapshot
    
    def profile(self, label: str):
        """
        Context manager for profiling a block.
        
        Args:
            label: Label for the profiled block
        
        Example:
            >>> with profiler.profile("training_step"):
            ...     loss = compute_loss()
        """
        return MemoryProfileContext(self, label)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get memory profiling summary.
        
        Returns:
            Summary dictionary
        """
        summary = {
            'current_allocated_mb': torch.cuda.memory_allocated() / (1024 ** 2),
            'current_reserved_mb': torch.cuda.memory_reserved() / (1024 ** 2),
            'peak_allocated_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
            'snapshots': {},
            'timings': {},
        }
        
        for label, snapshots in self._snapshots.items():
            if snapshots:
                summary['snapshots'][label] = {
                    'count': len(snapshots),
                    'avg_allocated': sum(s['allocated'] for s in snapshots) / len(snapshots),
                    'max_allocated': max(s['allocated'] for s in snapshots),
                }
        
        for label, timings in self._timings.items():
            if timings:
                summary['timings'][label] = {
                    'count': len(timings),
                    'total_time': sum(timings),
                    'avg_time': sum(timings) / len(timings),
                    'max_time': max(timings),
                }
        
        return summary
    
    def export_trace(self, path: str) -> None:
        """
        Export profiling trace.
        
        Args:
            path: Path to save trace
        """
        trace = {
            'summary': self.get_summary(),
            'snapshots': dict(self._snapshots),
            'timings': dict(self._timings),
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(trace, f, indent=2)
        
        logger.info(f"Memory trace exported to {path}")
    
    def start_profiler(self) -> None:
        """Start PyTorch profiler."""
        if self.enabled:
            self._profiler = torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs/profiler'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
            )
            self._profiler.start()
    
    def step_profiler(self) -> None:
        """Step PyTorch profiler."""
        if self._profiler is not None:
            self._profiler.step()
    
    def stop_profiler(self) -> None:
        """Stop PyTorch profiler."""
        if self._profiler is not None:
            self._profiler.stop()
            self._profiler = None


class MemoryProfileContext:
    """Context manager for memory profiling."""
    
    def __init__(self, profiler: MemoryProfiler, label: str):
        self.profiler = profiler
        self.label = label
        self.start_time: Optional[float] = None
        self.start_memory: Optional[Dict] = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.start_memory = self.profiler.snapshot(f"{self.label}_start")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        end_memory = self.profiler.snapshot(f"{self.label}_end")
        
        if self.start_memory and end_memory:
            duration = end_time - self.start_time
            self.profiler._timings[self.label].append(duration)
        
        return False


class WandBLogger:
    """
    Weights & Biases Logger for experiment tracking.
    
    Integrates with W&B for experiment tracking, visualization,
    and collaboration.
    
    Args:
        project: W&B project name
        entity: W&B entity/team name
        config: Configuration to log
        mode: W&B mode ('online', 'offline', 'disabled')
        
    Example:
        >>> logger = WandBLogger(project='llm-training', config=cfg)
        >>> logger.log({'loss': 2.5, 'step': 100})
    """
    
    def __init__(
        self,
        project: str = 'llm-training',
        entity: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        mode: str = 'online',
        tags: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        self.project = project
        self.entity = entity
        self.mode = mode
        self.tags = tags
        self.name = name
        
        self._initialized = False
        self._run = None
        
        # Try to initialize W&B
        try:
            import wandb
            
            self._wandb = wandb
            
            # Initialize
            self._run = wandb.init(
                project=project,
                entity=entity,
                config=config,
                mode=mode,
                tags=tags,
                name=name,
            )
            self._initialized = True
            
            logger.info(f"W&B initialized: {wandb.run.get_url()}")
        
        except ImportError:
            logger.warning("wandb not installed. Install with: pip install wandb")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")
    
    def log(
        self,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
    ) -> None:
        """
        Log metrics to W&B.
        
        Args:
            metrics: Metrics dictionary
            step: Optional step number
        """
        if not self._initialized:
            return
        
        try:
            self._wandb.log(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log to W&B: {e}")
    
    def log_model(
        self,
        model: nn.Module,
        name: str = 'model',
    ) -> None:
        """
        Log model to W&B.
        
        Args:
            model: Model to log
            name: Model name
        """
        if not self._initialized:
            return
        
        try:
            # Log model architecture
            model_artifact = self._wandb.Artifact(name, type='model')
            
            # Save model config
            config_path = f'/tmp/{name}_config.json'
            with open(config_path, 'w') as f:
                json.dump(str(model), f)
            model_artifact.add_file(config_path)
            
            self._wandb.log_artifact(model_artifact)
        except Exception as e:
            logger.warning(f"Failed to log model to W&B: {e}")
    
    def log_table(
        self,
        name: str,
        columns: List[str],
        data: List[List[Any]],
    ) -> None:
        """
        Log table to W&B.
        
        Args:
            name: Table name
            columns: Column names
            data: Table data
        """
        if not self._initialized:
            return
        
        try:
            table = self._wandb.Table(columns=columns, data=data)
            self._wandb.log({name: table})
        except Exception as e:
            logger.warning(f"Failed to log table to W&B: {e}")
    
    def watch(
        self,
        model: nn.Module,
        log: str = 'gradients',
        log_freq: int = 100,
    ) -> None:
        """
        Watch model parameters and gradients.
        
        Args:
            model: Model to watch
            log: What to log ('gradients', 'parameters', 'all')
            log_freq: Logging frequency
        """
        if not self._initialized:
            return
        
        try:
            self._wandb.watch(model, log=log, log_freq=log_freq)
        except Exception as e:
            logger.warning(f"Failed to watch model: {e}")
    
    def finish(self) -> None:
        """Finish W&B run."""
        if self._initialized and self._run is not None:
            self._wandb.finish()
            logger.info("W&B run finished")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False


class TensorBoardLogger:
    """
    TensorBoard Logger for experiment tracking.
    
    Args:
        log_dir: Directory for logs
        flush_secs: Flush interval in seconds
        
    Example:
        >>> logger = TensorBoardLogger(log_dir='./logs')
        >>> logger.log_scalar('loss', 2.5, step=100)
    """
    
    def __init__(
        self,
        log_dir: str = './logs',
        flush_secs: int = 30,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(
            str(self.log_dir),
            flush_secs=flush_secs,
        )
        
        logger.info(f"TensorBoard logs at: {self.log_dir}")
    
    def log_scalar(
        self,
        tag: str,
        value: float,
        step: int,
    ) -> None:
        """Log scalar value."""
        self.writer.add_scalar(tag, value, step)
    
    def log_scalars(
        self,
        tag: str,
        values: Dict[str, float],
        step: int,
    ) -> None:
        """Log multiple scalars."""
        self.writer.add_scalars(tag, values, step)
    
    def log_histogram(
        self,
        tag: str,
        values: torch.Tensor,
        step: int,
    ) -> None:
        """Log histogram."""
        self.writer.add_histogram(tag, values, step)
    
    def log_image(
        self,
        tag: str,
        image: torch.Tensor,
        step: int,
    ) -> None:
        """Log image."""
        self.writer.add_image(tag, image, step)
    
    def log_text(
        self,
        tag: str,
        text: str,
        step: int,
    ) -> None:
        """Log text."""
        self.writer.add_text(tag, text, step)
    
    def log_model(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
    ) -> None:
        """Log model graph."""
        self.writer.add_graph(model, example_input)
    
    def close(self) -> None:
        """Close writer."""
        self.writer.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


class TrainingMonitor:
    """
    Comprehensive Training Monitor.
    
    Combines all monitoring components for complete training visibility.
    
    Args:
        config: Monitoring configuration
        log_dir: Directory for logs
        
    Example:
        >>> monitor = TrainingMonitor(log_dir='./logs')
        >>> monitor.start()
        >>> for step, batch in enumerate(dataloader):
        ...     loss = train_step(batch)
        ...     monitor.update(step, loss=loss)
    """
    
    def __init__(
        self,
        log_dir: str = './logs',
        log_interval: int = 10,
        use_wandb: bool = False,
        wandb_config: Optional[Dict[str, Any]] = None,
    ):
        self.log_dir = Path(log_dir)
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        
        # Initialize components
        self.loss_tracker = LossTracker(log_interval=log_interval)
        self.gradient_monitor: Optional[GradientMonitor] = None
        self.gpu_monitor = GPUMonitor(log_interval=log_interval)
        self.memory_profiler = MemoryProfiler(enabled=True)
        
        # Loggers
        self.tb_logger = TensorBoardLogger(log_dir=str(self.log_dir / 'tensorboard'))
        self.wandb_logger: Optional[WandBLogger] = None
        
        if use_wandb:
            wandb_config = wandb_config or {}
            self.wandb_logger = WandBLogger(
                project=wandb_config.get('project', 'llm-training'),
                entity=wandb_config.get('entity'),
                config=wandb_config.get('config'),
            )
        
        # Timing
        self._start_time: Optional[float] = None
        self._last_step_time: Optional[float] = None
        self._step_times: deque = deque(maxlen=100)
    
    def start(
        self,
        model: Optional[nn.Module] = None,
    ) -> None:
        """
        Start monitoring.
        
        Args:
            model: Optional model for gradient monitoring
        """
        self._start_time = time.time()
        self._last_step_time = time.time()
        
        if model is not None:
            self.gradient_monitor = GradientMonitor(model, log_interval=self.log_interval)
        
        logger.info("Training monitor started")
    
    def update(
        self,
        step: int,
        epoch: int = 0,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        batch_size: int = 1,
        num_tokens: int = 0,
    ) -> TrainingMetrics:
        """
        Update monitor with new metrics.
        
        Args:
            step: Current step
            epoch: Current epoch
            loss: Current loss
            learning_rate: Current learning rate
            gradient_norm: Gradient norm
            batch_size: Batch size
            num_tokens: Number of tokens
        
        Returns:
            TrainingMetrics
        """
        # Update timing
        current_time = time.time()
        if self._last_step_time is not None:
            step_time = current_time - self._last_step_time
            self._step_times.append(step_time)
        self._last_step_time = current_time
        
        # Calculate throughput
        elapsed = current_time - self._start_time if self._start_time else 1
        samples_per_second = batch_size / max(step_time, 1e-6) if step_time else 0
        tokens_per_second = num_tokens / max(step_time, 1e-6) if step_time else 0
        
        # Update loss tracker
        if loss is not None:
            self.loss_tracker.update(loss, step)
        
        # Get GPU stats
        gpu_stats = self.gpu_monitor.get_stats()
        gpu_memory = gpu_stats.get('total_memory_used', 0)
        gpu_util = gpu_stats.get('avg_utilization', 0)
        
        # Create metrics
        metrics = TrainingMetrics(
            step=step,
            epoch=epoch,
            loss=loss or 0,
            learning_rate=learning_rate or 0,
            gradient_norm=gradient_norm or 0,
            samples_per_second=samples_per_second,
            tokens_per_second=tokens_per_second,
            gpu_memory_used=gpu_memory,
            gpu_utilization=gpu_util,
        )
        
        # Log periodically
        if step % self.log_interval == 0:
            self._log_metrics(metrics, step)
        
        return metrics
    
    def _log_metrics(
        self,
        metrics: TrainingMetrics,
        step: int,
    ) -> None:
        """Log metrics to all loggers."""
        # TensorBoard
        self.tb_logger.log_scalar('loss', metrics.loss, step)
        self.tb_logger.log_scalar('learning_rate', metrics.learning_rate, step)
        self.tb_logger.log_scalar('gradient_norm', metrics.gradient_norm, step)
        self.tb_logger.log_scalar('samples_per_second', metrics.samples_per_second, step)
        self.tb_logger.log_scalar('tokens_per_second', metrics.tokens_per_second, step)
        self.tb_logger.log_scalar('gpu_memory_gb', metrics.gpu_memory_used, step)
        self.tb_logger.log_scalar('gpu_utilization', metrics.gpu_utilization, step)
        
        # W&B
        if self.wandb_logger:
            self.wandb_logger.log(metrics.to_dict(), step=step)
        
        # Console
        avg_loss = self.loss_tracker.get_average()
        loss_trend = self.loss_tracker.get_trend()
        
        logger.info(
            f"Step {step} | Loss: {metrics.loss:.4f} (avg: {avg_loss:.4f}, {loss_trend}) | "
            f"LR: {metrics.learning_rate:.2e} | "
            f"Grad: {metrics.gradient_norm:.4f} | "
            f"Throughput: {metrics.samples_per_second:.1f} samples/s | "
            f"GPU: {metrics.gpu_memory_used:.1f}GB"
        )
    
    def log_gradients(self, step: int) -> None:
        """Log gradient statistics."""
        if self.gradient_monitor:
            stats = self.gradient_monitor.get_stats()
            
            self.tb_logger.log_scalar('gradient_total_norm', stats['total_norm'], step)
            self.tb_logger.log_scalar('gradient_max_norm', stats['max_norm'], step)
            self.tb_logger.log_scalar('gradient_zeros', stats['num_zeros'], step)
            
            if self.wandb_logger:
                self.wandb_logger.log({
                    'gradient/total_norm': stats['total_norm'],
                    'gradient/max_norm': stats['max_norm'],
                    'gradient/zeros': stats['num_zeros'],
                }, step=step)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            'loss': {
                'current': self.loss_tracker.get_average(),
                'global_avg': self.loss_tracker.get_global_average(),
                'std': self.loss_tracker.get_std(),
                'trend': self.loss_tracker.get_trend(),
            },
            'memory': self.memory_profiler.get_summary(),
            'gpu': self.gpu_monitor.get_stats(),
            'timing': {
                'avg_step_time': sum(self._step_times) / max(len(self._step_times), 1),
                'total_time': time.time() - self._start_time if self._start_time else 0,
            },
        }
    
    def finish(self) -> None:
        """Finish monitoring."""
        self.tb_logger.close()
        if self.wandb_logger:
            self.wandb_logger.finish()
        self.gpu_monitor.cleanup()
        
        # Export summary
        summary = self.get_summary()
        summary_path = self.log_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Training monitor finished. Summary at {summary_path}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False
