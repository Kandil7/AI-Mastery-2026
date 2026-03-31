"""
LoRA (Low-Rank Adaptation) - Module 2.4.2

Production-ready LoRA implementation from scratch:
- LoRA configuration
- LoRA linear layer
- LoRA model wrapper
- LoRA trainer

References:
- "LoRA: Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
"""

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for LoRA."""
    # LoRA dimensions
    r: int = 8  # Rank
    alpha: float = 16.0  # Scaling factor
    dropout: float = 0.05  # Dropout probability
    
    # Target modules
    target_modules: List[str] = field(default_factory=lambda: ['q_proj', 'v_proj'])
    modules_to_save: Optional[List[str]] = None
    
    # Initialization
    bias: str = 'none'  # 'none', 'all', 'lora_only'
    init_weights: bool = True
    
    # Type
    task_type: str = 'CAUSAL_LM'
    
    @property
    def scaling(self) -> float:
        """Get LoRA scaling factor."""
        return self.alpha / self.r


class LoRALayer(nn.Module):
    """
    Base LoRA layer.
    
    Implements the LoRA decomposition: W + BA
    where W is frozen, and B, A are trainable.
    
    Args:
        r: Rank of the decomposition
        alpha: Scaling factor
        dropout: Dropout probability
        merge_weights: Whether to merge weights for inference
        
    Example:
        >>> lora = LoRALayer(in_features=768, out_features=768, r=8)
        >>> output = lora(input)
    """
    
    def __init__(
        self,
        r: int,
        alpha: float,
        dropout: float = 0.0,
        merge_weights: bool = True,
    ):
        super().__init__()
        
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.dropout = dropout
        self.merge_weights = merge_weights
        
        self.merged = False
    
    def reset_parameters(self) -> None:
        """Reset LoRA parameters."""
        pass
    
    def merge(self) -> None:
        """Merge LoRA weights with base weights."""
        pass
    
    def unmerge(self) -> None:
        """Unmerge LoRA weights."""
        pass


class LoRALinear(nn.Module):
    """
    LoRA-enabled Linear layer.
    
    Replaces a standard linear layer with LoRA decomposition.
    The original weights are frozen, and low-rank matrices are trained.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        r: Rank of LoRA decomposition
        alpha: Scaling factor
        dropout: Dropout probability
        bias: Whether to use bias
        merge_weights: Whether to merge weights for inference
        
    Example:
        >>> lora_linear = LoRALinear(768, 768, r=8, alpha=16)
        >>> output = lora_linear(input)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
        bias: bool = False,
        merge_weights: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        self.merge_weights = merge_weights
        
        # Original linear layer (frozen)
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features),
            requires_grad=False,
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # LoRA matrices
        # A: (r, in_features) - initialized with Kaiming uniform
        # B: (out_features, r) - initialized with zeros
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Reset parameters
        self.reset_parameters()
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.merged = False
    
    def reset_parameters(self) -> None:
        """Initialize LoRA parameters."""
        # Initialize A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # Initialize B with zeros (important for starting from identity transformation)
        nn.init.zeros_(self.lora_B)
    
    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        
        if mode:
            # Unmerge when training
            if self.merge_weights and self.merged:
                self.unmerge()
        else:
            # Merge when evaluating
            if self.merge_weights and not self.merged:
                self.merge()
        
        return self
    
    def merge(self) -> None:
        """Merge LoRA weights with base weights."""
        if self.merged:
            return
        
        # W_merged = W + (B @ A) * scaling
        delta_weight = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        self.weight.data.add_(delta_weight)
        
        self.merged = True
    
    def unmerge(self) -> None:
        """Unmerge LoRA weights."""
        if not self.merged:
            return
        
        # W_original = W_merged - (B @ A) * scaling
        delta_weight = torch.matmul(self.lora_B, self.lora_A) * self.scaling
        self.weight.data.sub_(delta_weight)
        
        self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if self.merged:
            # Use merged weights
            return F.linear(x, self.weight, self.bias)
        else:
            # Compute separately: Wx + BAx * scaling
            base_output = F.linear(x, self.weight, self.bias)
            lora_output = F.linear(x, self.lora_A)
            lora_output = F.linear(lora_output, self.lora_B)
            lora_output = lora_output * self.scaling
            lora_output = self.dropout(lora_output)
            
            return base_output + lora_output
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ) -> 'LoRALinear':
        """
        Create LoRALinear from existing Linear layer.
        
        Args:
            linear: Original linear layer
            r: Rank
            alpha: Scaling factor
            dropout: Dropout probability
        
        Returns:
            LoRALinear with copied weights
        """
        lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            bias=linear.bias is not None,
        )
        
        # Copy original weights
        lora_linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            lora_linear.bias.data.copy_(linear.bias.data)
        
        return lora_linear


class LoRAEmbedding(nn.Module):
    """
    LoRA-enabled Embedding layer.
    
    Applies LoRA to embedding layers.
    
    Args:
        num_embeddings: Vocabulary size
        embedding_dim: Embedding dimension
        r: Rank of LoRA decomposition
        alpha: Scaling factor
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        r: int = 8,
        alpha: float = 16.0,
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        
        # Original embedding (frozen)
        self.weight = nn.Parameter(
            torch.zeros(num_embeddings, embedding_dim),
            requires_grad=False,
        )
        
        # LoRA matrices for embeddings
        # For embeddings, we use a different approach:
        # LoRA_A: (r, embedding_dim)
        # LoRA_B: (num_embeddings, r)
        self.lora_A = nn.Parameter(torch.zeros(r, embedding_dim))
        self.lora_B = nn.Parameter(torch.zeros(num_embeddings, r))
        
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        """Initialize parameters."""
        # Initialize embedding with normal distribution
        nn.init.normal_(self.weight, mean=0, std=0.02)
        
        # Initialize LoRA A with Kaiming uniform
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        
        # Initialize LoRA B with zeros
        nn.init.zeros_(self.lora_B)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Base embedding lookup
        base_embed = F.embedding(input_ids, self.weight)
        
        # LoRA contribution
        # Get LoRA B rows for input tokens
        lora_B_selected = F.embedding(input_ids, self.lora_B)
        # Multiply by LoRA A
        lora_output = torch.matmul(lora_B_selected, self.lora_A) * self.scaling
        
        return base_embed + lora_output
    
    @classmethod
    def from_embedding(
        cls,
        embedding: nn.Embedding,
        r: int = 8,
        alpha: float = 16.0,
    ) -> 'LoRAEmbedding':
        """Create LoRAEmbedding from existing Embedding layer."""
        lora_embed = cls(
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
            r=r,
            alpha=alpha,
        )
        
        # Copy original weights
        lora_embed.weight.data.copy_(embedding.weight.data)
        
        return lora_embed


class LoRAModel:
    """
    LoRA Model Wrapper.
    
    Wraps a model and applies LoRA to specified modules.
    
    Args:
        model: Base model to wrap
        config: LoRA configuration
        
    Example:
        >>> lora_model = LoRAModel(model, lora_config)
        >>> lora_model.enable_adapter_layers()
        >>> output = lora_model(input_ids)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: LoRAConfig,
    ):
        self.model = model
        self.config = config
        
        self._find_and_replace()
        
        # Mark model for training
        self._mark_only_adapter_as_trainable()
    
    def _find_and_replace(self) -> None:
        """Find and replace target modules with LoRA versions."""
        target_modules = self.config.target_modules
        
        # Count replaced modules
        replaced = 0
        
        for name, module in self.model.named_modules():
            # Check if this module should be replaced
            if not any(target in name for target in target_modules):
                continue
            
            # Get parent module and child name
            parent = self._get_parent_module(name)
            child_name = name.split('.')[-1]
            
            if isinstance(module, nn.Linear):
                # Replace with LoRALinear
                lora_linear = LoRALinear.from_linear(
                    module,
                    r=self.config.r,
                    alpha=self.config.alpha,
                    dropout=self.config.dropout,
                )
                
                if parent is not None:
                    setattr(parent, child_name, lora_linear)
                    replaced += 1
        
        logger.info(f"Replaced {replaced} modules with LoRA")
    
    def _get_parent_module(self, name: str) -> Optional[nn.Module]:
        """Get parent module of a named module."""
        if '.' not in name:
            return self.model
        
        parent_name = '.'.join(name.split('.')[:-1])
        return dict(self.model.named_modules()).get(parent_name)
    
    def _mark_only_adapter_as_trainable(self) -> None:
        """Freeze base model, only train LoRA parameters."""
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze LoRA parameters
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
        
        # Also unfreeze modules_to_save if specified
        if self.config.modules_to_save:
            for name, param in self.model.named_parameters():
                if any(save in name for save in self.config.modules_to_save):
                    param.requires_grad = True
        
        # Handle bias
        if self.config.bias == 'all':
            for name, param in self.model.named_parameters():
                if 'bias' in name:
                    param.requires_grad = True
        elif self.config.bias == 'lora_only':
            for name, param in self.model.named_parameters():
                if 'lora_' in name and 'bias' in name:
                    param.requires_grad = True
    
    def enable_adapter_layers(self) -> None:
        """Enable adapter layers for training."""
        self.model.train()
        self._mark_only_adapter_as_trainable()
    
    def disable_adapter_layers(self) -> None:
        """Disable adapter layers for inference."""
        self.model.eval()
        
        # Merge weights for faster inference
        for module in self.model.modules():
            if isinstance(module, LoRALinear):
                module.merge()
    
    def get_adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get only adapter state dict."""
        adapter_state = {}
        
        for name, param in self.model.named_parameters():
            if 'lora_' in name or (
                self.config.modules_to_save and
                any(save in name for save in self.config.modules_to_save)
            ):
                adapter_state[name] = param.detach().cpu()
        
        return adapter_state
    
    def load_adapter(self, path: Union[str, Path]) -> None:
        """Load adapter weights from file."""
        path = Path(path)
        
        if path.is_dir():
            path = path / 'adapter_model.bin'
        
        state_dict = torch.load(path, map_location='cpu')
        
        # Load into model
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        
        logger.info(f"Loaded adapter from {path}")
    
    def save_adapter(
        self,
        path: Union[str, Path],
        safe_serialization: bool = True,
    ) -> None:
        """Save adapter weights to file."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        adapter_state = self.get_adapter_state_dict()
        
        if safe_serialization:
            try:
                import safetensors.torch
                safetensors.torch.save_file(
                    adapter_state,
                    path / 'adapter_model.safetensors',
                )
            except ImportError:
                logger.warning("safetensors not installed, using torch.save")
                torch.save(adapter_state, path / 'adapter_model.bin')
        else:
            torch.save(adapter_state, path / 'adapter_model.bin')
        
        # Save config
        with open(path / 'adapter_config.json', 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Saved adapter to {path}")
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped model."""
        try:
            return getattr(self.model, name)
        except AttributeError:
            return super().__getattr__(name)
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass through wrapped model."""
        return self.model(*args, **kwargs)


def get_lora_linear_modules(model: nn.Module) -> List[str]:
    """Get names of all linear modules in a model."""
    linear_modules = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_modules.append(name)
    
    return linear_modules


def get_lora_target_modules(
    model: nn.Module,
    module_types: Optional[List[str]] = None,
) -> List[str]:
    """
    Get target modules for LoRA based on module types.
    
    Args:
        model: Model to analyze
        module_types: Types of modules to target
    
    Returns:
        List of module names
    """
    module_types = module_types or ['q_proj', 'v_proj']
    
    target_modules = []
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(mt in name for mt in module_types):
                target_modules.append(name)
    
    return target_modules


def inject_lora(
    model: nn.Module,
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: Optional[List[str]] = None,
) -> LoRAModel:
    """
    Convenience function to inject LoRA into a model.
    
    Args:
        model: Model to modify
        r: Rank
        alpha: Scaling factor
        dropout: Dropout probability
        target_modules: Modules to target
    
    Returns:
        LoRAModel wrapper
    """
    if target_modules is None:
        # Auto-detect target modules
        target_modules = get_lora_target_modules(model)
    
    config = LoRAConfig(
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )
    
    return LoRAModel(model, config)


class LoRATrainer:
    """
    Trainer for LoRA fine-tuning.
    
    Args:
        model: LoRA-wrapped model
        config: LoRA configuration
        tokenizer: Tokenizer
        train_dataset: Training dataset
        eval_dataset: Optional evaluation dataset
        
    Example:
        >>> trainer = LoRATrainer(lora_model, config, tokenizer, dataset)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: LoRAModel,
        config: LoRAConfig,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        output_dir: str = './lora_output',
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
    ):
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer for LoRA parameters."""
        # Get only trainable parameters (LoRA params)
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        
        # Separate weight decay
        no_decay = ['bias', 'LayerNorm.weight']
        
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for p in trainable_params
                    if not any(nd in n for n, p in self.model.named_parameters() 
                              if p is not None) or not any(nd in name for nd in no_decay)
                ],
                'weight_decay': self.weight_decay,
            },
            {
                'params': [
                    p for p in trainable_params
                    if any(nd in name for nd in no_decay)
                ],
                'weight_decay': 0.0,
            },
        ]
        
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    
    def _create_scheduler(self) -> LRScheduler:
        """Create learning rate scheduler."""
        from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
        
        num_training_steps = (
            len(self.train_dataset) // 
            (self.batch_size * self.gradient_accumulation_steps) *
            self.num_epochs
        )
        
        num_warmup_steps = int(num_training_steps * self.warmup_ratio)
        
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
    
    def train(self) -> Dict[str, Any]:
        """
        Run LoRA training.
        
        Returns:
            Training metrics
        """
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        
        logger.info("Starting LoRA training...")
        
        # Create dataloader
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        
        # Training loop
        self.model.enable_adapter_layers()
        self.model.to(self.device)
        
        progress_bar = tqdm(total=len(train_dataloader) * self.num_epochs)
        
        training_metrics = {'loss': [], 'eval_loss': []}
        
        for epoch in range(self.num_epochs):
            self.model.train()
            
            for batch in train_dataloader:
                # Move to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                )
                
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Optimizer step
                if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    
                    self.global_step += 1
                    progress_bar.update(1)
                    
                    training_metrics['loss'].append(loss.item() * self.gradient_accumulation_steps)
                
                # Evaluation
                if self.global_step % 100 == 0 and self.eval_dataset:
                    eval_loss = self.evaluate()
                    training_metrics['eval_loss'].append(eval_loss)
                    
                    if eval_loss < self.best_eval_loss:
                        self.best_eval_loss = eval_loss
                        self.save_adapter('best')
        
        progress_bar.close()
        
        # Save final adapter
        self.save_adapter('final')
        
        logger.info("LoRA training complete!")
        
        return training_metrics
    
    def _collate_fn(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch of features."""
        return {
            'input_ids': torch.tensor([f['input_ids'] for f in features], dtype=torch.long),
            'attention_mask': torch.tensor([f['attention_mask'] for f in features], dtype=torch.long),
            'labels': torch.tensor([f['labels'] for f in features], dtype=torch.long),
        }
    
    def evaluate(self) -> float:
        """Evaluate the model."""
        from torch.utils.data import DataLoader
        
        self.model.eval()
        
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.batch_size,
            collate_fn=self._collate_fn,
        )
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels'],
                )
                
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
                total_loss += loss.item()
                num_batches += 1
        
        return total_loss / max(num_batches, 1)
    
    def save_adapter(self, name: str) -> None:
        """Save adapter."""
        self.model.save_adapter(self.output_dir / name)
