"""
QLoRA (Quantized LoRA) - Module 2.4.3

Production-ready QLoRA implementation:
- 4-bit quantization
- NF4 (Normal Float 4) quantizer
- Quantized LoRA model
- QLoRA trainer

References:
- "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class QuantizationType(str, Enum):
    """Quantization types."""
    FP4 = "fp4"
    NF4 = "nf4"
    INT4 = "int4"
    INT8 = "int8"


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA."""
    # Quantization settings
    quantization_type: QuantizationType = QuantizationType.NF4
    bits: int = 4
    double_quant: bool = True  # Quantize quantization constants
    quant_zero: bool = True  # Quantize zero point
    
    # LoRA settings
    r: int = 64
    alpha: float = 16.0
    dropout: float = 0.1
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ['q_proj', 'v_proj', 'k_proj', 'o_proj']


class NFQuantizer:
    """
    Normal Float 4 (NF4) Quantizer.
    
    NF4 is optimized for normally distributed weights,
    which is common in pretrained models.
    
    Reference:
        "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
    """
    
    # NF4 quantization levels (pre-computed)
    # These values are optimized for standard normal distribution
    NF4_LEVELS = torch.tensor([
        -1.0,
        -0.6962,
        -0.5251,
        -0.3949,
        -0.2844,
        -0.1848,
        -0.0911,
        0.0,
        0.0796,
        0.1609,
        0.2461,
        0.3379,
        0.4407,
        0.5626,
        0.7230,
        1.0,
    ])
    
    def __init__(self, bits: int = 4):
        self.bits = bits
        self.num_levels = 2 ** bits
        
        if bits == 4:
            self.levels = self.NF4_LEVELS
        else:
            # Generate levels for other bit widths
            self.levels = self._generate_levels(bits)
    
    def _generate_levels(self, bits: int) -> torch.Tensor:
        """Generate quantization levels for given bit width."""
        num_levels = 2 ** bits
        
        # Use percent point function (inverse CDF) of normal distribution
        from scipy.stats import norm
        
        # Generate evenly spaced probabilities
        probs = torch.linspace(0.0001, 0.9999, num_levels)
        
        # Convert to quantization levels using inverse CDF
        levels = torch.tensor(norm.ppf(probs.numpy()))
        
        # Normalize to [-1, 1]
        levels = levels / levels.abs().max()
        
        return levels
    
    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize tensor to NF4.
        
        Args:
            tensor: Input tensor
        
        Returns:
            Tuple of (quantized indices, scale, zero_point)
        """
        # Compute absolute maximum for scaling
        abs_max = tensor.abs().max()
        
        if abs_max == 0:
            # Handle zero tensor
            return (
                torch.zeros_like(tensor, dtype=torch.uint8),
                torch.tensor(1.0, device=tensor.device),
                torch.tensor(0.0, device=tensor.device),
            )
        
        # Scale tensor to [-1, 1]
        scaled = tensor / abs_max
        
        # Find nearest quantization level for each value
        levels = self.levels.to(tensor.device)
        
        # Compute distances to all levels
        distances = (scaled.unsqueeze(-1) - levels).abs()
        
        # Find nearest level index
        indices = distances.argmin(dim=-1).to(torch.uint8)
        
        # Zero point (for asymmetric quantization)
        zero_point = torch.tensor(0.0, device=tensor.device)
        
        return indices, abs_max, zero_point
    
    def dequantize(
        self,
        indices: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
    ) -> torch.Tensor:
        """
        Dequantize NF4 indices to float.
        
        Args:
            indices: Quantized indices
            scale: Scale factor
            zero_point: Zero point
        
        Returns:
            Dequantized tensor
        """
        levels = self.levels.to(indices.device)
        
        # Look up quantization levels
        dequantized = levels[indices]
        
        # Apply scale and zero point
        dequantized = dequantized * scale + zero_point
        
        return dequantized
    
    def quantize_block(
        self,
        tensor: torch.Tensor,
        block_size: int = 1024,
    ) -> Dict[str, torch.Tensor]:
        """
        Quantize tensor in blocks for better precision.
        
        Args:
            tensor: Input tensor
            block_size: Size of each quantization block
        
        Returns:
            Dictionary with quantized data
        """
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()
        
        num_blocks = (len(flat_tensor) + block_size - 1) // block_size
        
        # Pad if necessary
        padded_length = num_blocks * block_size
        if len(flat_tensor) < padded_length:
            flat_tensor = F.pad(flat_tensor, (0, padded_length - len(flat_tensor)))
        
        # Reshape into blocks
        blocks = flat_tensor.view(num_blocks, block_size)
        
        # Quantize each block
        all_indices = []
        all_scales = []
        
        for block in blocks:
            indices, scale, _ = self.quantize(block)
            all_indices.append(indices)
            all_scales.append(scale)
        
        return {
            'indices': torch.stack(all_indices),
            'scales': torch.stack(all_scales),
            'original_shape': torch.tensor(original_shape),
            'block_size': torch.tensor(block_size),
        }
    
    def dequantize_block(
        self,
        quant_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Dequantize block-quantized tensor.
        
        Args:
            quant_data: Quantized data dictionary
        
        Returns:
            Dequantized tensor
        """
        indices = quant_data['indices']
        scales = quant_data['scales']
        original_shape = tuple(quant_data['original_shape'].tolist())
        
        # Dequantize each block
        dequantized_blocks = []
        
        for i, block_indices in enumerate(indices):
            dequantized = self.dequantize(
                block_indices,
                scales[i],
                torch.tensor(0.0, device=block_indices.device),
            )
            dequantized_blocks.append(dequantized)
        
        # Concatenate and reshape
        dequantized = torch.cat(dequantized_blocks)
        original_size = int(torch.prod(torch.tensor(original_shape)).item())
        dequantized = dequantized[:original_size]
        
        return dequantized.view(original_shape)


class QuantizedLinear(nn.Module):
    """
    Quantized Linear layer for QLoRA.
    
    Stores weights in 4-bit quantized format and
    dequantizes on-the-fly during forward pass.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        quant_config: Quantization configuration
        bias: Whether to use bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quant_config: Optional[QLoRAConfig] = None,
        bias: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Quantization config
        if quant_config is None:
            quant_config = QLoRAConfig()
        self.quant_config = quant_config
        
        # Initialize quantizer
        self.quantizer = NFQuantizer(bits=quant_config.bits)
        
        # Weight storage (quantized)
        # We store as uint8 for 4-bit quantization
        self.register_buffer(
            'quant_weight',
            torch.zeros((out_features, in_features), dtype=torch.uint8),
        )
        self.register_buffer('weight_scale', torch.ones(1))
        self.register_buffer('weight_zero_point', torch.zeros(1))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # For double quantization
        if quant_config.double_quant:
            self.register_buffer('scale_scale', torch.ones(1))
            self.register_buffer('scale_zero_point', torch.zeros(1))
    
    def quantize_weight(self, weight: torch.Tensor) -> None:
        """Quantize and store weight."""
        if self.quant_config.double_quant:
            # Block-wise quantization
            quant_data = self.quantizer.quantize_block(weight)
            self.quant_weight = quant_data['indices']
            self.weight_scale = quant_data['scales']
        else:
            # Simple quantization
            indices, scale, zero_point = self.quantizer.quantize(weight)
            self.quant_weight = indices
            self.weight_scale = scale
            self.weight_zero_point = zero_point
    
    def get_weight(self) -> torch.Tensor:
        """Get dequantized weight."""
        if self.quant_config.double_quant:
            return self.quantizer.dequantize_block({
                'indices': self.quant_weight,
                'scales': self.weight_scale,
                'original_shape': torch.tensor([self.out_features, self.in_features]),
                'block_size': torch.tensor(1024),
            })
        else:
            return self.quantizer.dequantize(
                self.quant_weight,
                self.weight_scale,
                self.weight_zero_point,
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with on-the-fly dequantization."""
        weight = self.get_weight()
        return F.linear(x, weight, self.bias)
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        quant_config: Optional[QLoRAConfig] = None,
    ) -> 'QuantizedLinear':
        """Create QuantizedLinear from existing Linear layer."""
        quant_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            quant_config=quant_config,
            bias=linear.bias is not None,
        )
        
        # Quantize and store weight
        quant_linear.quantize_weight(linear.weight.data)
        
        # Copy bias
        if linear.bias is not None:
            quant_linear.bias.data.copy_(linear.bias.data)
        
        return quant_linear


class QuantizedLoRALinear(nn.Module):
    """
    Quantized LoRA Linear layer.
    
    Combines 4-bit quantization with LoRA adaptation.
    The base weight is quantized and frozen,
    while LoRA adapters are full precision.
    
    Args:
        in_features: Input feature dimension
        out_features: Output feature dimension
        r: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout probability
        quant_config: Quantization configuration
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 64,
        alpha: float = 16.0,
        dropout: float = 0.1,
        quant_config: Optional[QLoRAConfig] = None,
        bias: bool = False,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r if r > 0 else 1.0
        
        # Quantization config
        if quant_config is None:
            quant_config = QLoRAConfig()
        self.quant_config = quant_config
        
        # Initialize quantizer
        self.quantizer = NFQuantizer(bits=quant_config.bits)
        
        # Quantized base weight (frozen)
        self.register_buffer(
            'quant_weight',
            torch.zeros((out_features, in_features), dtype=torch.uint8),
        )
        self.register_buffer('weight_scale', torch.ones(1))
        
        # LoRA adapters (trainable, full precision)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize LoRA weights
        self.reset_lora_parameters()
    
    def reset_lora_parameters(self) -> None:
        """Initialize LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def quantize_weight(self, weight: torch.Tensor) -> None:
        """Quantize and store base weight."""
        indices, scale, _ = self.quantizer.quantize(weight)
        self.quant_weight = indices
        self.weight_scale = scale
    
    def get_base_weight(self) -> torch.Tensor:
        """Get dequantized base weight."""
        return self.quantizer.dequantize(
            self.quant_weight,
            self.weight_scale,
            torch.tensor(0.0, device=self.quant_weight.device),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Dequantize base weight
        base_weight = self.get_base_weight()
        
        # Base output
        base_output = F.linear(x, base_weight, self.bias)
        
        # LoRA output
        lora_output = F.linear(x, self.lora_A)
        lora_output = F.linear(lora_output, self.lora_B)
        lora_output = lora_output * self.scaling
        lora_output = self.dropout(lora_output)
        
        return base_output + lora_output
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        r: int = 64,
        alpha: float = 16.0,
        dropout: float = 0.1,
        quant_config: Optional[QLoRAConfig] = None,
    ) -> 'QuantizedLoRALinear':
        """Create QuantizedLoRALinear from existing Linear layer."""
        quant_lora_linear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            r=r,
            alpha=alpha,
            dropout=dropout,
            quant_config=quant_config,
            bias=linear.bias is not None,
        )
        
        # Quantize base weight
        quant_lora_linear.quantize_weight(linear.weight.data)
        
        # Copy bias
        if linear.bias is not None:
            quant_lora_linear.bias.data.copy_(linear.bias.data)
        
        return quant_lora_linear


class QuantizedLoRAModel:
    """
    Quantized LoRA Model Wrapper.
    
    Wraps a model with 4-bit quantization and LoRA adapters.
    
    Args:
        model: Base model to quantize
        config: QLoRA configuration
        
    Example:
        >>> qlora_model = QuantizedLoRAModel(model, qlora_config)
        >>> qlora_model.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: QLoRAConfig,
    ):
        self.model = model
        self.config = config
        
        self._quantize_and_inject_lora()
        self._mark_only_adapter_as_trainable()
    
    def _quantize_and_inject_lora(self) -> None:
        """Quantize model and inject LoRA adapters."""
        replaced = 0
        
        for name, module in list(self.model.named_modules()):
            # Check if this is a target module
            if not any(target in name for target in self.config.target_modules):
                continue
            
            if isinstance(module, nn.Linear):
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = dict(self.model.named_modules())[parent_name]
                else:
                    parent = self.model
                
                # Create quantized LoRA linear
                quant_lora = QuantizedLoRALinear.from_linear(
                    module,
                    r=self.config.r,
                    alpha=self.config.alpha,
                    dropout=self.config.dropout,
                    quant_config=self.config,
                )
                
                setattr(parent, child_name, quant_lora)
                replaced += 1
        
        logger.info(f"Quantized and injected LoRA into {replaced} modules")
    
    def _mark_only_adapter_as_trainable(self) -> None:
        """Freeze quantized weights, only train LoRA parameters."""
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Unfreeze LoRA parameters
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
    
    def get_memory_footprint(self) -> float:
        """Get model memory footprint in GB."""
        total_params = sum(p.numel() for p in self.model.parameters())
        total_buffers = sum(b.numel() for b in self.model.buffers())
        
        # Estimate: 4-bit for quantized weights, 32-bit for LoRA
        quantized_params = sum(
            p.numel() for n, p in self.model.named_parameters()
            if 'quant_weight' in n
        )
        lora_params = sum(
            p.numel() for n, p in self.model.named_parameters()
            if 'lora_' in n
        )
        
        # Memory in bytes
        quantized_bytes = quantized_params * 0.5  # 4-bit = 0.5 bytes
        lora_bytes = lora_params * 4  # 32-bit = 4 bytes
        buffer_bytes = total_buffers * 4
        
        total_bytes = quantized_bytes + lora_bytes + buffer_bytes
        
        return total_bytes / (1024 ** 3)  # Convert to GB
    
    def enable_adapter_layers(self) -> None:
        """Enable adapter layers for training."""
        self.model.train()
        self._mark_only_adapter_as_trainable()
    
    def get_adapter_state_dict(self) -> Dict[str, torch.Tensor]:
        """Get only adapter state dict."""
        adapter_state = {}
        
        for name, param in self.model.named_parameters():
            if 'lora_' in name:
                adapter_state[name] = param.detach().cpu()
        
        return adapter_state
    
    def save_adapter(
        self,
        path: Union[str, Path],
    ) -> None:
        """Save adapter weights."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        adapter_state = self.get_adapter_state_dict()
        
        torch.save(adapter_state, path / 'adapter_model.bin')
        
        # Save config
        with open(path / 'adapter_config.json', 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Saved QLoRA adapter to {path}")
    
    def load_adapter(
        self,
        path: Union[str, Path],
    ) -> None:
        """Load adapter weights."""
        path = Path(path)
        
        if path.is_dir():
            path = path / 'adapter_model.bin'
        
        state_dict = torch.load(path, map_location='cpu')
        
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        
        logger.info(f"Loaded QLoRA adapter from {path}")
    
    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to wrapped model."""
        try:
            return getattr(self.model, name)
        except AttributeError:
            return super().__getattr__(name)
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass."""
        return self.model(*args, **kwargs)


class QLoRATrainer:
    """
    Trainer for QLoRA fine-tuning.
    
    Args:
        model: QLoRA-wrapped model
        config: QLoRA configuration
        tokenizer: Tokenizer
        train_dataset: Training dataset
        
    Example:
        >>> trainer = QLoRATrainer(qlora_model, config, tokenizer, dataset)
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: QuantizedLoRAModel,
        config: QLoRAConfig,
        tokenizer: Any,
        train_dataset: Any,
        eval_dataset: Optional[Any] = None,
        output_dir: str = './qlora_output',
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 1,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        bf16: bool = True,
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
        self.bf16 = bf16
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Mixed precision
        self.use_amp = bf16
        self.scaler = torch.cuda.amp.GradScaler() if bf16 and torch.cuda.is_available() else None
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.global_step = 0
        self.best_eval_loss = float('inf')
        
        # Log memory footprint
        logger.info(f"QLoRA memory footprint: {model.get_memory_footprint():.2f} GB")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for QLoRA parameters."""
        # Get only trainable parameters
        trainable_params = [
            p for p in self.model.parameters() if p.requires_grad
        ]
        
        return torch.optim.AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
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
            eta_min=0,
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
        """Run QLoRA training."""
        from torch.utils.data import DataLoader
        from tqdm import tqdm
        
        logger.info("Starting QLoRA training...")
        
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        
        self.model.enable_adapter_layers()
        self.model.to(self.device)
        
        progress_bar = tqdm(total=len(train_dataloader) * self.num_epochs)
        
        training_metrics = {'loss': [], 'eval_loss': []}
        accumulation_step = 0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            
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
                
                loss = loss / self.gradient_accumulation_steps
                
                # Backward pass
                if self.use_amp and self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accumulation_step += 1
                
                # Optimizer step
                if accumulation_step % self.gradient_accumulation_steps == 0:
                    if self.use_amp and self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.max_grad_norm,
                    )
                    
                    if self.use_amp and self.scaler:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
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
        
        self.save_adapter('final')
        
        logger.info("QLoRA training complete!")
        
        return training_metrics
    
    def _collate_fn(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate batch."""
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
