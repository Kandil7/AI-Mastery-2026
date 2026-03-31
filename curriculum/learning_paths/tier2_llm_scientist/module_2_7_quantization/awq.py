"""
AWQ Quantization - Module 2.7.4

Activation-Aware Weight Quantization.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class AWQConfig:
    """AWQ configuration."""
    bits: int = 4
    group_size: int = 128
    zero_point: bool = True
    sym: bool = False
    version: str = "gemm"


class AWQLinear(nn.Module):
    """
    AWQ Quantized Linear Layer.
    
    Uses activation-aware weight quantization to preserve
    important weights based on activation magnitude.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: AWQConfig,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Quantized weight storage
        self.register_buffer(
            'qweight',
            torch.zeros((out_features, in_features // (32 // config.bits)), dtype=torch.int32),
        )
        self.register_buffer('scales', torch.zeros((out_features, in_features // config.group_size)))
        self.register_buffer('qzeros', torch.zeros((out_features, in_features // config.group_size), dtype=torch.int32))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Activation statistics for AWQ
        self.register_buffer('act_scales', torch.zeros(in_features))
        self._n_samples = 0
        
        self._ready = False
    
    def collect_activation(self, x: torch.Tensor) -> None:
        """Collect activation statistics."""
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        x = x.detach().abs()
        
        # Update running max of activations
        if self._n_samples == 0:
            self.act_scales = x.max(dim=0)[0]
        else:
            self.act_scales = torch.maximum(self.act_scales, x.max(dim=0)[0])
        
        self._n_samples += 1
    
    def quantize(self, weight: torch.Tensor) -> None:
        """Quantize weights using AWQ."""
        if self._n_samples == 0:
            # Fallback to standard quantization
            self.act_scales = torch.ones(self.in_features, device=weight.device)
        
        # Scale weights by activation
        weight = weight / self.act_scales.view(1, -1)
        
        # Group-wise quantization
        for i in range(0, self.in_features, self.config.group_size):
            j = min(i + self.config.group_size, self.in_features)
            
            W_group = weight[:, i:j]
            
            # Compute scale
            if self.config.sym:
                max_val = W_group.abs().max(dim=1, keepdim=True)[0]
                scale = max_val / ((2 ** (self.config.bits - 1)) - 1)
            else:
                min_val = W_group.min(dim=1, keepdim=True)[0]
                max_val = W_group.max(dim=1, keepdim=True)[0]
                scale = (max_val - min_val) / ((2 ** self.config.bits) - 1)
                scale = scale.clamp(min=1e-5)
            
            # Store scale
            self.scales[:, i // self.config.group_size] = scale.squeeze() * self.act_scales[i:j].max()
            
            # Quantize
            Q_group = torch.round(W_group / scale).clamp(
                0 if not self.config.sym else -(2 ** (self.config.bits - 1)),
                (2 ** self.config.bits) - 1 if not self.config.sym else (2 ** (self.config.bits - 1)) - 1
            )
            
            # Pack weights
            self._pack_weights(Q_group, i)
        
        self._ready = True
    
    def _pack_weights(
        self,
        Q: torch.Tensor,
        offset: int,
    ) -> None:
        """Pack quantized weights."""
        # Simple packing for 4-bit
        if self.config.bits == 4:
            Q = Q.to(torch.uint8)
            # Pack two 4-bit values into one byte
            packed = Q[:, ::2] << 4 | Q[:, 1::2]
            self.qweight[:, offset // 8:(offset + Q.shape[1]) // 8] = packed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if not self._ready:
            self.collect_activation(x)
            return F.linear(x, self.weight, self.bias)
        
        # Dequantize
        W = self.dequantize()
        return F.linear(x, W, self.bias)
    
    def dequantize(self) -> torch.Tensor:
        """Dequantize weights."""
        W = self.qweight.float()
        
        # Unpack if 4-bit
        if self.config.bits == 4:
            W = torch.cat([
                (W >> 4) & 0xF,
                W & 0xF,
            ], dim=-1)
        
        # Apply scales
        W = W * self.scales.repeat_interleave(self.config.group_size, dim=-1)
        
        return W


class AWQQuantizer:
    """
    AWQ Model Quantizer.
    
    Applies activation-aware weight quantization.
    """
    
    def __init__(self, config: AWQConfig):
        self.config = config
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: List[torch.Tensor],
    ) -> nn.Module:
        """
        Quantize model using AWQ.
        
        Args:
            model: Model to quantize
            calibration_data: Calibration dataset
        
        Returns:
            Quantized model
        """
        logger.info("Starting AWQ quantization...")
        
        # Replace linear layers
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                awq_linear = AWQLinear(
                    module.in_features,
                    module.out_features,
                    self.config,
                    bias=module.bias is not None,
                )
                awq_linear.weight = module.weight
                
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                
                setattr(parent, child_name, awq_linear)
        
        # Collect activations
        model.eval()
        
        with torch.no_grad():
            for data in calibration_data:
                model(data)
        
        # Quantize each layer
        for name, module in model.named_modules():
            if isinstance(module, AWQLinear):
                module.quantize(module.weight.data)
                logger.info(f"Quantized {name}")
        
        logger.info("AWQ quantization complete")
        
        return model
