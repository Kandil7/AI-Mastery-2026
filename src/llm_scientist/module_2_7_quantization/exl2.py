"""
EXL2 Quantization - Module 2.7.5

EXL2: Extreme Low-bit Quantization (2-8 bit).
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class EXL2Config:
    """EXL2 configuration."""
    bits: int = 4
    group_size: int = 128
    exllama_version: int = 2
    weight_bits: int = 4
    scale_bits: int = 16


class EXL2Linear(nn.Module):
    """
    EXL2 Quantized Linear Layer.
    
    Supports 2-8 bit quantization with EXL2 format.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: EXL2Config,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # EXL2 uses packed int32 weights
        pack_factor = 32 // config.bits
        
        self.register_buffer(
            'qweight',
            torch.zeros((out_features, in_features // pack_factor), dtype=torch.int32),
        )
        self.register_buffer(
            'qscale',
            torch.zeros((out_features, in_features // config.group_size), dtype=torch.float16),
        )
        self.register_buffer(
            'qzero',
            torch.zeros((out_features, in_features // config.group_size), dtype=torch.int32),
        )
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._ready = False
    
    def quantize(self, weight: torch.Tensor) -> None:
        """Quantize weights to EXL2 format."""
        bits = self.config.bits
        group_size = self.config.group_size
        
        # Calculate quantization parameters
        qmax = (2 ** bits) - 1
        
        # Group-wise quantization
        for i in range(0, self.in_features, group_size):
            j = min(i + group_size, self.in_features)
            
            W_group = weight[:, i:j]
            
            # Find min/max for this group
            min_val = W_group.min(dim=1, keepdim=True)[0]
            max_val = W_group.max(dim=1, keepdim=True)[0]
            
            # Compute scale
            scale = (max_val - min_val) / qmax
            scale = scale.clamp(min=1e-5)
            
            # Store scale
            self.qscale[:, i // group_size] = scale.squeeze().half()
            
            # Quantize
            Q_group = torch.round((W_group - min_val) / scale)
            Q_group = Q_group.clamp(0, qmax)
            
            # Pack weights
            self._pack_weights(Q_group, i, bits)
            
            # Store zero point
            zero_point = torch.round(-min_val / scale).to(torch.int32)
            self.qzero[:, i // group_size] = zero_point.squeeze()
        
        self._ready = True
    
    def _pack_weights(
        self,
        Q: torch.Tensor,
        offset: int,
        bits: int,
    ) -> None:
        """Pack weights into int32."""
        pack_factor = 32 // bits
        
        # Reshape for packing
        Q = Q.to(torch.int32)
        
        if bits == 4:
            # Pack 8 values into one int32
            packed = torch.zeros(
                (Q.shape[0], Q.shape[1] // pack_factor),
                dtype=torch.int32,
                device=Q.device,
            )
            
            for i in range(pack_factor):
                packed |= (Q[:, i::pack_factor] << (i * bits))
            
            self.qweight[:, offset // pack_factor:(offset + Q.shape[1]) // pack_factor] = packed
        
        elif bits == 8:
            # Pack 4 values into one int32
            packed = torch.zeros(
                (Q.shape[0], Q.shape[1] // pack_factor),
                dtype=torch.int32,
                device=Q.device,
            )
            
            for i in range(pack_factor):
                packed |= (Q[:, i::pack_factor] << (i * bits))
            
            self.qweight[:, offset // pack_factor:(offset + Q.shape[1]) // pack_factor] = packed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if not self._ready:
            return F.linear(x, self.weight, self.bias)
        
        # Dequantize for forward
        W = self.dequantize()
        return F.linear(x, W, self.bias)
    
    def dequantize(self) -> torch.Tensor:
        """Dequantize weights."""
        bits = self.config.bits
        group_size = self.config.group_size
        qmax = (2 ** bits) - 1
        
        # Unpack weights
        W = torch.zeros(
            (self.out_features, self.in_features),
            dtype=torch.float16,
            device=self.qweight.device,
        )
        
        pack_factor = 32 // bits
        mask = qmax
        
        for i in range(pack_factor):
            unpacked = (self.qweight >> (i * bits)) & mask
            col_start = i * pack_factor
            col_end = min(col_start + pack_factor, self.in_features)
            
            if col_end > col_start:
                W[:, col_start:col_end] = unpacked[:, :col_end - col_start].float()
        
        # Apply scale and zero point
        for i in range(0, self.in_features, group_size):
            j = min(i + group_size, self.in_features)
            scale = self.qscale[:, i // group_size:i // group_size + 1]
            zero = self.qzero[:, i // group_size:i // group_size + 1]
            
            W[:, i:j] = W[:, i:j] * scale + zero
        
        return W.half()


class EXL2Quantizer:
    """
    EXL2 Model Quantizer.
    
    Applies EXL2 quantization for extreme compression.
    """
    
    def __init__(self, config: EXL2Config):
        self.config = config
    
    def quantize_model(
        self,
        model: nn.Module,
    ) -> nn.Module:
        """
        Quantize model using EXL2.
        
        Args:
            model: Model to quantize
        
        Returns:
            Quantized model
        """
        logger.info(f"Starting EXL2 quantization ({self.config.bits}-bit)...")
        
        # Replace linear layers
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                exl2_linear = EXL2Linear(
                    module.in_features,
                    module.out_features,
                    self.config,
                    bias=module.bias is not None,
                )
                exl2_linear.weight = module.weight
                
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                
                setattr(parent, child_name, exl2_linear)
        
        # Quantize each layer
        for name, module in model.named_modules():
            if isinstance(module, EXL2Linear):
                module.quantize(module.weight.data)
                logger.info(f"Quantized {name}")
        
        logger.info("EXL2 quantization complete")
        
        return model
    
    def get_compression_ratio(
        self,
        model: nn.Module,
    ) -> float:
        """Calculate compression ratio."""
        original_size = sum(
            p.numel() * 4  # FP32
            for p in model.parameters()
        )
        
        quantized_size = sum(
            m.qweight.numel() * 4 + m.qscale.numel() * 2
            for m in model.modules()
            if isinstance(m, EXL2Linear)
        )
        
        return original_size / quantized_size if quantized_size > 0 else 1.0
