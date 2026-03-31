"""
GPTQ Quantization - Module 2.7.3

GPTQ: Accurate Post-Training Quantization for LLMs.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class GPTQConfig:
    """GPTQ configuration."""
    bits: int = 4
    group_size: int = 128
    damp_percent: float = 0.01
    desc_act: bool = False
    static_groups: bool = False
    sym: bool = True
    true_sequential: bool = True


class GPTQLinear(nn.Module):
    """
    GPTQ Quantized Linear Layer.
    
    Implements GPTQ quantization for linear layers.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: GPTQConfig,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.config = config
        
        # Weight storage (quantized)
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
        
        # For calibration
        self.register_buffer('H', torch.zeros((in_features, in_features)))
        self.register_buffer('n_samples', torch.zeros(1))
        self._ready = False
    
    def collect_input(self, x: torch.Tensor) -> None:
        """Collect input for Hessian computation."""
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        
        x = x.detach()
        H = self.H * (self.n_samples / (self.n_samples + 1))
        
        if len(x.shape) == 3:
            x = x.reshape(-1, x.shape[-1])
        
        H += x.T @ x / x.shape[0]
        self.H = H
        self.n_samples += 1
    
    def quantize(self) -> None:
        """Perform GPTQ quantization."""
        if self.n_samples == 0:
            logger.warning("No calibration data collected")
            return
        
        W = self.weight.data.clone()
        H = self.H
        
        # Add damping
        damp = self.config.damp_percent * torch.mean(torch.diag(H))
        diag = torch.arange(self.in_features, device=H.device)
        H[diag, diag] += damp
        
        # Cholesky decomposition
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        
        # Quantize
        self._quantize_gptq(W, H)
        self._ready = True
    
    def _quantize_gptq(
        self,
        W: torch.Tensor,
        H: torch.Tensor,
    ) -> None:
        """GPTQ quantization algorithm."""
        Q = torch.zeros_like(W)
        Error = torch.zeros_like(W)
        
        Hinv = H @ H.T
        
        for i1 in range(0, self.in_features, self.config.group_size):
            i2 = min(i1 + self.config.group_size, self.in_features)
            
            # Get group
            W_group = W[:, i1:i2]
            
            # Quantize group
            if self.config.sym:
                max_val = W_group.abs().max(dim=1, keepdim=True)[0]
                scale = max_val / ((2 ** (self.config.bits - 1)) - 1)
            else:
                min_val = W_group.min(dim=1, keepdim=True)[0]
                max_val = W_group.max(dim=1, keepdim=True)[0]
                scale = (max_val - min_val) / ((2 ** self.config.bits) - 1)
            
            # Store scales
            self.scales[:, i1 // self.config.group_size] = scale.squeeze()
            
            # Quantize
            Q_group = torch.round(W_group / scale)
            Q[:, i1:i2] = Q_group * scale
        
        self.qweight = Q.to(torch.int32)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        if not self._ready:
            # During calibration
            self.collect_input(x)
            return F.linear(x, self.weight, self.bias)
        
        # Dequantize for forward pass
        W = self.dequantize()
        return F.linear(x, W, self.bias)
    
    def dequantize(self) -> torch.Tensor:
        """Dequantize weights."""
        # Simple dequantization
        return self.qweight.float() * self.scales


class GPTQQuantizer:
    """
    GPTQ Model Quantizer.
    
    Applies GPTQ quantization layer by layer.
    """
    
    def __init__(self, config: GPTQConfig):
        self.config = config
    
    def quantize_model(
        self,
        model: nn.Module,
        calibration_data: torch.Tensor,
    ) -> nn.Module:
        """
        Quantize model using GPTQ.
        
        Args:
            model: Model to quantize
            calibration_data: Calibration dataset
        
        Returns:
            Quantized model
        """
        logger.info("Starting GPTQ quantization...")
        
        # Replace linear layers
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear):
                gptq_linear = GPTQLinear(
                    module.in_features,
                    module.out_features,
                    self.config,
                    bias=module.bias is not None,
                )
                gptq_linear.weight = module.weight
                
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = dict(model.named_modules())[parent_name]
                else:
                    parent = model
                
                setattr(parent, child_name, gptq_linear)
        
        # Calibrate each layer
        model.eval()
        
        with torch.no_grad():
            for _ in range(len(calibration_data)):
                model(calibration_data[_])
        
        # Quantize each layer
        for name, module in model.named_modules():
            if isinstance(module, GPTQLinear):
                module.quantize()
                logger.info(f"Quantized {name}")
        
        logger.info("GPTQ quantization complete")
        
        return model
