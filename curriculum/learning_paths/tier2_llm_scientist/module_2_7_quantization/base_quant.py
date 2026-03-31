"""
Base Quantization - Module 2.7.1

Fundamental quantization implementations:
- FP32, FP16, BF16
- INT8 with zero-point
- Absmax quantization
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class QuantizationType(str, Enum):
    """Quantization types."""
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""
    quant_type: QuantizationType = QuantizationType.INT8
    bits: int = 8
    group_size: int = 128
    symmetrical: bool = True
    per_channel: bool = False
    
    # Zero-point settings
    zero_point: bool = True
    
    # Calibration settings
    calibration_samples: int = 512
    
    # Output settings
    compute_dtype: torch.dtype = torch.float16


class BaseQuantizer(ABC):
    """Abstract base class for quantizers."""
    
    @abstractmethod
    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize a tensor."""
        pass
    
    @abstractmethod
    def dequantize(
        self,
        quant_tensor: torch.Tensor,
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """Dequantize a tensor."""
        pass


class FP32Quantizer(BaseQuantizer):
    """
    FP32 (Full Precision) Quantizer.
    
    No actual quantization - keeps weights in FP32.
    Used as baseline for comparison.
    """
    
    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Keep tensor in FP32."""
        return tensor.float(), {'dtype': 'fp32'}
    
    def dequantize(
        self,
        quant_tensor: torch.Tensor,
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """Return as-is."""
        return quant_tensor


class FP16Quantizer(BaseQuantizer):
    """
    FP16 (Half Precision) Quantizer.
    
    Converts weights to FP16 for reduced memory usage.
    """
    
    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Convert to FP16."""
        return tensor.half(), {'dtype': 'fp16'}
    
    def dequantize(
        self,
        quant_tensor: torch.Tensor,
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """Convert back to FP32."""
        return quant_tensor.float()


class BF16Quantizer(BaseQuantizer):
    """
    BF16 (BFloat16) Quantizer.
    
    Converts weights to BFloat16 for reduced memory with
    better dynamic range than FP16.
    """
    
    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Convert to BF16."""
        return tensor.bfloat16(), {'dtype': 'bf16'}
    
    def dequantize(
        self,
        quant_tensor: torch.Tensor,
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """Convert back to FP32."""
        return quant_tensor.float()


class INT8Quantizer(BaseQuantizer):
    """
    INT8 Quantizer with symmetric quantization.
    
    Quantizes weights to 8-bit integers with scale factor.
    
    Formula: q = round(w / scale)
    Dequantize: w = q * scale
    """
    
    def __init__(self, symmetrical: bool = True):
        self.symmetrical = symmetrical
    
    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize to INT8."""
        if self.symmetrical:
            # Symmetric quantization
            max_val = tensor.abs().max()
            scale = max_val / 127.0
        else:
            # Asymmetric quantization
            min_val = tensor.min()
            max_val = tensor.max()
            scale = (max_val - min_val) / 255.0
        
        # Quantize
        quant_tensor = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
        
        params = {
            'scale': scale,
            'symmetrical': self.symmetrical,
            'dtype': 'int8',
        }
        
        if not self.symmetrical:
            params['zero_point'] = -min_val / scale
        
        return quant_tensor, params
    
    def dequantize(
        self,
        quant_tensor: torch.Tensor,
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """Dequantize from INT8."""
        scale = params['scale']
        
        dequant = quant_tensor.float() * scale
        
        if not params['symmetrical']:
            zero_point = params.get('zero_point', 0)
            dequant = dequant + zero_point
        
        return dequant


class ZeroPointQuantizer(BaseQuantizer):
    """
    Zero-Point Quantizer.
    
    Uses asymmetric quantization with explicit zero-point.
    Common in INT8 quantization for better handling of
    non-symmetric weight distributions.
    
    Formula: q = round(w / scale) + zero_point
    """
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.qmin = -(2 ** (bits - 1))
        self.qmax = 2 ** (bits - 1) - 1
    
    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize with zero-point."""
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Compute scale and zero-point
        scale = (max_val - min_val) / (self.qmax - self.qmin)
        zero_point = self.qmin - min_val / scale
        zero_point = round(zero_point).clamp(self.qmin, self.qmax)
        
        # Quantize
        quant_tensor = (tensor / scale + zero_point).round()
        quant_tensor = quant_tensor.clamp(self.qmin, self.qmax)
        
        if self.bits == 8:
            quant_tensor = quant_tensor.to(torch.int8)
        elif self.bits == 4:
            quant_tensor = quant_tensor.to(torch.int8)  # Pack later
        
        params = {
            'scale': scale,
            'zero_point': int(zero_point),
            'bits': self.bits,
            'min_val': min_val,
            'max_val': max_val,
        }
        
        return quant_tensor, params
    
    def dequantize(
        self,
        quant_tensor: torch.Tensor,
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """Dequantize with zero-point."""
        scale = params['scale']
        zero_point = params['zero_point']
        
        return (quant_tensor.float() - zero_point) * scale


class AbsMaxQuantizer(BaseQuantizer):
    """
    Absolute Maximum Quantizer.
    
    Uses the absolute maximum value for scaling.
    Simple and effective for many use cases.
    """
    
    def __init__(self, bits: int = 8):
        self.bits = bits
        self.qmax = 2 ** (bits - 1) - 1
    
    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize using absmax."""
        abs_max = tensor.abs().max()
        scale = abs_max / self.qmax
        
        # Quantize
        quant_tensor = (tensor / scale).round()
        quant_tensor = quant_tensor.clamp(-self.qmax - 1, self.qmax)
        
        if self.bits == 8:
            quant_tensor = quant_tensor.to(torch.int8)
        
        params = {
            'scale': scale,
            'abs_max': abs_max,
            'bits': self.bits,
        }
        
        return quant_tensor, params
    
    def dequantize(
        self,
        quant_tensor: torch.Tensor,
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """Dequantize from absmax."""
        scale = params['scale']
        return quant_tensor.float() * scale


class PerChannelQuantizer(BaseQuantizer):
    """
    Per-Channel Quantizer.
    
    Applies quantization separately to each output channel.
    Better preserves accuracy for weight matrices.
    """
    
    def __init__(
        self,
        base_quantizer: BaseQuantizer,
        axis: int = 0,
    ):
        self.base_quantizer = base_quantizer
        self.axis = axis
    
    def quantize(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Quantize per-channel."""
        # Move quantization axis to last dimension
        if self.axis != tensor.dim() - 1:
            tensor = tensor.transpose(self.axis, -1)
        
        original_shape = tensor.shape
        
        # Reshape for per-channel processing
        tensor = tensor.view(-1, tensor.shape[-1])
        
        quant_tensors = []
        all_params = {'scales': [], 'zero_points': []}
        
        for i in range(tensor.shape[1]):
            channel = tensor[:, i]
            quant_channel, params = self.base_quantizer.quantize(channel)
            quant_tensors.append(quant_channel)
            
            if 'scale' in params:
                all_params['scales'].append(params['scale'])
            if 'zero_point' in params:
                all_params['zero_points'].append(params['zero_point'])
        
        # Stack channels
        quant_tensor = torch.stack(quant_tensors, dim=1)
        quant_tensor = quant_tensor.view(original_shape)
        
        # Transpose back if needed
        if self.axis != tensor.dim() - 1:
            quant_tensor = quant_tensor.transpose(self.axis, -1)
        
        all_params['axis'] = self.axis
        all_params['original_shape'] = original_shape
        
        return quant_tensor, all_params
    
    def dequantize(
        self,
        quant_tensor: torch.Tensor,
        params: Dict[str, Any],
    ) -> torch.Tensor:
        """Dequantize per-channel."""
        axis = params['axis']
        original_shape = params['original_shape']
        scales = params['scales']
        
        if axis != quant_tensor.dim() - 1:
            quant_tensor = quant_tensor.transpose(axis, -1)
        
        quant_tensor = quant_tensor.view(-1, quant_tensor.shape[-1])
        
        dequant_tensors = []
        
        for i, scale in enumerate(scales):
            channel = quant_tensor[:, i]
            dequant = channel.float() * scale
            dequant_tensors.append(dequant)
        
        dequant_tensor = torch.stack(dequant_tensors, dim=1)
        dequant_tensor = dequant_tensor.view(original_shape)
        
        if axis != quant_tensor.dim() - 1:
            dequant_tensor = dequant_tensor.transpose(axis, -1)
        
        return dequant_tensor


class QuantizedLinear(nn.Module):
    """
    Quantized Linear Layer.
    
    Wraps a linear layer with quantization support.
    
    Args:
        in_features: Input features
        out_features: Output features
        quantizer: Quantizer to use
        bias: Whether to use bias
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        quantizer: BaseQuantizer,
        bias: bool = True,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.quantizer = quantizer
        
        # Weight
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Quantized weight storage
        self.register_buffer('quant_weight', None)
        self.register_buffer('weight_scale', None)
        self.register_buffer('weight_zero_point', None)
        
        self._quantized = False
    
    def quantize_weight(self) -> None:
        """Quantize the weight."""
        quant_weight, params = self.quantizer.quantize(self.weight.data)
        
        self.quant_weight = nn.Parameter(quant_weight, requires_grad=False)
        self.weight_scale = params.get('scale')
        
        if 'zero_point' in params:
            self.weight_zero_point = params['zero_point']
        
        self._quantized = True
    
    def get_weight(self) -> torch.Tensor:
        """Get dequantized weight."""
        if not self._quantized:
            return self.weight
        
        params = {'scale': self.weight_scale}
        if self.weight_zero_point is not None:
            params['zero_point'] = self.weight_zero_point
        
        return self.quantizer.dequantize(self.quant_weight, params)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        weight = self.get_weight()
        return F.linear(x, weight, self.bias)
    
    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        quantizer: BaseQuantizer,
    ) -> 'QuantizedLinear':
        """Create from existing linear layer."""
        quant_linear = cls(
            linear.in_features,
            linear.out_features,
            quantizer,
            bias=linear.bias is not None,
        )
        
        quant_linear.weight.data.copy_(linear.weight.data)
        if linear.bias is not None:
            quant_linear.bias.data.copy_(linear.bias.data)
        
        return quant_linear


def quantize_model(
    model: nn.Module,
    config: QuantizationConfig,
) -> nn.Module:
    """
    Quantize a model.
    
    Args:
        model: Model to quantize
        config: Quantization configuration
    
    Returns:
        Quantized model
    """
    # Create quantizer based on config
    if config.quant_type == QuantizationType.FP16:
        quantizer = FP16Quantizer()
    elif config.quant_type == QuantizationType.BF16:
        quantizer = BF16Quantizer()
    elif config.quant_type == QuantizationType.INT8:
        if config.per_channel:
            base_quantizer = INT8Quantizer(symmetrical=config.symmetrical)
            quantizer = PerChannelQuantizer(base_quantizer)
        else:
            quantizer = INT8Quantizer(symmetrical=config.symmetrical)
    else:
        quantizer = FP32Quantizer()
    
    # Replace linear layers
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model
            
            quant_linear = QuantizedLinear.from_linear(module, quantizer)
            setattr(parent, child_name, quant_linear)
    
    # Convert model dtype
    if config.quant_type == QuantizationType.FP16:
        model = model.half()
    elif config.quant_type == QuantizationType.BF16:
        model = model.bfloat16()
    
    logger.info(f"Model quantized to {config.quant_type.value}")
    
    return model


def get_quantization_memory_footprint(
    model: nn.Module,
    config: QuantizationConfig,
) -> Dict[str, float]:
    """
    Estimate memory footprint after quantization.
    
    Args:
        model: Model to analyze
        config: Quantization configuration
    
    Returns:
        Memory footprint in MB
    """
    total_params = sum(p.numel() for p in model.parameters())
    
    # Bytes per parameter based on quantization type
    bytes_per_param = {
        QuantizationType.FP32: 4,
        QuantizationType.FP16: 2,
        QuantizationType.BF16: 2,
        QuantizationType.INT8: 1,
        QuantizationType.INT4: 0.5,
        QuantizationType.NF4: 0.5,
    }.get(config.quant_type, 4)
    
    total_bytes = total_params * bytes_per_param
    
    return {
        'total_params': total_params,
        'bytes_per_param': bytes_per_param,
        'total_mb': total_bytes / (1024 ** 2),
        'total_gb': total_bytes / (1024 ** 3),
    }
