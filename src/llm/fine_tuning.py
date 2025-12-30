"""
Fine-Tuning Techniques
======================
Parameter-efficient fine-tuning (PEFT) methods: LoRA, QLoRA, Adapters.

Author: AI-Mastery-2026
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class LoRAConfig:
    """LoRA configuration."""
    r: int = 8
    alpha: float = 16.0
    dropout: float = 0.1
    target_modules: List[str] = None


class LoRALayer:
    """
    LoRA adaptation layer.
    
    Original: y = Wx
    With LoRA: y = Wx + (B @ A)x × (α/r)
    """
    
    def __init__(self, in_features: int, out_features: int,
                 r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        self.in_features = in_features
        self.out_features = out_features
        self.r = r
        self.scaling = alpha / r
        self.dropout = dropout
        
        self.A = np.random.randn(r, in_features) * 0.01
        self.B = np.zeros((out_features, r))
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        if training and self.dropout > 0:
            mask = np.random.binomial(1, 1 - self.dropout, x.shape)
            x = x * mask / (1 - self.dropout)
        hidden = x @ self.A.T
        return (hidden @ self.B.T) * self.scaling
    
    def merge_weights(self, base_weights: np.ndarray) -> np.ndarray:
        return base_weights + self.B @ self.A * self.scaling
    
    def get_trainable_params(self) -> int:
        return self.A.size + self.B.size


class LinearWithLoRA:
    """Linear layer with LoRA."""
    
    def __init__(self, in_features: int, out_features: int, r: int = 8):
        scale = np.sqrt(2.0 / in_features)
        self.weight = np.random.randn(out_features, in_features) * scale
        self.bias = np.zeros(out_features)
        self.lora = LoRALayer(in_features, out_features, r)
        self.base_frozen = False
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        output = x @ self.weight.T + self.bias
        return output + self.lora.forward(x, training)
    
    def merge_lora(self):
        self.weight = self.lora.merge_weights(self.weight)
        self.lora = None


class AdapterLayer:
    """Adapter module for PEFT."""
    
    def __init__(self, d_model: int, bottleneck: int = 64):
        scale = np.sqrt(2.0 / d_model)
        self.down = np.random.randn(d_model, bottleneck) * scale
        self.up = np.random.randn(bottleneck, d_model) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0, x @ self.down)
        return x + hidden @ self.up


def quantize_nf4(weights: np.ndarray) -> Tuple[np.ndarray, float]:
    """Simulate 4-bit quantization."""
    absmax = np.max(np.abs(weights))
    scale = absmax / 7.0
    quantized = np.clip(np.round(weights / scale), -8, 7).astype(np.int8)
    return quantized, scale


def dequantize_nf4(quantized: np.ndarray, scale: float) -> np.ndarray:
    """Dequantize 4-bit weights."""
    return quantized.astype(np.float32) * scale


__all__ = ['LoRAConfig', 'LoRALayer', 'LinearWithLoRA', 'AdapterLayer', 
           'quantize_nf4', 'dequantize_nf4']
