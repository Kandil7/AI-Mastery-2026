"""
GGUF Quantization - Module 2.7.2

GGUF format quantization for llama.cpp compatibility.
"""

import json
import logging
import struct
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class GGMLType(IntEnum):
    """GGML tensor types."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9


@dataclass
class GGUFConfig:
    """GGUF quantization configuration."""
    quant_type: str = "Q8_0"
    file_version: int = 2
    alignment: int = 32


class GGUFQuantizer:
    """
    GGUF Quantizer for llama.cpp compatibility.
    
    Supports various GGML quantization types.
    """
    
    def __init__(self, config: GGUFConfig):
        self.config = config
        self.ggml_type = GGMLType[config.quant_type]
    
    def quantize_tensor(
        self,
        tensor: torch.Tensor,
    ) -> Dict[str, Any]:
        """Quantize tensor to GGUF format."""
        if self.ggml_type == GGMLType.F32:
            return {'data': tensor.float(), 'type': 'F32'}
        elif self.ggml_type == GGMLType.F16:
            return {'data': tensor.half(), 'type': 'F16'}
        elif self.ggml_type == GGMLType.Q8_0:
            return self._quantize_q8_0(tensor)
        else:
            raise ValueError(f"Unsupported GGML type: {self.ggml_type}")
    
    def _quantize_q8_0(
        self,
        tensor: torch.Tensor,
    ) -> Dict[str, Any]:
        """Quantize to Q8_0 format."""
        # Q8_0: scale + int8 weights
        d = tensor.abs().max() / 127.0
        qs = (tensor / d).round().clamp(-128, 127).to(torch.int8)
        
        return {
            'data': qs,
            'scale': d,
            'type': 'Q8_0',
        }


class GGUFConverter:
    """
    GGUF Model Converter.
    
    Converts PyTorch models to GGUF format.
    """
    
    GGUF_MAGIC = 0x46554747  # "GGUF"
    GGUF_VERSION = 2
    
    def __init__(self, config: GGUFConfig):
        self.config = config
        self.quantizer = GGUFQuantizer(config)
        self.metadata: Dict[str, Any] = {}
        self.tensor_infos: List[Dict] = []
    
    def add_metadata(self, key: str, value: Any) -> None:
        """Add metadata."""
        self.metadata[key] = value
    
    def convert(
        self,
        model: nn.Module,
        output_path: str,
    ) -> None:
        """Convert model to GGUF format."""
        output_path = Path(output_path)
        
        logger.info(f"Converting model to GGUF: {output_path}")
        
        # Collect tensor info
        self.tensor_infos = []
        tensors_data = []
        
        for name, param in model.named_parameters():
            quant_data = self.quantizer.quantize_tensor(param.data)
            
            tensor_info = {
                'name': name,
                'shape': list(param.shape),
                'dtype': quant_data['type'],
                'nbytes': param.numel() * self._get_type_size(quant_data['type']),
            }
            
            self.tensor_infos.append(tensor_info)
            tensors_data.append((name, quant_data))
        
        # Write GGUF file
        with open(output_path, 'wb') as f:
            self._write_header(f)
            self._write_metadata(f)
            self._write_tensor_infos(f)
            self._write_tensors(f, tensors_data)
        
        logger.info(f"GGUF file written to {output_path}")
    
    def _get_type_size(self, dtype: str) -> int:
        """Get bytes per element for dtype."""
        sizes = {
            'F32': 4,
            'F16': 2,
            'Q8_0': 1,
        }
        return sizes.get(dtype, 4)
    
    def _write_header(self, f) -> None:
        """Write GGUF header."""
        f.write(struct.pack('<I', self.GGUF_MAGIC))
        f.write(struct.pack('<I', self.GGUF_VERSION))
        f.write(struct.pack('<Q', len(self.tensor_infos)))
    
    def _write_metadata(self, f) -> None:
        """Write metadata."""
        f.write(struct.pack('<Q', len(self.metadata)))
        for key, value in self.metadata.items():
            self._write_string(f, key)
            self._write_value(f, value)
    
    def _write_tensor_infos(self, f) -> None:
        """Write tensor info."""
        for info in self.tensor_infos:
            self._write_string(f, info['name'])
            f.write(struct.pack('<I', len(info['shape'])))
            for dim in info['shape']:
                f.write(struct.pack('<Q', dim))
            f.write(struct.pack('<I', GGMLType[info['dtype']]))
            f.write(struct.pack('<Q', info['nbytes']))
    
    def _write_tensors(self, f, tensors_data) -> None:
        """Write tensor data."""
        for name, quant_data in tensors_data:
            data = quant_data['data']
            f.write(data.numpy().tobytes())
    
    def _write_string(self, f, s: str) -> None:
        """Write string with length prefix."""
        encoded = s.encode('utf-8')
        f.write(struct.pack('<Q', len(encoded)))
        f.write(encoded)
    
    def _write_value(self, f, value: Any) -> None:
        """Write typed value."""
        if isinstance(value, str):
            f.write(struct.pack('<I', 1))  # String type
            self._write_string(f, value)
        elif isinstance(value, int):
            f.write(struct.pack('<I', 0))  # Int type
            f.write(struct.pack('<q', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', 2))  # Float type
            f.write(struct.pack('<d', value))
