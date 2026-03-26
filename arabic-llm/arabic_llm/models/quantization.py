"""
Arabic LLM - Quantization Utilities

Utilities for model quantization (4-bit, 8-bit).
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class QuantizationConfig:
    """Quantization configuration"""
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    llm_int8_threshold: float = 6.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "load_in_4bit": self.load_in_4bit,
            "load_in_8bit": self.load_in_8bit,
            "bnb_4bit_quant_type": self.bnb_4bit_quant_type,
            "bnb_4bit_compute_dtype": self.bnb_4bit_compute_dtype,
            "bnb_4bit_use_double_quant": self.bnb_4bit_use_double_quant,
        }


def create_quantization_config(
    use_4bit: bool = True,
    use_8bit: bool = False,
    quant_type: str = "nf4",
    compute_dtype: str = "float16",
    use_double_quant: bool = True,
) -> QuantizationConfig:
    """
    Create quantization configuration.
    
    Args:
        use_4bit: Use 4-bit quantization
        use_8bit: Use 8-bit quantization
        quant_type: Quantization type (nf4, fp4)
        compute_dtype: Compute dtype (float16, bfloat16)
        use_double_quant: Use nested quantization
        
    Returns:
        QuantizationConfig object
    """
    return QuantizationConfig(
        load_in_4bit=use_4bit,
        load_in_8bit=use_8bit,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=use_double_quant,
    )


def get_quantization_config(config: QuantizationConfig):
    """
    Get BitsAndBytesConfig from QuantizationConfig.
    
    Args:
        config: QuantizationConfig object
        
    Returns:
        BitsAndBytesConfig object
    """
    try:
        from transformers import BitsAndBytesConfig
        import torch
        
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        
        compute_dtype = dtype_map.get(
            config.bnb_4bit_compute_dtype,
            torch.float16
        )
        
        return BitsAndBytesConfig(
            load_in_4bit=config.load_in_4bit,
            load_in_8bit=config.load_in_8bit,
            bnb_4bit_quant_type=config.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=config.bnb_4bit_use_double_quant,
            llm_int8_threshold=config.llm_int8_threshold,
        )
        
    except ImportError as e:
        raise ImportError(
            "BitsAndBytes not available. "
            "Install with: pip install bitsandbytes"
        ) from e


def estimate_vram_usage(
    model_params: int,
    quantization: str = "4bit",
) -> float:
    """
    Estimate VRAM usage for model.
    
    Args:
        model_params: Model parameters in billions
        quantization: Quantization type (4bit, 8bit, 16bit)
        
    Returns:
        Estimated VRAM in GB
    """
    # Base VRAM per billion parameters
    vram_per_billion = {
        "4bit": 0.7,   # ~0.7 GB per billion params
        "8bit": 1.2,   # ~1.2 GB per billion params
        "16bit": 2.0,  # ~2.0 GB per billion params
    }
    
    base_vram = vram_per_billion.get(quantization, 0.7)
    
    # Add overhead for activations, gradients (approx 20%)
    overhead = 1.2
    
    return model_params * base_vram * overhead


def get_recommended_batch_size(
    model_params: int,
    vram_gb: int,
    quantization: str = "4bit",
) -> int:
    """
    Get recommended batch size based on VRAM.
    
    Args:
        model_params: Model parameters in billions
        vram_gb: Available VRAM in GB
        quantization: Quantization type
        
    Returns:
        Recommended batch size
    """
    model_vram = estimate_vram_usage(model_params, quantization)
    available_vram = vram_gb - model_vram
    
    # Approximate memory per sample (varies by sequence length)
    memory_per_sample = 0.1  # GB
    
    batch_size = int(available_vram / memory_per_sample)
    
    # Ensure reasonable batch size
    return max(1, min(batch_size, 32))
