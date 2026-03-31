"""
Configuration Module
====================

Centralized configuration management for AI-Mastery-2026.

Components:
- **settings**: Global settings and environment configuration
- **model_config**: Model-specific configurations
- **data_config**: Data pipeline configurations
- **api_config**: API and server configurations

Usage:
    >>> from src.config import get_settings, ModelConfig
    >>> settings = get_settings()
    >>> config = ModelConfig(hidden_dim=768, num_layers=12)
"""

from .settings import Settings, get_settings, Environment
from .model_config import (
    ModelConfig,
    TransformerConfig,
    TrainingConfig,
    LLMConfig,
    RAGConfig,
)
from .data_config import DataConfig, PreprocessingConfig

__all__ = [
    # Settings
    "Settings",
    "get_settings",
    "Environment",
    # Model Configs
    "ModelConfig",
    "TransformerConfig",
    "TrainingConfig",
    "LLMConfig",
    "RAGConfig",
    # Data Configs
    "DataConfig",
    "PreprocessingConfig",
]
