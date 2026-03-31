"""
Global Settings and Environment Configuration
==============================================

Centralized settings management with environment variable support.
"""

import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not required


class Environment(str, Enum):
    """Application environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class Settings:
    """
    Global application settings.

    Attributes:
        environment: Current environment (dev/test/prod)
        debug: Debug mode flag
        log_level: Logging level
        project_root: Root directory of the project
        data_dir: Directory for data files
        model_dir: Directory for model checkpoints
        cache_dir: Directory for cache files
        api_host: API server host
        api_port: API server port
        max_workers: Maximum number of worker threads
        batch_size: Default batch size for training
        device: Device for computation (cpu/cuda/mps)
    """
    # Environment
    environment: Environment = field(default_factory=lambda: Environment(
        os.getenv("ENVIRONMENT", "development")
    ))
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "true").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    model_dir: Path = field(default_factory=lambda: Path(os.getenv("MODEL_DIR", "models")))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", ".cache")))

    # API Settings
    api_host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    api_port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))

    # Performance
    max_workers: int = field(default_factory=lambda: int(os.getenv("MAX_WORKERS", "4")))
    batch_size: int = field(default_factory=lambda: int(os.getenv("BATCH_SIZE", "32")))

    # Device
    device: str = field(default_factory=lambda: os.getenv("DEVICE", "cpu"))

    # Feature flags
    enable_caching: bool = field(default_factory=lambda: os.getenv("ENABLE_CACHING", "true").lower() == "true")
    enable_monitoring: bool = field(default_factory=lambda: os.getenv("ENABLE_MONITORING", "true").lower() == "true")

    def __post_init__(self):
        """Validate and resolve paths after initialization."""
        # Resolve paths relative to project root
        if not self.data_dir.is_absolute():
            self.data_dir = self.project_root / self.data_dir
        if not self.model_dir.is_absolute():
            self.model_dir = self.project_root / self.model_dir
        if not self.cache_dir.is_absolute():
            self.cache_dir = self.project_root / self.cache_dir

        # Create directories if they don't exist
        if self.environment == Environment.DEVELOPMENT:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.environment == Environment.TESTING

    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary."""
        return {
            "environment": self.environment.value,
            "debug": self.debug,
            "log_level": self.log_level,
            "project_root": str(self.project_root),
            "data_dir": str(self.data_dir),
            "model_dir": str(self.model_dir),
            "cache_dir": str(self.cache_dir),
            "api_host": self.api_host,
            "api_port": self.api_port,
            "max_workers": self.max_workers,
            "batch_size": self.batch_size,
            "device": self.device,
            "enable_caching": self.enable_caching,
            "enable_monitoring": self.enable_monitoring,
        }


# Global settings instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """
    Get global settings instance (singleton pattern).

    Returns:
        Settings: Global settings instance
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings() -> None:
    """Reset global settings (useful for testing)."""
    global _settings
    _settings = None
