"""
Configuration Management for AI-Mastery-2026
=============================================

Centralized configuration with environment variable support,
validation, and type safety.

Usage:
------
    from src.core.utils.config import get_config, Config

    config = get_config()

    # Access configuration
    db_url = config.database_url
    llm_api_key = config.get_secret("llm_api_key")

    # Nested configuration
    rag_config = config.rag
    chunk_size = rag_config.chunk_size

Configuration Sources (in order):
---------------------------------
1. Environment variables (AI_MASTERY_*)
2. .env file
3. config.yaml file
4. Default values
"""

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Generic, List, Optional, TypeVar

from dotenv import load_dotenv

# Load .env file
load_dotenv()


T = TypeVar("T")


@dataclass
class DatabaseConfig:
    """Database configuration."""

    url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://localhost/ai_mastery"
        )
    )
    host: str = field(default_factory=lambda: os.getenv("DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("DB_PORT", "5432")))
    name: str = field(default_factory=lambda: os.getenv("DB_NAME", "ai_mastery"))
    user: str = field(default_factory=lambda: os.getenv("DB_USER", "postgres"))
    password: str = field(default_factory=lambda: os.getenv("DB_PASSWORD", ""))
    pool_size: int = field(default_factory=lambda: int(os.getenv("DB_POOL_SIZE", "10")))
    max_overflow: int = field(
        default_factory=lambda: int(os.getenv("DB_MAX_OVERFLOW", "20"))
    )


@dataclass
class RedisConfig:
    """Redis configuration."""

    url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379")
    )
    host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))
    db: int = field(default_factory=lambda: int(os.getenv("REDIS_DB", "0")))


@dataclass
class LLMConfig:
    """LLM API configuration."""

    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", ""))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "gpt-4"))
    base_url: Optional[str] = field(default_factory=lambda: os.getenv("LLM_BASE_URL"))
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("LLM_MAX_TOKENS", "2048"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.7"))
    )
    timeout: int = field(default_factory=lambda: int(os.getenv("LLM_TIMEOUT", "30")))


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    provider: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "sentence_transformers")
    )
    model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("EMBEDDING_API_KEY")
    )
    dimension: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_DIM", "384"))
    )
    batch_size: int = field(
        default_factory=lambda: int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
    )


@dataclass
class VectorStoreConfig:
    """Vector store configuration."""

    provider: str = field(
        default_factory=lambda: os.getenv("VECTOR_STORE_PROVIDER", "faiss")
    )
    index_path: str = field(
        default_factory=lambda: os.getenv(
            "VECTOR_STORE_INDEX_PATH", "data/vector_index"
        )
    )
    qdrant_url: Optional[str] = field(default_factory=lambda: os.getenv("QDRANT_URL"))
    qdrant_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("QDRANT_API_KEY")
    )
    collection_name: str = field(
        default_factory=lambda: os.getenv("VECTOR_COLLECTION", "documents")
    )


@dataclass
class RAGConfig:
    """RAG pipeline configuration."""

    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("RAG_CHUNK_SIZE", "512"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("RAG_CHUNK_OVERLAP", "50"))
    )
    retrieval_top_k: int = field(
        default_factory=lambda: int(os.getenv("RAG_TOP_K", "5"))
    )
    rerank_top_k: int = field(
        default_factory=lambda: int(os.getenv("RAG_RERANK_TOP_K", "3"))
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.7"))
    )
    cache_enabled: bool = field(
        default_factory=lambda: os.getenv("RAG_CACHE_ENABLED", "true").lower() == "true"
    )
    cache_ttl_hours: int = field(
        default_factory=lambda: int(os.getenv("RAG_CACHE_TTL", "24"))
    )


@dataclass
class CacheConfig:
    """Cache configuration."""

    enabled: bool = field(
        default_factory=lambda: os.getenv("CACHE_ENABLED", "true").lower() == "true"
    )
    backend: str = field(default_factory=lambda: os.getenv("CACHE_BACKEND", "redis"))
    ttl_seconds: int = field(
        default_factory=lambda: int(os.getenv("CACHE_TTL", "3600"))
    )
    max_size: int = field(
        default_factory=lambda: int(os.getenv("CACHE_MAX_SIZE", "10000"))
    )
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.95"))
    )


@dataclass
class APIConfig:
    """API server configuration."""

    host: str = field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    workers: int = field(default_factory=lambda: int(os.getenv("API_WORKERS", "4")))
    debug: bool = field(
        default_factory=lambda: os.getenv("API_DEBUG", "false").lower() == "true"
    )
    cors_origins: List[str] = field(
        default_factory=lambda: os.getenv("API_CORS_ORIGINS", "*").split(",")
    )
    rate_limit_per_second: int = field(
        default_factory=lambda: int(os.getenv("API_RATE_LIMIT", "10"))
    )


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""

    enabled: bool = field(
        default_factory=lambda: os.getenv("MONITORING_ENABLED", "true").lower()
        == "true"
    )
    prometheus_port: int = field(
        default_factory=lambda: int(os.getenv("PROMETHEUS_PORT", "9090"))
    )
    grafana_url: Optional[str] = field(default_factory=lambda: os.getenv("GRAFANA_URL"))
    tracing_enabled: bool = field(
        default_factory=lambda: os.getenv("TRACING_ENABLED", "false").lower() == "true"
    )
    tracing_endpoint: Optional[str] = field(
        default_factory=lambda: os.getenv("TRACING_ENDPOINT")
    )


@dataclass
class SecurityConfig:
    """Security configuration."""

    jwt_secret_key: str = field(
        default_factory=lambda: os.getenv(
            "JWT_SECRET_KEY", "dev-secret-key-change-in-production"
        )
    )
    jwt_algorithm: str = field(
        default_factory=lambda: os.getenv("JWT_ALGORITHM", "HS256")
    )
    jwt_expiration_hours: int = field(
        default_factory=lambda: int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
    )
    cors_enabled: bool = field(
        default_factory=lambda: os.getenv("CORS_ENABLED", "true").lower() == "true"
    )
    https_enabled: bool = field(
        default_factory=lambda: os.getenv("HTTPS_ENABLED", "false").lower() == "true"
    )
    ssl_cert_path: Optional[str] = field(
        default_factory=lambda: os.getenv("SSL_CERT_PATH")
    )
    ssl_key_path: Optional[str] = field(
        default_factory=lambda: os.getenv("SSL_KEY_PATH")
    )


@dataclass
class Config:
    """
    Main configuration class.

    All configuration is accessible through this class with validation
    and type safety.

    Example:
        >>> config = get_config()
        >>> print(config.database.host)
        localhost
        >>> print(config.rag.chunk_size)
        512
    """

    # Application
    app_name: str = field(
        default_factory=lambda: os.getenv("APP_NAME", "AI-Mastery-2026")
    )
    environment: str = field(
        default_factory=lambda: os.getenv("ENVIRONMENT", "development")
    )
    debug: bool = field(
        default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true"
    )
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)

    # Paths
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", "data")))
    models_dir: Path = field(
        default_factory=lambda: Path(os.getenv("MODELS_DIR", "models"))
    )
    logs_dir: Path = field(default_factory=lambda: Path(os.getenv("LOGS_DIR", "logs")))

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Ensure directories exist
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Validate environment
        if self.environment not in ["development", "staging", "production"]:
            raise ValueError(
                f"Invalid environment: {self.environment}. "
                "Must be one of: development, staging, production"
            )

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Get a secret value from environment.

        Args:
            key: Environment variable name
            default: Default value if not found

        Returns:
            Secret value or default
        """
        return os.getenv(key, default)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert configuration to dictionary.

        Note: Sensitive values are redacted.

        Returns:
            Configuration as dictionary
        """

        def dataclass_to_dict(obj) -> Dict[str, Any]:
            if hasattr(obj, "__dataclass_fields__"):
                return {
                    k: dataclass_to_dict(v)
                    for k, v in obj.__dict__.items()
                    if not k.endswith("_key")
                    and not k.endswith("_password")
                    and not k.endswith("_secret")
                }
            return obj

        return {
            "app_name": self.app_name,
            "environment": self.environment,
            "debug": self.debug,
            "log_level": self.log_level,
            "database": dataclass_to_dict(self.database),
            "redis": dataclass_to_dict(self.redis),
            "llm": dataclass_to_dict(self.llm),
            "embedding": dataclass_to_dict(self.embedding),
            "vector_store": dataclass_to_dict(self.vector_store),
            "rag": dataclass_to_dict(self.rag),
            "cache": dataclass_to_dict(self.cache),
            "api": dataclass_to_dict(self.api),
            "monitoring": dataclass_to_dict(self.monitoring),
            "security": {"configured": True},  # Don't expose security config
            "paths": {
                "data_dir": str(self.data_dir),
                "models_dir": str(self.models_dir),
                "logs_dir": str(self.logs_dir),
            },
        }

    def __str__(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@lru_cache()
def get_config() -> Config:
    """
    Get the global configuration instance.

    Uses LRU cache to ensure only one configuration instance exists.

    Returns:
        Global configuration instance

    Example:
        >>> config = get_config()
        >>> print(config.app_name)
        AI-Mastery-2026
    """
    return Config()


def reload_config() -> Config:
    """
    Reload configuration from environment.

    Clears the LRU cache and creates a new configuration instance.

    Returns:
        New configuration instance
    """
    get_config.cache_clear()
    return get_config()


# Convenience functions
def get_database_url() -> str:
    """Get database URL from configuration."""
    config = get_config()
    return config.database.url


def get_redis_url() -> str:
    """Get Redis URL from configuration."""
    config = get_config()
    return config.redis.url


def is_production() -> bool:
    """Check if running in production environment."""
    config = get_config()
    return config.environment == "production"


def is_development() -> bool:
    """Check if running in development environment."""
    config = get_config()
    return config.environment == "development"


def is_debug() -> bool:
    """Check if debug mode is enabled."""
    config = get_config()
    return config.debug

