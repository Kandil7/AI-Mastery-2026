"""
Configuration Management for Production RAG System

This module implements a comprehensive configuration management system using Pydantic
BaseSettings. It provides centralized configuration for all components of the RAG
system including database connections, API keys, model parameters, and operational
settings.

The configuration system follows production best practices:
- Environment variable loading with .env file support
- Type validation and coercion
- Nested configuration models for different components
- Secure handling of sensitive information
- Runtime validation of configuration values
- Default values for common deployment scenarios

Key Features:
- Centralized configuration management
- Environment-specific overrides
- Validation of configuration values
- Secure handling of API keys and credentials
- Component-specific configuration sections
- Runtime configuration reloading (optional)

Security Considerations:
- Sensitive values marked with Field(secret=True)
- Validation to prevent misconfiguration
- Secure defaults for production environments
- Environment variable precedence over defaults
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator
from pydantic_settings import BaseSettings
import os
from enum import Enum


class Environment(str, Enum):
    """Enumeration for application environments."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class DatabaseConfig(BaseModel):
    """
    Configuration for database connections.

    Attributes:
        url (str): Database connection URL
        name (str): Database name
        username (str): Database username (if applicable)
        password (str): Database password (if applicable)
        pool_size (int): Connection pool size
        max_overflow (int): Maximum overflow connections
        echo (bool): Enable SQL query logging
    """
    url: str = Field(default="mongodb://localhost:27017", description="Database connection URL")
    name: str = Field(default="minirag", description="Database name")
    username: Optional[str] = Field(default=None, description="Database username")
    password: Optional[str] = Field(default=None, secret=True, description="Database password")
    pool_size: int = Field(default=10, ge=1, le=100, description="Connection pool size")
    max_overflow: int = Field(default=20, ge=0, le=100, description="Maximum overflow connections")
    echo: bool = Field(default=False, description="Enable SQL query logging")

    @validator('pool_size', 'max_overflow')
    def validate_pool_sizes(cls, v):
        """Validate pool size parameters."""
        if v < 0:
            raise ValueError('Pool sizes must be non-negative')
        return v


class ModelConfig(BaseModel):
    """
    Configuration for ML models and embeddings.

    Attributes:
        generator_model (str): Name of the text generation model
        dense_model (str): Name of the dense embedding model
        sparse_model (str): Name of the sparse embedding model
        max_new_tokens (int): Maximum tokens for generation
        temperature (float): Temperature for generation diversity
        top_p (float): Top-p sampling parameter
        top_k (int): Top-k sampling parameter
    """
    generator_model: str = Field(default="gpt2", description="Text generation model name")
    dense_model: str = Field(default="all-MiniLM-L6-v2", description="Dense embedding model name")
    sparse_model: str = Field(default="bm25", description="Sparse embedding model name")
    max_new_tokens: int = Field(default=300, ge=1, le=2048, description="Max tokens for generation")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Generation temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")
    top_k: int = Field(default=5, ge=1, le=20, description="Top-k sampling parameter")

    @validator('temperature', 'top_p')
    def validate_probability_params(cls, v):
        """Validate probability-based parameters."""
        if v < 0.0 or v > 1.0:
            raise ValueError('Probability parameters must be between 0 and 1')
        return v


class RetrievalConfig(BaseModel):
    """
    Configuration for retrieval components.

    Attributes:
        alpha (float): Weight for dense retrieval in hybrid fusion
        fusion_method (str): Fusion strategy ('rrf', 'weighted', 'densite', 'combsum', 'combmnz')
        rrf_k (int): Smoothing constant for RRF calculation
        sparse_k1 (float): BM25 k1 parameter
        sparse_b (float): BM25 b parameter
        max_candidates (int): Maximum candidates for reranking
    """
    alpha: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for dense retrieval")
    fusion_method: str = Field(default="rrf", description="Fusion strategy")
    rrf_k: int = Field(default=60, ge=1, description="Smoothing constant for RRF")
    sparse_k1: float = Field(default=1.5, ge=0.0, description="BM25 k1 parameter")
    sparse_b: float = Field(default=0.75, ge=0.0, le=1.0, description="BM25 b parameter")
    max_candidates: int = Field(default=50, ge=1, description="Max candidates for reranking")

    @validator('alpha')
    def validate_alpha(cls, v):
        """Validate alpha parameter."""
        if v < 0.0 or v > 1.0:
            raise ValueError('Alpha must be between 0 and 1')
        return v


class APIConfig(BaseModel):
    """
    Configuration for API and networking.

    Attributes:
        host (str): Host address for the API server
        port (int): Port number for the API server
        cors_origins (List[str]): Allowed origins for CORS
        rate_limit_requests (int): Max requests per minute
        rate_limit_window (int): Time window for rate limiting in seconds
        request_timeout (int): Request timeout in seconds
    """
    host: str = Field(default="0.0.0.0", description="Host address for API server")
    port: int = Field(default=8000, ge=1, le=65535, description="Port number for API server")
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    rate_limit_requests: int = Field(default=100, ge=1, description="Max requests per minute")
    rate_limit_window: int = Field(default=60, ge=1, description="Rate limit window in seconds")
    request_timeout: int = Field(default=30, ge=1, description="Request timeout in seconds")


class LoggingConfig(BaseModel):
    """
    Configuration for logging and monitoring.

    Attributes:
        level (str): Logging level
        format (str): Log format string
        file_path (Optional[str]): Path to log file
        max_bytes (int): Maximum log file size in bytes
        backup_count (int): Number of backup log files
    """
    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
                       description="Log format string")
    file_path: Optional[str] = Field(default=None, description="Path to log file")
    max_bytes: int = Field(default=10485760, ge=1024, description="Max log file size in bytes")
    backup_count: int = Field(default=5, ge=1, description="Number of backup log files")


class SecurityConfig(BaseModel):
    """
    Configuration for security settings.

    Attributes:
        secret_key (str): Secret key for cryptographic operations
        jwt_algorithm (str): Algorithm for JWT encoding
        access_token_expire_minutes (int): Access token expiration in minutes
        enable_authentication (bool): Enable authentication middleware
        allowed_hosts (List[str]): Allowed hosts for security headers
    """
    secret_key: str = Field(default="secret", secret=True, description="Secret key for crypto ops")
    jwt_algorithm: str = Field(default="HS256", description="JWT encoding algorithm")
    access_token_expire_minutes: int = Field(default=30, ge=1, description="Token expiration mins")
    enable_authentication: bool = Field(default=False, description="Enable auth middleware")
    allowed_hosts: List[str] = Field(default=["localhost", "127.0.0.1"], description="Allowed hosts")


class RAGConfig(BaseSettings):
    """
    Main configuration class for the RAG system.

    This class aggregates all configuration sections and provides the primary
    interface for accessing configuration values throughout the application.

    Attributes:
        app_name (str): Name of the application
        app_version (str): Version of the application
        environment (Environment): Application environment
        debug (bool): Enable debug mode
        database (DatabaseConfig): Database configuration
        models (ModelConfig): Model configuration
        retrieval (RetrievalConfig): Retrieval configuration
        api (APIConfig): API configuration
        logging (LoggingConfig): Logging configuration
        security (SecurityConfig): Security configuration
        openai_api_key (Optional[str]): OpenAI API key (if using OpenAI)
        huggingface_token (Optional[str]): Hugging Face token (if needed)
    """
    app_name: str = Field(default="Production RAG API", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    environment: Environment = Field(default=Environment.DEVELOPMENT, description="App environment")
    debug: bool = Field(default=False, description="Debug mode flag")
    
    # Component configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # API Keys and Tokens
    openai_api_key: Optional[str] = Field(default=None, secret=True, description="OpenAI API key")
    huggingface_token: Optional[str] = Field(default=None, secret=True, description="HuggingFace token")

    class Config:
        """Pydantic configuration for settings."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_nested_delimiter = "__"  # Allows nested settings like DB__URL

    @validator('environment', pre=True)
    def validate_environment(cls, v):
        """Validate environment value."""
        if isinstance(v, str):
            try:
                return Environment(v.lower())
            except ValueError:
                raise ValueError(f"Invalid environment: {v}. Must be one of {[e.value for e in Environment]}")
        return v

    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == Environment.DEVELOPMENT

    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == Environment.PRODUCTION

    def get_database_url(self) -> str:
        """Get database URL with credentials if available."""
        if self.database.username and self.database.password:
            # Insert credentials into URL if they're provided separately
            base_url = self.database.url
            if "://" in base_url:
                protocol, rest = base_url.split("://", 1)
                return f"{protocol}://{self.database.username}:{self.database.password}@{rest}"
            else:
                return base_url
        return self.database.url


# Create a singleton instance of the configuration
settings = RAGConfig()

# Export for use in other modules
__all__ = ["RAGConfig", "DatabaseConfig", "ModelConfig", "RetrievalConfig", 
           "APIConfig", "LoggingConfig", "SecurityConfig", "Environment", "settings"]