"""
Core Configuration Module
=========================
Pydantic Settings for all environment-based configuration.
All settings are loaded from environment variables or .env file.

إعدادات التكوين الأساسية - تُحمّل من متغيرات البيئة
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Design Decision: Using pydantic-settings for type-safe configuration
    that validates at startup rather than runtime failures.

    قرار التصميم: استخدام pydantic-settings للتحقق من الإعدادات عند بدء التشغيل
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # =========================================================================
    # Application / التطبيق
    # =========================================================================
    app_name: str = Field(default="rag-engine-mini", description="Application name")
    env: Literal["dev", "staging", "prod"] = Field(default="dev", description="Environment")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")

    # =========================================================================
    # Security / الأمان
    # =========================================================================
    api_key_header: str = Field(default="X-API-KEY", description="Header for API key auth")
    jwt_secret_key: str | None = Field(
        default=None, description="Secret key for JWT signing (use env var!)"
    )
    jwt_algorithm: str = Field(
        default="HS256", description="JWT algorithm (HS256 for dev, RS256 for production)"
    )
    jwt_access_expire_minutes: int = Field(
        default=15, description="Access token lifetime in minutes"
    )
    jwt_refresh_expire_days: int = Field(default=7, description="Refresh token lifetime in days")

    # =========================================================================
    # Database / قاعدة البيانات
    # =========================================================================
    database_url: str = Field(
        default="postgresql+psycopg://postgres:postgres@localhost:5432/rag",
        description="PostgreSQL connection URL",
    )
    db_pool_size: int = Field(default=5, ge=1, le=20)
    db_max_overflow: int = Field(default=10, ge=0, le=50)
    use_real_db: bool = Field(default=False, description="Use Postgres if True, memory if False")

    # =========================================================================
    # Redis / ريديس
    # =========================================================================
    redis_url: str = Field(default="redis://localhost:6379/0")
    celery_broker_url: str = Field(default="redis://localhost:6379/1")
    celery_result_backend: str = Field(default="redis://localhost:6379/2")
    embedding_cache_ttl: int = Field(default=604800, description="7 days in seconds")

    # =========================================================================
    # Vector Store (Qdrant) / مخزن المتجهات
    # =========================================================================
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection: str = Field(default="chunks")
    qdrant_api_key: str | None = Field(default=None, description="For Qdrant Cloud")
    embedding_dim: int = Field(default=1536, description="Must match embedding model")

    # =========================================================================
    # Embeddings / التضمينات
    # =========================================================================
    embeddings_backend: Literal["openai", "local"] = Field(default="openai")
    openai_embed_model: str = Field(default="text-embedding-3-small")
    local_embed_model: str = Field(default="all-MiniLM-L6-v2")
    local_embed_device: Literal["cpu", "cuda"] = Field(default="cpu")

    # =========================================================================
    # LLM Provider / مزود نموذج اللغة
    # =========================================================================
    llm_backend: Literal["openai", "ollama", "gemini", "huggingface"] = Field(default="openai")

    # OpenAI
    openai_api_key: str | None = Field(default=None)
    openai_chat_model: str = Field(default="gpt-4o-mini")
    openai_max_tokens: int = Field(default=700)
    openai_temperature: float = Field(default=0.2, ge=0, le=2)

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434")
    ollama_chat_model: str = Field(default="llama3.1")
    ollama_embed_model: str = Field(default="nomic-embed-text")

    # Gemini
    gemini_api_key: str | None = Field(default=None)
    gemini_model: str = Field(default="gemini-1.5-flash")

    # Hugging Face
    hf_api_key: str | None = Field(default=None)
    hf_model: str = Field(default="mistralai/Mistral-7B-Instruct-v0.2")
    hf_use_inference_api: bool = Field(default=True)

    # =========================================================================
    # Reranking / إعادة الترتيب
    # =========================================================================
    rerank_backend: Literal["cross_encoder", "llm", "none"] = Field(default="cross_encoder")
    cross_encoder_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    cross_encoder_device: Literal["cpu", "cuda"] = Field(default="cpu")
    rerank_top_n: int = Field(default=8, ge=1, le=50)

    # =========================================================================
    # Retrieval / الاسترجاع
    # =========================================================================
    default_k_vector: int = Field(default=30, ge=1, le=200)
    default_k_keyword: int = Field(default=30, ge=1, le=200)
    default_fused_limit: int = Field(default=40, ge=1, le=200)
    rrf_k: int = Field(default=60, description="RRF constant")

    # =========================================================================
    # Chunking / التقطيع
    # =========================================================================
    chunk_max_tokens: int = Field(default=512, ge=50, le=2000)
    chunk_overlap_tokens: int = Field(default=50, ge=0, le=500)
    chunk_encoding: str = Field(default="cl100k_base", description="tiktoken encoding")

    # File Upload / رفع الملفات
    # =========================================================================
    upload_dir: str = Field(default="./uploads")
    max_upload_mb: int = Field(default=20, ge=1, le=100)
    allowed_extensions: str = Field(default="pdf,docx,txt")

    # Web Search (Stage 5)
    tavily_api_key: str | None = Field(default=None)

    @property
    def allowed_extensions_list(self) -> list[str]:
        """Parse comma-separated extensions into list."""
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def max_upload_bytes(self) -> int:
        """Convert MB to bytes."""
        return self.max_upload_mb * 1024 * 1024

    # =========================================================================
    # Workers / العمال
    # =========================================================================
    celery_worker_concurrency: int = Field(default=4)
    celery_task_time_limit: int = Field(default=600)

    # =========================================================================
    # Observability / المراقبة
    # =========================================================================
    enable_metrics: bool = Field(default=True)
    metrics_path: str = Field(default="/metrics")
    request_id_header: str = Field(default="X-Request-ID")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Using @lru_cache ensures settings are only loaded once.
    استخدام @lru_cache يضمن تحميل الإعدادات مرة واحدة فقط
    """
    return Settings()


# Convenience alias
settings = get_settings()
