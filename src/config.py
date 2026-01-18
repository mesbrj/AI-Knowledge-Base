"""Configuration management for the application."""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Application
    app_name: str = Field(default="doc-rag-system", alias="APP_NAME")
    app_version: str = Field(default="1.0.0", alias="APP_VERSION")
    environment: Literal["development", "production", "testing"] = Field(
        default="development", alias="ENVIRONMENT"
    )
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # API
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    api_prefix: str = Field(default="/api/v1", alias="API_PREFIX")

    # Ollama
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="deepseek-r1:14b", alias="OLLAMA_MODEL")
    ollama_embedding_model: str = Field(
        default="nomic-embed-text:v1.5", alias="OLLAMA_EMBEDDING_MODEL"
    )
    ollama_temperature: float = Field(default=0.7, alias="OLLAMA_TEMPERATURE")
    ollama_request_timeout: int = Field(default=300, alias="OLLAMA_REQUEST_TIMEOUT")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_collection_name: str = Field(default="documents", alias="QDRANT_COLLECTION_NAME")
    qdrant_vector_size: int = Field(default=768, alias="QDRANT_VECTOR_SIZE")
    qdrant_api_key: str | None = Field(default=None, alias="QDRANT_API_KEY")

    # Storage
    storage_type: Literal["local", "s3"] = Field(default="local", alias="STORAGE_TYPE")
    storage_local_path: str = Field(default="./data/documents", alias="STORAGE_LOCAL_PATH")
    s3_bucket: str | None = Field(default=None, alias="S3_BUCKET")
    s3_region: str = Field(default="us-east-1", alias="S3_REGION")
    s3_access_key: str | None = Field(default=None, alias="S3_ACCESS_KEY")
    s3_secret_key: str | None = Field(default=None, alias="S3_SECRET_KEY")

    # Document Processing
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    max_concurrent_workers: int = Field(default=4, alias="MAX_CONCURRENT_WORKERS")

    # RAG
    rag_top_k: int = Field(default=5, alias="RAG_TOP_K")
    rag_score_threshold: float = Field(default=0.7, alias="RAG_SCORE_THRESHOLD")
    rag_max_context_length: int = Field(default=4000, alias="RAG_MAX_CONTEXT_LENGTH")

    # Summarization
    summary_max_length: int = Field(default=500, alias="SUMMARY_MAX_LENGTH")
    summary_min_length: int = Field(default=100, alias="SUMMARY_MIN_LENGTH")

    # Redis (Optional)
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")
    redis_max_connections: int = Field(default=10, alias="REDIS_MAX_CONNECTIONS")


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
