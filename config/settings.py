"""Central application settings.

Loads configuration from environment variables (and an optional ``.env`` file)
using Pydantic ``BaseSettings``. Every tunable system parameter lives here so
that the rest of the codebase can import a single, validated ``settings`` object
instead of reaching into ``os.environ`` directly.
"""

from __future__ import annotations

from functools import lru_cache
from typing import List

try:  # Pydantic v2 moved BaseSettings into ``pydantic-settings``.
    from pydantic_settings import BaseSettings, SettingsConfigDict

    _PYDANTIC_V2 = True
except ImportError:  # pragma: no cover - fallback for Pydantic v1 installs.
    from pydantic import BaseSettings  # type: ignore

    SettingsConfigDict = dict  # type: ignore
    _PYDANTIC_V2 = False

from pydantic import Field


class Settings(BaseSettings):
    """Strongly-typed, environment-driven application configuration."""

    # ------------------------------------------------------------------ #
    # Service metadata
    # ------------------------------------------------------------------ #
    app_name: str = Field(default="Enterprise Hybrid RAG & Semantic Search Engine")
    app_version: str = Field(default="1.0.0")
    environment: str = Field(default="development", description="dev | staging | prod")
    log_level: str = Field(default="INFO")
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    # ------------------------------------------------------------------ #
    # Embedding / vector configuration
    # ------------------------------------------------------------------ #
    embedding_model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformers model used for dense embeddings.",
    )
    vector_dimensionality: int = Field(
        default=384, description="Dense embedding vector size for the chosen model."
    )
    embedding_device: str = Field(
        default="cpu", description="Torch device: 'cpu' or 'cuda'."
    )
    embedding_batch_size: int = Field(default=32)

    # ------------------------------------------------------------------ #
    # Qdrant configuration
    # ------------------------------------------------------------------ #
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_grpc_port: int = Field(default=6334)
    qdrant_api_key: str = Field(default="")
    qdrant_collection_name: str = Field(default="enterprise_documents")
    qdrant_prefer_grpc: bool = Field(default=False)
    qdrant_timeout: float = Field(default=10.0)

    # ------------------------------------------------------------------ #
    # Redis / semantic cache configuration
    # ------------------------------------------------------------------ #
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: str = Field(default="")
    redis_ttl_seconds: int = Field(
        default=3600, description="Time-to-live for cached query/response entries."
    )
    semantic_cache_prefix: str = Field(default="semcache")

    # ------------------------------------------------------------------ #
    # Retrieval / search parameters
    # ------------------------------------------------------------------ #
    similarity_threshold: float = Field(
        default=0.88,
        description="Cosine similarity threshold for a semantic cache hit.",
    )
    retrieval_top_k: int = Field(default=3, description="Chunks fetched per query.")
    chunk_size_tokens: int = Field(default=512)
    chunk_overlap_tokens: int = Field(default=64)

    # ------------------------------------------------------------------ #
    # vLLM / inference configuration
    # ------------------------------------------------------------------ #
    vllm_endpoint: str = Field(
        default="http://localhost:8001/v1",
        description="OpenAI-compatible base URL exposed by the vLLM server.",
    )
    vllm_model: str = Field(default="meta-llama/Meta-Llama-3-8B-Instruct")
    vllm_api_key: str = Field(default="EMPTY")
    llm_temperature: float = Field(default=0.2)
    llm_max_tokens: int = Field(default=256)
    llm_top_p: float = Field(default=0.95)
    llm_request_timeout: float = Field(default=60.0)

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #
    evaluation_output_dir: str = Field(default="./artifacts/evaluation")

    if _PYDANTIC_V2:
        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            env_prefix="",
            case_sensitive=False,
            extra="ignore",
        )
    else:  # pragma: no cover - Pydantic v1 config style.

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            case_sensitive = False

    @property
    def qdrant_url(self) -> str:
        """Return the fully-qualified HTTP URL for the Qdrant node."""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def cors_origins(self) -> List[str]:
        """Permissive CORS origins for local development."""
        return ["*"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide :class:`Settings` instance."""
    return Settings()


# Convenience singleton importable as ``from config.settings import settings``.
settings: Settings = get_settings()
