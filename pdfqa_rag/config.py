"""Pydantic Settings config for pdfqa-rag.

All values resolve from environment variables or a ``.env`` file.

For local GPU servers (vLLM / Ollama / llama.cpp), set ``EMBED_BASE_URL``
and ``LLM_BASE_URL`` to point at your OpenAI-compatible endpoints.
Leave them empty to use the public OpenAI API.

Model naming follows ravi-engine's ``LLMFactory`` convention:
  - Bare name (``gpt-4o-mini``) → OpenAI
  - Provider-prefixed (``groq/llama-3.3-70b``) → that provider
  - ``compatible/<name>`` + ``base_url`` → any local server
"""

from __future__ import annotations

from typing import ClassVar

from anyio import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbedConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBED_", extra="ignore")

    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    """Model string passed to ``create_embedding_client()``.
    Prefix selects the backend:
      sentence-transformers/<name>  →  CPU local, no server needed
      <openai-model-name>           →  OpenAI API
      <gemini-model-name>           →  Gemini API
    """

    base_url: str = ""
    """OpenAI-compatible base URL for GPU servers. Unused for sentence-transformers."""

    dimensions: int = 384
    """Must match the model: 384 for all-MiniLM-L6-v2, 768 for all-mpnet-base-v2."""

    batch_size: int = 64
    """Texts per embed call."""

    api_key: str = "local"
    """API key. Use 'local' for servers that don't enforce auth."""


class LLMConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")

    model: str = "compatible/llama-3.1-8b-instruct"
    """LLMFactory model string. Prefix 'compatible/' for local servers."""

    base_url: str = ""
    """OpenAI-compatible base URL for local GPU servers."""

    temperature: float = 0.1
    max_tokens: int = 512
    api_key: str = "local"


class StoreConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="STORE_", extra="ignore")

    backend: str = "pgvector"

    database_url: str = Field(
        "postgresql+asyncpg://postgres:postgres@localhost:5435/pdfqa",
        alias="DATABASE_URL",
    )

    default_collection: str = "pdfqa"


class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PIPELINE_", extra="ignore")

    concurrency: int = 8
    """Parallel embed calls during batch ingestion."""

    checkpoint_every: int = 500

    data_dir: str = "data/pdfQA-Annotations"

    categories: list[str] = Field(default_factory=lambda: ["real-pdfQA"])


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    log_level: str = "INFO"

class Settings(BaseSettings):
    ROOT_DIR: ClassVar[Path] = Path(__file__).parent.parent

settings = Settings()