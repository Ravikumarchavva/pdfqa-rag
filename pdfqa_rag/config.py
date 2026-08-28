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

    lancedb_path: str = "data/lancedb"
    """Used only when backend == "lancedb" — see LanceDBVectorStore."""

    default_collection: str = "pdfqa"


class BlobStoreConfig(BaseSettings):
    """S3-compatible object store for raw source documents (PDFs/text).

    Defaults match ``docker-compose.yml``'s SeaweedFS S3 gateway — run
    ``make infra-up`` first. Swappable for AWS S3 / MinIO in production
    with no code change, since access is purely via the S3 API.
    """

    model_config = SettingsConfigDict(env_prefix="STORAGE_", extra="ignore")

    endpoint_url: str = "http://localhost:8333"
    access_key: str = "local"
    secret_key: str = "local"
    bucket: str = "pdfqa-raw"
    region: str = "us-east-1"


class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="PIPELINE_", extra="ignore")

    concurrency: int = 8
    """Parallel embed calls during batch ingestion."""

    checkpoint_every: int = 500

    data_dir: str = "data/pdfQA-Annotations"

    categories: list[str] = Field(default_factory=lambda: ["real-pdfQA"])


class DocumentIntelligenceConfig(BaseSettings):
    """document-intelligence service (PaddleOCR extraction) — see
    agent-substrate/src/substrate/runtimes/document_intelligence/client.py.

    Defaults to localhost; point at a GPU host for larger batches, e.g.
    ``DOC_INTEL_SERVICE_URL=http://192.168.0.11:8021``.
    """

    model_config = SettingsConfigDict(env_prefix="DOC_INTEL_", extra="ignore")

    service_url: str = "http://localhost:8021"
    timeout_s: float = 90.0
    num_replicas: int = 3
    """Endpoints derived by dataset_ingest_gpu._default_endpoints() (ports
    incrementing from service_url's port). Real corporate PDFs run 30-180+
    pages; 90s is too short for those — raise DOC_INTEL_TIMEOUT_S for a
    full run. Set to 6 to match a doubled-up (2-per-GPU) deployment."""


class MultimodalEmbedConfig(BaseSettings):
    """llama-embed / llama-rerank sidecars (Qwen3-VL text+image embedding) —
    see agent-substrate/src/substrate/runtimes/embedding_reranker/service/embedding.py.

    Separate from ``EmbedConfig`` above: that one wraps ravi-engine's
    generic text-only ``create_embedding_client`` (sentence-transformers/
    OpenAI/Gemini); this targets the multimodal llama-server sidecars
    specifically, which embed_text and embed_image into the same space.
    Point at a GPU host for larger batches, e.g.
    ``MM_EMBED_EMBED_SERVER_URL=http://192.168.0.11:8031``.
    """

    model_config = SettingsConfigDict(env_prefix="MM_EMBED_", extra="ignore")

    embed_server_url: str = "http://localhost:8031"
    rerank_server_url: str = "http://localhost:8032"
    timeout_s: float = 120.0
    """Per-request httpx timeout. EmbeddingReranker's own default (30s) is
    tuned for same-host calls; a remote GPU host (e.g. epyc over LAN) needs
    more headroom for larger image payloads plus real inference time."""


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    embed: EmbedConfig = Field(default_factory=EmbedConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    storage: BlobStoreConfig = Field(default_factory=BlobStoreConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    doc_intel: DocumentIntelligenceConfig = Field(default_factory=DocumentIntelligenceConfig)
    mm_embed: MultimodalEmbedConfig = Field(default_factory=MultimodalEmbedConfig)
    log_level: str = "INFO"

class Settings(BaseSettings):
    ROOT_DIR: ClassVar[Path] = Path(__file__).parent.parent

settings = Settings()