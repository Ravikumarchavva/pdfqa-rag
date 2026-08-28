"""RAGPipeline factory — thin wiring, no reimplementation.

Delegates entirely to ravi-engine's existing factories:
  - ``create_embedding_client()`` from substrate.integrations.llm
  - ``LLMFactory`` from substrate.integrations.llm
  - ``RAGPipeline`` from substrate.capabilities.knowledge.pipeline
  - ``PgVectorStore`` via store/factory.py
  - ``DocumentIngestPipeline``/``ask`` from substrate.capabilities.knowledge
    (multimodal PDF ingestion + query — real orchestration logic that lives
    in agent-substrate, not pdfqa_rag, so any downstream consumer gets it;
    this module only translates pdfqa_rag's own Config objects into the
    constructor args those classes actually take)

All heavy imports are deferred to function bodies so ``import pdfqa_rag``
stays fast.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pdfqa_rag.config import (
    AppConfig,
    BlobStoreConfig,
    ChunkingConfig,
    DocumentIntelligenceConfig,
    EmbedConfig,
    LLMConfig,
    MultimodalEmbedConfig,
)
from pdfqa_rag.store.factory import build_vector_store

if TYPE_CHECKING:
    from substrate.capabilities.knowledge import DocumentIngestPipeline
    from substrate.capabilities.knowledge.pipeline import RAGPipeline
    from substrate.capabilities.storage.s3 import S3FileStore
    from substrate.kernel.llm import EmbeddingClient, LLMClient
    from substrate.runtimes.document_intelligence.client import ExtractionClient
    from substrate.runtimes.embedding_reranker.service.embedding import EmbeddingReranker

logger = logging.getLogger(__name__)


def build_embed_client(cfg: EmbedConfig) -> EmbeddingClient:
    """Build an embedding client via ravi-engine's ``create_embedding_client``.

    The model prefix selects the backend automatically:
      ``sentence-transformers/<name>``  →  CPU local (no server needed)
      ``<openai-model>``                →  OpenAI API
      ``<gemini-model>``                →  Gemini API
    """
    from substrate.integrations.llm import create_embedding_client

    logger.debug("Embed: model=%s base_url=%s", cfg.model, cfg.base_url or "(default)")
    return create_embedding_client(
        cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url or None,
    )


def build_llm_client(cfg: LLMConfig) -> LLMClient:
    """Build a generation client using ravi-engine's ``LLMFactory``."""
    from substrate.integrations.llm import LLMFactory

    logger.debug("LLM: model=%s base_url=%s", cfg.model, cfg.base_url or "(openai)")
    return LLMFactory(cfg.model, cfg.api_key).build(
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        base_url=cfg.base_url or None,
    )


async def build_pipeline(cfg: AppConfig) -> RAGPipeline:
    """Wire embed client + vector store into a ready-to-use RAGPipeline."""
    from substrate.capabilities.knowledge.pipeline import RAGPipeline

    embed = build_embed_client(cfg.embed)
    store = build_vector_store(cfg.store, cfg.embed.dimensions)
    await store.ensure_table()
    return RAGPipeline(embedding_client=embed, vector_store=store)


def build_extraction_client(cfg: DocumentIntelligenceConfig) -> ExtractionClient:
    """Build a document-intelligence client using agent-substrate's own class."""
    from substrate.runtimes.document_intelligence.client import ExtractionClient

    logger.debug("ExtractionClient: service_url=%s", cfg.service_url)
    return ExtractionClient(base_url=cfg.service_url, timeout_s=cfg.timeout_s)


def build_multimodal_embedder(cfg: MultimodalEmbedConfig) -> EmbeddingReranker:
    """Build the llama-embed/llama-rerank sidecar client using agent-substrate's own class."""
    from substrate.runtimes.embedding_reranker.service.embedding import EmbeddingReranker

    logger.debug(
        "EmbeddingReranker: embed=%s rerank=%s",
        cfg.embed_server_url,
        cfg.rerank_server_url,
    )
    return EmbeddingReranker(
        embed_server_url=cfg.embed_server_url,
        rerank_server_url=cfg.rerank_server_url,
        timeout=cfg.timeout_s,
    )


def build_blob_store(cfg: BlobStoreConfig) -> S3FileStore:
    """Build the SeaweedFS-backed blob store for source PDFs + extracted
    images. Uses agent-substrate's own S3FileStore (wraps S3Connector,
    speaks the S3 API — compatible with SeaweedFS's S3 gateway)."""
    from substrate.capabilities.storage.s3 import S3FileStore

    logger.debug("BlobStore: endpoint=%s bucket=%s", cfg.endpoint_url, cfg.bucket)
    return S3FileStore(
        endpoint_url=cfg.endpoint_url,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        bucket=cfg.bucket,
        region=cfg.region,
    )


def build_segmenter(cfg: ChunkingConfig):
    """Sentence segmenter for the chunker, or ``None`` for the built-in
    regex default. ``"sat"`` needs agent-substrate's ``chunking`` extra and
    downloads its model on first use, so it is constructed only when
    actually selected — never eagerly."""
    if cfg.segmenter == "regex":
        return None
    if cfg.segmenter == "sat":
        from substrate.capabilities.knowledge.segmentation import SaTSegmenter

        logger.info("Segmenter: SaT (%s)", cfg.sat_model)
        return SaTSegmenter(cfg.sat_model)
    raise ValueError(f"Unknown segmenter {cfg.segmenter!r}. Expected 'regex' or 'sat'.")


def build_document_ingest_pipeline(
    cfg: AppConfig, store, blob_store: S3FileStore | None = None
) -> DocumentIngestPipeline:
    """Wire extraction client + multimodal embedder + store (+ optional blob
    store) into a ready-to-use ``DocumentIngestPipeline`` (see
    substrate.capabilities.knowledge) — pass in the store yourself (e.g.
    from ``build_vector_store``) since its dimensions depend on the
    embedding model, which this function has no opinion about. Omit
    ``blob_store`` to inline extracted images instead of uploading them.
    """
    from substrate.capabilities.knowledge import DocumentIngestPipeline

    extraction = build_extraction_client(cfg.doc_intel)
    embedder = build_multimodal_embedder(cfg.mm_embed)
    return DocumentIngestPipeline(
        extraction,
        embedder,
        store,
        blob_store=blob_store,
        key_prefix=cfg.storage.key_prefix,
        chunk_size=cfg.chunking.size,
        chunk_overlap=cfg.chunking.overlap,
        segmenter=build_segmenter(cfg.chunking),
    )
