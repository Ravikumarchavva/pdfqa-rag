"""RAGPipeline factory — thin wiring, no reimplementation.

Delegates entirely to ravi-engine's existing factories:
  - ``create_embedding_client()`` from ravi.integrations.llm
  - ``LLMFactory`` from ravi.integrations.llm
  - ``RAGPipeline`` from ravi.capabilities.knowledge.pipeline
  - ``PgVectorStore`` via store/factory.py

All heavy imports are deferred to function bodies so ``import pdfqa_rag``
stays fast.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pdfqa_rag.config import AppConfig, EmbedConfig, LLMConfig
from pdfqa_rag.store.factory import build_vector_store

if TYPE_CHECKING:
    from ravi.capabilities.knowledge.pipeline import RAGPipeline
    from ravi.kernel.llm import EmbeddingClient, LLMClient

logger = logging.getLogger(__name__)


def build_embed_client(cfg: EmbedConfig) -> EmbeddingClient:
    """Build an embedding client via ravi-engine's ``create_embedding_client``.

    The model prefix selects the backend automatically:
      ``sentence-transformers/<name>``  →  CPU local (no server needed)
      ``<openai-model>``                →  OpenAI API
      ``<gemini-model>``                →  Gemini API
    """
    from ravi.integrations.llm import create_embedding_client

    logger.debug("Embed: model=%s base_url=%s", cfg.model, cfg.base_url or "(default)")
    return create_embedding_client(
        cfg.model,
        api_key=cfg.api_key,
        base_url=cfg.base_url or None,
    )


def build_llm_client(cfg: LLMConfig) -> LLMClient:
    """Build a generation client using ravi-engine's ``LLMFactory``."""
    from ravi.integrations.llm import LLMFactory

    logger.debug("LLM: model=%s base_url=%s", cfg.model, cfg.base_url or "(openai)")
    return LLMFactory(cfg.model, cfg.api_key).build(
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        base_url=cfg.base_url or None,
    )


async def build_pipeline(cfg: AppConfig) -> RAGPipeline:
    """Wire embed client + vector store into a ready-to-use RAGPipeline."""
    from ravi.capabilities.knowledge.pipeline import RAGPipeline

    embed = build_embed_client(cfg.embed)
    store = build_vector_store(cfg.store, cfg.embed.dimensions)
    await store.ensure_table()
    return RAGPipeline(embedding_client=embed, vector_store=store)
