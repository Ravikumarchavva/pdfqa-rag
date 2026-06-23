"""VectorStore factory — maps config to a concrete store implementation.

To add a new backend (e.g. ChromaDB, Qdrant):
1. Add a new branch in ``build_vector_store``.
2. Expose the same interface: ``VectorStore`` Protocol from ``agent_substratekernel.vector``.
"""

from __future__ import annotations

import logging

from pdfqa_rag.config import StoreConfig

logger = logging.getLogger(__name__)


def build_vector_store(cfg: StoreConfig, dimensions: int):
    """Instantiate the configured vector store.

    Args:
        cfg: Store configuration (backend, database_url, …).
        dimensions: Embedding dimensionality — must match the embed model.

    Returns:
        A concrete ``VectorStore`` instance (not yet connected).
        Call ``await store.ensure_table()`` before first use.
    """
    if cfg.backend == "pgvector":
        return _build_pgvector(cfg, dimensions)

    raise NotImplementedError(
        f"Vector store backend {cfg.backend!r} is not supported yet. "
        "Add a new branch in pdfqa_rag/store/factory.py."
    )


def _build_pgvector(cfg: StoreConfig, dimensions: int):
    from agent_substratecapabilities.vector.pgvector_store import PgVectorStore
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    logger.debug(
        "Building PgVectorStore (dims=%d) against %s",
        dimensions,
        _redact_url(cfg.database_url),
    )
    engine = create_async_engine(cfg.database_url, pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return PgVectorStore(session_factory=session_factory, engine=engine, dimensions=dimensions)


def _redact_url(url: str) -> str:
    """Replace password in DSN with *** for safe logging."""
    import re
    return re.sub(r"://([^:@]+):([^@]+)@", r"://\1:***@", url)
