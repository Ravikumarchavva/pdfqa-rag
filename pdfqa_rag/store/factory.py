"""VectorStore factory — maps config to a concrete store implementation.

To add a new backend (e.g. ChromaDB, Qdrant):
1. Add a new branch in ``build_vector_store``.
2. Expose the same interface: ``VectorStore`` Protocol from ``substrate.kernel.storage.vector``.
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
    if cfg.backend == "in_memory":
        return _build_in_memory()
    if cfg.backend == "lancedb":
        return _build_lancedb(cfg)

    raise NotImplementedError(
        f"Vector store backend {cfg.backend!r} is not supported yet. "
        "Add a new branch in pdfqa_rag/store/factory.py."
    )


def _build_in_memory():
    """Dependency-free store for dev/notebook use — see
    substrate.agents.storage.InMemoryVectorStore. No ``ensure_table()``;
    state lives only in the current process.
    """
    from substrate.agents.storage import InMemoryVectorStore

    logger.debug("Building InMemoryVectorStore (dev-only, no persistence)")
    return InMemoryVectorStore()


def _build_lancedb(cfg: StoreConfig):
    """File-based store for dev — see substrate.capabilities.vector.LanceDBVectorStore.
    No ``ensure_table()``; the directory + per-collection tables are created
    on first use. Persists across process restarts, unlike ``in_memory``.
    Requires the ``rag`` extra on agent-substrate (``lancedb``, ~110MB).
    """
    from substrate.capabilities.vector import LanceDBVectorStore

    logger.debug("Building LanceDBVectorStore at %s", cfg.lancedb_path)
    return LanceDBVectorStore(cfg.lancedb_path)


def _build_pgvector(cfg: StoreConfig, dimensions: int):
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from substrate.capabilities.vector.pgvector_store import PgVectorStore

    logger.debug(
        "Building PgVectorStore (dims=%d) against %s",
        dimensions,
        _redact_url(cfg.database_url),
    )
    connect_args = _ssl_connect_args() if cfg.require_ssl else {}
    engine = create_async_engine(
        cfg.database_url,
        pool_pre_ping=True,
        pool_size=cfg.pool_size,
        max_overflow=cfg.max_overflow,
        connect_args=connect_args,
    )
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return PgVectorStore(session_factory=session_factory, engine=engine, dimensions=dimensions)


def _ssl_connect_args() -> dict:
    """asyncpg's own ``connect()`` has no ``sslmode`` parameter (only
    ``ssl``) — SQLAlchemy's asyncpg dialect passes a URL's query params
    straight through as ``connect()`` kwargs with no translation, so a
    Postgres URL carrying psycopg's conventional ``?sslmode=require`` would
    fail with ``TypeError: unexpected keyword argument 'sslmode'`` rather
    than actually enabling TLS. Passing a real ``ssl.SSLContext`` via
    ``connect_args`` instead sidesteps that whole ambiguity — this is what
    ``StoreConfig.require_ssl=True`` triggers. A hosted-Postgres URL should
    NOT itself carry ``?sslmode=`` when this is used.
    """
    import ssl

    return {"ssl": ssl.create_default_context()}


def _redact_url(url: str) -> str:
    """Replace password in DSN with *** for safe logging."""
    import re
    return re.sub(r"://([^:@]+):([^@]+)@", r"://\1:***@", url)
