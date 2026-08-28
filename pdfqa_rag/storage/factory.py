"""BlobStore factory — thin wiring around agent-substrate's S3FileStore.

Delegates entirely to ``substrate.capabilities.storage.s3.S3FileStore``
(aiobotocore-based), which speaks the plain S3 API — so SeaweedFS locally
and AWS S3 (or any S3-compatible service) in production are interchangeable via config alone.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pdfqa_rag.config import BlobStoreConfig

if TYPE_CHECKING:
    from substrate.capabilities.storage.s3 import S3FileStore

logger = logging.getLogger(__name__)


def build_blob_store(cfg: BlobStoreConfig) -> S3FileStore:
    """Instantiate an ``S3FileStore`` against the configured endpoint.

    Call ``await store.connect()`` before first use — it opens the
    session and creates the bucket if it doesn't exist yet.
    """
    from substrate.capabilities.storage.s3 import S3FileStore

    logger.debug(
        "Blob store: endpoint=%s bucket=%s", cfg.endpoint_url, cfg.bucket
    )
    return S3FileStore(
        endpoint_url=cfg.endpoint_url,
        access_key=cfg.access_key,
        secret_key=cfg.secret_key,
        bucket=cfg.bucket,
        region=cfg.region,
    )
