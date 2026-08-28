"""Checkpointed, GPU-parallel PDF ingestion driver for the multimodal RAG
evaluation run — dispatches files across N document-intelligence-gpu
endpoints via a shared work queue, sharing one embedder and one vector
store. Real, found-not-assumed reason for N endpoints instead of more
concurrency against one: a single document-intelligence-gpu instance
appears to serialize extraction requests internally (concurrent requests
to it made even small files time out; sequential succeeded every time).

Distinct from ``DocumentIngestPipeline.ingest_dataset`` (single endpoint,
``concurrency=1`` by design — the per-instance limitation above) and
``pdfqa_rag.pipeline.batch.BatchIngestor`` (chunk-level checkpoint, a
different ingestion path entirely, coupled to the old text-only
``RAGPipeline``). This composes ``DocumentIngestPipeline.ingest_file``
directly with its own file-level checkpoint and multi-endpoint dispatch.

Usage::

    uv run python -m pdfqa_rag.pipeline.dataset_ingest_gpu \\
        --dataset ClimRetrieve --collection pdfqa-pilot

    # Full benchmark, resumable if interrupted — rerun with the same
    # --collection and it picks up where it left off via the checkpoint.
    uv run python -m pdfqa_rag.pipeline.dataset_ingest_gpu \\
        --collection pdfqa-benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path

from pdfqa_rag.config import AppConfig, DocumentIntelligenceConfig, settings
from pdfqa_rag.pipeline.factory import build_extraction_client, build_multimodal_embedder
from pdfqa_rag.store.factory import build_vector_store

logger = logging.getLogger(__name__)

# Qwen3-VL-Embedding-2B — must match the dimensions the embed sidecar
# actually produces (verified this session: 2048-dim vectors).
_EMBEDDING_DIMENSIONS = 2048


class Checkpoint:
    """Append-only JSONL checkpoint keyed by PDF stem (== QAPair.file_name,
    per pdfqa_rag/data/loader.py — the join key back to ground truth).
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self.done: set[str] = self._load()

    def _load(self) -> set[str]:
        if not self._path.exists():
            return set()
        done: set[str] = set()
        with self._path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["file_name"])
                except (KeyError, json.JSONDecodeError):
                    pass
        logger.info("Checkpoint %s: %d files already ingested", self._path, len(done))
        return done

    async def record(
        self,
        file_name: str,
        *,
        status: str,
        text_docs: int = 0,
        image_docs: int = 0,
        error: str | None = None,
    ) -> None:
        entry = {
            "file_name": file_name,
            "status": status,
            "text_docs": text_docs,
            "image_docs": image_docs,
            "error": error,
        }
        async with self._lock:
            with self._path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            self.done.add(file_name)


async def ingest_dataset_gpu(
    dataset_dir: Path,
    *,
    collection: str,
    endpoints: list[str],
    checkpoint_path: Path,
    limit: int | None = None,
) -> dict[str, int]:
    """Ingest every PDF under ``dataset_dir`` across ``len(endpoints)``
    document-intelligence-gpu instances in parallel (one worker per
    endpoint, pulling from a shared work queue), resumable via
    ``checkpoint_path``. Each worker's own pipeline still processes its
    files one at a time internally (the per-instance serialization
    finding still applies) — parallelism comes from N independent workers,
    not from any one of them handling concurrent requests.
    """
    from substrate.capabilities.knowledge import DocumentIngestPipeline, ExtractionFailedError
    from substrate.runtimes.embedding_reranker.service.embedding import EmbeddingServiceError

    cfg = AppConfig()
    store = build_vector_store(cfg.store, dimensions=_EMBEDDING_DIMENSIONS)
    if hasattr(store, "ensure_table"):
        await store.ensure_table()
    embedder = build_multimodal_embedder(cfg.mm_embed)

    pipelines = [
        DocumentIngestPipeline(
            build_extraction_client(
                DocumentIntelligenceConfig(service_url=url, timeout_s=cfg.doc_intel.timeout_s)
            ),
            embedder,
            store,
        )
        for url in endpoints
    ]

    checkpoint = Checkpoint(checkpoint_path)
    pdfs = sorted(dataset_dir.glob("**/*.pdf"))
    if limit is not None:
        pdfs = pdfs[:limit]
    pending = [p for p in pdfs if p.stem not in checkpoint.done]
    logger.info(
        "Ingesting %d/%d files (%d already done, %d workers)",
        len(pending),
        len(pdfs),
        len(pdfs) - len(pending),
        len(pipelines),
    )

    queue: asyncio.Queue[Path] = asyncio.Queue()
    for p in pending:
        queue.put_nowait(p)

    stats = {"files": 0, "failed": 0, "text_docs": 0, "image_docs": 0}

    async def _worker(pipeline: DocumentIngestPipeline, worker_id: int) -> None:
        while True:
            try:
                path = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            try:
                n_text, n_img = await pipeline.ingest_file(path, collection=collection)
            except (OSError, ExtractionFailedError, EmbeddingServiceError) as exc:
                stats["failed"] += 1
                await checkpoint.record(path.stem, status="failed", error=str(exc))
                logger.warning("[worker %d] Failed %s: %s", worker_id, path.name, exc)
            else:
                stats["files"] += 1
                stats["text_docs"] += n_text
                stats["image_docs"] += n_img
                await checkpoint.record(
                    path.stem, status="ok", text_docs=n_text, image_docs=n_img
                )
                logger.info(
                    "[worker %d] Ingested %s: %d text, %d image docs",
                    worker_id,
                    path.name,
                    n_text,
                    n_img,
                )

    await asyncio.gather(*(_worker(p, i) for i, p in enumerate(pipelines)))
    await embedder.aclose()
    return stats


def _default_endpoints(cfg: AppConfig) -> list[str]:
    """Derive 3 document-intelligence-gpu endpoints from the configured
    base doc_intel service_url by incrementing its port — matches the
    8021/8022/8023 convention document-intelligence-gpu-0/1/2 use in
    docker-compose.yml.
    """
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(cfg.doc_intel.service_url)
    base_port = parts.port or 8021
    urls = []
    for i in range(3):
        netloc = f"{parts.hostname}:{base_port + i}"
        urls.append(urlunsplit((parts.scheme, netloc, parts.path, "", "")))
    return urls


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="GPU-parallel checkpointed PDF ingestion")
    parser.add_argument(
        "--dataset",
        help=(
            "Limit to one dataset dir under "
            "data/pdfQA-Benchmark/real-pdfQA/01.2_Input_Files_PDF/ "
            "(e.g. ClimRetrieve). Omit to ingest the full benchmark."
        ),
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint JSONL path (default: <collection>.checkpoint.jsonl)",
    )
    args = parser.parse_args()

    cfg = AppConfig()
    # settings.ROOT_DIR is an anyio.Path (async-native), not stdlib
    # pathlib.Path -- its glob() isn't directly sorted()-compatible the
    # way ingest_dataset_gpu()'s synchronous glob call needs. Force a
    # stdlib Path here, same fix already used for the extraction notebook.
    base_dir = Path(str(settings.ROOT_DIR)) / "data/pdfQA-Benchmark/real-pdfQA/01.2_Input_Files_PDF"
    dataset_dir = base_dir / args.dataset if args.dataset else base_dir
    checkpoint_path = Path(args.checkpoint or f"{args.collection}.checkpoint.jsonl")
    endpoints = _default_endpoints(cfg)

    logger.info("Dataset dir: %s", dataset_dir)
    logger.info("Endpoints: %s", endpoints)
    logger.info("Collection: %s", args.collection)

    stats = asyncio.run(
        ingest_dataset_gpu(
            dataset_dir,
            collection=args.collection,
            endpoints=endpoints,
            checkpoint_path=checkpoint_path,
            limit=args.limit,
        )
    )
    logger.info("Done: %s", stats)


if __name__ == "__main__":
    main()
