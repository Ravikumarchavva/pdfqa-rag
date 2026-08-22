"""Resumable, concurrent batch ingestion with progress tracking.

``BatchIngestor`` wraps ``RAGPipeline.ingest_documents`` with:
- tqdm progress bar
- configurable async concurrency (semaphore-limited batches)
- checkpoint file so large runs survive interruption

Checkpoint format: one JSON line per chunk, e.g.::

    {"id": "abc123", "file_name": "Apple_2024", "dataset": "ClimateFinanceBench"}

On ``--resume``, already-checkpointed chunk IDs are skipped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from pdfqa_rag.data.models import SourceChunk

if TYPE_CHECKING:
    from substrate.capabilities.knowledge.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class BatchIngestor:
    """Ingest ``SourceChunk`` lists through a ``RAGPipeline`` in concurrent batches.

    Args:
        pipeline: A ready-to-use ``RAGPipeline`` instance.
        collection: Target collection in the vector store.
        batch_size: Chunks per embed API call.
        concurrency: Maximum parallel embed calls (semaphore count).
        checkpoint_every: Write checkpoint every N completed chunks.
        checkpoint_path: Where to write the checkpoint file.
            Defaults to ``{collection}.checkpoint.jsonl`` in cwd.
    """

    def __init__(
        self,
        pipeline: RAGPipeline,
        *,
        collection: str,
        batch_size: int = 64,
        concurrency: int = 8,
        checkpoint_every: int = 500,
        checkpoint_path: Path | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._collection = collection
        self._batch_size = batch_size
        self._concurrency = concurrency
        self._checkpoint_every = checkpoint_every
        self._checkpoint_path = checkpoint_path or Path(f"{collection}.checkpoint.jsonl")

    # ── Public API ─────────────────────────────────────────────────────────────

    def load_checkpoint(self) -> set[str]:
        """Return the set of chunk IDs already written to the checkpoint file."""
        if not self._checkpoint_path.exists():
            return set()
        ids: set[str] = set()
        with self._checkpoint_path.open() as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["id"])
                except (KeyError, json.JSONDecodeError):
                    pass
        logger.info("Checkpoint: %d chunks already ingested", len(ids))
        return ids

    async def ingest(
        self,
        chunks: list[SourceChunk],
        *,
        resume: bool = False,
    ) -> int:
        """Embed and store all chunks.

        Args:
            chunks: Source chunks to ingest.
            resume: If True, load checkpoint and skip already-ingested chunks.

        Returns:
            Number of chunks newly ingested this run.
        """
        done_ids = self.load_checkpoint() if resume else set()

        # Assign stable IDs deterministically by (file_name, chunk_index, text-hash)
        # so the same chunk always gets the same ID across runs.
        pending = [
            (chunk, _stable_id(chunk))
            for chunk in chunks
            if _stable_id(chunk) not in done_ids
        ]

        if not pending:
            logger.info("All %d chunks already ingested (checkpoint is current)", len(chunks))
            return 0

        logger.info(
            "Ingesting %d/%d chunks (batch=%d, concurrency=%d)",
            len(pending),
            len(chunks),
            self._batch_size,
            self._concurrency,
        )

        sem = asyncio.Semaphore(self._concurrency)
        total_ingested = 0
        checkpoint_buf: list[dict] = []

        # Split into batches
        batches = [
            pending[i : i + self._batch_size]
            for i in range(0, len(pending), self._batch_size)
        ]

        with tqdm(total=len(pending), unit="chunk", desc="Ingesting") as bar:
            async def _ingest_batch(batch: list[tuple[SourceChunk, str]]) -> int:
                from substrate.kernel.storage.vector import Document

                async with sem:
                    documents = [
                        Document.from_text(
                            chunk.text,
                            metadata={
                                "file_name": chunk.file_name,
                                "dataset": chunk.dataset,
                                "category": chunk.category,
                                "chunk_index": chunk.chunk_index,
                            },
                            id=chunk_id,
                        )
                        for chunk, chunk_id in batch
                    ]
                    n = await self._pipeline.ingest_documents(
                        documents, collection=self._collection
                    )
                    return n, batch

            tasks = [asyncio.create_task(_ingest_batch(b)) for b in batches]
            for coro in asyncio.as_completed(tasks):
                n, batch = await coro
                total_ingested += n
                bar.update(n)

                # Accumulate checkpoint entries
                for chunk, chunk_id in batch:
                    checkpoint_buf.append({
                        "id": chunk_id,
                        "file_name": chunk.file_name,
                        "dataset": chunk.dataset,
                    })

                # Flush checkpoint periodically
                if len(checkpoint_buf) >= self._checkpoint_every:
                    self._flush_checkpoint(checkpoint_buf)
                    checkpoint_buf.clear()

        # Final flush
        if checkpoint_buf:
            self._flush_checkpoint(checkpoint_buf)

        logger.info("Ingested %d new chunks into collection '%s'", total_ingested, self._collection)
        return total_ingested

    # ── Internal ───────────────────────────────────────────────────────────────

    def _flush_checkpoint(self, entries: list[dict]) -> None:
        with self._checkpoint_path.open("a") as f:
            for entry in entries:
                f.write(json.dumps(entry) + "\n")
        logger.debug("Checkpoint flushed (%d entries)", len(entries))


def _stable_id(chunk: SourceChunk) -> str:
    """Deterministic chunk ID — stable across runs for the same content."""
    key = f"{chunk.file_name}:{chunk.chunk_index}:{chunk.text[:64]}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))
