"""Ingest all FinanceBench PDFs into pgvector.

Usage:
    uv run python ingest.py                        # ingest
    uv run python ingest.py --clean                # wipe collection first, then ingest
    uv run python ingest.py --collection my_col    # custom collection name
    uv run python ingest.py --pdf-dir /other/path  # custom PDF directory
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import click


@click.command()
@click.option("--collection", default="financebench", show_default=True)
@click.option(
    "--pdf-dir",
    default="data/pdfQA-Benchmark/real-pdfQA/01.2_Input_Files_PDF/FinanceBench",
    show_default=True,
)
@click.option("--clean", is_flag=True, default=False, help="Delete existing rows before ingesting.")
@click.option("--embed-batch", default=None, type=int, help="Override embed batch size.")
@click.option("--parse-workers", default=None, type=int, help="Override number of parse workers.")
def main(collection: str, pdf_dir: str, clean: bool, embed_batch: int | None, parse_workers: int | None) -> None:
    asyncio.run(_run(
        collection=collection,
        pdf_dir=Path(pdf_dir),
        clean=clean,
        embed_batch_override=embed_batch,
        parse_workers_override=parse_workers,
    ))


async def _run(
    *,
    collection: str,
    pdf_dir: Path,
    clean: bool,
    embed_batch_override: int | None,
    parse_workers_override: int | None,
) -> None:
    import torch
    from tqdm import tqdm

    from agent_substratecapabilities.llm import SentenceTransformersEmbeddingClient
    from agent_substratekernel.vector import Document
    from pdfqa_rag.config import AppConfig
    from pdfqa_rag.pipeline.parse import parse_pdf_worker
    from pdfqa_rag.store.factory import build_vector_store

    device = "cuda" if torch.cuda.is_available() else "cpu"

    embed_batch = embed_batch_override or 64  # small = GPU starts immediately, no stalls
    n_workers = parse_workers_override or (os.cpu_count() or 4)
    queue_max = max(embed_batch * 8, 1024)   # always >> embed_batch so parsers never block

    cfg = AppConfig()
    embed_client = SentenceTransformersEmbeddingClient(batch_size=embed_batch, device=device)
    store = build_vector_store(cfg.store, dimensions=cfg.embed.dimensions)
    await store.ensure_table()

    if clean:
        deleted = await store.delete_collection(collection)
        click.echo(f"Cleared {deleted} existing rows from '{collection}'.")

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        raise click.ClickException(f"No PDFs found in {pdf_dir}")

    click.echo(
        f"device={device}  embed_batch={embed_batch}  "
        f"parse_workers={n_workers}  pdfs={len(pdf_files)}  collection={collection}"
    )

    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_max)
    loop = asyncio.get_event_loop()
    ctx = mp.get_context("spawn")

    async def producer(pbar):
        try:
            jobs = [
                loop.run_in_executor(
                    pool,
                    parse_pdf_worker,
                    (str(p), {"file_name": p.stem, "dataset": "FinanceBench"}),
                )
                for p in pdf_files
            ]
            for fut in asyncio.as_completed(jobs):
                try:
                    docs = await fut
                except Exception as exc:
                    click.echo(f"  parse failed: {exc}", err=True)
                    pbar.update(1)
                    continue
                for doc in docs:
                    await queue.put(doc)
                pbar.update(1)
        finally:
            await queue.put(None)

    async def consumer(pbar):
        batch: list[Document] = []
        total = 0

        async def flush():
            nonlocal total
            if not batch:
                return
            result = await embed_client.embed([d.text for d in batch])
            await store.add(batch, result.embeddings, collection=collection)
            pbar.update(len(batch))
            total += len(batch)
            batch.clear()

        while True:
            doc = await queue.get()
            if doc is None:
                break
            batch.append(doc)
            if len(batch) >= embed_batch:
                await flush()
        await flush()
        return total

    parse_bar  = tqdm(total=len(pdf_files), desc="Parsing (docling+RapidOCR)")
    ingest_bar = tqdm(desc="Embedded + stored", unit="chunk")

    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        prod = asyncio.create_task(producer(parse_bar))
        total = await consumer(ingest_bar)
        await prod

    parse_bar.close()
    ingest_bar.close()
    click.echo(f"\nDone — {total} pages embedded + stored in '{collection}'.")
    click.echo("Re-running is idempotent (deterministic page IDs + ON CONFLICT DO NOTHING).")


if __name__ == "__main__":
    main()
