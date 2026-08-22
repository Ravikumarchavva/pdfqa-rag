"""``pdfqa-rag ingest-docs`` — embed full downloaded documents into pgvector."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command(name="ingest-docs")
@click.option(
    "--dataset",
    default="FinanceBench",
    show_default=True,
    help="Dataset name whose cached documents to ingest.",
)
@click.option(
    "--file-name",
    default=None,
    help="Ingest a single document. Default: all cached documents for the dataset.",
)
@click.option(
    "--collection",
    default="pdfqa_docs",
    show_default=True,
    help="Vector store collection name (separate from annotation snippets).",
)
@click.option(
    "--format",
    "fmt",
    default="text",
    type=click.Choice(["text", "pdf"]),
    show_default=True,
    help="Format of cached files to load.",
)
@click.option(
    "--cache-dir",
    default="data/documents",
    show_default=True,
    help="Local cache directory containing downloaded files.",
)
@click.option(
    "--chunk-size",
    default=1024,
    show_default=True,
    help="Character chunk size (text format only).",
)
@click.option(
    "--chunk-overlap",
    default=128,
    show_default=True,
    help="Character overlap between chunks (text format only).",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip chunks already in the checkpoint file.",
)
def ingest_docs(
    dataset: str,
    file_name: str | None,
    collection: str,
    fmt: str,
    cache_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    resume: bool,
) -> None:
    """Embed full downloaded documents and store them in the vector store.

    Run ``pdfqa-rag download`` first to fetch the source files.

    Uses a separate collection (``pdfqa_docs``) from the annotation snippet
    KB (``pdfqa``) so you can compare retrieval quality between them.

    Example:

        pdfqa-rag ingest-docs --dataset FinanceBench --collection finance_full
    """
    asyncio.run(_ingest_docs(
        dataset=dataset, file_name=file_name, collection=collection,
        fmt=fmt, cache_dir=cache_dir, chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, resume=resume,
    ))


async def _ingest_docs(
    dataset: str,
    file_name: str | None,
    collection: str,
    fmt: str,
    cache_dir: str,
    chunk_size: int,
    chunk_overlap: int,
    resume: bool,
) -> None:
    from pdfqa_rag.config import AppConfig
    from pdfqa_rag.data.downloader import DocumentDownloader
    from pdfqa_rag.data.models import SourceChunk
    from pdfqa_rag.pipeline.batch import BatchIngestor
    from pdfqa_rag.pipeline.factory import build_pipeline
    from pdfqa_rag.storage.factory import build_blob_store

    cfg = AppConfig()
    cache_path = Path(cache_dir)
    blob_store = build_blob_store(cfg.storage)
    await blob_store.connect()
    downloader = DocumentDownloader(cache_dir=cache_path, format=fmt, blob_store=blob_store)

    # Determine which files to ingest
    if file_name:
        file_names = [file_name]
    else:
        ext = "txt" if fmt == "text" else "pdf"
        dataset_dir = cache_path / dataset
        if not dataset_dir.exists():
            raise click.ClickException(
                f"Cache directory not found: {dataset_dir}\n"
                "Run: pdfqa-rag download --dataset " + dataset
            )
        file_names = [p.stem for p in sorted(dataset_dir.glob(f"*.{ext}"))]
        if not file_names:
            raise click.ClickException(
                f"No {ext} files found in {dataset_dir}\n"
                "Run: pdfqa-rag download --dataset " + dataset
            )

    click.echo(f"Loading {len(file_names)} document(s) from cache ...")
    docs_map = await downloader.load_many(
        file_names,
        dataset=dataset,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        concurrency=cfg.pipeline.concurrency,
    )

    all_documents = [doc for docs in docs_map.values() for doc in docs]
    click.echo(f"Total chunks to ingest: {len(all_documents)}")

    if not all_documents:
        click.echo("Nothing to ingest.")
        return

    # Convert to SourceChunk-compatible format for BatchIngestor
    # BatchIngestor works with SourceChunk, but ingest_documents works with Document directly
    pipeline = await build_pipeline(cfg)

    ingestor = BatchIngestor(
        pipeline,
        collection=collection,
        batch_size=cfg.embed.batch_size,
        concurrency=cfg.pipeline.concurrency,
        checkpoint_every=cfg.pipeline.checkpoint_every,
        checkpoint_path=Path(f"{collection}.checkpoint.jsonl"),
    )

    # Build SourceChunk-like objects so BatchIngestor can handle resumption.
    # We use the Document objects directly via pipeline.ingest_documents instead.
    from substrate.kernel.storage.vector import Document as KernelDocument
    n = await pipeline.ingest_documents(all_documents, collection=collection)

    click.echo(f"\nDone. {n} chunks ingested into collection '{collection}'.")
    click.echo(
        f"\nEvaluate against ground truth:\n"
        f"  pdfqa-rag eval --collection {collection} --dataset {dataset} --k 5"
    )
