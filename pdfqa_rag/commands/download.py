"""``pdfqa-rag download`` — fetch source documents from HuggingFace."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--dataset",
    default="FinanceBench",
    show_default=True,
    help="Dataset name to download documents for.",
)
@click.option(
    "--file-name",
    default=None,
    help="Download a single document by file_name (e.g. ADOBE_2022_10K). "
         "Default: all documents referenced in the annotations.",
)
@click.option(
    "--format",
    "fmt",
    default="text",
    type=click.Choice(["text", "pdf"]),
    show_default=True,
    help="'text' = pre-extracted PyMuPDF .txt (fast). 'pdf' = original PDF.",
)
@click.option(
    "--cache-dir",
    default="data/documents",
    show_default=True,
    help="Local directory to cache downloaded files.",
)
@click.option(
    "--data-dir",
    default="data/pdfQA-Annotations",
    show_default=True,
    help="Path to pdfQA-Annotations directory.",
)
@click.option(
    "--concurrency",
    default=4,
    show_default=True,
    help="Parallel downloads.",
)
@click.option(
    "--hf-token",
    default=None,
    envvar="HF_TOKEN",
    help="HuggingFace token (if needed). Also reads HF_TOKEN env var.",
)
def download(
    dataset: str,
    file_name: str | None,
    fmt: str,
    cache_dir: str,
    data_dir: str,
    concurrency: int,
    hf_token: str | None,
) -> None:
    """Download source documents from HuggingFace pdfQA-Benchmark.

    By default downloads all documents referenced in the annotation JSONs
    for the chosen dataset.  Use --file-name to fetch a single document.

    Example — download all FinanceBench text files:

        pdfqa-rag download --dataset FinanceBench --format text

    Download a single document:

        pdfqa-rag download --dataset FinanceBench --file-name ADOBE_2022_10K
    """
    asyncio.run(_download(
        dataset=dataset, file_name=file_name, fmt=fmt,
        cache_dir=cache_dir, data_dir=data_dir,
        concurrency=concurrency, hf_token=hf_token,
    ))


async def _download(
    dataset: str,
    file_name: str | None,
    fmt: str,
    cache_dir: str,
    data_dir: str,
    concurrency: int,
    hf_token: str | None,
) -> None:
    from pdfqa_rag.config import AppConfig
    from pdfqa_rag.data.downloader import DocumentDownloader
    from pdfqa_rag.data.loader import load_annotations
    from pdfqa_rag.storage.factory import build_blob_store

    blob_store = build_blob_store(AppConfig().storage)
    await blob_store.connect()

    downloader = DocumentDownloader(
        cache_dir=Path(cache_dir),
        format=fmt,
        hf_token=hf_token,
        blob_store=blob_store,
    )

    if file_name:
        file_names = [file_name]
    else:
        # Collect all file_names referenced in this dataset's annotations
        result = load_annotations(Path(data_dir), datasets=[dataset])
        file_names = sorted({qa.file_name for qa in result.qa_pairs})
        click.echo(f"Found {len(file_names)} documents in {dataset} annotations:")
        for n in file_names:
            cached = " [cached]" if downloader.is_cached(n, dataset) else ""
            click.echo(f"  {n}{cached}")
        click.echo()

    click.echo(
        f"Downloading {len(file_names)} document(s) "
        f"[format={fmt}, cache={cache_dir}] ..."
    )

    docs_map = await downloader.load_many(
        file_names,
        dataset=dataset,
        concurrency=concurrency,
    )

    click.echo(f"\nDone. {len(docs_map)}/{len(file_names)} documents loaded:")
    total_chunks = 0
    for name, docs in sorted(docs_map.items()):
        click.echo(f"  {name}: {len(docs)} chunks")
        total_chunks += len(docs)
    click.echo(f"\nTotal: {total_chunks} chunks ready for ingestion.")
    click.echo(
        "\nNext step:\n"
        "  pdfqa-rag ingest-docs --dataset FinanceBench --cache-dir data/documents"
    )
