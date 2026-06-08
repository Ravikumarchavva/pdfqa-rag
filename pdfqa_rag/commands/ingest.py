"""``pdfqa-rag ingest`` — load annotations and embed source chunks into pgvector."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--category",
    default="real-pdfQA",
    show_default=True,
    help="Annotation category to load ('real-pdfQA' or 'syn-pdfQA').",
)
@click.option(
    "--dataset",
    default=None,
    help="Restrict to a single dataset (e.g. 'ClimateFinanceBench'). Default: all.",
)
@click.option(
    "--collection",
    default="pdfqa",
    show_default=True,
    help="Target collection name in the vector store.",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip chunks already recorded in the checkpoint file.",
)
@click.option(
    "--data-dir",
    default=None,
    help="Path to pdfQA-Annotations directory. Overrides PIPELINE_DATA_DIR env var.",
)
def ingest(
    category: str,
    dataset: str | None,
    collection: str,
    resume: bool,
    data_dir: str | None,
) -> None:
    """Embed annotation source chunks and store them in the vector store.

    Reads annotation JSONs from PIPELINE_DATA_DIR (or --data-dir), extracts
    inline source text, deduplicates, and ingests via the configured embedding
    server and pgvector store.

    Progress is written to ``<collection>.checkpoint.jsonl`` so interrupted
    runs can be resumed with ``--resume``.
    """
    asyncio.run(_ingest(category=category, dataset=dataset, collection=collection,
                        resume=resume, data_dir=data_dir))


async def _ingest(
    category: str,
    dataset: str | None,
    collection: str,
    resume: bool,
    data_dir: str | None,
) -> None:
    from pdfqa_rag.config import AppConfig
    from pdfqa_rag.data.loader import load_annotations
    from pdfqa_rag.pipeline.batch import BatchIngestor
    from pdfqa_rag.pipeline.factory import build_pipeline

    cfg = AppConfig()

    effective_data_dir = Path(data_dir or cfg.pipeline.data_dir)
    click.echo(f"Loading annotations from {effective_data_dir} ...")
    result = load_annotations(
        effective_data_dir,
        categories=[category],
        datasets=[dataset] if dataset else None,
    )
    click.echo(result.summary())

    if not result.source_chunks:
        click.echo("No inline source chunks found — nothing to ingest.")
        return

    click.echo(f"\nBuilding pipeline (embed={cfg.embed.model}, store={cfg.store.backend}) ...")
    pipeline = await build_pipeline(cfg)

    ingestor = BatchIngestor(
        pipeline,
        collection=collection,
        batch_size=cfg.embed.batch_size,
        concurrency=cfg.pipeline.concurrency,
        checkpoint_every=cfg.pipeline.checkpoint_every,
    )

    n = await ingestor.ingest(result.source_chunks, resume=resume)
    click.echo(f"\nDone. {n} chunks ingested into collection '{collection}'.")
