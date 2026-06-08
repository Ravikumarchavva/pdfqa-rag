"""``pdfqa-rag query`` — semantic search + optional generation."""

from __future__ import annotations

import asyncio
import logging

import click

logger = logging.getLogger(__name__)


@click.command()
@click.argument("question")
@click.option(
    "--collection",
    default="pdfqa",
    show_default=True,
    help="Vector store collection to search.",
)
@click.option(
    "--limit", "-k",
    default=5,
    show_default=True,
    help="Number of chunks to retrieve.",
)
@click.option(
    "--generate",
    is_flag=True,
    default=False,
    help="Generate an LLM answer from the retrieved context.",
)
@click.option(
    "--dataset",
    default=None,
    help="Filter retrieval to a specific dataset (metadata filter).",
)
def query(
    question: str,
    collection: str,
    limit: int,
    generate: bool,
    dataset: str | None,
) -> None:
    """Retrieve the top-K source chunks most relevant to QUESTION.

    With --generate, also produces an LLM answer grounded in the retrieved context.

    Example:

        pdfqa-rag query "What are Apple's Scope 1 emissions in FY2023?" --generate
    """
    asyncio.run(_query(question=question, collection=collection, limit=limit,
                       generate=generate, dataset=dataset))


async def _query(
    question: str,
    collection: str,
    limit: int,
    generate: bool,
    dataset: str | None,
) -> None:
    from pdfqa_rag.config import AppConfig
    from pdfqa_rag.pipeline.factory import build_llm_client, build_pipeline

    cfg = AppConfig()
    pipeline = await build_pipeline(cfg)

    metadata_filter = {"dataset": dataset} if dataset else None

    results = await pipeline.query(
        question,
        collection=collection,
        limit=limit,
        filter=metadata_filter,
    )

    if not results:
        click.echo("No results found.")
        return

    click.echo(f"\nTop-{limit} results for: {question!r}\n")
    for i, r in enumerate(results, 1):
        meta = r.metadata or {}
        source = meta.get("file_name", "?")
        ds = meta.get("dataset", "?")
        score = f"{r.score:.4f}" if r.score is not None else "N/A"
        preview = r.text[:200].replace("\n", " ").strip()
        click.echo(f"[{i}] score={score}  file={source}  dataset={ds}")
        click.echo(f"     {preview}")
        if len(r.text) > 200:
            click.echo(f"     ... ({len(r.text)} chars total)")
        click.echo()

    if generate:
        click.echo("─" * 60)
        click.echo("Generating answer ...\n")
        llm = build_llm_client(cfg.llm)
        answer = await pipeline.query_with_context(
            question,
            collection=collection,
            model_client=llm,
            limit=limit,
            filter=metadata_filter,
        )
        click.echo(f"Answer:\n{answer}")
