"""CLI entrypoint — `python -m pdfqa_rag` or `pdfqa-rag`."""

from __future__ import annotations

import logging

import click

from pdfqa_rag.commands.download import download
from pdfqa_rag.commands.eval import eval as eval_cmd
from pdfqa_rag.commands.ingest import ingest
from pdfqa_rag.commands.ingest_docs import ingest_docs
from pdfqa_rag.commands.query import query


@click.group()
@click.option(
    "--log-level",
    default="INFO",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    show_default=True,
    help="Logging verbosity.",
)
def cli(log_level: str) -> None:
    """pdfqa-rag — modular RAG pipeline for the pdfQA benchmark."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )


cli.add_command(download)
cli.add_command(ingest)
cli.add_command(ingest_docs)
cli.add_command(query)
cli.add_command(eval_cmd, name="eval")


if __name__ == "__main__":
    cli()
