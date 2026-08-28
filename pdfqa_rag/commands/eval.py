"""``pdfqa-rag eval`` — offline retrieval quality evaluation."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import click

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--collection",
    default="pdfqa",
    show_default=True,
    help="Vector store collection to evaluate against.",
)
@click.option(
    "--k",
    default=5,
    show_default=True,
    help="Recall@k cutoff.",
)
@click.option(
    "--category",
    default="real-pdfQA",
    show_default=True,
    help="Annotation category to evaluate.",
)
@click.option(
    "--dataset",
    default=None,
    help="Restrict evaluation to one dataset (e.g. 'ClimateFinanceBench').",
)
@click.option(
    "--output",
    default=None,
    help="Path to write per-question JSON results (optional).",
)
@click.option(
    "--data-dir",
    default=None,
    help="Path to pdfQA-Annotations directory. Overrides PIPELINE_DATA_DIR.",
)
@click.option(
    "--match",
    default="substring",
    type=click.Choice(["substring", "exact"]),
    show_default=True,
    help="Text matching strategy for hit detection.",
)
def eval(
    collection: str,
    k: int,
    category: str,
    dataset: str | None,
    output: str | None,
    data_dir: str | None,
    match: str,
) -> None:
    """Evaluate retrieval quality against ground-truth annotation sources.

    For each QA pair with inline source text, retrieves the top-k chunks
    and checks whether any ground-truth source appears in the results.

    Reports Recall@k and MRR, broken down per dataset.

    Example:

        pdfqa-rag eval --k 5 --dataset ClimateFinanceBench --output results.json
    """
    asyncio.run(_eval(collection=collection, k=k, category=category,
                      dataset=dataset, output=output, data_dir=data_dir,
                      match_fn=match))


async def _eval(
    collection: str,
    k: int,
    category: str,
    dataset: str | None,
    output: str | None,
    data_dir: str | None,
    match_fn: str,
) -> None:
    from pdfqa_rag.config import AppConfig
    from pdfqa_rag.data.loader import load_annotations
    from pdfqa_rag.metrics.retrieval import (
        EvalResult,
        aggregate,
        mrr,
        per_dataset_breakdown,
        recall_at_k,
        score_result,
    )
    from pdfqa_rag.pipeline.factory import build_pipeline

    cfg = AppConfig()
    effective_data_dir = Path(data_dir or cfg.pipeline.data_dir)

    click.echo(f"Loading annotations from {effective_data_dir} ...")
    result = load_annotations(
        effective_data_dir,
        categories=[category],
        datasets=[dataset] if dataset else None,
    )

    evaluable = [qa for qa in result.qa_pairs if qa.has_inline_sources]
    click.echo(
        f"{len(evaluable)}/{len(result.qa_pairs)} QA pairs have inline sources "
        f"(evaluable at k={k})\n"
    )

    if not evaluable:
        click.echo("Nothing to evaluate.")
        return

    pipeline = await build_pipeline(cfg)

    eval_results: list[EvalResult] = []
    with click.progressbar(evaluable, label="Evaluating") as bar:
        for qa in bar:
            retrieved = await pipeline.query(
                qa.question,
                collection=collection,
                limit=k,
            )
            retrieved_texts = [r.to_text() for r in retrieved]
            hit, rank = score_result(qa.source_texts, retrieved_texts, match_fn=match_fn)
            eval_results.append(
                EvalResult(
                    question=qa.question,
                    file_name=qa.file_name,
                    dataset=qa.dataset,
                    ground_truth_texts=qa.source_texts,
                    retrieved_texts=retrieved_texts,
                    hit=hit,
                    rank=rank,
                )
            )

    # ── Print results ──────────────────────────────────────────────────────────
    click.echo("\n" + "═" * 60)
    click.echo(f"  Evaluation: {category} | match={match_fn} | k={k}")
    click.echo("═" * 60)

    breakdown = per_dataset_breakdown(eval_results)
    for ds_metrics in breakdown.values():
        click.echo(f"  {ds_metrics}")

    click.echo("─" * 60)
    overall = aggregate(eval_results)
    click.echo(
        f"  Overall — Recall@{k}: {overall.recall:.1%}  MRR: {overall.mrr:.3f}  "
        f"({overall.num_hits}/{overall.num_questions})"
    )
    click.echo("═" * 60)

    # ── Optional JSON output ───────────────────────────────────────────────────
    if output:
        payload = {
            "config": {
                "collection": collection,
                "k": k,
                "category": category,
                "dataset": dataset,
                "match_fn": match_fn,
            },
            "overall": {
                "recall_at_k": overall.recall,
                "mrr": overall.mrr,
                "num_hits": overall.num_hits,
                "num_questions": overall.num_questions,
            },
            "per_dataset": {
                ds: {
                    "recall": m.recall,
                    "mrr": m.mrr,
                    "num_hits": m.num_hits,
                    "num_questions": m.num_questions,
                }
                for ds, m in breakdown.items()
            },
            "results": [
                {
                    "question": r.question,
                    "file_name": r.file_name,
                    "dataset": r.dataset,
                    "hit": r.hit,
                    "rank": r.rank,
                }
                for r in eval_results
            ],
        }
        Path(output).write_text(json.dumps(payload, indent=2))
        click.echo(f"\nResults written to {output}")
