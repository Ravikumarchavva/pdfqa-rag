"""Annotation JSON loader — filesystem traversal + parsing.

Directory layout expected:
    <data_dir>/
        real-pdfQA/<dataset>/<file>.json
        syn-pdfQA/<dataset>/<file>.json

Each JSON file is a list of QA entries (see pdfQA-Annotations README).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from pdfqa_rag.data.models import DatasetStats, LoadResult, QAPair, SourceChunk
from pdfqa_rag.data.splitter import dedup_chunks, extract_source_text

logger = logging.getLogger(__name__)


def load_annotations(
    data_dir: Path | str,
    *,
    categories: list[str] | None = None,
    datasets: list[str] | None = None,
) -> LoadResult:
    """Load annotation JSONs from ``data_dir`` and return a ``LoadResult``.

    Args:
        data_dir: Root of the ``pdfQA-Annotations`` directory.
        categories: Which top-level categories to include.
            Defaults to ``["real-pdfQA"]``.  Pass ``None`` for all.
        datasets: Optional allowlist of dataset names (e.g.
            ``["ClimateFinanceBench", "FinanceBench"]``).  ``None`` = all.

    Returns:
        A ``LoadResult`` with qa_pairs, deduped source_chunks, and per-dataset stats.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {data_dir}")

    if categories is None:
        categories = ["real-pdfQA"]

    all_qa_pairs: list[QAPair] = []
    all_raw_chunks: list[SourceChunk] = []
    stats_list: list[DatasetStats] = []

    for category in categories:
        category_dir = data_dir / category
        if not category_dir.is_dir():
            logger.warning("Category directory not found, skipping: %s", category_dir)
            continue

        for dataset_dir in sorted(category_dir.iterdir()):
            if not dataset_dir.is_dir():
                continue
            dataset_name = dataset_dir.name
            if datasets and dataset_name not in datasets:
                continue

            qa_pairs, raw_chunks, stats = _load_dataset(
                dataset_dir, category=category, dataset=dataset_name
            )
            all_qa_pairs.extend(qa_pairs)
            all_raw_chunks.extend(raw_chunks)
            stats_list.append(stats)
            logger.info("%s", stats)

    deduped = dedup_chunks(all_raw_chunks)
    logger.info(
        "Total: %d QA pairs, %d raw chunks → %d after dedup",
        len(all_qa_pairs),
        len(all_raw_chunks),
        len(deduped),
    )
    return LoadResult(qa_pairs=all_qa_pairs, source_chunks=deduped, stats=stats_list)


def _load_dataset(
    dataset_dir: Path,
    *,
    category: str,
    dataset: str,
) -> tuple[list[QAPair], list[SourceChunk], DatasetStats]:
    stats = DatasetStats(category=category, dataset=dataset)
    qa_pairs: list[QAPair] = []
    raw_chunks: list[SourceChunk] = []

    for json_file in sorted(dataset_dir.glob("*.json")):
        stats.num_files += 1
        try:
            entries = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", json_file.name, exc)
            continue

        if not isinstance(entries, list):
            logger.warning("Expected list in %s, got %s", json_file.name, type(entries))
            continue

        for entry in entries:
            qa, chunks, skipped = _parse_entry(
                entry, category=category, dataset=dataset
            )
            if qa is not None:
                qa_pairs.append(qa)
                raw_chunks.extend(chunks)
                stats.num_qa_pairs += 1
                stats.num_source_chunks += len(chunks)
                stats.num_skipped_sources += skipped

    deduped_count = len(dedup_chunks(raw_chunks))
    stats.num_deduped_chunks = deduped_count
    return qa_pairs, raw_chunks, stats


def _parse_entry(
    entry: dict,
    *,
    category: str,
    dataset: str,
) -> tuple[QAPair | None, list[SourceChunk], int]:
    """Parse a single annotation entry. Returns (qa_pair, chunks, skipped_count)."""
    question = entry.get("question", "").strip()
    answer = entry.get("answer", "").strip()
    file_name = entry.get("file_name", "unknown")
    raw_sources: list[str] = entry.get("sources", [])

    if not question:
        return None, [], 0

    chunks: list[SourceChunk] = []
    skipped = 0

    for i, raw in enumerate(raw_sources):
        text = extract_source_text(str(raw))
        if text is None:
            skipped += 1
            continue
        chunks.append(
            SourceChunk(
                text=text,
                file_name=file_name,
                dataset=dataset,
                category=category,
                chunk_index=i,
            )
        )

    qa = QAPair(
        question=question,
        answer=answer,
        source_chunks=chunks,
        file_name=file_name,
        dataset=dataset,
        category=category,
        num_sources=len(raw_sources),
        num_sources_skipped=skipped,
        question_type=entry.get("question_type"),
        complexity=entry.get("complexity"),
        source_sampling_strategy=entry.get("source_sampling_strategy"),
    )
    return qa, chunks, skipped
