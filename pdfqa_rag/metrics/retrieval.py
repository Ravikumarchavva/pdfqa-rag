"""Retrieval quality metrics for RAG evaluation.

All functions are pure — no I/O, no async.  The eval command calls these
after accumulating per-question results.

Metrics:
    recall_at_k  — fraction of QA pairs where ≥1 ground truth span is retrieved
    mrr          — mean reciprocal rank of the first hit
    per_dataset_breakdown — aggregate both metrics by dataset name
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Per-question evaluation result."""

    question: str
    file_name: str
    dataset: str
    ground_truth_texts: list[str]
    retrieved_texts: list[str]
    hit: bool
    rank: int | None
    """1-based rank of the first hit, or None if no hit."""


@dataclass
class AggregateMetrics:
    """Aggregated recall and MRR over a set of EvalResults."""

    num_questions: int = 0
    num_hits: int = 0
    reciprocal_rank_sum: float = 0.0
    dataset: str = "all"

    @property
    def recall(self) -> float:
        return self.num_hits / self.num_questions if self.num_questions else 0.0

    @property
    def mrr(self) -> float:
        return self.reciprocal_rank_sum / self.num_questions if self.num_questions else 0.0

    def __str__(self) -> str:
        return (
            f"[{self.dataset}] "
            f"Recall: {self.recall:.1%}  MRR: {self.mrr:.3f}  "
            f"({self.num_hits}/{self.num_questions})"
        )


def score_result(
    ground_truth: list[str],
    retrieved: list[str],
    *,
    match_fn: str = "substring",
) -> tuple[bool, int | None]:
    """Return (hit, rank) for a single question.

    Args:
        ground_truth: List of expected evidence texts.
        retrieved: List of retrieved texts (ordered by score, best first).
        match_fn: How to compare texts. ``"substring"`` = gt ⊆ retrieved or
            retrieved ⊆ gt.  ``"exact"`` = full string equality.

    Returns:
        ``(True, rank)`` if any ground-truth text matches any retrieved text,
        ``(False, None)`` otherwise.
    """
    if not ground_truth or not retrieved:
        return False, None

    _match = _substring_match if match_fn == "substring" else _exact_match

    for rank, r_text in enumerate(retrieved, start=1):
        for gt_text in ground_truth:
            if _match(gt_text, r_text):
                return True, rank
    return False, None


def recall_at_k(results: list[EvalResult], k: int) -> float:
    """Fraction of questions with a hit in the top-k retrieved texts."""
    if not results:
        return 0.0
    hits = sum(1 for r in results if r.hit and r.rank is not None and r.rank <= k)
    return hits / len(results)


def mrr(results: list[EvalResult]) -> float:
    """Mean Reciprocal Rank across all questions."""
    if not results:
        return 0.0
    total = sum(1.0 / r.rank for r in results if r.rank is not None)
    return total / len(results)


def per_dataset_breakdown(
    results: list[EvalResult],
) -> dict[str, AggregateMetrics]:
    """Group results by dataset and compute recall + MRR per group."""
    grouped: dict[str, list[EvalResult]] = defaultdict(list)
    for r in results:
        grouped[r.dataset].append(r)

    breakdown: dict[str, AggregateMetrics] = {}
    for dataset, ds_results in sorted(grouped.items()):
        metrics = AggregateMetrics(dataset=dataset)
        for r in ds_results:
            metrics.num_questions += 1
            if r.hit and r.rank is not None:
                metrics.num_hits += 1
                metrics.reciprocal_rank_sum += 1.0 / r.rank
        breakdown[dataset] = metrics

    return breakdown


def aggregate(results: list[EvalResult], label: str = "all") -> AggregateMetrics:
    """Compute aggregate recall + MRR over all results."""
    metrics = AggregateMetrics(dataset=label)
    for r in results:
        metrics.num_questions += 1
        if r.hit and r.rank is not None:
            metrics.num_hits += 1
            metrics.reciprocal_rank_sum += 1.0 / r.rank
    return metrics


# ── Match functions ────────────────────────────────────────────────────────────

def _substring_match(gt: str, retrieved: str) -> bool:
    """True if either text is a substring of the other (case-insensitive)."""
    gt_norm = gt.lower().strip()
    r_norm = retrieved.lower().strip()
    # Use a 50-char anchor from gt to avoid very short false positives
    anchor = gt_norm[:50]
    return anchor in r_norm or r_norm[:50] in gt_norm


def _exact_match(gt: str, retrieved: str) -> bool:
    return gt.strip() == retrieved.strip()
