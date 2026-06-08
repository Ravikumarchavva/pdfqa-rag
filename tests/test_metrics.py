"""Unit tests for metrics/retrieval.py — pure functions, no I/O."""

from __future__ import annotations

import pytest

from pdfqa_rag.metrics.retrieval import (
    EvalResult,
    aggregate,
    mrr,
    per_dataset_breakdown,
    recall_at_k,
    score_result,
)


def _result(hit: bool, rank: int | None, dataset: str = "DS") -> EvalResult:
    return EvalResult(
        question="q",
        file_name="f",
        dataset=dataset,
        ground_truth_texts=["gt"],
        retrieved_texts=["r"],
        hit=hit,
        rank=rank,
    )


class TestScoreResult:
    def test_exact_match_hit(self):
        gt = ["Scope 1: 55,200 metric tons CO2e"]
        retrieved = ["Scope 1: 55,200 metric tons CO2e"]
        hit, rank = score_result(gt, retrieved, match_fn="exact")
        assert hit is True
        assert rank == 1

    def test_no_match(self):
        hit, rank = score_result(["foo bar baz qux"], ["completely different text here"], match_fn="exact")
        assert hit is False
        assert rank is None

    def test_substring_match_gt_in_retrieved(self):
        gt = ["Scope 1 emissions"]
        retrieved = ["Total Scope 1 emissions were 55,200 MT CO2e in FY2023"]
        hit, rank = score_result(gt, retrieved, match_fn="substring")
        assert hit is True
        assert rank == 1

    def test_rank_reflects_position(self):
        gt = ["target text in document"]
        retrieved = ["irrelevant chunk", "another irrelevant", "target text in document"]
        hit, rank = score_result(gt, retrieved, match_fn="substring")
        assert hit is True
        assert rank == 3

    def test_empty_ground_truth(self):
        hit, rank = score_result([], ["some text"], match_fn="substring")
        assert hit is False
        assert rank is None

    def test_empty_retrieved(self):
        hit, rank = score_result(["some text"], [], match_fn="substring")
        assert hit is False
        assert rank is None


class TestRecallAtK:
    def test_all_hits(self):
        results = [_result(True, 1), _result(True, 2), _result(True, 3)]
        assert recall_at_k(results, k=5) == 1.0

    def test_no_hits(self):
        results = [_result(False, None), _result(False, None)]
        assert recall_at_k(results, k=5) == 0.0

    def test_half_hits(self):
        results = [_result(True, 1), _result(False, None)]
        assert recall_at_k(results, k=5) == 0.5

    def test_rank_cutoff_respected(self):
        # Hit at rank 6 shouldn't count for k=5
        results = [_result(True, 6)]
        assert recall_at_k(results, k=5) == 0.0

    def test_empty_list(self):
        assert recall_at_k([], k=5) == 0.0


class TestMRR:
    def test_first_rank_mrr(self):
        results = [_result(True, 1), _result(True, 1)]
        assert mrr(results) == 1.0

    def test_rank_2_mrr(self):
        results = [_result(True, 2)]
        assert mrr(results) == pytest.approx(0.5)

    def test_mixed_mrr(self):
        # rank 1 → 1.0, rank 2 → 0.5 → avg = 0.75
        results = [_result(True, 1), _result(True, 2)]
        assert mrr(results) == pytest.approx(0.75)

    def test_no_hits_mrr_zero(self):
        results = [_result(False, None), _result(False, None)]
        assert mrr(results) == 0.0

    def test_empty_list(self):
        assert mrr([]) == 0.0


class TestPerDatasetBreakdown:
    def test_groups_by_dataset(self):
        results = [
            _result(True, 1, dataset="A"),
            _result(False, None, dataset="A"),
            _result(True, 1, dataset="B"),
        ]
        breakdown = per_dataset_breakdown(results)
        assert set(breakdown.keys()) == {"A", "B"}
        assert breakdown["A"].num_questions == 2
        assert breakdown["B"].num_questions == 1

    def test_recall_per_dataset(self):
        results = [
            _result(True, 1, dataset="A"),
            _result(False, None, dataset="A"),
        ]
        breakdown = per_dataset_breakdown(results)
        assert breakdown["A"].recall == 0.5


class TestAggregate:
    def test_aggregate_label(self):
        results = [_result(True, 1)]
        m = aggregate(results, label="test")
        assert m.dataset == "test"
        assert m.recall == 1.0
        assert m.mrr == 1.0
