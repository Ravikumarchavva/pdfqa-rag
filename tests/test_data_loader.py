"""Integration tests for data/loader.py — reads actual annotation files."""

from __future__ import annotations

from pathlib import Path

import pytest

from pdfqa_rag.data.loader import load_annotations
from pdfqa_rag.data.models import LoadResult, QAPair, SourceChunk

ANNOTATIONS_DIR = Path(__file__).parent.parent / "data" / "pdfQA-Annotations"
SKIP_IF_NO_DATA = pytest.mark.skipif(
    not ANNOTATIONS_DIR.exists(),
    reason="pdfQA-Annotations submodule not initialised",
)


@SKIP_IF_NO_DATA
class TestLoadAnnotations:
    def test_returns_load_result(self):
        result = load_annotations(ANNOTATIONS_DIR)
        assert isinstance(result, LoadResult)

    def test_real_pdfqa_loaded_by_default(self):
        result = load_annotations(ANNOTATIONS_DIR)
        assert result.total_qa_pairs > 0
        assert all(qa.category == "real-pdfQA" for qa in result.qa_pairs)

    def test_source_chunks_are_deduplicated(self):
        result = load_annotations(ANNOTATIONS_DIR)
        keys = [(c.file_name, c.text) for c in result.source_chunks]
        assert len(keys) == len(set(keys)), "Dedup failed — duplicate (file, text) keys"

    def test_qa_pairs_have_questions(self):
        result = load_annotations(ANNOTATIONS_DIR)
        assert all(len(qa.question) > 0 for qa in result.qa_pairs)

    def test_dataset_filter(self):
        result = load_annotations(
            ANNOTATIONS_DIR,
            datasets=["ClimateFinanceBench"],
        )
        assert all(qa.dataset == "ClimateFinanceBench" for qa in result.qa_pairs)
        assert result.total_qa_pairs > 0

    def test_stats_match_data(self):
        result = load_annotations(ANNOTATIONS_DIR)
        total_from_stats = sum(s.num_qa_pairs for s in result.stats)
        assert total_from_stats == result.total_qa_pairs

    def test_missing_dir_raises(self):
        with pytest.raises(FileNotFoundError):
            load_annotations(Path("/nonexistent/path"))

    def test_source_texts_are_nonempty(self):
        result = load_annotations(ANNOTATIONS_DIR)
        for chunk in result.source_chunks:
            assert chunk.text.strip(), f"Empty source text in {chunk.file_name}"

    def test_finance_bench_present(self):
        result = load_annotations(ANNOTATIONS_DIR)
        datasets = {qa.dataset for qa in result.qa_pairs}
        assert len(datasets) >= 3, f"Expected ≥3 datasets, got {sorted(datasets)}"
