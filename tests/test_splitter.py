"""Unit tests for data/splitter.py — no I/O, no network."""

from __future__ import annotations

import pytest

from pdfqa_rag.data.models import SourceChunk
from pdfqa_rag.data.splitter import dedup_chunks, extract_source_text


class TestExtractSourceText:
    # doc1{...} wrapper format (ClimateFinanceBench)
    def test_inline_text_extracted(self):
        raw = "doc1{Greenhouse gas emissions\nScope 1: 55,200 MT CO2e}"
        assert extract_source_text(raw) == "Greenhouse gas emissions\nScope 1: 55,200 MT CO2e"

    def test_doc_prefix_any_number(self):
        assert extract_source_text("doc42{some text}") == "some text"

    def test_empty_braces_returns_none(self):
        assert extract_source_text("doc1{}") is None

    def test_whitespace_only_inner_returns_none(self):
        assert extract_source_text("doc1{   }") is None

    def test_leading_trailing_whitespace_stripped_from_wrapper(self):
        result = extract_source_text("doc1{  hello world  }")
        assert result == "hello world"

    def test_multiline_inner_text_preserved(self):
        raw = "doc1{line1\nline2\nline3}"
        assert extract_source_text(raw) == "line1\nline2\nline3"

    # ID-only references (syn-pdfQA) → None
    def test_id_reference_returns_none(self):
        assert extract_source_text("Source_1042") is None

    def test_id_reference_various_prefixes(self):
        assert extract_source_text("Doc_999") is None
        assert extract_source_text("Ref_42") is None

    # Plain text sources (FinanceBench, FeTaQA, ClimRetrieve, …)
    def test_plain_text_returned_as_is(self):
        raw = "Net income\n$ 4,756\nDepreciation 856\nTotal 7,838"
        assert extract_source_text(raw) == raw

    def test_plain_text_with_markdown_table(self):
        raw = "| Year | Award | Result |\n| 2016 | Best R&B | Nominated |"
        assert extract_source_text(raw) == raw

    def test_empty_string_returns_none(self):
        assert extract_source_text("") is None

    def test_very_short_text_returns_none(self):
        assert extract_source_text("hi") is None


class TestDedupChunks:
    def _make(self, text: str, file_name: str = "f1", idx: int = 0) -> SourceChunk:
        return SourceChunk(
            text=text,
            file_name=file_name,
            dataset="DS",
            category="real-pdfQA",
            chunk_index=idx,
        )

    def test_no_duplicates_unchanged(self):
        chunks = [self._make("a"), self._make("b"), self._make("c")]
        assert dedup_chunks(chunks) == chunks

    def test_exact_duplicates_removed(self):
        c = self._make("hello")
        result = dedup_chunks([c, c, c])
        assert result == [c]

    def test_same_text_different_files_kept(self):
        c1 = self._make("hello", file_name="f1")
        c2 = self._make("hello", file_name="f2")
        assert dedup_chunks([c1, c2]) == [c1, c2]

    def test_first_occurrence_preserved(self):
        c1 = self._make("hello", idx=0)
        c2 = self._make("hello", idx=1)
        result = dedup_chunks([c1, c2])
        assert result == [c1]
        assert result[0].chunk_index == 0

    def test_empty_list(self):
        assert dedup_chunks([]) == []

    def test_order_preserved(self):
        chunks = [self._make(t) for t in ["z", "a", "m", "b"]]
        assert [c.text for c in dedup_chunks(chunks)] == ["z", "a", "m", "b"]
