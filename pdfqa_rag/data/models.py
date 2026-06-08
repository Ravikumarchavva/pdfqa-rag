"""Pure data models — no I/O, no external dependencies."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SourceChunk:
    """A single evidence span extracted from an annotation source list.

    Immutable and hashable so it can be used in sets for deduplication.
    The ``chunk_index`` is its ordinal position within the file's source
    list — useful for exact-match evaluation against ground-truth sources.
    """

    text: str
    file_name: str
    dataset: str
    category: str
    chunk_index: int


@dataclass
class QAPair:
    """One question/answer pair with its ground-truth source chunks.

    ``source_chunks`` contains only chunks whose text could be extracted
    (real-pdfQA inline text).  ID-only references (syn-pdfQA) are skipped
    and reflected in ``num_sources_skipped``.
    """

    question: str
    answer: str
    source_chunks: list[SourceChunk]
    file_name: str
    dataset: str
    category: str
    num_sources: int
    num_sources_skipped: int = 0
    question_type: str | None = None
    complexity: str | None = None
    source_sampling_strategy: str | None = None

    @property
    def has_inline_sources(self) -> bool:
        return len(self.source_chunks) > 0

    @property
    def source_texts(self) -> list[str]:
        return [c.text for c in self.source_chunks]


@dataclass
class DatasetStats:
    """Aggregate statistics for a loaded dataset."""

    category: str
    dataset: str
    num_files: int = 0
    num_qa_pairs: int = 0
    num_source_chunks: int = 0
    num_deduped_chunks: int = 0
    num_skipped_sources: int = 0

    def __str__(self) -> str:
        return (
            f"{self.category}/{self.dataset}: "
            f"{self.num_qa_pairs} QA pairs, "
            f"{self.num_deduped_chunks} unique chunks "
            f"({self.num_skipped_sources} ID-only sources skipped)"
        )


@dataclass
class LoadResult:
    """Return type of load_annotations() — QA pairs + deduped source chunks + stats."""

    qa_pairs: list[QAPair] = field(default_factory=list)
    source_chunks: list[SourceChunk] = field(default_factory=list)
    stats: list[DatasetStats] = field(default_factory=list)

    @property
    def total_qa_pairs(self) -> int:
        return len(self.qa_pairs)

    @property
    def total_source_chunks(self) -> int:
        return len(self.source_chunks)

    def summary(self) -> str:
        lines = [str(s) for s in self.stats]
        lines.append(
            f"Total: {self.total_qa_pairs} QA pairs, "
            f"{self.total_source_chunks} unique source chunks"
        )
        return "\n".join(lines)
