"""Text extraction and deduplication helpers for annotation source lists.

Source formats vary by dataset:

- ``doc1{text}``   — ClimateFinanceBench: inline text wrapped in a doc-ref tag
- Plain text        — FinanceBench, FeTaQA, ClimRetrieve, PaperText, Tat-QA, …
- ``Source_NNN``   — syn-pdfQA only: opaque ID referencing the full PDF; skip

``extract_source_text`` normalises both inline-text formats to plain strings,
and returns ``None`` for ID-only references so the loader can skip them.
"""

from __future__ import annotations

import re

from pdfqa_rag.data.models import SourceChunk

_INLINE_RE = re.compile(r"^doc\d+\{(.*)\}$", re.DOTALL)
_ID_ONLY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*_\d+$")
_MIN_SOURCE_LEN = 10  # shorter strings that don't match any pattern are skipped


def extract_source_text(raw: str) -> str | None:
    """Return usable plain text from a source entry, or ``None`` to skip.

    Handles three cases:
    1. ``doc1{...}`` wrapper (ClimateFinanceBench) — extract inner text.
    2. Plain multi-line text (FinanceBench, FeTaQA, …) — return as-is.
    3. ID reference like ``Source_1042`` (syn-pdfQA) — return ``None``.
    """
    raw = raw.strip()
    if not raw:
        return None

    # Case 1: doc-ref wrapper
    m = _INLINE_RE.match(raw)
    if m:
        text = m.group(1).strip()
        return text if text else None

    # Case 3: opaque source ID (syn-pdfQA) — single-token, no whitespace, ID pattern
    if "\n" not in raw and " " not in raw and _ID_ONLY_RE.match(raw):
        return None

    # Case 2: plain text — keep if long enough to be meaningful
    return raw if len(raw) >= _MIN_SOURCE_LEN else None


def dedup_chunks(chunks: list[SourceChunk]) -> list[SourceChunk]:
    """Remove duplicate source chunks by ``(file_name, text)`` key.

    The same evidence span often appears in multiple QA pairs for the
    same document.  Deduplication prevents re-embedding identical text
    and inflating the retrieval index.

    Order is preserved (first occurrence wins).
    """
    seen: set[tuple[str, str]] = set()
    result: list[SourceChunk] = []
    for chunk in chunks:
        key = (chunk.file_name, chunk.text)
        if key not in seen:
            seen.add(key)
            result.append(chunk)
    return result
