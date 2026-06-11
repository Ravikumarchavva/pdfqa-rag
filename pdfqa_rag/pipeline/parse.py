"""Process-pool worker for parsing PDFs with Docling.

Docling uses RapidOCR (PaddleOCR ONNX) for fast extraction and
HybridChunker for structure-aware chunking — chunks follow heading
boundaries rather than page breaks, so each chunk is semantically
coherent and embeddings carry document structure.

``parse_pdf_worker`` is module-level (picklable) so ProcessPoolExecutor
can fan it out across all CPU cores while the main process runs the GPU
embedder.
"""

from __future__ import annotations

import hashlib
import uuid
from typing import Any

from ravi.kernel.vector import Document


def _chunk_id(source: str, index: int, text: str) -> str:
    """Deterministic content-addressed chunk ID — same content always same UUID."""
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}|{index}|{digest}"))


def parse_pdf_worker(args: tuple[str, dict[str, Any]]) -> list[Document]:
    """Parse one PDF into semantically-chunked Documents. Runs in a worker process.

    Uses Docling's DocumentConverter (RapidOCR) for extraction and
    HybridChunker for structure-aware chunking. contextualize() prepends
    the heading hierarchy to each chunk so embeddings carry section context.

    Args:
        args: ``(pdf_path, metadata)`` tuple.

    Returns:
        List of ``Document`` objects with deterministic IDs, ready for embedding.
    """
    import logging
    import warnings

    warnings.filterwarnings("ignore")
    logging.getLogger("docling").setLevel(logging.ERROR)
    logging.getLogger("rapidocr").setLevel(logging.ERROR)
    logging.getLogger("RapidOCR").setLevel(logging.ERROR)

    pdf_path, metadata = args

    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker

    result = DocumentConverter().convert(pdf_path)
    doc = result.document
    chunker = HybridChunker()

    documents: list[Document] = []
    for i, chunk in enumerate(chunker.chunk(doc)):
        # contextualize prepends the heading hierarchy → richer embedding signal
        text = chunker.contextualize(chunk)
        if not text.strip():
            continue

        # Pull page number from first item's provenance if available
        page_no = None
        try:
            prov = chunk.meta.doc_items[0].prov
            if prov:
                page_no = prov[0].page_no
        except (AttributeError, IndexError):
            pass

        chunk_meta = {
            **metadata,
            "chunk_index": i,
            "headings": chunk.meta.headings or [],
        }
        if page_no is not None:
            chunk_meta["page_number"] = page_no

        documents.append(Document(
            text=text,
            metadata=chunk_meta,
            id=_chunk_id(pdf_path, i, chunk.text),  # stable ID on raw text, not contextualized
        ))

    return documents
