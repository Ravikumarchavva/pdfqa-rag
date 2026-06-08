"""Download source documents from the HuggingFace pdfQA-Benchmark dataset.

HuggingFace repo: ``pdfqa/pdfQA-Benchmark``

Path conventions inside the repo:
    Text (pre-extracted by PyMuPDF):
        real-pdfQA/01.1_Input_Files_Non_PDF/{dataset}/{file_name}__pymupdf.txt
    PDF (original):
        real-pdfQA/01.2_Input_Files_PDF/{dataset}/{file_name}.pdf

``DocumentDownloader`` resolves annotation ``file_name`` values to HF paths,
downloads to a local cache directory, and returns kernel ``Document`` objects
ready for ``RAGPipeline.ingest_documents()``.

Usage::

    downloader = DocumentDownloader(cache_dir=Path("data/documents"))
    docs = await downloader.load("ADOBE_2022_10K", dataset="FinanceBench")
    # docs is a list[Document] — one per page/chunk
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Literal
from pdfqa_rag.config import settings
from ravi.kernel.vector import Document

logger = logging.getLogger(__name__)

HF_REPO_ID = "pdfqa/pdfQA-Benchmark"
HF_REPO_TYPE = "dataset"

# HF path templates
_TXT_TEMPLATE = "real-pdfQA/01.1_Input_Files_Non_PDF/{dataset}/{file_name}__pymupdf.txt"
_PDF_TEMPLATE = "real-pdfQA/01.2_Input_Files_PDF/{dataset}/{file_name}.pdf"


class DocumentDownloader:
    """Download and cache documents from HuggingFace, then parse into Documents.

    Args:
        cache_dir: Local directory where downloaded files are stored.
            Structure: ``{cache_dir}/{dataset}/{file_name}.{ext}``
        format: ``"text"`` uses the pre-extracted PyMuPDF .txt files (fast,
            no GPU/OCR needed). ``"pdf"`` downloads the original PDFs and
            runs ``PDFLoader`` (slower, more accurate table/layout handling).
        hf_token: HuggingFace token — only needed if the repo is private.
    """

    def __init__(
        self,
        cache_dir: Path | str = settings.ROOT_DIR / 'data' / 'documents',
        format: Literal["text", "pdf"] = "text",
        hf_token: str | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._format = format
        self._hf_token = hf_token

    # ── Public API ─────────────────────────────────────────────────────────────

    async def load(
        self,
        file_name: str,
        dataset: str,
        *,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
    ) -> list[Document]:
        """Download (if not cached) and parse a single document.

        Args:
            file_name: The annotation ``file_name`` value (e.g. ``ADOBE_2022_10K``).
            dataset: Dataset name (e.g. ``FinanceBench``).
            chunk_size: Character chunk size for text format.
                Ignored for PDF format (which chunks per page).
            chunk_overlap: Character overlap between consecutive chunks.

        Returns:
            List of ``Document`` objects ready for embedding.
        """
        local_path = await self._ensure_cached(file_name, dataset)
        return self._parse(local_path, file_name=file_name, dataset=dataset,
                           chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    async def load_many(
        self,
        file_names: list[str],
        dataset: str,
        *,
        chunk_size: int = 1024,
        chunk_overlap: int = 128,
        concurrency: int = 4,
    ) -> dict[str, list[Document]]:
        """Download and parse multiple documents concurrently.

        Returns a dict mapping ``file_name`` → ``list[Document]``.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _one(name: str) -> tuple[str, list[Document]]:
            async with sem:
                docs = await self.load(name, dataset, chunk_size=chunk_size,
                                       chunk_overlap=chunk_overlap)
                return name, docs

        tasks = [asyncio.create_task(_one(n)) for n in file_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        out: dict[str, list[Document]] = {}
        for name, result in zip(file_names, results):
            if isinstance(result, Exception):
                logger.error("Failed to load %s: %s", name, result)
            else:
                _, docs = result
                out[name] = docs
                logger.info("Loaded %s: %d documents", name, len(docs))

        return out

    def is_cached(self, file_name: str, dataset: str) -> bool:
        return self._local_path(file_name, dataset).exists()

    # ── Internal ───────────────────────────────────────────────────────────────

    def _local_path(self, file_name: str, dataset: str) -> Path:
        ext = "txt" if self._format == "text" else "pdf"
        return self._cache_dir / dataset / f"{file_name}.{ext}"

    def _hf_path(self, file_name: str, dataset: str) -> str:
        template = _TXT_TEMPLATE if self._format == "text" else _PDF_TEMPLATE
        return template.format(dataset=dataset, file_name=file_name)

    async def _ensure_cached(self, file_name: str, dataset: str) -> Path:
        local = self._local_path(file_name, dataset)
        if local.exists():
            logger.debug("Cache hit: %s", local)
            return local

        local.parent.mkdir(parents=True, exist_ok=True)
        hf_path = self._hf_path(file_name, dataset)
        logger.info("Downloading %s from HuggingFace ...", hf_path)

        # Run blocking download in a thread so we don't block the event loop
        downloaded = await asyncio.to_thread(self._hf_download, hf_path)

        # Copy from HF cache to our structured cache dir
        import shutil
        shutil.copy2(downloaded, local)
        logger.info("Cached to %s (%d bytes)", local, local.stat().st_size)
        return local

    def _hf_download(self, hf_path: str) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(
            repo_id=HF_REPO_ID,
            repo_type=HF_REPO_TYPE,
            filename=hf_path,
            token=self._hf_token,
        )

    def _parse(
        self,
        path: Path,
        *,
        file_name: str,
        dataset: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[Document]:
        if self._format == "text":
            return self._parse_text(path, file_name=file_name, dataset=dataset,
                                    chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        else:
            return self._parse_pdf(path, file_name=file_name, dataset=dataset)

    def _parse_text(
        self,
        path: Path,
        *,
        file_name: str,
        dataset: str,
        chunk_size: int,
        chunk_overlap: int,
    ) -> list[Document]:
        """Split a pre-extracted .txt file into fixed-size character chunks."""
        from ravi.capabilities.knowledge.chunking import TextChunker

        text = path.read_text(encoding="utf-8", errors="replace")
        chunker = TextChunker(chunk_size=chunk_size, overlap=chunk_overlap)
        base_meta = {
            "file_name": file_name,
            "dataset": dataset,
            "source_format": "pymupdf_text",
            "source_path": str(path),
        }
        docs = chunker.chunk(text, metadata=base_meta)
        logger.info(
            "Parsed %s → %d chunks (chunk_size=%d)", path.name, len(docs), chunk_size
        )
        return docs

    def _parse_pdf(
        self, path: Path, *, file_name: str, dataset: str
    ) -> list[Document]:
        """Parse a PDF with pdfplumber — one Document per page."""
        import asyncio as _asyncio
        from ravi.capabilities.knowledge.loaders.pdf_loader import PDFLoader

        loader = PDFLoader(extract_tables=True)
        base_meta = {
            "file_name": file_name,
            "dataset": dataset,
            "source_format": "pdf",
            "source_path": str(path),
        }
        docs = _asyncio.run(loader.load(path, metadata=base_meta))
        logger.info("Parsed %s → %d pages", path.name, len(docs))
        return docs
