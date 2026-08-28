"""Checkpointed, GPU-parallel PDF ingestion driver for the multimodal RAG
evaluation run — a two-stage producer/consumer pipeline across N
document-intelligence-gpu endpoints.

Real, found-not-assumed reason for a staged design instead of one worker
doing extract-then-process per file (the original shape): each worker's
timeline was ``[extract -> GPU busy] then [S3 uploads + embed + Postgres
inserts -> GPU idle]`` — the GPU replica sat unused for most of every
file's processing time. Splitting extraction (stage 1, GPU-bound, one
worker per replica) from chunk/embed/store (stage 2, network/DB-bound, a
separate worker pool) lets stage 1 keep pulling the next batch immediately
instead of waiting on stage 2's tail.

Also real, found-not-assumed: a single document-intelligence-gpu instance
appears to serialize extraction requests internally (concurrent requests to
it made even small files time out; sequential succeeded every time) — so
stage-1 concurrency per replica must stay 1; parallelism comes from N
independent replicas, not from any one of them handling concurrent
requests.

Distinct from ``DocumentIngestPipeline.ingest_dataset`` (single endpoint,
``concurrency=1`` by design — the per-instance limitation above) and
``pdfqa_rag.pipeline.batch.BatchIngestor`` (chunk-level checkpoint, a
different ingestion path entirely, coupled to the old text-only
``RAGPipeline``). This composes ``DocumentIngestPipeline.extract_files``/
``process_extracted`` directly with its own file-level checkpoint,
page-budget batching, and staged dispatch.

Usage::

    uv run python -m pdfqa_rag.pipeline.dataset_ingest_gpu \\
        --dataset ClimRetrieve --collection pdfqa-pilot

    # Full benchmark, resumable if interrupted — rerun with the same
    # --collection and it picks up where it left off via the checkpoint.
    uv run python -m pdfqa_rag.pipeline.dataset_ingest_gpu \\
        --collection pdfqa-benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from pdfqa_rag.config import AppConfig, DocumentIntelligenceConfig, settings
from pdfqa_rag.pipeline.factory import (
    build_blob_store,
    build_extraction_client,
    build_multimodal_embedder,
)
from pdfqa_rag.store.factory import build_vector_store

logger = logging.getLogger(__name__)

# Qwen3-VL-Embedding-2B — must match the dimensions the embed sidecar
# actually produces (verified this session: 2048-dim vectors).
_EMBEDDING_DIMENSIONS = 2048


class Checkpoint:
    """Append-only JSONL checkpoint keyed by PDF stem (== QAPair.file_name,
    per pdfqa_rag/data/loader.py — the join key back to ground truth).

    Doubles as this run's telemetry: ``pages``/``extract_ms``/
    ``postprocess_ms``/``stage`` are free (the checkpoint write already
    happens once per file) and are what makes later throughput changes
    measurable instead of guessed.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = asyncio.Lock()
        self.done: set[str] = self._load()

    def _load(self) -> set[str]:
        if not self._path.exists():
            return set()
        done: set[str] = set()
        with self._path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["file_name"])
                except (KeyError, json.JSONDecodeError):
                    pass
        logger.info("Checkpoint %s: %d files already ingested", self._path, len(done))
        return done

    async def record(
        self,
        file_name: str,
        *,
        status: str,
        text_docs: int = 0,
        image_docs: int = 0,
        error: str | None = None,
        pages: int | None = None,
        stage: str | None = None,
        extract_ms: float | None = None,
        postprocess_ms: float | None = None,
    ) -> None:
        entry = {
            "file_name": file_name,
            "status": status,
            "text_docs": text_docs,
            "image_docs": image_docs,
            "error": error,
            "pages": pages,
            "stage": stage,
            "extract_ms": extract_ms,
            "postprocess_ms": postprocess_ms,
        }
        async with self._lock:
            with self._path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
            self.done.add(file_name)


def _annotated_stems(annotations_dir: Path) -> set[str]:
    """Only the files with real ground truth are worth ingesting for this
    eval — ``dataset_dir`` (data/pdfQA-Benchmark) holds ~4300 PDFs total,
    but only ~514 have a matching annotation JSON (QAPair.file_name is the
    join key, == the PDF stem). Ingesting the rest would burn hours of GPU
    time on files nothing can ever be scored against.
    """
    from pdfqa_rag.data.loader import load_annotations

    result = load_annotations(annotations_dir)
    return {qa.file_name for qa in result.qa_pairs}


# ── Instrumentation ───────────────────────────────────────────────────────


async def _monitor_event_loop_lag(interval_s: float = 0.25, report_every_s: float = 60.0):
    """Background task: logs event-loop scheduling lag every ``report_every_s``.

    The diagnostic for the shared-event-loop design this driver runs under
    (many workers, one loop) — a sync call anywhere (file I/O, base64,
    JSON parsing of a large response) blocks every other worker for exactly
    as long as it takes, and this is what makes that visible instead of
    just "throughput feels low." Target: p99 under ~50ms. Cancelled cleanly
    when the run's main task group finishes.
    """
    samples: list[float] = []
    last_report = time.monotonic()
    try:
        while True:
            t0 = time.monotonic()
            await asyncio.sleep(interval_s)
            lag = time.monotonic() - t0 - interval_s
            samples.append(max(0.0, lag))
            now = time.monotonic()
            if now - last_report >= report_every_s and samples:
                samples.sort()
                p50 = samples[len(samples) // 2]
                p99 = samples[int(len(samples) * 0.99)]
                logger.info(
                    "Event-loop lag: p50=%.1fms p99=%.1fms (n=%d)",
                    p50 * 1000,
                    p99 * 1000,
                    len(samples),
                )
                samples.clear()
                last_report = now
    except asyncio.CancelledError:
        pass


async def _monitor_gpu_utilization(interval_s: float = 5.0):
    """Background task: logs mean per-GPU utilization every ``interval_s``.

    Best-effort — silently stops (logs once) if ``nvidia-smi`` isn't
    available, e.g. the driver isn't running on the GPU host itself. Mean
    utilization across the run is the headline number for whether the
    staged design actually kept the replicas busy (target: >80%).
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError("nvidia-smi not usable")
    except (FileNotFoundError, RuntimeError):
        logger.info("GPU utilization monitor disabled (nvidia-smi not available here)")
        return

    samples: dict[str, list[int]] = {}
    try:
        while True:
            await asyncio.sleep(interval_s)
            try:
                proc = await asyncio.create_subprocess_exec(
                    "nvidia-smi",
                    "--query-gpu=index,utilization.gpu",
                    "--format=csv,noheader,nounits",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                stdout, _ = await proc.communicate()
                for line in stdout.decode().strip().splitlines():
                    idx, util = (p.strip() for p in line.split(","))
                    samples.setdefault(idx, []).append(int(util))
            except Exception as exc:
                logger.debug("GPU utilization sample failed: %s", exc)
    except asyncio.CancelledError:
        if samples:
            means = {idx: sum(v) / len(v) for idx, v in samples.items()}
            logger.info(
                "Mean GPU utilization this run: %s",
                ", ".join(f"gpu{idx}={mean:.0f}%" for idx, mean in sorted(means.items())),
            )


# ── Scheduling ────────────────────────────────────────────────────────────


async def _page_counts(paths: list[Path], *, concurrency: int = 16) -> dict[Path, int]:
    """``{path: page_count}`` via pypdf, read in threads (pypdf only parses
    the xref table for this, not page content — cheap even for hundreds of
    files). Never fails scheduling on a broken PDF: falls back to an
    estimate from file size, since a genuinely corrupt file will fail in
    extraction anyway and get checkpointed as failed there."""
    import pypdf

    sem = asyncio.Semaphore(concurrency)

    def _count(p: Path) -> int:
        try:
            return len(pypdf.PdfReader(str(p), strict=False).pages)
        except Exception:
            return max(1, p.stat().st_size // 40_000)

    async def _one(p: Path) -> tuple[Path, int]:
        async with sem:
            return p, await asyncio.to_thread(_count, p)

    results = await asyncio.gather(*(_one(p) for p in paths))
    return dict(results)


async def _load_or_compute_page_counts(
    paths: list[Path], checkpoint_path: Path
) -> dict[Path, int]:
    """Cached to ``<checkpoint>.pagecounts.json`` (keyed by filename, not
    full path, so a rerun from a different working directory or dataset
    root still hits the cache) — a resumed run shouldn't re-read every
    PDF's xref table just to rebuild the same schedule."""
    cache_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".pagecounts.json")
    cached: dict[str, int] = {}
    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            cached = {}

    missing = [p for p in paths if p.name not in cached]
    if missing:
        counts = await _page_counts(missing)
        cached.update({p.name: n for p, n in counts.items()})
        try:
            cache_path.write_text(json.dumps(cached))
        except OSError:
            pass

    return {p: cached[p.name] for p in paths}


def _build_batches(
    paths: list[Path],
    page_counts: dict[Path, int],
    *,
    max_pages_per_batch: int,
    max_files_per_batch: int,
) -> list[list[Path]]:
    """Longest-processing-time-first batching: sort files by page count
    descending, greedily pack a batch until either page or file budget is
    hit, emit batches biggest-first.

    Real, found-not-assumed motivation, two problems solved at once:
    (1) a FIFO queue over arbitrarily-sized files left GPUs idle at the end
    of a run while one worker finished a straggler (observed directly —
    2 of 3 GPUs at 0% while one worker processed a 450-page file); LPT
    scheduling minimizes exactly this kind of makespan gap. (2) a batch's
    extraction timeout has to cover its combined page count in one
    blocking call — grouping a few small files together (small combined
    timeout) instead of accidentally pairing them with a 450-page report
    (which now gets its own batch, since it alone likely already hits
    max_pages_per_batch) means the per-batch timeout formula stays
    meaningful instead of one fixed ceiling shared by wildly different
    batch sizes.
    """
    ordered = sorted(paths, key=lambda p: page_counts.get(p, 1), reverse=True)
    batches: list[list[Path]] = []
    current: list[Path] = []
    current_pages = 0
    for p in ordered:
        pages = page_counts.get(p, 1)
        if current and (
            current_pages + pages > max_pages_per_batch
            or len(current) >= max_files_per_batch
        ):
            batches.append(current)
            current = []
            current_pages = 0
        current.append(p)
        current_pages += pages
    if current:
        batches.append(current)
    return batches


def _extraction_timeout_s(
    total_pages: int, *, base_s: float, per_page_s: float, floor_s: float, ceiling_s: float
) -> float:
    """``clamp(base + per_page * pages, floor, ceiling)`` — computed per
    batch by the caller (which has the page counts), not inside
    ExtractionClient (which has no idea what a page is). ``floor_s`` keeps
    a tiny batch from getting an unreasonably short timeout; ``ceiling_s``
    keeps a huge one from turning a genuinely hung replica into an
    unbounded stall — 1800s (30min) is the recommended ceiling."""
    return max(floor_s, min(ceiling_s, base_s + per_page_s * total_pages))


# ── Ingestion ─────────────────────────────────────────────────────────────

# Sentinel pushed onto the hand-off queue once per stage-2 worker, once all
# stage-1 workers have finished — the shutdown signal. Not a real payload.
_SHUTDOWN = object()


async def ingest_dataset_gpu(
    dataset_dir: Path,
    *,
    collection: str,
    endpoints: list[str],
    checkpoint_path: Path,
    limit: int | None = None,
    annotated_stems: set[str] | None = None,
    max_pages_per_batch: int = 120,
    max_files_per_batch: int = 6,
    postprocess_workers: int = 8,
    extracted_queue_maxsize: int = 8,
    timeout_base_s: float = 30.0,
    timeout_per_page_s: float = 2.0,
    timeout_floor_s: float = 60.0,
    timeout_ceiling_s: float = 1800.0,
    use_blob_store: bool = True,
) -> dict[str, int]:
    """Two-stage producer/consumer ingestion across ``len(endpoints)``
    document-intelligence-gpu instances.

    Stage 1 (``len(endpoints)`` workers, one per replica): pulls a
    page-budget batch, extracts it in one ``extract_files`` call sized to
    the batch's own page count, and pushes each file's result onto a
    bounded hand-off queue — draining its own batch list as it goes so a
    blocked ``put`` never pins a whole batch's extracted images in memory.

    Stage 2 (``postprocess_workers`` workers): pulls one file at a time from
    the hand-off queue and runs ``process_extracted`` (chunk/embed/upload/
    store) — the part of the pipeline that's network/DB-bound, not
    GPU-bound, so it scales independently of the replica count. All stage-2
    workers share ``pipelines[0]`` (same embedder/store/blob_store as every
    other pipeline instance; only the extraction client differs, and stage
    2 never calls it) — this also gives the upload concurrency semaphore in
    ``DocumentIngestPipeline`` a genuinely global cap for free.

    The checkpoint is written ONLY by stage 2, after ``process_extracted``
    returns (success or failure) — one writer, and the same invariant
    ``ingest_file``'s checkpoint always had: an entry means the file is
    fully in the store, which is what makes ``--resume`` correct. A
    stage-1 failure (extraction error, or an unexpected exception in that
    worker) travels the queue as an exception payload and is recorded by
    stage 2 when it's pulled, tagged ``stage="extract"`` vs
    ``stage="process"`` so a failure's origin is visible in the checkpoint.

    Accepted, deliberate loss on a crash: a file already extracted but not
    yet processed is re-extracted on resume (bounded by
    ``extracted_queue_maxsize`` + ``postprocess_workers`` in flight) —
    keeping the queue small is itself the mitigation; this does not try to
    persist in-flight extraction results.
    """
    from substrate.capabilities.knowledge import DocumentIngestPipeline
    from substrate.capabilities.knowledge.document_ingest_pipeline import ExtractedFile

    cfg = AppConfig()
    store = build_vector_store(cfg.store, dimensions=_EMBEDDING_DIMENSIONS)
    if hasattr(store, "ensure_table"):
        await store.ensure_table()
    embedder = build_multimodal_embedder(cfg.mm_embed)
    blob_store = None
    if use_blob_store:
        blob_store = build_blob_store(cfg.storage)
        await blob_store.connect()

    pipelines = [
        DocumentIngestPipeline(
            build_extraction_client(
                DocumentIntelligenceConfig(service_url=url, timeout_s=cfg.doc_intel.timeout_s)
            ),
            embedder,
            store,
            blob_store=blob_store,
            key_prefix=cfg.storage.key_prefix,
        )
        for url in endpoints
    ]

    checkpoint = Checkpoint(checkpoint_path)
    pdfs = sorted(dataset_dir.glob("**/*.pdf"))
    if annotated_stems is not None:
        before = len(pdfs)
        pdfs = [p for p in pdfs if p.stem in annotated_stems]
        logger.info("Filtered to annotated files only: %d/%d", len(pdfs), before)
    if limit is not None:
        pdfs = pdfs[:limit]
    pending = [p for p in pdfs if p.stem not in checkpoint.done]
    logger.info(
        "Ingesting %d/%d files (%d already done, %d stage-1 workers, %d stage-2 workers)",
        len(pending),
        len(pdfs),
        len(pdfs) - len(pending),
        len(pipelines),
        postprocess_workers,
    )

    page_counts = await _load_or_compute_page_counts(pending, checkpoint_path)
    batches = _build_batches(
        pending,
        page_counts,
        max_pages_per_batch=max_pages_per_batch,
        max_files_per_batch=max_files_per_batch,
    )
    logger.info(
        "Scheduled %d batches (LPT, max_pages=%d, max_files=%d)",
        len(batches),
        max_pages_per_batch,
        max_files_per_batch,
    )

    batch_queue: asyncio.Queue[list[Path]] = asyncio.Queue()
    for b in batches:
        batch_queue.put_nowait(b)

    extracted_queue: asyncio.Queue = asyncio.Queue(maxsize=extracted_queue_maxsize)
    stats = {"files": 0, "failed": 0, "text_docs": 0, "image_docs": 0}

    async def _stage1(pipeline: DocumentIngestPipeline, worker_id: int) -> None:
        while True:
            try:
                batch = batch_queue.get_nowait()
            except asyncio.QueueEmpty:
                return

            batch_pages = sum(page_counts.get(p, 1) for p in batch)
            timeout_s = _extraction_timeout_s(
                batch_pages,
                base_s=timeout_base_s,
                per_page_s=timeout_per_page_s,
                floor_s=timeout_floor_s,
                ceiling_s=timeout_ceiling_s,
            )
            t0 = time.monotonic()
            try:
                results = await pipeline.extract_files(batch, timeout_s=timeout_s)
            except Exception as exc:
                # A stage-1 worker dying takes a whole GPU replica out of
                # the run for good -- must not be possible. Fail every file
                # in this batch and keep pulling the next one.
                logger.exception(
                    "[stage1 %d] Unexpected error extracting batch: %s", worker_id, exc
                )
                results = [exc for _ in batch]
            extract_ms = (time.monotonic() - t0) * 1000
            logger.info(
                "[stage1 %d] Extracted %d file(s), %d pages, timeout=%.0fs in %.0fms",
                worker_id,
                len(batch),
                batch_pages,
                timeout_s,
                extract_ms,
            )

            # Drain destructively so a blocked put() never pins the rest of
            # this batch's already-extracted images in memory.
            while results:
                path = batch.pop(0)
                result = results.pop(0)
                pages = page_counts.get(path, None)
                await extracted_queue.put((path, result, pages, extract_ms))

    async def _stage2(pipeline: DocumentIngestPipeline, worker_id: int) -> None:
        while True:
            item = await extracted_queue.get()
            if item is _SHUTDOWN:
                return
            path, result, pages, extract_ms = item

            if isinstance(result, Exception):
                stats["failed"] += 1
                await checkpoint.record(
                    path.stem,
                    status="failed",
                    error=str(result),
                    pages=pages,
                    stage="extract",
                    extract_ms=extract_ms,
                )
                logger.warning("[stage2 %d] Failed %s (extract): %s", worker_id, path.name, result)
                continue

            assert isinstance(result, ExtractedFile)
            t0 = time.monotonic()
            try:
                n_text, n_img = await pipeline.process_extracted(result, collection=collection)
            except Exception as exc:
                # Same reasoning as stage 1: a store.add()/embed error must
                # not silently kill this worker for the rest of the run.
                stats["failed"] += 1
                postprocess_ms = (time.monotonic() - t0) * 1000
                await checkpoint.record(
                    path.stem,
                    status="failed",
                    error=str(exc),
                    pages=pages,
                    stage="process",
                    extract_ms=extract_ms,
                    postprocess_ms=postprocess_ms,
                )
                logger.exception("[stage2 %d] Failed %s (process): %s", worker_id, path.name, exc)
                continue

            postprocess_ms = (time.monotonic() - t0) * 1000
            stats["files"] += 1
            stats["text_docs"] += n_text
            stats["image_docs"] += n_img
            await checkpoint.record(
                path.stem,
                status="ok",
                text_docs=n_text,
                image_docs=n_img,
                pages=pages,
                stage="process",
                extract_ms=extract_ms,
                postprocess_ms=postprocess_ms,
            )
            logger.info(
                "[stage2 %d] Ingested %s: %d text, %d image docs (extract=%.0fms, process=%.0fms)",
                worker_id,
                path.name,
                n_text,
                n_img,
                extract_ms,
                postprocess_ms,
            )

    lag_task = asyncio.create_task(_monitor_event_loop_lag())
    gpu_task = asyncio.create_task(_monitor_gpu_utilization())
    try:
        stage2_tasks = [
            asyncio.create_task(_stage2(pipelines[0], i)) for i in range(postprocess_workers)
        ]
        await asyncio.gather(*(_stage1(p, i) for i, p in enumerate(pipelines)))
        for _ in range(postprocess_workers):
            await extracted_queue.put(_SHUTDOWN)
        await asyncio.gather(*stage2_tasks)
    finally:
        lag_task.cancel()
        gpu_task.cancel()
        await asyncio.gather(lag_task, gpu_task, return_exceptions=True)

    await embedder.aclose()
    if blob_store is not None:
        await blob_store.disconnect()
    return stats


def _default_endpoints(cfg: AppConfig) -> list[str]:
    """Derive cfg.doc_intel.num_replicas document-intelligence-gpu endpoints
    from the configured base doc_intel service_url by incrementing its
    port — matches the 8021/8022/8023(/8024/8025/8026) convention
    document-intelligence-gpu-0..5 use in docker-compose.yml.
    """
    from urllib.parse import urlsplit, urlunsplit

    parts = urlsplit(cfg.doc_intel.service_url)
    base_port = parts.port or 8021
    urls = []
    for i in range(cfg.doc_intel.num_replicas):
        netloc = f"{parts.hostname}:{base_port + i}"
        urls.append(urlunsplit((parts.scheme, netloc, parts.path, "", "")))
    return urls


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="GPU-parallel checkpointed PDF ingestion")
    parser.add_argument(
        "--dataset",
        help=(
            "Limit to one dataset dir under "
            "data/pdfQA-Benchmark/real-pdfQA/01.2_Input_Files_PDF/ "
            "(e.g. ClimRetrieve). Omit to ingest the full benchmark."
        ),
    )
    parser.add_argument("--collection", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint JSONL path (default: <collection>.checkpoint.jsonl)",
    )
    parser.add_argument(
        "--max-pages-per-batch",
        type=int,
        default=120,
        help="Page budget per stage-1 extraction batch (LPT-scheduled). Bounds "
        "worst-case predict() call time and the per-batch timeout.",
    )
    parser.add_argument(
        "--max-files-per-batch",
        type=int,
        default=6,
        dest="max_files_per_batch",
        help="File count cap per batch, alongside --max-pages-per-batch.",
    )
    parser.add_argument(
        "--file-batch-size",
        type=int,
        default=argparse.SUPPRESS,  # only touches the namespace if explicitly passed
        dest="max_files_per_batch",
        help="Deprecated alias for --max-files-per-batch.",
    )
    parser.add_argument(
        "--postprocess-workers",
        type=int,
        default=8,
        help="Stage-2 (chunk/embed/upload/store) worker count -- independent "
        "of the extraction replica count. Match to llama-embed-gpu's "
        "--parallel setting.",
    )
    parser.add_argument(
        "--no-blob-store",
        action="store_true",
        help="Inline extracted images as base64 instead of uploading them -- "
        "for local/dev runs without an object store configured.",
    )
    parser.add_argument(
        "--include-unannotated",
        action="store_true",
        help=(
            "Ingest every PDF under the dataset dir, not just the ~514 with "
            "a matching ground-truth annotation. Off by default -- this "
            "driver exists for the evaluation, and the other ~3800 files "
            "can never be scored against anything."
        ),
    )
    args = parser.parse_args()

    cfg = AppConfig()
    # settings.ROOT_DIR is an anyio.Path (async-native), not stdlib
    # pathlib.Path -- its glob() isn't directly sorted()-compatible the
    # way ingest_dataset_gpu()'s synchronous glob call needs. Force a
    # stdlib Path here, same fix already used for the extraction notebook.
    base_dir = Path(str(settings.ROOT_DIR)) / "data/pdfQA-Benchmark/real-pdfQA/01.2_Input_Files_PDF"
    dataset_dir = base_dir / args.dataset if args.dataset else base_dir
    checkpoint_path = Path(args.checkpoint or f"{args.collection}.checkpoint.jsonl")
    endpoints = _default_endpoints(cfg)

    annotated_stems = None
    if not args.include_unannotated:
        annotations_dir = Path(str(settings.ROOT_DIR)) / "data/pdfQA-Annotations"
        annotated_stems = _annotated_stems(annotations_dir)
        logger.info("Loaded %d annotated file stems", len(annotated_stems))

    logger.info("Dataset dir: %s", dataset_dir)
    logger.info("Endpoints: %s", endpoints)
    logger.info("Collection: %s", args.collection)

    stats = asyncio.run(
        ingest_dataset_gpu(
            dataset_dir,
            collection=args.collection,
            endpoints=endpoints,
            checkpoint_path=checkpoint_path,
            limit=args.limit,
            annotated_stems=annotated_stems,
            max_pages_per_batch=args.max_pages_per_batch,
            max_files_per_batch=args.max_files_per_batch,
            postprocess_workers=args.postprocess_workers,
            use_blob_store=not args.no_blob_store,
        )
    )
    logger.info("Done: %s", stats)


if __name__ == "__main__":
    main()
