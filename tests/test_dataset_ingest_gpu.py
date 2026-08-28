"""dataset_ingest_gpu.py — pure scheduling/timeout logic and the staged
producer/consumer driver, exercised against fakes (no real GPU services,
no real S3/Postgres)."""

from __future__ import annotations

import json
from pathlib import Path

from pdfqa_rag.pipeline.dataset_ingest_gpu import (
    Checkpoint,
    _build_batches,
    _extraction_timeout_s,
)

# ── _build_batches (LPT scheduling) ──────────────────────────────────────


def _paths(n: int) -> list[Path]:
    return [Path(f"file{i}.pdf") for i in range(n)]


def test_build_batches_puts_a_huge_file_in_its_own_batch():
    """A file that alone exceeds the page budget must not be starved --
    it becomes a batch of one rather than looping forever trying to add
    more files to a batch that's already over budget."""
    p = _paths(1)
    counts = {p[0]: 450}

    batches = _build_batches(p, counts, max_pages_per_batch=120, max_files_per_batch=6)

    assert batches == [[p[0]]]


def test_build_batches_processes_largest_files_first():
    """LPT: sorted descending by page count, so the biggest work is
    scheduled first -- minimizes the makespan gap where small stragglers
    finish fast and leave GPUs idle while one huge file finishes last."""
    p = _paths(3)
    counts = {p[0]: 5, p[1]: 450, p[2]: 20}

    batches = _build_batches(p, counts, max_pages_per_batch=120, max_files_per_batch=6)

    # p[1] (450 pages) must be its own first batch.
    assert batches[0] == [p[1]]


def test_build_batches_packs_small_files_together_under_the_page_budget():
    p = _paths(4)
    counts = {p[0]: 5, p[1]: 5, p[2]: 5, p[3]: 5}

    batches = _build_batches(p, counts, max_pages_per_batch=120, max_files_per_batch=6)

    assert batches == [p]  # all 4 fit in one batch (20 pages total)


def test_build_batches_respects_max_files_per_batch_even_under_page_budget():
    p = _paths(8)
    counts = {f: 1 for f in p}  # trivially small pages, file-count is the binding limit

    batches = _build_batches(p, counts, max_pages_per_batch=120, max_files_per_batch=3)

    assert [len(b) for b in batches] == [3, 3, 2]


def test_build_batches_missing_page_count_defaults_to_one_page():
    p = _paths(2)
    counts = {p[0]: 5}  # p[1] missing -- must not KeyError

    batches = _build_batches(p, counts, max_pages_per_batch=120, max_files_per_batch=6)

    assert sum(len(b) for b in batches) == 2


def test_build_batches_empty_input():
    assert _build_batches([], {}, max_pages_per_batch=120, max_files_per_batch=6) == []


# ── _extraction_timeout_s ─────────────────────────────────────────────────


def test_extraction_timeout_scales_with_pages():
    small = _extraction_timeout_s(20, base_s=30, per_page_s=2.0, floor_s=60, ceiling_s=1800)
    large = _extraction_timeout_s(450, base_s=30, per_page_s=2.0, floor_s=60, ceiling_s=1800)
    assert large > small


def test_extraction_timeout_respects_floor():
    t = _extraction_timeout_s(1, base_s=30, per_page_s=2.0, floor_s=60, ceiling_s=1800)
    assert t == 60  # 30 + 2*1 = 32, below the 60s floor


def test_extraction_timeout_respects_ceiling():
    t = _extraction_timeout_s(100_000, base_s=30, per_page_s=2.0, floor_s=60, ceiling_s=1800)
    assert t == 1800


# ── Checkpoint (telemetry fields) ────────────────────────────────────────


async def test_checkpoint_records_stage_and_timing_fields(tmp_path):
    path = tmp_path / "run.checkpoint.jsonl"
    checkpoint = Checkpoint(path)

    await checkpoint.record(
        "doc1",
        status="ok",
        text_docs=3,
        image_docs=1,
        pages=12,
        stage="process",
        extract_ms=450.0,
        postprocess_ms=1200.0,
    )

    lines = path.read_text().strip().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["pages"] == 12
    assert entry["stage"] == "process"
    assert entry["extract_ms"] == 450.0
    assert entry["postprocess_ms"] == 1200.0


async def test_checkpoint_resume_skips_done_files_regardless_of_status(tmp_path):
    path = tmp_path / "run.checkpoint.jsonl"
    checkpoint = Checkpoint(path)
    await checkpoint.record("doc1", status="ok")
    await checkpoint.record("doc2", status="failed", stage="extract")

    resumed = Checkpoint(path)

    assert resumed.done == {"doc1", "doc2"}


# ── Staged driver (fakes: no real network) ───────────────────────────────


class _FakeExtractResponse:
    def __init__(self, *, success=True, markdown="# A\n\nBody.", page_count=1, error=None):
        self.success = success
        self.markdown = markdown
        self.page_count = page_count
        self.images = []
        self.error = error


class _FakeExtractionClient:
    def __init__(self, *, fail_files: set[str] | None = None) -> None:
        self._fail_files = fail_files or set()

    async def extract_batch(self, items, *, timeout_s=None):
        return [
            _FakeExtractResponse(success=filename not in self._fail_files, error="boom")
            for _, filename, _ in items
        ]


class _FakeEmbedder:
    async def embed_texts(self, texts):
        return [[0.1, 0.2] for _ in texts]

    async def embed_text(self, text):
        return [0.1, 0.2]

    async def embed_images(self, images):
        return [[0.3, 0.4] for _ in images]

    async def embed_image(self, data):
        return [0.3, 0.4]

    async def aclose(self):
        pass


class _FakeStore:
    def __init__(self):
        self.added = []

    async def add(self, documents, *, collection):
        self.added.append((documents, collection))
        return [d.id for d in documents]


async def test_ingest_dataset_gpu_end_to_end_with_fakes(tmp_path, monkeypatch):
    """Real staged run (stage 1 -> queue -> stage 2 -> checkpoint) against
    fakes for every network dependency -- proves the two stages actually
    hand off correctly and every file lands in the checkpoint exactly
    once, success and failure both."""
    from pdfqa_rag.pipeline import dataset_ingest_gpu as mod

    dataset_dir = tmp_path / "pdfs"
    dataset_dir.mkdir()
    for i in range(5):
        (dataset_dir / f"doc{i}.pdf").write_bytes(b"%PDF-1.4 fake\n%%EOF")

    fake_store = _FakeStore()
    monkeypatch.setattr(mod, "build_vector_store", lambda cfg, dimensions: fake_store)
    monkeypatch.setattr(mod, "build_multimodal_embedder", lambda cfg: _FakeEmbedder())
    monkeypatch.setattr(
        mod,
        "build_extraction_client",
        lambda cfg: _FakeExtractionClient(fail_files={"doc2.pdf"}),
    )

    checkpoint_path = tmp_path / "run.checkpoint.jsonl"
    stats = await mod.ingest_dataset_gpu(
        dataset_dir,
        collection="kb",
        endpoints=["http://fake-0", "http://fake-1"],
        checkpoint_path=checkpoint_path,
        max_pages_per_batch=120,
        max_files_per_batch=6,
        postprocess_workers=2,
        use_blob_store=False,
    )

    assert stats["files"] == 4
    assert stats["failed"] == 1

    lines = [json.loads(line) for line in checkpoint_path.read_text().strip().splitlines()]
    assert len(lines) == 5
    by_name = {entry["file_name"]: entry for entry in lines}
    assert by_name["doc2"]["status"] == "failed"
    assert by_name["doc2"]["stage"] == "extract"
    assert by_name["doc0"]["status"] == "ok"
    assert by_name["doc0"]["stage"] == "process"


async def test_ingest_dataset_gpu_resumes_via_checkpoint(tmp_path, monkeypatch):
    from pdfqa_rag.pipeline import dataset_ingest_gpu as mod

    dataset_dir = tmp_path / "pdfs"
    dataset_dir.mkdir()
    (dataset_dir / "doc0.pdf").write_bytes(b"%PDF-1.4 fake\n%%EOF")
    (dataset_dir / "doc1.pdf").write_bytes(b"%PDF-1.4 fake\n%%EOF")

    checkpoint_path = tmp_path / "run.checkpoint.jsonl"
    checkpoint_path.write_text(json.dumps({"file_name": "doc0", "status": "ok"}) + "\n")

    fake_store = _FakeStore()
    monkeypatch.setattr(mod, "build_vector_store", lambda cfg, dimensions: fake_store)
    monkeypatch.setattr(mod, "build_multimodal_embedder", lambda cfg: _FakeEmbedder())
    extraction = _FakeExtractionClient()
    monkeypatch.setattr(mod, "build_extraction_client", lambda cfg: extraction)

    stats = await mod.ingest_dataset_gpu(
        dataset_dir,
        collection="kb",
        endpoints=["http://fake-0"],
        checkpoint_path=checkpoint_path,
        postprocess_workers=1,
        use_blob_store=False,
    )

    # Only doc1 should have been (re)processed -- doc0 was already done.
    assert stats["files"] == 1
    lines = [json.loads(line) for line in checkpoint_path.read_text().strip().splitlines()]
    assert [entry["file_name"] for entry in lines] == ["doc0", "doc1"]
