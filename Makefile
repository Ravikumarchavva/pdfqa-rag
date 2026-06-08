.PHONY: sync lint test ingest query eval clean infra-up infra-down infra-logs

# ── Infrastructure ────────────────────────────────────────────────────────────

infra-up:
	docker compose up -d --wait

infra-down:
	docker compose down

infra-logs:
	docker compose logs -f postgres

# ── Dev ───────────────────────────────────────────────────────────────────────

sync:
	uv sync

lint:
	uv run ruff check pdfqa_rag tests
	uv run ruff format --check pdfqa_rag tests

fmt:
	uv run ruff format pdfqa_rag tests
	uv run ruff check --fix pdfqa_rag tests

test:
	uv run pytest tests/ -q

# ── Data commands (require .env with DATABASE_URL + EMBED_* set) ───────────────

ingest:
	uv run pdfqa-rag ingest

ingest-resume:
	uv run pdfqa-rag ingest --resume

query:
	uv run pdfqa-rag query "$(Q)" --limit 5

eval:
	uv run pdfqa-rag eval --k 5 --output results.json

# ── Smoke-check (no DB / network required) ────────────────────────────────────

smoke:
	uv run python -c "\
	from pathlib import Path; \
	from pdfqa_rag.data.loader import load_annotations; \
	pairs, chunks = load_annotations(Path('data/pdfQA-Annotations')); \
	print(f'{len(pairs)} QA pairs, {len(chunks)} unique source chunks') \
	"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.checkpoint.jsonl" -delete
