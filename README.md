# pdfqa-rag
experimenting rag

## Infra

`make infra-up` starts:
- **postgres** (pgvector) — chunk embeddings
- **seaweedfs** (S3 gateway on `:8333`) — durable store for raw downloaded PDFs/text.
  `download` and `ingest-docs` persist HF downloads here first; the local
  `data/documents` cache dir is just an ephemeral scratch copy for parsing,
  and gets refilled from SeaweedFS alone if deleted. Swappable for AWS S3 /
  MinIO in production via `STORAGE_*` env vars — no code change.
