# Backend image: the FastAPI product surface over the whole Python platform.
#
# On first boot the entrypoint fetches the DuckDB corpus and builds the hybrid index if one
# isn't present (mounted volume persists it across restarts), then serves the API. Dry by
# default — set SIAP_ALLOW_LIVE=1 and mount creds to allow real Bedrock spend.
FROM python:3.12-slim

# git: the corpus fetch shallow-clones duckdb-web. build-essential kept out — the pinned
# wheels (faiss-cpu, numpy) ship manylinux binaries, so no compiler is needed.
RUN apt-get update && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Deps first for layer caching. The MiniLM embedder model downloads at ingest time, not here.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    TENANT=duckdb \
    EMBEDDER=sentence-transformers/all-MiniLM-L6-v2

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=180s --retries=20 \
    CMD curl -fsS http://localhost:8000/api/health || exit 1

ENTRYPOINT ["bash", "docker-entrypoint.sh"]
