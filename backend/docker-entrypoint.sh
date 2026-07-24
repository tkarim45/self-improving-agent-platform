#!/usr/bin/env bash
# First-boot data bootstrap, then serve. Idempotent: skips fetch/ingest when the index is
# already built (a mounted data/ volume persists it across restarts).
set -euo pipefail

TENANT="${TENANT:-duckdb}"
EMBEDDER="${EMBEDDER:-sentence-transformers/all-MiniLM-L6-v2}"
INDEX_DIR="data/index/${TENANT}"

if [[ -f "${INDEX_DIR}/manifest.json" ]]; then
  echo "[entrypoint] index present at ${INDEX_DIR} — skipping fetch/ingest"
else
  echo "[entrypoint] no index — fetching corpus and building it (first boot; a few minutes)"
  python -m src.corpus fetch
  python -m src.ingest "data/corpus/${TENANT}" --tenant "${TENANT}" --embedder "${EMBEDDER}"
fi

echo "[entrypoint] starting API on 0.0.0.0:8000 (live_enabled=${SIAP_ALLOW_LIVE:-unset})"
exec python -m src.api --host 0.0.0.0 --port 8000
