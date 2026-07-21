"""Per-tenant hybrid index: BM25 + dense, with persistence.

Milestone 0 builds and persists both index sides and searches each independently.
Reciprocal-rank fusion and the cross-encoder reranker land in Milestone 1 (see
docs/02-build-plan.md), which is why there is no combined `search` here yet — a fused score
with no retrieval eval set behind it is a number nobody can defend.

Tenant isolation is by directory: `data/index/<tenant>/`. One index per tenant means a
retrieval bug can leak nothing across tenants, which is worth more than sharing a vector
index for the sake of memory.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.index.bm25 import BM25Retriever
from src.index.dense import DenseRetriever
from src.index.embedders import get_embedder
from src.types import Chunk

MANIFEST = "manifest.json"
CHUNKS = "chunks.jsonl"
DENSE = "dense.faiss"
FORMAT_VERSION = 1


def _chunk_to_json(c: Chunk) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "tenant": c.tenant,
        "text": c.text,
        "source_path": c.source_path,
        "heading_path": list(c.heading_path),
        "ordinal": c.ordinal,
        "metadata": c.metadata,
    }


def _chunk_from_json(d: dict) -> Chunk:
    return Chunk(
        chunk_id=d["chunk_id"],
        doc_id=d["doc_id"],
        tenant=d["tenant"],
        text=d["text"],
        source_path=d["source_path"],
        heading_path=tuple(d.get("heading_path", ())),
        ordinal=d.get("ordinal", 0),
        metadata=d.get("metadata", {}),
    )


class HybridIndex:
    def __init__(self, tenant: str, embedder_name: str = "hashing", root: str | Path = "data/index"):
        self.tenant = tenant
        self.embedder_name = embedder_name
        self.root = Path(root)
        self.embedder = get_embedder(embedder_name)
        self.sparse = BM25Retriever()
        self.dense = DenseRetriever(self.embedder)
        self._chunks: list[Chunk] = []
        self._seen: set[str] = set()

    @property
    def path(self) -> Path:
        return self.root / self.tenant

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> int:
        """Add chunks, skipping ones already indexed. Returns the number actually added.

        Chunk ids are content-addressed (src.types.stable_id), so re-ingesting an unchanged
        page is a no-op and re-ingesting an edited one adds only what changed. That is the
        whole incremental re-index story for Milestone 0.
        """
        fresh = [c for c in chunks if c.chunk_id not in self._seen and c.tenant == self.tenant]
        if not fresh:
            return 0
        self.sparse.add(fresh)
        self.dense.add(fresh)
        self._chunks.extend(fresh)
        self._seen.update(c.chunk_id for c in fresh)
        return len(fresh)

    def save(self) -> Path:
        self.path.mkdir(parents=True, exist_ok=True)
        with (self.path / CHUNKS).open("w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(json.dumps(_chunk_to_json(c), ensure_ascii=False) + "\n")
        self.dense.save(self.path / DENSE)
        (self.path / MANIFEST).write_text(
            json.dumps(
                {
                    "format_version": FORMAT_VERSION,
                    "tenant": self.tenant,
                    "embedder": self.embedder_name,
                    "dim": self.embedder.dim,
                    "n_chunks": len(self._chunks),
                    "n_docs": len({c.doc_id for c in self._chunks}),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return self.path

    @classmethod
    def load(cls, tenant: str, root: str | Path = "data/index") -> HybridIndex:
        path = Path(root) / tenant
        manifest_path = path / MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"no index for tenant '{tenant}' at {path} — run ingest first")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"index format v{manifest.get('format_version')} != v{FORMAT_VERSION} — re-ingest"
            )

        index = cls(tenant, embedder_name=manifest["embedder"], root=root)
        with (path / CHUNKS).open(encoding="utf-8") as f:
            chunks = [_chunk_from_json(json.loads(line)) for line in f if line.strip()]

        # BM25 rebuilds from text (cheap); the dense side reloads its vectors rather than
        # re-embedding, which is the expensive half.
        index.sparse.add(chunks)
        index.dense.load(path / DENSE, chunks)
        index._chunks = chunks
        index._seen = {c.chunk_id for c in chunks}
        return index
