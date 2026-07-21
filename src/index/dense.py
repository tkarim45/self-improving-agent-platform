"""Dense (vector) retrieval on FAISS."""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from src.interfaces import Embedder, Retriever
from src.types import Chunk, ScoredChunk


class DenseRetriever(Retriever):
    """Flat inner-product FAISS index over L2-normalized vectors, so IP == cosine.

    Flat is exact and the right default at docs scale. Swapping in IVF/HNSW is an index-type
    change only, and vector-db-benchmark already has the recall-vs-latency numbers for when
    that trade becomes worth making.
    """

    name = "dense"

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self._index = faiss.IndexFlatIP(embedder.dim)
        self._chunks: list[Chunk] = []

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        vecs = self.embedder.encode([c.contextualized() for c in chunks])
        arr = np.asarray(vecs, dtype="float32")
        faiss.normalize_L2(arr)
        self._index.add(arr)
        self._chunks.extend(chunks)

    def search(self, query: str, tenant: str, k: int = 10) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        arr = np.asarray(self.embedder.encode([query]), dtype="float32")
        faiss.normalize_L2(arr)
        # Over-fetch, because the tenant filter is applied after the search. Correct as long
        # as one tenant's chunks live in one index; a shared index needs a real IDSelector.
        depth = min(len(self._chunks), max(k * 4, k))
        scores, ids = self._index.search(arr, depth)
        out: list[ScoredChunk] = []
        for score, idx in zip(scores[0], ids[0], strict=True):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            if chunk.tenant != tenant:
                continue
            out.append(ScoredChunk(chunk=chunk, score=float(score), retriever=self.name))
            if len(out) >= k:
                break
        return out

    def save(self, path: Path) -> None:
        faiss.write_index(self._index, str(path))

    def load(self, path: Path, chunks: list[Chunk]) -> None:
        self._index = faiss.read_index(str(path))
        self._chunks = list(chunks)
        if self._index.ntotal != len(self._chunks):
            raise ValueError(
                f"index/chunk mismatch: {self._index.ntotal} vectors vs {len(self._chunks)} "
                "chunks — the index is stale, re-run ingest"
            )
