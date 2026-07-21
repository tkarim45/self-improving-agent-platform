"""Sparse (keyword) retrieval over a tenant's chunks."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from src.index.embedders import tokenize
from src.interfaces import Retriever
from src.types import Chunk, ScoredChunk


class BM25Retriever(Retriever):
    """BM25 over contextualized chunk text.

    The index is rebuilt on `add` rather than updated incrementally. rank-bm25 has no
    incremental API, and on a docs-sized corpus a rebuild is well under a second, so the
    simpler thing is also the fast enough thing. Revisit if a tenant passes ~100k chunks.
    """

    name = "bm25"

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._tokens: list[set[str]] = []
        self._bm25: BM25Okapi | None = None

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self._chunks.extend(chunks)
        self._rebuild()

    def _rebuild(self) -> None:
        corpus = [tokenize(c.contextualized()) for c in self._chunks]
        self._tokens = [set(doc) for doc in corpus]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, tenant: str, k: int = 10) -> list[ScoredChunk]:
        if self._bm25 is None:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        # Relevance is gated on token overlap, not on `score > 0`. Okapi IDF is negative for
        # any term carried by more than half the corpus, so on a small index a real match can
        # score below zero — thresholding the score would silently drop it.
        wanted = set(tokens)
        ranked = sorted(
            (
                (score, chunk)
                for score, chunk, chunk_tokens in zip(
                    scores, self._chunks, self._tokens, strict=True
                )
                if chunk.tenant == tenant and wanted & chunk_tokens
            ),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return [
            ScoredChunk(chunk=chunk, score=float(score), retriever=self.name)
            for score, chunk in ranked[:k]
        ]
