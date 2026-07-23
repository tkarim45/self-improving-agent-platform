"""Second-stage rerankers.

A cross-encoder reads (query, chunk) jointly instead of comparing two independently-computed
vectors, so it can judge relevance a bi-encoder cannot. It is also far too slow to run over a
whole corpus, which is the entire reason for the retrieve-then-rerank split: first stage
recalls ~50 cheaply, second stage orders them precisely.

`embedding-reranker-bench` measured the payoff on this hardware — cross-encoder rerank lifted
every embedder to recall@1 = 1.0. `production-rag-lab` measured the limit: reranking a
first stage that is already saturated buys nothing and still costs 2.7x latency. Both numbers
are why this is a swappable component with an identity implementation, not a hardcoded step.

The reranker is a **flywheel fine-tune target in M5** (mined (query, good, bad) triples).
"""

from __future__ import annotations

from src.index.embedders import tokenize
from src.interfaces import Reranker
from src.types import ScoredChunk


class IdentityReranker(Reranker):
    """No-op. The control arm in any rerank comparison."""

    name = "identity"

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int = 5) -> list[ScoredChunk]:
        return candidates[:k]


class LexicalReranker(Reranker):
    """Jaccard overlap between query and chunk tokens.

    Weak on purpose. It exists so the pipeline's rerank path is exercised in tests with no
    model download, and as a floor: a learned reranker that cannot beat token overlap is not
    earning its latency.
    """

    name = "lexical"

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int = 5) -> list[ScoredChunk]:
        wanted = set(tokenize(query))
        if not wanted:
            return candidates[:k]
        scored = [
            ScoredChunk(
                chunk=c.chunk,
                score=len(wanted & set(tokenize(c.chunk.contextualized())))
                / len(wanted | set(tokenize(c.chunk.contextualized()))),
                retriever=f"{c.retriever}>{self.name}",
            )
            for c in candidates
        ]
        return sorted(scored, key=lambda c: c.score, reverse=True)[:k]


class CrossEncoderReranker(Reranker):
    """Real cross-encoder. Model loads lazily; ms-marco-MiniLM-L-6-v2 is ~90 MB."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self.model_name = model_name
        self.name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: list[ScoredChunk], k: int = 5) -> list[ScoredChunk]:
        if not candidates:
            return []
        model = self._load()
        pairs = [(query, c.chunk.contextualized()) for c in candidates]
        scores = model.predict(pairs, show_progress_bar=False)
        scored = [
            ScoredChunk(chunk=c.chunk, score=float(s), retriever=f"{c.retriever}>rerank")
            for c, s in zip(candidates, scores, strict=True)
        ]
        return sorted(scored, key=lambda c: c.score, reverse=True)[:k]


def get_reranker(name: str) -> Reranker:
    if name in ("", "none", "identity"):
        return IdentityReranker()
    if name == "lexical":
        return LexicalReranker()
    return CrossEncoderReranker(name)
