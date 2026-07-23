"""The composed retrieval path: first stage -> fusion -> rerank.

Every stage is switchable from config, because the point of M1 is to *measure* which stages
earn their cost on this corpus, not to assume the maximal pipeline is best. `production-rag-lab`
found the same technique doing nothing on a saturated first stage and a lot on a weak one, so
"is reranking worth it" is only answerable per-corpus, against an eval set.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.index.graph import DocGraph, boost_by_links
from src.index.store import HybridIndex
from src.interfaces import Reranker, Retriever
from src.retrieval.fusion import DEFAULT_K, reciprocal_rank_fusion
from src.retrieval.rerank import IdentityReranker, get_reranker
from src.types import Chunk, ScoredChunk

MODES = ("bm25", "dense", "hybrid")


@dataclass
class RetrievalConfig:
    """A versioned retrieval configuration. The unit the flywheel promotes in M5."""

    mode: str = "hybrid"
    fetch_k: int = 50
    rrf_k: int = DEFAULT_K
    reranker: str = "none"
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    graph_boost: float = 0.0
    graph_seed_n: int = 3
    version: str = "v0"

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

    @property
    def label(self) -> str:
        stem = self.mode
        if self.mode == "hybrid" and (self.bm25_weight, self.dense_weight) != (1.0, 1.0):
            stem += f"(bm25={self.bm25_weight},dense={self.dense_weight})"
        if self.graph_boost:
            stem += f"+graph({self.graph_boost})"
        return stem if self.reranker in ("", "none") else f"{stem}+rerank"


class HybridRetriever(Retriever):
    def __init__(
        self,
        index: HybridIndex,
        config: RetrievalConfig | None = None,
        reranker: Reranker | None = None,
        graph: DocGraph | None = None,
    ) -> None:
        self.index = index
        self.config = config or RetrievalConfig()
        self.reranker = reranker or get_reranker(self.config.reranker)
        self.graph = graph

    @property
    def reranks(self) -> bool:
        return not isinstance(self.reranker, IdentityReranker)

    def add(self, chunks: list[Chunk]) -> None:
        self.index.add(chunks)

    def first_stage(self, query: str, tenant: str, depth: int) -> list[ScoredChunk]:
        cfg = self.config
        if cfg.mode == "bm25":
            return self.index.sparse.search(query, tenant, k=depth)
        if cfg.mode == "dense":
            return self.index.dense.search(query, tenant, k=depth)
        runs = {
            "bm25": self.index.sparse.search(query, tenant, k=depth),
            "dense": self.index.dense.search(query, tenant, k=depth),
        }
        return reciprocal_rank_fusion(
            runs,
            k=cfg.rrf_k,
            weights={"bm25": cfg.bm25_weight, "dense": cfg.dense_weight},
            top_k=depth,
        )

    def search(self, query: str, tenant: str, k: int = 10) -> list[ScoredChunk]:
        # Over-fetch whenever a later stage reorders: both the reranker and the graph boost
        # can only work on what the first stage recalled, so first-stage depth caps the
        # ceiling of the whole pipeline.
        boosts = self.reranks or (self.config.graph_boost > 0 and self.graph is not None)
        depth = max(self.config.fetch_k, k) if boosts else k
        candidates = self.first_stage(query, tenant, depth)

        if self.config.graph_boost > 0 and self.graph is not None:
            candidates = boost_by_links(
                candidates,
                self.graph,
                seed_n=self.config.graph_seed_n,
                boost=self.config.graph_boost,
            )

        if not self.reranks:
            return candidates[:k]
        return self.reranker.rerank(query, candidates, k=k)
