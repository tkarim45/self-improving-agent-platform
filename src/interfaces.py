"""The five stable interfaces (docs/01-architecture.md).

Keep these signatures stable. Every milestone adds implementations behind them, never new
required parameters, so that a mock built in Milestone 0 still satisfies the contract the
flywheel calls in Milestone 5.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from src.types import Answer, CandidateConfig, Chunk, ScoredChunk, Trace


class Embedder(ABC):
    """Text to vector. Swappable: hashing (offline tests) or sentence-transformers (real)."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @property
    def name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class Retriever(ABC):
    """`Retriever.search(query, tenant, k) -> [Chunk]` from the architecture doc.

    Returns ScoredChunk rather than bare Chunk so fusion and reranking upstream have
    something to fuse on. `.chunk` gets you the Chunk.
    """

    @abstractmethod
    def search(self, query: str, tenant: str, k: int = 10) -> list[ScoredChunk]: ...

    @abstractmethod
    def add(self, chunks: list[Chunk]) -> None: ...


class Reranker(ABC):
    """Second-stage ordering. A flywheel fine-tune target in Milestone 5."""

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[ScoredChunk], k: int = 5
    ) -> list[ScoredChunk]: ...


class Agent(ABC):
    """`Agent.run(query, tenant) -> Answer{text, citations, trace}`."""

    @abstractmethod
    def run(self, query: str, tenant: str) -> Answer: ...


class Judge(ABC):
    """`Judge.score(trace) -> {faithfulness, groundedness, task_success}`."""

    @abstractmethod
    def score(self, trace: Trace) -> dict[str, float]: ...


class Trainer(ABC):
    """`Trainer.improve(failures) -> CandidateConfig`."""

    @abstractmethod
    def improve(self, failures: list[Trace]) -> CandidateConfig: ...


class Promoter(ABC):
    """`Promoter.evaluate(candidate, incumbent, golden) -> promote|reject`."""

    @abstractmethod
    def evaluate(
        self,
        candidate: CandidateConfig,
        incumbent: CandidateConfig,
        golden: list[dict],
    ) -> bool: ...
