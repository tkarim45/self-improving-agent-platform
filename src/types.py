"""Core data types shared by every subsystem.

These are deliberately plain dataclasses with no dependency on any model, index, or
provider. Everything downstream (retrieval, agent, eval, flywheel) speaks in these, so an
implementation can be swapped from mock to local to cloud without touching its callers.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A source document before chunking."""

    doc_id: str
    tenant: str
    text: str
    source_path: str
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text, traceable back to its source.

    `heading_path` is what makes a citation readable to a human ("Docs > SQL > Aggregates"),
    and it is also fed to the embedder as a contextual prefix.
    """

    chunk_id: str
    doc_id: str
    tenant: str
    text: str
    source_path: str
    heading_path: tuple[str, ...] = ()
    ordinal: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        where = " > ".join(self.heading_path) if self.heading_path else self.source_path
        return f"[{self.chunk_id}] {where}"

    def contextualized(self) -> str:
        """Text with its heading path prepended.

        Retrieval quality on doc corpora improves when a chunk carries the section it came
        from, because a chunk about `PIVOT` rarely repeats the word "SQL" in its body.
        """
        if not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n\n" + self.text


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    retriever: str = ""

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id


@dataclass
class Citation:
    chunk_id: str
    quote: str = ""


@dataclass
class Trace:
    """Everything one request did. The flywheel's raw input.

    Written for every request in Milestone 3, scored in Milestone 4, mined in Milestone 5.
    """

    trace_id: str
    tenant: str
    query: str
    answer: str = ""
    retrieved: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    model_tier: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    config_version: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    steps: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    trace: Trace | None = None


@dataclass
class CandidateConfig:
    """A proposed improvement produced by the flywheel (Milestone 5)."""

    version: str
    component: str  # "reranker" | "router" | "embedder" | "prompt"
    artifact_path: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)


def stable_id(*parts: str, length: int = 12) -> str:
    """Deterministic short id.

    Content-addressed on purpose: re-ingesting an unchanged document yields the same chunk
    ids, so incremental re-index is a set difference and citations stay valid across runs.
    """
    h = hashlib.sha256("\x00".join(parts).encode("utf-8")).hexdigest()
    return h[:length]
