"""Embedders behind one interface.

`HashingEmbedder` exists so the whole pipeline is testable with no model download, no
network, and no memory pressure on an 8 GB machine. It is a real (if weak) lexical
embedder, not a stub returning zeros, so tests that assert "the right chunk ranks first"
are actually exercising vector search.

`SentenceTransformerEmbedder` is the real one. Default is all-MiniLM-L6-v2 (~90 MB, 384-d),
which the embedding-reranker-bench results say is a sane starting point on this hardware.
"""

from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache

from src.interfaces import Embedder

_TOKEN = re.compile(r"[a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class HashingEmbedder(Embedder):
    """Hashed bag-of-words with sublinear term weighting, L2-normalized.

    Deterministic across processes: the hash is sha1 of the token, not Python's salted
    `hash()`, so an index built in one run still matches queries in the next.
    """

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"hashing-{self._dim}"

    def _bucket(self, token: str) -> tuple[int, float]:
        digest = hashlib.sha1(token.encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "big") % self._dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return idx, sign

    def encode(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            counts: dict[str, int] = {}
            for token in tokenize(text):
                counts[token] = counts.get(token, 0) + 1
            for token, count in counts.items():
                idx, sign = self._bucket(token)
                vec[idx] += sign * (1.0 + math.log(count))
            norm = math.sqrt(sum(v * v for v in vec))
            out.append([v / norm for v in vec] if norm else vec)
        return out


class SentenceTransformerEmbedder(Embedder):
    """Real dense embeddings. Model loads lazily so importing this module stays cheap."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None
        self._dim: int | None = None

    @property
    def name(self) -> str:
        return self.model_name

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            # Renamed in sentence-transformers 5.x; keep working on both.
            getter = getattr(self._model, "get_embedding_dimension", None) or (
                self._model.get_sentence_embedding_dimension
            )
            self._dim = getter()
        return self._model

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._load()
        assert self._dim is not None
        return self._dim

    def encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]


@lru_cache(maxsize=4)
def get_embedder(name: str = "hashing") -> Embedder:
    """Resolve an embedder by config name. Cached so an index reload does not re-download."""
    if name.startswith("hashing"):
        _, _, dim = name.partition("-")
        return HashingEmbedder(dim=int(dim) if dim else 256)
    return SentenceTransformerEmbedder(name)
