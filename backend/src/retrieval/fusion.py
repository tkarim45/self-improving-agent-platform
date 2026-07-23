"""Reciprocal rank fusion.

BM25 scores and cosine similarities are not on a comparable scale — BM25 is unbounded and
corpus-dependent, cosine is [-1, 1] — so summing or normalizing them is guesswork that
changes meaning as the corpus grows. RRF ignores scores entirely and fuses on rank position,
which is why it survives that mismatch.

    score(d) = sum over runs of 1 / (k + rank(d))

`k` (60 by convention, from Cormack et al. 2009) damps the top of each list: without it a
rank-1 hit would score 1.0 and dominate any amount of agreement further down. Larger k means
flatter weighting and more influence from consensus across runs.
"""

from __future__ import annotations

from src.types import ScoredChunk

DEFAULT_K = 60


def reciprocal_rank_fusion(
    runs: dict[str, list[ScoredChunk]],
    k: int = DEFAULT_K,
    weights: dict[str, float] | None = None,
    top_k: int = 10,
) -> list[ScoredChunk]:
    """Fuse several ranked runs into one.

    `runs` maps a retriever name to its ranked hits, best first. `weights` optionally scales
    a run's contribution — a lever the flywheel can tune in M5 once there is evidence about
    which side carries a given query type.
    """
    if not runs:
        return []
    weights = weights or {}

    scores: dict[str, float] = {}
    chunks: dict[str, ScoredChunk] = {}
    sources: dict[str, list[str]] = {}

    for run_name, hits in runs.items():
        weight = weights.get(run_name, 1.0)
        for rank, hit in enumerate(hits, start=1):
            cid = hit.chunk_id
            scores[cid] = scores.get(cid, 0.0) + weight / (k + rank)
            sources.setdefault(cid, []).append(run_name)
            # Keep the first chunk object seen; they are identical across runs by id.
            chunks.setdefault(cid, hit)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        ScoredChunk(
            chunk=chunks[cid].chunk,
            score=score,
            retriever="+".join(sorted(set(sources[cid]))),
        )
        for cid, score in ranked
    ]
