"""Document link graph, for multi-hop retrieval.

The DuckDB docs cross-reference each other with Jekyll link tags:

    {% link docs/current/sql/query_syntax/qualify.md %}

That is an author-curated edge — a maintainer decided these two pages are related — which
makes it a better multi-hop signal than co-occurrence or embedding similarity, both of which
would just rediscover topical closeness the dense retriever already has.

The use is deliberately narrow. A multi-hop query fails in a specific way: the first stage
retrieves the page the query *sounds* like and never surfaces the bridge page, or surfaces it
below the cutoff. Boosting candidates that are linked from a top-ranked page targets exactly
that, and does nothing on single-hop queries where the answer is already at rank 1.
"""

from __future__ import annotations

import re
from collections import defaultdict

import networkx as nx

from src.types import Document

# {% link docs/current/sql/statements/copy.md %} -> sql/statements/copy.md
_LINK = re.compile(r"\{%\s*link\s+docs/[^/\s]+/(?P<path>[^\s%]+\.md)\s*%\}")


class DocGraph:
    """Directed page graph. Edges point from the citing page to the cited page."""

    def __init__(self) -> None:
        self.g = nx.DiGraph()

    def __len__(self) -> int:
        return self.g.number_of_nodes()

    @property
    def n_edges(self) -> int:
        return self.g.number_of_edges()

    def add_documents(self, docs: list[Document]) -> None:
        known = {d.source_path for d in docs}
        for doc in docs:
            self.g.add_node(doc.source_path, title=doc.title)
            for match in _LINK.finditer(doc.text):
                target = match.group("path")
                # Only keep edges inside the ingested corpus; a link to a page we did not
                # ingest is a dangling edge that would inflate the graph's apparent density.
                if target in known and target != doc.source_path:
                    self.g.add_edge(doc.source_path, target)

    def neighbors(self, page: str, undirected: bool = True) -> set[str]:
        """Pages one hop away. Undirected by default — 'A links to B' relates them both ways."""
        if page not in self.g:
            return set()
        out = set(self.g.successors(page))
        if undirected:
            out |= set(self.g.predecessors(page))
        return out

    def stats(self) -> dict[str, float]:
        n, e = len(self), self.n_edges
        degrees = [d for _, d in self.g.degree()]
        isolated = sum(1 for d in degrees if d == 0)
        return {
            "pages": n,
            "edges": e,
            "mean_degree": (sum(degrees) / n) if n else 0.0,
            "isolated_pages": isolated,
        }


def build_graph(docs: list[Document]) -> DocGraph:
    graph = DocGraph()
    graph.add_documents(docs)
    return graph


def boost_by_links(
    candidates,
    graph: DocGraph,
    seed_n: int = 3,
    boost: float = 0.5,
):
    """Re-score candidates: a page linked from one of the top `seed_n` pages gets a bump.

    The bump is proportional to the seed's own score, so a link from a confident hit counts
    for more than a link from a marginal one, and it is capped at one bump per candidate to
    stop a densely-linked hub page from accumulating boosts and swamping the ranking.
    """
    from src.types import ScoredChunk

    if not candidates or boost <= 0:
        return candidates

    seeds: list[tuple[str, float]] = []
    for hit in candidates:
        page = hit.chunk.source_path
        if page not in [p for p, _ in seeds]:
            seeds.append((page, hit.score))
        if len(seeds) >= seed_n:
            break

    linked: dict[str, float] = defaultdict(float)
    for page, score in seeds:
        for neighbor in graph.neighbors(page):
            linked[neighbor] = max(linked[neighbor], score)

    rescored = [
        ScoredChunk(
            chunk=c.chunk,
            score=c.score + boost * linked.get(c.chunk.source_path, 0.0),
            retriever=f"{c.retriever}+graph" if c.chunk.source_path in linked else c.retriever,
        )
        for c in candidates
    ]
    return sorted(rescored, key=lambda c: c.score, reverse=True)
