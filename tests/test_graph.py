from __future__ import annotations

from src.index.graph import boost_by_links, build_graph
from src.types import Chunk, Document, ScoredChunk


def doc(path: str, text: str) -> Document:
    return Document(doc_id=path, tenant="t", text=text, source_path=path, title=path)


def link(path: str) -> str:
    return f"{{% link docs/current/{path} %}}"


DOCS = [
    doc("a.md", f"see {link('b.md')} and {link('c.md')}"),
    doc("b.md", f"back to {link('a.md')}"),
    doc("c.md", "no links here"),
    doc("d.md", f"points at a missing page {link('nope.md')}"),
]


def test_link_tags_become_edges():
    g = build_graph(DOCS)
    assert g.g.has_edge("a.md", "b.md")
    assert g.g.has_edge("a.md", "c.md")


def test_links_to_pages_outside_the_corpus_are_dropped():
    """A dangling edge would inflate apparent graph density and boost a page we never indexed."""
    g = build_graph(DOCS)
    assert "nope.md" not in g.g
    assert g.neighbors("d.md") == set()


def test_neighbors_are_undirected_by_default():
    g = build_graph(DOCS)
    assert g.neighbors("c.md") == {"a.md"}  # only an inbound edge
    assert g.neighbors("c.md", undirected=False) == set()


def test_self_links_are_ignored():
    g = build_graph([doc("x.md", link("x.md"))])
    assert g.neighbors("x.md") == set()


def test_stats_report_isolated_pages():
    stats = build_graph(DOCS).stats()
    assert stats["pages"] == 4
    assert stats["isolated_pages"] == 1  # d.md, whose only link was dangling


def scored(path: str, score: float) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(chunk_id=path, doc_id=path, tenant="t", text=path, source_path=path),
        score=score,
        retriever="dense",
    )


def test_boost_promotes_a_page_linked_from_the_top_hit():
    g = build_graph(DOCS)
    # c.md starts last. a.md is the top hit and links to it; z.md is linked from nothing.
    hits = [scored("a.md", 1.0), scored("z.md", 0.9), scored("c.md", 0.8)]
    out = [c.chunk.source_path for c in boost_by_links(hits, g, seed_n=1, boost=0.15)]
    assert out.index("c.md") < out.index("z.md")


def test_a_strong_boost_lets_a_neighbour_overtake_its_own_seed():
    """The cliff the M1 sweep measured: at boost >= 0.2 neighbours swamp the real hits.

    Pinned as a test because it is the failure mode, not a bug — it is why the shipped
    default is 0.05.
    """
    g = build_graph(DOCS)
    hits = [scored("a.md", 1.0), scored("c.md", 0.8)]
    out = [c.chunk.source_path for c in boost_by_links(hits, g, seed_n=1, boost=0.5)]
    assert out[0] == "c.md"


def test_boost_marks_which_hits_the_graph_touched():
    g = build_graph(DOCS)
    out = boost_by_links([scored("a.md", 1.0), scored("c.md", 0.8)], g, seed_n=1, boost=0.5)
    boosted = [c for c in out if c.chunk.source_path == "c.md"][0]
    assert boosted.retriever.endswith("+graph")


def test_zero_boost_is_a_no_op():
    g = build_graph(DOCS)
    hits = [scored("a.md", 1.0), scored("c.md", 0.8)]
    assert boost_by_links(hits, g, boost=0.0) is hits


def test_boost_on_empty_candidates():
    assert boost_by_links([], build_graph(DOCS), boost=0.5) == []


def test_unlinked_pages_keep_their_score():
    g = build_graph(DOCS)
    out = boost_by_links([scored("a.md", 1.0), scored("z.md", 0.9)], g, seed_n=1, boost=0.5)
    assert [c.score for c in out if c.chunk.source_path == "z.md"] == [0.9]
