from __future__ import annotations

from src.ingest.chunker import HeadingChunker, split_sections
from src.types import Document


def test_heading_path_is_captured(sample_doc):
    chunks = HeadingChunker(min_chars=0).chunk(sample_doc)
    paths = {c.heading_path for c in chunks}
    assert ("Window Functions", "Syntax", "QUALIFY") in paths


def test_hash_inside_code_fence_is_not_a_heading(sample_doc):
    """A `#` comment inside ```sql must not open a new section."""
    for chunk in HeadingChunker(min_chars=0).chunk(sample_doc):
        assert not any("this hash is a comment" in h for h in chunk.heading_path)


def test_code_fence_survives_chunking(sample_doc):
    chunks = HeadingChunker(min_chars=0).chunk(sample_doc)
    qualify = [c for c in chunks if "QUALIFY rn <= 3" in c.text]
    assert len(qualify) == 1
    # The fence opened and closed inside one chunk.
    assert qualify[0].text.count("```") % 2 == 0


def test_contextualized_prepends_heading_path(sample_doc):
    chunk = HeadingChunker(min_chars=0).chunk(sample_doc)[0]
    assert chunk.contextualized().startswith("Window Functions")


def test_small_sections_are_merged():
    doc = Document(
        doc_id="d",
        tenant="t",
        text="\n".join(f"## S{i}\n\nshort body {i}." for i in range(6)),
        source_path="p.md",
        title="T",
    )
    merged = HeadingChunker(min_chars=300, max_chars=1800).chunk(doc)
    unmerged = HeadingChunker(min_chars=0, max_chars=1800).chunk(doc)
    assert len(merged) < len(unmerged)


def test_long_section_is_split_under_max_chars():
    body = "\n\n".join(f"Paragraph {i} " + "filler words here. " * 20 for i in range(40))
    doc = Document(doc_id="d", tenant="t", text=f"# Big\n\n{body}", source_path="p.md", title="T")
    chunks = HeadingChunker(max_chars=1200, overlap=100).chunk(doc)
    assert len(chunks) > 1
    assert all(len(c.text) <= 1200 + 200 for c in chunks)


def test_chunk_ids_are_stable_across_runs(sample_doc):
    a = [c.chunk_id for c in HeadingChunker().chunk(sample_doc)]
    b = [c.chunk_id for c in HeadingChunker().chunk(sample_doc)]
    assert a == b and len(set(a)) == len(a)


def test_split_sections_on_empty_text():
    assert split_sections("") == []
