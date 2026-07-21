from __future__ import annotations

import pytest

from src.ingest.loaders import load_path, strip_frontmatter


def test_strip_frontmatter_extracts_title():
    body, meta = strip_frontmatter("---\ntitle: Aggregates\nlayout: docs\n---\n\nBody text.\n")
    assert meta["title"] == "Aggregates"
    assert body.strip() == "Body text."


def test_strip_frontmatter_noop_without_block():
    body, meta = strip_frontmatter("# Heading\n\ntext")
    assert meta == {} and body.startswith("# Heading")


def test_load_directory_skips_unsupported_and_empty(tmp_path):
    (tmp_path / "a.md").write_text("---\ntitle: A\n---\n\nalpha content")
    (tmp_path / "b.png").write_bytes(b"\x89PNG")
    (tmp_path / "empty.md").write_text("---\ntitle: E\n---\n")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.md").write_text("gamma content")

    docs = load_path(tmp_path, tenant="duckdb")
    assert {d.source_path for d in docs} == {"a.md", "nested/c.md"}
    assert all(d.tenant == "duckdb" for d in docs)


def test_title_falls_back_to_filename(tmp_path):
    (tmp_path / "window-functions.md").write_text("body")
    assert load_path(tmp_path, "t")[0].title == "window functions"


def test_missing_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_path(tmp_path / "nope", "t")
