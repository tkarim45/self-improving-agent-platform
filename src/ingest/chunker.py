"""Heading-aware markdown chunking.

Why not fixed-size windows: on a documentation corpus a fixed window routinely cuts a SQL
example away from the sentence that explains it, and the resulting chunk cites a section it
is only half from. Splitting on the heading tree keeps a chunk semantically whole and gives
every chunk a heading path for citation.

Two guards matter and are easy to get wrong:
  1. A `#` inside a fenced code block is a comment, not a heading.
  2. A long section still has to be split, but on paragraph boundaries, never mid-fence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from src.types import Chunk, Document, stable_id

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


@dataclass
class Section:
    heading_path: tuple[str, ...]
    text: str


def split_sections(text: str, root_title: str = "") -> list[Section]:
    """Split markdown into sections keyed by their heading path."""
    stack: list[tuple[int, str]] = []
    sections: list[Section] = []
    buf: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush() -> None:
        body = "\n".join(buf).strip()
        buf.clear()
        if not body:
            return
        path = tuple(h for _, h in stack)
        if root_title:
            path = (root_title, *path)
        sections.append(Section(heading_path=path, text=body))

    for line in text.splitlines():
        fence = _FENCE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence, fence_marker = False, ""
            buf.append(line)
            continue

        m = None if in_fence else _HEADING.match(line)
        if m:
            flush()
            level, title = len(m.group(1)), m.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            buf.append(line)

    flush()
    return sections


def _split_long(text: str, max_chars: int, overlap: int) -> list[str]:
    """Split an oversized section on paragraph boundaries, never inside a code fence."""
    paras, cur, in_fence, fence_marker = [], [], False, ""
    for block in text.split("\n\n"):
        cur.append(block)
        for line in block.splitlines():
            fence = _FENCE.match(line)
            if fence:
                marker = fence.group(1)
                if not in_fence:
                    in_fence, fence_marker = True, marker
                elif marker == fence_marker:
                    in_fence, fence_marker = False, ""
        if not in_fence:
            paras.append("\n\n".join(cur))
            cur = []
    if cur:
        paras.append("\n\n".join(cur))

    out: list[str] = []
    buf = ""
    for para in paras:
        if buf and len(buf) + len(para) + 2 > max_chars:
            out.append(buf)
            tail = buf[-overlap:] if overlap else ""
            buf = (tail + "\n\n" + para).strip() if tail else para
        else:
            buf = f"{buf}\n\n{para}" if buf else para
    if buf.strip():
        out.append(buf)
    # A single paragraph bigger than max_chars (a giant table or code block) stays whole
    # rather than being cut mid-statement.
    return out or [text]


class HeadingChunker:
    """Section-aware chunker with merge-small / split-large passes."""

    def __init__(self, max_chars: int = 1800, min_chars: int = 300, overlap: int = 150) -> None:
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.overlap = overlap

    def chunk(self, doc: Document) -> list[Chunk]:
        sections = split_sections(doc.text, root_title=doc.title)
        merged = self._merge_small(sections)

        chunks: list[Chunk] = []
        for section in merged:
            for piece in _split_long(section.text, self.max_chars, self.overlap):
                body = piece.strip()
                if not body:
                    continue
                ordinal = len(chunks)
                chunks.append(
                    Chunk(
                        chunk_id=stable_id(doc.doc_id, str(ordinal), body[:120]),
                        doc_id=doc.doc_id,
                        tenant=doc.tenant,
                        text=body,
                        source_path=doc.source_path,
                        heading_path=section.heading_path,
                        ordinal=ordinal,
                        metadata={"title": doc.title},
                    )
                )
        return chunks

    def _merge_small(self, sections: list[Section]) -> list[Section]:
        """Fold runt sections into the next one.

        Docs pages are full of two-line sections ("### Syntax" followed by one example).
        Alone they retrieve badly, because there is not enough text for either BM25 or a
        dense model to key on. Merged siblings keep the shallower heading path.
        """
        out: list[Section] = []
        pending: Section | None = None
        for section in sections:
            if pending is not None:
                combined = f"{pending.text}\n\n{section.text}"
                if len(combined) <= self.max_chars:
                    common = _common_prefix(pending.heading_path, section.heading_path)
                    section = Section(heading_path=common or pending.heading_path, text=combined)
                else:
                    out.append(pending)
                pending = None
            if len(section.text) < self.min_chars:
                pending = section
            else:
                out.append(section)
        if pending is not None:
            out.append(pending)
        return out


def _common_prefix(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    shared: list[str] = []
    for x, y in zip(a, b, strict=False):
        if x != y:
            break
        shared.append(x)
    return tuple(shared)
