from __future__ import annotations

import pytest

from src.types import Document

SAMPLE = """---
title: Window Functions
layout: docs
---

Window functions let a query refer to other rows.

# Syntax

The general form follows the SQL standard.

## QUALIFY

QUALIFY filters the result of a window function, the way HAVING filters a GROUP BY.

```sql
# this hash is a comment, not a heading
SELECT name, row_number() OVER (ORDER BY score) AS rn
FROM players
QUALIFY rn <= 3;
```

## Framing

A frame clause bounds which rows the window sees, using ROWS or RANGE.
"""


@pytest.fixture
def sample_doc() -> Document:
    return Document(
        doc_id="doc1",
        tenant="duckdb",
        text=SAMPLE.split("---", 2)[2],
        source_path="sql/window.md",
        title="Window Functions",
    )
