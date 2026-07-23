# Retrieval eval — DuckDB docs (Milestone 1)

- Corpus: `duckdb-web@89998bfe4a0c docs/current`
- Index: 4556 chunks, embedder `sentence-transformers/all-MiniLM-L6-v2`
- Queries: 35 (32 single-hop, 3 multi-hop)
- Labels are page-level; ranked chunks are collapsed to ranked unique pages.

## Results

| config | R@1 | R@3 | R@5 | R@10 | MRR | nDCG@10 | latency |
|---|---|---|---|---|---|---|---|
| bm25 | 0.129 | 0.400 | 0.471 | 0.586 | 0.301 | 0.365 | 12.6 ms |
| dense | 0.357 | 0.571 | 0.771 | 0.814 | 0.535 | 0.592 | 45.5 ms |
| hybrid | 0.257 | 0.571 | 0.643 | 0.800 | 0.466 | 0.537 | 24.1 ms |
| hybrid(bm25=0.3,dense=1.0) | 0.343 | 0.600 | 0.743 | 0.814 | 0.536 | 0.598 | 20.4 ms |
| dense+graph(0.05) | 0.371 | 0.600 | 0.786 | 0.843 | 0.573 | 0.618 | 11.4 ms |
| dense+rerank | 0.357 | 0.671 | 0.771 | 0.814 | 0.558 | 0.608 | 4813.1 ms |
| hybrid+rerank | 0.300 | 0.529 | 0.657 | 0.757 | 0.485 | 0.536 | 4411.8 ms |

## Multi-hop (3 queries)

`coverage` = every required page present in the top 10, which is what a multi-hop
answer actually needs. `recall@10` gives partial credit and so reads higher.

| config | recall@10 | coverage@10 |
|---|---|---|
| bm25 | 0.500 | 0.333 |
| dense | 0.833 | 0.667 |
| hybrid | 0.667 | 0.333 |
| hybrid(bm25=0.3,dense=1.0) | 0.833 | 0.667 |
| dense+graph(0.05) | 0.833 | 0.667 |
| dense+rerank | 0.667 | 0.333 |
| hybrid+rerank | 0.667 | 0.333 |

## Missed by every config

None.
