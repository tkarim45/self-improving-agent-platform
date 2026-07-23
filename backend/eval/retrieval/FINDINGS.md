# M1 retrieval — what the numbers say

Generated numbers live in [`report.md`](report.md) (regenerate with `make eval`). This file
is the analysis, written after the run, including the parts that went against expectation.

Setup: 411 DuckDB doc pages → 4,556 chunks, MiniLM-L6-v2 embeddings, 35 labeled queries
(32 single-hop, 3 multi-hop), page-level labels, on an M1 (8 GB).

## Headline

**The cheapest arm won.** `dense+graph(0.05)` leads on R@1, R@10, MRR and nDCG at **11 ms**
per query, while `dense+rerank` costs **4,800–11,700 ms** across runs — a **400–1,000×**
latency multiplier — and still loses on R@1, R@10 and nDCG.

| config | R@1 | R@10 | MRR | nDCG@10 | latency |
|---|---|---|---|---|---|
| bm25 | 0.129 | 0.586 | 0.301 | 0.365 | 8 ms |
| dense | 0.357 | 0.814 | 0.535 | 0.592 | 44 ms |
| hybrid (1:1 RRF) | 0.257 | 0.800 | 0.466 | 0.537 | 20 ms |
| hybrid (bm25 0.3) | 0.343 | 0.814 | 0.536 | 0.598 | 19 ms |
| **dense+graph(0.05)** | **0.371** | **0.843** | **0.573** | **0.618** | **11 ms** |
| dense+rerank | 0.357 | 0.814 | 0.558 | 0.608 | 4,813 ms |
| hybrid+rerank | 0.300 | 0.757 | 0.485 | 0.536 | 4,412 ms |

Latency is indicative, not a benchmark: single M1 under varying load. The rerank arms ranged
4,412–11,697 ms across runs; the fast arms are stable to within a few ms. The order-of-
magnitude gap is far too large to be noise, but no single figure here should be quoted as
precise.

## 1. Naive hybrid is *worse* than its own better leg

Equal-weight RRF scored R@1 0.257 against dense's 0.357. Fusing a strong retriever with a
weak one drags the strong one down — BM25 manages R@1 0.129 here.

The weighted arm separates the two possible explanations. At `bm25_weight=0.3` hybrid
recovers to R@1 0.343 / nDCG 0.598, edging out plain dense on nDCG. So RRF was not the wrong
fusion; it was being fed a lopsided pair of runs and treating them as equals.

Worth stating plainly: "add hybrid search" is not free. On this corpus, adding BM25 at equal
weight costs 10 points of R@1.

## 2. Why BM25 is weak here, and why that is deliberate

The queries were written the way a user asks in an issue thread, avoiding the page's own
title wording — "turn the distinct values of a column into separate columns" rather than
"PIVOT". A query that quotes the title makes BM25 look far better than it is in production.

This is a **property of the eval set, not of BM25**. A corpus whose users paste exact error
strings or function names would invert this, which is exactly why the weight is a config
field the flywheel can tune in M5 rather than a constant.

## 3. The cross-encoder does not earn its latency — and never fixes rank 1

Reranking improved the middle of the ranking (R@3 0.571 → 0.671) but left **R@1 completely
unchanged at 0.357**, for 400–1,000× the latency. It reorders what is already there and
cannot conjure the right page into a pool that never contained it.

`hybrid+rerank` (0.300 R@1) losing to `dense+rerank` (0.357) makes the same point from the
other side: rerank quality is capped by first-stage quality. Reranking a worse pool gives a
worse result, at identical cost.

This matches `production-rag-lab`'s finding that reranking helps only where there is
headroom, and it sharpens it — here the headroom is in ranks 2–10, not at rank 1.

## 4. The graph did NOT solve multi-hop, which is what it was built for

Multi-hop coverage@10 (all required pages present, the thing a multi-hop answer actually
needs) sits at **0.667 for plain dense and 0.667 for dense+graph**. No improvement.

Worse, the boost has a cliff. Swept on the same 35 queries:

| boost | R@1 | R@10 | MRR | multi-hop coverage |
|---|---|---|---|---|
| 0.00 | 0.357 | 0.814 | 0.535 | 0.667 |
| 0.02 | 0.371 | 0.814 | 0.560 | 0.667 |
| **0.05** | **0.371** | **0.843** | **0.573** | **0.667** |
| 0.10 | 0.357 | 0.814 | 0.567 | 0.667 |
| 0.20 | 0.371 | 0.814 | 0.586 | **0.333** |
| 0.50 | 0.314 | 0.757 | 0.538 | **0.333** |

At ≥0.2 coverage halves. The docs average **5.7 links per page**, so boosting all neighbours
of the top 3 seeds injects ~17 topically-adjacent pages and floods the top 10 — pushing out
the genuine second hop. The mechanism intended to *find* the bridge page is what evicts it.

So the honest read: the link graph is a **mild general-purpose ranking nudge** (+0.038 MRR,
+0.029 R@10 at boost 0.05) that happens to be nearly free, and **not** a multi-hop solution.
The cliff is pinned as a test so a future config change cannot silently cross it.

Reranking is actively *harmful* to multi-hop: coverage drops 0.667 → 0.333, because a
cross-encoder scores each chunk against the query independently and the bridge page rarely
looks relevant on its own. Multi-hop needs joint reasoning over a *set*, which is an agent
concern (M2), not a ranking one.

Caveat: 3 multi-hop queries. Each is worth 0.333 of the coverage metric, so these moves are
directional only. Expanding that slice is M2 work.

## 5. One label was wrong, and the retriever found it

`q15` ("where do I put my access key so it can reach cloud storage") was missed by every
config. Inspection showed the retriever was right and the label was wrong: it was labeled
`create_secret.md`, a 2-chunk bare syntax reference, while `configuration/secrets_manager.md`
— which dense ranked **4th** — is the page that actually explains where credentials live.

The label was widened after reading both pages. The correction is recorded inline in
`duckdb.yaml` so it stays auditable, because quietly editing labels until the numbers improve
is how an eval set stops meaning anything.

The general lesson, which will recur: on a docs corpus the answer is routinely split between
a **reference page** and a **concept page**, so single-page labels systematically understate
recall.

## What ships as the M1 default

`dense` first stage + `graph_boost=0.05`. Best measured quality, ~11 ms, no reranker.

The cross-encoder stays in the codebase behind a config flag rather than being deleted: it is
the M5 fine-tune target, and a *tuned* reranker is a different proposition from this
off-the-shelf one. But it is **off by default**, on evidence.

## Threats to validity

- 35 queries is small. Differences under ~0.05 on any metric are not resolvable.
- 3 multi-hop queries is very small (see above).
- Labels are author-written, so they inherit the author's idea of what a user asks. Real
  query patterns from the DuckDB issue tracker arrive with the M6 simulator.
- One embedder (MiniLM-L6-v2) and one cross-encoder. `embedding-reranker-bench` covers the
  model-choice axis; this eval deliberately holds it fixed.
