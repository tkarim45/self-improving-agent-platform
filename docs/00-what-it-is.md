# 00 — What this project is

## Problem

RAG and agent systems are shipped and then frozen. They answer the same way forever,
degrade silently as documents change, and nobody can prove they work. Enterprises cancel
these projects (Gartner: >40% of agentic projects by 2027) because there is no evidence
the agent is correct, no way to catch regressions, and no mechanism to improve it.

## What we build

A multi-tenant **domain agent platform** with three things most demos lack:

1. **Grounded agentic answers** — not just retrieval, but a plan→retrieve→tool→critic
   agent that cites every claim and can call tools (SQL, web, internal APIs) via MCP.
2. **Continuous evaluation in production** — every answer is scored online; a sampled
   subset is judged by a calibrated LLM-judge; a CI gate blocks any change that
   regresses quality.
3. **A self-improvement flywheel** — failed traces become new eval cases; a small
   on-device model (reranker / embedder / router) is re-fine-tuned via MLX-LoRA; the new
   config is shadow-tested and promoted **only when it measurably beats the incumbent**.

## Goals

- Prove the system's answer quality **rises over time with no human in the retrain loop**.
- Keep it running on an **Apple M1 with 8 GB RAM** — local small models do the cheap work
  and are the fine-tune targets; Claude does the heavy reasoning; big scale-out is an
  optional documented cloud step.
- Ship a real **product surface**: chat UI with citations + an admin console with eval,
  cost, latency, and quality dashboards.

## Non-goals

- Not a general chatbot — it's grounded in a tenant's ingested corpus.
- Not a from-scratch model trainer — that's the sibling `ondevice-model-lifecycle` repo.
  Here, fine-tuning is limited to small reranker/router adapters.
- Not a Kubernetes platform demo — multi-tenant scale-out is documented, not required.

## Success criteria (definition of done)

| # | Criterion | Evidence |
|---|---|---|
| S1 | Grounded agent answers a domain question with correct inline citations | Trace + citation check |
| S2 | Every production answer is scored online (faithfulness, groundedness, task-success) | Metrics dashboard |
| S3 | CI gate blocks a deliberately-worse prompt/config from merging | Red PR in CI |
| S4 | The flywheel retrains a small model from mined failures and shadow-tests it | Promotion log |
| S5 | **Answer quality trends up over N simulated weeks with zero human retrain intervention** | The headline curve |
| S6 | Cost-per-query stays bounded while quality rises | Cost line on the same chart |
| S7 | Whole thing boots and runs on the M1 (cheap path fully local) | `make dev` + screen recording |

## Chosen domain — COMMITTED (Milestone 0, 2026-07-21)

**Open-source support agent over the DuckDB documentation.**

- **Corpus:** `github.com/duckdb/duckdb-web`, `docs/current` only — 411 pages, 4.1 MB,
  Jekyll markdown. Fetched by `python -m src.corpus fetch`, never committed.
- **Task:** answer a DuckDB user's question with a grounded, cited answer, the way a good
  maintainer would in an issue thread.
- **Query supply for the Milestone 6 simulator:** the DuckDB issue tracker and Discord
  are full of real user questions, so the usage stream can be replayed from real query
  patterns rather than invented ones.

Why this over the two originally-listed candidates:

- **Not SEC filings.** `sec-rag-analyst` (shipped) and `agentic-filing-analyst`
  (scaffolded) already cover that corpus. A third would prove nothing new.
- **Less memorized than FastAPI/Postgres.** Claude knows popular frameworks well enough to
  answer from parametric memory, which would mask retrieval failures — the system would
  look grounded while the retriever was broken. DuckDB is real but niche enough that a
  wrong retrieval shows up as a wrong answer, so groundedness is actually measurable.
- **Verifiable ground truth.** Answers are SQL semantics, so a golden case can be checked
  by running the query, not just judged by another model. That matters in Milestone 5,
  where a judge-only signal is what reward-hacking exploits.

One caveat to design around: docs change under us. The corpus is pinned by commit sha
(`--ref`), so an improvement curve is not silently confounded by upstream doc edits.

The flywheel needs a stable task to measure improvement against, and this is it.
