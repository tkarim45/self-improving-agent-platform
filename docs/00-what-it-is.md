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

## Chosen domain

Pick **one** real domain with public documents and a natural task, e.g.:

- **Financial-filing analyst** — SEC 10-K/10-Q/8-K; task = multi-step analysis with
  citations. (Extends the existing `sec-rag-analyst` repo.)
- **Legal-intake triage** — natural fit with existing MY Law Company client domain.

Commit to one in Milestone 0. The flywheel needs a stable task to measure improvement
against.
