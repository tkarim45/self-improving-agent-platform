# Self-Improving Domain Agent Platform ("the RAG-agent OS")

> A multi-tenant platform where an organization drops in its documents + tools and a
> fleet of LLM agents answers, acts, and **measurably improves over time** through a
> closed evaluation-and-retraining loop — running on an Apple M1 (8 GB) laptop, with
> heavy reasoning on the Claude API and the parts that improve fine-tuned on-device.

**Status:** 📐 Spec / scaffold. No measured results yet. This repo is the build plan;
code lands milestone by milestone (see [`docs/02-build-plan.md`](docs/02-build-plan.md)).

**Capstone #1 of 3.** Sibling repos: `ondevice-model-lifecycle`,
`realtime-decision-intelligence`. Estimated effort: **6–12 months solo**.

---

## The one-sentence pitch

Most RAG/agent demos are static — they answer the same way on day 100 as day 1. This
platform closes the loop: every production trace is scored, failures are mined into new
eval cases, a **small on-device reranker/router is re-fine-tuned (MLX-LoRA)**, the new
config is shadow-tested, and it's promoted **only on measured lift**. The headline
deliverable is a **before/after quality curve proving the system improves with no human
in the retrain loop, at bounded cost.**

## Why this project exists

The mid-2026 hiring signal (see `../research.md`) says the gap in a strong AI portfolio
is not another RAG variant or eval framework — those are baseline. The gap is **one
integrated, deployed, domain-grounded, evaluated, actually-used system**. This is that
system, and the self-improvement flywheel is the differentiator nobody else ships.

It also answers the Gartner prediction that >40% of agentic projects get cancelled: this
agent comes with the eval + governance to prove it works.

## Headline metric (the thing to demo)

A chart with **weeks on the x-axis and answer quality (faithfulness / task-success) on
the y-axis**, trending up, annotated with each automated promotion event — while a
second line shows cost-per-query staying flat or falling. If that curve goes up without
a human touching the retrainer, the project succeeds.

## What runs where (Apple M1, 8 GB — the core constraint)

| Role | Where |
|---|---|
| Heavy agent reasoning, LLM-judge | **Claude API / AWS Bedrock** (cloud, temp 0) |
| Cheap router tier, offline demo | **Local Qwen2.5-1.5B / Llama-3.2-1B** via llama.cpp / MLX (Metal) |
| Reranker + embedder + router (the parts that improve) | **MLX-LoRA / QLoRA fine-tuned on-device (≤1.5B)** |
| Vector + keyword retrieval | FAISS + BM25 (laptop-light; no pgvector) |
| Multi-tenant K8s scale-out | Optional, documented cloud step — **not** required on the laptop |

Usable model memory on this machine is ~4–5 GB after the OS. Everything is scoped to fit.

## Architecture (one glance)

```
             ┌─────────────── Continuous Eval Flywheel ───────────────┐
             │                                                         │
  docs ──▶ Ingestion ──▶ Hybrid Index (BM25 + dense + graph)          │
                              │                                        │
  query ─▶ Agent (plan→retrieve→tool→critic) ─▶ answer + citations    │
                              │                          │             │
                     tools via MCP            every trace scored online│
                              │                          │             │
                     cost-aware model router    sampled ▶ LLM-judge    │
                                                         │             │
                        mine failures ▶ new eval cases ▶ MLX-LoRA ─────┘
                              retrain reranker/router ▶ shadow ▶ promote-on-lift
```

Full detail: [`docs/01-architecture.md`](docs/01-architecture.md).

## Repository layout (target)

```
self-improving-agent-platform/
├── README.md
├── .gitignore
├── requirements.txt
├── docs/
│   ├── 00-what-it-is.md      # problem, goals, success criteria
│   ├── 01-architecture.md    # subsystems, data flow, interfaces
│   ├── 02-build-plan.md      # ← phased, step-by-step milestones (build from here)
│   └── 03-setup.md           # env, models, keys, first run
├── src/                      # (created in Milestone 0)
├── eval/                     # golden sets, judge configs, CI gate
├── configs/                  # versioned agent/retrieval/router configs
└── data/                     # gitignored; fetched by scripts
```

## Quickstart (once Milestone 0 is done)

```bash
# 1. Env (personal conda env — never base; see ~/.claude/CLAUDE.md)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate personal
pip install -r requirements.txt

# 2. Local model for the cheap tier (llama.cpp + a small GGUF)
#    see docs/03-setup.md for the exact model + download command

# 3. Cloud creds (Claude / Bedrock) come from the global ~/.env
set -a; source ~/.env; set +a

# 4. Run the API + console (target)
make dev
```

## How to build it

Read the docs in order, then work **milestone by milestone** through
[`docs/02-build-plan.md`](docs/02-build-plan.md). Each milestone is independently
demoable and ends with a concrete artifact. Do **not** try to build all subsystems at
once — the flywheel only makes sense once a basic agent + eval exist.

## Tech stack

Python 3.12 · Claude (Anthropic / Bedrock) · llama.cpp + MLX (local models) · FAISS +
rank-bm25 · networkx (GraphRAG) · MCP · FastAPI + async workers · Redis · SQLite/Postgres
· Next.js (admin console) · Prometheus/Grafana · DVC/MLflow · GitHub Actions · Docker.

## License

Private. All rights reserved (personal portfolio project).
