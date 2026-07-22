# Self-Improving Domain Agent Platform ("the RAG-agent OS")

> A multi-tenant platform where an organization drops in its documents + tools and a
> fleet of LLM agents answers, acts, and **measurably improves over time** through a
> closed evaluation-and-retraining loop — running on an Apple M1 (8 GB) laptop, with
> heavy reasoning on the Claude API and the parts that improve fine-tuned on-device.

**Status:** 🔨 In build — **M0–M3 of 8 done** (2026-07-22). Domain committed; ingestion, a
measured retrieval stack, a grounded tool-using agent, and a guardrail + tracing boundary all
run end to end against real AWS Bedrock; 165 offline tests pass. **No judged answer quality
until M4, no flywheel until M5.** Code lands milestone by milestone (see
[`docs/02-build-plan.md`](docs/02-build-plan.md)).

**Guardrails + tracing (M3):** input/tool/output gates (secret + PII redaction, injection
block, unsafe-SQL policy), per-request traces to SQLite, and ordered provider failover. On a
live run every gate fired — an injection **blocked at $0.00** before reaching the model, a
pasted AWS key **redacted** before it hit the model or the trace. The finding: redaction is
load-bearing but **not semantically free** — a bare `[AWS_ACCESS_KEY]` placeholder read to the
model as literal user input and it misdiagnosed the question. Details in
[`eval/agent/M3_FINDINGS.md`](eval/agent/M3_FINDINGS.md).

**Agent today:** plan → retrieve → tool → critic, with inline citations checked against what
was actually retrieved, a sandboxed DuckDB tool the agent uses to verify its own SQL, and a
cost-aware router. On 5 real questions: **5/5 grounded, 0 invalid citations, $0.25.** The
headline finding is negative — **the router cost 2.7× an always-cheap baseline for no
measurable grounding gain**, with one question consuming 75% of the run. Details in
[`eval/agent/FINDINGS.md`](eval/agent/FINDINGS.md).

**Retrieval today:** dense + link-graph boost, **R@1 0.371 / R@10 0.843 / MRR 0.573 at ~11 ms
per query** on 35 labeled queries. Full numbers in
[`eval/retrieval/report.md`](eval/retrieval/report.md), analysis in
[`eval/retrieval/FINDINGS.md`](eval/retrieval/FINDINGS.md). Three results worth the click: a
naive equal-weight hybrid scored *worse* than dense alone; the cross-encoder reranker cost
400–1,000× the latency and never once improved rank 1; and the link graph failed at the
multi-hop job it was built for.

**Domain:** open-source support agent over the **DuckDB documentation** (411 pages,
`docs/current`, pinned by commit sha). Rationale in
[`docs/00-what-it-is.md`](docs/00-what-it-is.md) — briefly, it avoids overlapping the two
existing SEC-filing repos, and DuckDB is niche enough that Claude cannot answer from
memory, so a retrieval failure shows up as a wrong answer instead of being masked.

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

## Quickstart (working today)

```bash
# 1. Env (personal conda env, never base; see ~/.claude/CLAUDE.md)
source ~/miniconda3/etc/profile.d/conda.sh && conda activate personal
make install

# 2. Fetch the corpus (411 DuckDB doc pages -> data/corpus/duckdb, gitignored)
make corpus

# 3. Build the hybrid index (BM25 + FAISS)
make ingest

# 4. Tests: 124 offline, no network, no cloud spend
make test

# 5. Retrieval eval (add eval-full for the slow cross-encoder arms)
make eval

# 6. The agent. `agent-dry` costs nothing; `agent-demo` SPENDS on real Bedrock (~$0.25)
make agent-dry
set -a; source ~/.env; set +a && make agent-demo

# 7. View persisted request traces (cost, latency, grounding, guard actions)
make traces
```

Every agent run is bounded three ways — a per-question spend limit that raises rather than
continuing, a run-wide model-call ceiling, and a search budget. All three were added because
a run hit them, not as a precaution.

Ingest defaults to the `hashing` embedder, which needs no download and indexes the whole
corpus in **0.7 s** — that keeps the test suite and iteration fast. For real semantic
retrieval, pass a model (**65 s** for 4,556 chunks on an M1):

```bash
python -m src.ingest data/corpus/duckdb --tenant duckdb --rebuild \
  --embedder sentence-transformers/all-MiniLM-L6-v2 \
  --query "how do I filter the result of a window function"
```

Later milestones add the local GGUF cheap tier and cloud creds (see
[`docs/03-setup.md`](docs/03-setup.md)); neither is needed for the steps above.

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

MIT — see [`LICENSE`](LICENSE).

The DuckDB documentation corpus is **not** covered by this license. It is fetched at build
time from [duckdb/duckdb-web](https://github.com/duckdb/duckdb-web) under its own terms and
is never committed here.
