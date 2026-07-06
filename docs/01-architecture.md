# 01 — Architecture

## Subsystems

### 1. Ingestion & indexing
- Multi-format parsing (PDF / HTML / tables / code). Layout-aware chunking.
- **Hybrid index**: BM25 (rank-bm25) + dense (sentence-transformers → FAISS) + a
  **knowledge graph** (networkx) for multi-hop questions.
- Incremental re-index on document change. Per-tenant namespace isolation.

### 2. Agentic reasoning layer
- Loop: **plan → retrieve → tool-call → critic → answer**.
- Tools exposed via **MCP** (SQL over a tenant DB, web search, calculator, internal API).
- **Cost-aware model router**: cheap local model (Qwen2.5-1.5B) for easy queries,
  Claude for hard ones. Router is one of the models the flywheel improves.
- Every answer carries inline citations back to source chunks.

### 3. Continuous evaluation harness (the flywheel input)
- **Online scoring** on every trace: faithfulness, groundedness, task-success (cheap
  heuristic + local-model scorers).
- **Sampled deep eval**: a fraction of traces judged by a **calibrated LLM-judge**
  (Claude), periodically re-calibrated against a small human-labeled set.
- **CI regression gate**: any prompt / model / retrieval change runs the golden eval
  set; merge blocked if score drops below threshold.

### 4. Self-improvement loop (the research core)
```
low-score / failed traces
      │ mine + cluster failure modes
      ▼
auto-generate hard eval cases  ──▶  grow golden set
      │
      ▼
MLX-LoRA fine-tune small reranker/embedder  +  distill router policy → local classifier
      │
      ▼
shadow / A-B the new config against incumbent on the golden set
      │
      ▼
promote ONLY on measured lift  ──▶  version bump + rollback point
```
This is where the on-device training happens: the reranker/router are ≤1.5B, small
enough to MLX-LoRA-tune on the M1; the heavy reasoning model is never retrained.

### 5. Product surface
- **Chat UI** (Next.js): answer + inline citations + confidence.
- **Admin console**: ingest manager; eval dashboards; cost / latency / quality trend
  charts; promotion history; RBAC; audit log.

### 6. Ops
- Containerized services; request tracing; per-tenant cost metering.
- Provider failover (Bedrock ↔ Anthropic ↔ local).
- **Guardrails** at the boundary: prompt-injection detection, PII redaction, unsafe
  tool-call blocking. (Reuse the existing `llm-guardrails` repo.)

## Data flow (request path)

```
user query
  → guardrails (input)
  → router picks model tier
  → agent: plan → hybrid retrieve → (tool calls) → critic
  → answer + citations
  → guardrails (output)
  → response to user
  ↳ trace written (prompt, tokens, latency, cost, retrieval set, scores)
  ↳ online scorers run
  ↳ sample → LLM-judge queue
```

## Key interfaces (define these early, keep stable)

- `Retriever.search(query, tenant, k) -> [Chunk]`
- `Agent.run(query, tenant) -> Answer{text, citations, trace}`
- `Judge.score(trace) -> {faithfulness, groundedness, task_success}`
- `Trainer.improve(failures) -> CandidateConfig`
- `Promoter.evaluate(candidate, incumbent, golden) -> promote|reject`

Stable interfaces let you swap implementations (mock → local → cloud) without touching
the rest of the system — critical for building on a memory-constrained laptop.

## Build-order dependency (why the milestones are ordered as they are)

```
Ingestion+Index ─▶ Agent ─▶ Eval harness ─▶ Flywheel ─▶ Product surface ─▶ Ops
                                    ▲
                        (Flywheel is meaningless without a working Agent + Eval)
```
