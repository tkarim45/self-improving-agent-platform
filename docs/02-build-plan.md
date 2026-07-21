# 02 — Build plan (step by step, milestone by milestone)

Build in this order. Each milestone is independently demoable and ends with a concrete
artifact you could show. Do not skip ahead — the flywheel (M5) is meaningless without a
working agent (M2) and eval (M4). Rough calendar assumes solo, part-time; compress if
full-time.

Golden rule for the M1: **mock first, then local, then cloud.** Get every subsystem
working against a cheap mock, swap in a local small model, only reach for the Claude API
where reasoning quality demands it. This keeps iteration fast and memory low.

---

## Milestone 0 — Foundations (Week 1–2)

**Goal:** repo skeleton, env, chosen domain, one document ingested end-to-end.

Steps:
1. Pick the domain (financial-filing analyst **or** legal-intake triage). Commit it in
   `docs/00-what-it-is.md`.
2. Create `src/` package layout, `pyproject.toml`/`requirements.txt`, `Makefile`
   (`dev`, `test`, `lint`, `eval`), pre-commit (ruff + black), pytest skeleton.
3. Set up the `personal` conda env; verify `llama.cpp` runs a small GGUF on Metal
   (see `docs/03-setup.md`). Confirm Claude API reachable via `~/.env`.
4. Write `Retriever`, `Agent`, `Judge` interface stubs (abstract classes) — no logic yet.
5. Ingest **one** document: parse → chunk → embed → store in FAISS + BM25.

**Artifact:** `python -m src.ingest <doc>` populates an index; a smoke test retrieves a
chunk by query.

---

## Milestone 1 — Hybrid retrieval (Week 3–4)

**Goal:** solid retrieval before any agent logic.

Steps:
1. Implement BM25 (rank-bm25) and dense (sentence-transformers, a small model that fits
   memory) retrievers; fuse with Reciprocal Rank Fusion.
2. Add a cross-encoder reranker (start with an off-the-shelf small one — this becomes a
   flywheel fine-tune target later).
3. Build a **small labeled retrieval eval set** (20–50 query→relevant-chunk pairs) and
   measure Recall@k / MRR. (Reuse method from the existing `rag-architectures` repo.)
4. Add the knowledge-graph index (networkx) for multi-hop; test on a 2-hop question.

**Artifact:** retrieval eval report (Recall@k table) committed under `eval/`.

---

## Milestone 2 — The agent (Week 5–8)

**Goal:** grounded, citing agent with tools.

Steps:
1. Implement the plan→retrieve→tool→critic loop. Start with Claude as the reasoning
   model (correctness first, optimize later).
2. Add inline citations: every claim maps to a retrieved chunk id; verify no
   uncited claims slip through (a cheap post-check).
3. Stand up **MCP tools**: a read-only SQL tool over a tenant SQLite DB, a calculator,
   optionally web search. (Reuse patterns from the existing `mcp-analytics-server` repo.)
4. Add the **cost-aware router**: heuristic first (query length / has-tool-need →
   local vs Claude), measured on a small labeled difficulty set.

**Artifact:** `Agent.run(query)` returns a cited answer; a demo notebook shows 5 domain
questions answered with citations + which model tier served each.

---

## Milestone 3 — Guardrails & ops basics (Week 9–10)

**Goal:** safe boundary + traceability, before scaling usage.

Steps:
1. Wire input/output guardrails (reuse `llm-guardrails`): injection detection, PII
   redaction, unsafe-tool-call block.
2. Emit a structured **trace** per request (prompt, tokens, latency, cost, retrieval set,
   model tier) to SQLite. (Reuse `llm-observability` tracer.)
3. Add provider failover (Bedrock ↔ Anthropic ↔ local) with a dead-primary test.

**Artifact:** a trace viewer (even CLI) showing per-request cost + latency; a test
proving a prompt-injection attempt is blocked.

---

## Milestone 4 — Continuous evaluation harness (Week 11–14)

**Goal:** score every answer; gate every change. This is the flywheel's sensor.

Steps:
1. Implement **online scorers** (cheap, run on every trace): groundedness (are cited
   chunks actually retrieved?), a heuristic faithfulness check, task-success proxy.
2. Implement the **LLM-judge** (Claude, G-Eval CoT rubric) on a sampled fraction. (Reuse
   `llm-as-judge-system` / `agent-eval-harness`.)
3. **Calibrate the judge** against a small human-labeled set; record agreement; add a
   drift check that flags when judge↔human agreement falls.
4. Build the **golden eval set** (start 50–100 cases) and a **CI gate**: GitHub Action
   runs the golden eval on every PR; fails if score < threshold. Prove it with a
   deliberately-worse prompt PR that goes red.

**Artifact:** a red CI run on a bad-prompt PR + a green run on a good one; a judge
calibration report.

---

## Milestone 5 — The self-improvement flywheel (Week 15–22) — THE CORE

**Goal:** close the loop. This is the differentiator and the hardest part.

Steps:
1. **Failure mining**: cluster low-score / failed traces by failure mode (bad retrieval
   vs bad reasoning vs bad routing).
2. **Auto-generate hard eval cases** from failures; grow the golden set (guard against
   contamination — new cases must be genuinely held out).
3. **On-device fine-tune** (MLX-LoRA):
   - Retrain the **reranker** on mined (query, good-chunk, bad-chunk) triples.
   - Distill the **router** into a small local classifier from observed
     (query → which-tier-was-right) data.
   - Optionally LoRA-tune a small **embedder**.
   Keep every base ≤1.5B so it fits the M1 (see `docs/03-setup.md` for MLX commands).
4. **Shadow / A-B**: run candidate config alongside incumbent on held-out golden set;
   compute lift with confidence (reuse the `experimentation-engine` stats).
5. **Promote-on-lift**: only if the candidate significantly beats incumbent; version-bump
   the config, keep a rollback point. Log every promotion.
6. **Safety valves**: cap retrain frequency; detect judge reward-hacking (does a promoted
   config score high on the judge but low on a frozen human-labeled canary set?); auto
   roll back on canary regression.

**Artifact:** a promotion log showing at least one automated retrain→shadow→promote cycle
that improved held-out score.

---

## Milestone 6 — The improvement curve (Week 23–26) — THE HEADLINE

**Goal:** prove the system improves unattended. This is what you demo.

Steps:
1. Build a **usage simulator**: a stream of domain queries over "N weeks" (replay real
   query patterns or generate them), feeding traces into the flywheel.
2. Run the loop unattended across the simulated weeks. Log quality + cost each week.
3. Plot the **headline curve**: quality up, cost flat/down, promotion events annotated.
4. Write the honest analysis: where it improved, where it plateaued, whether the judge
   held calibration, what reward-hacking (if any) appeared and how it was caught.

**Artifact:** the before/after curve + a written finding. This is the portfolio piece.

---

## Milestone 7 — Product surface & polish (Week 27–34)

**Goal:** make it a product, not a script.

Steps:
1. Next.js **chat UI** with inline citations + confidence.
2. **Admin console**: ingest manager, eval/cost/latency/quality dashboards, promotion
   history, RBAC, audit log.
3. Multi-tenant isolation hardening.
4. `docker-compose` for the full local stack; README quickstart that actually works.
5. A 60–90s screen recording of the whole loop for the portfolio.

**Artifact:** deployed demo (local or a small cloud instance) + recording.

---

## Optional cloud burst (any time after M5)

If you want multi-tenant scale-out or a larger fine-tune base than the M1 allows: script
a single-instance cloud deploy (K8s) and/or a rented-A100 fine-tune. **Document it as
optional** — the project stands alone on the laptop without it.

---

## Milestone checklist

- [x] **M0 Foundations — done 2026-07-21.** Domain committed (DuckDB docs support agent);
      `src/` package + Makefile + ruff + pytest; the five interfaces from `docs/01`;
      heading-aware chunker; BM25 + FAISS hybrid index with per-tenant dirs, content-addressed
      chunk ids (so re-ingest is incremental) and save/load; `python -m src.corpus fetch` and
      `python -m src.ingest`. **Measured:** 407 docs → 4,556 chunks, 0.7 s with the hashing
      embedder / 65 s with MiniLM-L6-v2, 30 tests passing offline.
      *Two findings worth keeping:* (1) BM25 relevance cannot be thresholded on `score > 0` —
      Okapi IDF goes negative for any term in more than half the corpus, so a real match on a
      small index scores below zero; overlap-gate instead. (2) The docs repo ships ~7 copies of
      every page (one per release), so a fetch must pin one version or the index fills with
      near-duplicates and "the correct chunk" stops being well-defined.
- [ ] M1 Hybrid retrieval — Recall@k report
- [ ] M2 Agent — cited answers + tools + router
- [ ] M3 Guardrails & tracing
- [ ] M4 Eval harness — CI gate proven red/green
- [ ] M5 Flywheel — one automated promote cycle
- [ ] M6 Improvement curve — the headline chart
- [ ] M7 Product surface — deployed demo + recording
