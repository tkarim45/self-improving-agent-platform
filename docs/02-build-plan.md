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

**DONE 2026-07-21.** [`eval/retrieval/report.md`](../eval/retrieval/report.md) (generated) +
[`eval/retrieval/FINDINGS.md`](../eval/retrieval/FINDINGS.md) (analysis). 35 labeled queries
(32 single-hop, 3 multi-hop), page-level labels, 75 tests passing.

**Shipped default: `dense` + `graph_boost=0.05`** — R@1 0.371, R@10 0.843, MRR 0.573,
nDCG@10 0.618, ~11 ms/query.

Four findings, each of which changed the default:

1. **Naive hybrid is worse than its own better leg.** Equal-weight RRF scored R@1 0.257 vs
   dense's 0.357; down-weighting BM25 to 0.3 recovered it to 0.343. RRF was not wrong, it was
   being fed a lopsided pair and treating them as equals. "Add hybrid search" is not free.
2. **The cross-encoder never fixed rank 1** — R@1 stayed at exactly 0.357 — while improving
   R@3 (0.571 → 0.671) at 400–1,000× the latency (11 ms → 4,800–11,700 ms across runs). It
   reorders the pool; it cannot conjure a page into it. `hybrid+rerank` losing to
   `dense+rerank` says the same thing: rerank quality is capped by first-stage quality.
3. **The graph did not solve multi-hop, the thing it was built for.** Coverage@10 held at
   0.667 with and without it, and at boost ≥0.2 it *halved* to 0.333 — the docs average 5.7
   links/page, so a strong boost floods the top 10 with neighbours and evicts the real second
   hop. Shipped at the swept optimum 0.05 as a mild ranking nudge, with the cliff pinned as a
   test. Reranking is worse still for multi-hop (coverage 0.667 → 0.333): a cross-encoder
   scores each chunk alone, and a bridge page rarely looks relevant alone. Joint reasoning
   over a *set* is an agent problem (M2), not a ranking one.
4. **One label was wrong and the retriever was right.** `q15` was missed by every config; the
   labeled page was a bare syntax reference while the page dense ranked 4th was the real
   answer. Widened after reading both, corrected inline in `duckdb.yaml`. On a docs corpus the
   answer is routinely split between a reference page and a concept page, so single-page
   labels systematically understate recall.

Caveats: 35 queries (differences under ~0.05 are not resolvable), only 3 multi-hop queries
(each worth 0.333 of that metric), one embedder, and latency measured on a loaded laptop.

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

**DONE 2026-07-21.** [`eval/agent/FINDINGS.md`](../eval/agent/FINDINGS.md) + raw records.
`make agent-demo`. Real AWS Bedrock; cheap = Haiku 4.5, strong = **Sonnet 4.6** (substituted:
Sonnet 5, Opus 4.8 and Opus 4.5 are all listed on this account but return 403 "not available",
and the Mantle endpoint 403s too — only the legacy InvokeModel path serves). 124 tests, all
offline via a scriptable fake provider.

**5/5 answers grounded, 0 invalid citations, $0.25 total.**

**Headline: routing to the strong tier bought nothing measurable and cost 2.7×.** Against an
always-cheap baseline, the heuristic router produced the same 5/5 grounded, same 0 invalid
citations, and the same ~50% claim-citation rate — for $0.2525 vs $0.0937. The single question
it sent to Sonnet cost **$0.189, 75% of the whole run** and 4.0× the same question on Haiku.
Honest caveat: the citation check measures *grounding*, not correctness — Sonnet wrote a much
fuller answer (25 claims vs 11) and that may be worth something this system cannot yet see.
That is M4's job. Until then always-cheap is the defensible default.

Five findings, each of which changed the code:

1. **Bounding a conversation does not bound a run.** `max_iterations` caps one conversation,
   but escalation restarts it and the critic's revision runs it again — worst case
   `max_iterations × 2 + 1 + max_iterations` calls. Hit **10 calls / $0.22 on one question**
   before the spend limit killed it. Added a run-wide `max_llm_calls` ceiling.
2. **The agent searched in circles until the money ran out** — 12 `search_docs` calls, 35,205
   input tokens, *no answer*, on a question the DuckDB docs largely cannot answer. Nothing
   told it to stop. Added a search budget that forces a decision, plus prompt language.
3. **The grounding metric under-reported on correctly-cited answers** — 0% on an answer with
   three valid citations, because models put the citation after the period (`applied.
   [403bd...]`) and the splitter stranded it. Rescored that answer **0% → 67%**. An
   under-reporting metric would have fed M5 good answers labelled as failures.
4. **The model split its answer across turns and the loop silently truncated it** — the first
   real answer began mid-sentence. A prompt fix, but an invisible failure.
5. **Bedrock IDs cannot be constructed** — the suffix convention is not uniform (Sonnet 4.6
   carries no date suffix, Haiku 4.5 does), and a plausible constructed ID 400s. Read them
   from `list_inference_profiles()`.

Caveats: 5 questions is directional only; grounding is not correctness; the escalation path
saw **zero** live escalations because the cheap tier never failed, so it is untested on real
traffic.

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

**DONE 2026-07-22.** [`eval/agent/M3_FINDINGS.md`](../eval/agent/M3_FINDINGS.md). Trace viewer
`python -m src.ops traces|summary|show`. Input/tool/output guardrails (secret + PII redaction,
signature-based injection block, unsafe-SQL policy gate), per-request traces to SQLite (flat
columns + loss-free JSON), and ordered provider failover with a dead-primary test. 165 tests.

Live run (5 requests, real Bedrock, $0.11): every gate fired — injection **blocked at $0.00**
(never reached the model, tier=none, still traced for mining), a pasted AWS key **redacted**
before it reached the model or the trace, three clean questions grounded on the cheap tier.

**The finding: redaction is not semantically free.** The secret question reached the model as
"...KEY_ID '[AWS_ACCESS_KEY]' ... why does read_parquet 401?", and the model confidently
misdiagnosed it — "you are passing the placeholder text literally" — answering a question the
user never asked, created by the redaction. The secret genuinely never leaks (verified absent
from the SQLite row including the JSON payload), but the placeholder reads to the model as
literal user input. Load-bearing yet not free; the fix (signal that a redaction occurred) is
an M4/M5 change once the judge can measure whether it helps.

Design lines worth keeping: **IPv4 is deliberately not redacted** (a DuckDB question is full
of host addresses; redact what is a secret, not what looks like a number), `SET`/`PRAGMA`/SQL
comments are **not** injection signals (a technical corpus makes an over-eager detector worse
than none), a `SpendLimitExceeded` is **not** failed over (the caller's budget, not a provider
fault), and a plain bug propagates rather than being masked by trying the next provider.

Caveats: the injection detector is signature-based (novel phrasings pass; indirect injection
through retrieved content is undefended, low-risk only because the corpus is trusted docs),
and the redaction-semantics finding rests on one live example.

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

**DONE 2026-07-22.** [`eval/golden/FINDINGS.md`](../eval/golden/FINDINGS.md). Online scorers
(groundedness, citation rate, task-success, abstention — deterministic, every trace), an
**execution-based objective oracle** (SQL answers checked by running them), an LLM-judge
(faithfulness/relevance/completeness, verdict recomputed from scores, malformed JSON fails
safe), a 12-case golden set (8 exec / 3 reference / 1 abstain, every `expected` verified by
running it), and a CI gate (replay mode = deterministic + free on every PR; live mode = real
Bedrock + judge, manual). 28 new tests (193 total).

**The gate flips: good prompt 92% GREEN → bad prompt 67% RED** (same 12 cases, threshold 75%;
replay CLI exits 0/1). CI is PR-only + manual dispatch, deliberately not on push to main.

Two findings:

1. **A gate on execution alone would have missed the regression.** The bad prompt
   ("you know DuckDB, don't search, citations unnecessary") barely moved the exec cases
   (8/8 → 7/8) — the model knows basic SQL from memory — but collapsed the citation cases
   (**reference 2/3 → 0/3**). Execution catches wrong SQL; reference catches ungrounded
   answers; you need both. The split confirms the M0 rationale from the other side: the
   DuckDB-specific cases that need retrieval are exactly the ones the bad prompt fails.
2. **Judge-vs-execution agreement 9/9, and the one disagreement was execution's test bug.**
   On g01 the model returned the correct rows with an extra column; the too-strict `expected`
   made execution FAIL while the judge (correctly) PASSED. Execution is objective about "do
   these rows match", but *what rows to expect* is a human judgment that can be wrong — and
   the judge caught it. Both signals are kept; a config that games the judge but fails
   execution is the M5 reward-hacking this pairing defends against.

Caveats: 12 cases (mechanism proven, threshold is a starting guess); the ASOF case fails green
too (a real retrieval miss, left honest); the judge is calibrated against execution, not
humans (better than humans for SQL, unaudited on free prose).

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

**STAGE 1 DONE 2026-07-23** (router distillation; MLX-LoRA reranker is stage 2).
[`eval/flywheel/FINDINGS.md`](../eval/flywheel/FINDINGS.md); the promotion log itself is
[`configs/promotions.jsonl`](../configs/promotions.jsonl). Full loop, automated per cycle:
traces → mine (failure modes, hash train/holdout split) → train → shadow (replay against
*observed* per-tier outcomes, zero eval spend) → canary (the frozen M4 execution-oracle gate —
the reward-hacking guard) → promote-on-dominance → active config → `--router active`.
Rollback drill run and verified. 23 new tests (216 total). Flywheel spend $1.67 / 66 requests.

**Cycle 1 REJECTED, correctly** — 38 organic-style queries produced zero "strong was needed"
labels (both escalations failed on strong too: retrieval failures, not routing), and the
holdout showed no lift. Logged, because a flywheel that only records wins is marketing.

**Cycle 2 PROMOTED** — a live shadow A/B priced the heuristic's known weakness (14
reasoning-worded but doc-answerable queries, both arms): heuristic routed **14/14 strong,
$0.8747, 11/14 grounded**; always-cheap **$0.1920, 14/14 grounded**. Holdout: quality
100%→100% at **−25% cost** → promoted. **The first promotion is a demotion**: the candidate is
a *declared-degenerate* constant policy that names itself as such — the flywheel reaching M2's
manual conclusion (the router's strong-routing was waste) automatically, and acting on it
under guards. Also: **Sonnet grounded worse than Haiku** on the batch (fuller answers, more
uncited claims, thinking-off tool reluctance).

Design lines: replay-first shadow (unknown choices block promotion — "price them live");
canary = execution oracle, checked before lift; structural hash-split contamination guard;
dominance-only promotion; degenerate models declared, not dressed up; sub-8-example datasets
refuse to fit.

Caveats: the shadow batch was *authored* to probe the heuristic's weakness (proof of
mechanism, not a production-savings rate — M6's simulator supplies organic volume); holdout
n=14; grounding ≠ correctness; escalated-trace cost split (20/80) is the M2 ratio applied as
an approximation; stage 2 (reranker fine-tune, hard-case golden growth) not yet trained.

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
- [x] **M1 Hybrid retrieval — done 2026-07-21.** RRF fusion, cross-encoder reranker, link
      graph, 35-query labeled eval, from-scratch Recall@k/MRR/nDCG. Default = dense +
      graph_boost 0.05 (R@1 0.371, R@10 0.843, MRR 0.573, ~11 ms). 75 tests.
- [x] **M2 Agent — done 2026-07-21.** plan→retrieve→tool→critic loop on real Bedrock, inline
      citations with a grounding post-check, sandboxed DuckDB SQL tool, cost-aware router +
      escalation. 5/5 grounded, 0 invalid citations, $0.25. Router costs 2.7× always-cheap
      for no measurable grounding gain. 124 tests.
- [x] **M3 Guardrails & tracing — done 2026-07-22.** Input/tool/output guardrails (secret+PII
      redaction, injection block, unsafe-SQL gate), SQLite trace store + CLI viewer, ordered
      provider failover with a dead-primary test. Live run: every gate fired, $0.11, 165 tests.
      Finding: redaction protects the secret but can make the model misdiagnose the request.
- [x] **M4 Eval harness — done 2026-07-22.** Online scorers + execution oracle + LLM-judge +
      12-case golden set + CI gate (PR-only). Gate flips 92% green → 67% red on a bad prompt.
      Judge-vs-execution 9/9 (the one disagreement was a too-strict golden case, judge caught
      it). Finding: an execution-only gate misses grounding regressions; you need citation
      cases too. 193 tests.
- [x] **M5 Flywheel (stage 1) — done 2026-07-23.** Router distillation loop closed: mine →
      train → shadow → canary → promote, all automated. Cycle 1 rejected (no evidence),
      cycle 2 promoted (quality held, −25% cost, live-shadow-priced). First promotion = a
      declared-degenerate always-cheap policy: the flywheel killing M2's measured router
      waste automatically. Rollback verified. Stage 2 (MLX reranker) pending. 216 tests.
- [ ] M6 Improvement curve — the headline chart
- [ ] M7 Product surface — deployed demo + recording
