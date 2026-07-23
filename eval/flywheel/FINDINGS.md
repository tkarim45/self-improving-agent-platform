# M5 (stage 1) — the flywheel closes, and its first act is to stop wasting money

Artifacts: [`configs/promotions.jsonl`](../../configs/promotions.jsonl) (the promotion log —
one rejected cycle, one promoted), [`configs/active.json`](../../configs/active.json) (what
production serves via `--router active`). Run 2026-07-23, real AWS Bedrock, total flywheel
spend **$1.67** across 66 traced requests.

Stage 1 closes the loop on the cheapest trainable component — the **router**, distilled from
observed outcomes per the plan ("distill the router policy into a small local classifier").
The MLX-LoRA reranker is stage 2. The loop that ran, fully automated per cycle:

```
traces (SQLite) -> mine (failure modes, hash train/holdout split)
                -> train candidate (refuses < 8 examples; declares degenerate data)
                -> shadow (replay: price both routers against OBSERVED per-tier outcomes)
                -> canary (frozen M4 golden gate — the execution-oracle reward-hacking guard)
                -> decide (promote only on dominance) -> promotion log -> active config
```

## The two cycles

**Cycle 1 — REJECTED, correctly.** 38 organic-style queries ($0.60): 36/38 grounded on the
tier chosen, 2 escalations — and both escalations *failed on strong too* (retrieval failures,
not routing ones), so the mined dataset contained **zero "strong was needed" labels**. The
trained candidate degenerated to always-cheap, and on the holdout split the heuristic had
chosen cheap everywhere anyway: identical quality, identical cost, `REJECT: no measured lift`.
The flywheel declined to churn configs without evidence. That rejection is logged, because a
flywheel that only records its wins is marketing.

**Cycle 2 — PROMOTED.** The heuristic's known weakness (M2: reasoning-worded queries route
strong) was priced with a live shadow A/B: 14 reasoning-worded but doc-answerable questions
("compare VARCHAR and TEXT...", "why does ORDER BY not persist...") run under BOTH routers:

| arm | routed | grounded | cost |
|---|---|---|---|
| incumbent heuristic | 14/14 strong | 11/14 | **$0.8747** |
| candidate always-cheap | 14/14 cheap | **14/14** | **$0.1920** |

With both tiers now observed, cycle 2's shadow priced the holdout at quality 100% -> 100% and
cost **-25%**: `PROMOTE`. The active config flipped to the learned router; `--router active`
serves it; a rollback drill restored the heuristic and a roll-forward re-applied the winner.

## What was actually learned (honest version)

**The first promotion is a demotion.** The candidate router is *degenerate* — a constant
always-cheap policy, and it says so in its own routing reason ("training data contained only
'cheap' outcomes"). It is not intelligence; it is the flywheel discovering, from observed
outcomes, that the incumbent heuristic's strong-routing bought nothing measurable — the same
conclusion M2 reached manually ($0.2525 vs $0.0937, 2.7×), now reached *automatically* and
acted on with a guarded promotion. Killing measured waste is a legitimate first act for a
self-improving system, and it is exactly what the data supported — nothing more.

**Sonnet grounded *worse* than Haiku on the shadow batch** (11/14 vs 14/14). Consistent with
M2's caveat from the other side: the strong tier writes fuller answers carrying more uncited
claims, and with thinking disabled it reaches for the search tool less readily. Grounding is
not correctness — but on the only quality signal the system has, cheap won outright.

**"Strong is needed" evidence remains absent.** Two escalations fired in 66 requests and both
failed on strong as well. Until traffic contains queries where cheap fails *and* strong
rescues, the learned router has nothing to learn beyond always-cheap — and the mining
deliberately refuses to treat "routed strong directly and it worked" as evidence strong was
needed, because that would bake the incumbent's waste into the training data.

## Design decisions worth keeping

- **Replay-first shadow.** Both routers are priced against *observed* per-tier outcomes — no
  token is spent to evaluate a candidate. A choice with no observation is counted `unknown`,
  and any unknown blocks promotion ("price them live before promoting"). Cycle 1's rejection
  and the live A/B that followed are that rule working as designed.
- **The canary is the execution oracle.** A candidate must not regress the frozen M4 golden
  gate before any lift is even considered. A config that games soft metrics but breaks
  executable ground truth is rejected outright — the reward-hacking defense, wired in.
- **Structural contamination guard.** Train/holdout is a deterministic hash of the query, so
  the same query always lands on the same side across re-mining. Nothing evaluated was
  trained on.
- **Dominance-only promotion.** Promote only if quality holds and cost drops (or quality
  rises at no added cost). Quality-up-at-higher-cost is rejected — conservative on purpose
  for an autonomous loop; loosening it is a deliberate decision for later, not a default.
- **Degenerate models are declared, not dressed up.** Single-class training data produces a
  constant policy that names itself as such, and a dataset under 8 examples refuses to fit
  at all.

## Threats to validity

- **The shadow batch was authored to probe the heuristic's known weakness**, not sampled
  from organic traffic. That makes the 25% holdout saving a proof of *mechanism* on a real
  weakness, not an estimate of production savings. M6's usage simulator supplies the organic
  volume that turns this into a rate.
- **Holdout n = 14 priced queries.** The decision rule saw real numbers, but small ones.
- **Grounding ≠ correctness** (M4's judge and oracle exist for that; they gate the canary,
  not the shadow's quality metric, which is grounding-based).
- **The cost split for escalated traces** (cheap attempt 20% / strong attempt 80%) is the
  M2-measured ratio, applied as an approximation rather than per-trace accounting.
- Stage 2 (MLX-LoRA reranker from mined triples) and hard-case golden-set growth are built
  as data products (`hard_cases()`) but not yet trained/promoted.
