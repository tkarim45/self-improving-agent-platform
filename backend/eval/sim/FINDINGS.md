# M6 — the improvement curve: six unattended weeks

![the curve](curve.png)

Artifacts: [`curve.png`](curve.png), [`weekly.json`](weekly.json) (per-week metrics),
[`promotions.jsonl`](promotions.jsonl) (every cycle's decision). Run 2026-07-23 on real AWS
Bedrock, **$1.77 for the whole six-week simulation** (72 served queries + 2 shadow samples +
6 flywheel cycles).

## What the curve shows

Six simulated weeks of DuckDB support traffic (12 queries/week, deterministic weekly samples
from a weighted pool that repeats the way real support traffic repeats), served by the agent
under whatever config the flywheel had last promoted, with one flywheel cycle at the end of
each week. **No human touched the loop between week 0 and week 5.**

| | grounded | cost/query |
|---|---|---|
| weeks 0–3 (heuristic router) | **93.8%** | **$0.0294** |
| weeks 4–5 (promoted learned router) | **87.5%** | **$0.0135** — 54% cheaper |

Endpoints: quality **92% → 92%**, cost **2.6¢ → 1.1¢**. One promotion event, at week 3,
annotated on the curve.

## The unattended narrative — the part that matters

The promotion log tells the story better than the curve does:

```
week 1 cycle: REJECT — holdout too small (3 < 5) to support a claim
week 2 cycle: REJECT — candidate made 1 choice with no observed outcome — price it live
              -> the shadow sampler then ran exactly that query under the candidate's tier
week 3 cycle: PROMOTE — quality held (100% -> 100%) at 23% lower cost
week 4 cycle: REJECT — no measured lift on either axis
week 5 cycle: REJECT — no measured lift on either axis
```

Every guard fired in sequence, unattended: too little data → wait; unpriceable choice →
spend a bounded shadow budget to price it; evidence in → promote; nothing further to gain →
decline to churn. The system gathered the evidence it needed by itself — the week-2
rejection *caused* the week-2 shadow sample that enabled the week-3 promotion.

**The escalation safety net is visible in the tier mix.** After the promotion the router
proposes cheap on everything, yet weeks 4–5 show `strong: 2` and `strong: 1` — escalation
firing when the cheap answer came back ungrounded. The learned policy is always-cheap, but
the *system* is not: the recovery path catches what the router gets wrong, and each catch is
a future training label.

## Honest analysis (the M6 step-4 questions)

**Where it improved:** cost, by 54% per query, at the promotion event, exactly as the shadow
predicted (−23% on its holdout; the production saving is larger because live traffic
contains more of the reasoning-worded queries the heuristic was over-routing).

**Where it plateaued:** immediately after. Weeks 4–5 cycles found no further lift and said
so. With an always-cheap policy in place there is nothing left for *this* component to save —
further improvement has to come from a different component (the reranker, the prompt) or
from "strong needed" evidence that has not yet appeared: both weeks-4/5 escalations failed
on strong too (one was the unanswerable-by-design question, one the genuinely hard index
question), so they teach routing nothing.

**Quality:** 93.8% → 87.5% is a drop of ~1.5 answers across 24 post-promotion queries, with
week 5 back at 11/12. At n=24 this is not resolvable from noise (a two-proportion test at
these counts is far from significance), and the frozen golden canary passed in every cycle —
but it is a *real number moving the wrong way*, reported rather than smoothed. If it
persisted at volume, the promotion should roll back; the mechanism for that exists and was
drilled in M5.

**Judge calibration:** not re-checked weekly — the weekly quality signal is the grounding
scorer plus the frozen execution-oracle canary, and the LLM-judge was not re-run per week
(cost discipline). M4's 9/9 judge-vs-oracle agreement is the standing calibration. This is a
scope cut, stated as one.

**Reward hacking:** none observed — and the canary ran in every one of the six cycles, so a
config that gamed the soft metrics while breaking executable ground truth would have been
rejected at the gate it was designed to fail.

## Threats to validity

- **Authored traffic.** The pool composition (lookup-heavy, a slice of reasoning-worded
  queries, a few hard/unanswerable) is my model of support traffic, not a log of real users.
  Weekly repetition mirrors reality; the mix proportions are assumptions.
- **12 queries/week.** Every weekly number carries ±1-answer granularity of ~8%.
- **One promotion event.** The curve demonstrates the loop, not a trend — a longer run with
  more trainable components (stage-2 reranker) is what would make it a curve rather than a
  step.
- **Grounding ≠ correctness**, as everywhere in this project; the execution oracle guards
  the canary, not the weekly traffic metric.
- The simulation state is fully isolated (`data/sim/`) and deterministic (hash-seeded
  sampling, caller-supplied timestamps): a re-run reproduces the same traffic schedule.
