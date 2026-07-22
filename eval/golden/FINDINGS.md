# M4 — continuous evaluation harness

Artifacts: [`report_good.md`](report_good.md) (green), [`report_bad.md`](report_bad.md) (red),
per-case records in `records.json`. Golden set: [`duckdb.yaml`](duckdb.yaml). Run 2026-07-22,
real AWS Bedrock.

M4 is the flywheel's sensor: online scorers on every trace, an LLM-judge on a sample, a golden
eval set, and a CI gate. The domain gives this milestone something most eval harnesses do not
have — **an objective oracle.** DuckDB answers are SQL, so a golden case is checked by *running
the query*, not by asking a model whether it looks right. That is the M0 rationale realized,
and it is what lets M4 calibrate the LLM-judge against ground truth instead of a subjective
label.

## The gate flips: 92% green → 67% red

Same 12 golden cases, two system prompts:

| prompt | exec | reference | abstain | total | gate |
|---|---|---|---|---|---|
| good (grounded, cites, searches) | 8/8 | 2/3 | 1/1 | **92%** | ✅ PASS |
| bad ("you know DuckDB, don't search, citations unnecessary") | 7/8 | **0/3** | 1/1 | **67%** | ❌ FAIL |

The deliberately-worse prompt drops the gate from 92% to 67%, below the 75% threshold. That is
the artifact the milestone asks for: a red gate on a bad-prompt change, a green gate on a good
one. The replay CLI returns exit 0 green / exit 1 red, which is what the CI job keys on.

### The finding: a gate on execution alone would have missed the regression

Look at *where* the bad prompt did its damage. The execution cases barely moved (8/8 → 7/8):
the model knows basic DuckDB SQL — `unnest`, `coalesce`, `regexp_extract`, `range` — from
memory, so "don't search" costs it almost nothing there. The collapse is entirely in the
**reference (citation) cases: 2/3 → 0/3.** "Citations unnecessary" means it stops citing, and
every reference case fails.

So the two case kinds catch different regressions. Execution catches *wrong SQL*; reference
catches *ungrounded answers*. A gate built only on the objective execution oracle would have
scored the bad prompt 7/8 on exec and called it fine — missing that the prompt had destroyed
grounding. You need both, and the split confirms the M0 domain rationale from the other side:
the cases that need retrieval (QUALIFY's column semantics, PIVOT, ASOF, secrets) are exactly
the ones the bad prompt fails, while generic SQL the model has memorized is resilient.

## Judge-vs-execution calibration: 9/9 agreement, and the one disagreement was execution's bug

The LLM-judge ran alongside execution on the 9 objectively-scorable cases (8 exec + 1
abstain). After correcting one golden case, judge and execution **agree 9/9**.

The interesting part is the disagreement *before* the correction. On `g01` (top-2-by-score
via QUALIFY) the model wrote `SELECT name, score FROM t QUALIFY row_number() OVER (ORDER BY
score DESC) <= 2` — the **correct rows**, with an extra `score` column. My golden `expected`
was `[["a"], ["c"]]` (name only), so execution called it a mismatch — FAIL. The judge called
it faithful — PASS.

The judge was right. Execution was strict about a column the question never pinned down, and
the too-narrow expected value made the objective oracle reject a correct answer. This is the
honest, non-obvious calibration result: **execution is objective about "do these exact rows
match", but *what rows to expect* is a human test-design judgment that can be wrong** — and
here the LLM-judge caught the design bug the rigid oracle could not. The fix was to widen the
question ("returning name and score") and the expected value, not to trust one signal over the
other. Both are kept; their disagreements are where you look for a bug in either the answer or
the test.

This matters for M5. The flywheel there is vulnerable to a promoted config that games the
judge. The defense is exactly this pairing: a config that scores high on the judge but fails
execution is reward-hacking, and a config that fails the judge while passing execution points
at a test bug. Neither signal is trusted alone.

## The pieces

- **Online scorers** (deterministic, run on every trace, zero cost): groundedness (are cited
  ids actually retrieved), citation rate, task-success proxy, and abstention detection. A
  correct abstention counts as *success* — saying "the docs don't cover this" beats inventing
  an answer, which is the whole reason M0 picked an un-memorized corpus. These are labelled
  proxies, not truth.
- **Execution checker**: pulls SQL from an answer's code fences, runs each candidate in the
  same sandboxed in-memory DuckDB the agent uses (`enable_external_access=false`, verified in
  M2), compares rows order-insensitively. "Any candidate passes" because answers often show two
  equivalent forms. A `COPY`/`ATTACH` inside an answer is skipped, not executed.
- **LLM-judge**: Claude (strong tier) scores faithfulness / relevance / completeness 1–5 with
  chain-of-thought, against the retrieved passages *only* — it grades grounding, not its own
  opinion of DuckDB. The verdict is recomputed from the scores, never trusted from the model's
  own "verdict" field, and malformed JSON fails safe (harshest verdict) so a parse error can
  never pass a bad answer.
- **Golden set**: 12 cases (8 exec, 3 reference, 1 abstain). Honest about size — the plan
  floats 50–100, but a golden set is only worth its size if every case is genuinely checkable,
  and every `expected` here was verified by running its reference query.
- **CI gate**: replay mode (deterministic, free, no secrets) runs on every PR — it scores
  frozen answer records and catches regressions in the scoring/execution code. The live gate
  (real Bedrock + judge) is a manual/nightly job, because a quality gate that spends on every
  PR is a quality gate nobody keeps green.

## Cost

Good live run (12 cases + judge on 9): **$0.31**. Bad run (recovery disabled for speed):
similar. The online scorers and the execution oracle are free; only the judge costs tokens.

## Threats to validity

- **12 cases.** Small. The gate mechanism is proven; the *threshold* (75%) is a starting
  guess, not tuned against a distribution of good and bad configs.
- **g10 (ASOF) fails in the green run too** — a genuine retrieval miss, consistent with M1's
  finding that multi-hop/ASOF is the weak spot. Left failing rather than papered over.
- **Judge calibrated against execution, not humans.** For SQL that is arguably *better* than
  human labels (execution is objective), but the reference and free-prose parts of an answer
  are not execution-checkable, and there the judge is unaudited.
- The bad-prompt run disabled escalation and the critic for speed, so it measures the *raw*
  prompt, not the prompt-plus-recovery the good run had. That is the fair comparison for
  "does the gate react to the prompt", but it is not identical machinery on both sides.
