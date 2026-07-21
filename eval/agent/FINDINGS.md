# M2 agent — what the first real run says

Raw records: [`m2_demo.json`](m2_demo.json) (heuristic router) and
[`m2_always_cheap.json`](m2_always_cheap.json) (baseline). Regenerate with
`make agent-demo`.

Setup: 5 DuckDB questions, real AWS Bedrock, cheap = Haiku 4.5 ($1/$5 per MTok), strong =
Sonnet 4.6 ($3/$15). Retrieval is M1's shipped default (dense + graph boost 0.05). Thinking
disabled on both tiers so the router compares one variable. Run 2026-07-21.

## Headline

**Routing to the strong tier bought nothing measurable and cost 2.7×.**

| | heuristic router | always-cheap |
|---|---|---|
| grounded answers | 5/5 | 5/5 |
| invalid citations | **0** | **0** |
| claims carrying a citation | 27/53 (51%) | 17/34 (50%) |
| total cost | **$0.2525** | **$0.0937** |
| mean latency | 22.4 s | 11.6 s |

The heuristic sent one of five questions to Sonnet. That single question cost **$0.189 — 75%
of the entire run**, and 4.0× what the same question cost on Haiku ($0.047). On the only
quality signal M2 has, it returned the same result: both grounded, both zero invalid
citations.

## What that does and does not prove

It does **not** prove Sonnet is no better. The citation check measures *grounding*, not
*correctness* — whether claims are attributed to retrieved passages, not whether they are
right. Sonnet wrote a substantially fuller answer on that question (25 claim sentences to
Haiku's 11), and a fuller correct answer is worth something the checker cannot see.

What it proves is narrower and still useful: **the router is spending 2.7× for a benefit
this system cannot currently detect.** That is the honest state of it, and it is precisely
what M4's LLM-judge exists to settle. Until then, `always-cheap` is the defensible default
and the heuristic router is unproven.

Note also the router never *escalated*. Zero escalations across both runs, because the cheap
tier produced a grounded answer every time. The escalation path — the half that reacts to
measured failure rather than guessing — got no exercise here. A 5-question demo where the
cheap model never fails cannot evaluate it.

## Findings

### 1. Bounding one conversation does not bound a run

`max_iterations=6` caps a single conversation. It does not cap the run: escalation restarts
the loop and the critic's revision pass runs it a third time, so the worst case is
`max_iterations × 2 + 1 + max_iterations` model calls. Observed live at **10 calls and $0.22
on a single question** before the spend limit killed it. Fixed with a run-wide
`max_llm_calls` ceiling checked in every phase, and pinned as a test.

The spend limit is what caught this, and it caught it three separate times during
development. A per-question cost ceiling that *raises* rather than continuing is the control
that made this milestone safe to iterate on.

### 2. The agent searched in circles until the money ran out

Asked "why would a hash join be slower than a merge join here", the strong tier issued **12
`search_docs` calls across 6 turns, consumed 35,205 input tokens, and produced no answer at
all.** Nothing in the loop told it that repeated searching was futile, so it kept going.

Two fixes: a search-call budget on the tool that returns an explicit "stop searching and
answer with what you have — saying the docs do not cover it is a valid answer", and prompt
language capping searches at about three. After both, the same question completes.

The underlying cause is worth keeping: the question is largely **not answerable from the
DuckDB docs**. The correct behaviour is to say so quickly, and an agent with no stopping rule
will instead spend the budget discovering that.

### 3. The grounding metric was under-reporting on correctly-cited answers

The checker reported **0% citation rate on an answer carrying three valid citations.** Models
place the citation after the closing period — `...have already been applied. [403bd30b848d]`
— which is exactly what the prompt asks for. The sentence splitter broke on `. [`, stranding
the citation as its own fragment and marking the claim it supported as uncited.

Fixed by normalizing a trailing citation to just before the period so it stays attached, then
segmenting normally. Rescoring the saved answers moved that question from **0% → 67%**; two
other answers were unaffected.

An under-reporting metric is as dangerous as an over-reporting one, and less likely to be
noticed. This one would have fed M5's flywheel a stream of well-grounded answers labelled as
failures — training the system to fix what was never broken.

### 4. The model split its answer across turns, and the loop silently truncated it

The first real answer began mid-thought: *"In this example, `QUALIFY rn = 1` filters..."*. The
model had emitted the opening of its answer alongside a tool call, then continued after the
result. A loop that returns only the final turn drops everything before it.

This is a prompting problem, not a loop problem — the fix is an explicit instruction to write
the complete answer in one final message. Worth recording because the failure is invisible:
the answer looks complete, just oddly abrupt.

### 5. The SQL tool gets used, and the sandbox holds

The agent called `run_sql` unprompted on 3 of 5 questions to check syntax before recommending
it. That is the behaviour the executable-answer corpus was chosen for in M0, working as
intended.

The sandbox was verified rather than assumed: `enable_external_access=false` raises
`PermissionException` on `COPY ... TO`, confirmed before the tool was trusted and pinned as a
test. Model-generated SQL is untrusted input.

### 6. Bedrock model IDs cannot be constructed

`us.anthropic.claude-sonnet-5-20260514-v1:0` — a plausible ID built from the naming
convention — returns *"The provided model identifier is invalid"*. The real ID carries **no
date suffix** (`us.anthropic.claude-sonnet-5`), while Haiku 4.5 does
(`us.anthropic.claude-haiku-4-5-20251001-v1:0`). The convention is not uniform; the IDs were
read from `bedrock.list_inference_profiles()`.

Availability is a separate question from listing. Sonnet 5, Opus 4.8 and Opus 4.5 are all
*listed* on this account and all return **403 "not available for this account"**. The
Mantle (Messages-API) endpoint also 403s; only the legacy InvokeModel path serves. Hence the
strong tier is **Sonnet 4.6, substituted for the intended Sonnet 5**.

## Threats to validity

- **5 questions.** Everything above is directional. A cost ratio computed on one strong-tier
  question is one data point, not a rate.
- **Grounding ≠ correctness.** No answer here has been checked for factual accuracy. That is
  M4.
- **The escalation path is untested live** — zero escalations occurred, because the cheap
  tier never produced an ungrounded answer.
- **Citation rate around 50%** across both runs. Roughly half of claim sentences carry no
  citation. Some of that is the claim heuristic being blunt (transitions and framing get
  counted); some is real. Not separated yet.
- **One corpus, one domain, one retrieval config.**
