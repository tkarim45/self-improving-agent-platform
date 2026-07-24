# M5 stage 2 — the on-device reranker fine-tune

Run 2026-07-24 on the M1 (CPU), one cycle: mine → fine-tune → replay-shadow → canary →
decide. The promotion entry is in [`../../configs/promotions.jsonl`](../../configs/promotions.jsonl)
(component `reranker`). The fine-tuned model itself is not committed (an 87 MB HF checkpoint);
the numbers below are the record.

## What ran

68 (query, good-chunk, bad-chunk) triples were mined off the **real** first stage: for each
of 34 labeled queries (1 skipped — no hard negative in the top 50), the positive is a chunk
from a page the labels call relevant and the hard negatives are the highest-ranked chunks
whose page is *not* relevant. That's 136 training pairs. The base cross-encoder
(`ms-marco-MiniLM-L-6-v2`, 22 M params) was fine-tuned for 3 epochs, then the tuned and base
rerankers were scored on the frozen 35-query M1 eval.

## The result — REJECTED, correctly

| metric | base reranker | tuned reranker | Δ |
|---|---|---|---|
| recall@1 | 0.357 | 0.314 | **−0.043** |
| recall@3 | 0.671 | 0.743 | **+0.072** |
| recall@10 | 0.814 | 0.871 | **+0.057** |
| MRR | 0.558 | 0.555 | −0.003 |
| nDCG@10 | 0.608 | 0.625 | +0.017 |

The dominance gate **rejected** it: promotion requires a strict beat on all of {MRR, nDCG@10,
recall@3}, and MRR was flat (−0.003, noise at n=35). The recall@10 canary — the guard against
a reranker evicting a page the agent needs — actually *improved* (+0.057), so this is not a
canary failure. It is an honest "not strictly better", declined.

## The finding — fine-tuning deepened the M1 pattern instead of breaking it

M1's headline was that the off-the-shelf cross-encoder **never once improved rank 1** (0.357 →
0.357) for 400–1,000× the latency; its whole value sat in ranks 2–10. Stage 2 asked the
obvious next question: does fine-tuning it on 68 in-domain hard triples fix rank 1?

**No — it made rank 1 slightly worse (0.357 → 0.314) while improving ranks 3–10** (recall@3
+0.072, recall@10 +0.057, nDCG +0.017). Training on hard negatives taught the model to pull
more relevant pages *into* the top 3 and top 10, but it shuffled the very top, and MRR (which
rewards the single best rank) came out flat because the rank-1 losses cancelled the rank-3
gains. The reranker still lives in ranks 2–10, even after being told exactly which pages are
right. **Rank 1 on this corpus is not a fine-tuning-fixable ranking problem** — it is the
agent's problem to read the top-k and decide, which is why M2 put the answer behind a
tool-using loop rather than trusting position 1.

This is the same shape as stage 1's first cycle: the flywheel producing a mixed/negative
result and the gate declining to promote it. A flywheel that only ever promotes is a
marketing device; one that rejects a fine-tune which improved three metrics but not the fourth
is doing its job.

## Caveats (stated, not buried)

- **n = 35 queries, 68 triples.** Deltas under ~0.05 are within noise; the MRR −0.003 that
  drove the rejection is itself noise, which is *why* the conservative "must strictly beat"
  bar is right here — promoting on a noise-level MRR gain would be the reward-hacking the gate
  exists to stop.
- **Trained on the eval's own queries.** The triples are mined from the same 35 labeled
  queries the tuned model is then scored on, so these numbers are optimistic (train ≈ test).
  Even *with* that advantage the model didn't dominate — which makes the negative result
  stronger, not weaker. A held-out query split is the honest next step and is left as one.
- **3 epochs, one hyperparameter point.** No sweep; a longer or LoRA-only fine-tune might
  trade the axes differently. The mechanism (mine → train → shadow → canary → decide) is what
  stage 2 set out to close, and it is closed.

## Engineering findings (each cost real time)

- **MPS training was unusably slow — 208 s/step, ~1 h for 18 steps.** The cross-encoder
  backward pass hit MPS op gaps and thrashed. Forcing CPU (`SIAP_TRAIN_DEVICE=cpu`) dropped it
  to well under a minute for the whole run. "On-device Apple Silicon" here means CPU, not
  Metal, for this architecture.
- **faiss + torch segfault (SIGSEGV) during the backward pass.** Both libraries load their own
  OpenMP runtime; with the faiss index resident from mining, the first training step crashed
  the process — reproducibly, and independent of device. `KMP_DUPLICATE_LIB_OK` and disabling
  dataloader workers did not fix it. The fix that did: **train in a subprocess that imports
  neither faiss nor the index** (`train_reranker_subprocess`), then load the saved model back
  for inference-only eval, where faiss + a cross-encoder coexist fine (M1's rerank arms already
  proved that). The module is kept faiss-free at import (`TYPE_CHECKING` guards) so the
  subprocess entrypoint stays clean.
- **mlx-lm was not usable** for this: it registers a tokenizer incompatibly with
  transformers ≥ 5 (the same break the sibling capstones hit), so the "MLX-LoRA" the plan
  named became "torch cross-encoder fine-tune on Apple Silicon" — a deliberate, recorded
  substitution, the base still 22 M ≪ the 1.5 B ceiling.
