"""M5 stage 2 — the on-device reranker fine-tune.

Stage 1 distilled the *router* (a TF-IDF+LogReg classifier) from observed per-tier outcomes.
Stage 2 does the other half the flywheel promised: mine (query, good-chunk, bad-chunk) triples
from the labeled retrieval eval and fine-tune the **cross-encoder reranker** on them, on-device
(Apple Silicon, Metal/MPS via PyTorch — the same laptop everything else runs on).

Why torch-on-MPS and not MLX-LoRA: the reranker is a Hugging Face cross-encoder
(ms-marco-MiniLM-L-6-v2, 22M params). MLX-LoRA would mean re-implementing the cross-encoder
in MLX; sentence-transformers already trains this exact architecture on Metal via MPS. The
milestone's *intent* — fine-tune the reranker on mined triples, on the M1, keep the base
small (22M << the 1.5B ceiling) — is met, and it actually runs to completion instead of
fighting the mlx-lm/transformers version break that the sibling capstones hit. Recorded as a
deliberate substitution, the same way M2 substituted Sonnet 4.6 for an unavailable Sonnet 5.

Mining is grounded in *real* first-stage mistakes, not synthetic pairs: for each labeled
query, a positive is a chunk from a page the labels call relevant; a hard negative is a chunk
the first stage ranked highly whose page is NOT relevant. Those are exactly the confusions a
reranker exists to fix.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # keep this module faiss-free at import time (see train_reranker docstring)
    from src.eval.retrieval import EvalQuery
    from src.retrieval.pipeline import HybridRetriever

# Fewer than this many training pairs and a fine-tune is noise, not learning — the same
# refuse-to-fit discipline the router trainer applies at 8 examples.
MIN_PAIRS = 24


@dataclass
class Triple:
    query: str
    positive: str  # chunk text from a relevant page
    negative: str  # chunk text a first stage ranked high from a NON-relevant page


@dataclass
class MiningReport:
    triples: list[Triple] = field(default_factory=list)
    queries_used: int = 0
    queries_skipped: int = 0  # no positive or no hard negative found in the fetch window

    @property
    def n_pairs(self) -> int:
        # each triple yields one positive (label 1) and one negative (label 0) pair
        return 2 * len(self.triples)


def _page_of(chunk) -> str:
    return chunk.source_path


def mine_triples(
    retriever: HybridRetriever,
    queries: list[EvalQuery],
    tenant: str,
    fetch_k: int = 50,
    max_neg_per_query: int = 2,
) -> MiningReport:
    """One or more (query, positive, hard-negative) triples per labeled query, read off the
    real first stage. A query contributes nothing if the fetch window holds no relevant chunk
    (positive missing) or is entirely relevant (no hard negative) — both are logged, not faked.
    """
    report = MiningReport()
    for q in queries:
        hits = retriever.search(q.query, tenant, k=fetch_k)
        positives = [h.chunk for h in hits if _page_of(h.chunk) in set(q.relevant)]
        # Hard negatives: highly-ranked chunks whose page the labels do NOT mark relevant.
        negatives = [h.chunk for h in hits if _page_of(h.chunk) not in set(q.relevant)]
        if not positives or not negatives:
            report.queries_skipped += 1
            continue
        report.queries_used += 1
        pos = positives[0]
        for neg in negatives[:max_neg_per_query]:
            report.triples.append(
                Triple(
                    query=q.query,
                    positive=pos.contextualized(),
                    negative=neg.contextualized(),
                )
            )
    return report


def _pairs_and_labels(triples: list[Triple]) -> tuple[list[list[str]], list[float]]:
    pairs: list[list[str]] = []
    labels: list[float] = []
    for t in triples:
        pairs.append([t.query, t.positive])
        labels.append(1.0)
        pairs.append([t.query, t.negative])
        labels.append(0.0)
    return pairs, labels


@dataclass
class TrainReport:
    out_dir: str
    base_model: str
    n_pairs: int
    epochs: int
    device: str
    refused: bool = False
    reason: str = ""


def _pick_device() -> str:
    # SIAP_TRAIN_DEVICE lets a caller force cpu (MPS has occasional op gaps); default to MPS.
    forced = os.environ.get("SIAP_TRAIN_DEVICE", "").strip().lower()
    if forced in {"cpu", "mps", "cuda"}:
        return forced
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def train_reranker(
    triples: list[Triple],
    out_dir: Path,
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 2,
    batch_size: int = 16,
    lr: float = 2e-5,
) -> TrainReport:
    """Fine-tune the cross-encoder on mined pairs and save it to `out_dir`. Returns a report;
    refuses (without writing a model) when there is too little signal to fit."""
    pairs, labels = _pairs_and_labels(triples)
    device = _pick_device()
    if len(pairs) < MIN_PAIRS:
        return TrainReport(
            out_dir=str(out_dir),
            base_model=base_model,
            n_pairs=len(pairs),
            epochs=0,
            device=device,
            refused=True,
            reason=f"only {len(pairs)} pairs (< {MIN_PAIRS}) — refusing to fit on noise",
        )

    from datasets import Dataset
    from sentence_transformers.cross_encoder import (
        CrossEncoder,
        CrossEncoderTrainer,
        CrossEncoderTrainingArguments,
    )
    from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss

    model = CrossEncoder(base_model, num_labels=1, device=device)
    ds = Dataset.from_dict(
        {
            "query": [p[0] for p in pairs],
            "passage": [p[1] for p in pairs],
            "label": labels,
        }
    )
    loss = BinaryCrossEntropyLoss(model)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = CrossEncoderTrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=0.1,
        logging_steps=1_000_000,  # quiet; this is a tiny run
        save_strategy="no",
        report_to=[],
        # The HF Trainer auto-selects MPS on Apple Silicon; use_cpu forces CPU when asked
        # (SIAP_TRAIN_DEVICE=cpu), which is the escape hatch for MPS op gaps.
        use_cpu=(device == "cpu"),
        # No forked dataloader workers and no pinned memory: faiss (mining) and torch both
        # load OpenMP, and forking a worker after both are in memory segfaults on macOS. A
        # single in-process loader over ~136 pairs costs nothing.
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )
    trainer = CrossEncoderTrainer(model=model, args=args, train_dataset=ds, loss=loss)
    trainer.train()
    model.save_pretrained(str(out_dir))
    (out_dir / "train_meta.json").write_text(
        json.dumps(
            {"base_model": base_model, "n_pairs": len(pairs), "epochs": epochs, "device": device},
            indent=2,
        ),
        encoding="utf-8",
    )
    return TrainReport(
        out_dir=str(out_dir),
        base_model=base_model,
        n_pairs=len(pairs),
        epochs=epochs,
        device=device,
    )


def train_reranker_subprocess(
    triples: list[Triple],
    out_dir: Path,
    base_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    epochs: int = 2,
    python: str = "",
) -> TrainReport:
    """Train in a fresh process that never imports faiss.

    faiss (used for mining) and torch both bring their own OpenMP runtime; with both resident
    the cross-encoder's backward pass segfaults on macOS. Isolating training in a subprocess
    that only imports datasets + sentence_transformers sidesteps it entirely, and the trained
    model is loaded back for inference-only eval where faiss+torch coexist fine (M1 proved it).
    """
    import subprocess
    import sys
    import tempfile

    pairs = _pairs_and_labels(triples)[0]
    device = _pick_device()
    if len(pairs) < MIN_PAIRS:
        return TrainReport(str(out_dir), base_model, len(pairs), 0, device, True,
                           f"only {len(pairs)} pairs (< {MIN_PAIRS}) — refusing to fit on noise")

    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump([{"query": t.query, "positive": t.positive, "negative": t.negative}
                   for t in triples], f)
        triples_path = f.name

    env = dict(os.environ)
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    proc = subprocess.run(
        [python or sys.executable, "-m", "src.flywheel.reranker_train",
         triples_path, str(out_dir), base_model, str(epochs)],
        env=env, capture_output=True, text=True,
    )
    Path(triples_path).unlink(missing_ok=True)
    if proc.returncode != 0:
        return TrainReport(str(out_dir), base_model, len(pairs), 0, device, True,
                           f"training subprocess failed (rc={proc.returncode}): "
                           f"{proc.stderr.strip()[-300:]}")
    return TrainReport(str(out_dir), base_model, len(pairs), epochs, device)


def _main(argv: list[str]) -> int:
    """Subprocess entrypoint (faiss-free): train_reranker over triples read from a JSON file.
        python -m src.flywheel.reranker_train <triples.json> <out_dir> <base_model> <epochs>
    """
    triples_path, out_dir, base_model, epochs = argv[1], argv[2], argv[3], int(argv[4])
    raw = json.loads(Path(triples_path).read_text(encoding="utf-8"))
    triples = [Triple(query=r["query"], positive=r["positive"], negative=r["negative"])
               for r in raw]
    report = train_reranker(triples, Path(out_dir), base_model=base_model, epochs=epochs)
    if report.refused:
        print(report.reason, flush=True)
        return 1
    print(f"trained {report.n_pairs} pairs, {report.epochs} epochs on {report.device} "
          f"-> {report.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(_main(sys.argv))
