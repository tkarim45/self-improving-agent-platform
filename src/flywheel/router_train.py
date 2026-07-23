"""Router distillation — the flywheel's first trainable component.

Distills observed routing outcomes into a small local classifier (TF-IDF + logistic
regression), per the plan's "distill the router policy into a small local classifier from
observed (query -> which-tier-was-right) data". Chosen as the first flywheel target because
its training signal already exists (the escalation path produces labels as a side effect),
training is instant on an M1, and `llm-router` measured the pattern working for real: on live
Bedrock traffic its learned router matched always-large quality at 71% of the cost, precisely
because it adapted to where the small model actually failed.

The classifier is deliberately tiny and inspectable. If the training data contains no
"strong" labels — cheap sufficed everywhere, which is what the M2/M4 runs observed — the
learned router degenerates to always-cheap. That is not a bug: it is the flywheel learning
that the incumbent heuristic's strong-routing was waste, and the shadow eval will price
exactly that claim.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.flywheel.mining import RouterDataset
from src.interfaces import Trainer
from src.types import CandidateConfig

MIN_EXAMPLES = 8  # below this, a fitted router is noise wearing a lab coat


class LearnedRouter:
    """Drop-in for HeuristicRouter/AlwaysRouter: route(query) -> RoutingDecision."""

    def __init__(self, model=None, single_label: str | None = None, version: str = "learned"):
        self._model = model
        self._single = single_label
        self.version = version

    def route(self, query: str):
        from src.agent.router import RoutingDecision

        if self._single is not None:
            return RoutingDecision(
                tier=self._single,
                reason=f"learned({self.version}): degenerate — training data contained "
                f"only '{self._single}' outcomes",
                score=0.0,
            )
        proba = self._model.predict_proba([query])[0]
        classes = list(self._model.classes_)
        p_strong = proba[classes.index("strong")] if "strong" in classes else 0.0
        tier = "strong" if p_strong >= 0.5 else "cheap"
        return RoutingDecision(
            tier=tier,
            reason=f"learned({self.version}): P(strong)={p_strong:.2f}",
            score=float(p_strong),
        )

    # --- persistence -------------------------------------------------------------------

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if self._single is not None:
            path.write_text(
                json.dumps({"kind": "single", "label": self._single, "version": self.version})
            )
            return
        import pickle

        with path.open("wb") as f:
            pickle.dump({"kind": "sklearn", "model": self._model, "version": self.version}, f)

    @classmethod
    def load(cls, path: Path) -> LearnedRouter:
        raw = path.read_bytes()
        if raw[:1] == b"{":
            data = json.loads(raw)
            return cls(single_label=data["label"], version=data["version"])
        import pickle

        data = pickle.loads(raw)  # noqa: S301 - our own artifact, path-controlled
        return cls(model=data["model"], version=data["version"])


class RouterTrainer(Trainer):
    """`Trainer.improve(failures) -> CandidateConfig` for the router component."""

    def __init__(self, out_dir: Path = Path("configs/candidates")) -> None:
        self.out_dir = out_dir

    def train(self, dataset: RouterDataset, version: str) -> tuple[LearnedRouter, dict]:
        if len(dataset) < MIN_EXAMPLES:
            raise ValueError(
                f"only {len(dataset)} labeled examples; refusing to fit below "
                f"{MIN_EXAMPLES} — a router trained on nothing is noise"
            )

        counts = dataset.label_counts
        if len(counts) == 1:
            # Single-class data: no classifier to fit. The candidate IS the constant policy,
            # stated as such rather than dressed up as a model.
            label = next(iter(counts))
            router = LearnedRouter(single_label=label, version=version)
            info = {"kind": "single", "label": label, "n": len(dataset), "labels": counts}
            return router, info

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
            LogisticRegression(max_iter=1000, class_weight="balanced"),
        )
        model.fit(dataset.queries, dataset.labels)
        router = LearnedRouter(model=model, version=version)
        info = {"kind": "sklearn", "n": len(dataset), "labels": counts}
        return router, info

    def improve(self, failures: list) -> CandidateConfig:  # interface adapter
        raise NotImplementedError("use train(dataset, version); the CLI wires mining to it")

    def to_candidate(self, router: LearnedRouter, info: dict, version: str) -> CandidateConfig:
        path = self.out_dir / f"router_{version}.bin"
        router.save(path)
        return CandidateConfig(
            component="router",
            version=version,
            artifact_path=str(path),
            params=info,
            provenance={"trained_on": info.get("n", 0), "labels": info.get("labels", {})},
        )
