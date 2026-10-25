"""Baseline: one TF-IDF + logistic regression model per task, each trained on the rows that have that label.

Fast, small and easy to explain. It is the bar the IndoBERT model has to clear, and the fallback that
runs without torch.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .data import Review
from .labels import TASKS

FILENAME = "baseline.joblib"


def _pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)),
    ])
    return Pipeline([("features", features),
                     ("clf", LogisticRegression(max_iter=3000, class_weight="balanced", C=4.0))])


class BaselineModel:
    kind = "baseline"

    def __init__(self, pipelines: dict[str, Pipeline], version: str = "baseline"):
        self.pipelines = pipelines
        self.version = version

    @property
    def tasks(self) -> list[str]:
        return list(self.pipelines)

    @classmethod
    def train(cls, rows: list[Review], version: str = "baseline") -> BaselineModel:
        pipelines = {}
        for task in TASKS:
            labelled = [r for r in rows if r.label(task)]
            if len({r.label(task) for r in labelled}) < 2:
                continue  # nothing to learn for this task yet
            p = _pipeline()
            p.fit([r.text for r in labelled], [r.label(task) for r in labelled])
            pipelines[task] = p
        if not pipelines:
            raise ValueError("no task has at least two labels in the training data")
        return cls(pipelines, version)

    def save(self, model_dir: str | Path, meta: dict | None = None) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipelines, model_dir / FILENAME)
        info = {"kind": "baseline", "version": self.version, "tasks": self.tasks, **(meta or {})}
        (model_dir / "model_meta.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, model_dir: str | Path) -> BaselineModel:
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
        return cls(joblib.load(model_dir / FILENAME), version=f"baseline:{meta.get('version', model_dir.name)}")

    def predict_proba(self, texts: list[str]) -> dict[str, np.ndarray]:
        """Per task, probabilities in the order of ``labels.TASKS[task]`` (missing classes get 0)."""
        out = {}
        for task, p in self.pipelines.items():
            raw = p.predict_proba(texts)
            classes = list(p.classes_)
            full = np.zeros((len(texts), len(TASKS[task])), dtype=np.float32)
            for j, key in enumerate(TASKS[task]):
                if key in classes:
                    full[:, j] = raw[:, classes.index(key)]
            out[task] = full
        return out
