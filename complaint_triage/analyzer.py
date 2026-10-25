"""Combine model output (and the urgency rules where the model has no urgency head) into one result."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from .labels import LABEL_INFO, TASKS
from .rules import urgency_by_rules

# A prediction below this confidence is flagged for a person to confirm.
REVIEW_BELOW = 0.55


class Model(Protocol):
    kind: str
    version: str

    @property
    def tasks(self) -> list[str]: ...

    def predict_proba(self, texts: list[str]) -> dict[str, np.ndarray]: ...


class TaskResult(BaseModel):
    label: str
    name: str
    confidence: float
    source: str  # "model" or "rules"


class Analysis(BaseModel):
    aspect: TaskResult
    sentiment: TaskResult
    urgency: TaskResult
    needs_review: bool
    matched_phrases: list[str]
    model_version: str
    latency_ms: float


class Analyzer:
    def __init__(self, model: Model, review_below: float = REVIEW_BELOW):
        self.model = model
        self.review_below = review_below

    @property
    def version(self) -> str:
        return self.model.version

    def analyze_batch(self, texts: list[str]) -> list[Analysis]:
        started = time.perf_counter()
        proba = self.model.predict_proba(texts) if texts else {}
        per_item_ms = (time.perf_counter() - started) * 1000 / max(1, len(texts))
        results = []
        for i, text in enumerate(texts):
            parts: dict[str, TaskResult] = {}
            for task in TASKS:
                if task in proba:
                    p = proba[task][i]
                    k = int(np.argmax(p))
                    key = TASKS[task][k]
                    parts[task] = TaskResult(label=key, name=LABEL_INFO[task][key].name,
                                             confidence=round(float(p[k]), 4), source="model")
            rule_label, phrases = urgency_by_rules(text)
            if "urgency" not in parts:
                parts["urgency"] = TaskResult(label=rule_label, name=LABEL_INFO["urgency"][rule_label].name,
                                              confidence=1.0, source="rules")
            for task in ("aspect", "sentiment"):
                if task not in parts:
                    raise ValueError(f"model {self.version} has no {task} head")
            needs_review = any(r.source == "model" and r.confidence < self.review_below for r in parts.values())
            results.append(Analysis(**parts, needs_review=needs_review, matched_phrases=phrases,
                                    model_version=self.version, latency_ms=round(per_item_ms, 2)))
        return results

    def analyze(self, text: str) -> Analysis:
        return self.analyze_batch([text])[0]


def resolve_model_dir(model_dir: str | Path) -> Path:
    """A local directory, or ``hf://owner/repo[@revision]`` for a model published on the Hugging Face Hub
    (downloaded once into the Hub cache)."""
    value = str(model_dir)
    if not value.startswith("hf://"):
        return Path(value)
    from huggingface_hub import snapshot_download

    repo, _, revision = value[len("hf://"):].partition("@")
    return Path(snapshot_download(repo_id=repo, revision=revision or None))


def load_model(model_dir: str | Path) -> Model:
    import json

    model_dir = resolve_model_dir(model_dir)
    meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
    if meta.get("kind") == "multitask":
        from .multitask import MultiTaskModel

        return MultiTaskModel(model_dir)
    from .baseline import BaselineModel

    return BaselineModel.load(model_dir)