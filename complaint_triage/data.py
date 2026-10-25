"""Review records, JSONL I/O, de-duplication and splitting.

    {"id": "ush-000123", "app": "BRImo", "text": "...", "rating": 1, "date": "2026-03-04",
     "app_version": "2.71.0", "aspect": "kualitas_transaksi", "sentiment": "negatif",
     "urgency": "kritis", "source": "ush_brimo"}

Any of the three labels may be missing: the public dataset has aspect and sentiment, urgency exists only
for the hand-annotated subset, and freshly scraped reviews have none. Training masks missing labels.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

from .labels import TASKS


class Review(BaseModel):
    id: str
    app: str = "BRImo"
    text: str = Field(min_length=1)
    rating: int | None = Field(default=None, ge=1, le=5)
    date: str | None = None  # YYYY-MM-DD
    app_version: str | None = None
    aspect: str | None = None
    sentiment: str | None = None
    urgency: str | None = None
    source: str = "unknown"
    synthetic: bool = False

    @field_validator("aspect", "sentiment", "urgency")
    @classmethod
    def _known(cls, v, info):
        if v is not None and v not in TASKS[info.field_name]:
            raise ValueError(f"unknown {info.field_name} {v!r}; expected one of {TASKS[info.field_name]}")
        return v

    @field_validator("date")
    @classmethod
    def _iso_date(cls, v):
        if v is not None and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError(f"date must be YYYY-MM-DD, got {v!r}")
        return v

    def label(self, task: str) -> str | None:
        return getattr(self, task)


def load_jsonl(path: str | Path) -> list[Review]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(Review.model_validate_json(line))
                except ValueError as exc:
                    raise ValueError(f"{path}:{n}: {exc}") from exc
    return rows


def save_jsonl(rows: Iterable[Review], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r.model_dump(exclude_none=True), ensure_ascii=False) + "\n")


_SPACE = re.compile(r"\s+")


def normalise_text(text: str) -> str:
    return _SPACE.sub(" ", text.lower()).strip()


def deduplicate(rows: list[Review]) -> tuple[list[Review], int]:
    """Drop repeated texts. App reviews repeat a lot ("mantap", "aplikasi bagus"); keeping copies would
    put the same text in train and test."""
    seen: set[str] = set()
    out = []
    for r in rows:
        key = normalise_text(r.text)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out, len(rows) - len(out)


def split_by_time(rows: list[Review], val_frac: float = 0.15, test_frac: float = 0.15
                  ) -> dict[str, list[Review]]:
    """Oldest reviews train, newest test. Mirrors deployment, where the model scores reviews written after
    it was trained, and exposes drift (new app versions, new problems) instead of hiding it."""
    dated = sorted((r for r in rows if r.date), key=lambda r: (r.date, r.id))
    undated = [r for r in rows if not r.date]
    n = len(dated)
    n_test, n_val = int(n * test_frac), int(n * val_frac)
    train = dated[: n - n_test - n_val] + undated
    return {"train": train, "val": dated[n - n_test - n_val: n - n_test], "test": dated[n - n_test:]}


def split_stratified(rows: list[Review], task: str = "aspect", val_frac: float = 0.15, test_frac: float = 0.15,
                     seed: int = 42) -> dict[str, list[Review]]:
    """Random split stratified on one task's label (rows without that label are stratified as 'none')."""
    by_label: dict[str, list[Review]] = defaultdict(list)
    for r in rows:
        by_label[r.label(task) or "none"].append(r)
    rng = random.Random(seed)
    out: dict[str, list[Review]] = {"train": [], "val": [], "test": []}
    for key in sorted(by_label):
        group = by_label[key]
        rng.shuffle(group)
        n_test, n_val = round(len(group) * test_frac), round(len(group) * val_frac)
        out["test"] += group[:n_test]
        out["val"] += group[n_test:n_test + n_val]
        out["train"] += group[n_test + n_val:]
    for part in out.values():
        rng.shuffle(part)
    return out


def summary(rows: list[Review]) -> dict:
    dates = sorted(r.date for r in rows if r.date)
    return {
        "reviews": len(rows),
        "apps": dict(Counter(r.app for r in rows)),
        "date_range": [dates[0], dates[-1]] if dates else None,
        **{f"{task}_labelled": sum(1 for r in rows if r.label(task)) for task in TASKS},
        **{task: dict(sorted(Counter(r.label(task) for r in rows if r.label(task)).items())) for task in TASKS},
        "synthetic": sum(r.synthetic for r in rows),
    }
