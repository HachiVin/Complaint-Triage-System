"""Per-task metrics on a labelled split, plus what matters operationally: critical recall, review rate,
latency and throughput."""

from __future__ import annotations

import time

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

from .data import Review
from .labels import TASKS
from .rules import urgency_by_rules


def _report(task: str, y_true: list[str], y_pred: list[str]) -> dict:
    labels = [k for k in TASKS[task] if k in set(y_true) | set(y_pred)]
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = [k for k in TASKS[task] if k in set(y_true)]
    out = {
        "n": len(y_true),
        "accuracy": round(float(np.mean(np.array(y_true) == np.array(y_pred))), 4),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)), 4),
        "per_label": {k: {"precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
                          "f1": round(float(f[i]), 4), "support": int(s[i])} for i, k in enumerate(labels)},
        "confusion": {"labels": labels, "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()},
    }
    if task == "urgency" and "kritis" in out["per_label"]:
        out["kritis_recall"] = out["per_label"]["kritis"]["recall"]
        out["kritis_precision"] = out["per_label"]["kritis"]["precision"]
    return out


def evaluate_model(model, rows: list[Review], batch_size: int = 64, review_below: float = 0.55) -> dict:
    texts = [r.text for r in rows]
    started = time.perf_counter()
    chunks = [model.predict_proba(texts[i: i + batch_size]) for i in range(0, len(texts), batch_size)]
    seconds = time.perf_counter() - started
    proba = {t: np.concatenate([c[t] for c in chunks]) for t in (chunks[0] if chunks else {})}

    single = []
    for t in texts[: min(30, len(texts))]:
        t0 = time.perf_counter()
        model.predict_proba([t])
        single.append((time.perf_counter() - t0) * 1000)

    out: dict = {"version": model.version, "n": len(rows), "tasks": {}}
    low = np.zeros(len(rows), dtype=bool)
    for task, p in proba.items():
        idx = [i for i, r in enumerate(rows) if r.label(task)]
        low |= p.max(axis=1) < review_below
        if not idx:
            continue
        pred = [TASKS[task][int(np.argmax(p[i]))] for i in idx]
        out["tasks"][task] = _report(task, [rows[i].label(task) for i in idx], pred)
    out["needs_review_rate"] = round(float(low.mean()), 4) if len(rows) else None
    out["throughput_per_s"] = round(len(rows) / seconds, 1) if seconds else None
    out["latency_ms_single"] = {"p50": round(float(np.percentile(single, 50)), 2),
                                "p95": round(float(np.percentile(single, 95)), 2)} if single else None
    return out


def evaluate_rules(rows: list[Review]) -> dict:
    idx = [r for r in rows if r.urgency]
    if not idx:
        return {"n": 0}
    return _report("urgency", [r.urgency for r in idx], [urgency_by_rules(r.text)[0] for r in idx])
