"""IndoBERT with three classification heads (aspect, sentiment, urgency) trained together.

One encoder pass per review serves all three tasks, which keeps scoring 100,000 reviews cheap. The loss
on each head is masked where a review has no label for that task, so the 100k reviews with aspect and
sentiment labels and the ~800 with urgency labels train the same model. The urgency head uses class
weights and its "kritis" recall is tracked separately, because missing a critical complaint is the
expensive mistake.

Train on a GPU (notebooks/train_colab.ipynb). Inference runs on CPU.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .data import Review
from .labels import TASKS

DEFAULT_MODEL = "indobenchmark/indobert-base-p1"
IGNORE = -100


@dataclass
class TrainConfig:
    model_name: str = DEFAULT_MODEL
    epochs: int = 4
    batch_size: int = 32
    lr: float = 3e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_length: int = 128
    patience: int = 2
    seed: int = 42
    task_weights: dict[str, float] = field(default_factory=lambda: {"aspect": 1.0, "sentiment": 1.0, "urgency": 2.0})
    max_train_rows: int | None = None  # subsample for quick experiments
    fp16: bool = True  # mixed precision on CUDA (about twice as fast on a Colab T4); ignored on CPU/MPS
    log_every: int = 200  # steps between progress lines


def _torch():
    import torch

    return torch


def _device():
    torch = _torch()
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_module(encoder_name_or_path: str, tasks: list[str]):
    torch = _torch()
    from transformers import AutoModel

    class MultiTaskModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = AutoModel.from_pretrained(encoder_name_or_path)
            hidden = self.encoder.config.hidden_size
            self.dropout = torch.nn.Dropout(0.1)
            self.heads = torch.nn.ModuleDict({t: torch.nn.Linear(hidden, len(TASKS[t])) for t in tasks})

        def forward(self, input_ids, attention_mask):
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            # mean pooling over real tokens; more stable than [CLS] for short, informal reviews
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            pooled = self.dropout(pooled)
            return {t: head(pooled) for t, head in self.heads.items()}

    return MultiTaskModel()


def _labels(rows: list[Review], task: str) -> list[int]:
    return [TASKS[task].index(r.label(task)) if r.label(task) else IGNORE for r in rows]


def _class_weights(labels: list[int], n: int) -> np.ndarray:
    counts = np.bincount([x for x in labels if x != IGNORE], minlength=n).astype(np.float64)
    w = np.ones(n)
    present = counts > 0
    if present.any():
        w[present] = counts[present].sum() / (present.sum() * counts[present])
    return w


def task_metrics(task: str, y_true: list[int], y_pred: list[int]) -> dict:
    from sklearn.metrics import f1_score, recall_score

    pairs = [(t, p) for t, p in zip(y_true, y_pred, strict=True) if t != IGNORE]
    if not pairs:
        return {"n": 0}
    t, p = zip(*pairs, strict=True)
    present = sorted(set(t))
    out = {"n": len(t), "macro_f1": round(float(f1_score(t, p, labels=present, average="macro", zero_division=0)), 4),
           "accuracy": round(float(np.mean(np.array(t) == np.array(p))), 4)}
    if task == "urgency" and TASKS[task].index("kritis") in present:
        k = TASKS[task].index("kritis")
        out["kritis_recall"] = round(float(recall_score(t, p, labels=[k], average="macro", zero_division=0)), 4)
    return out


def train_multitask(train: list[Review], val: list[Review], out_dir: str | Path, config: TrainConfig | None = None,
                    log=print) -> dict:
    torch = _torch()
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer, get_linear_schedule_with_warmup

    cfg = config or TrainConfig()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = _device()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if cfg.max_train_rows and len(train) > cfg.max_train_rows:
        rng = random.Random(cfg.seed)
        # keep every urgency-labelled row; it is the scarce one
        scarce = [r for r in train if r.urgency]
        rest = [r for r in train if not r.urgency]
        train = scarce + rng.sample(rest, max(0, cfg.max_train_rows - len(scarce)))

    tasks = [t for t in TASKS if len({r.label(t) for r in train if r.label(t)}) >= 2]
    if not tasks:
        raise ValueError("no task has at least two labels in the training data")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = build_module(cfg.model_name, tasks).to(device)

    def encode(rows: list[Review]):
        enc = tokenizer([r.text for r in rows], truncation=True, max_length=cfg.max_length)
        labels = {t: _labels(rows, t) for t in tasks}
        return [{"ids": enc["input_ids"][i], **{t: labels[t][i] for t in tasks}} for i in range(len(rows))]

    def collate(batch):
        width = max(len(b["ids"]) for b in batch)
        ids = torch.full((len(batch), width), tokenizer.pad_token_id or 0, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, b in enumerate(batch):
            ids[i, : len(b["ids"])] = torch.tensor(b["ids"])
            mask[i, : len(b["ids"])] = 1
        return ids, mask, {t: torch.tensor([b[t] for b in batch]) for t in tasks}

    train_ds, val_ds = encode(train), encode(val)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size * 2, collate_fn=collate)

    loss_fns = {
        t: torch.nn.CrossEntropyLoss(
            weight=torch.tensor(_class_weights([d[t] for d in train_ds], len(TASKS[t])), dtype=torch.float,
                                device=device),
            ignore_index=IGNORE)
        for t in tasks
    }
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": model.heads.parameters(), "lr": cfg.head_lr, "weight_decay": 0.0},
    ])
    total = max(1, len(train_dl) * cfg.epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(cfg.warmup_ratio * total), total)

    use_amp = cfg.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def autocast():
        return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp)

    def evaluate() -> dict:
        model.eval()
        preds = {t: [] for t in tasks}
        truth = {t: [] for t in tasks}
        with torch.inference_mode(), autocast():
            for ids, mask, labels in val_dl:
                logits = model(ids.to(device), mask.to(device))
                for t in tasks:
                    preds[t] += logits[t].argmax(-1).cpu().tolist()
                    truth[t] += labels[t].tolist()
        return {t: task_metrics(t, truth[t], preds[t]) for t in tasks}

    def score(metrics: dict) -> float:
        # selection score: mean macro-F1 over tasks with validation labels, plus kritis recall if present
        vals = [m["macro_f1"] for m in metrics.values() if m.get("n")]
        if "urgency" in metrics and "kritis_recall" in metrics["urgency"]:
            vals.append(metrics["urgency"]["kritis_recall"])
        return float(np.mean(vals)) if vals else 0.0

    log(f"training on {device} ({'fp16' if use_amp else 'fp32'}): {len(train_ds):,} rows, "
        f"{len(train_dl):,} steps per epoch, tasks {tasks}")
    best, best_epoch, bad, history = -1.0, 0, 0, []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        started, running = time.time(), 0.0
        for step, (ids, mask, labels) in enumerate(train_dl, 1):
            with autocast():
                logits = model(ids.to(device), mask.to(device))
                loss = 0.0
                for t in tasks:
                    y = labels[t].to(device)
                    if (y != IGNORE).any():
                        loss = loss + cfg.task_weights.get(t, 1.0) * loss_fns[t](logits[t].float(), y)
            if not isinstance(loss, float):
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                running += loss.item()
            optimizer.zero_grad()
            if cfg.log_every and step % cfg.log_every == 0:
                elapsed = time.time() - started
                eta = elapsed / step * (len(train_dl) - step)
                log(f"epoch {epoch} step {step}/{len(train_dl)} loss {running / step:.4f} "
                    f"({elapsed / 60:.1f} min, about {eta / 60:.1f} min left in this epoch)")
        metrics = evaluate()
        s = score(metrics)
        row = {"epoch": epoch, "train_loss": round(running / max(1, len(train_dl)), 4), "val": metrics,
               "selection_score": round(s, 4), "seconds": round(time.time() - started, 1)}
        history.append(row)
        log(json.dumps(row))
        if s > best:
            best, best_epoch, bad = s, epoch, 0
            model.encoder.save_pretrained(out_dir / "encoder")
            tokenizer.save_pretrained(out_dir / "encoder")
            torch.save(model.heads.state_dict(), out_dir / "heads.pt")
        else:
            bad += 1
            if bad >= cfg.patience:
                log(f"early stop after {cfg.patience} epochs without improvement")
                break

    meta = {"kind": "multitask", "version": out_dir.name, "tasks": tasks, "labels": {t: list(TASKS[t]) for t in tasks},
            "best_epoch": best_epoch, "best_selection_score": round(best, 4), "history": history,
            "config": asdict(cfg), "train_size": len(train), "val_size": len(val)}
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


class MultiTaskModel:
    """CPU inference wrapper with the same ``predict_proba`` contract as ``BaselineModel``."""

    kind = "multitask"

    def __init__(self, model_dir: str | Path, max_length: int = 128):
        torch = _torch()
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "model_meta.json").read_text(encoding="utf-8"))
        for t, labels in meta["labels"].items():
            if labels != list(TASKS[t]):
                raise ValueError(f"checkpoint labels for {t} do not match the current label set")
        self.torch = torch
        self.tasks = meta["tasks"]
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir / "encoder")
        self.module = build_module(str(model_dir / "encoder"), self.tasks)
        self.module.heads.load_state_dict(torch.load(model_dir / "heads.pt", map_location="cpu"))
        # GPU when there is one (batch scoring in Colab); CPU otherwise (the API).
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.module.to(self.device).eval()
        self.max_length = max_length
        self.version = f"multitask:{meta.get('version', model_dir.name)}"

    def predict_proba(self, texts: list[str], batch_size: int = 64) -> dict[str, np.ndarray]:
        torch = self.torch
        out = {t: [] for t in self.tasks}
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                enc = self.tokenizer(texts[i: i + batch_size], truncation=True, max_length=self.max_length,
                                     padding=True, return_tensors="pt")
                logits = self.module(enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device))
                for t in self.tasks:
                    out[t].append(torch.softmax(logits[t].float(), -1).cpu().numpy())
        return {t: (np.concatenate(v) if v else np.zeros((0, len(TASKS[t])))).astype(np.float32)
                for t, v in out.items()}