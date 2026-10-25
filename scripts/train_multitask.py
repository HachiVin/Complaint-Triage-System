"""Fine-tune IndoBERT with aspect, sentiment and urgency heads. Run on a GPU (Colab/Kaggle).

    python scripts/train_multitask.py data/processed models/indobert-mt
    python scripts/train_multitask.py data/processed models/indobert-mt --max-train-rows 30000   # quicker
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from complaint_triage.data import load_jsonl  # noqa: E402
from complaint_triage.multitask import DEFAULT_MODEL, TrainConfig, train_multitask  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument("--max-train-rows", type=int, default=None)
    ap.add_argument("--urgency-weight", type=float, default=2.0)
    args = ap.parse_args()

    data = Path(args.data_dir)
    cfg = TrainConfig(model_name=args.model, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                      max_length=args.max_length, max_train_rows=args.max_train_rows,
                      task_weights={"aspect": 1.0, "sentiment": 1.0, "urgency": args.urgency_weight})
    meta = train_multitask(load_jsonl(data / "train.jsonl"), load_jsonl(data / "val.jsonl"), args.out_dir, cfg)
    print(f"best epoch {meta['best_epoch']}, selection score {meta['best_selection_score']}")


if __name__ == "__main__":
    main()
