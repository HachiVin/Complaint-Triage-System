"""Train the TF-IDF baseline (one model per task) and report validation metrics.

    python scripts/train_baseline.py data/processed models/baseline --version v1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from complaint_triage.baseline import BaselineModel  # noqa: E402
from complaint_triage.data import load_jsonl  # noqa: E402
from complaint_triage.evaluation import evaluate_model  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--version", default=None)
    args = ap.parse_args()

    data = Path(args.data_dir)
    train, val = load_jsonl(data / "train.jsonl"), load_jsonl(data / "val.jsonl")
    synthetic = any(r.synthetic for r in train)
    version = args.version or ("demo-synthetic" if synthetic else Path(args.out_dir).name)
    model = BaselineModel.train(train, version=version)
    model.save(args.out_dir, meta={"train_size": len(train), "synthetic_data": synthetic})
    metrics = evaluate_model(BaselineModel.load(args.out_dir), val)
    (Path(args.out_dir) / "val_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    for task, m in metrics["tasks"].items():
        extra = f", kritis recall {m['kritis_recall']}" if "kritis_recall" in m else ""
        print(f"{task:9} macro-F1 {m['macro_f1']:.3f} (n={m['n']}){extra}")


if __name__ == "__main__":
    main()
