"""Train the demo baseline on the fictional reviews so the dashboard and Docker image work out of the box.

    python scripts/make_demo_model.py

Its version contains "demo", so /health reports demo_model=true and the dashboard shows a banner.
Its scores mean nothing: the demo data is generated from templates. Real numbers come from
scripts/train_baseline.py and scripts/train_multitask.py on the annotated dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from complaint_triage.baseline import BaselineModel  # noqa: E402
from complaint_triage.data import load_jsonl  # noqa: E402


def main() -> None:
    rows = load_jsonl(ROOT / "data" / "sample" / "demo_reviews.jsonl")
    model = BaselineModel.train(rows, version="demo-synthetic")
    out = ROOT / "models" / "baseline"
    model.save(out, meta={"train_size": len(rows), "synthetic_data": True,
                          "note": "demo only, trained on fictional template reviews"})
    print(f"saved {out} ({len(rows)} rows, tasks: {', '.join(model.tasks)})")


if __name__ == "__main__":
    main()
