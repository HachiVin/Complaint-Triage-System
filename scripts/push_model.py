"""Publish a trained model folder to the Hugging Face Hub, so the API can load it with MODEL_DIR=hf://...

    pip install huggingface_hub==2.3.0
    python scripts/push_model.py models/indobert-mt <hf-username>/complaint-triage-indobert

Needs a Hugging Face access token with write permission (huggingface.co/settings/tokens). The script
asks for it; it is used for this upload only and never written to disk. HF_TOKEN in the environment
also works.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

CARD = """---
language: id
library_name: transformers
base_model: indobenchmark/indobert-base-p1
datasets:
- USH BRImo Dataset (Mendeley Data, CC BY 4.0, doi:10.17632/7253k33f62.1)
tags:
- text-classification
- multi-task
- indonesian
- customer-complaints
---

# Complaint Triage System: IndoBERT multi-task

One IndoBERT encoder with three heads that label an Indonesian mobile-banking app review with its
**service aspect** (9 classes), **sentiment** (3) and **urgency** (kritis / tinggi / normal).
Code, data pipeline and evaluation: https://github.com/HachiVin/Complaint-Triage-System

This is not a plain `AutoModelForSequenceClassification` checkpoint: `encoder/` is the fine-tuned
IndoBERT, `heads.pt` holds the three linear heads, and `model_meta.json` the label order. Load it with
`complaint_triage.multitask.MultiTaskModel` from the repository above.

## Results (test set: newest 15% of reviews, 7,814 reviews)

{results}

**Urgency labels are silver (not human-verified)**: keyword rules cross-checked with an LLM. Aspect and
sentiment use the dataset's own labels. See the repository README for the method and its limits.

## Intended use

Prioritising which app reviews a customer-experience team reads first. A *kritis* prediction puts a
review in front of a person; it is not a decision about the customer.
"""


def results_table(results_path: Path) -> str:
    if not results_path.exists():
        return "(run scripts/evaluate.py first to include the numbers here)"
    data = json.loads(results_path.read_text(encoding="utf-8"))
    rows = ["| Model | Aspect macro-F1 | Sentiment macro-F1 | Urgency macro-F1 (silver) |", "|---|---|---|---|"]
    for name, m in data["sets"]["test"]["models"].items():
        t = m["tasks"]
        rows.append(f"| {name} | " + " | ".join(f"{t[k]['macro_f1']:.3f}" if k in t else "–"
                                                 for k in ("aspect", "sentiment", "urgency")) + " |")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model_dir")
    ap.add_argument("repo_id", help="<hf-username>/<model-name>")
    ap.add_argument("--results", default="docs/results.json")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import HfApi

    model_dir = Path(args.model_dir)
    for needed in ("model_meta.json", "heads.pt", "encoder"):
        if not (model_dir / needed).exists():
            sys.exit(f"{model_dir} has no {needed}; is this the folder from train_multitask.py?")
    token = os.getenv("HF_TOKEN") or getpass.getpass("Hugging Face token (write), input hidden: ").strip()
    api = HfApi(token=token)
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    (model_dir / "README.md").write_text(CARD.format(results=results_table(Path(args.results))), encoding="utf-8")
    print(f"uploading {model_dir} -> {args.repo_id} (about 500 MB, a few minutes)...")
    api.upload_folder(repo_id=args.repo_id, folder_path=str(model_dir), commit_message="upload model")
    print(f"done: https://huggingface.co/{args.repo_id}")
    print(f"use it with MODEL_DIR=hf://{args.repo_id}")


if __name__ == "__main__":
    main()