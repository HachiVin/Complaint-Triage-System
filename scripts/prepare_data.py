"""Merge annotated files, de-duplicate and split into train/val/test.

    python scripts/prepare_data.py data/annotated/brimo.jsonl data/processed
    python scripts/prepare_data.py data/annotated/brimo.jsonl data/processed --split random

Default is a time split (newest 15% of reviews are the test set). Extra files given with --external
(e.g. scraped reviews of another bank, annotated) become a separate ``external.jsonl`` test set and are
never trained on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from complaint_triage.data import deduplicate, load_jsonl, save_jsonl, split_by_time, split_stratified, summary  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("out_dir")
    ap.add_argument("--split", choices=("time", "random"), default="time")
    ap.add_argument("--external", nargs="*", default=[])
    ap.add_argument("--allow-synthetic", action="store_true")
    args = ap.parse_args()

    rows = [r for path in args.datasets for r in load_jsonl(path)]
    if any(r.synthetic for r in rows) and not args.allow_synthetic:
        sys.exit("synthetic rows found; results must come from real data (use --allow-synthetic to test the pipeline)")
    rows, dropped = deduplicate(rows)
    parts = split_by_time(rows) if args.split == "time" else split_stratified(rows)
    out = Path(args.out_dir)
    for name, part in parts.items():
        save_jsonl(part, out / f"{name}.jsonl")
    if args.external:
        ext = [r for path in args.external for r in load_jsonl(path)]
        save_jsonl(ext, out / "external.jsonl")
        parts["external"] = ext
    report = {name: summary(part) for name, part in parts.items()}
    report["duplicates_dropped"] = dropped
    (out / "split_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    for name, part in parts.items():
        s = summary(part)
        print(f"{name:9} {s['reviews']:>7,} reviews  dates {s['date_range']}  urgency labelled {s['urgency_labelled']}")
    if not any(r.urgency for r in parts["test"]):
        print("note: test split has no urgency labels; annotate some recent reviews too (see annotation guide)")


if __name__ == "__main__":
    main()
