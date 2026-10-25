"""Batch pipeline: load reviews into PostgreSQL and score them with a model.

    python scripts/ingest.py data/interim/brimo.jsonl --model models/indobert-mt
    python scripts/ingest.py data/interim/livin.jsonl --model models/indobert-mt --labelled-by annotator

Re-running is safe: reviews are upserted by (source, id) and predictions by (review, model_version), so
scoring with a new model adds a new version next to the old one; v_reviews shows the latest.
Needs DATABASE_URL.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from complaint_triage.analyzer import Analyzer, load_model  # noqa: E402
from complaint_triage.data import load_jsonl  # noqa: E402
from complaint_triage.db import Database  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jsonl", nargs="+")
    ap.add_argument("--model", required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--labelled-by", default="dataset")
    args = ap.parse_args()

    dsn = os.getenv("DATABASE_URL")
    if not dsn:
        sys.exit("DATABASE_URL is not set")
    db = Database(dsn)
    db.apply_schema()
    analyzer = Analyzer(load_model(args.model))
    rows = [r for path in args.jsonl for r in load_jsonl(path)]
    started = time.perf_counter()
    for i in range(0, len(rows), args.batch_size):
        batch = rows[i: i + args.batch_size]
        ids = db.upsert_reviews(batch, labelled_by=args.labelled_by)
        db.save_predictions(ids, analyzer.analyze_batch([r.text for r in batch]))
        done = i + len(batch)
        rate = done / (time.perf_counter() - started)
        print(f"\r{done:,}/{len(rows):,} reviews  {rate:,.0f}/s", end="", flush=True)
    print(f"\nscored with {analyzer.version}")
    db.close()


if __name__ == "__main__":
    main()
