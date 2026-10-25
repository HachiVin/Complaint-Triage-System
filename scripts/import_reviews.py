"""Convert a raw review file into the project's JSONL format.

USH BRImo dataset (Mendeley, V1; the file inside the zip is a CSV named "USH Brimo Dataset"):

    python scripts/import_reviews.py "data/raw/USH Brimo Dataset" data/interim/brimo.jsonl \
        --app BRImo --source ush_brimo --text-col Ulasan --aspect-col Aspek --sentiment-col Sentimen \
        --rating-col Rating --date-col Tanggal --version-col "Versi App" --id-col none

Reviews scraped with scripts/scrape_play.py already use these default column names.

Reviewer names and profile pictures are never imported. Label values are mapped through
complaint_triage/labels.py; an unknown value stops the import and lists every unknown value found.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from complaint_triage.data import Review, deduplicate, save_jsonl, summary  # noqa: E402
from complaint_triage.labels import normalise_label  # noqa: E402
from scripts.inspect_dataset import read_any  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--app", required=True)
    ap.add_argument("--source", required=True, help="e.g. ush_brimo, play_scrape")
    ap.add_argument("--text-col", default="content")
    ap.add_argument("--aspect-col")
    ap.add_argument("--sentiment-col")
    ap.add_argument("--urgency-col")
    ap.add_argument("--rating-col", default="score")
    ap.add_argument("--date-col", default="at")
    ap.add_argument("--version-col", default="appVersion")
    ap.add_argument("--id-col", default="reviewId")
    ap.add_argument("--min-words", type=int, default=3, help="drop reviews shorter than this")
    ap.add_argument("--keep-duplicates", action="store_true")
    args = ap.parse_args()

    df = read_any(args.src)
    missing = [c for c in (args.text_col,) if c not in df.columns]
    if missing:
        sys.exit(f"column(s) {missing} not found; columns are {list(df.columns)}")

    def col(name) -> list:
        """A column as a plain list (fast to index), or all None when the column is absent."""
        return df[name].tolist() if name and name in df.columns else [None] * len(df)

    texts = col(args.text_col)
    dates = [d.date().isoformat() if pd.notna(d) else None for d in pd.to_datetime(col(args.date_col),
                                                                                  errors="coerce")]
    raw_labels = {"aspect": col(args.aspect_col), "sentiment": col(args.sentiment_col),
                  "urgency": col(args.urgency_col)}
    ids, ratings, versions = col(args.id_col), col(args.rating_col), col(args.version_col)

    def present(v) -> bool:
        return v is not None and str(v) not in ("nan", "None", "")

    unknown: dict[str, Counter] = {"aspect": Counter(), "sentiment": Counter(), "urgency": Counter()}
    rows, short = [], 0
    for i in range(len(df)):
        text = str(texts[i] or "").strip()
        if not text or text.lower() == "nan" or len(text.split()) < args.min_words:
            short += 1
            continue
        labels = {}
        for task, values in raw_labels.items():
            try:
                labels[task] = normalise_label(task, values[i])
            except ValueError:
                unknown[task][str(values[i])] += 1
                labels[task] = None
        rid = ids[i]
        if not present(rid):  # no id column: id from text, date and row position
            rid = hashlib.sha1(f"{text}|{dates[i]}|{i}".encode()).hexdigest()[:16]
        rows.append(Review(
            id=f"{args.source}-{rid}", app=args.app, text=text,
            rating=int(ratings[i]) if present(ratings[i]) else None,
            date=dates[i], app_version=str(versions[i]) if present(versions[i]) else None,
            source=args.source, **labels,
        ))

    if any(unknown.values()):
        print("Unknown label values (add them to the aliases in complaint_triage/labels.py, then re-run):")
        for task, counts in unknown.items():
            for value, n in counts.most_common():
                print(f"  {task}: {value!r} x{n}")
        sys.exit(1)

    dropped = 0
    if not args.keep_duplicates:
        rows, dropped = deduplicate(rows)
    save_jsonl(rows, args.dst)
    print(f"{len(rows):,} reviews -> {args.dst} ({short:,} too short, {dropped:,} duplicate texts dropped)")
    print(summary(rows))


if __name__ == "__main__":
    main()
