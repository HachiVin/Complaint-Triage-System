"""First look at a raw review file (CSV, XLSX or JSONL) before deciding how to import it.

    python scripts/inspect_dataset.py data/raw/ush_brimo.csv

Prints columns, missing values, the most common values of short text columns (likely labels), text
length, duplicate rate and a few sample rows. Run this before scripts/import_reviews.py.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd


def read_any(path: str) -> pd.DataFrame:
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path)
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    sys.exit("could not decode the file")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--samples", type=int, default=5)
    args = ap.parse_args()

    df = read_any(args.path)
    print(f"rows: {len(df):,}   columns: {len(df.columns)}\n")
    print("column            dtype     missing   unique")
    for c in df.columns:
        print(f"{str(c)[:16]:<17} {str(df[c].dtype)[:8]:<9} {df[c].isna().mean():>6.1%} {df[c].nunique():>9,}")

    text_cols = [c for c in df.columns if df[c].dtype == object]
    longest = max(text_cols, key=lambda c: df[c].astype(str).str.len().median(), default=None)
    if longest is not None:
        lengths = df[longest].astype(str).str.split().str.len()
        dup = df[longest].astype(str).str.lower().str.strip().duplicated().mean()
        print(f"\nlikely text column: {longest!r}  words median {lengths.median():.0f}, "
              f"p95 {lengths.quantile(.95):.0f}, duplicate texts {dup:.1%}")

    print("\nvalue counts of columns with <= 30 distinct values (candidate labels):")
    for c in df.columns:
        if 1 < df[c].nunique() <= 30:
            print(f"\n[{c}]")
            print(df[c].value_counts(dropna=False).head(30).to_string())

    for c in df.columns:
        if "date" in str(c).lower() or "time" in str(c).lower() or "at" == str(c).lower()[-2:]:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.5:
                print(f"\ndate column {c!r}: {parsed.min()} .. {parsed.max()}")

    print("\nsample rows:")
    with pd.option_context("display.max_colwidth", 120, "display.width", 200):
        print(df.sample(min(args.samples, len(df)), random_state=1).to_string())


if __name__ == "__main__":
    main()
