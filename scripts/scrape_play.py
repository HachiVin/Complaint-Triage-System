"""Collect recent Google Play reviews of other banking apps, to test whether the model works beyond BRImo.

Run on your own machine (needs internet):

    pip install google-play-scraper
    python scripts/scrape_play.py id.co.bankmandiri.livin livin.csv --count 2000
    python scripts/import_reviews.py livin.csv data/interim/livin.jsonl --app "Livin by Mandiri" --source play_scrape

Only the review text, rating, date, app version and review id are saved. Reviewer names and profile
images are dropped. Keep request volume modest and use the data for research only.
Find an app id in its Play Store URL (…/details?id=<app id>).
"""

from __future__ import annotations

import argparse
import csv
import time


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("app_id")
    ap.add_argument("out_csv")
    ap.add_argument("--count", type=int, default=2000)
    ap.add_argument("--lang", default="id")
    ap.add_argument("--country", default="id")
    args = ap.parse_args()

    from google_play_scraper import Sort, reviews

    collected, token = [], None
    while len(collected) < args.count:
        batch, token = reviews(args.app_id, lang=args.lang, country=args.country, sort=Sort.NEWEST,
                               count=min(200, args.count - len(collected)), continuation_token=token)
        if not batch:
            break
        collected += batch
        time.sleep(1.0)
        if token is None:
            break

    with open(args.out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["reviewId", "content", "score", "at", "appVersion"])
        w.writeheader()
        for r in collected:
            w.writerow({"reviewId": r["reviewId"], "content": r["content"], "score": r["score"],
                        "at": r["at"].isoformat() if r.get("at") else "", "appVersion": r.get("appVersion") or ""})
    print(f"{len(collected)} reviews -> {args.out_csv}")


if __name__ == "__main__":
    main()
