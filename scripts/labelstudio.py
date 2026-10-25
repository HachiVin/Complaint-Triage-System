"""Urgency annotation with Label Studio.

1. Pick reviews to annotate and write Label Studio tasks, pre-filled with the rule-based urgency:

       python scripts/labelstudio.py sample data/interim/brimo.jsonl data/interim/ls_urgency.json --n 800

   Critical complaints are rare, so the sample over-represents negative reviews and reviews the rules
   flag. That is fine for training; the test set's natural distribution is reported separately
   (see docs/annotation_guide.md).

2. After annotating, export JSON from Label Studio and merge the labels back:

       python scripts/labelstudio.py merge data/interim/brimo.jsonl data/interim/ls_export.json \
           data/annotated/brimo.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from complaint_triage.data import load_jsonl, save_jsonl, summary  # noqa: E402
from complaint_triage.rules import urgency_by_rules  # noqa: E402


def sample(src: str, dst: str, n: int, seed: int) -> None:
    rows = [r for r in load_jsonl(src) if not r.urgency]
    rng = random.Random(seed)
    flagged = [r for r in rows if urgency_by_rules(r.text)[0] != "normal"]
    flagged_ids = {r.id for r in flagged}
    negative = [r for r in rows if r.sentiment == "negatif" and r.id not in flagged_ids]
    negative_ids = {r.id for r in negative}
    rest = [r for r in rows if r.id not in flagged_ids and r.id not in negative_ids]
    picked = []
    for pool, share in ((flagged, 0.4), (negative, 0.35), (rest, 0.25)):
        rng.shuffle(pool)
        picked += pool[: int(n * share)]
    if len(picked) < n:  # top up from whatever is left
        picked_ids = {r.id for r in picked}
        left = [r for r in rows if r.id not in picked_ids]
        rng.shuffle(left)
        picked += left[: n - len(picked)]
    rng.shuffle(picked)
    tasks = []
    for r in picked:
        label, phrases = urgency_by_rules(r.text)
        tasks.append({
            "data": {"id": r.id, "text": r.text, "app": r.app, "rating": r.rating, "aspect": r.aspect,
                     "sentiment": r.sentiment, "rule_hits": ", ".join(phrases)},
            "predictions": [{"model_version": "rules", "result": [
                {"from_name": "urgency", "to_name": "text", "type": "choices", "value": {"choices": [label]}}]}],
        })
    Path(dst).write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(tasks)} tasks -> {dst} (rules flagged {len(flagged)} of {len(rows)} unlabelled reviews)")


def merge(dataset: str, export: str, dst: str) -> None:
    rows = load_jsonl(dataset)
    by_id = {r.id: r for r in rows}
    done = 0
    for task in json.loads(Path(export).read_text(encoding="utf-8")):
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not anns:
            continue
        choice = next((x["value"]["choices"][0] for x in anns[-1]["result"] if x["type"] == "choices"
                       and x["from_name"] == "urgency"), None)
        rid = task["data"]["id"]
        if choice and rid in by_id:
            by_id[rid].urgency = choice
            done += 1
    save_jsonl(rows, dst)
    print(f"{done} urgency labels merged -> {dst}")
    print(summary(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("src")
    s.add_argument("dst")
    s.add_argument("--n", type=int, default=800)
    s.add_argument("--seed", type=int, default=42)
    m = sub.add_parser("merge")
    m.add_argument("dataset")
    m.add_argument("export")
    m.add_argument("dst")
    args = ap.parse_args()
    if args.cmd == "sample":
        sample(args.src, args.dst, args.n, args.seed)
    else:
        merge(args.dataset, args.export, args.dst)


if __name__ == "__main__":
    main()
