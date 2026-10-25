"""Urgency annotation with Label Studio.

1. Pick reviews to annotate and write Label Studio tasks, pre-filled with the rule-based urgency:

       python scripts/labelstudio.py sample data/interim/brimo.jsonl data/interim/ls_urgency.json --n 800

   Critical complaints are rare, so the sample over-represents reviews the rules flag (20% kritis,
   20% tinggi) and negative reviews (35%); the other 25% is drawn from the rest. That is fine for
   training; the natural distribution is reported separately (see docs/annotation_guide.md).

   Tasks are ordered test period first, then validation, then training (field "periode"), so
   annotating from the top and stopping early still leaves a complete, hand-labelled test set.

1b. Optional, when time is short: add a second, independent annotator (an LLM, or a colleague) and
   cross-check it against the rules:

       python scripts/labelstudio.py crosscheck data/interim/ls_urgency.json \
           data/annotation/second_opinion.jsonl data/interim/ls_urgency_crosscheck.json

   Outside the test period, where the rules and the second annotator agree, the label is imported as
   already annotated (cek = "tidak"). Everything else is left for a person (cek = "ya") and put at the
   top of the list in this order: every test-period review (always decided by a person), then
   validation and training reviews where the two disagree or the second annotator was unsure.
   Disagreements nobody gets to are simply left unlabelled.

1c. No time for any manual review yet: build *silver* labels straight from the crosscheck file,
   without Label Studio:

       python scripts/labelstudio.py silver data/interim/brimo.jsonl data/interim/ls_urgency_crosscheck.json \
           data/annotated/brimo.jsonl

   Training and validation get the labels the rules and the second annotator agree on; the test period
   gets the second annotator's label (unsure ones are skipped). Nothing here is human-verified, and the
   script writes docs/label_provenance.json saying so. Replace it later with step 2.

2. After annotating, export JSON from Label Studio and merge the labels back:

       python scripts/labelstudio.py merge data/interim/brimo.jsonl data/interim/ls_export.json \
           data/annotated/brimo.jsonl --tasks data/interim/ls_urgency_crosscheck.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from complaint_triage.data import deduplicate, load_jsonl, save_jsonl, split_by_time, summary  # noqa: E402
from complaint_triage.rules import urgency_by_rules  # noqa: E402

PERIOD_ORDER = {"test": 0, "val": 1, "train": 2}


def sample(src: str, dst: str, n: int, seed: int) -> None:
    every, _ = deduplicate(load_jsonl(src))
    # The same time split as scripts/prepare_data.py, so each task knows where it will end up.
    period = {r.id: name for name, part in split_by_time(every).items() for r in part}
    rows = [r for r in every if not r.urgency]
    rng = random.Random(seed)
    rule = {r.id: urgency_by_rules(r.text)[0] for r in rows}
    # Rules flag far more "tinggi" than "kritis" (about 7 to 1 on BRImo), so the two are sampled
    # separately; otherwise the sample would hold only a few dozen likely-critical reviews.
    flag_kritis = [r for r in rows if rule[r.id] == "kritis"]
    flag_tinggi = [r for r in rows if rule[r.id] == "tinggi"]
    flagged = flag_kritis + flag_tinggi
    flagged_ids = {r.id for r in flagged}
    negative = [r for r in rows if r.sentiment == "negatif" and r.id not in flagged_ids]
    negative_ids = {r.id for r in negative}
    rest = [r for r in rows if r.id not in flagged_ids and r.id not in negative_ids]
    picked = []
    for pool, share in ((flag_kritis, 0.2), (flag_tinggi, 0.2), (negative, 0.35), (rest, 0.25)):
        rng.shuffle(pool)
        picked += pool[: int(n * share)]
    if len(picked) < n:  # top up from whatever is left
        picked_ids = {r.id for r in picked}
        left = [r for r in rows if r.id not in picked_ids]
        rng.shuffle(left)
        picked += left[: n - len(picked)]
    rng.shuffle(picked)
    # Test-period reviews first, then validation, then training: whoever stops early still leaves a
    # fully labelled test set behind.
    picked.sort(key=lambda r: PERIOD_ORDER[period[r.id]])
    tasks = []
    for r in picked:
        label, phrases = urgency_by_rules(r.text)
        # One ready-made line of context: Label Studio's Text tag shows one field, not a template.
        meta = (f"Periode: {period[r.id]} · Aplikasi: {r.app} · Rating: {r.rating or '-'} · "
                f"Aspek: {r.aspect or '-'} · Sentimen: {r.sentiment or '-'} · "
                f"Kata pemicu: {', '.join(phrases) or '-'}")
        tasks.append({
            "data": {"id": r.id, "text": r.text, "meta": meta, "periode": period[r.id], "app": r.app,
                     "rating": r.rating,
                     "aspect": r.aspect, "sentiment": r.sentiment, "rule_hits": ", ".join(phrases)},
            "predictions": [{"model_version": "rules", "result": [
                {"from_name": "urgency", "to_name": "text", "type": "choices", "value": {"choices": [label]}}]}],
        })
    Path(dst).write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(tasks)} tasks -> {dst} (rules flagged {len(flagged)} of {len(rows)} unlabelled reviews)")
    counts = {name: sum(period[r.id] == name for r in picked) for name in PERIOD_ORDER}
    print(f"order: {counts['test']} test, then {counts['val']} val, then {counts['train']} train")


def _urgency_result(label: str) -> list[dict]:
    return [{"from_name": "urgency", "to_name": "text", "type": "choices", "value": {"choices": [label]}}]


def crosscheck(tasks_path: str, second: str, dst: str) -> None:
    """Combine the rule guess (already in the tasks) with a second annotator (jsonl: id, urgency, unsure, by)."""
    other = {}
    for line in Path(second).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            other[item["id"]] = item
    tasks = json.loads(Path(tasks_path).read_text(encoding="utf-8"))
    if any("periode" not in t["data"] for t in tasks):
        sys.exit("tasks have no 'periode' field; re-create them with the current `sample` command")
    counts = {"auto": 0, "test": 0, "beda": 0, "ragu": 0, "tanpa_opini": 0}
    for task in tasks:
        data = task["data"]
        rule = urgency_by_rules(data["text"])[0]
        item = other.get(data["id"])
        task.pop("annotations", None)
        if data["periode"] == "test":
            reason = "test"
        elif item is None:
            reason = "tanpa_opini"
        elif item.get("unsure"):
            reason = "ragu"
        elif item["urgency"] != rule:
            reason = "beda"
        else:
            reason = "auto"
        counts[reason] += 1
        data["aturan"] = rule
        data["opini_kedua"] = item["urgency"] if item else "-"
        data["opini_ragu"] = "ya" if item and item.get("unsure") else "tidak"
        data["cek"] = "tidak" if reason == "auto" else "ya"
        data["alasan_cek"] = {"auto": "-", "test": "periode test, wajib diputuskan manusia",
                              "beda": "aturan dan opini kedua berbeda", "ragu": "opini kedua ragu",
                              "tanpa_opini": "belum ada opini kedua"}[reason]
        if reason == "auto":
            task["annotations"] = [{"result": _urgency_result(rule)}]
            task["predictions"] = []
        elif item is not None:
            task["predictions"] = [{"model_version": item.get("by", "second_opinion"),
                                    "result": _urgency_result(item["urgency"])}]
    # Work list first: test period, then validation, then training; accepted ones at the end.
    tasks.sort(key=lambda t: (t["data"]["cek"] == "tidak", PERIOD_ORDER[t["data"]["periode"]]))
    Path(dst).write_text(json.dumps(tasks, ensure_ascii=False, indent=1), encoding="utf-8")
    review = len(tasks) - counts["auto"]
    print(f"{len(tasks)} tasks -> {dst}")
    print(f"  {counts['auto']} imported as annotated (rules and second opinion agree, outside the test period)")
    print(f"  {review} for a person (cek = ya), at the top: {counts['test']} test period (required), then "
          f"{counts['beda']} disagreements and {counts['ragu'] + counts['tanpa_opini']} unsure (as time allows)")


def silver(dataset: str, tasks_path: str, dst: str, provenance: str = "docs/label_provenance.json") -> None:
    """Urgency labels without human review, from a crosscheck file. Meant as a first version only."""
    rows = load_jsonl(dataset)
    by_id = {r.id: r for r in rows}
    counts = {"agreed": 0, "second_opinion_test": 0, "skipped": 0}
    per_period = {name: {"kritis": 0, "tinggi": 0, "normal": 0} for name in PERIOD_ORDER}
    for task in json.loads(Path(tasks_path).read_text(encoding="utf-8")):
        data = task["data"]
        if "cek" not in data:
            sys.exit("not a crosscheck file; run `crosscheck` first")
        rid, period = data["id"], data["periode"]
        if data["cek"] == "tidak":
            label, how = data["aturan"], "agreed"
        elif period == "test" and data["opini_kedua"] != "-" and data.get("opini_ragu") != "ya":
            label, how = data["opini_kedua"], "second_opinion_test"
        else:
            counts["skipped"] += 1
            continue
        if rid in by_id:
            by_id[rid].urgency = label
            counts[how] += 1
            per_period[period][label] += 1
    save_jsonl(rows, dst)
    info = {
        "status": "silver: not human-verified",
        "method": "train/val: keyword rules and LLM second opinion agree; test period: LLM second opinion",
        "second_opinion": "data/annotation/second_opinion.jsonl",
        "counts": counts, "labels_per_period": per_period,
        "next_step": "human review of every test-period review in Label Studio, then `merge`",
    }
    Path(provenance).parent.mkdir(parents=True, exist_ok=True)
    Path(provenance).write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"{counts['agreed'] + counts['second_opinion_test']} silver urgency labels -> {dst} "
          f"({counts['agreed']} agreed, {counts['second_opinion_test']} test period from the second opinion, "
          f"{counts['skipped']} skipped)")
    print(f"per period: {per_period}")
    print(f"provenance -> {provenance}  (NOT human-verified; say so wherever results are shown)")


def _choice(result: list[dict], name: str) -> str | None:
    return next((x["value"]["choices"][0] for x in result
                 if x.get("type") == "choices" and x.get("from_name") == name and x["value"].get("choices")), None)


def merge(dataset: str, export: str, dst: str, tasks_path: str | None = None) -> None:
    """Copy urgency labels into the dataset. Reviews marked "ragu" (unsure) get no label, so they are used
    neither for training nor for testing; they are listed in <dst>.ragu.jsonl to discuss and relabel."""
    rows = load_jsonl(dataset)
    by_id = {r.id: r for r in rows}
    done, unsure, changed, auto = 0, [], 0, 0
    test_done = 0
    for task in json.loads(Path(export).read_text(encoding="utf-8")):
        anns = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        rid = task["data"]["id"]
        test_done += task["data"].get("periode") == "test" and bool(anns)
        if not anns or rid not in by_id:
            continue
        result = anns[-1]["result"]
        choice = _choice(result, "urgency")
        if _choice(result, "ragu"):
            unsure.append({"id": rid, "text": task["data"]["text"], "picked": choice,
                           "rules": urgency_by_rules(task["data"]["text"])[0]})
            continue
        if choice:
            by_id[rid].urgency = choice
            done += 1
            changed += choice != urgency_by_rules(task["data"]["text"])[0]
            auto += task["data"].get("cek") == "tidak"
    save_jsonl(rows, dst)
    ragu_path = Path(dst).with_suffix(".ragu.jsonl")
    ragu_path.write_text("".join(json.dumps(u, ensure_ascii=False) + "\n" for u in unsure), encoding="utf-8")
    print(f"{done} urgency labels merged -> {dst}")
    print(f"{changed} of them differ from the rule-based guess ({changed / max(done, 1):.0%})")
    if auto:
        print(f"  {auto} accepted from the crosscheck (rules and second opinion agreed), "
              f"{done - auto} decided by a person")
    print(f"{len(unsure)} marked ragu, left unlabelled -> {ragu_path}")
    # Label Studio exports annotated tasks only, so the number to reach comes from the task file.
    test_total = None
    if tasks_path:
        test_total = sum(t["data"].get("periode") == "test"
                         for t in json.loads(Path(tasks_path).read_text(encoding="utf-8")))
    print(f"{test_done} test-period reviews annotated" + (f" of {test_total}" if test_total is not None else ""))
    if test_total is not None and test_done < test_total:
        print(f"WARNING: {test_total - test_done} test-period reviews are not annotated yet; finish them "
              f"before training, or the test set is incomplete")
    print(summary(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("src")
    s.add_argument("dst")
    s.add_argument("--n", type=int, default=800)
    s.add_argument("--seed", type=int, default=42)
    c = sub.add_parser("crosscheck")
    c.add_argument("tasks")
    c.add_argument("second_opinion")
    c.add_argument("dst")
    v = sub.add_parser("silver")
    v.add_argument("dataset")
    v.add_argument("crosscheck")
    v.add_argument("dst")
    v.add_argument("--provenance", default="docs/label_provenance.json")
    m = sub.add_parser("merge")
    m.add_argument("dataset")
    m.add_argument("export")
    m.add_argument("dst")
    m.add_argument("--tasks", help="the task file imported into Label Studio, to check the test period is complete")
    args = ap.parse_args()
    if args.cmd == "sample":
        sample(args.src, args.dst, args.n, args.seed)
    elif args.cmd == "crosscheck":
        crosscheck(args.tasks, args.second_opinion, args.dst)
    elif args.cmd == "silver":
        silver(args.dataset, args.crosscheck, args.dst, args.provenance)
    else:
        merge(args.dataset, args.export, args.dst, args.tasks)


if __name__ == "__main__":
    main()