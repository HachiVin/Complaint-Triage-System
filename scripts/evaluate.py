"""Compare models on the test split (and the external set, if present). Writes docs/results.md/.json.

    python scripts/evaluate.py data/processed --baseline models/baseline --multitask models/indobert-mt

With --human data/annotation/human_test.jsonl, urgency on the test split is scored against the
human-verified labels only (reviews marked ragu are left out); aspect and sentiment are unchanged.
The silver labels those reviews had before are compared with the human ones and reported too.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.metrics import cohen_kappa_score  # noqa: E402

from complaint_triage.data import Review, load_jsonl, summary  # noqa: E402
from complaint_triage.evaluation import evaluate_model, evaluate_rules  # noqa: E402


def size_mb(path: Path) -> float:
    files = [f for f in path.rglob("*") if f.is_file()]
    return round(sum(f.stat().st_size for f in files) / 1e6, 1)


def load_human(path: str) -> dict[str, str]:
    """id -> urgency from human_test.jsonl, without the reviews the annotator marked ragu."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if not item.get("ragu"):
                out[item["id"]] = item["urgency"]
    return out


def apply_human(rows: list[Review], labels: dict[str, str]) -> tuple[list[Review], dict]:
    """Urgency = the human label where there is one, nothing elsewhere. Also compares the old (silver)
    labels with the human ones on the reviews that have both."""
    missing = set(labels) - {r.id for r in rows}
    if missing:
        sys.exit(f"{len(missing)} human-labelled ids are not in the test split (e.g. {sorted(missing)[0]}); "
                 "was the test split rebuilt from different data?")
    pairs = [(r.urgency, labels[r.id]) for r in rows if r.urgency and r.id in labels]
    out = [r.model_copy(update={"urgency": labels.get(r.id)}) for r in rows]
    agreement = {"n": len(pairs)}
    if pairs:
        old, new = zip(*pairs, strict=True)
        agreement["agreement"] = round(sum(a == b for a, b in pairs) / len(pairs), 4)
        agreement["cohen_kappa"] = round(float(cohen_kappa_score(old, new)), 4)
    return out, agreement


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--baseline")
    ap.add_argument("--multitask")
    ap.add_argument("--out", default="docs")
    ap.add_argument("--human", help="human-verified urgency labels for the test split (human_test.jsonl)")
    args = ap.parse_args()

    data = Path(args.data_dir)
    sets = {"test": load_jsonl(data / "test.jsonl")}
    silver_vs_human = None
    if args.human:
        sets["test"], silver_vs_human = apply_human(sets["test"], load_human(args.human))
    if (data / "external.jsonl").exists():
        sets["external"] = load_jsonl(data / "external.jsonl")

    models = []
    if args.baseline:
        from complaint_triage.baseline import BaselineModel

        models.append(("TF-IDF + LogReg", BaselineModel.load(args.baseline), size_mb(Path(args.baseline))))
    if args.multitask:
        from complaint_triage.multitask import MultiTaskModel

        models.append(("IndoBERT multi-task", MultiTaskModel(args.multitask), size_mb(Path(args.multitask))))

    results = {"date": date.today().isoformat(), "sets": {}}
    lines = [f"# Results ({results['date']})", ""]
    for set_name, rows in sets.items():
        res = {"data": summary(rows), "models": {}, "rules_urgency": evaluate_rules(rows)}
        for name, model, size in models:
            print(f"{set_name}: {name} ...")
            m = evaluate_model(model, rows)
            m["size_mb"] = size
            res["models"][name] = m
        results["sets"][set_name] = res

        lines += [f"## {set_name} ({len(rows):,} reviews, {res['data']['date_range']})", "",
                  "| Model | Aspect macro-F1 | Sentiment macro-F1 | Urgency macro-F1 | Kritis recall | "
                  "Needs review | Throughput | p50 latency | Size |", "|---|---|---|---|---|---|---|---|---|"]
        ru = res["rules_urgency"]
        if ru.get("n"):
            lines.append(f"| Rules (urgency only) | – | – | {ru['macro_f1']:.3f} | {ru.get('kritis_recall', '–')} "
                         "| – | – | – | – |")
        for name, m in res["models"].items():
            t = m["tasks"]

            def f1(task, t=t):
                return f"{t[task]['macro_f1']:.3f}" if task in t else "–"

            kr = t.get("urgency", {}).get("kritis_recall", "–")
            lines.append(f"| {name} | {f1('aspect')} | {f1('sentiment')} | {f1('urgency')} | {kr} | "
                         f"{m['needs_review_rate']:.1%} | {m['throughput_per_s']}/s | "
                         f"{m['latency_ms_single']['p50']} ms | {m['size_mb']} MB |")
        n_urg = sum(r.urgency is not None for r in rows)
        if n_urg:
            n_kritis = sum(r.urgency == "kritis" for r in rows)
            lines.append(f"Urgency is scored on the {n_urg} reviews with an urgency label ({n_kritis} kritis), "
                         f"so one kritis review is worth {1 / max(n_kritis, 1):.0%} of kritis recall.")
        lines.append("")

    provenance = Path(args.out) / "label_provenance.json"
    if args.human:
        n_h = sum(r.urgency is not None for r in sets["test"])
        results["urgency_label_status"] = "human-verified test set"
        results["silver_vs_human"] = silver_vs_human
        lines += [f"> **Urgency on the test split: human-verified, {n_h} reviews** (blind annotation, one "
                  f"annotator, sample enriched with rule-flagged reviews; `{Path(args.human).as_posix()}`).", ""]
        if silver_vs_human and silver_vs_human["n"]:
            lines += [f"Silver labels of v0.1 against the human labels, on the {silver_vs_human['n']} reviews that "
                      f"have both: {silver_vs_human['agreement']:.1%} agreement, Cohen's kappa "
                      f"{silver_vs_human['cohen_kappa']:.3f}.", ""]
    elif provenance.exists():
        info = json.loads(provenance.read_text(encoding="utf-8"))
        results["urgency_label_status"] = info.get("status")
        lines += [f"> **Urgency labels: {info.get('status')}.** {info.get('method')}. "
                  f"See `{provenance.as_posix()}`.", ""]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()