"""Compare models on the test split (and the external set, if present). Writes docs/results.md/.json.

    python scripts/evaluate.py data/processed --baseline models/baseline --multitask models/indobert-mt
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from complaint_triage.data import load_jsonl, summary  # noqa: E402
from complaint_triage.evaluation import evaluate_model, evaluate_rules  # noqa: E402


def size_mb(path: Path) -> float:
    files = [f for f in path.rglob("*") if f.is_file()]
    return round(sum(f.stat().st_size for f in files) / 1e6, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--baseline")
    ap.add_argument("--multitask")
    ap.add_argument("--out", default="docs")
    args = ap.parse_args()

    data = Path(args.data_dir)
    sets = {"test": load_jsonl(data / "test.jsonl")}
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
    if provenance.exists():
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