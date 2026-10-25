from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from complaint_triage.data import load_jsonl, save_jsonl

ROOT = Path(__file__).resolve().parent.parent


def run(*args):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True, text=True)


def test_import_maps_columns_and_labels(tmp_path):
    src = tmp_path / "raw.csv"
    pd.DataFrame({
        "reviewId": ["r1", "r2", "r3", "r4"],
        "userName": ["Budi", "Sari", "Andi", "Tono"],
        "content": ["transfer gagal saldo terpotong", "aplikasi sangat membantu sekali", "ok",
                    "aplikasi sangat membantu sekali"],
        "score": [1, 5, 4, 5], "at": ["2026-09-01 10:00:00", "2026-09-02 11:00:00", "2026-09-03", "2026-09-04"],
        "appVersion": ["1.0", None, "1.0", "1.1"],
        "aspect": ["Transaction Quality", "General", "General", "General"],
        "sentiment": ["Negative", "Positive", "Positive", "Positive"],
    }).to_csv(src, index=False)
    dst = tmp_path / "out.jsonl"
    res = run("scripts/import_reviews.py", src, dst, "--app", "BRImo", "--source", "ush_brimo",
              "--aspect-col", "aspect", "--sentiment-col", "sentiment")
    assert res.returncode == 0, res.stderr
    rows = load_jsonl(dst)
    assert len(rows) == 2  # "ok" too short, one duplicate dropped
    assert rows[0].aspect == "kualitas_transaksi" and rows[0].sentiment == "negatif"
    assert rows[0].date == "2026-09-01" and rows[1].app_version is None
    assert "Budi" not in dst.read_text(encoding="utf-8")


def test_import_stops_on_unknown_label(tmp_path):
    src = tmp_path / "raw.csv"
    pd.DataFrame({"content": ["transfer gagal saldo terpotong"], "aspect": ["Taste"]}).to_csv(src, index=False)
    res = run("scripts/import_reviews.py", src, tmp_path / "o.jsonl", "--app", "X", "--source", "t",
              "--aspect-col", "aspect")
    assert res.returncode == 1
    assert "'Taste'" in res.stdout


def test_labelstudio_sample_and_merge(tmp_path, records):
    data = tmp_path / "data.jsonl"
    save_jsonl([r.model_copy(update={"urgency": None}) for r in records[:60]], data)
    tasks_file = tmp_path / "tasks.json"
    assert run("scripts/labelstudio.py", "sample", data, tasks_file, "--n", "20").returncode == 0
    tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
    assert len(tasks) == 20 and len({t["data"]["id"] for t in tasks}) == 20
    assert all(t["data"]["meta"].startswith("Periode:") for t in tasks)
    order = {"test": 0, "val": 1, "train": 2}
    ranks = [order[t["data"]["periode"]] for t in tasks]
    assert ranks == sorted(ranks)  # test period first, so stopping early keeps the test set complete
    export = [{"data": t["data"], "annotations": [{"result": [
        {"from_name": "urgency", "to_name": "text", "type": "choices", "value": {"choices": ["tinggi"]}}]}]}
        for t in tasks[:5]]
    export.append({"data": tasks[5]["data"], "annotations": [{"result": [
        {"from_name": "urgency", "to_name": "text", "type": "choices", "value": {"choices": ["kritis"]}},
        {"from_name": "ragu", "to_name": "text", "type": "choices", "value": {"choices": ["ragu"]}}]}]})
    (tmp_path / "export.json").write_text(json.dumps(export), encoding="utf-8")
    out = tmp_path / "merged.jsonl"
    assert run("scripts/labelstudio.py", "merge", data, tmp_path / "export.json", out).returncode == 0
    merged = load_jsonl(out)
    assert sum(r.urgency == "tinggi" for r in merged) == 5
    assert sum(r.urgency is not None for r in merged) == 5  # the "ragu" one stays unlabelled
    ragu = (tmp_path / "merged.ragu.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ragu) == 1 and json.loads(ragu[0])["picked"] == "kritis"


def test_labelstudio_crosscheck_and_merge(tmp_path, records):
    from complaint_triage.rules import urgency_by_rules

    data = tmp_path / "data.jsonl"
    save_jsonl([r.model_copy(update={"urgency": None}) for r in records], data)
    tasks_file = tmp_path / "tasks.json"
    assert run("scripts/labelstudio.py", "sample", data, tasks_file, "--n", "60").returncode == 0
    tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
    # Second opinion: agrees with the rules on even positions, disagrees on odd ones, unsure every 10th.
    second = tmp_path / "second.jsonl"
    with second.open("w", encoding="utf-8") as f:
        for i, t in enumerate(tasks):
            rule = urgency_by_rules(t["data"]["text"])[0]
            label = rule if i % 2 == 0 else ("tinggi" if rule == "normal" else "normal")
            f.write(json.dumps({"id": t["data"]["id"], "urgency": label, "unsure": i % 10 == 0, "by": "t"}) + "\n")
    cross_file = tmp_path / "cross.json"
    res = run("scripts/labelstudio.py", "crosscheck", tasks_file, second, cross_file)
    assert res.returncode == 0, res.stderr
    cross = json.loads(cross_file.read_text(encoding="utf-8"))
    position = {t["data"]["id"]: i for i, t in enumerate(tasks)}
    for t in cross:
        i = position[t["data"]["id"]]
        auto = i % 2 == 0 and i % 10 != 0 and t["data"]["periode"] != "test"
        assert t["data"]["cek"] == ("tidak" if auto else "ya")
        assert bool(t.get("annotations")) == auto  # test period and disagreements are left to a person
    flags = [t["data"]["cek"] for t in cross]
    assert flags == sorted(flags, key=lambda c: c == "tidak")  # the work list comes first

    # Merge: only the accepted ones are annotated, so the test period is reported as unfinished.
    export = [t for t in cross if t.get("annotations")]
    (tmp_path / "export.json").write_text(json.dumps(export), encoding="utf-8")
    res = run("scripts/labelstudio.py", "merge", data, tmp_path / "export.json", tmp_path / "merged.jsonl",
              "--tasks", cross_file)
    assert res.returncode == 0, res.stderr
    assert "accepted from the crosscheck" in res.stdout
    if any(t["data"]["periode"] == "test" for t in cross):
        assert "WARNING" in res.stdout


def test_labelstudio_silver(tmp_path, records):
    from complaint_triage.rules import urgency_by_rules

    data = tmp_path / "data.jsonl"
    save_jsonl([r.model_copy(update={"urgency": None}) for r in records], data)
    tasks_file = tmp_path / "tasks.json"
    assert run("scripts/labelstudio.py", "sample", data, tasks_file, "--n", "60").returncode == 0
    tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
    second = tmp_path / "second.jsonl"
    with second.open("w", encoding="utf-8") as f:
        for i, t in enumerate(tasks):
            rule = urgency_by_rules(t["data"]["text"])[0]
            label = rule if i % 2 == 0 else ("tinggi" if rule == "normal" else "normal")
            f.write(json.dumps({"id": t["data"]["id"], "urgency": label, "unsure": i % 7 == 0, "by": "t"}) + "\n")
    cross = tmp_path / "cross.json"
    assert run("scripts/labelstudio.py", "crosscheck", tasks_file, second, cross).returncode == 0
    out, prov = tmp_path / "silver.jsonl", tmp_path / "prov.json"
    res = run("scripts/labelstudio.py", "silver", data, cross, out, "--provenance", prov)
    assert res.returncode == 0, res.stderr
    labelled = {r.id: r.urgency for r in load_jsonl(out) if r.urgency}
    for i, t in enumerate(tasks):
        tid, period = t["data"]["id"], t["data"]["periode"]
        rule = urgency_by_rules(t["data"]["text"])[0]
        if i % 7 == 0:
            assert tid not in labelled  # unsure is never used
        elif period == "test":
            assert labelled[tid] == (rule if i % 2 == 0 else ("tinggi" if rule == "normal" else "normal"))
        elif i % 2 == 0:
            assert labelled[tid] == rule  # agreed
        else:
            assert tid not in labelled  # disagreement outside the test period is skipped
    assert json.loads(prov.read_text(encoding="utf-8"))["status"].startswith("silver")


def test_prepare_refuses_synthetic_without_flag(tmp_path):
    res = run("scripts/prepare_data.py", "data/sample/demo_reviews.jsonl", tmp_path / "p")
    assert res.returncode != 0
    res = run("scripts/prepare_data.py", "data/sample/demo_reviews.jsonl", tmp_path / "p", "--allow-synthetic")
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "p" / "test.jsonl").exists() and (tmp_path / "p" / "split_report.json").exists()