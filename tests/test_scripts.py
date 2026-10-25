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
    export = [{"data": t["data"], "annotations": [{"result": [
        {"from_name": "urgency", "to_name": "text", "type": "choices", "value": {"choices": ["tinggi"]}}]}]}
        for t in tasks[:5]]
    (tmp_path / "export.json").write_text(json.dumps(export), encoding="utf-8")
    out = tmp_path / "merged.jsonl"
    assert run("scripts/labelstudio.py", "merge", data, tmp_path / "export.json", out).returncode == 0
    assert sum(r.urgency == "tinggi" for r in load_jsonl(out)) == 5


def test_prepare_refuses_synthetic_without_flag(tmp_path):
    res = run("scripts/prepare_data.py", "data/sample/demo_reviews.jsonl", tmp_path / "p")
    assert res.returncode != 0
    res = run("scripts/prepare_data.py", "data/sample/demo_reviews.jsonl", tmp_path / "p", "--allow-synthetic")
    assert res.returncode == 0, res.stderr
    assert (tmp_path / "p" / "test.jsonl").exists() and (tmp_path / "p" / "split_report.json").exists()
