from __future__ import annotations

from pathlib import Path

import pytest

from complaint_triage.analyzer import Analyzer
from complaint_triage.baseline import BaselineModel
from complaint_triage.data import Review, deduplicate, load_jsonl, save_jsonl, split_by_time, split_stratified
from complaint_triage.evaluation import evaluate_model, evaluate_rules
from complaint_triage.labels import TASKS, normalise_label
from complaint_triage.rules import urgency_by_rules


def test_normalise_label_aliases_and_unknown():
    assert normalise_label("aspect", "Transaction Quality") == "kualitas_transaksi"
    assert normalise_label("sentiment", "Negative") == "negatif"
    assert normalise_label("urgency", None) is None
    with pytest.raises(ValueError):
        normalise_label("aspect", "rasa kopi")


def test_review_rejects_unknown_label():
    with pytest.raises(ValueError):
        Review(id="x", text="aplikasi lambat", aspect="bukan_aspek")


def test_jsonl_roundtrip(tmp_path, records):
    path = tmp_path / "rows.jsonl"
    save_jsonl(records[:5], path)
    assert load_jsonl(path) == records[:5]


def test_deduplicate_ignores_case_and_spaces():
    rows = [Review(id="a", text="Mantap  jiwa"), Review(id="b", text="mantap jiwa"), Review(id="c", text="lambat")]
    kept, dropped = deduplicate(rows)
    assert [r.id for r in kept] == ["a", "c"] and dropped == 1


def test_split_by_time_puts_newest_in_test(records):
    parts = split_by_time(records)
    assert max(r.date for r in parts["train"] if r.date) <= min(r.date for r in parts["test"])
    assert sum(len(p) for p in parts.values()) == len(records)


def test_split_stratified_keeps_every_row(records):
    parts = split_stratified(records, task="aspect")
    ids = [r.id for p in parts.values() for r in p]
    assert sorted(ids) == sorted(r.id for r in records)


@pytest.mark.parametrize("text, expected", [
    ("Transfer gagal tapi saldo terpotong, sudah 3 hari", "kritis"),
    ("ada transaksi yang tidak saya lakukan di rekening", "kritis"),
    ("tidak bisa login sejak update kemarin", "tinggi"),
    ("OTP tidak masuk masuk", "tinggi"),
    ("tampilannya bagus, mudah dipakai", "normal"),
])
def test_urgency_rules(text, expected):
    assert urgency_by_rules(text)[0] == expected


def test_baseline_probabilities_follow_task_order(baseline):
    proba = baseline.predict_proba(["transfer gagal saldo terpotong", "aplikasi bagus"])
    for task, p in proba.items():
        assert p.shape == (2, len(TASKS[task]))
        assert abs(float(p[0].sum()) - 1) < 1e-4


def test_baseline_save_load(tmp_path, baseline):
    baseline.save(tmp_path)
    loaded = BaselineModel.load(tmp_path)
    assert loaded.version == "baseline:test"
    assert set(loaded.tasks) == set(baseline.tasks)


def test_analyzer_falls_back_to_rules_without_urgency_head(records):
    no_urgency = [r.model_copy(update={"urgency": None}) for r in records]
    model = BaselineModel.train(no_urgency, version="no-urgency")
    assert "urgency" not in model.tasks
    result = Analyzer(model).analyze("Transfer gagal tapi saldo terpotong")
    assert result.urgency.label == "kritis" and result.urgency.source == "rules"
    assert result.matched_phrases


def test_evaluate_model_report_shape(baseline, records):
    report = evaluate_model(baseline, records[:80])
    assert set(report["tasks"]) == {"aspect", "sentiment", "urgency"}
    assert "kritis_recall" in report["tasks"]["urgency"]
    assert report["latency_ms_single"]["p50"] >= 0


def test_resolve_model_dir(monkeypatch, tmp_path):
    import huggingface_hub

    from complaint_triage.analyzer import resolve_model_dir

    assert resolve_model_dir("models/baseline") == Path("models/baseline")
    calls = []
    monkeypatch.setattr(huggingface_hub, "snapshot_download",
                        lambda repo_id, revision=None: calls.append((repo_id, revision)) or str(tmp_path))
    assert resolve_model_dir("hf://someone/model@v1") == tmp_path
    assert resolve_model_dir("hf://someone/model") == tmp_path
    assert calls == [("someone/model", "v1"), ("someone/model", None)]


def test_evaluate_rules(records):
    report = evaluate_rules(records)
    assert 0 <= report["macro_f1"] <= 1