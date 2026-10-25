from __future__ import annotations

from complaint_triage.analyzer import Analyzer, load_model
from complaint_triage.data import split_stratified
from complaint_triage.labels import TASKS


def test_multitask_trains_saves_and_loads(tmp_path, tiny_bert, records):
    from complaint_triage.multitask import TrainConfig, train_multitask

    parts = split_stratified(records, task="aspect")
    # Urgency labels only on part of the rows: the masked loss must cope with missing labels.
    train = [r if i % 2 else r.model_copy(update={"urgency": None}) for i, r in enumerate(parts["train"])]
    cfg = TrainConfig(model_name=str(tiny_bert), epochs=1, batch_size=16, max_length=32, lr=1e-3)
    out = tmp_path / "mt"
    meta = train_multitask(train, parts["val"], out, cfg)
    assert meta["kind"] == "multitask"
    assert (out / "heads.pt").exists() and (out / "encoder").is_dir()

    model = load_model(out)
    proba = model.predict_proba(["transfer gagal saldo terpotong", "bagus"])
    assert set(proba) == set(TASKS)
    assert proba["aspect"].shape == (2, len(TASKS["aspect"]))

    result = Analyzer(model).analyze("tidak bisa login")
    assert result.urgency.source == "model"
    assert result.model_version.startswith("multitask")
