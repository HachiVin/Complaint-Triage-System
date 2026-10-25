from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

from app.main import create_app
from complaint_triage.analyzer import Analyzer


@pytest.fixture
def client(baseline):
    with TestClient(create_app(analyzer=Analyzer(baseline))) as c:
        yield c


def test_classify(client):
    r = client.post("/api/classify", json={"text": "Transfer gagal tapi saldo terpotong sudah 3 hari"})
    assert r.status_code == 200
    body = r.json()
    assert body["urgency"]["label"] == "kritis"
    assert body["aspect"]["label"] == "kualitas_transaksi"


def test_classify_validation(client):
    assert client.post("/api/classify", json={"text": ""}).status_code == 422


def test_without_database(client):
    assert client.get("/health").json()["database"] == "not_configured"
    assert client.get("/api/overview").status_code == 503
    assert client.post("/api/ask", json={"question": "berapa ulasan?"}).status_code == 503


def test_labels_and_index(client):
    labels = client.get("/api/labels").json()
    assert {x["key"] for x in labels["urgency"]} == {"kritis", "tinggi", "normal"}
    assert "Complaint Triage System" in client.get("/").text


def test_corrections_can_be_switched_off(baseline, monkeypatch):
    monkeypatch.setenv("ALLOW_CORRECTIONS", "0")
    with TestClient(create_app(analyzer=Analyzer(baseline))) as c:
        assert c.get("/health").json()["corrections"] is False
        r = c.post("/api/correct", json={"review_id": 1, "task": "urgency", "label": "normal"})
        assert r.status_code == 403


def test_rate_limit(baseline, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_CLASSIFY", "2")
    with TestClient(create_app(analyzer=Analyzer(baseline))) as c:
        codes = [c.post("/api/classify", json={"text": "aplikasi lambat"}).status_code for _ in range(3)]
        assert codes == [200, 200, 429]
        assert c.get("/metrics").json()["rate_limited"] == 1


# ---- with PostgreSQL ------------------------------------------------------------------------------------

@pytest.fixture
def db_client(loaded_db, baseline, scripted_llm):
    from complaint_triage.agent import AnalystAgent

    llm = scripted_llm(responses=[
        AIMessage(content="", tool_calls=[{"name": "run_sql", "type": "tool_call", "id": "c1",
                                           "args": {"query": "SELECT count(*) AS n FROM v_reviews"}}]),
        AIMessage(content="Total 448 ulasan."),
    ])
    agent = AnalystAgent(loaded_db, llm=llm)
    with TestClient(create_app(analyzer=Analyzer(baseline), db=loaded_db, agent=agent)) as c:
        yield c


def test_overview(db_client):
    o = db_client.get("/api/overview", params={"weeks": 4}).json()
    assert o["reviews"] > 0 and o["period_end"]
    assert o["kritis"] >= 0 and o["aspects"]
    assert all(set(w) == {"week", "aspect", "reviews"} for w in o["weekly_negative"])
    one = db_client.get("/api/overview", params={"app": "KitaPay", "weeks": 4}).json()
    assert one["reviews"] < o["reviews"]


def test_planted_incident_shows_up(db_client):
    """The demo data has a payment incident in DemoBank 3.2.0; the version table must surface it."""
    o = db_client.get("/api/overview", params={"app": "DemoBank Mobile", "weeks": 12}).json()
    worst = max(o["versions"], key=lambda v: v["kritis"])
    assert worst["app_version"] == "3.2.0"


def test_queue_and_correction(db_client, loaded_db):
    rows = db_client.get("/api/queue", params={"urgency": "kritis", "limit": 5}).json()
    assert rows and all(r["urgency"] == "kritis" for r in rows)
    rid = rows[0]["review_id"]
    def fix(review_id, label):
        return db_client.post("/api/correct", json={"review_id": review_id, "task": "urgency", "label": label})

    assert fix(rid, "normal").status_code == 204
    assert fix(rid, "x").status_code == 422
    assert fix(10**9, "normal").status_code == 404
    quality = db_client.get("/api/quality").json()
    assert any(c["task"] == "urgency" for c in quality["corrections"])


def test_ask(db_client):
    r = db_client.post("/api/ask", json={"question": "berapa total ulasan?"}).json()
    assert r["ok"] and r["queries"] == ["SELECT count(*) AS n FROM v_reviews"]
    assert db_client.get("/metrics").json()["agent"]["runs"] == 1


def test_upsert_returns_ids_in_input_order(loaded_db, records):
    ids = loaded_db.upsert_reviews(records[:30])
    with loaded_db.pool.connection() as conn:
        texts = dict(conn.execute("SELECT id, text FROM reviews WHERE id = ANY(%s)", (ids,)).fetchall())
    assert [texts[i] for i in ids] == [r.text for r in records[:30]]
    assert len(set(ids)) == len({(r.source, r.id) for r in records[:30]})


def test_readonly_path_blocks_writes(loaded_db):
    from complaint_triage.sqlguard import UnsafeQuery

    cols, rows = loaded_db.run_readonly("SELECT app, count(*) AS n FROM v_reviews GROUP BY app ORDER BY app")
    assert cols == ["app", "n"] and len(rows) == 2
    with pytest.raises(UnsafeQuery):
        loaded_db.run_readonly("UPDATE reviews SET text = 'x'")