from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from complaint_triage.agent import AnalystAgent


class FakeDB:
    def __init__(self):
        self.queries, self.logged = [], []

    def run_readonly(self, sql):
        from complaint_triage.sqlguard import check

        check(sql)
        self.queries.append(sql)
        return ["aspect", "n"], [["kualitas_transaksi", 42]]

    def log_agent(self, *args):
        self.logged.append(args)
        return 1


def call(name, args, cid):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid, "type": "tool_call"}])


def test_agent_queries_then_answers(scripted_llm):
    llm = scripted_llm(responses=[
        call("run_sql", {"query": "SELECT aspect, count(*) AS n FROM v_reviews GROUP BY aspect"}, "c1"),
        AIMessage(content="Keluhan terbanyak: kualitas transaksi (42 ulasan)."),
    ])
    db = FakeDB()
    run = AnalystAgent(db, llm=llm).ask("aspek apa yang paling banyak dikeluhkan?")
    assert run.ok and "42" in run.answer
    assert run.queries == ["SELECT aspect, count(*) AS n FROM v_reviews GROUP BY aspect"]
    assert run.steps == 2
    assert db.logged and db.logged[0][3] is True


def test_rejected_query_is_returned_to_the_model(scripted_llm):
    seen = []

    class Recording(scripted_llm):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append(messages[-1].content)
            return super()._generate(messages, stop, run_manager, **kwargs)

    llm = Recording(responses=[
        call("run_sql", {"query": "DELETE FROM reviews"}, "c1"),
        call("run_sql", {"query": "SELECT count(*) FROM v_reviews"}, "c2"),
        AIMessage(content="Ada 42 ulasan."),
    ])
    run = AnalystAgent(FakeDB(), llm=llm).ask("berapa ulasan?")
    assert run.ok
    assert any("QUERY REJECTED" in str(s) for s in seen)
    assert len(run.queries) == 2


def test_label_guide_tool(scripted_llm):
    llm = scripted_llm(responses=[call("label_guide", {}, "c1"), AIMessage(content="ok")])
    agent = AnalystAgent(FakeDB(), llm=llm)
    run = agent.ask("apa arti kritis?")
    assert run.ok and run.queries == []


def test_step_limit(scripted_llm):
    loop = call("run_sql", {"query": "SELECT count(*) FROM v_reviews"}, "c1")
    run = AnalystAgent(FakeDB(), llm=scripted_llm(responses=[loop]), max_steps=2).ask("loop")
    assert not run.ok and run.error == "step limit reached"


def test_run_sql_output_is_json(scripted_llm):
    agent = AnalystAgent(FakeDB(), llm=scripted_llm(responses=[AIMessage(content="x")]))
    run_sql = agent._tools()[0]
    out = json.loads(run_sql.invoke({"query": "SELECT aspect, count(*) n FROM v_reviews GROUP BY 1"}))
    assert out["row_count"] == 1 and out["columns"] == ["aspect", "n"]
