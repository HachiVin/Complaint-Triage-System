"""Analyst agent: answers business questions about reviews by writing SQL against ``v_reviews``.

A small LangGraph graph: the LLM either calls a tool or answers. Tools:

* ``run_sql``: read-only query through ``sqlguard`` (see db.run_readonly). Errors are returned to the
  LLM as text so it can fix its own query.
* ``label_guide``: the meaning of every aspect, sentiment and urgency value, so the LLM filters with
  the right keys instead of guessing.

The LLM is any OpenAI-compatible endpoint (default Groq, free tier). Every run is logged to
``agent_log`` with its queries, steps, latency and tokens.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .labels import ASPECT_LABELS, SENTIMENT_LABELS, URGENCY_LABELS
from .sqlguard import MAX_ROWS, UnsafeQuery

SCHEMA_DOC = f"""
View v_reviews (one row per review, with its latest model prediction):
  review_id bigint, app text, text text, rating smallint (1-5, may be null), review_date date,
  app_version text (may be null), aspect text, sentiment text, urgency text,
  aspect_conf real, sentiment_conf real, urgency_conf real, needs_review boolean, model_version text
aspect values: {", ".join(x.key for x in ASPECT_LABELS)}
sentiment values: {", ".join(x.key for x in SENTIMENT_LABELS)}
urgency values: {", ".join(x.key for x in URGENCY_LABELS)}
Results are capped at {MAX_ROWS} rows.
""".strip()

SYSTEM_PROMPT = f"""Kamu analis data untuk tim customer experience sebuah bank. Jawab pertanyaan tentang ulasan
aplikasi mobile banking dengan menulis query PostgreSQL ke view v_reviews memakai tool run_sql.

{SCHEMA_DOC}

Aturan:
- Selalu ambil angka dari query, jangan menebak. Kalau datanya tidak cukup, katakan begitu.
- "Minggu ini" atau "bulan ini" dihitung dari tanggal ulasan terbaru (SELECT max(review_date) FROM v_reviews),
  bukan dari tanggal hari ini, karena data bisa saja bukan data hari ini.
- "N hari terakhir" berarti N hari kalender termasuk tanggal terbaru:
  review_date > max(review_date) - N (bukan >=, yang menghasilkan N+1 hari). Tulis rentang tanggalnya.
- Saat memberi contoh ulasan, kutip teksnya apa adanya dan sebutkan review_id.
- app_version kosong (NULL) artinya versinya tidak tercatat, bukan sebuah versi. Saat membandingkan atau
  meranking versi aplikasi, saring dengan app_version IS NOT NULL; kalau jumlah yang tidak tercatat besar,
  sebutkan terpisah.
- Jawab dalam bahasa Indonesia, ringkas, dengan angka. Sebutkan periode dan jumlah ulasan yang dipakai.
- Tulis angka dengan format Indonesia: titik untuk ribuan dan koma untuk desimal (14.020; 46,3%).
- Hanya pertanyaan tentang data ulasan yang dijawab; tolak permintaan lain dengan sopan.
"""


@dataclass
class AgentRun:
    question: str
    answer: str | None = None
    queries: list[str] = field(default_factory=list)
    steps: int = 0
    ok: bool = False
    error: str | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    llm_model: str = ""


def default_llm():
    from langchain_openai import ChatOpenAI

    key = os.getenv("LLM_API_KEY")
    if not key:
        raise RuntimeError("LLM_API_KEY is not set")
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-20b"),
        base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"),
        api_key=key, temperature=0, timeout=60, max_retries=2,
    )


class AnalystAgent:
    def __init__(self, db, llm=None, max_steps: int = 8):
        self.db = db
        self.llm = llm or default_llm()
        self.max_steps = max_steps
        self.graph = self._build()

    def _tools(self):
        agent = self

        @tool
        def run_sql(query: str) -> str:
            """Run one read-only PostgreSQL SELECT against the view v_reviews and return JSON rows."""
            try:
                cols, rows = agent.db.run_readonly(query)
            except UnsafeQuery as exc:
                return f"QUERY REJECTED: {exc}"
            except Exception as exc:  # syntax errors, timeouts: let the model fix its query
                return f"QUERY ERROR: {type(exc).__name__}: {str(exc).splitlines()[0]}"
            return json.dumps({"columns": cols, "rows": rows, "row_count": len(rows)}, ensure_ascii=False,
                              default=str)

        @tool
        def label_guide() -> str:
            """Describe every aspect, sentiment and urgency value used in v_reviews."""
            lines = []
            for title, labels in (("aspect", ASPECT_LABELS), ("sentiment", SENTIMENT_LABELS),
                                  ("urgency", URGENCY_LABELS)):
                lines += [f"{title}={x.key}: {x.description}" for x in labels]
            return "\n".join(lines)

        return [run_sql, label_guide]

    def _build(self):
        tools = self._tools()
        model = self.llm.bind_tools(tools)

        def call_model(state: MessagesState):
            return {"messages": [model.invoke(state["messages"])]}

        g = StateGraph(MessagesState)
        g.add_node("agent", call_model)
        g.add_node("tools", ToolNode(tools))
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", tools_condition)
        g.add_edge("tools", "agent")
        return g.compile()

    def ask(self, question: str, log: bool = True) -> AgentRun:
        run = AgentRun(question=question, llm_model=getattr(self.llm, "model_name", type(self.llm).__name__))
        started = time.perf_counter()
        try:
            state = self.graph.invoke(
                {"messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]},
                config={"recursion_limit": 2 * self.max_steps + 1},
            )
            messages = state["messages"]
            for m in messages:
                if isinstance(m, AIMessage):
                    run.steps += 1
                    usage = getattr(m, "usage_metadata", None) or {}
                    run.prompt_tokens += usage.get("input_tokens", 0)
                    run.completion_tokens += usage.get("output_tokens", 0)
                    for call in m.tool_calls or []:
                        if call["name"] == "run_sql":
                            run.queries.append(call["args"].get("query", ""))
            final = messages[-1]
            if isinstance(final, AIMessage) and not final.tool_calls:
                run.answer = final.content if isinstance(final.content, str) else str(final.content)
                run.ok = bool(run.answer.strip())
            elif isinstance(final, ToolMessage):
                run.error = "agent stopped after a tool call"
        except Exception as exc:
            name = type(exc).__name__
            run.error = "step limit reached" if name == "GraphRecursionError" else f"{name}: {exc}"
        run.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        if log:
            try:
                self.db.log_agent(run.question, run.queries, run.answer, run.ok, run.error, run.steps,
                                  run.latency_ms, run.prompt_tokens, run.completion_tokens, run.llm_model)
            except Exception:
                pass
        return run