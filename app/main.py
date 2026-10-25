"""Complaint Triage System API and dashboard.

Environment:
    MODEL_DIR            model directory (baseline or multitask), default models/baseline;
                         hf://owner/repo loads a model published on the Hugging Face Hub
    DATABASE_URL         PostgreSQL; without it only /api/classify works
    AGENT_DATABASE_URL   optional read-only DSN for the agent (role agent_ro, sql/readonly_role.sql)
    LLM_API_KEY          enables /api/ask (LLM_BASE_URL and LLM_MODEL optional, default Groq gpt-oss-20b)
    SEED_DEMO            1 = load the fictional demo reviews when the database is empty
    RATE_LIMIT_CLASSIFY  requests per client per minute, default 60
    RATE_LIMIT_ASK       agent questions per client per minute, default 5
    ALLOW_CORRECTIONS    0 = public demo: label corrections are refused (default 1)
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from complaint_triage.analyzer import Analysis, Analyzer, load_model
from complaint_triage.labels import LABEL_INFO, TASKS

from .monitoring import Metrics, RateLimiter

log = logging.getLogger("complaint_triage")
STATIC = Path(__file__).parent / "static"
ROOT = Path(__file__).resolve().parent.parent


class ClassifyRequest(BaseModel):
    text: str = Field(min_length=1, max_length=3000)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    answer: str | None
    ok: bool
    error: str | None
    queries: list[str]
    steps: int
    latency_ms: float


class CorrectRequest(BaseModel):
    review_id: int = Field(gt=0)
    task: Literal["aspect", "sentiment", "urgency"]
    label: str


def seed_demo(db, analyzer: Analyzer) -> int:
    from complaint_triage.data import load_jsonl

    with db.pool.connection() as conn:
        if conn.execute("SELECT count(*) FROM reviews").fetchone()[0]:
            return 0
    rows = load_jsonl(ROOT / "data" / "sample" / "demo_reviews.jsonl")
    ids = db.upsert_reviews(rows)
    db.save_predictions(ids, analyzer.analyze_batch([r.text for r in rows]))
    return len(rows)


def create_app(analyzer: Analyzer | None = None, db=None, agent=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.analyzer = analyzer or Analyzer(load_model(os.getenv("MODEL_DIR", "models/baseline")))
        app.state.db = db
        if app.state.db is None and os.getenv("DATABASE_URL"):
            from complaint_triage.db import Database

            app.state.db = Database(os.environ["DATABASE_URL"], agent_dsn=os.getenv("AGENT_DATABASE_URL"))
            app.state.db.apply_schema()
            if os.getenv("SEED_DEMO") == "1":
                n = seed_demo(app.state.db, app.state.analyzer)
                if n:
                    log.info("seeded %d demo reviews", n)
        app.state.agent = agent
        if app.state.agent is None and app.state.db is not None and os.getenv("LLM_API_KEY"):
            from complaint_triage.agent import AnalystAgent

            app.state.agent = AnalystAgent(app.state.db)
        yield
        if app.state.db is not None and db is None:
            app.state.db.close()

    app = FastAPI(title="Complaint Triage System API", version="0.1.0", lifespan=lifespan,
                  description="Analisis keluhan nasabah mobile banking: aspek, sentimen, urgensi, dan agent analis.")
    app.state.metrics = Metrics()
    app.state.limiter = RateLimiter({"classify": int(os.getenv("RATE_LIMIT_CLASSIFY", "60")),
                                     "ask": int(os.getenv("RATE_LIMIT_ASK", "5"))})

    def require_db():
        if app.state.db is None:
            raise HTTPException(503, "database not configured (set DATABASE_URL)")
        return app.state.db

    def limit(bucket: str, request: Request):
        client = request.client.host if request.client else "unknown"
        if not app.state.limiter.allow(bucket, client):
            app.state.metrics.limited()
            raise HTTPException(429, "Terlalu banyak permintaan, coba lagi sebentar lagi.")

    @app.post("/api/classify", response_model=Analysis)
    def classify(req: ClassifyRequest, request: Request):
        limit("classify", request)
        t0 = time.perf_counter()
        result = app.state.analyzer.analyze(req.text)
        app.state.metrics.record("classify", (time.perf_counter() - t0) * 1000)
        app.state.metrics.classified(result.urgency.label)
        return result

    @app.get("/api/apps")
    def apps():
        return require_db().apps()

    @app.get("/api/overview")
    def overview(app_name: str | None = Query(None, alias="app"), weeks: int = Query(4, ge=1, le=52)):
        t0 = time.perf_counter()
        out = require_db().overview(app_name or None, weeks)
        app.state.metrics.record("overview", (time.perf_counter() - t0) * 1000)
        return out

    @app.get("/api/queue")
    def queue(app_name: str | None = Query(None, alias="app"),
              urgency: Literal["kritis", "tinggi", "normal"] = "kritis", aspect: str | None = None,
              limit_rows: int = Query(50, alias="limit", ge=1, le=200)):
        if aspect and aspect not in TASKS["aspect"]:
            raise HTTPException(422, f"aspect must be one of {list(TASKS['aspect'])}")
        return require_db().queue(app_name or None, urgency, limit_rows, aspect)

    @app.post("/api/correct", status_code=204)
    def correct(req: CorrectRequest):
        if os.getenv("ALLOW_CORRECTIONS", "1") == "0":
            raise HTTPException(403, "Mode demo publik: koreksi label dimatikan.")
        if req.label not in TASKS[req.task]:
            raise HTTPException(422, f"label must be one of {list(TASKS[req.task])}")
        try:
            require_db().correct(req.review_id, req.task, req.label)
        except KeyError:
            raise HTTPException(404, "review not found") from None

    @app.get("/api/quality")
    def quality():
        return require_db().model_quality()

    @app.post("/api/ask", response_model=AskResponse)
    def ask(req: AskRequest, request: Request):
        if app.state.agent is None:
            raise HTTPException(503, "agent not configured (needs DATABASE_URL and LLM_API_KEY)")
        limit("ask", request)
        run = app.state.agent.ask(req.question)
        app.state.metrics.record("ask", run.latency_ms)
        app.state.metrics.agent_run(run.ok, run.prompt_tokens + run.completion_tokens)
        if not run.ok:
            app.state.metrics.error("ask")
        return AskResponse(answer=run.answer, ok=run.ok, error=run.error, queries=run.queries, steps=run.steps,
                           latency_ms=run.latency_ms)

    @app.get("/api/labels")
    def labels():
        return {task: [{"key": x.key, "name": x.name, "description": x.description} for x in info.values()]
                for task, info in LABEL_INFO.items()}

    @app.get("/health")
    def health():
        db_state = "not_configured" if app.state.db is None else app.state.db.ping()
        version = app.state.analyzer.version
        return {"status": "ok" if db_state is not False else "degraded", "model_version": version,
                "demo_model": "demo" in version, "database": db_state, "agent": app.state.agent is not None,
                "corrections": os.getenv("ALLOW_CORRECTIONS", "1") != "0"}

    @app.get("/metrics")
    def metrics():
        return {"model_version": app.state.analyzer.version, **app.state.metrics.snapshot()}

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app


app = create_app()