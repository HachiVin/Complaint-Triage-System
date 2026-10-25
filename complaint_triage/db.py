"""PostgreSQL access: schema, ingestion, prediction storage, dashboard queries and the agent's
read-only query path. Plain SQL throughout, so every query can be read and run in psql."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from .analyzer import Analysis
from .data import Review
from .sqlguard import check

SQL_DIR = Path(__file__).resolve().parent.parent / "sql"


def external_id(review: Review) -> str:
    return review.id or hashlib.sha1(review.text.encode("utf-8")).hexdigest()[:16]


class Database:
    def __init__(self, dsn: str, agent_dsn: str | None = None, min_size: int = 1, max_size: int = 5):
        from psycopg_pool import ConnectionPool

        self.pool = ConnectionPool(dsn, min_size=min_size, max_size=max_size, open=True)
        self.agent_pool = ConnectionPool(agent_dsn, min_size=1, max_size=2, open=True) if agent_dsn else self.pool

    def close(self) -> None:
        if self.agent_pool is not self.pool:
            self.agent_pool.close()
        self.pool.close()

    def apply_schema(self) -> None:
        with self.pool.connection() as conn:
            conn.execute((SQL_DIR / "schema.sql").read_text(encoding="utf-8"))

    def ping(self) -> bool:
        try:
            with self.pool.connection() as conn:
                conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    # ---- writes ---------------------------------------------------------------------------------------

    def upsert_reviews(self, reviews: list[Review], labelled_by: str = "dataset") -> list[int]:
        """Insert reviews (ignoring ones already present) and their labels. Returns database ids in order."""
        ids = []
        with self.pool.connection() as conn, conn.cursor() as cur:
            for r in reviews:
                cur.execute(
                    """
                    INSERT INTO reviews (source, external_id, app, text, rating, review_date, app_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (source, external_id) DO UPDATE SET text = EXCLUDED.text
                    RETURNING id
                    """,
                    (r.source, external_id(r), r.app, r.text, r.rating, r.date, r.app_version),
                )
                rid = cur.fetchone()[0]
                ids.append(rid)
                if r.aspect or r.sentiment or r.urgency:
                    cur.execute(
                        """
                        INSERT INTO labels (review_id, aspect, sentiment, urgency, labelled_by)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (review_id) DO UPDATE SET
                            aspect = COALESCE(EXCLUDED.aspect, labels.aspect),
                            sentiment = COALESCE(EXCLUDED.sentiment, labels.sentiment),
                            urgency = COALESCE(EXCLUDED.urgency, labels.urgency),
                            labelled_by = EXCLUDED.labelled_by, updated_at = now()
                        """,
                        (rid, r.aspect, r.sentiment, r.urgency, labelled_by),
                    )
        return ids

    def save_predictions(self, review_ids: list[int], results: list[Analysis]) -> None:
        rows = [(rid, a.model_version, a.aspect.label, a.aspect.confidence, a.sentiment.label, a.sentiment.confidence,
                 a.urgency.label, a.urgency.confidence, a.urgency.source, a.needs_review)
                for rid, a in zip(review_ids, results, strict=True)]
        with self.pool.connection() as conn, conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO predictions (review_id, model_version, aspect, aspect_conf, sentiment, sentiment_conf,
                                         urgency, urgency_conf, urgency_source, needs_review)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (review_id, model_version) DO UPDATE SET
                    aspect = EXCLUDED.aspect, aspect_conf = EXCLUDED.aspect_conf,
                    sentiment = EXCLUDED.sentiment, sentiment_conf = EXCLUDED.sentiment_conf,
                    urgency = EXCLUDED.urgency, urgency_conf = EXCLUDED.urgency_conf,
                    urgency_source = EXCLUDED.urgency_source, needs_review = EXCLUDED.needs_review,
                    predicted_at = now()
                """,
                rows,
            )

    def correct(self, review_id: int, task: str, corrected: str) -> None:
        """A CX agent fixes a label: logged as a correction and stored as ground truth."""
        with self.pool.connection() as conn:
            row = conn.execute(f"SELECT {task}, model_version FROM v_reviews WHERE review_id = %s",  # noqa: S608
                               (review_id,)).fetchone()
            if row is None:
                raise KeyError("review not found")
            conn.execute("INSERT INTO corrections (review_id, task, predicted, corrected, model_version) "
                         "VALUES (%s, %s, %s, %s, %s)", (review_id, task, row[0], corrected, row[1]))
            conn.execute(
                f"""
                INSERT INTO labels (review_id, {task}, labelled_by) VALUES (%s, %s, 'cx')
                ON CONFLICT (review_id) DO UPDATE SET {task} = EXCLUDED.{task}, labelled_by = 'cx', updated_at = now()
                """,  # noqa: S608 - task is validated by the caller against labels.TASKS
                (review_id, corrected),
            )

    def log_agent(self, question: str, queries: list[str], answer: str | None, ok: bool, error: str | None,
                  steps: int, latency_ms: float, prompt_tokens: int, completion_tokens: int, llm_model: str) -> int:
        with self.pool.connection() as conn:
            return conn.execute(
                """
                INSERT INTO agent_log (question, queries, answer, ok, error, steps, latency_ms, prompt_tokens,
                                       completion_tokens, llm_model)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id
                """,
                (question, json.dumps(queries), answer, ok, error, steps, latency_ms, prompt_tokens,
                 completion_tokens, llm_model),
            ).fetchone()[0]

    # ---- reads for the dashboard ----------------------------------------------------------------------

    def apps(self) -> list[str]:
        with self.pool.connection() as conn:
            return [r[0] for r in conn.execute("SELECT DISTINCT app FROM v_reviews ORDER BY app").fetchall()]

    def latest_date(self, app: str | None) -> date | None:
        with self.pool.connection() as conn:
            return conn.execute("SELECT max(review_date) FROM v_reviews WHERE (%s::text IS NULL OR app = %s)",
                                (app, app)).fetchone()[0]

    def overview(self, app: str | None, weeks: int) -> dict:
        """KPIs for the last ``weeks`` weeks of data (anchored on the newest review, not today)."""
        end = self.latest_date(app)
        if end is None:
            return {"reviews": 0, "weeks": weeks}
        params = {"app": app, "end": end, "days": weeks * 7}
        where = "(%(app)s::text IS NULL OR app = %(app)s) AND review_date > %(end)s::date - %(days)s"
        with self.pool.connection() as conn:
            k = conn.execute(
                f"""
                SELECT count(*), avg((sentiment = 'negatif')::int), count(*) FILTER (WHERE urgency = 'kritis'),
                       count(*) FILTER (WHERE urgency = 'tinggi'), avg(rating), avg(needs_review::int)
                FROM v_reviews WHERE {where}
                """, params).fetchone()
            prev = conn.execute(
                """
                SELECT count(*) FILTER (WHERE urgency = 'kritis') FROM v_reviews
                WHERE (%(app)s::text IS NULL OR app = %(app)s)
                  AND review_date >  %(end)s::date - 2 * %(days)s AND review_date <= %(end)s::date - %(days)s
                """, params).fetchone()[0]
            aspects = conn.execute(
                f"""
                SELECT aspect, count(*) AS n, avg((sentiment = 'negatif')::int) AS neg,
                       count(*) FILTER (WHERE urgency = 'kritis') AS kritis
                FROM v_reviews WHERE {where} GROUP BY aspect ORDER BY n DESC
                """, params).fetchall()
            weekly = conn.execute(
                f"""
                SELECT date_trunc('week', review_date)::date AS week, aspect, count(*)
                FROM v_reviews WHERE {where} AND sentiment = 'negatif'
                GROUP BY 1, 2 ORDER BY 1, 2
                """, params).fetchall()
            versions = conn.execute(
                f"""
                SELECT app_version, count(*), avg((sentiment = 'negatif')::int),
                       count(*) FILTER (WHERE urgency = 'kritis')
                FROM v_reviews WHERE {where} AND app_version IS NOT NULL
                GROUP BY app_version HAVING count(*) >= 5 ORDER BY 2 DESC LIMIT 8
                """, params).fetchall()
        return {
            "app": app, "weeks": weeks, "period_end": end.isoformat(),
            "reviews": k[0], "negative_share": round(float(k[1] or 0), 3), "kritis": k[2], "tinggi": k[3],
            "kritis_previous_period": prev, "avg_rating": round(float(k[4]), 2) if k[4] is not None else None,
            "needs_review_share": round(float(k[5] or 0), 3),
            "aspects": [{"aspect": a, "reviews": n, "negative_share": round(float(neg), 3), "kritis": kr}
                        for a, n, neg, kr in aspects],
            "weekly_negative": [{"week": w.isoformat(), "aspect": a, "reviews": n} for w, a, n in weekly],
            "versions": [{"app_version": v, "reviews": n, "negative_share": round(float(neg), 3), "kritis": kr}
                         for v, n, neg, kr in versions],
        }

    def queue(self, app: str | None, urgency: str, limit: int = 50, aspect: str | None = None) -> list[dict]:
        with self.pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT review_id, app, review_date, rating, app_version, aspect, sentiment, urgency, urgency_conf,
                       needs_review, text
                FROM v_reviews
                WHERE urgency = %(u)s AND (%(app)s::text IS NULL OR app = %(app)s)
                  AND (%(aspect)s::text IS NULL OR aspect = %(aspect)s)
                ORDER BY review_date DESC NULLS LAST, urgency_conf DESC
                LIMIT %(limit)s
                """, {"u": urgency, "app": app, "aspect": aspect, "limit": limit}).fetchall()
        keys = ("review_id", "app", "review_date", "rating", "app_version", "aspect", "sentiment", "urgency",
                "urgency_conf", "needs_review", "text")
        out = [dict(zip(keys, r, strict=True)) for r in rows]
        for r in out:
            r["review_date"] = r["review_date"].isoformat() if r["review_date"] else None
        return out

    def model_quality(self) -> dict:
        """Live accuracy per model version from CX corrections and from rows that have ground truth."""
        with self.pool.connection() as conn:
            rows = conn.execute(
                """
                SELECT v.model_version,
                       count(*) FILTER (WHERE l.aspect IS NOT NULL)                         AS aspect_n,
                       avg((v.aspect = l.aspect)::int) FILTER (WHERE l.aspect IS NOT NULL)       AS aspect_acc,
                       count(*) FILTER (WHERE l.urgency IS NOT NULL)                        AS urgency_n,
                       avg((v.urgency = l.urgency)::int) FILTER (WHERE l.urgency IS NOT NULL)    AS urgency_acc
                FROM v_reviews v JOIN labels l ON l.review_id = v.review_id
                GROUP BY v.model_version ORDER BY v.model_version
                """).fetchall()
            corrections = conn.execute(
                "SELECT model_version, task, count(*) FROM corrections GROUP BY 1, 2 ORDER BY 1, 2").fetchall()
        return {
            "by_version": [{"model_version": v, "aspect_labelled": an,
                            "aspect_accuracy": round(float(aa), 3) if aa is not None else None,
                            "urgency_labelled": un,
                            "urgency_accuracy": round(float(ua), 3) if ua is not None else None}
                           for v, an, aa, un, ua in rows],
            "corrections": [{"model_version": v, "task": t, "count": n} for v, t, n in corrections],
        }

    # ---- the agent's only path to data ---------------------------------------------------------------

    def run_readonly(self, sql: str, timeout_ms: int = 5000) -> tuple[list[str], list[list]]:
        """Validate, then run in a READ ONLY transaction with a timeout. Returns (columns, rows)."""
        safe = check(sql)
        with self.agent_pool.connection() as conn:
            with conn.transaction():
                conn.execute("SET TRANSACTION READ ONLY")
                conn.execute(f"SET LOCAL statement_timeout = {int(timeout_ms)}")
                cur = conn.execute(safe)
                cols = [d.name for d in cur.description]
                rows = [[_jsonable(v) for v in row] for row in cur.fetchall()]
        return cols, rows


def _jsonable(v):
    if isinstance(v, date):
        return v.isoformat()
    if hasattr(v, "__float__") and not isinstance(v, (int, float, bool)):
        return float(v)
    return v
