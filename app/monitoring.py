"""In-process service metrics and a per-client rate limiter. Long-term history lives in PostgreSQL
(predictions, corrections, agent_log)."""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque

import numpy as np


class Metrics:
    def __init__(self, window: int = 1000):
        self._lock = threading.Lock()
        self.started = time.time()
        self.requests: Counter[str] = Counter()
        self.errors: Counter[str] = Counter()
        self.rate_limited = 0
        self.latency: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self.urgency: Counter[str] = Counter()
        self.agent = {"runs": 0, "ok": 0, "tokens": 0}

    def record(self, endpoint: str, ms: float) -> None:
        with self._lock:
            self.requests[endpoint] += 1
            self.latency[endpoint].append(ms)

    def error(self, endpoint: str) -> None:
        with self._lock:
            self.errors[endpoint] += 1

    def limited(self) -> None:
        with self._lock:
            self.rate_limited += 1

    def classified(self, urgency: str) -> None:
        with self._lock:
            self.urgency[urgency] += 1

    def agent_run(self, ok: bool, tokens: int) -> None:
        with self._lock:
            self.agent["runs"] += 1
            self.agent["ok"] += int(ok)
            self.agent["tokens"] += tokens

    def snapshot(self) -> dict:
        with self._lock:
            lat = {k: {"p50": round(float(np.percentile(v, 50)), 1), "p95": round(float(np.percentile(v, 95)), 1)}
                   for k, v in self.latency.items() if v}
            runs = self.agent["runs"]
            return {
                "uptime_s": round(time.time() - self.started), "requests": dict(self.requests),
                "errors": dict(self.errors), "rate_limited": self.rate_limited, "latency_ms": lat,
                "classified_urgency": dict(self.urgency),
                "agent": {**self.agent, "success_rate": round(self.agent["ok"] / runs, 3) if runs else None,
                          "mean_tokens": round(self.agent["tokens"] / runs, 1) if runs else None},
            }


class RateLimiter:
    """Sliding one-minute window per client and bucket (LLM calls are limited separately)."""

    def __init__(self, per_minute: dict[str, int]):
        self.per_minute = per_minute
        self._hits: dict[tuple[str, str], deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, bucket: str, client: str) -> bool:
        limit = self.per_minute.get(bucket, 0)
        if limit <= 0:
            return True
        now = time.monotonic()
        with self._lock:
            hits = self._hits.setdefault((bucket, client), deque())
            while hits and now - hits[0] > 60:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True
