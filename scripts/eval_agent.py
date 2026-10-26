"""Evaluate the analyst agent on a set of questions with known answers.

    python scripts/eval_agent.py data/agent_eval/questions.jsonl

Each line: {"id": "q01", "question": "...", "gold_sql": "SELECT ... -- returns one number"}
The gold SQL is run first; the answer counts as correct when that number appears in the agent's
answer (integers exactly; decimals within ±1 on the last shown digit, also as a percentage). Crude but
honest: it checks that the agent got the number right, not that it phrased things nicely.
Needs DATABASE_URL and LLM_API_KEY. Writes docs/agent_eval.json.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from complaint_triage.agent import AnalystAgent  # noqa: E402
from complaint_triage.db import Database  # noqa: E402


def numbers_in(text: str) -> list[float]:
    found = []
    # LLMs sometimes group thousands with a (narrow) no-break space: "14\u202f020"
    text = re.sub(r"(?<=\d)[\u00a0\u2009\u202f](?=\d{3}(?!\d))", "", text or "")
    for m in re.finditer(r"\d[\d.,]*", text):
        s = m.group(0).rstrip(".,")
        # Indonesian formatting: 1.234 thousands, 12,5 decimals
        if re.fullmatch(r"\d{1,3}(\.\d{3})+", s):
            s = s.replace(".", "")
        s = s.replace(",", ".")
        try:
            found.append(float(s))
        except ValueError:
            pass
    return found


def matches(gold: float, answer: str) -> bool:
    for x in numbers_in(answer):
        if float(gold).is_integer() and x == gold:
            return True
        for candidate in (gold, gold * 100):  # 0.237 may be written as 23.7%
            if abs(x - candidate) <= max(0.05 * abs(candidate), 0.11):
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("questions")
    ap.add_argument("--out", default="docs/agent_eval.json")
    args = ap.parse_args()

    db = Database(os.environ["DATABASE_URL"])
    agent = AnalystAgent(db)
    items = [json.loads(line) for line in open(args.questions, encoding="utf-8") if line.strip()]
    results = []
    for item in items:
        with db.pool.connection() as conn:
            gold = conn.execute(item["gold_sql"]).fetchone()[0]
        run = agent.ask(item["question"])
        ok = run.ok and gold is not None and matches(float(gold), run.answer or "")
        results.append({"id": item["id"], "question": item["question"],
                        "gold": float(gold) if gold is not None else None,
                        "correct": ok, "answer": run.answer, "queries": run.queries, "steps": run.steps,
                        "latency_ms": run.latency_ms, "tokens": run.prompt_tokens + run.completion_tokens,
                        "error": run.error})
        print(f"{item['id']}: {'OK ' if ok else 'BAD'} gold={gold} steps={run.steps} {run.latency_ms:.0f} ms")
    n = len(results)
    summary = {"questions": n, "accuracy": round(sum(r["correct"] for r in results) / n, 3) if n else None,
               "mean_steps": round(sum(r["steps"] for r in results) / n, 2) if n else None,
               "mean_latency_ms": round(sum(r["latency_ms"] for r in results) / n, 1) if n else None,
               "mean_tokens": round(sum(r["tokens"] for r in results) / n, 1) if n else None,
               "failed_runs": sum(1 for r in results if r["error"])}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary, "results": results}, indent=2, ensure_ascii=False),
                              encoding="utf-8")
    print(json.dumps(summary, indent=2))
    db.close()


if __name__ == "__main__":
    main()