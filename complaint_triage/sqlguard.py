"""Checks on SQL written by the LLM before it reaches the database.

Defence in depth, three layers:
1. this guard: one statement, SELECT/WITH only, only the ``v_reviews`` view, no dangerous functions,
   and a row limit is always applied;
2. the query runs in a READ ONLY transaction with a statement timeout;
3. in deployment the agent connects as ``agent_ro`` (sql/readonly_role.sql), which can read only the view.
"""

from __future__ import annotations

import re

ALLOWED_RELATIONS = {"v_reviews"}
MAX_ROWS = 200

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|vacuum|analyze|comment|"
    r"call|do|execute|prepare|listen|notify|set|reset|lock|refresh|merge|"
    r"pg_sleep|pg_read_file|pg_ls_dir|lo_import|lo_export|dblink|current_setting|set_config)\b",
    re.IGNORECASE,
)
_RELATION = re.compile(r"\b(?:from|join)\s+([a-zA-Z_][\w.]*)", re.IGNORECASE)
# EXTRACT(MONTH FROM d), SUBSTRING(x FROM 1) and TRIM(... FROM x) use FROM without naming a table.
_FROM_FUNCS = re.compile(r"\b(?:extract|substring|trim|overlay|position)\s*\([^()]*\)", re.IGNORECASE)
_CTE_NAME = re.compile(r"(?:\bwith\s+(?:recursive\s+)?|,\s*)([a-zA-Z_]\w*)\s+as\s*\(", re.IGNORECASE)


class UnsafeQuery(ValueError):
    pass


def _strip_comments_and_strings(sql: str) -> str:
    sql = re.sub(r"--[^\n]*", " ", sql)
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"'(?:''|[^'])*'", "''", sql)


def check(sql: str) -> str:
    """Return the query wrapped with a row limit, or raise ``UnsafeQuery`` with a reason the LLM can use."""
    raw = sql.strip().rstrip(";").strip()
    if not raw:
        raise UnsafeQuery("empty query")
    bare = _strip_comments_and_strings(raw)
    if ";" in bare:
        raise UnsafeQuery("only one statement is allowed")
    if not re.match(r"^\s*(select|with)\b", bare, re.IGNORECASE):
        raise UnsafeQuery("only SELECT queries are allowed")
    bad = _FORBIDDEN.search(bare)
    if bad:
        raise UnsafeQuery(f"keyword or function not allowed: {bad.group(0)}")
    ctes = {m.lower() for m in _CTE_NAME.findall(bare)}
    for rel in _RELATION.findall(_FROM_FUNCS.sub(" fn() ", bare)):
        name = rel.lower().split(".")[-1]
        if name not in ALLOWED_RELATIONS and name not in ctes:
            raise UnsafeQuery(f"table {rel!r} is not available; query v_reviews")
    if "v_reviews" not in bare.lower():
        raise UnsafeQuery("query must read from v_reviews")
    return f"SELECT * FROM ({raw}) AS q LIMIT {MAX_ROWS}"
