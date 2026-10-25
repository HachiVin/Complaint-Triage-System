from __future__ import annotations

import pytest

from complaint_triage.sqlguard import MAX_ROWS, UnsafeQuery, check


@pytest.mark.parametrize("sql", [
    "SELECT aspect, count(*) FROM v_reviews GROUP BY aspect",
    "select * from v_reviews where text ilike '%drop table%';",
    "SELECT EXTRACT(MONTH FROM review_date) AS m, count(*) FROM v_reviews GROUP BY 1",
    "SELECT substring(text FROM 1 FOR 50) FROM v_reviews",
    "WITH recent AS (SELECT * FROM v_reviews WHERE review_date > DATE '2026-09-01') "
    "SELECT app, count(*) FROM recent GROUP BY app",
    "SELECT a.app FROM v_reviews a JOIN v_reviews b ON a.review_id = b.review_id",
])
def test_allowed(sql):
    out = check(sql)
    assert out.endswith(f"LIMIT {MAX_ROWS}")
    assert ";" not in out


@pytest.mark.parametrize("sql, reason", [
    ("", "empty"),
    ("DELETE FROM v_reviews", "only SELECT"),
    ("SELECT 1; DROP TABLE reviews", "one statement"),
    ("SELECT * FROM reviews", "not available"),
    ("SELECT * FROM labels JOIN v_reviews USING (review_id)", "not available"),
    ("SELECT * FROM pg_catalog.pg_user", "not available"),
    ("SELECT pg_sleep(10) FROM v_reviews", "not allowed"),
    ("SELECT current_setting('server_version') FROM v_reviews", "not allowed"),
    ("WITH x AS (DELETE FROM reviews RETURNING *) SELECT * FROM x", "not allowed"),
    ("SELECT 1", "must read from v_reviews"),
])
def test_rejected(sql, reason):
    with pytest.raises(UnsafeQuery, match=reason):
        check(sql)


def test_keywords_inside_comments_and_strings_are_ignored():
    check("SELECT count(*) FROM v_reviews -- delete later\n WHERE text ILIKE '%update%'")
