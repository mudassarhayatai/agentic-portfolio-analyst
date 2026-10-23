import pytest

from src.tools.sql_fallback import run_readonly_sql, validate_readonly_sql

ATTACK_QUERIES = [
    "DROP TABLE transactions",
    "SELECT * FROM transactions; DROP TABLE transactions",
    "ATTACH DATABASE '/tmp/evil.db' AS evil",
    "SELECT * FROM sqlite_master",
    "UPDATE transactions SET price = 0",
    "WITH cte AS (SELECT * FROM transactions) SELECT * FROM cte",
]


@pytest.mark.parametrize("query", ATTACK_QUERIES)
def test_attack_queries_are_rejected(query):
    is_valid, error = validate_readonly_sql(query)
    assert is_valid is False
    assert error  # must always explain why, not just fail silently


def test_valid_select_is_accepted():
    is_valid, error = validate_readonly_sql("SELECT symbol FROM transactions")
    assert is_valid is True
    assert error == ""


def test_limit_is_enforced_even_when_absent(db_path):
    result = run_readonly_sql(db_path, "SELECT * FROM transactions", max_rows=2)
    assert result["success"] is True
    assert result["data"]["row_count"] == 2
    assert "LIMIT 2" in result["data"]["query_executed"]


def test_query_timeout_actually_fires(db_path):
    """Forces full evaluation with COUNT(*) so LIMIT can't let SQLite
    short-circuit the cross join early -- an earlier version of this test
    used a plain SELECT and never actually exercised the timeout at all,
    because LIMIT let the query finish almost instantly regardless."""
    query = (
        "SELECT COUNT(*) FROM transactions a, transactions b, transactions c, "
        "transactions d, transactions e, transactions f, transactions g, "
        "transactions h, transactions i, transactions j, transactions k, "
        "transactions l, transactions m"
    )
    result = run_readonly_sql(db_path, query, timeout_seconds=0.2)
    assert result["success"] is False
    assert "timed out" in result["error"].lower()