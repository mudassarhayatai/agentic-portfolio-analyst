"""
Section 4: Guarded SQL fallback tool.

For questions the curated tools (Section 3) don't cover. The read-only
connection (Section 1's get_connection) is NOT sufficient on its own as a
safety boundary: SQLite's ATTACH DATABASE can bring in a second, writable
database file even over a read-only main connection. So this validates
the query text itself, in layers, before it ever reaches SQLite:

  1. Exactly one statement (reject stacked queries separated by ';')
  2. Must start with SELECT
  3. No DDL/DML/ATTACH/PRAGMA keywords, anywhere in the query
  4. No CTEs (WITH ...) -- see the note below on why
  5. Every table referenced must be on an explicit allowlist (blocks
     probing sqlite_master or anything not in this schema)
  6. A LIMIT is enforced, capped, regardless of what the query asked for
  7. A real wall-clock timeout via SQLite's progress handler, since
     SQLite has no native per-statement timeout

"""

import re
import sqlite3
import time
from typing import Optional

from src.data.db import get_connection

ALLOWED_TABLES = {"transactions", "dividends", "market_price", "source_documents", "trade_confirmations"}

FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "drop", "alter", "create", "replace",
    "attach", "detach", "pragma", "vacuum", "reindex", "trigger", "with",
}

DEFAULT_MAX_ROWS = 100
MAX_ROWS_CAP = 500
DEFAULT_TIMEOUT_SECONDS = 5.0


def validate_readonly_sql(query: str) -> tuple[bool, str]:
    """Pure validation, no DB access -- returns (is_valid, error_message).
    Deliberately conservative: reject anything ambiguous rather than try
    to be clever about what's "probably fine."""
    stripped = query.strip().rstrip(";").strip()

    if not stripped:
        return False, "Empty query."

    if ";" in stripped:
        return False, "Multiple statements are not allowed -- submit exactly one SELECT."

    first_word = re.match(r"^\s*(\w+)", stripped, re.IGNORECASE)
    if not first_word or first_word.group(1).lower() != "select":
        return False, "Only SELECT queries are allowed."

    tokens = set(re.findall(r"[a-zA-Z_]+", stripped.lower()))
    forbidden_found = FORBIDDEN_KEYWORDS.intersection(tokens)
    if forbidden_found:
        return False, f"Query contains forbidden keyword(s): {', '.join(sorted(forbidden_found))}."

    referenced_tables = set(re.findall(r"(?:from|join)\s+([a-zA-Z_][a-zA-Z0-9_]*)", stripped, re.IGNORECASE))
    if not referenced_tables:
        return False, "Could not identify a table in the query."
    unknown_tables = referenced_tables - ALLOWED_TABLES
    if unknown_tables:
        return False, (
            f"Query references unrecognized table(s): {', '.join(sorted(unknown_tables))}. "
            f"Allowed tables: {', '.join(sorted(ALLOWED_TABLES))}."
        )

    return True, ""


def enforce_row_limit(query: str, max_rows: int) -> str:
    """Ensures a capped LIMIT is present, regardless of what was asked
    for -- a query with no LIMIT is a real risk once these tables grow
    past the small size they are today."""
    existing = re.search(r"\blimit\s+(\d+)", query, re.IGNORECASE)
    if existing:
        if int(existing.group(1)) > max_rows:
            return re.sub(r"\blimit\s+\d+", f"LIMIT {max_rows}", query, flags=re.IGNORECASE)
        return query
    return f"{query.rstrip().rstrip(';')} LIMIT {max_rows}"


def run_readonly_sql(
    db_path: str,
    query: str,
    max_rows: int = DEFAULT_MAX_ROWS,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict:
    is_valid, error = validate_readonly_sql(query)
    if not is_valid:
        return {"success": False, "error": error}

    max_rows = min(max_rows, MAX_ROWS_CAP)
    safe_query = enforce_row_limit(query, max_rows)

    start_time = time.monotonic()

    def _progress_handler() -> int:
        # Called every N SQLite VM instructions (see set_progress_handler
        # below). Returning nonzero aborts the currently executing query
        # and raises sqlite3.OperationalError -- this IS how you impose a
        # wall-clock timeout via Python's sqlite3 module; there's no
        # separate "query timeout" parameter to just set.
        return 1 if (time.monotonic() - start_time) > timeout_seconds else 0

    try:
        with get_connection(db_path) as conn:
            conn.set_progress_handler(_progress_handler, 1000)
            cursor = conn.execute(safe_query)
            rows = cursor.fetchmany(max_rows + 1)  # +1 to detect truncation
    except sqlite3.OperationalError as e:
        elapsed = time.monotonic() - start_time
        if elapsed >= timeout_seconds:
            return {"success": False, "error": f"Query timed out after {timeout_seconds}s."}
        return {"success": False, "error": f"SQL error: {e}"}
    except Exception as e:
        return {"success": False, "error": f"Unexpected error running query: {e}"}

    truncated = len(rows) > max_rows
    result_rows = [dict(row) for row in rows[:max_rows]]

    return {
        "success": True,
        "data": {
            "query_executed": safe_query,
            "row_count": len(result_rows),
            "truncated": truncated,
            "rows": result_rows,
        },
    }