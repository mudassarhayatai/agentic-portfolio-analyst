"""
Section 1: Read-only database access layer.

Every tool this agent uses -- curated (Section 3) or the SQL fallback
(Section 4) -- reads through this module, never sqlite3 directly. Two
things matter here more than in a typical "just query the DB" script:

1. The connection is opened in SQLite's URI read-only mode (`?mode=ro`),
   enforced by SQLite itself at the driver level -- not just "the code
   happens not to call INSERT/UPDATE." An agent that eventually gets to
   run less-constrained SQL (Section 4) needs a real safety floor here,
   not a promise kept by convention.
2. Every function returns plain dicts/lists, not raw cursor rows -- this
   is the layer the agent's tool wrappers call directly, so the output
   needs to already be JSON-serializable for the agent's tool-call loop.

"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional


@contextmanager
def get_connection(db_path: str):
    """Opens the database in SQLite's URI read-only mode -- any write
    attempt is rejected at the driver level, before it would even reach
    the database file. This is the actual safety mechanism the
    SQL-fallback tool (Section 4) will depend on, not just code
    discipline in the functions below."""
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _rows_to_dicts(rows) -> list[dict]:
    return [dict(row) for row in rows]


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------

def get_transactions(
    db_path: str,
    symbol: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[dict]:
    """Raw BUY/SELL rows, optionally filtered. Ordered by trade_date, then
    id -- callers doing cost-basis matching (Section 2) depend on this
    ordering being chronological, not just display convenience."""
    query = "SELECT * FROM transactions WHERE 1=1"
    params: list = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if start_date:
        query += " AND trade_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND trade_date <= ?"
        params.append(end_date)
    query += " ORDER BY trade_date ASC, id ASC"

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return _rows_to_dicts(rows)


def list_held_symbols(db_path: str) -> list[str]:
    """Every symbol that's ever appeared in a transaction -- not
    necessarily still held (net quantity could be zero); just traded."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT DISTINCT symbol FROM transactions ORDER BY symbol").fetchall()
    return [r["symbol"] for r in rows]


def get_current_holdings(db_path: str, symbol: Optional[str] = None) -> list[dict]:
    """Net quantity currently held per symbol: SUM(BUY qty) - SUM(SELL
    qty). This is a net-position view, NOT a cost basis -- computing that
    correctly needs FIFO/weighted-average matching (Section 2), which a
    single SUM can't express (it can't tell you which specific lot's cost
    applies to remaining shares once partial sells are involved). A symbol
    with net_quantity <= 0 here means "fully exited," not an error --
    filtered out by the HAVING clause so this only returns what's
    currently actually held."""
    query = """
        SELECT
            symbol,
            SUM(CASE WHEN action = 'BUY' THEN quantity ELSE -quantity END) AS net_quantity,
            SUM(CASE WHEN action = 'BUY' THEN net_value ELSE 0 END) AS total_buy_value,
            SUM(CASE WHEN action = 'BUY' THEN quantity ELSE 0 END) AS total_buy_quantity
        FROM transactions
        WHERE 1=1
    """
    params: list = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    query += " GROUP BY symbol HAVING net_quantity != 0 ORDER BY symbol"

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return _rows_to_dicts(rows)


# ---------------------------------------------------------------------------
# Dividends
# ---------------------------------------------------------------------------

def get_dividends(
    db_path: str,
    symbol: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> list[dict]:
    query = "SELECT * FROM dividends WHERE 1=1"
    params: list = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)
    if start_date:
        query += " AND payment_date >= ?"
        params.append(start_date)
    if end_date:
        query += " AND payment_date <= ?"
        params.append(end_date)
    query += " ORDER BY payment_date ASC"

    with get_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()
    return _rows_to_dicts(rows)


def get_dividend_summary(
    db_path: str,
    symbol: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> dict:
    """Total net dividend income over a window, optionally per symbol.
    ASSUMPTION: net_dividend in your schema is the amount actually paid
    out (post-withholding-tax) -- confirm before using this for a real tax
    figure. If net_dividend is actually pre-tax, this needs a
    withholding-tax column that isn't in the DDL you gave me, and this
    summary would overstate net income."""
    rows = get_dividends(db_path, symbol, start_date, end_date)
    total = sum(r["net_dividend"] for r in rows)
    return {
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "total_net_dividend": total,
        "num_payments": len(rows),
    }


# ---------------------------------------------------------------------------
# Market prices
# ---------------------------------------------------------------------------
# ASSUMED SCHEMA (not verified against actual DDL -- see module docstring):
#   table: market_price(Symbol TEXT, price REAL, fetched_date TEXT)

def get_latest_price(db_path: str, symbol: str) -> Optional[dict]:
    """Most recent stored price for a symbol, by fetched_date. Returns
    None if the symbol has never been fetched -- callers must handle this
    explicitly (a newly bought or delisted symbol might have no price row
    yet) rather than assuming a price always exists."""
    query = """
        SELECT symbol, price, fetched_date
        FROM market_price
        WHERE Symbol = ?
        ORDER BY fetched_date DESC
        LIMIT 1
    """
    with get_connection(db_path) as conn:
        row = conn.execute(query, [symbol]).fetchone()
    return dict(row) if row else None