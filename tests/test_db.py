import sqlite3

import pytest

from src.data.db import (
    get_connection,
    get_current_holdings,
    get_dividend_summary,
    get_latest_price,
    get_transactions,
)


def test_current_holdings_excludes_fully_exited_symbol(db_path):
    holdings = {h["symbol"]: h for h in get_current_holdings(db_path)}
    assert "HBL" in holdings
    assert "ENGRO" not in holdings  # bought 50, sold 50 -- net zero


def test_current_holdings_for_never_traded_symbol_is_empty(db_path):
    assert get_current_holdings(db_path, symbol="NOPE") == []


def test_transactions_returned_in_chronological_order(db_path):
    txns = get_transactions(db_path, symbol="LUCK")
    dates = [t["trade_date"] for t in txns]
    assert dates == sorted(dates)


def test_dividend_summary_totals_correctly(db_path):
    summary = get_dividend_summary(db_path, symbol="HBL")
    assert summary["total_net_dividend"] == 1600.0
    assert summary["num_payments"] == 2


def test_latest_price_picks_the_most_recent_row(db_path):
    price = get_latest_price(db_path, "HBL")
    assert price["price"] == 148.0
    assert price["fetched_date"] == "2025-01-15"


def test_latest_price_is_none_when_never_fetched(db_path):
    assert get_latest_price(db_path, "LUCK") is None


def test_connection_rejects_writes(db_path):
    """The actual safety mechanism everything else in this project
    depends on -- not just asserted, verified."""
    with pytest.raises(sqlite3.OperationalError):
        with get_connection(db_path) as conn:
            conn.execute(
                "INSERT INTO transactions (source_document_id, trade_date, symbol, action, quantity, price) "
                "VALUES (1, '2025-01-01', 'HACK', 'BUY', 1, 1.0)"
            )
            conn.commit()