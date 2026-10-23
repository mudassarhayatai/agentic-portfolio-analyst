"""
Section 3: Curated agent tools.

Thin, typed wrappers around Section 1 (db.py) and Section 2
(cost_basis.py). Two things distinguish these from calling db.py
directly:

1. Every tool returns a plain dict, always {"success": True, "data": ...}
   or {"success": False, "error": "..."} -- the agent loop (Section 5)
   needs one consistent shape to reason about "did this work," not a mix
   of return values, None, and exceptions to handle differently.
2. Every tool catches its own errors and returns them as data, never lets
   an exception escape into the agent loop -- a bad symbol name should
   produce an observation the agent can act on (try something else, tell
   the user), not a crash.
"""

from typing import Optional

from src.data.cost_basis import compute_symbol_cost_basis
from src.data.db import (
    get_current_holdings,
    get_dividend_summary,
    get_latest_price,
    get_transactions,
)


def _ok(data: dict) -> dict:
    return {"success": True, "data": data}


def _err(message: str) -> dict:
    return {"success": False, "error": message}


def tool_list_portfolio_symbols(db_path: str) -> dict:
    try:
        holdings = get_current_holdings(db_path)
        symbols = [h["symbol"] for h in holdings]
        return _ok({"symbols": symbols, "count": len(symbols)})
    except Exception as e:
        return _err(f"Could not list portfolio symbols: {e}")


def tool_get_holdings(db_path: str, symbol: Optional[str] = None) -> dict:
    try:
        holdings = get_current_holdings(db_path, symbol=symbol)
        if symbol and not holdings:
            return _ok({
                "symbol": symbol, "held": False,
                "message": f"No current holding found for {symbol} (never bought, or fully sold).",
            })
        return _ok({"holdings": holdings})
    except Exception as e:
        return _err(f"Could not retrieve holdings: {e}")


def tool_get_current_price(db_path: str, symbol: str) -> dict:
    try:
        price = get_latest_price(db_path, symbol)
        if price is None:
            return _err(f"No stored price found for {symbol} -- it may not have been fetched yet.")
        return _ok(price)
    except Exception as e:
        return _err(f"Could not retrieve price for {symbol}: {e}")


def tool_get_dividend_income(
    db_path: str, symbol: Optional[str] = None,
    start_date: Optional[str] = None, end_date: Optional[str] = None,
) -> dict:
    try:
        return _ok(get_dividend_summary(db_path, symbol=symbol, start_date=start_date, end_date=end_date))
    except Exception as e:
        return _err(f"Could not retrieve dividend income: {e}")


def tool_get_transaction_history(
    db_path: str, symbol: Optional[str] = None,
    start_date: Optional[str] = None, end_date: Optional[str] = None,
) -> dict:
    try:
        txns = get_transactions(db_path, symbol=symbol, start_date=start_date, end_date=end_date)
        return _ok({"transactions": txns, "count": len(txns)})
    except Exception as e:
        return _err(f"Could not retrieve transaction history: {e}")


def tool_calculate_unrealized_gain(db_path: str, symbol: str) -> dict:
    """Needs BOTH a current holding and a stored price to succeed --
    returns a distinguishable error for each missing piece, since the
    agent (or a human reading the trace) needs to know which one to fix."""
    try:
        cost_basis_result = compute_symbol_cost_basis(db_path, symbol)
        position = cost_basis_result["open_position"]
        if position["total_quantity"] <= 0:
            return _ok({"symbol": symbol, "held": False, "message": f"No current holding in {symbol}."})

        price_row = get_latest_price(db_path, symbol)
        if price_row is None:
            return _err(f"Holding exists for {symbol}, but no current price is stored.")

        quantity = position["total_quantity"]
        cost_basis = position["total_cost_basis"]
        market_value = price_row["price"] * quantity
        unrealized_gain = market_value - cost_basis

        result = {
            "symbol": symbol,
            "quantity": quantity,
            "average_cost_per_share": position["average_cost_per_share"],
            "current_price": price_row["price"],
            "price_as_of": price_row["fetched_date"],
            "cost_basis": cost_basis,
            "market_value": market_value,
            "unrealized_gain": unrealized_gain,
            "unrealized_gain_pct": (unrealized_gain / cost_basis * 100) if cost_basis else None,
        }
        if cost_basis_result["warnings"]:
            result["data_warnings"] = cost_basis_result["warnings"]
        return _ok(result)
    except Exception as e:
        return _err(f"Could not calculate unrealized gain for {symbol}: {e}")


def tool_calculate_realized_gain(
    db_path: str, symbol: str,
    start_date: Optional[str] = None, end_date: Optional[str] = None,
) -> dict:
    try:
        cost_basis_result = compute_symbol_cost_basis(db_path, symbol)
        trades = cost_basis_result["realized_trades"]
        if start_date:
            trades = [t for t in trades if t["sell_trade_date"] >= start_date]
        if end_date:
            trades = [t for t in trades if t["sell_trade_date"] <= end_date]

        result = {
            "symbol": symbol, "start_date": start_date, "end_date": end_date,
            "num_sell_transactions": len(trades),
            "total_realized_gain": sum(t["realized_gain"] for t in trades),
            "trades": trades,
        }
        if cost_basis_result["warnings"]:
            result["data_warnings"] = cost_basis_result["warnings"]
        return _ok(result)
    except Exception as e:
        return _err(f"Could not calculate realized gain for {symbol}: {e}")