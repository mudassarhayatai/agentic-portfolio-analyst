"""
Section 2: FIFO cost-basis engine.

A net-quantity SUM (Section 1's get_current_holdings) tells you how many
shares you hold, not what they cost -- once there's been ANY partial sell,
"average price paid" isn't a single number without a matching method. This
module implements FIFO: each SELL consumes the earliest still-open BUY
lot(s) first, which is both the most common convention and the one most
tax authorities default to expect for capital gains reporting.

ASSUMPTION (stated in db.py too, repeated here because it's load-bearing
for every number this module produces): `net_value` on a BUY is the total
cash paid out including commission/taxes/other_charges (raises cost
basis); on a SELL it's the net proceeds after those same deductions
(lowers realized gain). This is standard brokerage-statement convention,
but it's still an assumption about your specific data, not something
verified against your actual rows.

"""

from collections import deque
from typing import Optional


def compute_fifo(transactions: list[dict]) -> dict:
    """
    transactions: chronologically ordered BUY/SELL rows for ONE symbol
    (exactly what db.get_transactions(db_path, symbol=...) returns --
    order matters here, this function trusts it rather than re-sorting).

    Returns:
      open_lots         -- remaining unsold lots, oldest first (their
                            total cost basis is what Section 3's
                            unrealized-gain calculation needs)
      realized_trades    -- one entry per SELL: which lots it consumed,
                            proceeds, cost basis, realized gain
      total_realized_gain
      warnings           -- non-fatal data issues (see below), returned
                            rather than raised, so a caller can decide
                            whether to surface them to the user, block on
                            them, or just log them
    """
    open_lots: deque = deque()
    realized_trades: list[dict] = []
    warnings: list[str] = []
    total_realized_gain = 0.0

    for txn in transactions:
        if txn["action"] == "BUY":
            quantity = txn["quantity"]
            cost_per_share = (txn["net_value"] / quantity) if quantity else 0.0
            open_lots.append({
                "trade_date": txn["trade_date"],
                "quantity": quantity,
                "cost_per_share": cost_per_share,
                "transaction_id": txn.get("id"),
            })

        elif txn["action"] == "SELL":
            sell_quantity = txn["quantity"]
            remaining_to_sell = sell_quantity
            proceeds_per_share = (txn["net_value"] / sell_quantity) if sell_quantity else 0.0
            cost_basis_consumed = 0.0
            lots_consumed: list[dict] = []

            while remaining_to_sell > 0:
                if not open_lots:
                    
                    warnings.append(
                        f"SELL on {txn['trade_date']} (txn {txn.get('id')}) short by "
                        f"{remaining_to_sell} shares of matching BUY lots -- possible missing "
                        f"BUY transaction or unrecorded bonus/right shares."
                    )
                    break

                lot = open_lots[0]
                consumed_qty = min(lot["quantity"], remaining_to_sell)
                cost_basis_consumed += consumed_qty * lot["cost_per_share"]
                lots_consumed.append({
                    "trade_date": lot["trade_date"],
                    "quantity": consumed_qty,
                    "cost_per_share": lot["cost_per_share"],
                })
                lot["quantity"] -= consumed_qty
                remaining_to_sell -= consumed_qty
                if lot["quantity"] <= 0:
                    open_lots.popleft()

            matched_quantity = sell_quantity - remaining_to_sell
            realized_proceeds = proceeds_per_share * matched_quantity
            realized_gain = realized_proceeds - cost_basis_consumed
            total_realized_gain += realized_gain

            realized_trades.append({
                "sell_trade_date": txn["trade_date"],
                "sell_transaction_id": txn.get("id"),
                "quantity_sold": sell_quantity,
                "quantity_matched": matched_quantity,
                "proceeds": realized_proceeds,
                "cost_basis": cost_basis_consumed,
                "realized_gain": realized_gain,
            })

    return {
        "open_lots": list(open_lots),
        "realized_trades": realized_trades,
        "total_realized_gain": total_realized_gain,
        "warnings": warnings,
    }


def summarize_open_position(fifo_result: dict) -> dict:
    """Aggregate the remaining open lots into a single quantity + average
    cost -- what Section 3's unrealized-gain tool needs, without exposing
    every individual lot to that caller."""
    open_lots = fifo_result["open_lots"]
    total_quantity = sum(lot["quantity"] for lot in open_lots)
    total_cost = sum(lot["quantity"] * lot["cost_per_share"] for lot in open_lots)
    average_cost_per_share = (total_cost / total_quantity) if total_quantity else 0.0
    return {
        "total_quantity": total_quantity,
        "total_cost_basis": total_cost,
        "average_cost_per_share": average_cost_per_share,
    }


def compute_symbol_cost_basis(db_path: str, symbol: str) -> dict:
    """Top-level entry point: pulls this symbol's transactions from
    Section 1's db layer and runs the full FIFO calculation. This is the
    function Section 3's tools actually call."""
    from src.data.db import get_transactions

    transactions = get_transactions(db_path, symbol=symbol)
    fifo_result = compute_fifo(transactions)
    return {
        "symbol": symbol,
        **fifo_result,
        "open_position": summarize_open_position(fifo_result),
    }