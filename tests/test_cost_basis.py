import pytest

from src.data.cost_basis import compute_fifo, compute_symbol_cost_basis
from src.data.db import get_transactions


def test_fifo_partial_sell_matches_correct_lots_and_cost(db_path):
    """The real FIFO test: LUCK has two BUY lots (100@50, 100@60) and one
    SELL of 150 -- this should consume all of lot 1 and half of lot 2."""
    txns = get_transactions(db_path, symbol="LUCK")
    result = compute_fifo(txns)

    trade = result["realized_trades"][0]
    # cost_per_share = net_value / quantity (includes fees), NOT the raw
    # price column -- lot 1's net_value is 5050 for 100 shares (50.5/share,
    # not 50.0), same reasoning as lot 2's 60.5.
    expected_cost_basis = (100 * 50.5) + (50 * 60.5)  # all of lot 1 + half of lot 2
    assert trade["quantity_matched"] == 150
    assert trade["cost_basis"] == pytest.approx(expected_cost_basis)
    assert trade["proceeds"] == pytest.approx(10450.0)
    assert trade["realized_gain"] == pytest.approx(10450.0 - expected_cost_basis)

    # remaining open lot: 50 shares left over from the SECOND buy lot
    assert len(result["open_lots"]) == 1
    assert result["open_lots"][0]["quantity"] == 50
    assert result["open_lots"][0]["cost_per_share"] == pytest.approx(60.5)  # 6050/100
    assert result["warnings"] == []


def test_fifo_oversell_warns_instead_of_crashing(db_path):
    txns = get_transactions(db_path, symbol="OVERSOLD")
    result = compute_fifo(txns)

    assert len(result["warnings"]) == 1
    assert "short by" in result["warnings"][0]
    assert result["realized_trades"][0]["quantity_matched"] == 0  # no BUY lots existed at all


def test_open_position_summary_for_simple_holding(db_path):
    result = compute_symbol_cost_basis(db_path, "HBL")
    position = result["open_position"]
    assert position["total_quantity"] == 300
    assert position["total_cost_basis"] == pytest.approx(12050.0 + 27050.0)
    assert position["average_cost_per_share"] == pytest.approx((12050.0 + 27050.0) / 300)


def test_fully_exited_symbol_has_no_open_position(db_path):
    result = compute_symbol_cost_basis(db_path, "ENGRO")
    assert result["open_position"]["total_quantity"] == 0