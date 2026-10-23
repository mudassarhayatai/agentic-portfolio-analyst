import pytest

from src.tools.registry import call_tool, get_tool_schemas


def test_calculate_unrealized_gain_end_to_end(db_path):
    result = call_tool("calculate_unrealized_gain", db_path, symbol="HBL")
    assert result["success"] is True
    data = result["data"]
    assert data["quantity"] == 300
    assert data["cost_basis"] == pytest.approx(39100.0)
    assert data["market_value"] == pytest.approx(148.0 * 300)
    assert data["unrealized_gain"] == pytest.approx(148.0 * 300 - 39100.0)


def test_calculate_unrealized_gain_for_unheld_symbol(db_path):
    result = call_tool("calculate_unrealized_gain", db_path, symbol="ENGRO")
    assert result["success"] is True
    assert result["data"]["held"] is False


def test_calculate_unrealized_gain_missing_price_is_a_distinct_error(db_path):
    """LUCK is held (50 shares remain after the partial sell) but has no
    market_price row -- this must fail with a specific, actionable error,
    not silently return a wrong number or a generic failure."""
    result = call_tool("calculate_unrealized_gain", db_path, symbol="LUCK")
    assert result["success"] is False
    assert "no current price" in result["error"].lower()


def test_calculate_realized_gain_reflects_fifo(db_path):
    result = call_tool("calculate_realized_gain", db_path, symbol="LUCK")
    assert result["success"] is True
    assert result["data"]["num_sell_transactions"] == 1
    # cost/share = net_value/quantity (50.5, 60.5), not the raw price column
    expected_cost_basis = (100 * 50.5) + (50 * 60.5)
    assert result["data"]["total_realized_gain"] == pytest.approx(10450.0 - expected_cost_basis)


def test_unknown_tool_returns_structured_error_not_exception(db_path):
    result = call_tool("not_a_real_tool", db_path)
    assert result["success"] is False


def test_every_registered_tool_has_a_valid_schema():
    for schema in get_tool_schemas():
        assert "name" in schema and schema["name"]
        assert "description" in schema and schema["description"]
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"