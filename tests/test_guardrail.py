import time

import pytest

from src.agent.guardrails import (
    GuardrailConfig,
    GuardrailViolation,
    RunTracker,
    run_with_timeout,
    validate_tool_call,
)
from src.tools.registry import TOOL_REGISTRY


def test_iteration_budget_trips_after_max():
    tracker = RunTracker(config=GuardrailConfig(max_iterations=2))
    tracker.iterations = 2
    with pytest.raises(GuardrailViolation):
        tracker.check_iteration_budget()


def test_tool_call_budget_trips_after_max():
    tracker = RunTracker(config=GuardrailConfig(max_tool_calls_per_run=3))
    tracker.tool_calls_made = 3
    with pytest.raises(GuardrailViolation):
        tracker.check_tool_call_budget()


def test_run_timeout_trips_after_budget_exceeded():
    tracker = RunTracker(config=GuardrailConfig(run_timeout_seconds=0.05))
    time.sleep(0.06)
    with pytest.raises(GuardrailViolation):
        tracker.check_run_timeout()


def test_validate_tool_call_rejects_unknown_tool():
    is_valid, error = validate_tool_call("not_a_tool", {}, TOOL_REGISTRY)
    assert is_valid is False
    assert "unknown tool" in error.lower()


def test_validate_tool_call_rejects_missing_required_param():
    is_valid, error = validate_tool_call("calculate_unrealized_gain", {}, TOOL_REGISTRY)
    assert is_valid is False
    assert "missing required" in error.lower()


def test_validate_tool_call_rejects_unexpected_param():
    is_valid, error = validate_tool_call("get_current_price", {"symbol": "HBL", "made_up_param": 1}, TOOL_REGISTRY)
    assert is_valid is False
    assert "unexpected parameter" in error.lower()


def test_validate_tool_call_accepts_correct_call():
    is_valid, error = validate_tool_call("get_current_price", {"symbol": "HBL"}, TOOL_REGISTRY)
    assert is_valid is True
    assert error == ""


def test_run_with_timeout_returns_normally_when_fast_enough():
    result = run_with_timeout(lambda: {"success": True, "data": "ok"}, timeout_seconds=1.0)
    assert result["success"] is True


def test_run_with_timeout_fires_on_slow_callable():
    result = run_with_timeout(lambda: (time.sleep(1.0), {"success": True})[1], timeout_seconds=0.1)
    assert result["success"] is False
    assert "timed out" in result["error"].lower()