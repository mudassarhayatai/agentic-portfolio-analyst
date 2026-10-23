"""
Tests for Section 5's hand-rolled ReAct loop. Uses a scripted
FakeLLMClient instead of a real model -- but the tool layer it calls
through is real (Sections 3/4 + this fixture's real data), so these
exercise the actual agent loop AND actual tool execution together; only
the LLM call itself is faked.
"""

from src.agent.guardrails import GuardrailConfig
from src.agent.react_agent import run_agent


class FakeLLMClient:
    """Returns a pre-scripted sequence of responses, one per .chat() call.
    Raises if called more times than scripted, so a runaway loop in the
    code under test fails loudly instead of silently looping forever."""

    def __init__(self, responses: list[dict]):
        self.responses = responses
        self.call_count = 0

    def chat(self, messages, tools):
        if self.call_count >= len(self.responses):
            raise AssertionError(f"FakeLLMClient called more times than scripted ({len(self.responses)})")
        response = self.responses[self.call_count]
        self.call_count += 1
        return response


def test_immediate_final_answer_no_tool_calls(db_path):
    client = FakeLLMClient([{"content": "I don't need any tools for this.", "tool_calls": []}])
    result = run_agent("What's 2+2?", db_path, client)
    assert result["success"] is True
    assert result["tool_calls_made"] == 0
    assert result["iterations"] == 1


def test_single_tool_call_then_final_answer(db_path):
    client = FakeLLMClient([
        {"content": "", "tool_calls": [{"name": "get_current_price", "arguments": {"symbol": "HBL"}}]},
        {"content": "HBL's current price is 148.0 as of 2025-01-15.", "tool_calls": []},
    ])
    result = run_agent("What's HBL's current price?", db_path, client)
    assert result["success"] is True
    assert result["tool_calls_made"] == 1
    # the tool actually ran against the real fixture DB, not a mock
    tool_step = result["trace"][0]
    assert tool_step["observation"]["success"] is True
    assert tool_step["observation"]["data"]["price"] == 148.0


def test_multi_step_chain_holdings_then_price(db_path):
    """Mirrors the real multi-step case from this project's design:
    check holdings, then check price, then answer."""
    client = FakeLLMClient([
        {"content": "", "tool_calls": [{"name": "get_holdings", "arguments": {"symbol": "HBL"}}]},
        {"content": "", "tool_calls": [{"name": "get_current_price", "arguments": {"symbol": "HBL"}}]},
        {"content": "You hold 300 shares of HBL, currently priced at 148.0.", "tool_calls": []},
    ])
    result = run_agent("How much is my HBL position worth?", db_path, client)
    assert result["success"] is True
    assert result["tool_calls_made"] == 2
    assert len(result["trace"]) == 3  # 2 tool calls + 1 final answer


def test_max_iterations_guardrail_stops_a_runaway_agent(db_path):
    """A FakeLLMClient that NEVER gives a final answer -- the agent must
    stop itself via the guardrail, not loop until the script runs out."""
    client = FakeLLMClient([
        {"content": "", "tool_calls": [{"name": "get_current_price", "arguments": {"symbol": "HBL"}}]}
        for _ in range(10)  # more scripted responses than the configured max
    ])
    config = GuardrailConfig(max_iterations=3)
    result = run_agent("infinite question", db_path, client, config)
    assert result["success"] is False
    assert "maximum of 3 reasoning steps" in result["stopped_reason"]
    assert client.call_count == 3  # never called a 4th time


def test_unknown_tool_call_is_handled_not_crashed(db_path):
    client = FakeLLMClient([
        {"content": "", "tool_calls": [{"name": "delete_everything", "arguments": {}}]},
        {"content": "I don't have a tool for that, so I can't answer.", "tool_calls": []},
    ])
    result = run_agent("do something destructive", db_path, client)
    assert result["success"] is True  # the AGENT recovered gracefully
    assert result["trace"][0]["observation"]["success"] is False
    assert "unknown tool" in result["trace"][0]["observation"]["error"].lower()


def test_missing_required_parameter_is_caught_before_execution(db_path):
    client = FakeLLMClient([
        {"content": "", "tool_calls": [{"name": "calculate_unrealized_gain", "arguments": {}}]},  # missing "symbol"
        {"content": "I need a symbol to calculate that.", "tool_calls": []},
    ])
    result = run_agent("calculate my gain", db_path, client)
    assert result["success"] is True
    observation = result["trace"][0]["observation"]
    assert observation["success"] is False
    assert "missing required" in observation["error"].lower()