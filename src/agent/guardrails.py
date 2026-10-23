"""
Shared guardrails for both agent implementations. Defined once here so
Section 5's hand-rolled loop and Section 7's LangGraph variant provably
enforce the SAME rules
"""

import concurrent.futures
import time
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class GuardrailConfig:
    max_iterations: int = 8
    max_tool_calls_per_run: int = 10
    tool_timeout_seconds: float = 90.0
    run_timeout_seconds: float = 180.0


class GuardrailViolation(Exception):
    """Raised when a guardrail trips. The agent loop catches this and
    turns it into a clean, explained stop -- never a raw crash or an
    infinite loop."""


@dataclass
class RunTracker:
    """State for ONE agent run -- iteration count, tool calls made,
    elapsed time. Passed through the loop explicitly rather than kept as
    global/module state, so concurrent runs (e.g. the eval harness
    running many tasks) never interfere with each other."""
    config: GuardrailConfig
    start_time: float = field(default_factory=time.monotonic)
    iterations: int = 0
    tool_calls_made: int = 0

    def check_iteration_budget(self) -> None:
        if self.iterations >= self.config.max_iterations:
            raise GuardrailViolation(
                f"Stopped after reaching the maximum of {self.config.max_iterations} reasoning steps without a final answer."
            )

    def check_tool_call_budget(self) -> None:
        if self.tool_calls_made >= self.config.max_tool_calls_per_run:
            raise GuardrailViolation(
                f"Stopped after reaching the maximum of {self.config.max_tool_calls_per_run} tool calls."
            )

    def check_run_timeout(self) -> None:
        elapsed = time.monotonic() - self.start_time
        if elapsed >= self.config.run_timeout_seconds:
            raise GuardrailViolation(
                f"Stopped after exceeding the {self.config.run_timeout_seconds}s total run budget ({elapsed:.1f}s elapsed)."
            )


def validate_tool_call(tool_name: str, tool_args: dict, tool_registry: dict) -> tuple[bool, str]:
    """Schema-validate a proposed tool call BEFORE executing it, against
    the same schema the LLM was shown. This is what stops a malformed or
    hallucinated tool call from ever reaching Section 3/4's actual
    database code -- the model proposing a bad call is expected and
    normal; letting it execute isn't."""
    entry = tool_registry.get(tool_name)
    if entry is None:
        return False, f"Unknown tool '{tool_name}'. Available: {list(tool_registry.keys())}"

    schema = entry["schema"]["parameters"]
    required = schema.get("required", [])
    missing = [p for p in required if p not in tool_args]
    if missing:
        return False, f"Missing required parameter(s) for '{tool_name}': {missing}"

    allowed_params = set(schema.get("properties", {}).keys())
    unexpected = set(tool_args.keys()) - allowed_params
    if unexpected:
        return False, f"Unexpected parameter(s) for '{tool_name}': {sorted(unexpected)}"

    return True, ""


def run_with_timeout(callable_fn: Callable[[], dict], timeout_seconds: float) -> dict:
    """Runs a zero-argument callable with a hard wall-clock timeout via a
    background thread. NOTE: this does not forcibly kill a truly stuck
    thread (Python threads can't be killed externally) -- it stops
    WAITING for it and reports a timeout. Sufficient for well-behaved
    tools; not a substitute for Section 4's own internal query timeout on
    the SQL fallback specifically, which actually aborts SQLite execution."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(callable_fn)
        try:
            return future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            return {"success": False, "error": f"Tool call timed out after {timeout_seconds}s."}