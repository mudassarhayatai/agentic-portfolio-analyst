"""
Section 6: Self-correction / reflection.

Two deterministic checks run after every agent answer, cheap and fully
testable (no LLM call needed for the check itself):

1. Numeric grounding -- every number stated in the final answer should
   trace back to a number that actually appeared in a tool observation.
   

2. Suspicious no-tool-use -- the question looks like it needs portfolio
   data (mentions holdings/prices/gains/etc.) but the trace shows zero
   tool calls. Heuristic, not proof -- a genuinely tool-free question
   ("what does unrealized gain mean?") would also trip this without
   actually being a problem, which is why it contributes to a retry
   trigger rather than being treated as a hard failure on its own.

When either check fails, run_agent_with_reflection re-runs Section 5's
agent (fresh trace) with a critique appended to the question, up to
max_reflection_attempts times -- not by modifying run_agent itself, to
keep Section 5 untouched and this layered cleanly on top of it.
"""

import re
from typing import Iterable, Optional

from src.agent.guardrails import GuardrailConfig
from src.agent.react_agent import run_agent

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")

PORTFOLIO_KEYWORDS = {
    "holding", "holdings", "shares", "dividend", "dividends", "gain", "loss",
    "portfolio", "price", "bought", "sold", "worth", "value", "invested",
}


def extract_numbers(text: str) -> list[float]:
    """Pulls numeric values out of free text. Dates (YYYY-MM-DD) are
    masked out first -- without this, a date like '2025-01-15' gets
    mis-parsed by the number regex as three separate numbers (2025, -1,
    -15), which is exactly the kind of noise that would make this check
    unusable in a domain where every tool observation carries a date."""
    text_without_dates = _DATE_RE.sub(" ", text)
    numbers = []
    for match in _NUMBER_RE.finditer(text_without_dates):
        cleaned = match.group().replace(",", "")
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue
    return numbers


def _walk_numbers(obj) -> Iterable[float]:
    if isinstance(obj, bool):
        return  # bool is a subclass of int in Python -- exclude True/False explicitly
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_numbers(v)
    elif isinstance(obj, str):
        yield from extract_numbers(obj)


def extract_numbers_from_trace(trace: list[dict]) -> set:
    """Every number that appeared anywhere in a tool observation across
    the run -- this is the "grounded" set the final answer gets checked
    against. Rounded to 2 decimal places so float noise doesn't cause
    spurious mismatches."""
    grounded = set()
    for step in trace:
        if step.get("type") == "tool_call":
            for n in _walk_numbers(step.get("observation")):
                grounded.add(round(n, 2))
    return grounded


def check_numeric_grounding(answer: str, trace: list[dict], tolerance: float = 0.01) -> dict:
    answer_numbers = extract_numbers(answer)
    grounded_numbers = extract_numbers_from_trace(trace)

    ungrounded = [n for n in answer_numbers if not any(abs(n - g) <= tolerance for g in grounded_numbers)]

    return {
        "answer_numbers": answer_numbers,
        "ungrounded_numbers": ungrounded,
        "is_grounded": len(ungrounded) == 0,
    }


def check_tool_usage(question: str, trace: list[dict]) -> dict:
    tool_calls_made = sum(1 for step in trace if step.get("type") == "tool_call")
    looks_like_data_question = any(kw in question.lower() for kw in PORTFOLIO_KEYWORDS)
    return {
        "tool_calls_made": tool_calls_made,
        "looks_like_data_question": looks_like_data_question,
        "suspicious_no_tool_use": looks_like_data_question and tool_calls_made == 0,
    }


def run_agent_with_reflection(
    question: str,
    db_path: str,
    llm_client,
    config: Optional[GuardrailConfig] = None,
    max_reflection_attempts: int = 1,
) -> dict:
    config = config or GuardrailConfig()
    result = run_agent(question, db_path, llm_client, config)

    reflection_log: list[dict] = []
    attempt = 0

    while attempt < max_reflection_attempts and result["success"]:
        grounding = check_numeric_grounding(result["answer"], result["trace"])
        usage = check_tool_usage(question, result["trace"])
        reflection_log.append({"attempt": attempt, "grounding": grounding, "tool_usage": usage})

        needs_retry = (not grounding["is_grounded"]) or usage["suspicious_no_tool_use"]
        if not needs_retry:
            break

        attempt += 1
        critique_parts = []
        if not grounding["is_grounded"]:
            critique_parts.append(
                f"Your previous answer stated number(s) {grounding['ungrounded_numbers']} that don't directly "
                f"match any tool result. If these are computed from multiple tool results, that's fine -- just "
                f"make sure every INPUT to the calculation came from a tool, not assumed knowledge."
            )
        if usage["suspicious_no_tool_use"]:
            critique_parts.append(
                "You answered a portfolio-data question without calling any tool -- use the available tools "
                "to get real data before answering."
            )

        revised_question = f"{question}\n\n[SELF-CORRECTION NOTE from a previous attempt: {' '.join(critique_parts)}]"
        result = run_agent(revised_question, db_path, llm_client, config)

    result["reflection"] = {"attempts": attempt, "log": reflection_log}
    return result