"""
Section 5: Hand-rolled ReAct agent loop.

The actual "agent" logic: decide -> act -> observe -> repeat, with every
guardrail from Section 4 enforced at the point it matters, not just
described in a prompt. 
"""

import json
import time
from typing import Optional

from src.agent.guardrails import (
    GuardrailConfig,
    GuardrailViolation,
    RunTracker,
    run_with_timeout,
    validate_tool_call,
)
from src.tools.registry import TOOL_REGISTRY, call_tool, get_tool_schemas

SYSTEM_PROMPT = """You are a portfolio analysis assistant. You answer questions about the user's PSX stock portfolio ONLY by calling the tools available to you -- you have no built-in knowledge of the user's actual holdings, transactions, or dividends, and must never guess or state a number you didn't get from a tool.

Rules:
1. Call tools to gather whatever data you need before answering. Chain multiple tool calls if the question requires it (e.g. check holdings before calculating a gain).
2. When you have enough information, respond with a final answer (no further tool call) citing the specific numbers the tools returned.
3. If a tool returns an error, or says something isn't currently held, say so plainly rather than guessing or retrying the same call over and over.
4. If you genuinely cannot answer with the available tools, say so explicitly and explain what's missing.
5. very important: Use only PKR currency when returing the answer. 
"""


def _stopped_result(question: str, trace: list[dict], reason: str, tracker: RunTracker) -> dict:
    return {
        "question": question,
        "answer": f"I had to stop before finishing: {reason}",
        "success": False,
        "stopped_reason": reason,
        "trace": trace,
        "iterations": tracker.iterations,
        "tool_calls_made": tracker.tool_calls_made,
    }


def run_agent(question: str, db_path: str, llm_client, config: Optional[GuardrailConfig] = None) -> dict:
    """Runs the full plan -> act -> observe loop for one question.
    Returns a structured result -- success/failure, the final answer, the
    full step trace (what Section 8's eval harness will use to build a
    failure taxonomy, not just a pass/fail count), and guardrail usage
    counts (what Section 8 will report as "how often did guardrails
    actually fire")."""
    config = config or GuardrailConfig()
    tracker = RunTracker(config=config)
    tool_schemas = get_tool_schemas()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    trace: list[dict] = []

    while True:
        try:
            tracker.check_iteration_budget()
            tracker.check_run_timeout()
        except GuardrailViolation as e:
            return _stopped_result(question, trace, str(e), tracker)

        tracker.iterations += 1

        response = llm_client.chat(messages, tools=tool_schemas)
        tool_calls = response.get("tool_calls") or []
        content = response.get("content") or ""

        if not tool_calls:
            # No tool call requested -- the model is giving its final answer.
            trace.append({"type": "final_answer", "content": content, "timestamp": time.time()})
            return {
                "question": question,
                "answer": content,
                "success": True,
                "stopped_reason": None,
                "trace": trace,
                "iterations": tracker.iterations,
                "tool_calls_made": tracker.tool_calls_made,
            }

        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

        for call in tool_calls:
            tool_name = call.get("name", "")
            tool_args = call.get("arguments") or {}

            try:
                tracker.check_tool_call_budget()
            except GuardrailViolation as e:
                return _stopped_result(question, trace, str(e), tracker)

            is_valid, validation_error = validate_tool_call(tool_name, tool_args, TOOL_REGISTRY)
            if not is_valid:
                observation = {"success": False, "error": validation_error}
            else:
                tracker.tool_calls_made += 1
                observation = run_with_timeout(
                    lambda: call_tool(tool_name, db_path, **tool_args),
                    config.tool_timeout_seconds,
                )

            trace.append({"type": "tool_call", "tool": tool_name, "args": tool_args, "observation": observation, "timestamp": time.time()})
            messages.append({"role": "tool", "tool_name": tool_name, "content": json.dumps(observation)})


class OllamaAgentClient:
    """Real LLM client for run_agent(), using Ollama's native tool-calling
    support via /api/chat. NOT verified against a live Ollama server in
    this sandbox -- the request/response shape below is my best
    understanding of Ollama's tool-calling wire format, but exact field
    names have genuinely shifted across Ollama versions before, so treat
    this as the piece most likely to need a small adjustment once you run
    it for real, not a guaranteed-correct implementation the way
    run_agent() itself is."""

    def __init__(self, model: str = "qwen3.5:0.8b", host: str = "http://localhost:11434"):
        self.model = model
        self.host = host.rstrip("/")

    def chat(self, messages: list[dict], tools: list[dict]) -> dict:
        import urllib.request

        ollama_tools = [{"type": "function", "function": schema} for schema in tools]
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": ollama_tools,
            "stream": False,
            "options": {"temperature": 0.1},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        message = data.get("message", {})
        raw_tool_calls = message.get("tool_calls") or []
        tool_calls = [
            {"name": tc["function"]["name"], "arguments": tc["function"].get("arguments", {})}
            for tc in raw_tool_calls
        ]
        return {"content": message.get("content", ""), "tool_calls": tool_calls}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/portfolio.db")
    parser.add_argument("--question", required=True)
    parser.add_argument("--ollama-model", default="qwen3.5:0.8b")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--max-iterations", type=int, default=8)
    args = parser.parse_args()

    client = OllamaAgentClient(model=args.ollama_model, host=args.ollama_host)
    config = GuardrailConfig(max_iterations=args.max_iterations)

    try:
        result = run_agent(args.question, args.db, client, config)
    except Exception as e:
        print(f"Could not reach Ollama at {args.ollama_host} ({e})")
        raise SystemExit(0)

    print(f"=== Answer ({'success' if result['success'] else 'stopped: ' + str(result['stopped_reason'])}) ===")
    print(result["answer"])
    print(f"\n({result['iterations']} iterations, {result['tool_calls_made']} tool calls)")
    print("\n=== Trace ===")
    for step in result["trace"]:
        print(json.dumps(step, indent=2, default=str))