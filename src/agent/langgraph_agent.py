"""
Section 7: LangGraph variant.

Same tools (Section 3/4), same guardrails (Section 4's guardrails.py),
same result shape as Section 5's hand-rolled run_agent() -- rebuilt as an
explicit state graph instead of a while-loop. The point of having both in
this repo isn't "LangGraph is better" or "hand-rolled is better" -- it's
demonstrating the mechanics were understood first (Section 5), then
shown fluent in the modern ecosystem's idiom for the same problem, on a
task that actually has real branching structure (tool-call vs. no-tool-
call vs. guardrail-stop), unlike Project 1's RAG pipeline, which was a
straight line and didn't justify a graph framework at all.

Both implementations reuse the EXACT SAME guardrails.py functions
(validate_tool_call, run_with_timeout, GuardrailConfig) -- not two
separate reimplementations of "stop after N steps" that could quietly
drift apart.


"""

import json
from typing import Annotated, Optional, TypedDict

from src.agent.guardrails import GuardrailConfig, run_with_timeout, validate_tool_call
from src.agent.react_agent import SYSTEM_PROMPT
from src.tools.registry import TOOL_REGISTRY, call_tool, get_tool_schemas


class AgentState(TypedDict):
    messages: Annotated[list, "add_messages reducer, see build_agent_graph"]
    iterations: int
    tool_calls_made: int
    db_path: str


def _messages_to_trace(messages: list) -> list[dict]:
    """Converts a LangGraph run's final message list into the SAME trace
    shape Section 5's run_agent() produces -- {"type": "tool_call", ...}
    / {"type": "final_answer", ...} -- so Section 8's eval harness can
    treat both agent implementations interchangeably without knowing
    which one produced a given result.

    Walks AI messages with tool_calls, matches each against the following
    ToolMessage(s) by tool_call_id, and treats a tool-call-free AI message
    as the final answer."""
    trace: list[dict] = []
    tool_messages_by_id = {m.tool_call_id: m for m in messages if getattr(m, "type", None) == "tool"}

    for m in messages:
        if getattr(m, "type", None) != "ai":
            continue
        tool_calls = getattr(m, "tool_calls", None) or []
        if not tool_calls:
            trace.append({"type": "final_answer", "content": m.content})
            continue
        for call in tool_calls:
            tool_message = tool_messages_by_id.get(call["id"])
            observation = json.loads(tool_message.content) if tool_message else {"success": False, "error": "No matching tool result found."}
            trace.append({"type": "tool_call", "tool": call["name"], "args": call.get("args", {}), "observation": observation})

    return trace


def build_agent_graph(llm, config: Optional[GuardrailConfig] = None):
    """Builds and compiles the graph. `llm` must support LangChain's
    .bind_tools()/.invoke() interface (e.g. langchain_ollama.ChatOllama).
    Import of langgraph/langchain_core is done here, inside the function,
    so this module stays importable without those packages -- only
    calling build_agent_graph() actually requires them."""
    from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
    from langgraph.graph import END, StateGraph
    from langgraph.graph.message import add_messages

    config = config or GuardrailConfig()

    class _State(TypedDict):
        messages: Annotated[list, add_messages]
        iterations: int
        tool_calls_made: int
        db_path: str

    bound_llm = llm.bind_tools([{"type": "function", "function": schema} for schema in get_tool_schemas()])

    def agent_node(state: _State) -> dict:
        response = bound_llm.invoke(state["messages"])
        return {"messages": [response], "iterations": state["iterations"] + 1}

    def tool_node(state: _State) -> dict:
        last_message = state["messages"][-1]
        tool_messages = []
        tool_calls_made = state["tool_calls_made"]

        for call in last_message.tool_calls:
            tool_name, tool_args, call_id = call["name"], call.get("args", {}), call["id"]

            is_valid, error = validate_tool_call(tool_name, tool_args, TOOL_REGISTRY)
            if not is_valid:
                observation = {"success": False, "error": error}
            else:
                tool_calls_made += 1
                observation = run_with_timeout(
                    lambda tn=tool_name, ta=tool_args: call_tool(tn, state["db_path"], **ta),
                    config.tool_timeout_seconds,
                )

            tool_messages.append(ToolMessage(content=json.dumps(observation), tool_call_id=call_id))

        return {"messages": tool_messages, "tool_calls_made": tool_calls_made}

    def guardrail_stop_node(state: _State) -> dict:
        reason = (
            f"Stopped after reaching a guardrail limit "
            f"({state['iterations']} iterations, {state['tool_calls_made']} tool calls)."
        )
        return {"messages": [AIMessage(content=f"I had to stop before finishing: {reason}")]}

    def route(state: _State) -> str:
        if state["iterations"] >= config.max_iterations or state["tool_calls_made"] >= config.max_tool_calls_per_run:
            return "guardrail_stop"
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "tools"
        return "end"

    graph = StateGraph(_State)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("guardrail_stop", guardrail_stop_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", route, {"tools": "tools", "end": END, "guardrail_stop": "guardrail_stop"})
    graph.add_edge("tools", "agent")
    graph.add_edge("guardrail_stop", END)

    return graph.compile()


def run_agent_langgraph(question: str, db_path: str, llm, config: Optional[GuardrailConfig] = None) -> dict:
    """Same signature/return shape as Section 5's run_agent(), so both
    are interchangeable to any caller (the eval harness, the API)."""
    from langchain_core.messages import HumanMessage, SystemMessage

    config = config or GuardrailConfig()
    graph = build_agent_graph(llm, config)

    initial_state = {
        "messages": [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=question)],
        "iterations": 0,
        "tool_calls_made": 0,
        "db_path": db_path,
    }
    final_state = graph.invoke(initial_state)

    trace = _messages_to_trace(final_state["messages"])
    last_ai_message = final_state["messages"][-1]
    guardrail_tripped = "I had to stop before finishing" in getattr(last_ai_message, "content", "")

    return {
        "question": question,
        "answer": last_ai_message.content,
        "success": not guardrail_tripped,
        "stopped_reason": last_ai_message.content if guardrail_tripped else None,
        "trace": trace,
        "iterations": final_state["iterations"],
        "tool_calls_made": final_state["tool_calls_made"],
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="data/demo.db")
    parser.add_argument("--question", required=True)
    parser.add_argument("--ollama-model", default="qwen3.5:0.8b")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--max-iterations", type=int, default=8)
    args = parser.parse_args()

    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Install with: pip install langgraph langchain-core langchain-ollama")
        raise SystemExit(1)

    # NOTE: base_url is my best understanding of ChatOllama's constructor
    # param for pointing at a non-default Ollama host -- unverified here,
    # same live-untested caveat as the rest of this file.
    llm = ChatOllama(model=args.ollama_model, base_url=args.ollama_host, temperature=0.1)
    config = GuardrailConfig(max_iterations=args.max_iterations)

    try:
        result = run_agent_langgraph(args.question, args.db, llm, config)
    except Exception as e:
        print(f"Could not run the LangGraph agent (is Ollama running at {args.ollama_host}?): {e}")
        raise SystemExit(0)

    print(f"=== Answer ({'success' if result['success'] else 'stopped: ' + str(result['stopped_reason'])}) ===")
    print(result["answer"])
    print(f"\n({result['iterations']} iterations, {result['tool_calls_made']} tool calls)")
    print("\n=== Trace ===")
    for step in result["trace"]:
        print(json.dumps(step, indent=2, default=str))