"""
Shared agent-runner factory. Both eval/run_eval.py (Section 8) and
src/api/main.py (Section 10) need to build "a callable that takes a
question and returns an agent result" for either the react or langgraph
variant
"""

from typing import Callable, Optional

from src.agent.guardrails import GuardrailConfig


def build_agent_runner(
    variant: str,
    db_path: str,
    ollama_model: str = "qwen3.5:0.8b",
    ollama_host: str = "http://localhost:11434",
    config: Optional[GuardrailConfig] = None,
    use_reflection: bool = True,
) -> Callable[[str], dict]:
    config = config or GuardrailConfig()

    if variant == "react":
        from src.agent.react_agent import OllamaAgentClient, run_agent
        from src.agent.reflection import run_agent_with_reflection

        client = OllamaAgentClient(model=ollama_model, host=ollama_host)
        if use_reflection:
            return lambda q: run_agent_with_reflection(q, db_path, client, config)
        return lambda q: run_agent(q, db_path, client, config)

    if variant == "langgraph":
        from langchain_ollama import ChatOllama

        from src.agent.langgraph_agent import run_agent_langgraph

        llm = ChatOllama(model=ollama_model, base_url=ollama_host, temperature=0.1)
        return lambda q: run_agent_langgraph(q, db_path, llm, config)

    raise ValueError(f"Unknown agent variant: {variant!r}. Use 'react' or 'langgraph'.")