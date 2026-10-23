"""
Section 10: FastAPI serving layer.

Wraps whichever agent implementation (Section 5's hand-rolled ReAct loop,
or Section 7's LangGraph variant -- pick via AGENT_VARIANT) behind an
HTTP API. The agent runner (LLM client + tool wiring) is built ONCE at
startup via Section 5/7/8's shared factory.py, not rebuilt per request --


Usage (once Ollama is running and DB_PATH points at data/demo.db or your
real portfolio.db):
    uvicorn src.api.main:app --port 8001

    curl -X POST http://localhost:8001/query -H "Content-Type: application/json" \
        -d '{"question": "What is my unrealized gain on HBL?"}'


"""

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # project root -> makes `src` importable

from src.agent.factory import build_agent_runner
from src.agent.guardrails import GuardrailConfig

# ---------------------------------------------------------------------------
# Config -- environment-variable driven, sensible local-dev defaults
# ---------------------------------------------------------------------------

CONFIG_PATH = Path("config.yml")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f) or {}

DB_PATH = config.get("db_path", "data/demo.db")

AGENT_VARIANT = config.get("agent_variant", "langgraph")

OLLAMA_MODEL = config.get("ollama_model", "qwen3.5:0.8b")
OLLAMA_HOST = config.get("ollama_host", "http://localhost:11434")

USE_REFLECTION = bool(config.get("use_reflection", True))

MAX_ITERATIONS = int(config.get("max_iterations", 8))
MAX_TOOL_CALLS = int(config.get("max_tool_calls", 10))

TOOL_TIMEOUT_SECONDS = float(
    config.get("tool_timeout_seconds", 90)
)

RUN_TIMEOUT_SECONDS = float(
    config.get("run_timeout_seconds", 180)
)


# ---------------------------------------------------------------------------
# App state -- built once in lifespan(), reused across every request
# ---------------------------------------------------------------------------

class AppState:
    agent_runner = None
    ready = False
    startup_error: Optional[str] = None


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = GuardrailConfig(
        max_iterations=MAX_ITERATIONS,
        max_tool_calls_per_run=MAX_TOOL_CALLS,
        tool_timeout_seconds=TOOL_TIMEOUT_SECONDS,
        run_timeout_seconds=RUN_TIMEOUT_SECONDS,
    )
    try:
        state.agent_runner = build_agent_runner(
            variant=AGENT_VARIANT, db_path=DB_PATH,
            ollama_model=OLLAMA_MODEL, ollama_host=OLLAMA_HOST,
            config=config, use_reflection=USE_REFLECTION,
        )
        state.ready = True
        print(f"[startup] Ready. variant={AGENT_VARIANT} reflection={USE_REFLECTION} db={DB_PATH}")
    except Exception as e:
        state.startup_error = str(e)
        state.ready = False
        print(f"[startup] FAILED to build agent runner: {e}")
    yield
   

app = FastAPI(title="Portfolio Agent API", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    include_trace: bool = True


class QueryResponse(BaseModel):
    question: str
    answer: str
    success: bool
    stopped_reason: Optional[str] = None
    iterations: int
    tool_calls_made: int
    agent_variant: str
    reflection_attempts: Optional[int] = None
    trace: Optional[list] = None
    latency_ms: int


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Check this before /query -- a silent startup failure (bad DB path,
    Ollama unreachable, missing langgraph if AGENT_VARIANT=langgraph)
    surfaces here with the actual error, instead of a confusing 503 later."""
    return {
        "status": "ok" if state.ready else "not_ready",
        "agent_variant": AGENT_VARIANT,
        "db_path": DB_PATH,
        "reflection_enabled": USE_REFLECTION,
        "startup_error": state.startup_error,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if not state.ready or state.agent_runner is None:
        raise HTTPException(status_code=503, detail=f"Service not ready -- check /health. {state.startup_error or ''}")

    start = time.monotonic()
    try:
        result = state.agent_runner(request.question)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Agent run failed: {e}")

    latency_ms = int((time.monotonic() - start) * 1000)
    reflection = result.get("reflection")

    return QueryResponse(
        question=result.get("question", request.question),
        answer=result.get("answer", ""),
        success=result.get("success", False),
        stopped_reason=result.get("stopped_reason"),
        iterations=result.get("iterations", 0),
        tool_calls_made=result.get("tool_calls_made", 0),
        agent_variant=AGENT_VARIANT,
        reflection_attempts=reflection["attempts"] if reflection else None,
        trace=result.get("trace") if request.include_trace else None,
        latency_ms=latency_ms,
    )