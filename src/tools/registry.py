"""
Tool registry: maps a tool name to its function and JSON schema. This is
the single source of truth the agent loop (Section 5) uses both to tell
the model what's available and to dispatch a call -- no tool call reaches
Section 1/2's actual database code except through call_tool() below.
"""

from src.tools import portfolio_tools as t
from src.tools.sql_fallback import run_readonly_sql

TOOL_REGISTRY = {
    "list_portfolio_symbols": {
        "function": t.tool_list_portfolio_symbols,
        "schema": {
            "name": "list_portfolio_symbols",
            "description": "List all stock symbols currently held (net quantity > 0). Use this first if the question doesn't name a specific symbol.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    "get_holdings": {
        "function": t.tool_get_holdings,
        "schema": {
            "name": "get_holdings",
            "description": "Current holdings (net quantity, total cost) for a symbol, or all symbols if none given.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string", "description": "e.g. 'HBL'. Omit for all holdings."}},
                "required": [],
            },
        },
    },
    "get_current_price": {
        "function": t.tool_get_current_price,
        "schema": {
            "name": "get_current_price",
            "description": "Most recently stored market price for a symbol.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    "get_dividend_income": {
        "function": t.tool_get_dividend_income,
        "schema": {
            "name": "get_dividend_income",
            "description": "Total net dividend income, optionally for one symbol and/or a payment_date range (YYYY-MM-DD).",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD, inclusive."},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD, inclusive."},
                },
                "required": [],
            },
        },
    },
    "get_transaction_history": {
        "function": t.tool_get_transaction_history,
        "schema": {
            "name": "get_transaction_history",
            "description": "Raw BUY/SELL transaction records, optionally filtered by symbol and/or trade_date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD, inclusive."},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD, inclusive."},
                },
                "required": [],
            },
        },
    },
    "calculate_unrealized_gain": {
        "function": t.tool_calculate_unrealized_gain,
        "schema": {
            "name": "calculate_unrealized_gain",
            "description": "Unrealized (paper) gain/loss for a currently-held symbol: (current price - avg cost/share) x quantity. Requires the symbol to be held AND have a stored current price.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        },
    },
    "calculate_realized_gain": {
        "function": t.tool_calculate_realized_gain,
        "schema": {
            "name": "calculate_realized_gain",
            "description": "Realized gain/loss from completed SELLs for a symbol, FIFO-matched, optionally within a sell-date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "start_date": {"type": "string", "description": "YYYY-MM-DD, inclusive."},
                    "end_date": {"type": "string", "description": "YYYY-MM-DD, inclusive."},
                },
                "required": ["symbol"],
            },
        },
    },
    "run_readonly_sql": {
        "function": lambda db_path, query: run_readonly_sql(db_path, query),
        "schema": {
            "name": "run_readonly_sql",
            "description": (
                "LAST RESORT ONLY -- use one of the other tools above if it covers the question. "
                "Runs a single read-only SELECT query against the portfolio database for questions "
                "the other tools don't cover. Allowed tables: transactions, dividends, market_price, "
                "source_documents, trade_confirmations. No CTEs, no writes, no ATTACH/PRAGMA. "
                "Results are capped and a LIMIT is enforced automatically."
            ),
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "A single SELECT statement."}},
                "required": ["query"],
            },
        },
    },
}


def get_tool_schemas() -> list[dict]:
    """What gets sent to the LLM so it knows what tools exist."""
    return [entry["schema"] for entry in TOOL_REGISTRY.values()]


def call_tool(name: str, db_path: str, **kwargs) -> dict:
    """Dispatch by name. An unknown tool name returns a structured error,
    not an exception -- the agent loop hands this back to the model as an
    observation ("that tool doesn't exist, try again"), rather than crashing."""
    entry = TOOL_REGISTRY.get(name)
    if entry is None:
        return {"success": False, "error": f"Unknown tool: '{name}'. Available tools: {list(TOOL_REGISTRY.keys())}"}
    return entry["function"](db_path, **kwargs)