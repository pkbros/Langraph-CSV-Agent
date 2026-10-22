"""Graph nodes: router -> planner -> executor -> check -> responder."""

import json
from typing import Any

from langchain_core.runnables import RunnableConfig

from .llm import ChatModel, LLMError
from .state import AgentState, STORE
from .tools import (
    TOOL_SIGNATURES,
    TOOLS,
    ToolError,
    call_tool,
    summarise_result,
)

MAX_HISTORY = 10


# ---------------------------------------------------------------------------
# prompts
# ---------------------------------------------------------------------------
ROUTER_SYSTEM = """You are the router of a CSV analyst agent.

Decide whether answering the user needs the loaded dataframe.
- "tool": the question asks about values, rows, totals, averages, comparisons,
  trends, filters, charts, or anything that must be computed from the data.
- "tool" also covers facts about the file itself: how many rows or columns,
  column names or types, missing values, a sample of rows, and overview
  questions such as "what is this data about?" or "summarise this file".
  The schema is only a map, not the data, so those answers need a tool.
- "chat": only greetings, thanks, questions about you (the assistant), or
  general conversation that needs no data.

Reply with JSON only, no prose:
{"route": "tool" | "chat", "reason": "<max 12 words>"}"""

PLANNER_SYSTEM = f"""You are the planner of a CSV analyst agent.

Pick exactly ONE tool and fill its arguments using ONLY the column names that
exist in the schema. Never invent a column.

Tools:
{chr(10).join('- ' + line for line in TOOL_SIGNATURES.values())}

Rules:
- "sales by region"        -> aggregate(group_by="region", column="sales", agg="sum")
- "how many rows"          -> describe_data()
- "what is this data about" -> describe_data()
- "summarise this file"    -> describe_data()
- "show orders over 500"   -> filter_rows(column="amount", op=">", value=500)
- "chart of sales by region" -> plot(kind="bar", x="region", y="sales")
- Use the exact column spelling from the schema, including case.
- If the user names a column that is not in the schema (for example "profit"),
  pass it unchanged and let the tool report the real columns. NEVER quietly
  swap it for a different metric such as "sales".
- If a previous attempt failed, read the error and fix the arguments. When the
  error lists available columns and one is an obvious spelling variant of the
  requested one, use it. Otherwise keep the requested column unchanged.

Reply with JSON only, no prose:
{{"tool_name": "<tool>", "tool_args": {{...}}}}"""

RESPONDER_SYSTEM = """You are a friendly data analyst answering questions about a CSV.

Write 1-3 sentences in plain English. Use the exact numbers from the tool
output. Do not invent rows, numbers or columns. A table or chart is shown to
the user next to your message, so do not repeat every row. If the output is a
chart, say what it shows. Write numbers naturally, for example 44,490 rather
than 44,490.0. Never mention JSON, tools or internal steps."""

CHAT_SYSTEM = """You are a friendly CSV analyst assistant. A CSV is loaded and its
schema appears below. Answer briefly in 1-2 sentences. Never say you cannot access
the file or ask the user to upload it. If they ask what the data is about, use the
column names and shape you were given. If they want numbers, invite a concrete
question (totals, averages, filters or charts)."""


def _trace(state: AgentState, line: str) -> list[str]:
    return [*state.get("trace", []), line]


def _short(value: Any, limit: int = 160) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str) if not isinstance(value, str) else value
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _history_block(state: AgentState) -> str:
    history = state.get("history") or []
    if not history:
        return "(none)"
    lines = []
    for turn in history[-MAX_HISTORY:]:
        lines.append(f"Q: {turn.get('question', '')}\nA: {turn.get('answer', '')}")
    return "\n".join(lines)


def router_schema(schema: str) -> str:
    """A lean schema for routing: shape plus column names/types, no sample rows.

    The router only needs to know whether the question is about the data and
    which columns exist. Ranges, unique values and sample rows go to the
    planner, which has to choose arguments.
    """
    if not schema:
        return "(no data loaded)"

    lines = []
    for line in schema.splitlines():
        if line.startswith(("Rows:", "Columns (")):
            lines.append(line)
        elif line.startswith("- "):
            lines.append(line.split(":", 1)[0])
    return "\n".join(lines) if lines else schema[:400]


def _fallback_note(exc: LLMError) -> str:
    if "rate limit" in str(exc).lower():
        return (
            "The model is rate-limited right now, so this answer came straight "
            "from the data tools. Try again in a few seconds."
        )
    return "The model is unavailable right now, so this answer came straight from the data tools."


# ---------------------------------------------------------------------------
# nodes
# ---------------------------------------------------------------------------
def make_router(llm: ChatModel):
    def router(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        schema = state.get("schema", "")
        user = (
            f"DATA SCHEMA\n{router_schema(schema)}\n\n"
            f"CONVERSATION SO FAR\n{_history_block(state)}\n\n"
            f"CURRENT QUESTION\n{state.get('question', '')}"
        )
        try:
            payload = llm.json(ROUTER_SYSTEM, user)
            route = str(payload.get("route", "")).strip().lower()
            reason = str(payload.get("reason", "")).strip()
            if route not in ("tool", "chat"):
                raise LLMError(f"invalid route {route!r}")
        except LLMError as exc:
            route = "tool" if schema else "chat"
            reason = f"router fallback ({exc})"
        if route == "tool" and not schema:
            route = "chat"
            reason = "no data loaded"

        return {
            "route": route,
            "trace": _trace(state, f"router → {route} ({reason or 'no reason given'})"),
        }

    return router


def make_planner(llm: ChatModel):
    def planner(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        error = state.get("error")
        error_block = (
            f"\n\nPREVIOUS ATTEMPT FAILED WITH:\n{error}\nFix the arguments." if error else ""
        )
        user = (
            f"DATA SCHEMA\n{state.get('schema', '')}\n\n"
            f"CONVERSATION SO FAR\n{_history_block(state)}\n\n"
            f"QUESTION\n{state.get('question', '')}{error_block}"
        )
        try:
            payload = llm.json(PLANNER_SYSTEM, user)
        except LLMError as exc:
            return {
                "tool_name": "",
                "tool_args": {},
                "error": f"The planner could not produce a plan: {exc}",
                "trace": _trace(state, f"planner → failed ({exc})"),
            }

        name = str(payload.get("tool_name") or payload.get("tool") or "").strip()
        args = payload.get("tool_args", payload.get("args", payload.get("arguments", {})))
        if not isinstance(args, dict):
            args = {}
        if name not in TOOLS:
            return {
                "tool_name": name,
                "tool_args": args,
                "error": f"Planner chose unknown tool '{name}'. Available: {', '.join(TOOLS)}.",
                "trace": _trace(state, f"planner → unknown tool {name!r}"),
            }

        return {
            "tool_name": name,
            "tool_args": args,
            "error": None if error is None else error,
            "trace": _trace(state, f"planner → {name}({_short(args, 120)})"),
        }

    return planner


def executor(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
    """Run the planned tool against the dataframe registered for this thread."""
    configurable = (config or {}).get("configurable", {}) if config else {}
    thread_id = str(configurable.get("thread_id", ""))
    df = STORE.frame(thread_id) if thread_id else None

    if df is None:
        message = "No CSV is loaded yet. Upload one in the sidebar and try again."
        return {
            "result": None,
            "error": message,
            "trace": _trace(state, f"executor → no dataframe for thread {thread_id or '(none)'}"),
        }

    name = state.get("tool_name", "")
    args = state.get("tool_args", {}) or {}
    if name not in TOOLS:
        message = state.get("error") or (
            f"The planner did not pick a valid tool (got '{name}'). "
            f"Available: {', '.join(TOOLS)}."
        )
        return {
            "result": None,
            "error": message,
            "trace": _trace(state, f"executor → skipped, invalid tool {name!r}"),
        }
    try:
        result = call_tool(name, df, args)
    except ToolError as exc:
        return {
            "result": None,
            "error": str(exc),
            "trace": _trace(state, f"executor → {name} failed: {_short(str(exc), 120)}"),
        }
    except Exception as exc:  # noqa: BLE001 - must never crash the graph
        return {
            "result": None,
            "error": f"Unexpected error while running {name}: {type(exc).__name__}: {exc}",
            "trace": _trace(state, f"executor → {name} crashed: {type(exc).__name__}"),
        }

    summary = summarise_result(result)
    return {
        "result": result,
        "error": None,
        "trace": _trace(state, f"executor → {name} ok ({_short(summary.splitlines()[0], 100)})"),
    }


def make_check(max_retries: int):
    def check(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        error = state.get("error")
        retries = int(state.get("retries", 0) or 0)

        if error and retries < max_retries:
            retries += 1
            return {
                "retries": retries,
                "will_retry": True,
                "trace": _trace(state, f"check → failed, retry {retries}/{max_retries}: {_short(error, 110)}"),
            }
        if error:
            return {
                "retries": retries,
                "will_retry": False,
                "trace": _trace(state, f"check → giving up after {retries} retries"),
            }
        return {
            "retries": retries,
            "will_retry": False,
            "trace": _trace(state, "check → result looks valid"),
        }

    return check


def _fallback_answer(result: dict[str, Any] | None) -> str:
    """Deterministic answer used when the phrasing call fails."""
    if not result:
        return "I could not compute anything for that question."

    kind = result.get("type")
    if kind == "describe":
        cols = ", ".join(str(c["name"]) for c in result.get("columns", [])[:8])
        return f"The file has {result['rows']:,} rows and {result['column_count']} columns: {cols}."

    if kind in ("table", "filter"):
        rows = result.get("rows") or []
        columns = result.get("columns") or []
        if kind == "filter":
            head = f"{result['matched']:,} of {result['total']:,} rows match."
        elif result.get("group_by"):
            head = f"Results grouped by {result['group_by']} ({result.get('agg')})."
        else:
            head = "Here is the result."
        if not rows or len(columns) < 2:
            return head
        label_col, value_col = columns[0], columns[1]
        top = rows[0]
        value = top.get(value_col)
        value_txt = f"{value:,.2f}".rstrip("0").rstrip(".") if isinstance(value, (int, float)) else str(value)
        return f"{head} The top entry is {top.get(label_col)} with {value_txt} {value_col}."

    if kind == "plot":
        return (
            f"Here is a {result.get('kind')} chart of {result.get('y_label')} "
            f"by {result.get('x_label')} ({result.get('point_count', 0)} points)."
        )

    return "Here is what I found in the data."


def make_responder(llm: ChatModel):
    def responder(state: AgentState, config: RunnableConfig | None = None) -> dict[str, Any]:
        question = state.get("question", "")
        error = state.get("error")
        result = state.get("result")
        route = state.get("route", "tool")

        if route == "chat":
            user = (
                f"DATA FILE\n{(state.get('schema') or '(no data loaded)')[:800]}\n\n"
                f"CONVERSATION SO FAR\n{_history_block(state)}\n\nUSER\n{question}"
            )
            try:
                answer = llm.text(CHAT_SYSTEM, user)
            except LLMError as exc:
                answer = (
                    "Hi! Your CSV is loaded. Ask me about totals, filters or charts, "
                    "or say 'summarise this dataset'."
                    if not state.get("history")
                    else "I'm having trouble reaching the model right now. Your file is "
                    "still loaded — please try again in a moment."
                )
                trace = _trace(state, f"responder → chat fallback ({_short(str(exc), 100)})")
            else:
                trace = _trace(state, "responder → chat reply")
        elif error:
            answer = (
                "I couldn't answer that one. "
                f"{error}\n\n"
                "Try rephrasing with an exact column name from the sidebar preview."
            )
            trace = _trace(state, "responder → friendly error message")
        else:
            user = (
                f"QUESTION\n{question}\n\n"
                f"TOOL OUTPUT\n{summarise_result(result) if result else '(nothing)'}"
            )
            try:
                answer = llm.text(RESPONDER_SYSTEM, user)
            except LLMError as exc:
                answer = f"{_fallback_answer(result)}\n\n_{_fallback_note(exc)}_"
                trace = _trace(state, f"responder → fallback ({_short(str(exc), 100)})")
            else:
                trace = _trace(state, "responder → final answer")

        history = [*(state.get("history") or []), {"question": question, "answer": answer}]
        history = history[-MAX_HISTORY:]
        return {"answer": answer, "history": history, "trace": trace}

    return responder


def after_router(state: AgentState) -> str:
    return "planner" if state.get("route") == "tool" else "responder"


def after_check(state: AgentState) -> str:
    return "planner" if state.get("will_retry") else "responder"
