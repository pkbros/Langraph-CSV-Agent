"""Offline heuristic stand-in for the LLM.

Used when ``GROQ_API_KEY`` is missing so the app and the demo still work.
It is deliberately small: keyword rules over the question and the schema that
the nodes put in the prompt. Real answers come from the Qwen model.
"""

from __future__ import annotations

import re
from typing import Any

_NUMERIC_HINTS = (
    "sales",
    "revenue",
    "amount",
    "price",
    "total",
    "units",
    "quantity",
    "qty",
    "score",
    "marks",
    "age",
    "cost",
    "profit",
    "count",
)


class FakeLLM:
    """Rule-based router/planner/responder with no network access."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _columns(prompt: str) -> list[tuple[str, str]]:
        """Pull (name, dtype) pairs out of the schema block."""
        pairs = re.findall(r"^- (\S+) \(([^)]+)\)", prompt, flags=re.MULTILINE)
        return [(name, dtype) for name, dtype in pairs]

    @staticmethod
    def _question(prompt: str) -> str:
        match = re.search(r"(?:CURRENT QUESTION|QUESTION)\n(.+?)(?:\n\n|\Z)", prompt, flags=re.DOTALL)
        return match.group(1).strip() if match else prompt.strip().splitlines()[-1]

    def _pick(self, prompt: str) -> dict[str, Any]:
        question = self._question(prompt).lower()
        columns = self._columns(prompt)
        numeric = [name for name, dtype in columns if "int" in dtype or "float" in dtype]
        categorical = [name for name, dtype in columns if name not in numeric]

        def find(words: tuple[str, ...], pool: list[str]) -> str | None:
            for name in pool:
                if name.lower() in question:
                    return name
            for name in pool:
                if any(word in name.lower() for word in words):
                    return name
            return None

        if any(word in question for word in ("how many rows", "number of rows", "describe", "summary", "summarize", "summarise", "columns", "missing", "schema", "what data", "data about", "overview")):
            return {"tool_name": "describe_data", "tool_args": {}}

        # filter: "over 500", "above 30", "older than 30"
        number = re.search(r"(?:over|above|greater than|more than|>|under|below|less than|<)\s*(-?\d+(?:\.\d+)?)", question)
        if number and any(word in question for word in ("over", "above", "greater", "more than", ">", "under", "below", "less", "less than", "<")):
            value = float(number.group(1))
            value = int(value) if value.is_integer() else value
            op = ">" if any(word in question for word in ("over", "above", "greater", "more than", ">")) else "<"
            column = find(("amount", "sales", "age", "price", "score"), numeric)
            return {"tool_name": "filter_rows", "tool_args": {"column": column or "", "op": op, "value": value}}

        # chart requests
        if any(word in question for word in ("chart", "plot", "graph", "visual", "histogram")):
            kind = "line" if "line" in question else "scatter" if "scatter" in question else "hist" if "histogram" in question else "bar"
            by_match = re.search(r"by ([a-z_ ]+?)(?:\?|$|,)", question)
            x = None
            if by_match:
                candidate = by_match.group(1).strip()
                x = next((name for name in [c for c, _ in columns] if name.lower() in candidate), None)
            x = x or find(("region", "category", "product", "month", "date", "name"), categorical) or (columns[0][0] if columns else None)
            if kind == "hist":
                return {"tool_name": "plot", "tool_args": {"kind": "hist", "x": find(("sales", "amount", "price", "age"), numeric) or x}}
            y = find(("sales", "revenue", "amount", "units", "price", "score", "marks", "profit"), numeric) or (numeric[0] if numeric else None)
            if x and y:
                return {"tool_name": "plot", "tool_args": {"kind": kind, "x": x, "y": y}}

        # aggregation: "sales by region", "average marks by class", "which region sold most"
        agg = "sum"
        if any(word in question for word in ("average", "mean")):
            agg = "mean"
        elif any(word in question for word in ("median",)):
            agg = "median"
        elif any(word in question for word in ("max", "highest", "largest", "top", "most")):
            agg = "sum" if any(word in question for word in ("total", "sales", "revenue")) else "max"
        elif any(word in question for word in ("count", "how many")):
            agg = "count"
        elif any(word in question for word in ("min", "lowest", "smallest", "least")):
            agg = "min"

        group = find(("region", "country", "category", "product", "month", "class", "name", "city", "department"), categorical)
        value = find(("sales", "revenue", "amount", "units", "price", "score", "marks", "profit"), numeric)

        if group and (value or agg == "count"):
            args: dict[str, Any] = {"group_by": group, "agg": agg}
            if value:
                args["column"] = value
            return {"tool_name": "aggregate", "tool_args": args}

        if value:
            metric = "total" if agg == "sum" else agg
            return {"tool_name": "aggregate", "tool_args": {"column": value, "agg": agg}}
        if columns:
            return {"tool_name": "describe_data", "tool_args": {}}
        return {"tool_name": "describe_data", "tool_args": {}}

    # -- ChatModel protocol ------------------------------------------------
    def json(self, system: str, user: str) -> dict[str, Any]:
        self.calls.append(("json", system[:40]))
        if "router" in system.lower():
            data_words = (
                "row", "column", "data", "csv", "total", "average", "mean", "sum", "max",
                "min", "count", "chart", "plot", "graph", "filter", "show", "which",
                "top", "highest", "lowest", "by ", "trend", "compare", "how many",
                "summarise", "summarize", "summary", "describe", "overview",
            )
            question = self._question(user)
            needs_data = "DATA SCHEMA" in user and (
                any(word in question.lower() for word in data_words) or "?" in question
            )
            return {"route": "tool" if needs_data else "chat", "reason": "offline heuristic"}
        payload = self._pick(user)
        return payload

    def text(self, system: str, user: str) -> str:
        self.calls.append(("text", system[:40]))
        if "general conversation" in system.lower() or "friendly csv analyst assistant" in system.lower():
            return (
                "I can summarise this file, compute totals or averages by group, "
                "filter rows, and draw charts. Ask something like "
                "'total sales by region' or 'show a bar chart of sales by region'."
            )
        output = ""
        match = re.search(r"TOOL OUTPUT\n(.+)", user, flags=re.DOTALL)
        if match:
            output = match.group(1).strip()
        first_lines = " ".join(output.splitlines()[:3])
        if len(first_lines) > 320:
            first_lines = first_lines[:317] + "…"
        return f"Based on the data: {first_lines}" if first_lines else "I could not find an answer in this file."
