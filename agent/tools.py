"""The four fixed tools the agent is allowed to call.

The LLM never writes pandas code. It only picks one of these functions and
supplies JSON arguments. Every argument is validated here; a bad argument
raises ``ToolError`` with a message the planner can act on during a retry.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import numpy as np
import pandas as pd


class ToolError(Exception):
    """A user/LLM-correctable problem: bad column, bad operator, bad agg..."""


ALLOWED_AGGS: tuple[str, ...] = ("sum", "mean", "median", "min", "max", "count", "std")
ALLOWED_OPS: tuple[str, ...] = ("==", "!=", ">", ">=", "<", "<=", "contains", "in")
PLOT_KINDS: tuple[str, ...] = ("bar", "line", "scatter", "hist")

MAX_GROUPS = 50
MAX_TABLE_ROWS = 200
MAX_PLOT_POINTS = 200
MAX_SCATTER_POINTS = 500
SAMPLE_ROWS = 5


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _json_safe(value: Any) -> Any:
    """Convert numpy/pandas scalars into JSON/Streamlit friendly values."""
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.datetime64,)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, 6)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timedelta,)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    return value


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {str(k): _json_safe(v) for k, v in row.items()}
        for row in df.to_dict(orient="records")
    ]


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _is_datetime(series: pd.Series) -> bool:
    return pd.api.types.is_datetime64_any_dtype(series)


def resolve_column(df: pd.DataFrame, name: Any, role: str = "column") -> str:
    """Return the real column name, tolerating case/whitespace differences."""
    if not isinstance(name, str) or not name.strip():
        raise ToolError(f"Missing {role} name. Available columns: {', '.join(map(str, df.columns))}.")

    if name in df.columns:
        return name

    lookup = {str(c).strip().lower(): c for c in df.columns}
    key = name.strip().lower()
    if key in lookup:
        return str(lookup[key])

    available = ", ".join(map(str, df.columns[:25]))
    raise ToolError(f"Column '{name}' was not found. Available columns: {available}.")


def _require_numeric(series: pd.Series, column: str, what: str) -> None:
    if not _is_numeric(series):
        raise ToolError(
            f"Column '{column}' is not numeric (dtype {series.dtype}); cannot {what}. "
            "Pick a numeric column."
        )


def _round_number(value: float) -> float:
    if value == 0:
        return 0.0
    return round(float(value), 4)


def _fmt_number(value: float) -> str:
    """Compact display for answers: 1310.0 -> 1,310 ; 0.3333 -> 0.3333."""
    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"
    return f"{value:,.4f}".rstrip("0").rstrip(".")


def _clean_label(value: Any) -> str:
    value = _json_safe(value)
    return "" if value is None else str(value)


def _table_result(
    df: pd.DataFrame,
    *,
    columns: Iterable[str],
    **meta: Any,
) -> dict[str, Any]:
    columns = list(columns)
    return {
        "type": "table",
        "columns": [str(c) for c in columns],
        "rows": _records(df[columns]),
        "row_count": int(len(df)),
        **{k: _json_safe(v) for k, v in meta.items()},
    }


# ---------------------------------------------------------------------------
# schema (given to the LLM, not a tool)
# ---------------------------------------------------------------------------
def build_schema(df: pd.DataFrame, sample_rows: int = SAMPLE_ROWS) -> str:
    """Compact text schema: shape, columns/types/example, first rows as a table."""
    lines = [
        f"Rows: {len(df):,}",
        f"Columns ({len(df.columns)}): " + ", ".join(map(str, df.columns)),
        "",
        "Column details:",
    ]
    for col in df.columns:
        series = df[col]
        example = series.dropna().iloc[0] if series.notna().any() else "null"
        extra = ""
        if _is_numeric(series) and series.notna().any():
            extra = f", range {_fmt_number(float(series.min()))}..{_fmt_number(float(series.max()))}"
        elif series.nunique(dropna=True) <= 12:
            uniques = ", ".join(_clean_label(v) for v in series.dropna().unique()[:12])
            extra = f", values: {uniques}"
        missing = int(series.isna().sum())
        missing_txt = f", {missing} missing" if missing else ""
        lines.append(
            f"- {col} ({series.dtype}): e.g. {_clean_label(example)}{extra}{missing_txt}"
        )

    head = df.head(sample_rows)
    if not head.empty:
        header = " | ".join(map(str, head.columns))
        divider = " | ".join("---" for _ in head.columns)
        lines += ["", f"First {len(head)} rows:", header, divider]
        for _, row in head.iterrows():
            lines.append(" | ".join(_clean_label(v) for v in row.values))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# tools
# ---------------------------------------------------------------------------
def describe_data(df: pd.DataFrame) -> dict[str, Any]:
    """Shape, dtypes, missing counts, unique counts and a sample of the data."""
    columns = []
    for col in df.columns:
        series = df[col]
        columns.append(
            {
                "name": str(col),
                "dtype": str(series.dtype),
                "missing": int(series.isna().sum()),
                "unique": int(series.nunique(dropna=True)),
            }
        )
    return {
        "type": "describe",
        "rows": int(len(df)),
        "columns": columns,
        "column_count": int(len(df.columns)),
        "sample": _records(df.head(SAMPLE_ROWS)),
        "sample_columns": [str(c) for c in df.columns],
        "numeric_columns": [str(c) for c in df.columns if _is_numeric(df[c])],
    }


def aggregate(
    df: pd.DataFrame,
    group_by: str | None = None,
    column: str | None = None,
    agg: str = "sum",
) -> dict[str, Any]:
    """Group by a column and reduce another column with one aggregation."""
    agg = (agg or "sum").strip().lower()
    if agg not in ALLOWED_AGGS:
        raise ToolError(
            f"Unknown aggregation '{agg}'. Use one of: {', '.join(ALLOWED_AGGS)}."
        )

    if column is None and agg != "count":
        raise ToolError(f"Aggregation '{agg}' needs a 'column' to aggregate.")

    value_col = resolve_column(df, column, "value column") if column is not None else None

    if value_col is not None and agg not in ("count",):
        _require_numeric(df[value_col], value_col, f"compute {agg}")

    if group_by is None or str(group_by).strip() == "":
        if value_col is None:
            total: Any = int(len(df))
        else:
            series = df[value_col].dropna()
            total = getattr(series, agg)() if agg != "count" else int(series.count())
        return {
            "type": "table",
            "columns": [agg] if value_col is None else [f"{agg}({value_col})"],
            "rows": [{agg if value_col is None else f"{agg}({value_col})": _round_number(float(total))}],
            "row_count": 1,
            "group_by": None,
            "value_column": value_col,
            "agg": agg,
        }

    group_col = resolve_column(df, group_by, "group_by column")

    working = df[[group_col] + ([value_col] if value_col is not None else [])].copy()
    working[group_col] = working[group_col].fillna("(missing)").astype(str).str.strip()
    working = working[working[group_col] != ""]

    if value_col is None:
        grouped = working.groupby(group_col, dropna=False).size()
    elif agg == "count":
        grouped = working.groupby(group_col, dropna=False)[value_col].count()
    else:
        grouped = working.groupby(group_col, dropna=False)[value_col].agg(agg)

    grouped = grouped.dropna()
    if grouped.empty:
        raise ToolError(f"No values to aggregate for '{value_col or group_col}'.")

    grouped = grouped.sort_values(ascending=False).head(MAX_GROUPS)
    total_groups = int(df[group_col].nunique(dropna=True))
    out = pd.DataFrame({group_col: grouped.index, agg: grouped.to_numpy()})
    result = _table_result(
        out,
        columns=[group_col, agg],
        group_by=group_col,
        value_column=value_col,
        agg=agg,
        total_groups=total_groups,
        truncated=max(0, total_groups - len(grouped)),
    )
    result["type"] = "table"
    return result


def filter_rows(
    df: pd.DataFrame,
    column: str,
    op: str,
    value: Any = None,
) -> dict[str, Any]:
    """Keep rows matching a simple condition, e.g. age > 30."""
    col = resolve_column(df, column)
    op = str(op or "").strip()
    if op not in ALLOWED_OPS:
        raise ToolError(
            f"Unknown operator '{op}'. Use one of: {', '.join(ALLOWED_OPS)}."
        )

    series = df[col]
    if op in ("contains",):
        if value is None:
            raise ToolError("Operator 'contains' needs a text value.")
        mask = series.astype(str).str.contains(str(value), case=False, na=False, regex=False)
    elif op == "in":
        if isinstance(value, str):
            candidates = [v.strip() for v in value.split(",") if v.strip()]
        elif isinstance(value, (list, tuple, set)):
            candidates = list(value)
        else:
            raise ToolError("Operator 'in' needs a list of values.")
        if not candidates:
            raise ToolError("Operator 'in' needs at least one value.")
        mask = series.isin(candidates)
        if not bool(mask.any()):
            # tolerate "30" vs 30 style mismatches
            typed = []
            for candidate in candidates:
                converted = _coerce(value=candidate, series=series)
                if converted is not None:
                    typed.append(converted)
            if typed:
                mask = series.isin(typed)
    else:
        if value is None:
            raise ToolError(f"Operator '{op}' needs a value.")
        coerced = _coerce(value=value, series=series)
        if coerced is None:
            raise ToolError(
                f"Could not compare {series.dtype} column '{col}' with value {value!r}."
            )
        try:
            if op == "==":
                mask = series == coerced
            elif op == "!=":
                mask = series != coerced
            elif op == ">":
                mask = series > coerced
            elif op == ">=":
                mask = series >= coerced
            elif op == "<":
                mask = series < coerced
            else:
                mask = series <= coerced
        except TypeError as exc:  # pragma: no cover - defensive
            raise ToolError(f"Could not compare column '{col}' with {value!r}: {exc}") from exc

    mask = mask.fillna(False)
    matched = df.loc[mask]
    if len(matched) > MAX_TABLE_ROWS:
        shown = matched.head(MAX_TABLE_ROWS)
        truncated = len(matched) - MAX_TABLE_ROWS
    else:
        shown = matched
        truncated = 0

    result = _table_result(
        shown,
        columns=list(df.columns),
        filter={"column": col, "op": op, "value": _json_safe(value)},
        matched=int(len(matched)),
        total=int(len(df)),
        truncated=truncated,
    )
    result["type"] = "filter"
    return result


def _coerce(value: Any, series: pd.Series) -> Any:
    """Best-effort conversion of a JSON value to the column's dtype."""
    if value is None:
        return None
    try:
        if _is_datetime(series):
            return pd.to_datetime(value)
        if _is_numeric(series):
            if isinstance(value, bool):
                return None
            return float(value)
        if pd.api.types.is_bool_dtype(series):
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "yes", "1"):
                    return True
                if lowered in ("false", "no", "0"):
                    return False
            return None
    except (TypeError, ValueError):
        return None
    return value


def plot(
    df: pd.DataFrame,
    kind: str = "bar",
    x: str | None = None,
    y: str | None = None,
    agg: str = "sum",
) -> dict[str, Any]:
    """Build a chart spec (data points, not a figure) for the UI to render."""
    kind = str(kind or "bar").strip().lower()
    if kind not in PLOT_KINDS:
        raise ToolError(f"Unknown chart kind '{kind}'. Use one of: {', '.join(PLOT_KINDS)}.")
    if x is None or str(x).strip() == "":
        raise ToolError("A chart needs an 'x' column.")
    x_col = resolve_column(df, x, "x column")

    if kind == "hist":
        _require_numeric(df[x_col], x_col, "draw a histogram")
        values = pd.to_numeric(df[x_col], errors="coerce").dropna()
        if values.empty:
            raise ToolError(f"Column '{x_col}' has no numeric values to plot.")
        counts, edges = np.histogram(values.to_numpy(dtype=float), bins=min(20, max(5, len(values) // 2)))
        points = [
            {"x": f"{_fmt_number(float(edges[i]))}–{_fmt_number(float(edges[i + 1]))}", "y": int(counts[i])}
            for i in range(len(counts))
        ]
        return {
            "type": "plot",
            "kind": "hist",
            "x": x_col,
            "y": "count",
            "x_label": x_col,
            "y_label": "count",
            "points": points,
            "point_count": len(points),
        }

    if y is None or str(y).strip() == "":
        raise ToolError(f"A {kind} chart needs a 'y' column.")
    y_col = resolve_column(df, y, "y column")
    _require_numeric(df[y_col], y_col, "plot")

    if kind == "scatter":
        sample = df[[x_col, y_col]].dropna().head(MAX_SCATTER_POINTS)
        if sample.empty:
            raise ToolError(f"No rows with both '{x_col}' and '{y_col}' filled in.")
        points = [{"x": _json_safe(a), "y": _round_number(float(b))} for a, b in zip(sample[x_col], sample[y_col])]
        return {
            "type": "plot",
            "kind": "scatter",
            "x": x_col,
            "y": y_col,
            "x_label": x_col,
            "y_label": y_col,
            "points": points,
            "point_count": len(points),
        }

    agg = str(agg or "sum").strip().lower()
    if agg not in ALLOWED_AGGS:
        raise ToolError(f"Unknown aggregation '{agg}'. Use one of: {', '.join(ALLOWED_AGGS)}.")

    working = df[[x_col, y_col]].dropna(subset=[y_col]).copy()
    working[x_col] = working[x_col].fillna("(missing)").astype(str).str.strip()
    if working.empty:
        raise ToolError(f"No rows with '{y_col}' filled in.")

    grouped = working.groupby(x_col, dropna=False)[y_col].agg(agg)
    grouped = grouped.dropna()

    if kind == "line":
        if _is_datetime(df[x_col]):
            order = (
                df[[x_col, y_col]]
                .dropna(subset=[x_col])
                .groupby(x_col, dropna=False)[y_col]
                .agg(agg)
                .sort_index()
                .tail(MAX_PLOT_POINTS)
            )
            grouped = order
        else:
            grouped = grouped.sort_index().head(MAX_PLOT_POINTS)
    else:
        grouped = grouped.sort_values(ascending=False).head(MAX_PLOT_POINTS)

    points = [{"x": _clean_label(idx), "y": _round_number(float(v))} for idx, v in grouped.items()]
    return {
        "type": "plot",
        "kind": kind,
        "x": x_col,
        "y": y_col,
        "agg": agg,
        "x_label": x_col,
        "y_label": f"{agg}({y_col})",
        "points": points,
        "point_count": len(points),
    }


# ---------------------------------------------------------------------------
# dispatch + formatting
# ---------------------------------------------------------------------------
TOOLS: dict[str, Any] = {
    "describe_data": describe_data,
    "aggregate": aggregate,
    "filter_rows": filter_rows,
    "plot": plot,
}

TOOL_SIGNATURES: dict[str, str] = {
    "describe_data": "describe_data() -> shape, column types, missing counts, sample rows",
    "aggregate": 'aggregate(group_by, column, agg) -> table; agg in sum|mean|median|min|max|count|std; group_by optional',
    "filter_rows": 'filter_rows(column, op, value) -> matching rows; op in ==|!=|>|>=|<|<=|contains|in',
    "plot": 'plot(kind, x, y, agg) -> chart points; kind in bar|line|scatter|hist; y optional for hist',
}


def call_tool(name: str, df: pd.DataFrame, args: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run a tool by name, validating the name and refusing unknown kwargs."""
    if name not in TOOLS:
        raise ToolError(f"Unknown tool '{name}'. Use one of: {', '.join(TOOLS)}.")
    args = dict(args or {})
    func = TOOLS[name]
    allowed = set(func.__code__.co_varnames[1 : func.__code__.co_argcount])
    unexpected = set(args) - allowed
    if unexpected:
        raise ToolError(
            f"Tool '{name}' does not accept {', '.join(sorted(unexpected))}. "
            f"Accepted arguments: {', '.join(sorted(allowed))}."
        )
    return func(df, **args)


def summarise_result(result: dict[str, Any], max_rows: int = 8) -> str:
    """A short, LLM-readable rendering of a tool result."""
    kind = result.get("type")

    if kind == "describe":
        cols = ", ".join(f"{c['name']} ({c['dtype']})" for c in result["columns"][:20])
        return f"{result['rows']} rows, {result['column_count']} columns. Columns: {cols}."

    if kind in ("table", "filter"):
        rows = result.get("rows", [])
        shown = rows[:max_rows]
        header = " | ".join(result.get("columns", []))
        body = "\n".join(" | ".join(_clean_label(r.get(c)) for c in result.get("columns", [])) for r in shown)
        prefix = ""
        if kind == "filter":
            prefix = f"{result['matched']} of {result['total']} rows matched. "
        elif result.get("group_by"):
            prefix = f"Grouped by {result['group_by']} ({result.get('agg')}). "
        suffix = f"\n(and {len(rows) - len(shown)} more rows)" if len(rows) > len(shown) else ""
        return f"{prefix}{len(rows)} rows shown.\n{header}\n{body}{suffix}"

    if kind == "plot":
        points = result.get("points", [])
        shown = points[:10]
        body = ", ".join(f"{p['x']}: {_fmt_number(p['y'])}" for p in shown)
        return (
            f"{result['kind']} chart of {result.get('y_label')} by {result.get('x_label')} "
            f"with {len(points)} points. Top/leading values: {body}."
        )

    return str(result)[:1000]
