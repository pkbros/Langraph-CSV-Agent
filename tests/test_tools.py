"""Unit tests for the fixed tool functions."""

from __future__ import annotations

import pandas as pd
import pytest

from agent.tools import (
    ALLOWED_AGGS,
    ToolError,
    aggregate,
    build_schema,
    call_tool,
    describe_data,
    filter_rows,
    plot,
    resolve_column,
    summarise_result,
)


# ---------------------------------------------------------------------------
# schema + describe
# ---------------------------------------------------------------------------
def test_build_schema_lists_columns_types_and_sample(sales_df):
    schema = build_schema(sales_df)
    assert "Rows: 6" in schema
    assert "Columns (5)" in schema
    for column in sales_df.columns:
        assert column in schema
    assert "First 5 rows" in schema


def test_describe_data_reports_shape_and_missing(sales_df):
    df = sales_df.copy()
    df.loc[0, "sales"] = None
    result = describe_data(df)

    assert result["type"] == "describe"
    assert result["rows"] == 6
    assert result["column_count"] == 5
    sales = next(c for c in result["columns"] if c["name"] == "sales")
    assert sales["missing"] == 1
    assert len(result["sample"]) == 5
    assert "units" in result["numeric_columns"]
    assert "region" not in result["numeric_columns"]


def test_resolve_column_is_case_insensitive(sales_df):
    assert resolve_column(sales_df, "REGION") == "region"
    assert resolve_column(sales_df, " Sales ") == "sales"
    with pytest.raises(ToolError, match="not found"):
        resolve_column(sales_df, "revenue")


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------
def test_aggregate_sum_by_region(sales_df):
    result = aggregate(sales_df, group_by="region", column="sales", agg="sum")

    assert result["type"] == "table"
    assert result["group_by"] == "region"
    rows = {row["region"]: row["sum"] for row in result["rows"]}
    assert rows == {"West": 1610.0, "East": 910.0, "North": 210.0}
    # sorted descending
    assert result["rows"][0]["region"] == "West"


def test_aggregate_mean_and_count(sales_df):
    mean_result = aggregate(sales_df, group_by="region", column="sales", agg="mean")
    west = next(r for r in mean_result["rows"] if r["region"] == "West")
    assert west["mean"] == pytest.approx(1610 / 3, rel=1e-3)

    count_result = aggregate(sales_df, group_by="region", column="units", agg="count")
    assert next(r for r in count_result["rows"] if r["region"] == "West")["count"] == 3


def test_aggregate_without_group_by_returns_single_value(sales_df):
    result = aggregate(sales_df, column="sales", agg="sum")
    assert result["row_count"] == 1
    assert result["rows"][0]["sum(sales)"] == 2730.0


def test_aggregate_accepts_case_insensitive_columns(sales_df):
    result = aggregate(sales_df, group_by="Region", column="Sales", agg="sum")
    assert result["rows"][0]["region"] == "West"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"group_by": "region", "column": "revenue", "agg": "sum"}, "not found"),
        ({"group_by": "region", "column": "sales", "agg": "sums"}, "Unknown aggregation"),
        ({"group_by": "region", "column": "product", "agg": "sum"}, "not numeric"),
        ({"group_by": "region", "agg": "sum"}, "needs a 'column'"),
        ({"group_by": "nope", "column": "sales", "agg": "sum"}, "not found"),
    ],
)
def test_aggregate_rejects_bad_arguments(sales_df, kwargs, match):
    with pytest.raises(ToolError, match=match):
        aggregate(sales_df, **kwargs)


def test_all_allowed_aggs_run(sales_df):
    for agg in ALLOWED_AGGS:
        result = aggregate(sales_df, group_by="region", column="sales", agg=agg)
        assert result["rows"]


# ---------------------------------------------------------------------------
# filter_rows
# ---------------------------------------------------------------------------
def test_filter_rows_greater_than(sales_df):
    result = filter_rows(sales_df, column="sales", op=">", value=500)
    assert result["type"] == "filter"
    assert result["matched"] == 2
    assert all(row["sales"] > 500 for row in result["rows"])


def test_filter_rows_coerces_numeric_strings(sales_df):
    result = filter_rows(sales_df, column="units", op="<=", value="3")
    assert result["matched"] == 3
    assert all(row["units"] <= 3 for row in result["rows"])


def test_filter_rows_contains_and_in(sales_df):
    widget = filter_rows(sales_df, column="product", op="contains", value="wid")
    assert widget["matched"] == 3

    west_east = filter_rows(sales_df, column="region", op="in", value=["West", "East"])
    assert west_east["matched"] == 5

    sales_in = filter_rows(sales_df, column="sales", op="in", value=[400, 650])
    assert sales_in["matched"] == 2


def test_filter_rows_equality_and_dates(sales_df):
    result = filter_rows(sales_df, column="region", op="==", value="North")
    assert result["matched"] == 1
    assert result["rows"][0]["product"] == "Gizmo"

    dated = filter_rows(sales_df, column="date", op=">=", value="2024-03-01")
    assert dated["matched"] == 2


def test_filter_rows_no_matches_is_not_an_error(sales_df):
    result = filter_rows(sales_df, column="sales", op=">", value=10_000)
    assert result["matched"] == 0
    assert result["rows"] == []


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"column": "revenue", "op": ">", "value": 1}, "not found"),
        ({"column": "sales", "op": "~", "value": 1}, "Unknown operator"),
        ({"column": "sales", "op": ">", "value": None}, "needs a value"),
        ({"column": "product", "op": "contains", "value": None}, "needs a text value"),
        ({"column": "region", "op": "in", "value": 5}, "needs a list"),
        ({"column": "region", "op": "in", "value": []}, "at least one value"),
    ],
)
def test_filter_rows_rejects_bad_arguments(sales_df, kwargs, match):
    with pytest.raises(ToolError, match=match):
        filter_rows(sales_df, **kwargs)


# ---------------------------------------------------------------------------
# plot
# ---------------------------------------------------------------------------
def test_plot_bar_aggregates_by_x(sales_df):
    result = plot(sales_df, kind="bar", x="region", y="sales")

    assert result["type"] == "plot"
    assert result["kind"] == "bar"
    points = {p["x"]: p["y"] for p in result["points"]}
    assert points == {"West": 1610.0, "East": 910.0, "North": 210.0}
    assert result["points"][0]["x"] == "West"


def test_plot_line_sorts_by_x(sales_df):
    result = plot(sales_df, kind="line", x="date", y="sales")
    xs = [p["x"] for p in result["points"]]
    assert xs == sorted(xs)
    assert result["y_label"] == "sum(sales)"


def test_plot_scatter_keeps_raw_pairs(sales_df):
    result = plot(sales_df, kind="scatter", x="units", y="sales")
    assert result["point_count"] == 6
    assert all(isinstance(p["y"], (int, float)) for p in result["points"])


def test_plot_hist(sales_df):
    result = plot(sales_df, kind="hist", x="sales")
    assert result["kind"] == "hist"
    assert result["points"]
    assert all(isinstance(p["y"], int) for p in result["points"])


def test_plot_mean_aggregation(sales_df):
    result = plot(sales_df, kind="bar", x="region", y="sales", agg="mean")
    west = next(p for p in result["points"] if p["x"] == "West")
    assert west["y"] == pytest.approx(536.6667, rel=1e-3)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kind": "pie", "x": "region", "y": "sales"}, "Unknown chart kind"),
        ({"kind": "bar", "x": "region"}, "needs a 'y' column"),
        ({"kind": "bar", "y": "sales"}, "needs an 'x' column"),
        ({"kind": "bar", "x": "revenue", "y": "sales"}, "not found"),
        ({"kind": "bar", "x": "region", "y": "product"}, "not numeric"),
        ({"kind": "hist", "x": "region"}, "not numeric"),
        ({"kind": "bar", "x": "region", "y": "sales", "agg": "nope"}, "Unknown aggregation"),
    ],
)
def test_plot_rejects_bad_arguments(sales_df, kwargs, match):
    with pytest.raises(ToolError, match=match):
        plot(sales_df, **kwargs)


# ---------------------------------------------------------------------------
# dispatch + summary
# ---------------------------------------------------------------------------
def test_call_tool_dispatches_and_rejects_unknown(sales_df):
    result = call_tool("describe_data", sales_df, {})
    assert result["rows"] == 6

    with pytest.raises(ToolError, match="Unknown tool"):
        call_tool("run_python", sales_df, {})

    with pytest.raises(ToolError, match="does not accept"):
        call_tool("describe_data", sales_df, {"column": "sales"})


def test_call_tool_rejects_unexpected_kwargs(sales_df):
    with pytest.raises(ToolError, match="Accepted arguments"):
        call_tool("aggregate", sales_df, {"group_by": "region", "column": "sales", "limit": 5})


def test_summarise_result_is_human_readable(sales_df):
    text = summarise_result(aggregate(sales_df, group_by="region", column="sales", agg="sum"))
    assert "West" in text and "1610" in text

    plot_text = summarise_result(plot(sales_df, kind="bar", x="region", y="sales"))
    assert "bar chart" in plot_text

    describe_text = summarise_result(describe_data(sales_df))
    assert "6 rows" in describe_text
