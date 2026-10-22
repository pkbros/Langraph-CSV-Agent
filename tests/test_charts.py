"""Chart rendering (PNG export) tests."""

from __future__ import annotations

from agent.charts import render_png
from agent.tools import plot


def test_png_for_each_chart_kind(sales_df):
    for kind, kwargs in (
        ("bar", {"x": "region", "y": "sales"}),
        ("line", {"x": "date", "y": "sales"}),
        ("scatter", {"x": "units", "y": "sales"}),
        ("hist", {"x": "sales"}),
    ):
        png = render_png(plot(sales_df, kind=kind, **kwargs))
        assert png is not None
        assert png.startswith(b"\x89PNG")
        assert len(png) > 1000


def test_png_is_none_for_non_plots():
    assert render_png({"type": "table", "rows": []}) is None
    assert render_png(None) is None
    assert render_png({"type": "plot", "points": []}) is None
