"""Turn a plot tool result into a downloadable PNG (matplotlib, no display)."""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")  # must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402


def render_png(result: dict[str, Any], width: float = 8.0, height: float = 4.2) -> bytes | None:
    """Render a ``plot`` tool result. Returns PNG bytes, or None for non-plots."""
    if not result or result.get("type") != "plot":
        return None

    points = result.get("points") or []
    if not points:
        return None

    xs = [p["x"] for p in points]
    ys = [p["y"] for p in points]
    kind = result.get("kind", "bar")

    fig, ax = plt.subplots(figsize=(width, height), dpi=130)
    if kind == "line":
        ax.plot(range(len(xs)), ys, marker="o", linewidth=1.8, color="#2f6feb")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=45, ha="right")
    elif kind == "scatter":
        ax.scatter(xs, ys, alpha=0.75, color="#2f6feb", edgecolors="white", linewidths=0.5)
    else:  # bar or hist
        ax.bar(range(len(xs)), ys, color="#2f6feb", width=0.75)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels(xs, rotation=45 if len(xs) > 8 else 0, ha="right" if len(xs) > 8 else "center")

    ax.set_xlabel(result.get("x_label") or result.get("x") or "")
    ax.set_ylabel(result.get("y_label") or result.get("y") or "")
    ax.set_title(f"{kind.title()} chart")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()
