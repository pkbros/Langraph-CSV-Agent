"""CSV Analyst Agent - Streamlit UI.

Upload a CSV, ask questions in plain English, see the answer, table/chart and
the agent's steps. Run with:  streamlit run app.py
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path

import pandas as pd
import streamlit as st

from agent.charts import render_png
from agent.config import get_settings
from agent.data import LoadReport, read_csv_bytes
from agent.graph import build_graph, run_question
from agent.state import STORE
from agent.tools import build_schema

SAMPLE_PATH = Path(__file__).parent / "sample_data" / "sales.csv"


_EXCLUDED_METRICS = re.compile(r"^(id|index|idx|row|rowid|no|number|unnamed.*)$", re.IGNORECASE)
_PREFERRED_CATEGORIES = (
    "region", "category", "type", "group", "product", "segment", "month",
    "class", "department", "country", "city", "status", "name",
)


def _pick_metric(df: pd.DataFrame, numeric: list[str]) -> str:
    """Prefer a measured float column with many distinct values over an id."""
    usable = [column for column in numeric if not _EXCLUDED_METRICS.match(str(column).strip())]
    pool = usable or numeric

    def score(column: str) -> tuple[bool, int, float]:
        values = pd.to_numeric(df[column], errors="coerce")
        return (
            bool(pd.api.types.is_float_dtype(df[column])),
            int(values.nunique(dropna=True)),
            abs(float(values.sum())),
        )

    return max(pool, key=score)


def _pick_category(df: pd.DataFrame, categorical: list[str]) -> str | None:
    if not categorical:
        return None

    def rank(column: str) -> tuple[int, int, str]:
        name = str(column).lower()
        position = next(
            (i for i, preferred in enumerate(_PREFERRED_CATEGORIES) if preferred in name),
            len(_PREFERRED_CATEGORIES),
        )
        return (position, int(df[column].nunique(dropna=True)), str(column))

    return min(categorical, key=rank)


def suggestions_for(df: pd.DataFrame) -> list[str]:
    """Question chips that fit the uploaded columns instead of a fixed dataset."""
    numeric = [column for column in df.columns if pd.api.types.is_numeric_dtype(df[column])]
    categorical = [
        column
        for column in df.columns
        if column not in numeric and 1 < df[column].nunique(dropna=True) <= 50
    ]

    items = ["Summarise this dataset", "How many rows and columns are there?"]
    if numeric:
        number = _pick_metric(df, numeric)
        category = _pick_category(df, categorical)
        if category:
            items.append(f"Average {number} by {category}")
            items.append(f"Show a bar chart of average {number} by {category}")
        else:
            items.append(f"Show a histogram of {number}")
            median = float(pd.to_numeric(df[number], errors="coerce").median())
            pretty = f"{median:,.0f}" if median == int(median) else f"{median:,.2f}"
            items.append(f"Show the rows where {number} is above {pretty}")
    return items[:4]


def apply_streamlit_secrets() -> None:
    """On Streamlit Community Cloud, keys arrive via st.secrets, not the env."""
    try:
        secrets = st.secrets
        for key in ("GROQ_API_KEY", "GROQ_BASE_URL", "GROQ_MODEL"):
            if key in secrets and not os.getenv(key):
                os.environ[key] = str(secrets[key])
    except Exception:  # noqa: BLE001 - no secrets.toml locally
        return


apply_streamlit_secrets()
st.set_page_config(page_title="CSV Analyst Agent", page_icon="📊", layout="wide")


# ---------------------------------------------------------------------------
# cached resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_graph():
    settings = get_settings()
    return build_graph(settings=settings), settings


@st.cache_data(show_spinner=False)
def read_csv(raw: bytes) -> tuple[pd.DataFrame, LoadReport]:
    return read_csv_bytes(raw)


# ---------------------------------------------------------------------------
# session helpers
# ---------------------------------------------------------------------------
def new_thread() -> None:
    st.session_state.thread_id = uuid.uuid4().hex
    st.session_state.messages = []


def load_dataframe(df: pd.DataFrame, label: str) -> None:
    thread_id = st.session_state.thread_id
    STORE.put(thread_id, df, build_schema(df))
    st.session_state.df = df
    st.session_state.data_label = label
    st.session_state.schema = STORE.schema(thread_id)


def start_dataset(df: pd.DataFrame, label: str, signature: str, note: str = "") -> None:
    """Fresh thread + empty chat: memory must not leak across files."""
    new_thread()
    load_dataframe(df, label)
    st.session_state.signature = signature
    st.session_state.load_note = note


def reset_conversation() -> None:
    new_thread()
    df = st.session_state.get("df")
    if df is not None:
        load_dataframe(df, st.session_state.get("data_label", "data.csv"))


if "thread_id" not in st.session_state:
    new_thread()
if "df" not in st.session_state:
    st.session_state.df = None


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------
graph, settings = get_graph()

with st.sidebar:
    st.header("Data")

    uploaded = st.file_uploader("Upload a CSV (max 10 MB)", type=["csv"])
    use_sample = st.button("Use the sample sales file", width="stretch")

    if uploaded is not None:
        raw = uploaded.getvalue()
        signature = hashlib.md5(raw).hexdigest()
        if len(raw) > 10 * 1024 * 1024:
            st.error("That file is larger than 10 MB.")
        elif (
            st.session_state.get("signature") != signature
            and st.session_state.get("failed_signature") != signature
        ):
            try:
                frame, report = read_csv(raw)
            except Exception as exc:  # noqa: BLE001
                st.session_state.failed_signature = signature
                st.error(f"Could not read that CSV: {exc}")
                if st.session_state.get("df") is not None:
                    st.caption("The previously loaded data is still active.")
            else:
                st.session_state.failed_signature = None
                start_dataset(frame, uploaded.name, signature, report.describe())
                st.success(f"Loaded {uploaded.name}")

    if use_sample and SAMPLE_PATH.exists():
        raw = SAMPLE_PATH.read_bytes()
        frame, report = read_csv(raw)
        start_dataset(
            frame, "sample_data/sales.csv", hashlib.md5(raw).hexdigest(), report.describe()
        )

    df = st.session_state.get("df")
    if df is not None:
        st.caption(
            st.session_state.get("load_note")
            or f"{len(df):,} rows × {len(df.columns)} columns"
        )
        st.dataframe(df.head(10), width="stretch", hide_index=True)
        with st.expander("Column schema"):
            st.code(st.session_state.get("schema", ""), language="text")

        if st.button("Clear conversation", width="stretch"):
            reset_conversation()
            st.rerun()
    else:
        st.info("Upload a CSV or load the sample to begin.")

    st.divider()
    st.caption("Model")
    if settings.configured:
        st.success(f"{settings.model}\n\nvia Groq", icon="✅")
    else:
        st.warning(
            "No `GROQ_API_KEY` found.\n\nRunning the offline heuristic model — "
            "answers still work, wording is canned.",
            icon="⚠️",
        )

    with st.expander("How it works"):
        st.markdown(
            "`router` → `planner` → `executor` → `check` → `responder`\n\n"
            "If a tool fails, the error goes back to the planner, up to "
            f"{settings.max_retries} retries. No pandas code is generated by the model."
        )


# ---------------------------------------------------------------------------
# main area
# ---------------------------------------------------------------------------
st.title("📊 CSV Analyst Agent")
st.caption("Ask plain-English questions about your CSV. Every step the agent takes is shown.")


def render_result(result: dict | None, key: str) -> None:
    if not result:
        return
    kind = result.get("type")

    if kind in ("table", "filter"):
        frame = pd.DataFrame(result.get("rows") or [])
        if frame.empty:
            st.info("No rows matched.")
            return
        st.dataframe(frame, width="stretch", hide_index=True)
        extra = []
        if kind == "filter":
            extra.append(f"{result.get('matched', 0):,} of {result.get('total', 0):,} rows")
        if result.get("truncated"):
            extra.append(f"{result['truncated']:,} more not shown")
        if extra:
            st.caption(" · ".join(extra))

    elif kind == "describe":
        st.dataframe(pd.DataFrame(result.get("sample") or []), width="stretch", hide_index=True)
        summary = pd.DataFrame(result.get("columns") or [])
        if not summary.empty:
            with st.expander("Column report"):
                st.dataframe(summary, width="stretch", hide_index=True)

    elif kind == "plot":
        points = result.get("points") or []
        frame = pd.DataFrame(points)
        if frame.empty:
            return
        chart_kind = result.get("kind")
        if chart_kind == "line":
            st.line_chart(frame, x="x", y="y")
        elif chart_kind == "scatter":
            st.scatter_chart(frame, x="x", y="y")
        else:
            st.bar_chart(frame, x="x", y="y")
        st.caption(f"{result.get('kind', 'chart').title()} · {result.get('y_label')} by {result.get('x_label')}")
        png = render_png(result)
        if png:
            st.download_button(
                "Download PNG",
                data=png,
                file_name=f"{result.get('kind', 'chart')}_{result.get('x', 'chart')}.png",
                mime="image/png",
                key=f"png-{key}",
            )


for index, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_result(message.get("result"), key=str(index))
            trace = message.get("trace") or []
            if trace:
                with st.expander("Agent steps"):
                    st.code("\n".join(trace), language="text")


df = st.session_state.get("df")
prompt = st.chat_input("Ask about your data…")

if not st.session_state.messages and df is not None:
    st.markdown("**Try one of these**")
    cols = st.columns(2)
    for idx, suggestion in enumerate(suggestions_for(df)):
        if cols[idx % 2].button(suggestion, key=f"suggest-{idx}", width="stretch"):
            st.session_state.pending = suggestion
            st.rerun()

if prompt is None:
    prompt = st.session_state.pop("pending", None)

if prompt:
    if df is None:
        st.warning("Load a CSV in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Working through the graph…"):
                try:
                    state = run_question(
                        prompt,
                        st.session_state.thread_id,
                        graph=graph,
                        settings=settings,
                    )
                    answer = state.get("answer") or "I could not produce an answer."
                    result = state.get("result")
                    trace = state.get("trace") or []
                except Exception as exc:  # noqa: BLE001 - UI must not crash
                    answer = (
                        "Something went wrong while answering that. "
                        f"`{type(exc).__name__}: {exc}`"
                    )
                    result, trace = None, []
            st.markdown(answer)
            render_result(result, key=f"live-{len(st.session_state.messages)}")
            if trace:
                with st.expander("Agent steps", expanded=True):
                    st.code("\n".join(trace), language="text")

        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "result": result, "trace": trace}
        )
