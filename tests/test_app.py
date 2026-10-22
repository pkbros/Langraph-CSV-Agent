"""Headless UI smoke test using streamlit.testing.v1.AppTest (no browser)."""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture
def app(monkeypatch):
    # Force the deterministic offline model so the test never hits the network.
    monkeypatch.setenv("GROQ_API_KEY", "")
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    return at


def test_app_boots(app):
    assert not app.exception, app.exception
    assert app.title[0].value.startswith("📊")
    assert any("Upload a CSV" in str(item.value) for item in app.info)


def test_sample_then_question(app):
    app.sidebar.button[0].click().run()  # "Use the sample sales file"
    assert not app.exception, app.exception

    app.chat_input[0].set_value("Which region had the highest total sales?").run()
    assert not app.exception, app.exception

    rendered = "\n".join(str(m.value) for m in app.markdown)
    assert "West" in rendered
    # the trace panel is rendered as a code block inside an expander
    traces = [str(block.value) for block in app.code]
    assert any("router →" in trace and "executor →" in trace for trace in traces)


def test_bad_question_does_not_crash(app):
    app.sidebar.button[0].click().run()
    app.chat_input[0].set_value("tell me about the revenue column").run()
    assert not app.exception, app.exception
    rendered = "\n".join(str(m.value) for m in app.markdown)
    assert "couldn't answer" in rendered.lower() or "Based on the data" in rendered


PREAMBLE_CSV = (
    b"Report,Generated\n"
    b"Client,ACME\n"
    b"id,name,region,rep,units,sales,date,c1,c2,c3,c4,c5\n"
    b"1,Widget,West,Ana,4,400,2024-01-03,x,y,z,w,v\n"
    b"2,Gadget,East,Ben,2,260,2024-01-05,x,y,z,w,v\n"
    b"3,Gizmo,West,Cara,7,910,2024-02-11,x,y,z,w,v\n"
)


def test_upload_with_preamble_and_twelve_columns(app):
    app.file_uploader[0].upload("REC_0019_2744.csv", PREAMBLE_CSV, "text/csv").run()
    assert not app.exception, app.exception

    assert any("Loaded REC_0019_2744.csv" in str(item.value) for item in app.success)
    captions = " ".join(str(item.value) for item in app.caption)
    assert "preamble" in captions

    app.chat_input[0].set_value("How many rows and columns are there?").run()
    assert not app.exception, app.exception
    rendered = "\n".join(str(m.value) for m in app.markdown)
    assert "3 rows" in rendered and "12 columns" in rendered


def test_broken_upload_keeps_previous_data(app):
    app.sidebar.button[0].click().run()
    app.file_uploader[0].upload("broken.csv", b"a,b,c\n", "text/csv").run()
    assert not app.exception, app.exception

    assert any("Could not read that CSV" in str(item.value) for item in app.error)
    captions = " ".join(str(item.value) for item in app.caption)
    assert "previously loaded data is still active" in captions
    assert "96 rows" in captions


def test_generic_summary_suggestion_works(app):
    app.sidebar.button[0].click().run()
    buttons = [button for button in app.button if button.label == "Summarise this dataset"]
    assert buttons, [button.label for button in app.button]

    buttons[0].click().run()
    assert not app.exception, app.exception

    rendered = "\n".join(str(m.value) for m in app.markdown)
    assert "96 rows" in rendered and "6 columns" in rendered


def test_suggestions_follow_uploaded_columns(app):
    app.file_uploader[0].upload(
        "REC_0019_2744.csv",
        b"Report,Generated\nClient,ACME\n"
        b"conf,edge,active,a_ms,drop\n"
        b"0.826,0,1,50,0\n"
        b"0.921,0,1,100,0\n"
        b"0.949,0,1,150,0\n",
        "text/csv",
    ).run()
    assert not app.exception, app.exception

    labels = [button.label for button in app.button]
    assert "Summarise this dataset" in labels
    assert "Show a histogram of conf" in labels
    suggestions = [label for label in labels if label not in ("Use the sample sales file", "Clear conversation")]
    assert not any("sales" in label.lower() for label in suggestions)
