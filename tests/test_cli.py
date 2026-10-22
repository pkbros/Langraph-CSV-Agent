"""Terminal entry point, in offline demo mode."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.cli import main

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = ROOT / "sample_data" / "sales.csv"


def test_cli_answers_one_question(capsys):
    code = main([str(SAMPLE), "Which region had the highest total sales?", "--demo"])
    out = capsys.readouterr().out

    assert code == 0
    assert "96 rows" in out
    assert "router → tool" in out
    assert "planner → aggregate" in out
    assert "West" in out


def test_cli_missing_file(capsys):
    code = main([str(ROOT / "nope.csv"), "anything", "--demo"])
    assert code == 2
    assert "File not found" in capsys.readouterr().err


def test_cli_chart_question(capsys):
    code = main([str(SAMPLE), "Show a bar chart of sales by region", "--demo"])
    out = capsys.readouterr().out
    assert code == 0
    assert "planner → plot" in out


def test_cli_describe_question(capsys):
    code = main([str(SAMPLE), "How many rows and columns are there?", "--demo"])
    out = capsys.readouterr().out
    assert code == 0
    assert "planner → describe_data" in out
    assert "96" in out


def test_cli_without_question_is_interactive(capsys, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "quit")
    code = main([str(SAMPLE), "--demo"])
    assert code == 0
