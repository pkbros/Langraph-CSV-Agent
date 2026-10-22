"""Terminal entry point: python -m agent.cli sample_data/sales.csv "your question"

Useful for a quick smoke test before opening the Streamlit UI.
Pass --demo to force the offline heuristic model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .config import get_settings
from .data import CSVLoadError, read_csv_path
from .fake_llm import FakeLLM
from .graph import build_graph, run_question


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    parser = argparse.ArgumentParser(description="Ask a CSV a question from the terminal.")
    parser.add_argument("csv", type=Path, help="Path to a .csv file")
    parser.add_argument("question", nargs="*", help="Question (omit for interactive mode)")
    parser.add_argument("--demo", action="store_true", help="Use the offline heuristic model")
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"File not found: {args.csv}", file=sys.stderr)
        return 2

    try:
        df, report = read_csv_path(str(args.csv))
    except CSVLoadError as exc:
        print(f"Could not read {args.csv}: {exc}", file=sys.stderr)
        return 2

    settings = get_settings()
    llm = FakeLLM() if args.demo else None
    graph = build_graph(llm=llm, settings=settings)
    thread_id = "cli"

    print(f"Loaded {args.csv} - {report.describe()}")
    print(f"Model: {'offline heuristic (--demo)' if args.demo else settings.model}")
    if not args.demo and not settings.configured:
        print("GROQ_API_KEY is empty - falling back to the offline heuristic model.")
    print()

    if args.question:
        _ask(graph, thread_id, df, " ".join(args.question))
        return 0

    while True:
        try:
            question = input("ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question.lower() in ("exit", "quit", ":q"):
            return 0
        _ask(graph, thread_id, df, question, register=False)


def _ask(graph, thread_id: str, df, question: str, register: bool = True) -> None:
    state = run_question(question, thread_id, df=df if register else None, graph=graph)
    print("\n".join("  " + line for line in state.get("trace", [])))
    print(f"\n{state.get('answer', '(no answer)')}\n")
    result = state.get("result") or {}
    if result.get("type") in ("table", "filter"):
        print(pd.DataFrame(result.get("rows") or []).head(10).to_string(index=False))
        print()
    elif result.get("type") == "describe":
        print(pd.DataFrame(result.get("sample") or []).head(10).to_string(index=False))
        print()


if __name__ == "__main__":
    raise SystemExit(main())
