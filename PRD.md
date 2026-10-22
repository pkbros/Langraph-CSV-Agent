# CSV Analyst Agent: PRD

A small multi-agent style workflow built with LangGraph and a Streamlit UI. You upload a CSV, ask questions in plain English, and the agent picks the right tool, runs it, retries if it fails, and answers.

## 1. Why this project

Shows the core skills of an AI agent engineer in one small repo:
- LangGraph routing, state, memory, tools and retries
- A Python backend with clean, testable tool functions
- Tests and tracing
- A simple deployment

## 2. Goal

Let a non-technical user ask questions about a CSV file and get a correct answer or chart, with the steps the agent took visible on screen.

## 3. Users

Anyone with a small CSV (sales, marks, expenses) who does not want to write pandas code.

## 4. Scope (v1)

**In scope**
- Upload one CSV (up to 10 MB)
- Chat interface with follow-up questions
- Four tools: `describe_data`, `aggregate`, `filter_rows`, `plot`
- Router that decides which tool to use, or answers directly for general questions
- Retry up to 2 times when a tool fails
- Trace panel showing each node that ran
- Unit tests and one end-to-end test

**Out of scope**
- Multiple files or joins
- Login, user accounts, database
- Running free-form LLM generated code (see decision below)

## 5. Key design decision

The LLM does **not** write and run pandas code. It only picks a tool and fills in JSON arguments (column names, aggregation type). The tools are fixed Python functions. This is safer, easier to test, and easier to retry when the LLM picks a wrong column.

## 6. How it works

### Graph flow

```
START -> router -> planner -> executor -> check
                      ^                     |
                      |---- error, retry ---|
                                            |
                                      responder -> END
router -> responder (for general chat, no tool needed)
```

### Nodes

| Node | Job |
|---|---|
| router | Reads the question and data schema, picks: `tool` or `chat` |
| planner | Chooses the tool and returns JSON arguments |
| executor | Runs the chosen tool function on the dataframe |
| check | If result is an error and retries < 2, sends the error back to planner. Otherwise moves on |
| responder | Writes the final answer in plain English from the tool result |

### State

```python
class AgentState(TypedDict):
    question: str
    schema: str            # column names, types, 5 sample rows
    route: str             # "tool" or "chat"
    tool_name: str
    tool_args: dict
    result: Any            # table, number, or figure
    error: str | None
    retries: int
    answer: str
    trace: list[str]       # one line per node, shown in the UI
```

### Memory

LangGraph `MemorySaver` checkpointer with a `thread_id` per Streamlit session, so follow-ups like "now show that by month" work.

### Tools

| Tool | Example args |
|---|---|
| describe_data | none |
| aggregate | `{"group_by": "region", "column": "sales", "agg": "sum"}` |
| filter_rows | `{"column": "age", "op": ">", "value": 30}` |
| plot | `{"kind": "bar", "x": "region", "y": "sales"}` |

Each tool validates its arguments (column exists, agg is allowed) and returns a clear error message if not. That message is what the planner sees on retry.

## 7. Tech stack

- Python 3.11
- LangGraph and LangChain core
- Streamlit
- pandas and matplotlib
- Groq API (Llama-3 model), key set through `.env`
- pytest
- Optional: LangSmith tracing through environment variables

## 8. Repo structure

```
csv-analyst-agent/
  app.py                 # Streamlit UI
  agent/
    graph.py             # builds the LangGraph
    nodes.py             # router, planner, executor, check, responder
    state.py             # AgentState
    tools.py             # the four tool functions
    llm.py               # LLM client setup
  tests/
    test_tools.py
    test_graph.py
  sample_data/sales.csv
  .env.example
  requirements.txt
  README.md
```

## 9. UI

- Sidebar: upload CSV, preview of first rows
- Main: chat box, answers, tables and charts inline
- Expander under each answer: "Agent steps" showing the trace (router chose tool, planner args, retry 1 failed because column not found, etc.)

## 10. Testing

- `test_tools.py`: each tool with good args, bad column, bad operator
- `test_graph.py`: use a fake LLM that returns fixed JSON to check routing, the retry path, and that retries stop at 2
- One end-to-end test on `sample_data/sales.csv`

## 11. Success criteria

- Answers 5 sample questions correctly on the sample CSV
- Recovers from a wrong column name by retrying
- Never crashes on a bad question, shows a friendly message instead
- All tests pass in GitHub Actions
- Live demo link in the README

## 12. Plan

**Day 1**
- Tools and tests
- State and graph with router, planner, executor, check, responder
- Test the graph in the terminal

**Day 2**
- Streamlit UI with chat, charts and trace panel
- Memory with checkpointer
- GitHub Actions for pytest
- Deploy on Streamlit Community Cloud
- README with a screenshot or short GIF and the architecture diagram above

## 13. Possible v2

- Multiple CSVs
- Export the answer as a report
- Evaluation set of questions with expected answers
