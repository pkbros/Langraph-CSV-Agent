"""CSV loader tests: delimiter sniffing, preambles, encodings, bad rows."""

from __future__ import annotations

import pandas as pd
import pytest

from agent.data import CSVLoadError, read_csv_bytes


def _csv(text: str, encoding: str = "utf-8") -> bytes:
    return text.strip("\n").encode(encoding)


def test_plain_comma_file():
    frame, report = read_csv_bytes(_csv("""
name,score
ana,10
ben,20
"""))
    assert list(frame.columns) == ["name", "score"]
    assert len(frame) == 2
    assert frame["score"].tolist() == [10, 20]
    assert report.delimiter == ","
    assert report.skipped_preamble == 0
    assert report.skipped_bad_rows == 0


def test_semicolon_file_with_decimal_commas():
    frame, report = read_csv_bytes(_csv("""
region;sales;units
West;400,50;4
East;260,25;2
"""))
    assert list(frame.columns) == ["region", "sales", "units"]
    assert len(frame) == 2
    assert report.delimiter == ";"
    assert frame.loc[0, "sales"] == "400,50"


def test_tab_separated_file():
    frame, report = read_csv_bytes(_csv("""
a\tb\tc
1\t2\t3
"""))
    assert report.delimiter == "\t"
    assert frame.shape == (1, 3)


def test_preamble_lines_are_skipped():
    """Reproduces: Expected 2 fields in line 3, saw 12."""
    header = ",".join(f"col{i}" for i in range(1, 13))
    row = ",".join(str(i) for i in range(1, 13))
    frame, report = read_csv_bytes(
        _csv(f"""
Report,Generated
Client,ACME
{header}
{row}
{row}
""")
    )
    assert frame.shape == (2, 12)
    assert report.skipped_preamble == 2
    assert list(frame.columns)[:3] == ["col1", "col2", "col3"]


def test_excel_sep_directive():
    frame, report = read_csv_bytes(_csv("""
sep=;
name;value
a;1
"""))
    assert report.delimiter == ";"
    assert list(frame.columns) == ["name", "value"]
    assert len(frame) == 1


def test_malformed_row_is_skipped_and_counted():
    frame, report = read_csv_bytes(_csv("""
name,score
ana,10
this row has,too,many,fields
ben,20
"""))
    assert list(frame.columns) == ["name", "score"]
    assert len(frame) == 2
    assert report.skipped_bad_rows == 1
    assert "malformed row" in report.describe()


def test_quoted_multiline_value_is_not_skipped():
    """A quoted newline is one record, not two malformed rows."""
    frame, report = read_csv_bytes(_csv('''
name,note
ana,"line one
line two"
ben,"ok"
'''))
    assert frame.shape == (2, 2)
    assert report.skipped_bad_rows == 0
    assert "line two" in frame.loc[0, "note"]


def test_bom_and_cp1252_encoding():
    frame, report = read_csv_bytes(_csv("""
name,city
José,São Paulo
""", encoding="cp1252"))
    assert "José" in frame["name"].iloc[0]
    assert report.encoding in ("cp1252", "latin-1")


def test_quoted_commas_do_not_split_columns():
    frame, report = read_csv_bytes(_csv('''
name,note
ana,"hello, world"
ben,"a;b"
'''))
    assert frame.shape == (2, 2)
    assert report.delimiter == ","
    assert frame.loc[0, "note"] == "hello, world"


def test_blank_lines_and_empty_columns_are_dropped():
    frame, report = read_csv_bytes(_csv("""
a,b,
1,2,
3,4,
"""))
    assert list(frame.columns) == ["a", "b"]
    assert len(frame) == 2


def test_empty_file_raises():
    with pytest.raises(CSVLoadError, match="empty"):
        read_csv_bytes(b"")


def test_header_only_file_raises():
    with pytest.raises(CSVLoadError, match="no data rows"):
        read_csv_bytes(_csv("a,b,c"))


def test_single_column_file_still_loads():
    frame, report = read_csv_bytes(_csv("""
name
ana
ben
"""))
    assert frame.shape == (2, 1)


def test_sample_file_loads():
    path = __import__("pathlib").Path("sample_data/sales.csv")
    frame, report = read_csv_bytes(path.read_bytes())
    assert frame.shape == (96, 6)
    assert report.delimiter == ","
    assert report.rows == 96


def test_numeric_types_are_inferred():
    frame, _ = read_csv_bytes(_csv("""
a,b,c
1,2.5,hello
3,4.5,world
"""))
    assert pd.api.types.is_integer_dtype(frame["a"])
    assert pd.api.types.is_float_dtype(frame["b"])
