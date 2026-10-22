"""Robust CSV loading for real-world files.

Handles the things that break ``pd.read_csv`` in practice:

* preamble lines before the real header (report titles, metadata)
* semicolon / tab / pipe separated files, including European decimal commas
* ``sep=;`` directives written by Excel
* non-UTF-8 encodings (cp1252, latin-1)
* values quoted across multiple physical lines
* a handful of genuinely malformed rows (skipped, and reported)

Everything is pure Python plus pandas, so it is easy to unit test.
"""

from __future__ import annotations

import csv
import io
import re
from collections import Counter
from dataclasses import dataclass

import pandas as pd

CANDIDATE_DELIMITERS: tuple[str, ...] = (",", ";", "\t", "|")
DELIMITER_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}
ENCODINGS: tuple[str, ...] = ("utf-8-sig", "cp1252", "latin-1")

_SEP_DIRECTIVE = re.compile(r"^\s*sep=(.)\s*$", re.IGNORECASE)
_DECIMAL_COMMA = re.compile(r"^-?\d+,\d+$")


class CSVLoadError(ValueError):
    """The upload cannot be interpreted as a table."""


@dataclass(frozen=True)
class LoadReport:
    delimiter: str
    encoding: str
    skipped_preamble: int
    skipped_bad_rows: int
    rows: int
    columns: int

    def describe(self) -> str:
        parts = [f"{self.rows:,} rows × {self.columns} columns"]
        if self.delimiter != ",":
            parts.append(f"{DELIMITER_NAMES.get(self.delimiter, repr(self.delimiter))}-separated")
        if self.encoding not in ("utf-8-sig", "utf-8"):
            parts.append(f"encoding {self.encoding}")
        if self.skipped_preamble:
            parts.append(f"skipped {self.skipped_preamble} preamble line(s)")
        if self.skipped_bad_rows:
            parts.append(f"skipped {self.skipped_bad_rows} malformed row(s)")
        return " · ".join(parts)


def _decode(raw: bytes) -> tuple[str, str]:
    if not raw.strip():
        raise CSVLoadError("That file is empty.")
    for encoding in ENCODINGS:
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise CSVLoadError("Could not decode the file as text. Save it as UTF-8 and try again.")


def _count_fields(line: str, delimiter: str) -> int:
    try:
        return len(next(csv.reader([line], delimiter=delimiter)))
    except (csv.Error, StopIteration):
        return 0


def _detect_delimiter(lines: list[str]) -> str:
    """Pick the delimiter that yields the most (consistent) columns."""
    sample = lines[:2000]
    best_delimiter, best_score = ",", -1.0
    for delimiter in CANDIDATE_DELIMITERS:
        counts = [_count_fields(line, delimiter) for line in sample]
        field_counts = Counter(n for n in counts if n > 1)
        if not field_counts:
            continue
        modal, frequency = field_counts.most_common(1)[0]
        score = modal * 100 + min(frequency, 99)
        # "1,5" style decimals make a semicolon file look comma-separated
        if delimiter == ",":
            decimals = 0
            for line in sample:
                try:
                    cells = next(csv.reader([line], delimiter=","))
                except (csv.Error, StopIteration):
                    continue
                decimals += sum(1 for cell in cells if _DECIMAL_COMMA.match(cell.strip()))
            score -= min(decimals, 99) * 20
        if score > best_score:
            best_delimiter, best_score = delimiter, score
    return best_delimiter


def _parse_records(text: str, delimiter: str) -> list[list[str]]:
    """Parse CSV records, honouring quoted fields that span physical lines."""
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter, skipinitialspace=True)
        return [record for record in reader if any(cell.strip() for cell in record)]
    except csv.Error as exc:
        raise CSVLoadError(f"Could not parse the file: {exc}") from exc


def _find_table_start(records: list[list[str]]) -> tuple[int, int]:
    """Return (header index, expected field count), skipping preamble records."""
    counts = [len(record) for record in records]
    multi = [count for count in counts if count > 1]
    if not multi:
        return 0, (counts[0] if counts else 1)

    tail = [count for count in counts[-50:] if count > 1]
    target = Counter(tail).most_common(1)[0][0] if tail else Counter(multi).most_common(1)[0][0]

    for index, count in enumerate(counts):
        if count == target:
            return index, target
    return 0, counts[0]


def _rebuild_csv(header: list[str], rows: list[list[str]], delimiter: str) -> io.StringIO:
    """Re-serialise clean records so pandas can infer types safely."""
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=delimiter, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    buffer.seek(0)
    return buffer


def read_csv_bytes(raw: bytes) -> tuple[pd.DataFrame, LoadReport]:
    """Parse uploaded bytes into a dataframe plus a report of what was skipped."""
    text, encoding = _decode(raw)

    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise CSVLoadError("That file has no readable lines.")

    preamble_extra = 0
    forced_delimiter: str | None = None
    directive = _SEP_DIRECTIVE.match(lines[0])
    if directive:
        forced_delimiter = directive.group(1)
        lines = lines[1:]
        preamble_extra = 1

    if not lines:
        raise CSVLoadError("That file only contains a separator directive.")

    delimiter = forced_delimiter or _detect_delimiter(lines)
    records = _parse_records("\n".join(lines), delimiter)
    if not records:
        raise CSVLoadError("That file has no readable rows.")

    header_index, _ = _find_table_start(records)
    header = [cell.strip() for cell in records[header_index]]
    data = records[header_index + 1 :]
    if not data:
        raise CSVLoadError(
            "Found a header but no data rows. Check that the file has one header row "
            "followed by data."
        )

    good_rows = [row for row in data if len(row) == len(header)]
    bad_rows = len(data) - len(good_rows)
    skipped_preamble = header_index + preamble_extra

    preview = "\n".join(delimiter.join(record) for record in records[:3])[:300]
    if not good_rows:
        raise CSVLoadError(
            f"Found {len(data)} data row(s), but none matched the {len(header)}-column "
            "header. Check the delimiter or remove the extra summary rows.\n\n"
            f"First lines:\n{preview}"
        )

    buffer = _rebuild_csv(header, good_rows, delimiter)
    try:
        frame = pd.read_csv(
            buffer,
            sep=delimiter,
            engine="python",
            skipinitialspace=True,
        )
    except pd.errors.EmptyDataError as exc:
        raise CSVLoadError("No tabular data found after the header row.") from exc
    except (pd.errors.ParserError, csv.Error) as exc:
        raise CSVLoadError(
            f"Could not parse the table ({exc}). First lines:\n{preview}"
        ) from exc

    frame.columns = [str(column).strip() for column in frame.columns]
    frame = frame.dropna(axis=1, how="all")
    frame = frame.dropna(axis=0, how="all")
    frame = frame.reset_index(drop=True)

    if frame.shape[1] == 0:
        raise CSVLoadError(
            "Could not find any columns. First lines:\n"
            f"{preview}\n\n"
            "Supported delimiters: comma, semicolon, tab, pipe."
        )

    report = LoadReport(
        delimiter=delimiter,
        encoding=encoding,
        skipped_preamble=skipped_preamble,
        skipped_bad_rows=bad_rows,
        rows=len(frame),
        columns=frame.shape[1],
    )
    return frame, report


def read_csv_path(path: str) -> tuple[pd.DataFrame, LoadReport]:
    with open(path, "rb") as handle:
        return read_csv_bytes(handle.read())
