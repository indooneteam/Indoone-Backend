from __future__ import annotations

import csv
import io
import json
import math
from statistics import mean
from typing import Any


MAX_ANALYSIS_BYTES = 2_000_000
MAX_SAMPLE_ROWS = 10


def _numbers(values: list[Any]) -> list[float]:
    numbers: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            numbers.append(number)
    return numbers


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    columns = sorted({str(key) for row in records for key in row})
    summary: dict[str, Any] = {}
    for column in columns:
        values = [row.get(column) for row in records]
        numbers = _numbers(values)
        non_empty = [value for value in values if value not in (None, "")]
        item: dict[str, Any] = {
            "non_empty": len(non_empty),
            "missing": len(values) - len(non_empty),
        }
        if numbers:
            item.update({
                "numeric": True,
                "count": len(numbers),
                "min": min(numbers),
                "max": max(numbers),
                "mean": mean(numbers),
            })
        else:
            item["numeric"] = False
            item["unique_values"] = len({str(value) for value in non_empty})
        summary[column] = item
    return summary


def analyze_payload(filename: str, content: bytes) -> dict[str, Any]:
    if len(content) > MAX_ANALYSIS_BYTES:
        raise ValueError("analysis input is too large")
    name = filename.casefold()
    text = content.decode("utf-8-sig")

    if name.endswith(".csv"):
        reader = csv.DictReader(io.StringIO(text))
        records = [dict(row) for row in reader]
    elif name.endswith(".json") or name.endswith(".jsonl"):
        if name.endswith(".jsonl"):
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            payload = json.loads(text)
            records = payload if isinstance(payload, list) else [payload]
        if not all(isinstance(row, dict) for row in records):
            raise ValueError("JSON analysis expects an object or a list of objects")
        records = [dict(row) for row in records]
    else:
        raise ValueError("supported analysis formats are CSV, JSON, and JSONL")

    return {
        "filename": filename,
        "rows": len(records),
        "columns": sorted({str(key) for row in records for key in row}),
        "summary": _summarize_records(records),
        "sample": records[:MAX_SAMPLE_ROWS],
    }
