from __future__ import annotations

import io
import json

from openpyxl import Workbook

from app.capabilities.data_analysis import analyze_payload


def test_csv_analysis_reports_quality_and_stats() -> None:
    content = b"name,score\nA,10\nB,20\nB,20\n"
    result = analyze_payload("scores.csv", content)

    assert result["rows"] == 3
    assert result["duplicate_rows"] == 1
    assert result["summary"]["score"]["mean"] == 50 / 3
    assert result["summary"]["score"]["median"] == 20.0
    assert result["charts"][0]["type"] == "bar"


def test_json_analysis_accepts_list_of_objects() -> None:
    content = json.dumps([
        {"city": "Bengaluru", "value": 10},
        {"city": "Mysuru", "value": 30},
    ]).encode()
    result = analyze_payload("data.json", content)

    assert result["rows"] == 2
    assert result["summary"]["value"]["min"] == 10.0
    assert result["summary"]["value"]["max"] == 30.0


def test_xlsx_analysis_reads_first_row_as_headers() -> None:
    buffer = io.BytesIO()
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sales"
    sheet.append(["item", "amount"])
    sheet.append(["pen", 10])
    sheet.append(["book", 30])
    workbook.save(buffer)

    result = analyze_payload("sales.xlsx", buffer.getvalue())

    assert result["rows"] == 2
    assert "item" in result["columns"]
    assert "amount" in result["columns"]
    assert result["summary"]["amount"]["mean"] == 20.0
    assert result["sample"][0]["_sheet"] == "Sales"
