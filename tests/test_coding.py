from __future__ import annotations

import json

import pytest

from app.ai.coding import MAX_CODE_LENGTH, analyze_code, code_analysis_tool, explain_code, suggest_fixes, transform_code
from app.ai.tools import run_tool


def test_python_code_analysis_accepts_valid_code() -> None:
    result = analyze_code("python", "def add(a, b):\n    return a + b\n")
    assert result["valid"] is True
    assert result["language"] == "python"
    assert result["issues"] == []


def test_python_code_analysis_reports_syntax_location() -> None:
    result = analyze_code("python", "def add(a, b)\n    return a + b\n")
    assert result["valid"] is False
    assert result["issues"][0]["severity"] == "error"
    assert result["issues"][0]["line"] == 1


def test_generic_code_analysis_reports_unbalanced_delimiters() -> None:
    result = analyze_code("javascript", "function add(a, b) { return a + b;\n")
    assert result["valid"] is False
    assert "unclosed '{'" in result["issues"][0]["message"]


def test_python_code_explanation_detects_structure_without_execution() -> None:
    result = explain_code("python", "import math\n\nclass Calc:\n    def add(self, a, b):\n        if a > 0:\n            return a + b\n")
    assert result["valid"] is True
    assert result["non_empty_lines"] == 5
    assert set(result["concepts"]) == {"imports", "classes", "functions", "conditionals"}


def test_code_explanation_surfaces_syntax_error() -> None:
    result = explain_code("python", "def broken(:\n    pass\n")
    assert result["valid"] is False
    assert "syntax-error" in result["concepts"]
    assert result["issues"]


def test_fix_suggestions_are_non_executing() -> None:
    result = suggest_fixes("javascript", "function add(a, b) { return a + b;\n")
    assert result["valid"] is False
    assert result["issue_count"] == 1
    assert result["suggestions"][0]["line"] == 1
    assert result["safe"] is True
    assert result["execution"] == "not_performed"


def test_code_transformation_normalizes_text_without_execution() -> None:
    result = transform_code("python", "x = 1  \r\nprint(x)", "normalize")
    assert result["changed"] is True
    assert result["code"] == "x = 1\nprint(x)\n"
    assert result["safe"] is True
    assert result["execution"] == "not_performed"


def test_code_transform_tool_is_json_and_safe() -> None:
    payload = json.dumps({"language": "python", "code": "x = 1  ", "operation": "strip_trailing_whitespace"})
    result = run_tool("code_transform", payload)
    assert result.safe is True
    body = json.loads(result.output)
    assert body["code"] == "x = 1"


def test_code_transform_rejects_unknown_operation() -> None:
    with pytest.raises(ValueError, match="unsupported transform"):
        transform_code("python", "x = 1", "execute")


def test_code_analysis_tool_is_json_and_safe() -> None:
    payload = json.dumps({"language": "python", "code": "print('hello')"})
    result = run_tool("code_analysis", payload)
    assert result.safe is True
    body = json.loads(result.output)
    assert body["valid"] is True


def test_code_analysis_rejects_unsupported_language() -> None:
    with pytest.raises(ValueError, match="unsupported language"):
        analyze_code("brainfuck", "++")


def test_code_analysis_rejects_oversized_source() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        analyze_code("python", "x" * (MAX_CODE_LENGTH + 1))
