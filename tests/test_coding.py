from __future__ import annotations

import json

import pytest

from app.ai.coding import MAX_CODE_LENGTH, analyze_code, code_analysis_tool, explain_code
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
    assert result["non_empty_lines"] == 6
    assert set(result["concepts"]) == {"imports", "classes", "functions", "conditionals"}


def test_code_explanation_surfaces_syntax_error() -> None:
    result = explain_code("python", "def broken(:\n    pass\n")
    assert result["valid"] is False
    assert "syntax-error" in result["concepts"]
    assert result["issues"]


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
