from __future__ import annotations

import json

from app.ai.coding import suggest_fixes
from app.ai.tools import run_tool


def test_python_fix_suggestions_include_parser_location() -> None:
    result = suggest_fixes("python", "def broken(:\n    pass\n")
    assert result["valid"] is False
    assert result["issue_count"] == 1
    assert result["suggestions"][0]["line"] == 1
    assert result["execution"] == "not_performed"


def test_generic_fix_suggestion_matches_unclosed_delimiter() -> None:
    result = suggest_fixes("javascript", "function f() {\n")
    assert result["suggestions"][0]["issue"] == "unclosed '{'"
    assert "'}'" in result["suggestions"][0]["suggestion"]


def test_fix_suggestions_tool_is_json_and_safe() -> None:
    payload = json.dumps({"language": "javascript", "code": "function f() {"})
    result = run_tool("code_fix_suggestions", payload)
    assert result.safe is True
    body = json.loads(result.output)
    assert body["issue_count"] == 1
    assert body["safe"] is True
