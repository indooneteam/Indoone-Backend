from __future__ import annotations

import json

from app.ai.tools import MAX_CALCULATOR_ABS_VALUE, MAX_CALCULATOR_EXPONENT, run_tool


def test_text_stats_tool_is_deterministic() -> None:
    result = run_tool("text_stats", "hello world\nagain")
    assert result.safe is True
    assert json.loads(result.output) == {"characters": 17, "lines": 2, "words": 3}


def test_json_summary_tool_is_bounded() -> None:
    result = run_tool("json_summary", '{"name":"Indoone","version":2}')
    assert result.safe is True
    assert json.loads(result.output) == {"key_names": ["name", "version"], "keys": 2, "type": "object"}


def test_unknown_tool_is_unsafe() -> None:
    result = run_tool("not_a_tool", "payload")
    assert result.safe is False


def test_calculator_rejects_huge_exponent() -> None:
    result = run_tool("calculator", f"2**{MAX_CALCULATOR_EXPONENT + 1}")
    assert result.safe is False


def test_calculator_rejects_huge_literal() -> None:
    result = run_tool("calculator", str(MAX_CALCULATOR_ABS_VALUE + 1))
    assert result.safe is False


def test_tool_payload_limit_is_enforced() -> None:
    result = run_tool("text_stats", "x" * 100_001)
    assert result.safe is False
    assert "payload" in result.output.lower()


def test_tool_name_is_normalized() -> None:
    result = run_tool(" TEXT_STATS ", "hello")
    assert result.safe is True
    assert result.name == "text_stats"


def test_non_string_payload_is_rejected() -> None:
    result = run_tool("text_stats", None)  # type: ignore[arg-type]
    assert result.safe is False
    assert "payload" in result.output.lower()
