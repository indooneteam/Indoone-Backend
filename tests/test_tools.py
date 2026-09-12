from __future__ import annotations

import json

from app.ai.tools import run_tool


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
