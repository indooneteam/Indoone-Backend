from app.ai.agent import _serialize_result
from app.ai.tools import ToolResult


def test_agent_serializes_tool_truncation_metadata() -> None:
    result = ToolResult("text_stats", "partial", safe=True, truncated=True)
    assert _serialize_result(result) == {
        "name": "text_stats",
        "output": "partial",
        "safe": True,
        "retryable": False,
        "truncated": True,
    }


def test_agent_serializes_untruncated_result_explicitly() -> None:
    result = ToolResult("text_stats", "complete")
    assert _serialize_result(result)["truncated"] is False
