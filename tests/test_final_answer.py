from app.ai.final_answer import synthesize_tool_answer
from app.ai.tools import ToolResult


def test_synthesize_tool_answer_returns_single_safe_output() -> None:
    result = synthesize_tool_answer((ToolResult("calculator", "13"),))
    assert result == "13"


def test_synthesize_tool_answer_ignores_unsafe_results() -> None:
    result = synthesize_tool_answer((ToolResult("calculator", "13"), ToolResult("x", "bad", safe=False)))
    assert result == "13"


def test_synthesize_tool_answer_handles_empty_results() -> None:
    result = synthesize_tool_answer(())
    assert "enough reliable information" in result
