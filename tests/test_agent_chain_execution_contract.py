from app.ai.agent import execute_agent
from app.ai.tools import ToolResult


def test_unresolved_chain_step_is_blocked_not_executed() -> None:
    execution = execute_agent("calculate $result9 + 1")
    assert execution.results[0].safe is False
    assert execution.steps == ()
    assert execution.blocked_steps[0].index == 1
    assert execution.blocked_steps[0].tool == "calculator"


def test_truncated_chain_result_cannot_feed_next_tool(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_run_tool(tool: str, payload: str, **_: object) -> ToolResult:
        calls.append((tool, payload))
        if tool == "text_stats":
            return ToolResult("text_stats", "partial-output", safe=True, truncated=True)
        return ToolResult(tool, "done", safe=True)

    monkeypatch.setattr("app.ai.agent.run_tool", fake_run_tool)
    execution = execute_agent("text stats: hello; summarize json: {\"ok\":true}")

    assert calls == [("text_stats", "hello")]
    assert len(execution.steps) == 1
    assert execution.steps[0].tool == "text_stats"
    assert execution.results[0].safe is True
    assert execution.results[0].truncated is True
    assert len(execution.results) == 2
    assert execution.results[1].safe is False
    assert "truncated" in execution.results[1].output.lower()
    assert len(execution.blocked_steps) == 1
    assert execution.blocked_steps[0].tool == "json_summary"
