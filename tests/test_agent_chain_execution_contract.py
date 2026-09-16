from app.ai.agent import execute_agent
from app.ai.tools import ToolResult, TOOLS


def test_unresolved_chain_step_is_blocked_not_executed() -> None:
    execution = execute_agent("calculate $result9 + 1")
    assert execution.results[0].safe is False
    assert execution.steps == ()
    assert execution.blocked_steps[0].index == 1
    assert execution.blocked_steps[0].tool == "calculator"


def test_truncated_chain_result_cannot_feed_next_tool(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    def first_tool(_: str) -> str:
        calls.append(("first", ""))
        return "partial-output"

    def second_tool(payload: str) -> str:
        calls.append(("second", payload))
        return "done"

    monkeypatch.setitem(TOOLS, "text_stats", first_tool)
    monkeypatch.setitem(TOOLS, "json_summary", second_tool)

    def fake_plan_request(_: str):
        from app.ai.orchestrator import Plan
        return Plan(tool="text_stats", tool_payload="hello")

    monkeypatch.setattr("app.ai.agent.plan_request", fake_plan_request)
    execution = execute_agent("anything")

    assert calls == [("first", "")]
    assert execution.results[-1].safe is False
    assert "truncated" in execution.results[-1].output.lower()
    assert execution.blocked_steps == ()
