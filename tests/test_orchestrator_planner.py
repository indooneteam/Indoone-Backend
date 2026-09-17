from app.ai.orchestrator import plan_request


def test_planner_routes_text_stats() -> None:
    plan = plan_request("analyze text: hello world")
    assert plan.tool == "text_stats"
    assert plan.tool_payload == "hello world"
    assert plan.executable is True


def test_planner_routes_coding_tools() -> None:
    analysis = plan_request("review code: print('hi')")
    assert analysis.tool == "code_analysis"
    sandbox = plan_request("run code: print(1)")
    assert sandbox.tool == "sandbox_execution"
    assert sandbox.executable is True


def test_planner_keeps_unknown_general_request_non_executable() -> None:
    plan = plan_request("tell me a story")
    assert plan.executable is False
    assert plan.tool is None
